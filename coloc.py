import concurrent.futures
import itertools
import multiprocessing
import numpy as np
import ome_types
import os
import pandas as pd
import pathlib
import scipy.stats
import skimage.filters
import sklearn.mixture
import skimage.measure
import skimage.morphology
import skimage.transform
import sys
import threadpoolctl
import tqdm
import tifffile


def imread(path):
    tiff = tifffile.TiffFile(path)
    img = tiff.series[0].asarray()
    channel = tiff.metaseries_metadata["PlaneInfo"]["_IllumSetting_"]
    if not channel.endswith("-PS"):
        raise ValueError(f"Unexpected IllumSetting naming pattern: {channel}")
    channel = channel[:-3]
    if tiff.series[0].shape != (2048, 2048):
        raise ValueError(f"Unexpected image dimensions: {tiff.series[0].shape}")
    # Apply scale and shift (channel-dependent) to account for chromatic aberration. This is not
    # a great correction model but it seems good enough for our needs.
    if channel == "DAPI":
        pass
    elif channel == "FITC":
        pass
    elif channel == "TRITC":
        img = skimage.transform.rescale(img, 1.001, preserve_range=True)[2:2050, 2:2050]
    elif channel == "CY5":
        img = skimage.transform.rescale(img, 1.002, preserve_range=True)[1:2049, 2:2050]
    else:
        raise ValueError(f"Unexpected channel name: {channel}")
    img = img.astype(np.uint16, copy=False)
    # Crop the top and right to avoid illumination artifacts.
    img = img[400:, :-400]
    return img


def gmm_fit(img, n_components):

    assert img.ndim == 2

    yi, xi = np.floor(np.linspace(0, img.shape, 200, endpoint=False)).astype(int).T
    # Slice one dimension at a time. Should generally use less memory than a meshgrid.
    img = img[yi]
    img = img[:, xi]
    img_log = np.log(img[img > 0])
    if len(np.unique(img_log)) < 2:
        return img.min(), img.max()

    gmm = sklearn.mixture.GaussianMixture(n_components, max_iter=1000, tol=1e-6)
    gmm.fit(img_log.reshape((-1, 1)))
    means = gmm.means_[:, 0]
    i1, i2 = np.argsort(means)[-2:]
    mean1, mean2 = means[[i1, i2]]
    std1, std2 = gmm.covariances_[[i1, i2], 0, 0] ** 0.5

    d1 = scipy.stats.norm(mean1, std1)
    d2 = scipy.stats.norm(mean2, std2)
    w1, w2 = gmm.weights_[[i1, i2]]

    return ((d1, w1), (d2, w2))


def auto_threshold(img, n_components=2):

    ((d1, w1), (d2, w2)) = gmm_fit(img, n_components)
    mean1, std1 = d1.stats()
    mean2, std2 = d2.stats()

    x = np.linspace(mean1, mean2, 50)
    y1 = d1.pdf(x) * w1
    y2 = d2.pdf(x) * w2

    lmax = mean2 + 3 * std2
    lmin = x[np.argmin(np.abs(y1 - y2))]
    if lmin >= mean2:
        lmin = mean2 - 3 * std2
    vmin = max(np.exp(lmin), img.min(), 0)
    vmax = min(np.exp(lmax), img.max())

    return vmin, vmax


def parse_paths(directory, well):
    # FIXME Resolve multiple subdir issue (see comment at bottom of file)
    # and assert that this glob only returns a single entry.
    scan_path = sorted(directory.glob('*/*/TimePoint_1'))[-1]
    paths = list(scan_path.glob(f'*_{well}_s*_w?????????-*.tif'))
    df = pd.DataFrame({'Path': paths})
    site_channel = (
        df.Path
        .map(lambda x: x.name)
        .str.extract(r'_s(?P<Site>\d+)_w(?P<Channel>\d)')
    )
    for col in site_channel:
        site_channel[col] = site_channel[col].astype(int)
    df = df.assign(**site_channel)
    return df


def subtract_bg(img, block_size):
    return np.clip(img - skimage.filters.threshold_local(img, block_size), 0, np.inf).astype(np.uint16)


def calc_quality(img_dna):
    # quality = 5 seems to be a good cutoff based on visual inspection.
    img_filtered = skimage.filters.laplace(skimage.filters.gaussian(img_dna, 10))
    quality = np.sum(np.abs(img_filtered))
    return quality


def prepare_dna(img):
    vmin, vmax = auto_threshold(img, 3)
    img = np.clip(img - float(vmin), 0, 65535).astype(np.uint16)
    mask = skimage.morphology.remove_small_objects(img > 0, 500)
    img[~mask] = 0
    return img


def prepare_marker(img):
    vmin, vmax = auto_threshold(img, 3)
    img = np.clip(img - float(vmin), 0, 65535)
    mask = skimage.morphology.remove_small_objects(img > 0, 10)
    img[~mask] = 0
    img = subtract_bg(img, 51)
    return img


def prepare_v5(img, parental_level):
    img = np.clip(img - float(parental_level), 0, 65535).astype(np.uint16)
    mask = skimage.morphology.remove_small_objects(img > 0, 10)
    img[~mask] = 0
    img = subtract_bg(img, 51)
    return img


def pcc(a, b):
    ccs = np.fft.ifft2(a * b.conj())
    ccsmax = ccs.max()
    aamp = np.sum(np.real(a * a.conj())) / a.size
    bamp = np.sum(np.real(b * b.conj())) / b.size
    tamp = aamp * bamp
    if tamp > 0:
       return abs(ccsmax * ccsmax.conj() / (aamp * bamp))
    else:
       return 0


def calc_parental_v5_level(args):
    plate, well, column, experiment, directory = args
    df = parse_paths(directory, well)
    df = df[df['Channel'].isin([1, 2])]
    paths = (
        df
        .pivot(index="Site", columns="Channel", values="Path")
        .rename(columns=lambda c: f"Channel{c}")
        .rename_axis(columns=None)
        .reset_index()
    )
    results = []
    for pt in paths.itertuples():
        img_dna = imread(pt.Channel1)
        img_v5 = imread(pt.Channel2)
        vmin_dna, vmax_dna = auto_threshold(img_dna)
        mask_dna = img_dna > vmin_dna
        mask_dna = skimage.morphology.remove_small_objects(mask_dna, 1000)
        log_v5 = np.log(img_v5[mask_dna])
        v5_mean = np.mean(log_v5)
        v5_std = np.std(log_v5)
        v5_level = np.exp(v5_mean + v5_std * 3)
        quality = calc_quality(img_dna)
        results.append({
            'plate': plate,
            'column': column,
            'experiment': experiment,
            'site': pt.Site,
            'parental_v5': v5_level,
            'dna_positive': np.sum(mask_dna),
            'quality': quality,
        })
    return results


def process(args):
    results = []
    try:
        plate, well, cell_line, marker1, marker2, parental_v5, directory = args
        df = parse_paths(directory, well)
        if marker1 == 'blank':
            marker1 = marker2
            marker2 = None
        for site, dfs in df.groupby('Site'):
            path_dna, path_v5, *path_markers = dfs.sort_values('Channel')['Path'].values
            assert (not marker2 and len(path_markers) == 1) or (marker2 and len(path_markers) == 2)
            img_dna_raw = imread(path_dna)
            quality = calc_quality(img_dna_raw)
            img_dna = prepare_dna(img_dna_raw)
            img_v5 = prepare_v5(imread(path_v5), parental_v5)
            img_markers = [prepare_marker(imread(p)) for p in path_markers]
            ipairs = [
                ("Hoechst33342", img_dna, path_dna),
                (marker1, img_markers[0], path_markers[0]),
            ]
            if marker2:
                ipairs.append((marker2, img_markers[1], path_markers[1]))
            for marker, img, ipath in ipairs:
                if img.any() and img_v5.any():
                    r, pvalue = scipy.stats.pearsonr(img_v5.ravel(), img.ravel())
                    # TODO: should the threshold be 100 or 0
                    # TODO: compute m2 also
                    m1 = skimage.measure.manders_coloc_coeff(img_v5, img > 0)
                    v5m = skimage.morphology.dilation(img_v5 > 0, skimage.morphology.disk(40))
                    m2 = skimage.measure.manders_coloc_coeff(img, img_v5 > 0, v5m)
                else:
                    r = np.nan
                    pvalue = np.nan
                    m1 = np.nan
                    m2 = np.nan
                v5_pos_count = np.sum(img_v5 > 0)
                results.append({
                    "Plate": plate,
                    "Well": well,
                    "CellLine": cell_line,
                    "Site": site,
                    "Marker": marker,
                    "R": r,
                    "M1": m1,
                    "M2": m2,
                    "Quality": quality,
                    "V5ControlLevel": parental_v5,
                    "V5PositiveCount": v5_pos_count,
                    "PathV5": path_v5,
                    "Path": ipath,
                })
    except Exception as e:
        print(f"ERROR: process({args}) : {e}")
        pass
    return results


def main():
    threadpoolctl.threadpool_limits(1)

    if hasattr(os, "sched_getaffinity"):
        num_workers = len(os.sched_getaffinity(0))
    else:
        num_workers = multiprocessing.cpu_count()

    in_path = pathlib.Path(sys.argv[1])
    out_path = pathlib.Path(sys.argv[2])
    out_control_path = pathlib.Path(sys.argv[3])

    assert out_path.suffix == '.csv', 'Output filename must end in .csv'

    df_in = pd.read_csv(in_path)
    base_path = in_path.resolve().parent
    df_in = df_in[df_in['directory'].notna()]
    df_in['experiment'] = df_in['directory']
    df_in['directory'] = df_in['directory'].map(lambda p: base_path / p)
    # Make sure parental line is only in row 1 and row 1 contains only parental
    # line.  Except in plate 8, in which rows 7 and 8 also contain parental line.
    # FIXME: What do do with plate 8 rows 7,8?
    assert len(df_in[
        ((df_in['cell_line'] == 'parental') ^ (df_in['row'] == 1))
        & ~((df_in['plate'] == 8) & df_in['row'].isin([7, 8]))
    ]) == 0, 'parental lines out of place'

    # Filter out EdU/gH2AX wells -- not used for colocalization.
    df_in = df_in[df_in['tritc'] != 'EdU']
    assert (df_in['cy5'] != 'gH2AX').all()

    # FIXME: subset for testing, delete later
    #df_in = df_in[(df_in.plate.isin([10, 11])) | ((df_in.plate == 29) & (df_in.row.isin([1, 2])))]
    #df_in = df_in[(df_in.plate.isin([9, 11, 17]))]
    #df_in = df_in[df_in.plate == 17]
    print(df_in.groupby(['plate', 'row']).size())

    is_parental = df_in['row'] == 1
    df_parental = df_in[is_parental]
    df_test = df_in[~is_parental]

    p_args = df_parental[['plate', 'well', 'column', 'experiment', 'directory']].values
    print('Computing V5 intensity levels in parental cell line controls')
    with concurrent.futures.ThreadPoolExecutor(num_workers) as pool:
        results = list(tqdm.tqdm(pool.map(calc_parental_v5_level, p_args), total=len(p_args)))
    print()

    df_control = pd.DataFrame(itertools.chain.from_iterable(results))
    df_control.to_csv(out_control_path, index=False)
    df_test = pd.merge(
        df_test,
        (
            df_control
            [(df_control.quality > 5) & (df_control.parental_v5 < 10_000)]
            .groupby('experiment')
            ['parental_v5']
            .median()
            .reset_index()
        )
    )

    args = df_test[['plate', 'well', 'cell_line', 'tritc', 'cy5', 'parental_v5', 'directory']].values
    print('Computing colocalization metrics')
    with concurrent.futures.ThreadPoolExecutor(num_workers) as pool:
        results = list(tqdm.tqdm(pool.map(process, args), total=len(args)))
    print()
    rows = itertools.chain.from_iterable(results)
    df_out = pd.DataFrame(rows)

    df_out.to_csv(out_path, index=False)


if __name__  == "__main__":
    main()


"""

dfm = df_out.copy()
dfm['ColFacet'] = 'CL=' + dfm.CellLine + ' P=' + dfm.Plate.astype(str)
g = sns.catplot(dfm, col='ColFacet', col_wrap=15, x='Marker', hue='Marker', y='M1')
g.set_titles('{col_name}')
g.tick_params(axis='x', rotation=90)
g.figure.tight_layout()

sns.catplot(..., y='R')

"""

# FIXME:
# scipy/stats/_stats_py.py:4781: ConstantInputWarning: An input array is constant; the correlation coefficient is not defined.

# FIXME: investigate 'gmm failed' instances

# for f in `xsv select directory IDG_DK_localization_metadata_240529.csv  | sort | uniq | grep -- -`; do ls -1d $f/*/*; echo; done
#
# missing: IDG-DK-loc-plate26-V5-EdU
# * ...-gH2AX variant does exist
# multiple dates: IDG-DK-loc-plate23-V5-EdU
# * First one is empty -- delete?
# multiple scan IDs: (many)

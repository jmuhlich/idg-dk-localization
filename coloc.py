import concurrent.futures
import cv2
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


class SerialExecutor:
    def __init__(self):
        pass
    def map(self, process, args):
        return map(process, args)
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        return None


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

    yi, xi = np.floor(np.linspace(0, img.shape, 100, endpoint=False)).astype(int).T
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


# def subtract_bg(img, block_size):
#     return np.clip(img - skimage.filters.threshold_local(img, block_size), 0, np.inf).astype(np.uint16)

def subtract_bg(img):
    # OpenCV's medianBlur is very fast, but only supports apertures > 5px on
    # uint8 images. We will log-transform our image (to preserve the dynamic
    # range) and rescale it to uint8, compute the medianBlur, then reverse the
    # transformation. The result is not completely equivalent to the true
    # computation, but it's close enough for our needs.
    factor = np.log(65535) / 255
    img_log8 = (np.log(np.where(img, img, 1)) / factor).round().astype(np.uint8)
    # Aperture of 15px corresponds to a diameter of ~5 microns as per Dunn 2011.
    blur_log8 = cv2.medianBlur(img_log8, 15)
    blur = np.exp(blur_log8 * factor)
    subtracted = np.clip(img - blur, 0, 65535).astype(np.uint16)
    return subtracted

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
    img = np.clip(img - float(vmin), 0, 65535).astype(np.uint16)
    mask = skimage.morphology.remove_small_objects(img > 0, 10)
    img[~mask] = 0
    img = subtract_bg(img)
    return img

# def prepare_marker_alt(img):
#     vmin, vmax = auto_threshold(img, 3)
#     img = np.clip(img - float(vmin), 0, 65535).astype(np.uint16)
#     mask = skimage.morphology.remove_small_objects(img > 0, 10)
#     img[~mask] = 0
#     img = np.clip(img.astype(float) - cv2.medianBlur(img, 5), 0, np.inf).astype(np.uint16)
#     return img

# def prepare_marker_alt2(img):
#     vmin, vmax = auto_threshold(img, 3)
#     img = np.clip(img - float(vmin), 0, 65535).astype(np.uint16)
#     mask = skimage.morphology.remove_small_objects(img > 0, 10)
#     img[~mask] = 0
#     elt = skimage.morphology.square(15)
#     img = np.clip(img.astype(float) - skimage.filters.rank.median(img, elt), 0, np.inf).astype(np.uint16)
#     return img


def prepare_v5(img, parental_level):
    img = np.clip(img - float(parental_level), 0, 65535).astype(np.uint16)
    mask = skimage.morphology.remove_small_objects(img > 0, 10)
    img[~mask] = 0
    img = subtract_bg(img)
    return img

# def prepare_v5_alt(img, parental_level):
#     img = np.clip(img - float(parental_level), 0, 65535).astype(np.uint16)
#     mask = skimage.morphology.remove_small_objects(img > 0, 10)
#     img[~mask] = 0
#     img = np.clip(img.astype(float) - cv2.medianBlur(img, 5), 0, np.inf).astype(np.uint16)
#     return img

# def prepare_v5_alt2(img, parental_level):
#     img = np.clip(img - float(parental_level), 0, 65535).astype(np.uint16)
#     mask = skimage.morphology.remove_small_objects(img > 0, 10)
#     img[~mask] = 0
#     elt = skimage.morphology.square(15)
#     img = np.clip(img.astype(float) - skimage.filters.rank.median(img, elt), 0, np.inf).astype(np.uint16)
#     return img


def phase_cross_correlation(a, b):
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


def calc_well_metrics(args):
    results = []
    try:
        plate, well, cell_line, marker1, marker2, parental_v5, directory = args
        df = parse_paths(directory, well)
        m2_mask_elt = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (81, 81))
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
                    m1 = skimage.measure.manders_coloc_coeff(img_v5, img > 0)
                    v5m = cv2.dilate((img_v5 > 0).astype(np.uint8), m2_mask_elt)
                    m2 = skimage.measure.manders_coloc_coeff(img, img_v5 > 0, v5m)
                    pcc = phase_cross_correlation(img_v5, img)
                else:
                    r = np.nan
                    #pvalue = np.nan
                    m1 = np.nan
                    m2 = np.nan
                    pcc = np.nan
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
                    "PCC": pcc,
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


def setup():

    threadpoolctl.threadpool_limits(1)

    if hasattr(os, "sched_getaffinity"):
        num_workers = len(os.sched_getaffinity(0))
    else:
        num_workers = multiprocessing.cpu_count()

    in_path = pathlib.Path(sys.argv[1])
    out_path = pathlib.Path(sys.argv[2])
    out_control_path = pathlib.Path(sys.argv[3])

    assert out_path.suffix == '.csv', 'Output filename must end in .csv'
    assert out_control_path.suffix == '.csv', 'Controls output filename must end in .csv'

    return in_path, out_path, out_control_path, num_workers


def load_metadata(in_path):

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

    return df_in


def compute_controls(df_parental, pool):
    args = df_parental[['plate', 'well', 'column', 'experiment', 'directory']].values
    results = list(tqdm.tqdm(pool.map(calc_parental_v5_level, args), total=len(args)))
    df_control = pd.DataFrame(itertools.chain.from_iterable(results))
    qc_pass = (df_control.quality > 5) & (df_control.parental_v5 < 10_000)
    df_control["qc_pass"] = qc_pass
    return df_control


def merge_parental_v5(df_test, df_control):
    df_control_pass = df_control[df_control["qc_pass"]]
    df_median_v5 = df_control_pass.groupby('experiment')['parental_v5'].median().reset_index()
    df_test = pd.merge(df_test, df_median_v5)
    return df_test


def compute_metrics(df_test, pool):
    args = df_test[['plate', 'well', 'cell_line', 'tritc', 'cy5', 'parental_v5', 'directory']].values
    results = list(tqdm.tqdm(pool.map(calc_well_metrics, args), total=len(args)))
    rows = itertools.chain.from_iterable(results)
    df_metrics = pd.DataFrame(rows)
    return df_metrics


def main():

    in_path, out_path, out_control_path, num_workers = setup()
    pool = concurrent.futures.ThreadPoolExecutor(num_workers)
    df_meta = load_metadata(in_path)

    # FIXME: subset for testing, delete later
    #df_meta = df_meta[(df_meta.plate.isin([10, 11])) | ((df_meta.plate == 29) & (df_meta.row.isin([1, 2])))]
    #df_meta = df_meta[(df_meta.plate.isin([9, 11, 17]))]
    df_meta = df_meta[df_meta.plate == 11]
    print(df_meta.groupby(['plate', 'row']).size())

    is_parental = df_meta['row'] == 1
    df_parental = df_meta[is_parental]
    df_test = df_meta[~is_parental]

    print('Computing V5 intensity levels in parental cell line controls')
    df_control = compute_controls(df_parental, pool)
    df_control.to_csv(out_control_path, index=False)
    print()

    print('Computing colocalization metrics')
    df_test = merge_parental_v5(df_test, df_control)
    df_metrics = compute_metrics(df_test, pool)
    df_metrics.to_csv(out_path, index=False)
    print()


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

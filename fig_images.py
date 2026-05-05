import colour
import functools
import imageio.v3 as imageio
import numpy as np
import pandas as pd
import pathlib
import scipy.stats
import scipy.spatial
import skimage.exposure
import sklearn.cluster
import sklearn.mixture
import sys
import threadpoolctl
import tqdm

import coloc



class ArrayWrapper:
    "Quick hashable wrapper around a numpy array which we know is read-only"

    def __init__(self, a):
        self.array = a

    def __hash__(self):
        i = self.array.__array_interface__
        i['descr'] = tuple(['descr'])
        return hash(tuple(i.items()))

    def __eq__(self, other):
        return self.array.__array_interface__ == other.array.__array_interface__


def auto_threshold(img):
    img_wrap = ArrayWrapper(img)
    return auto_threshold_inner(img_wrap)


@functools.cache
def auto_threshold_inner(img_wrap):

    img = img_wrap.array

    img_log = np.log(img[img > 0])
    if len(np.unique(img_log)) < 2:
        return img.min(), img.max()

    gmm = sklearn.mixture.GaussianMixture(2, max_iter=1000, tol=1e-6)
    gmm.fit(img_log.reshape((-1,1)))
    means = gmm.means_[:, 0]
    i1, i2 = np.argsort(means)
    mean1, mean2 = means[[i1, i2]]
    std1, std2 = gmm.covariances_[[i1, i2], 0, 0] ** 0.5

    x = np.linspace(mean1, mean2, 50)
    y1 = scipy.stats.norm(mean1, std1).pdf(x) * gmm.weights_[i1]
    y2 = scipy.stats.norm(mean2, std2).pdf(x) * gmm.weights_[i2]

    lmax = mean2 + 2 * std2
    lmin = x[np.argmin(np.abs(y1 - y2))]
    if lmin >= mean2:
        lmin = mean2 - 2 * std2
    vmin = max(np.exp(lmin), img.min(), 0)
    vmax = min(np.exp(lmax), img.max())

    return vmin, vmax


def threshold_btubulin(img):
    gmm = sklearn.mixture.GaussianMixture(3, max_iter=1000, tol=1e-6)
    gmm.fit(np.log(img[img>0]).reshape(-1, 1))
    idxs = np.argsort(gmm.means_[:, 0])[[1]]
    lm1, = gmm.means_[idxs, 0]
    ls1, = gmm.covariances_[idxs, 0, 0] ** 0.5
    vmin = np.exp(lm1 - ls1 * 2)
    vmax = np.exp(lm1 + ls1 * 2)
    return vmin, vmax


def threshold_gm130(img):
    gmm = sklearn.mixture.GaussianMixture(3, max_iter=1000, tol=1e-6)
    gmm.fit(np.log(img[img>0]).reshape(-1, 1))
    idxs = np.argsort(gmm.means_[:, 0])[[2, 1]]
    lm1, lm2 = gmm.means_[idxs, 0]
    ls1, ls2 = gmm.covariances_[idxs, 0, 0] ** 0.5
    vmin = np.exp(lm2 - ls2 * 1)
    vmax = np.exp(lm1 + ls1 * 3)
    return vmin, vmax


def threshold_lamp(img):
    gmm = sklearn.mixture.GaussianMixture(3, max_iter=1000, tol=1e-6)
    gmm.fit(np.log(img[img>0]).reshape(-1, 1))
    idxs = np.argsort(gmm.means_[:, 0])[[2, 1]]
    lm1, lm2 = gmm.means_[idxs, 0]
    ls1, ls2 = gmm.covariances_[idxs, 0, 0] ** 0.5
    vmin = np.exp(lm2 - ls2)
    vmax = np.exp(lm1 + ls1 * 3)
    return vmin, vmax


def threshold_lamin(img):
    gmm = sklearn.mixture.GaussianMixture(3, max_iter=1000, tol=1e-6)
    gmm.fit(np.log(img[img>0]).reshape(-1, 1))
    idxs = np.argsort(gmm.means_[:, 0])[[2, 1]]
    lm1, lm2 = gmm.means_[idxs, 0]
    ls1, ls2 = gmm.covariances_[idxs, 0, 0] ** 0.5
    vmin = np.exp(lm2 - ls2)
    vmax = np.exp(lm1 + ls1 * 3)
    return vmin, vmax


def threshold_calnexin(img):
    gmm = sklearn.mixture.GaussianMixture(3, max_iter=1000, tol=1e-6)
    gmm.fit(np.log(img[img>0]).reshape(-1, 1))
    idxs = np.argsort(gmm.means_[:, 0])[[2, 1]]
    lm1, lm2 = gmm.means_[idxs, 0]
    ls1, ls2 = gmm.covariances_[idxs, 0, 0] ** 0.5
    vmin = np.exp(lm2 - ls2 * 3)
    vmax = np.exp(lm1 + ls1 * 2)
    return vmin, vmax


def threshold_cytochromec(img):
    gmm = sklearn.mixture.GaussianMixture(3, max_iter=1000, tol=1e-6)
    gmm.fit(np.log(img[img>0]).reshape(-1, 1))
    idxs = np.argsort(gmm.means_[:, 0])[[2, 1]]
    lm1, lm2 = gmm.means_[idxs, 0]
    ls1, ls2 = gmm.covariances_[idxs, 0, 0] ** 0.5
    vmin = np.exp(lm2 - ls2 * 3)
    vmax = np.exp(lm1 + ls1 * 3)
    return vmin, vmax


def threshold_streptavidin(img):
    gmm = sklearn.mixture.GaussianMixture(3, max_iter=1000, tol=1e-6)
    gmm.fit(np.log(img[img>0]).reshape(-1, 1))
    idxs = np.argsort(gmm.means_[:, 0])[[2, 1]]
    lm1, lm2 = gmm.means_[idxs, 0]
    ls1, ls2 = gmm.covariances_[idxs, 0, 0] ** 0.5
    # FIXME check these
    vmin = np.exp(lm2 - ls2)
    vmax = np.exp(lm1 + ls1 * 3)
    return vmin, vmax


tfuncs = {
    "b-tubulin": threshold_btubulin,
    "GM-130": threshold_gm130,
    "LAMP": threshold_lamp,
    "lamin": threshold_lamin,
    "calnexin": threshold_calnexin,
    "cytochromeC": threshold_cytochromec,
    "streptavidin": threshold_streptavidin,
}
@functools.cache
def auto_threshold_marker(img_wrap, name):
    func = tfuncs[name]
    vmin, vmax = func(img_wrap.array)
    vmin = max(vmin, img_wrap.array.min(), 0)
    vmax = min(vmax, img_wrap.array.max())
    return vmin, vmax


def gray(img, normalize=True):
    img = skimage.util.img_as_float32(img)
    if normalize:
        img = skimage.exposure.rescale_intensity(img)
    img = np.dstack([img, img, img])
    return img

def colorize(v5=None, marker=None, dna=None, marker_name=None):
    assert (marker is not None and marker_name is not None) or (marker is None and marker_name is None)
    shape = v5.shape if v5 is not None else marker.shape if marker is not None else dna.shape
    #sp = round(np.mean(shape) / 20)
    img_xyz = np.zeros(shape + (3,))
    for rimg, (L, C, h) in zip((marker, v5, dna), oklch_colors):
        if rimg is None:
            continue
        # rrange = (
        #     auto_threshold_marker(ArrayWrapper(rimg), marker_name) if rimg is marker
        #     else auto_threshold(rimg[::sp, ::sp])
        # )
        rrange = (0, 65535)
        lum_img = skimage.exposure.rescale_intensity(rimg, rrange, float)
        cimg = np.zeros_like(img_xyz)
        cimg[..., 0] = colour.lightness(lum_img) * L
        cimg[..., 1] = C
        cimg[..., 2] = h
        img_xyz += colour.Oklab_to_XYZ(colour.Oklch_to_Oklab(cimg))
    img = np.clip(colour.XYZ_to_sRGB(img_xyz), 0, 1)
    return img

threadpoolctl.threadpool_limits(1)

dfm = pd.read_csv(sys.argv[1])
dfms = pd.read_parquet(sys.argv[2])

project_path = pathlib.Path(__file__).parent.resolve()

base = project_path / "out" / "figures"
print(f"Saving images to: {base}")
base.mkdir(parents=True, exist_ok=True)

colour.set_domain_range_scale('1')
oklch_colors = colour.Oklab_to_Oklch(colour.XYZ_to_Oklab(colour.sRGB_to_XYZ(np.diag(np.ones(3)))))

CLUSTER_SIZE_TARGET = 50
CROP_WIDTH = 175
PADDING = 10

dfm['Marker'] = dfm['Marker'].replace('Hoechst33342', 'DNA')
dfm['CellLine'] = dfm['CellLine'].replace('2270 (2218c)', '2218c')
dfm = dfm[dfm['CellLine'] != 'parental']
dfm['PlateRowLine'] = dfm['Plate'].astype(str) + dfm['Well'].str[0] + ' ' + dfm['CellLine'].astype(str)

cell_line_locations = dfm[['Plate','Well','CellLine']].drop_duplicates()
cell_line_locations['PlateRowLine'] = (
    cell_line_locations['Plate'].astype(str)
    + cell_line_locations['Well'].str[0]
    + ' '
    + cell_line_locations['CellLine'].astype(str)
)
ignore_wells = dfm[dfm['Marker'].isin(['EdU','gH2AX'])][['Plate','Well']].drop_duplicates()
cell_line_locations = (
    pd.merge(cell_line_locations, ignore_wells, how='outer', indicator=True)
    .query('_merge=="left_only"')
    .drop(columns='_merge')
)
dfms['Marker'] = dfms['Marker'].replace('Hoechst33342', 'DNA')
dfms = pd.merge(dfms, cell_line_locations)

dfmsp = dfms[dfms['QcPass'] & dfms['V5Positive']]

dfmm = (
    dfmsp
    [dfmsp.Well.str[1:] >= '06'] # Ugly way to ignore wells with streptavidin or gH2AX+EdU
    .groupby(['Plate', 'CellLine', 'PlateRowLine', 'Marker'])
    ['M1']
    .median()
    .unstack('Marker')
    .apply(scipy.stats.zscore, nan_policy='omit')
    .stack(future_stack=True)
    .rename('M1')
    .reset_index()
)

m1m = dfmm.set_index(['PlateRowLine', 'Marker'])['M1'].unstack('Marker')

best_m1_marker = (
    m1m
    .stack()
    .sort_values(ascending=False)
    .reset_index()
    .drop_duplicates('PlateRowLine')
    .set_index('PlateRowLine')
    ['Marker']
)

empty_column = np.zeros((4 * CROP_WIDTH + 3 * PADDING, CROP_WIDTH, 3))
hpad = np.zeros((CROP_WIDTH, PADDING, 3))

gbs = dfms.groupby(['PlateRowLine', 'Marker'])
gbsp = dfmsp.groupby(['PlateRowLine', 'Marker'])

def save_figure(plate_row_line, marker):
    img = generate_figure(plate_row_line, marker)
    if img is not None:
        imageio.imwrite(base / f"{plate_row_line} {marker}.jpg", img, quality=95)

def generate_figure(plate_row_line, marker):
    if marker == 'DNA':
        marker_show = 'lamin'
    else:
        marker_show = marker
    try:
        cellsp = gbsp.get_group((plate_row_line, marker_show))
    except KeyError:
        try:
            marker_show = 'calnexin'
            cellsp = gbsp.get_group((plate_row_line, marker_show))
        except KeyError:
            print(f"No good cells for {plate_row_line} / {marker} (showing {marker_show})")
            return
    gbmsp = cellsp.groupby(['Well', 'Site'])
    well, site = gbmsp.size().sort_values().index[-1]
    cells = (
        gbs
        .get_group((plate_row_line, marker_show))
        .groupby(['Well', 'Site'])
        .get_group((well, site))
        .copy()
    )
    tree = scipy.spatial.KDTree(cells[['X','Y']])
    neighbors = tree.query_ball_tree(tree, r=CROP_WIDTH/2, p=1)
    cells_pass = cells['QcPass'] & cells['V5Positive']
    cells['NeighborPassCount'] = np.array([cells_pass.iloc[n].sum() for n in neighbors], float)
    cells['NeighborRejectCount'] = np.array([(~cells_pass).iloc[n].sum() for n in neighbors], float)
    cells['NeighborCountDiff'] = np.clip(
        cells['NeighborPassCount'] - cells['NeighborRejectCount'],
        min=0,
    )
    neighborsp = [[c for c in n if cells_pass.iloc[c]] for n, p in zip(neighbors, cells_pass) if p]
    cells.loc[cells_pass, 'NeighborM1Median'] = np.array([cells.iloc[n]['M1'].median() for n in neighborsp])
    cells.loc[cells_pass, 'NeighborV5Median'] = np.array([cells.iloc[n]['V5IntensityMean'].median() for n in neighborsp])
    cells.loc[cells_pass, 'Score'] = (
        cells['NeighborM1Median'] ** 0.5
        * cells['NeighborV5Median'] / cells['NeighborV5Median'].max()
        * (cells['NeighborCountDiff'] / cells['NeighborCountDiff'].max() / 2)
    )
    cx, cy = cells.loc[cells_pass].sort_values('Score', na_position='first')[['X','Y']].iloc[-1].round().astype(int)
    r = dfm[(dfm.PlateRowLine==plate_row_line) & (dfm.Well==well) & (dfm.Site==site) & (dfm.Marker==marker_show)].iloc[0]
    path_dna = (
        dfm.merge(pd.DataFrame([r[['Plate', 'Well', 'Site']]]))
        .query('Marker=="DNA"')
        .iloc[0]
        .Path
    )
    img_v5 = coloc.subtract_bg(coloc.imread(r.PathV5))
    if marker == 'Membrane':
        img_marker = coloc.calc_membrane_mask(coloc.load_mask(r.Plate, well, site)) * 0.5
    else:
        img_marker = coloc.subtract_bg(coloc.imread(r.Path))
    img_dna = coloc.imread(path_dna)
    x1 = np.clip(cx - CROP_WIDTH // 2, 0, img_v5.shape[1] - CROP_WIDTH)
    y1 = np.clip(cy - CROP_WIDTH // 2, 0, img_v5.shape[0] - CROP_WIDTH)
    x2 = x1 + CROP_WIDTH
    y2 = y1 + CROP_WIDTH
    crop_v5 = img_v5[y1:y2, x1:x2]
    crop_marker = img_marker[y1:y2, x1:x2]
    crop_dna = img_dna[y1:y2, x1:x2]
    panel_v5 = gray(crop_v5)
    panel_marker = gray(crop_marker, normalize=(marker != 'Membrane'))
    panel_dna = gray(crop_dna)
    # Quick and dirty RGB merge.
    panel_merge = np.dstack([panel_marker[..., 0], panel_v5[...,0], panel_dna[...,0]])
    img_out = np.hstack([panel_v5, hpad, panel_marker, hpad, panel_dna, hpad, panel_merge])
    img_out = skimage.util.img_as_ubyte(img_out)
    return img_out


for (plate_row_line, marker) in tqdm.tqdm(best_m1_marker.items(), total=len(best_m1_marker)):
    save_figure(plate_row_line, marker)

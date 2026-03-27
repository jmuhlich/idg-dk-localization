import colour
import concurrent.futures
import functools
import imageio.v3 as imageio
import numpy as np
import pandas as pd
import pathlib
import scipy.stats
import skimage.exposure
import sklearn.cluster
import sklearn.mixture
import sys
import threading
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


def gray(img, marker):
    img = skimage.util.img_as_float32(img)
    img = skimage.exposure.rescale_intensity(img)
    img = np.dstack([img, img, img])
    return img

# Functions from the colour package are somehow not thread-safe so we serialize
# all calls. Without locking, the functions generate wrong results.
c_lock = threading.Lock()
def colorize(v5=None, marker=None, dna=None, marker_name=None):
    assert (marker is not None and marker_name is not None) or (marker is None and marker_name is None)
    shape = v5.shape if v5 is not None else marker.shape if marker is not None else dna.shape
    sp = round(np.mean(shape) / 20)
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
        with c_lock:
            cimg[..., 0] = colour.lightness(lum_img) * L
        cimg[..., 1] = C
        cimg[..., 2] = h
        with c_lock:
            img_xyz += colour.Oklab_to_XYZ(colour.Oklch_to_Oklab(cimg))
    with c_lock:
        img = np.clip(colour.XYZ_to_sRGB(img_xyz), 0, 1)
    return img

threadpoolctl.threadpool_limits(1)

df = pd.read_csv(sys.argv[1])
dfs = pd.read_parquet(sys.argv[2])

base = pathlib.Path(__file__).parent.resolve() / "out" / "figures"
print(f"Saving images to: {base}")
base.mkdir(parents=True, exist_ok=True)

colour.set_domain_range_scale('1')
oklch_colors = colour.Oklab_to_Oklch(colour.XYZ_to_Oklab(colour.sRGB_to_XYZ(np.diag(np.ones(3)))))

CLUSTER_SIZE_TARGET = 50
CROP_WIDTH = 175
PADDING = 10
marker_order = [
    "b-tubulin", "GM-130", "LAMP", "lamin", "calnexin", "cytochromeC", "streptavidin", "Hoechst33342"
]
df = df[~df.Marker.isin(['Membrane', 'EdU', 'gH2AX']) & (df.CellLine != 'parental')]
df['Marker'] = pd.Categorical(df['Marker'], categories=marker_order, ordered=True)
df['RowName'] = df['Well'].str[0]
assert df.Marker.notna().all()

dfs = dfs[dfs['QcPass'] & dfs['V5Positive']]
dfs['RowName'] = dfs['Well'].str[0]

empty_column = np.zeros((4 * CROP_WIDTH + 3 * PADDING, CROP_WIDTH, 3))
vpad = np.zeros((PADDING, CROP_WIDTH, 3))
hpad = np.zeros((4 * CROP_WIDTH + 3 * PADDING, PADDING, 3))

gbs = dfs.groupby(['Plate', 'RowName', 'Marker'])

cluster_lock = threading.Lock()

def generate_figure(plate, row, cell_line, dfc):
    marker_panels = []
    for marker, dfg in dfc.groupby('Marker', observed=False):
        if marker == 'Hoechst33342':
            continue
        try:
            cells = gbs.get_group((plate, row, marker))
        except KeyError:
            marker_panels.append(empty_column)
            continue
        if len(cells) == 0:
            raise Exception(f"No cells in group: {(plate, row, marker)}")
        gbms = cells.groupby(['Well', 'Site'])
        well, site = gbms.size().sort_values().index[-1]
        cells = gbms.get_group((well, site)).copy()
        # dbscan can cause deadlocks if run from multiple threads in parallel.
        with cluster_lock:
            _, labels = sklearn.cluster.dbscan(cells[['X','Y']], metric='euclidean', eps=80, min_samples=10, n_jobs=1)
            if (labels == -1).all():
                _, labels = sklearn.cluster.dbscan(cells[['X','Y']], metric='euclidean', eps=100, min_samples=5, n_jobs=1)
                if (labels == -1).all():
                    # If no real clusters exist, use single cell with highest M1 as the "cluster".
                    idx = np.argsort(cells['M1']).iloc[-1]
                    labels[idx] = 0
        cells['Cluster'] = labels
        cluster_sizes = cells.Cluster.value_counts().drop(-1, errors='ignore')
        best_cluster = np.abs(cluster_sizes - CLUSTER_SIZE_TARGET).sort_values().index[0]
        # FIXME use cell closest to centroid, not centroid itself.
        cx, cy = cells[cells.Cluster==best_cluster][['X', 'Y']].mean().round().astype(int)
        r = dfg[(dfg.Well==well) & (dfg.Site==site)].iloc[0]
        path_dna = (
            df.merge(pd.DataFrame([r[['Plate', 'Well', 'Site']]]))
            .query('Marker=="Hoechst33342"')
            .iloc[0]
            .Path
        )
        img_v5 = coloc.imread(r.PathV5)
        img_marker = coloc.imread(r.Path)
        img_dna = coloc.imread(path_dna)
        x1 = np.clip(cx - CROP_WIDTH // 2, 0, img_v5.shape[1] - CROP_WIDTH)
        y1 = np.clip(cy - CROP_WIDTH // 2, 0, img_v5.shape[0] - CROP_WIDTH)
        x2 = x1 + CROP_WIDTH
        y2 = y1 + CROP_WIDTH
        crop_v5 = img_v5[y1:y2, x1:x2]
        crop_marker = img_marker[y1:y2, x1:x2]
        crop_dna = img_dna[y1:y2, x1:x2]
        panel_v5 = gray(crop_v5, marker='v5')
        panel_marker = gray(crop_marker, marker=marker)
        panel_dna = gray(crop_dna, marker='dna')
        #panel_merge = colorize(v5=crop_v5, marker=crop_marker, dna=crop_dna, marker_name=marker)
        panel_merge = np.dstack([panel_marker[..., 0], panel_v5[...,0], panel_dna[...,0]])
        panel_column = np.vstack([panel_merge, vpad, panel_v5, vpad, panel_marker, vpad, panel_dna])
        marker_panels.append(panel_column)
    img_out = np.hstack([np.hstack([p, hpad]) for p in marker_panels])[:, :-PADDING]
    img_out = skimage.util.img_as_ubyte(img_out)
    imageio.imwrite(base / f"{cell_line}-P{plate}.jpg", img_out, quality=95)

class SerialExecutor:
    def __init__(self):
        pass
    def map(self, process, *args):
        return map(process, *args)
    def submit(self, process, *args):
        f = concurrent.futures.Future()
        f.set_result(process(*args))
        return f
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        return None

pool = concurrent.futures.ThreadPoolExecutor(4)
#pool = SerialExecutor()
futures = [
    pool.submit(generate_figure, plate, row, cell_line, dfc)
    for (plate, row, cell_line), dfc in df.groupby(['Plate', 'RowName', 'CellLine'])
]
_ = list(tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures)))

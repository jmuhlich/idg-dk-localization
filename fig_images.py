import colour
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


def colorize(v5=None, marker=None, dna=None):
    shape = v5.shape if v5 is not None else marker.shape if marker is not None else dna.shape
    sp = round(np.mean(shape) / 20)
    img_xyz = np.zeros(shape + (3,))
    for rimg, (L, C, h) in zip((marker, v5, dna), oklch_colors):
        if rimg is None:
            continue
        rrange = auto_threshold(rimg[::sp, ::sp])
        lum_img = skimage.exposure.rescale_intensity(rimg, rrange, float)
        cimg = np.zeros_like(img_xyz)
        cimg[..., 0] = colour.lightness(lum_img) * L
        cimg[..., 1] = C
        cimg[..., 2] = h
        img_xyz += colour.Oklab_to_XYZ(colour.Oklch_to_Oklab(cimg))
    img = np.clip(colour.XYZ_to_sRGB(img_xyz), 0, 1)
    return img


threadpoolctl.threadpool_limits(1)

df = pd.read_csv(sys.argv[1])
dfs = pd.read_parquet(sys.argv[2])

base = pathlib.Path(__file__).parent.resolve() / "out" / "figures"
base.mkdir(parents=True, exist_ok=True)

colour.set_domain_range_scale('1')
oklch_colors = colour.Oklab_to_Oklch(colour.XYZ_to_Oklab(colour.sRGB_to_XYZ(np.diag(np.ones(3)))))

CLUSTER_SIZE_TARGET = 50
CROP_WIDTH = 175
PADDING = 10
marker_order = [
    "b-tubulin", "GM-130", "LAMP", "lamin", "calnexin", "cytochromeC", "streptavidin", "Hoechst33342"
]
df = df[df.Marker != 'Membrane']
df['Marker'] = pd.Categorical(df['Marker'], categories=marker_order, ordered=True)
assert df.Marker.notna().all()

empty_column = np.zeros((4 * CROP_WIDTH + 3 * PADDING, CROP_WIDTH, 3))
vpad = np.zeros((PADDING, CROP_WIDTH, 3))
hpad = np.zeros((4 * CROP_WIDTH + 3 * PADDING, PADDING, 3))

gbs = dfs.groupby(['Plate', 'Well', 'Site', 'Marker'])

for (plate, cell_line), dfc in tqdm.tqdm(df.groupby(['Plate', 'CellLine'])):
    if cell_line == 'parental':
        continue
    marker_panels = []
    for marker, dfg in dfc.groupby('Marker', observed=False):
        if marker == 'Hoechst33342':
            continue
        if len(dfg):
            r = dfg.sort_values('V5PositiveCells').iloc[-1]
            key = tuple(r[['Plate', 'Well', 'Site', 'Marker']])
        else:
            key = (None,) * 4
        try:
            cells = gbs.get_group(key)
            cells = cells[cells.V5Positive]
        except KeyError:
            cells = []
        if len(cells) == 0:
            marker_panels.append(empty_column)
            continue
        _, labels = sklearn.cluster.dbscan(cells[['X','Y']], metric='euclidean', eps=80, min_samples=10)
        if (labels == -1).all():
            _, labels = sklearn.cluster.dbscan(cells[['X','Y']], metric='euclidean', eps=100, min_samples=5)
            if (labels == -1).all():
                marker_panels.append(empty_column)
                continue
        cells['Cluster'] = labels
        cluster_sizes = cells.Cluster.value_counts().drop(-1, errors='ignore')
        best_cluster = np.abs(cluster_sizes - CLUSTER_SIZE_TARGET).sort_values().index[0]
        cx, cy = cells[cells.Cluster==best_cluster][['X', 'Y']].mean().round().astype(int)
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
        panel_v5 = colorize(v5=crop_v5)
        panel_marker = colorize(marker=crop_marker)
        panel_dna = colorize(dna=crop_dna)
        panel_merge = colorize(v5=crop_v5, marker=crop_marker, dna=crop_dna)
        panel_column = np.vstack([panel_merge, vpad, panel_v5, vpad, panel_marker, vpad, panel_dna])
        marker_panels.append(panel_column)
    img_out = np.hstack([np.hstack([p, hpad]) for p in marker_panels])[:, :-PADDING]
    img_out = skimage.util.img_as_ubyte(img_out)
    imageio.imwrite(base / f"{cell_line}-P{plate}.jpg", img_out)

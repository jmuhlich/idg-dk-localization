import concurrent.futures
import itertools
import multiprocessing
import numpy as np
import ome_types
import os
import pandas as pd
import pathlib
import scipy.stats
import sklearn.mixture
import skimage.measure
import threadpoolctl
import tqdm
import tifffile


def truestem(p):
    """Return stem with ALL suffixes stripped."""
    full_suffix = "".join(p.suffixes)
    return p.name[:p.name.index(full_suffix)]


def auto_threshold(img):

    assert img.ndim == 2

    yi, xi = np.floor(np.linspace(0, img.shape, 200, endpoint=False)).astype(int).T
    # Slice one dimension at a time. Should generally use less memory than a meshgrid.
    img = img[yi]
    img = img[:, xi]
    img_log = np.log(img[img > 0])
    if len(np.unique(img_log)) < 2:
        return img.min(), img.max()

    gmm = sklearn.mixture.GaussianMixture(3, max_iter=1000, tol=1e-6)
    gmm.fit(img_log.reshape((-1,1)))
    means = gmm.means_[:, 0]
    _, i1, i2 = np.argsort(means)
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


def process(p):
    img = tifffile.imread(p)
    metadata = ome_types.from_tiff(p)
    cell_line, label_1, label_2 = truestem(p).split("_")
    img_dna = img[0]
    img_v5 = img[1]
    channel_names = [c.name for c in metadata.images[0].pixels.channels]
    # Account for the decision to omit blank channels from ome-tiffs.
    if len(channel_names) == 4:
        labelimg_1 = img[2]
        labelimg_2 = img[3]
    else:
        channel_names.insert(2, "blank")
        labelimg_1 = None
        labelimg_2 = img[2]
    assert channel_names[0] == "Hoechst33342"
    assert channel_names[1] == "V5"
    assert channel_names[2] == label_1
    assert channel_names[3] == label_2
    ipairs = (
        ("Hoechst33342", img_dna),
        (label_1, labelimg_1),
        (label_2, labelimg_2),
    )
    results = []
    for label, labelimg in ipairs:
        if labelimg is None:
            continue
        r, pvalue = scipy.stats.pearsonr(img_v5.ravel(), labelimg.ravel())
        m1 = skimage.measure.manders_coloc_coeff(img_v5, labelimg > 100)
        results.append({
            "CellLine": cell_line,
            "Label": label,
            "R": r,
            "M1": m1,
        })
    return results


threadpoolctl.threadpool_limits(1)

if hasattr(os, "sched_getaffinity"):
    num_workers = len(os.sched_getaffinity(0))
else:
    num_workers = multiprocessing.cpu_count()

# TODO: Work from CSV and original TIFFs to subtract parental control V5 signal
paths = list((pathlib.Path(__file__).resolve().parent / "out").glob("*.ome.tif"))

with concurrent.futures.ThreadPoolExecutor(num_workers) as pool:
    results = list(tqdm.tqdm(pool.map(process, paths), total=len(paths)))
rows = itertools.chain.from_iterable(results)
df = pd.DataFrame(rows)

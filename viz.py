import dask.array as da
import napari
import pandas as pd
import sys
import tqdm
import zarr

import coloc


marker_to_channel = {
    'Hoechst33342': 0,
    'V5': 1,
    'streptavidin': 3,
    'LAMP': 2,
    'lamin': 3,
    'b-tubulin': 2,
    'GM-130': 3,
    'calnexin': 2,
    'cytochromeC': 3,
}

df = pd.read_csv(sys.argv[1])
plate, row = sys.argv[2:]

df = df[(df.Plate==int(plate)) & (df.Well.str.startswith(row))].copy()
df['Channel'] = df.Marker.map(marker_to_channel)

th, tw = coloc.imread(df.iloc[0].Path).shape
wh = 3 if df.Site.max() <= 9 else 4
ww = 3
ih = wh * th
iw = 12 * ww * tw
zimg = zarr.open_array(
    mode='w',
    shape=(4, ih, iw),
    chunks=(1, 512, 512),
    dtype='uint16',
)

loaded_v5 = set()
for row in tqdm.tqdm(df.itertuples(), total=len(df), desc='loading images'):
    col = int(row.Well[1:]) - 1
    field = row.Site - 1
    x = (col * ww + field % ww) * tw
    y = field // ww * th
    zimg[row.Channel, y:y+th, x:x+tw] = coloc.imread(row.Path)
    if row.PathV5 not in loaded_v5:
        zimg[marker_to_channel['V5'], y:y+th, x:x+tw] = coloc.imread(row.PathV5)
        loaded_v5.add(row.PathV5)

pyramids = []
for i in tqdm.tqdm(range(zimg.shape[0]), desc='generating image pyramids'):
    p = [da.from_zarr(zimg, zarr_format=2)[i]]
    p.append(p[0][::4, ::4].compute())
    p.append(p[1][::4, ::4].copy())
    pyramids.append(p)

colors = ("gray", "red", "green", "bop blue")
viewer = napari.Viewer()
for c, p in zip(colors, pyramids):
    viewer.add_image(
        p,
        contrast_limits=(0, 65535),
        colormap=c,
        blending="additive",
    )
napari.run()

import concurrent.futures
import napari
import numpy as np
import pandas as pd
import sys
import tifffile
import tqdm

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
dfs = pd.read_parquet(sys.argv[2]) if sys.argv[2].endswith('.parquet') else pd.read_csv(sys.argv[2])
plate, row = sys.argv[3:]

df = df[(df.Plate==int(plate)) & (df.Well.str.startswith(row))].copy()
if len(df) == 0:
    print("Requested plate and row are not present in the dataset")
    sys.exit(1)
dfs = dfs[(dfs.Plate==int(plate)) & (dfs.Well.str.startswith(row))].copy()
dfs = pd.merge(df[['Plate', 'Well', 'Site']], dfs)
dfs['Label'] = dfs['Label'].astype(int)

df['Channel'] = df.Marker.map(marker_to_channel)

wws = 400
th, tw = coloc.imread(df.iloc[0].Path).shape
wh = 3 if df.Site.max() <= 9 else 4
ww = 3
ih = wh * th
iw = 12 * (ww * tw + wws)
zimg = np.zeros( shape=(4 * 2, ih, iw), dtype='uint16')
zmask = np.zeros( shape=(ih, iw), dtype='uint32')

loaded_v5 = set()
for t in tqdm.tqdm(df.itertuples(), total=len(df), desc='loading images'):
    col = int(t.Well[1:]) - 1
    field = t.Site - 1
    x = (col * ww + field % ww) * tw + col * wws
    y = field // ww * th
    field = coloc.imread(t.Path)
    prep = coloc.prepare_dna if t.Marker == 'Hoechst33342' else coloc.prepare_marker
    zimg[t.Channel, y:y+th, x:x+tw] = field
    zimg[t.Channel + 4, y:y+th, x:x+tw] = prep(field)
    if t.PathV5 not in loaded_v5:
        ch_v5 = marker_to_channel['V5']
        field = coloc.imread(t.PathV5)
        zimg[ch_v5, y:y+th, x:x+tw] = field
        zimg[ch_v5 + 4, y:y+th, x:x+tw] = coloc.prepare_v5(field, t.V5ControlLevel)
        dfs.loc[(dfs.Site == t.Site) & (dfs.Well == t.Well), ['X', 'Y']] += [x, y]
        loaded_v5.add(t.PathV5)
    zmask[y:y+th, x:x+tw] = tifffile.imread(f'out/masks/{t.Plate}/{t.Well}_{t.Site}.tif')

pyramids = []
for i in tqdm.tqdm(range(zimg.shape[0]), desc='generating image pyramids'):
    p = [zimg[i]]
    for _ in range(4):
        p.append(p[-1][::2, ::2].copy())
    pyramids.append(p)
mpyramid = [zmask]
for _ in range(4):
    mpyramid.append(mpyramid[-1][::4, ::4].copy())

ew = 100
ec = '#303030'
bbox_rects = np.array([
    [
        [0 - ew / 2, x - ew / 2],
        [wh * th + ew / 2, x + ww * tw + ew / 2],
    ]
    for x in np.arange(12) * (ww * tw + wws)
])
features = pd.DataFrame([{'Well': f'{row}{i:02}'} for i in range(1, 12 + 1)])
well_markers = (
    df.groupby(['Well', 'Marker', 'Channel'])
    .first()
    .index
    .to_frame(index=False)
    .pivot(index='Well', columns='Channel', values='Marker')
    .reset_index()
    .rename(columns={2: 'Marker1', 3: 'Marker2'})
    .drop(columns=0)
)
features = pd.merge(features, well_markers, how='left').fillna('')
text_parameters1 = {
    'string': '{Well}',
    'size': 24,
    'color': '#ffffff',
    'anchor': 'upper_left',
    'translation': [-ew, 0],
}
text_parameters2 = {
    'string': '{Marker1}\n ',
    'size': 12,
    'color': '#00ff00',
    'anchor': 'upper_right',
    'translation': [-ew, 0],
}
text_parameters3 = {
    'string': '{Marker2}',
    'size': 12,
    'color': '#0080ff',
    'anchor': 'upper_right',
    'translation': [-ew, 0],
}

colors = ("gray", "red", "green", "bop blue")
channels = ('Hoechst', 'V5', 'TRITC', 'Cy5')
channels_raw = tuple(f'{c} (raw)' for c in channels)

def update_thumbnail(layer):
    layer.thumbnail = np.ones(layer._thumbnail_shape) * layer.colormap.map(0.7)

viewer = napari.Viewer()
for c, n, p in zip(colors * 2, channels_raw + channels, pyramids):
    layer = viewer.add_image(
        p,
        contrast_limits=(0, 65535),
        colormap=c,
        name=n,
        blending='additive',
        visible='raw' not in n,
    )
    layer._update_thumbnail = update_thumbnail.__get__(layer)
    layer._update_thumbnail()
viewer.add_labels(
    mpyramid,
    name='Segmentation',
)
viewer.add_shapes(
    bbox_rects,
    face_color='transparent',
    edge_color=ec,
    edge_width=ew,
    opacity=1,
    features=features,
    text=text_parameters1,
    name='Well Annotations',
)
viewer.add_shapes(
    bbox_rects,
    face_color='transparent',
    edge_color='transparent',
    features=features,
    text=text_parameters2,
    name='TRITC Markers',
)
viewer.add_shapes(
    bbox_rects,
    face_color='transparent',
    edge_color='transparent',
    features=features,
    text=text_parameters3,
    name='Cy5 Markers',
)
viewer.add_points(
    dfs[['Y','X']],
    face_color='white',
    border_width=0,
    size=0,
    antialiasing=0,
    features=dfs[['Label']],
    text=dict(
        string='{Label}',
        size=12,
        color='#ffffff',
        anchor='center',
    ),
    visible=False,
)
viewer.layers.selection = []
napari.run()

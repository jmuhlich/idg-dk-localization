import concurrent.futures
import napari
import numpy as np
import os
import pandas as pd
import pathlib
import sys
import tifffile
import tqdm

import coloc


marker_to_channel = {
    'Hoechst33342': 0,
    'V5': 1,
    'streptavidin': 3,
    'EdU': 2,
    'gH2AX': 3,
    'LAMP': 2,
    'lamin': 3,
    'b-tubulin': 2,
    'GM-130': 3,
    'calnexin': 2,
    'cytochromeC': 3,
}
channel_to_marker = {
    0: 'Hoechst33342',
    1: 'V5',
    2: 'TRITC',
    3: 'Cy5',
}

df = pd.read_csv(sys.argv[1])
dfs = pd.read_parquet(sys.argv[2]) if sys.argv[2].endswith('.parquet') else pd.read_csv(sys.argv[2])
plate, row = sys.argv[3:]

base = pathlib.Path(sys.argv[1]).parent

if row == 'A':
    v5control = True
    df = df[df.Plate == int(plate)]
    #df['Well'] = 'A' + df['column'].astype(str).str.zfill(2)
    dfwe = df[['Well', 'Experiment']].value_counts().index.to_frame(index=False)
    df = pd.concat([
        coloc.parse_paths(base / 'in' / r.Experiment, r.Well)
        #.query('Channel in (1,2)')
        .assign(Well=r.Well, Plate=int(plate))
        for r in dfwe.itertuples()
    ])
    df.Channel -= 1
    df['Marker'] = df.Channel.map(channel_to_marker)
    df = pd.merge(
        df,
        df.loc[df.Channel==1, ['Plate', 'Well', 'Site', 'Path']],
        on=['Plate', 'Well', 'Site'],
        suffixes=['', 'V5'],
    )
else:
    v5control = False
    df = df[(df.Plate == int(plate)) & (df.Well.str.startswith(row)) & df.Path.notna()].copy()
    df['Channel'] = df.Marker.map(marker_to_channel)

if len(df) == 0:
    print("Requested plate and row are not present in the dataset")
    sys.exit(1)
dfs = dfs[(dfs.Plate==int(plate)) & (dfs.Well.str.startswith(row))].copy()
dfs = pd.merge(df[['Plate', 'Well', 'Site']], dfs)
dfs['Label'] = dfs['Label'].astype(int)

wws = 400
th, tw = coloc.imread(df.iloc[-1].Path).shape
# We know max site will always be <= 12 for this dataset.
wh = 3 if df.Site.max() <= 9 else 4
ww = 3
ih = wh * th
iw = 12 * (ww * tw + wws)
zimg = np.zeros( shape=(4 * 2, ih, iw), dtype='uint16')
zmask = np.zeros( shape=(ih, iw), dtype='uint32')

num_workers = min(len(os.sched_getaffinity(0)), 8)

# V5 images are duplicated across multiple rows in df. We'll use this set to
# track which ones we've loaded to avoid redundant work.
loaded_v5 = set()
def load(t):
    col = int(t.Well[1:]) - 1
    field = t.Site - 1
    x = (col * ww + field % ww) * tw + col * wws
    y = field // ww * th
    try:
        field = coloc.imread(t.Path)
    except FileNotFoundError:
        print(f"Missing file: {t.Path}")
        return
    prep = coloc.prepare_dna if t.Marker == 'Hoechst33342' else coloc.prepare_marker
    zimg[t.Channel, y:y+th, x:x+tw] = field
    zimg[t.Channel + 4, y:y+th, x:x+tw] = prep(field)
    if t.PathV5 not in loaded_v5:
        ch_v5 = marker_to_channel['V5']
        field = coloc.imread(t.PathV5)
        zimg[ch_v5, y:y+th, x:x+tw] = field
        if hasattr(t, 'V5ControlLevel'):
            zimg[ch_v5 + 4, y:y+th, x:x+tw] = coloc.prepare_v5(field, t.V5ControlLevel)
        dfs.loc[(dfs.Site == t.Site) & (dfs.Well == t.Well), ['X', 'Y']] += [x, y]
        loaded_v5.add(t.PathV5)
    mask_path = base / 'out' / 'masks' / str(t.Plate) / f'{t.Well}_{t.Site}.tif'
    if mask_path.exists():
        zmask[y:y+th, x:x+tw] = tifffile.imread(mask_path)

with concurrent.futures.ThreadPoolExecutor(num_workers) as pool:
    list(tqdm.tqdm(pool.map(load, df.itertuples()), total=len(df), desc='loading images'))

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

colors = ("gray", "green", "red", "bop blue", "gray", "yellow", "red", "bop blue")
channels = ('Hoechst', 'V5', 'TRITC', 'Cy5')
channels_raw = tuple(f'{c} (raw)' for c in channels)

def update_thumbnail(layer):
    layer.thumbnail = np.ones(layer._thumbnail_shape) * layer.colormap.map(0.7)

viewer = napari.Viewer()
for c, n, p in zip(colors, channels_raw + channels, pyramids):
    layer = viewer.add_image(
        p,
        contrast_limits=(0, 65535),
        colormap=c,
        name=n,
        blending='additive',
        visible=n in ('Hoechst (raw)', 'V5 (raw)'),
    )
    layer._update_thumbnail = update_thumbnail.__get__(layer)
    layer._update_thumbnail()
labels_layer = viewer.add_labels(
    mpyramid,
    name='Segmentation',
    visible=False,
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

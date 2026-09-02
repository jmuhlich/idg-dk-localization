import imageio.v3 as imageio
import numpy as np
import pandas as pd
import pathlib
from PIL import Image, ImageDraw, ImageFont
import skimage.exposure
import sys
import threadpoolctl
import tqdm

import coloc


def gray(img, normalize=True):
    img = skimage.util.img_as_float32(img)
    if normalize:
        vmin, vmax = np.percentile(img, [0.1, 99.9])
        img = skimage.exposure.rescale_intensity(img, in_range=(vmin, vmax))
        img = np.clip(img, 0, 1)
    img = np.dstack([img, img, img])
    return img

def draw_label(s, fnt=ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 30)):
    img = Image.new("RGB", (175,175), (0, 0, 0))
    ImageDraw.Draw(img).text((10, 175/2), s, anchor='lm', font=fnt, fill=(255,255,255))
    return np.array(img)


threadpoolctl.threadpool_limits(1)

dfm = pd.read_csv(sys.argv[1])
dfcg = pd.read_csv(sys.argv[2])

project_path = pathlib.Path(__file__).parent.resolve()

base = project_path / "out" / "figures"
print(f"Saving images to: {base}")
base.mkdir(parents=True, exist_ok=True)

CROP_WIDTH = 175
PADDING = 10

dfm['CellLine'] = dfm['CellLine'].replace('2270 (2218c)', '2218c')
dfm['PlateRowLine'] = dfm['Plate'].astype(str) + dfm['Well'].str[0] + ' ' + dfm['CellLine'].astype(str)

dfcg['plate'] = dfcg.cell_id.str.extract(r'(^\d+)')

# TEMP
dfcg = dfcg[~(dfcg['marker'].isna() | dfcg['marker'].str.contains('N/A'))]
dfcg['cx'] = dfcg['cx'].astype(int)
dfcg['cy'] = dfcg['cy'].astype(int)

hpad = np.zeros((CROP_WIDTH, PADDING, 3))

paths = []

for r in tqdm.tqdm(dfcg.itertuples(), total=len(dfcg)):
    if r.marker == 'tbd':
        marker_show = 'b-tubulin'
    else:
        marker_show = r.marker
    m = dfm[(dfm.PlateRowLine==r.cell_id) & (dfm.Well==r.well) & (dfm.Site==r.site)]
    path_dna = m[m.Marker == 'Hoechst33342'].iloc[0].Path
    m_marker = m[m.Marker == marker_show].iloc[0]
    img_v5 = coloc.imread(m_marker.PathV5, crop=False)
    if marker_show == 'Membrane':
        img_marker = coloc.calc_membrane_mask(coloc.load_mask(r.plate, r.well, r.site), crop=False) * 0.8
    else:
        img_marker = coloc.imread(m_marker.Path, crop=False)
    img_dna = coloc.imread(path_dna, crop=False)
    x1 = np.clip(r.cx - CROP_WIDTH // 2, 0, img_v5.shape[1] - CROP_WIDTH)
    # Add 400 to cy since original coordinates assumed coloc.imread(..., crop=True).
    y1 = np.clip(r.cy + 400 - CROP_WIDTH // 2, 0, img_v5.shape[0] - CROP_WIDTH)
    x2 = x1 + CROP_WIDTH
    y2 = y1 + CROP_WIDTH
    #crop_v5 = coloc.subtract_bg(img_v5[y1:y2, x1:x2])
    crop_v5 = img_v5[y1:y2, x1:x2]
    crop_marker = img_marker[y1:y2, x1:x2]
    # if marker_show != 'Membrane':
    #     crop_marker = coloc.subtract_bg(crop_marker)
    crop_dna = img_dna[y1:y2, x1:x2]
    panel_v5 = gray(crop_v5)
    panel_marker = gray(crop_marker, normalize=(marker_show != 'Membrane'))
    panel_dna = gray(crop_dna)
    panel_merge = np.dstack([panel_marker[..., 0], panel_v5[...,0], panel_dna[...,0]])
    img_out = np.hstack([panel_v5, hpad, panel_marker, hpad, panel_dna, hpad, panel_merge])
    img_out = skimage.util.img_as_ubyte(img_out)
    path_out = base / f"{r.cell_id} {r.marker}.jpg"
    paths.append(path_out)
    imageio.imwrite(path_out, img_out, quality=95)

# Stack cell line thumbnail images into a single tall image, with the same ordering as the heatmap
# clustergram.
labels = dfcg.cell_id + '\n' + dfcg.marker + '\n' + dfcg.coloc_cluster_labels.astype(str)
gallery_img = np.vstack([
    np.hstack([draw_label(label), imageio.imread(path)])
    for label, path in zip(labels, paths)
])
imageio.imwrite(base / 'heatmap_gallery.jpg', gallery_img)

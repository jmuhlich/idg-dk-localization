import imageio.v3 as imageio
import numpy as np
import pandas as pd
import pathlib
from PIL import Image, ImageDraw, ImageFont
import scipy.stats
import scipy.spatial
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


threadpoolctl.threadpool_limits(1)

dfm = pd.read_csv(sys.argv[1])
dfms = pd.read_parquet(sys.argv[2])

project_path = pathlib.Path(__file__).parent.resolve()

base = project_path / "out" / "figures_streptavidin"
print(f"Saving images to: {base}")
base.mkdir(parents=True, exist_ok=True)

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
keep_wells = dfm[dfm['Marker'] == 'streptavidin'][['Plate','Well']].drop_duplicates()
cell_line_locations = pd.merge(cell_line_locations, keep_wells)
dfms['Marker'] = dfms['Marker'].replace('Hoechst33342', 'DNA')
dfms = pd.merge(dfms, cell_line_locations)

dfmsp = dfms[dfms['QcPass'] & dfms['V5Positive']]

empty_column = np.zeros((4 * CROP_WIDTH + 3 * PADDING, CROP_WIDTH, 3))
hpad = np.zeros((CROP_WIDTH, PADDING, 3))

gbs = dfms.groupby(['PlateRowLine', 'Marker'])
gbsp = dfmsp.groupby(['PlateRowLine', 'Marker'])


def generate_figure(plate_row_line, marker):

    marker_show = marker
    try:
        cellsp = gbsp.get_group((plate_row_line, marker_show))
    except KeyError:
        print(f"No good cells for {plate_row_line} / {marker_show})")
        return None, None
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
        cells['NeighborM1Median'] ** 2
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
    if marker.lower() == 'membrane':
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
    meta = {
        'PlateRowLine': plate_row_line,
        'well': well,
        'site': site,
        'marker': marker,
        'cx': cx,
        'cy': cy,
    }
    return img_out, meta


metas = []
for plate_row_line in tqdm.tqdm(cell_line_locations['PlateRowLine'].drop_duplicates()):
    img, meta = generate_figure(plate_row_line, 'streptavidin')
    if img is not None:
        imageio.imwrite(base / f"{plate_row_line} streptavidin.jpg", img, quality=95)
        metas.append(meta)
#pd.DataFrame(metas).to_csv(base / 'figure_metadata.csv', index=False)


# Stack cell line thumbnail images into a single tall image, with the same ordering as the heatmap
# clustergram.

# fnt = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 30)

# def draw_label(s):
#     img = Image.new("RGB", (175,175), (0, 0, 0))
#     ImageDraw.Draw(img).text((10, 175/2), s, anchor='lm', font=fnt, fill=(255,255,255))
#     return np.array(img)

# dfcg['Path'] = [list(base.glob(prl + '*'))[0] for prl in dfcg['cell_id']]
# dfcg['Marker'] = [p.stem.split(' ', 2)[2] for p in dfcg['Path']]
# gallery_img = np.vstack([
#     np.hstack([
#         draw_label(f"{r.cell_id}\n{r.Marker}\n{r.coloc_cluster_labels}"),
#         imageio.imread(r.Path),
#     ])
#     for r in dfcg.itertuples()
# ])
# imageio.imwrite(base / 'heatmap_gallery.jpg', gallery_img)

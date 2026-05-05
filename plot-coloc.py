import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns
import sklearn.cluster
import sklearn.decomposition
import sys

marker_order = ['DNA', 'GM-130', 'lamin', 'LAMP', 'b-tubulin', 'calnexin', 'cytochromeC', 'Membrane']

if len(sys.argv) == 4:
    path_control = sys.argv[1]
    path_metrics = sys.argv[2]
    path_metrics_cell = sys.argv[3]
elif len(sys.argv) == 1:
    ts = str(pd.Timestamp('now').date())
    print(f"No inputs specified; defaulting to  output files from today ({ts})")
    path_control = f"controls-{ts}.csv"
    path_metrics = f"metrics-{ts}.csv"
    path_metrics_cell = f"metrics-single-cell-{ts}.parquet"
else:
    print("Usage: plot-coloc.py [ctrl.csv metrics.csv metrics-sc.parquet]")
    sys.exit(1)

print("Input files:")
print(f"  Control metrics, per field          - {path_control}")
print(f"  Colocalization metrics, per field   - {path_metrics}")
print(f"  Colocalization metrics, single-cell - {path_metrics_cell}")

dfc = pd.read_csv(path_control)
dfm = pd.read_csv(path_metrics)
dfms = pd.read_parquet(path_metrics_cell)

dfc['Column'] = dfc['Well'].str[1:].astype(int)

dfm = pd.concat([dfm, dfm.Well.str.extract(r'(?P<Row>.)(?P<Column>..)')], axis=1)
dfm['Marker'] = dfm['Marker'].replace('Hoechst33342', 'DNA')
dfm['RowName'] = dfm['Row']
dfm['Row'] = dfm['Row'].map(ord) - ord('A') + 1
dfm['Column'] = dfm['Column'].astype(int)

dfm['CellLine'] = dfm['CellLine'].replace('2270 (2218c)', '2218c')
dfm = dfm[dfm['CellLine'] != 'parental']

cell_line_locations = dfm[['Plate','Well','CellLine']].value_counts().index.to_frame(index=False)
cell_line_locations['PlateRowLine'] = (
    cell_line_locations['Plate'].astype(str)
    + cell_line_locations['Well'].str[0]
    + ' '
    + cell_line_locations['CellLine'].astype(str)
)
ignore_wells = dfm[~dfm['Marker'].isin(marker_order)][['Plate','Well']].drop_duplicates()
cell_line_locations = (
    pd.merge(cell_line_locations, ignore_wells, how='outer', indicator=True)
    .query('_merge=="left_only"')
    .drop(columns='_merge')
)
dfms['Marker'] = dfms['Marker'].replace('Hoechst33342', 'DNA')
dfms = pd.merge(dfms, cell_line_locations)

dfmsp = dfms[dfms.QcPass & dfms.V5Positive]

plot_qc = False

# g = sns.FacetGrid(dfc.assign(plate=dfc.plate.astype('category'), color=(dfc.quality<5), x=(dfc.column+(dfc.site-1)%3/4), y=2-(dfc.site-1)//3), col='plate', col_wrap=5)
# g.map_dataframe(sns.scatterplot, x='x', y='y', hue='color')
# for ax in g.axes.ravel():
#     ax.set_aspect(0.25)

if plot_qc:

    sns.jointplot(
        dfc,
        x='NumCells',
        y='ParentalV5',
        hue='QcPass',
        kind='scatter',
        alpha=0.3,
        marginal_kws=dict(cut=0),
    )

    sns.catplot(
        dfc.assign(Column=dfc.Column.astype('category')),
        col='Plate',
        col_wrap=min(dfc.Plate.nunique(), 5),
        x='Column',
        hue='Column',
        y='ParentalV5',
        log_scale=True,
        kind='swarm',
        size=3,
    )

    well_v5positive = (
        dfms[(dfms.Marker=='DNA') & dfms.V5Positive]
        .groupby(['Plate', 'Well'])
        .size()
        .map(np.log)
        .rename('V5PositiveCount')
        .reset_index()
    )
    well_v5positive['Row'] = well_v5positive['Well'].str[0].map(ord) - ord('A') + 1
    well_v5positive['Column'] = well_v5positive['Well'].str[1:].astype(int)

    sm = plt.cm.ScalarMappable(
        cmap='summer',
        norm = plt.Normalize(
            well_v5positive.V5PositiveCount.min(),
            well_v5positive.V5PositiveCount.max(),
        ),
    )
    g = sns.FacetGrid(
        well_v5positive,
        col='Plate',
        col_wrap=min(dfc.Plate.nunique(), 5),
        height=1.55,
    )
    g.map_dataframe(
        sns.scatterplot,
        x='Column',
        y='Row',
        hue='V5PositiveCount',
        s=100,
        marker='s',
        ec='none',
        palette=sm.cmap,
        hue_norm=sm.norm,
    )
    g.axes[0].set_xticks([2, 4, 6, 8, 10])
    g.axes[0].set_yticks([2, 4, 6, 8])
    g.axes[0].set_xlim(1.5, 11.5)
    g.axes[0].set_ylim(8.5, 1.5)
    g.add_legend()
    g._legend.remove()
    cbar_ax = g.figure.add_axes([.92, 0.13, 0.015, 0.8])
    g.figure.colorbar(sm, cbar_ax, label='log( mean V5PositiveCount )')

    g = sns.catplot(
        dfmsp,
        col='PlateRowLine',
        col_wrap=20,
        x='M1',
        hue='Marker',
        y='Marker',
        order=marker_order,
        kind='violin',
        inner=None,
        linewidth=0,
        cut=0,
        aspect=1,
    )
    g.set_titles('{col_name}')
    for ax in g.axes:
        ax.spines[:].set_visible(False)
        ax.tick_params('y', length=0)

dfmm = (
    dfmsp
    .groupby(['Plate', 'CellLine', 'PlateRowLine', 'Marker'])
    [['M1', 'M2']]
    .median()
    .unstack('Marker')
    .apply(scipy.stats.zscore, nan_policy='omit')
    .dropna()
    .stack(future_stack=True)
    .reset_index()
)
m1m = dfmm.set_index(['PlateRowLine', 'Marker']).reindex(marker_order, level='Marker')[['M1','M2']].unstack('Marker')
m1m = m1m['M1'] # TEMP will we use M2 here or not?
# Next lines only needed if M2 is retained.
# m1m.columns = ['-'.join(c) for c in m1m.columns]
# m1m.columns.name = 'Marker'

cell_count = (
    m1m
    .stack()
    .sort_values(ascending=False)
    .reset_index()
    .drop_duplicates('PlateRowLine')
    .set_index('PlateRowLine')[['Marker']]
    .reset_index()
    .merge(
        dfmsp.groupby(['PlateRowLine', 'Marker']).size().rename('CellCount').reset_index()
    )
    .set_index('PlateRowLine')
    .CellCount
    .reindex_like(m1m)
)

# TEMP: random_state chosen to make the cluster containing the membrane-expressing lines end up with
# label 0 so it's colored green in the Pastel2 cmap.
m1m_cluster = sklearn.cluster.KMeans(n_clusters=5, n_init=10, random_state=0)
m1m_cluster.fit(m1m, sample_weight=cell_count)
m1m_labels = m1m_cluster.labels_
m1mpca = sklearn.decomposition.PCA().fit(m1m)
m1m_X_reduced = m1mpca.transform(m1m)

if plot_qc:
    sns.pairplot(m1m)

# PCA explained variance
plt.figure()
sns.pointplot(
    pd.DataFrame({
        'ExplainedVariance': m1mpca.explained_variance_,
        'PC': range(1, m1mpca.n_components_ + 1),
    }),
    x='PC',
    y='ExplainedVariance',
    markers='none',
)

# PCA loadings
g = sns.catplot(
    pd.DataFrame(
        m1mpca.components_.T,
        columns=range(1, m1mpca.n_components_ + 1)
    )
    .set_axis(m1m.columns, axis="index")
    .loc[:, 1:4]
    .stack()
    .rename_axis(index=['Metric','PC'])
    .rename('Loading')
    .reset_index(),
    row='PC',
    x='Metric',
    y='Loading',
    kind='bar',
    aspect=3,
    height=3,
    width=0.5,
)
g.tick_params(axis='x', rotation=45)
plt.tight_layout()
for ax in g.axes.flat:
    ax.axhline(0, c='lightgray', lw=1)

# PCA first 2 dimensions, colored by K-means cluster
plt.figure()
ax = sns.scatterplot(
    pd.DataFrame({
        "PC_1": m1m_X_reduced[:,0],
        "PC_2": m1m_X_reduced[:,1],
        "Cluster": m1m_labels,
    }),
    x="PC_1",
    y="PC_2",
    hue='Cluster',
    palette='Pastel2',
)
ax.set_title("PCA first 2 dimensions")

# Parallel coordinate plot of M1 for each K-means cluster
g = sns.catplot(
    m1m
    .assign(Cluster=m1m_labels)
    .set_index('Cluster', append=True)
    .set_axis(m1m.columns, axis="columns")
    .stack()
    .rename_axis(index={None: "Marker"})
    .rename("Value")
    .reset_index(),
    col='Cluster',
    col_wrap=3,
    hue='Cluster',
    x='Marker',
    y='Value',
    palette='Pastel2')
g.map_dataframe(sns.pointplot, x='Marker', y='Value', color='black', lw=1)
for ax in g.axes:
    ax.xaxis.set_tick_params(rotation=45)
g.figure.tight_layout()

is_maybe_membrane = [
    '14F 2270c', '15C 2067', '15D 2068', '15G 2135', '22G 2031', '23E 2412', '23F 2413', '23G 2414',
    '23H 1462', '26D 1736', '26F 1759', '25C 1758', '20B 1273', '21C 2135', '30H 2289', '31D 1465',
]
is_membrane = [
    '14F 2270c', '15C 2067', '15D 2068', '15G 2135', '23E 2412', '23F 2413', '23G 2414', '26F 1759', '25C 1758',
]
best_m1_marker = (
    m1m
    .stack()
    .sort_values(ascending=False)
    .reset_index()
    .drop_duplicates('PlateRowLine')
    .set_index('PlateRowLine')
    ['Marker']
)
cell_count_cmap = mpl.cm.ScalarMappable(
    mpl.colors.LogNorm(vmin=cell_count.min(), vmax=cell_count.max()),
    cmap='pink_r',
)
best_m1_marker_cmap = dict(zip(marker_order, sns.color_palette('Set1', len(marker_order))))
cluster_label_cmap = dict(zip(sorted(set(m1m_labels)), sns.color_palette('Pastel2', len(set(m1m_labels)))))
clustermap_row_colors = pd.concat(
    [
        pd.DataFrame(index=is_membrane).assign(Membrane='seagreen'),
        # pd.DataFrame(index=is_maybe_membrane).assign(maybe_Membrane='darkgreen'),
        pd.DataFrame({
            'KMeansCluster': m1m.assign(Cluster=m1m_labels)['Cluster'].map(cluster_label_cmap),
            'BestM1': best_m1_marker.map(best_m1_marker_cmap.get),
            'CellCount': cell_count.map(cell_count_cmap.to_rgba),
        }),
    ],
    axis=1,
)

# Explicitly calculate distance matrix so we can explicitly impose a high distance between members
# of different k-means clusters.
dist = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(m1m))
# max_intra_dist = 0.0
# for i1, m1 in enumerate(m1m.index):
#     for i2, m2 in enumerate(m1m.index):
#         c1 = m1m_labels[i1]
#         c2 = m1m_labels[i2]
#         if c1 != c2:
#             # Different cluster, flag with nan for later backfilling.
#             dist[i1, i2] = np.nan
#         else:
#             # Same cluster, keep track of highest seen distance.
#             max_intra_dist = max(max_intra_dist, dist[i1, i2])
# # Fill all inter-cluster entries with the maximum intra-cluster distance. This should be enough to
# # force each k-means cluster into its own hierarchical cluster branch (with some variance across
# # different linkage methods).
# dist[np.isnan(dist)] = max_intra_dist
# row_linkage = scipy.cluster.hierarchy.linkage(
#     scipy.spatial.distance.squareform(dist), method='average'
# )
row_linkage = scipy.cluster.hierarchy.weighted(
    scipy.spatial.distance.squareform(dist),
)

# Clustermap of M1 values
g = sns.clustermap(
    m1m,
    center=0,
    vmin=-2,
    vmax=2,
    xticklabels=True,
    yticklabels=True,
    linewidth=0,
    figsize=(5,13),
    row_colors=clustermap_row_colors,
    row_linkage=row_linkage,
    col_cluster=False,
)
g.ax_heatmap.yaxis.set_tick_params(labelsize=5)
g.ax_heatmap.xaxis.set_tick_params(labelsize=10)
g.ax_row_colors.xaxis.set_tick_params(labelsize=5)
plt.legend(
    [mpl.patches.Patch(facecolor=c) for c in best_m1_marker_cmap.values()],
    best_m1_marker_cmap.keys(),
    title='Best M1',
    bbox_to_anchor=(1, 1),
    bbox_transform=plt.gcf().transFigure,
    loc='upper right',
)
plt.colorbar(cell_count_cmap, ax=g.ax_col_dendrogram, shrink=0.9, label='CellCount', location='left')

# Stack cell line thumbnail images (generated by a separate script) into a single tall image, with
# the same ordering as the heatmap clustergram.
'''
import imageio.v3
from PIL import Image, ImageDraw, ImageFont
fnt = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 30)
def draw_label(s):
    img = Image.new("RGB", (175,175), (0, 0, 0))
    ImageDraw.Draw(img).text((10, 175/2), s, anchor='lm', font=fnt, fill=(255,255,255))
    return np.array(img)
gallery_img = np.vstack([
    np.hstack([
        draw_label(f"{prl}\n{m}"),
        imageio.v3.imread(f"out/figures/{prl} {m}.jpg"),
    ])
    for prl, m in best_m1_marker.loc[m1m.index[g.dendrogram_row.reordered_ind]].items()
])
imageio.v3.imwrite('out/heatmap_gallery.jpg', gallery_img)
'''

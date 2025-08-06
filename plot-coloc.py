import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.cluster
import sklearn.decomposition
import sys

marker_order = ["DNA", "b-tubulin", "GM-130", "LAMP", "lamin", "calnexin", "cytochromeC"]

dfc = pd.read_csv(sys.argv[1])
dfm = pd.read_csv(sys.argv[2])

dfm = pd.concat([dfm, dfm.Well.str.extract(r'(?P<Row>.)(?P<Column>..)')], axis=1)
dfm['Marker'] = dfm['Marker'].replace('Hoechst33342', 'DNA')
dfm['RowName'] = dfm['Row']
dfm['Row'] = dfm['Row'].map(ord) - ord('A') + 1
dfm['Column'] = dfm['Column'].astype(int)

# g = sns.FacetGrid(dfc.assign(plate=dfc.plate.astype('category'), color=(dfc.quality<5), x=(dfc.column+(dfc.site-1)%3/4), y=2-(dfc.site-1)//3), col='plate', col_wrap=5)
# g.map_dataframe(sns.scatterplot, x='x', y='y', hue='color')
# for ax in g.axes.ravel():
#     ax.set_aspect(0.25)

mask = (dfc.quality > 0) & (dfc.parental_v5 > 0)
sns.jointplot(
    x=np.log(dfc.quality[mask]),
    y=np.log(dfc.parental_v5[mask]),
    kind='hex',
)

mask = (dfc.quality > 0) & (dfc.dna_positive > 0)
sns.jointplot(
    x=np.log(dfc.quality[mask]),
    y=np.log(dfc.dna_positive[mask]),
    kind='hex',
)

g = sns.FacetGrid(dfc, col='plate', col_wrap=min(dfc.plate.nunique(), 5))
g.map_dataframe(
    sns.boxplot,
    x='column',
    order=sorted(dfc.column.unique()),
    y='parental_v5',
    log_scale=True,
)

dfcq = dfc[dfc.quality>=25]
g = sns.catplot(
    pd.merge(
        dfcq[['plate', 'column', 'experiment']].drop_duplicates(),
        dfcq.groupby('experiment')['parental_v5'].median().reset_index(),
    ),
    col='plate',
    col_wrap=min(dfc.plate.nunique(), 5),
    x='column',
    y='parental_v5',
    height=1.5,
    aspect=1.5,
)
plt.tight_layout()

well_v5positive_mean = np.log(dfm.groupby(['Plate','Column','Row'])[['V5PositiveCount']].mean()).reset_index()
well_v5positive_mean = well_v5positive_mean[well_v5positive_mean['V5PositiveCount'] != -np.inf]
sm = plt.cm.ScalarMappable(
    cmap='summer',
    norm = plt.Normalize(
        well_v5positive_mean.V5PositiveCount.min(),
        well_v5positive_mean.V5PositiveCount.max(),
    ),
)
g = sns.FacetGrid(
    well_v5positive_mean,
    col='Plate',
    col_wrap=min(dfc.plate.nunique(), 5),
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

#dfm['ColFacet'] = 'CL=' + dfm.CellLine + ' P=' + dfm.Plate.astype(str)
#dfmq = dfm[(dfm.Quality>30) & (dfm.V5PositiveCount>5000) & (dfm.Marker!='streptavidin')]
dfmq = dfm[(dfm.Quality>100) & (dfm.V5PositiveCount/dfm.Quality>10) & (dfm.Marker!='streptavidin')]

g = sns.catplot(dfmq, col='Plate', row='RowName', x='Marker', hue='Marker', y='M1')
g.set_titles('{col_name} / {row_name}')
g.tick_params(axis='x', rotation=90)
g.figure.tight_layout()

dfmm = (
    dfmq
    .groupby(['Plate', 'CellLine', 'RowName', 'Marker'])
    [['M1', 'M2', 'R']]
    .median()
    .unstack('Marker')
    .dropna()
    .stack(future_stack=True)
    .reset_index()
)
m1m = dfmm.set_index(['Plate', 'CellLine', 'RowName', 'Marker']).unstack('Marker')['M1'][marker_order]
m1m_labels = sklearn.cluster.KMeans(n_clusters=6, n_init=10).fit(m1m).labels_
m1mpca = sklearn.decomposition.PCA().fit(m1m)
m1m_X_reduced = m1mpca.transform(m1m)

sns.pairplot(m1m)

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

g = sns.catplot(
    pd.DataFrame(
        m1mpca.components_.T,
        columns=range(1, m1mpca.n_components_ + 1)
    )
    .set_axis(["-".join(x) for x in m1m.columns], axis="index")
    .loc[:, 1:4]
    .stack()
    .rename_axis(index=['Metric-Marker','PC'])
    .rename('Loading')
    .reset_index(),
    row='PC',
    x='Metric-Marker',
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
    palette='Set1',
)
ax.set_title("PCA first 2 dimensions");

g = sns.catplot(
    m1m
    .assign(Cluster=m1m_labels)
    .set_index('Cluster', append=True)
    .set_axis(["-".join(x) for x in m1m.columns], axis="columns")
    .stack()
    .rename_axis(index={None: "Metric-Marker"})
    .rename("Value")
    .reset_index(),
    col='Cluster',
    col_wrap=3,
    hue='Cluster',
    x='Metric-Marker',
    y='Value',
    palette='Set1')
g.map_dataframe(sns.pointplot, x='Metric-Marker', y='Value', color='black', lw=1)
#for ax in g.axes:
#    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
g.figure.tight_layout()

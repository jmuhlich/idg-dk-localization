import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sys

dfc = pd.read_csv(sys.argv[1])
dfm = pd.read_csv(sys.argv[2])

dfm = pd.concat([dfm, dfm.Well.str.extract(r'(?P<Row>.)(?P<Column>..)')], axis=1)
dfm['RowName'] = dfm['Row']
dfm['Row'] = dfm['Row'].map(ord) - ord('A') + 1
dfm['Column'] = dfm['Column'].astype(int)

# g = sns.FacetGrid(dfc.assign(plate=dfc.plate.astype('category'), color=(dfc.quality<5), x=(dfc.column+(dfc.site-1)%3/4), y=2-(dfc.site-1)//3), col='plate', col_wrap=5)
# g.map_dataframe(sns.scatterplot, x='x', y='y', hue='color')
# for ax in g.axes.ravel():
#     ax.set_aspect(0.25)


sns.jointplot(
    x=np.log(dfc.quality),
    y=np.log(dfc.parental_v5),
    kind='hex',
)

sns.jointplot(
    x=np.log(dfc.quality),
    y=np.log(dfc.dna_positive),
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
g.figure.colorbar(sm, cbar_ax, label='mean V5PositiveCount')

#dfm['ColFacet'] = 'CL=' + dfm.CellLine + ' P=' + dfm.Plate.astype(str)
g = sns.catplot(dfm, col='Plate', row='RowName', x='Marker', hue='Marker', y='M1')
g.set_titles('{col_name} / {row_name}')
g.tick_params(axis='x', rotation=90)
g.figure.tight_layout()

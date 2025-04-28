from coloc import *
import seaborn as sns


def pcc(a, b):
    ccs = np.fft.ifft2(a * b.conj())
    ccsmax = ccs.max()
    aamp = np.sum(np.real(a * a.conj())) / a.size
    bamp = np.sum(np.real(b * b.conj())) / b.size
    tamp = aamp * bamp
    if tamp > 0:
       return abs(ccsmax * ccsmax.conj() / (aamp * bamp))
    else:
       return 0


threadpoolctl.threadpool_limits(1)
num_workers = len(os.sched_getaffinity(0))
pool = concurrent.futures.ThreadPoolExecutor(num_workers)

base_path = pathlib.Path('in')
df_in = pd.read_csv(base_path / 'IDG_DK_localization_metadata_240725.csv')
dfc = pd.read_csv('c.csv')

df_in = df_in[df_in['directory'].notna()].copy()
df_in['experiment'] = df_in['directory']
df_in['directory'] = df_in['directory'].map(lambda p: base_path / p)
df_in = df_in[df_in['tritc'] != 'EdU']
df_test = pd.merge(
    df_in
    #[
        # ((df_in.plate==11) & (df_in.well.isin(('G11','G12'))))
        # | ((df_in.plate==17) & (df_in.well.isin(('B08','B09'))))
        #df_in.plate.isin((11,))
    #]
    ,
    dfc[dfc.quality > 5].groupby('experiment')['parental_v5'].median().reset_index(),
)
results = []
for (experiment, row), dfer in (t := tqdm.tqdm(df_test[df_test["column"] > 3].groupby(["experiment", "row"]))):
    t.set_postfix_str(dfer.apply("PL-{0.plate},R-{0.row}".format, axis=1).iloc[0])
    dfp = (
        pd.concat(
            (
                parse_paths(t.directory, t.well).sort_values(["Site", "Channel"])
                for t in dfer.itertuples()
            ),
            ignore_index=True,
        )
    )
    parental_v5 = float(dfer.iloc[0]["parental_v5"])
    q = list(pool.map(lambda p: calc_quality(imread(p)) > 5, dfp[dfp.Channel == 1]["Path"]))
    imgs_v5 = list(pool.map(
        lambda p: np.fft.fft2(skimage.filters.laplace(np.clip(imread(p) - parental_v5, 0, 65535).astype(np.uint16))),
        dfp[dfp.Channel == 2][q]["Path"]
    ))
    imgc = [
        list(pool.map(lambda p: np.fft.fft2(skimage.filters.laplace(imread(p))), dfp[dfp.Channel == c][q]["Path"]))
        for c in [1, 3, 4]
    ]
    s = [list(pool.map(pcc, imgs_v5, imgs)) for imgs in imgc]
    b = [
        list(pool.map(pcc, *zip(*((v, c) for iv, v in enumerate(imgs_v5) for ic, c in enumerate(imgs) if iv != ic))))
        for imgs in imgc
    ]
    cell_line = dfer.iloc[0]['cell_line']
    m1 = dfer.iloc[0]['tritc']
    m2 = dfer.iloc[0]['cy5']
    dfs = pd.DataFrame(dict(zip(["DNA", m1, m2], s))).rename_axis(columns='Marker').melt(value_name="PCC")
    dfs["Pop"] = "FG"
    dfc = pd.DataFrame(dict(zip(["DNA", m1, m2], b))).rename_axis(columns='Marker').melt(value_name="PCC")
    dfc["Pop"] = "BG"
    dfa = pd.concat([dfs, dfc])
    dfa["Experiment"] = experiment
    dfa["CellLine"] = cell_line
    dfa["Row"] = row
    results.append(dfa)

pool.shutdown()

scores = pd.concat(results)

# sns.FacetGrid(
#     scores.assign(col=scores.CellLine + "/" + scores.Marker),
#     col="col",
#     hue="Pop",
# ).map(sns.kdeplot, "PCC")

dfh = pd.merge(scores[scores['Pop']=='FG'].set_index(['CellLine', 'Marker']).sort_index()['PCC'], scores[scores['Pop']=='BG'].groupby(['CellLine', 'Marker'])['PCC'].max().rename('Control'), left_index=True, right_index=True)
hits = (dfh.PCC > dfh.Control).groupby(level=[0,1]).all()

sp = pd.merge(
    pd.merge(
        scores,
        hits[hits].reset_index()[['CellLine', 'Marker']],
    ),
    df_in.groupby(['cell_line', 'experiment', 'row', 'column', 'plate']).first().reset_index().iloc[:, :5],
    left_on=['CellLine', 'Experiment', 'Row'],
    right_on=['cell_line', 'experiment', 'row'],
)
sp["Plate"] = sp["plate"]
sp["RowName"] = (sp.Row + ord('A') - 1).map(chr)
sp["Column"] = sp["column"]
sp["Loc"] = sp["Plate"].astype(str) + " / " + sp["RowName"]
del sp["cell_line"]
del sp["experiment"]
del sp["row"]
del sp["column"]
del sp["plate"]
sp = sp.sort_values(["Plate", "RowName", "Pop"])
marker_order = ["DNA", "b-tubulin", "GM-130", "LAMP", "lamin", "calnexin", "cytochromeC"]

# Stem plot of phase cross correlation values vs. background (scrambled field
# labels) controls. Only plot "hits" i.e. locations where all FG values are
# above all BG values. This is a pretty stringent cutoff!
g = sns.FacetGrid(sp, col="Marker", row="Loc", hue="Pop", col_order=marker_order, height=0.3, aspect=8, palette="Set1", margin_titles=True)
g.tight_layout = lambda *args, **kwargs: None
g.map_dataframe(lambda data, color, label: (plt.setp(stem("PCC", [1]*len(data), data=data, linefmt=color, basefmt=color, markerfmt='none')[1], 'lw', 1)))
for ax in g.axes.flat:
    if ax.texts:
        txt = ax.texts[0]
        ax.text(*txt.get_unitless_position(), txt.get_text(), transform=ax.transAxes, va='center', rotation=0)
        ax.texts[0].remove()

# g = sns.catplot(pd.merge(scores, df_in.groupby("experiment")["plate"].first().reset_index(), left_on="Experiment", right_on="experiment").sort_values('Pop'), col='plate', row='Row', x='Marker', hue='Pop', y='PCC', s=10, palette="Set1", legend=False)
# g.set_titles('{col_name} / {row_name}')
# g.tick_params(axis='x', rotation=90)

# for (c, m), dfcm in scores.groupby(['CellLine', 'Marker']):
#     print(c, m)
#     print(scipy.stats.median_test(dfcm[dfcm.Pop=="BG"].PCC, dfcm[dfcm.Pop=="FG"].PCC))
#     print()

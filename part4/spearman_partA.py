"""Calculations 4-5: Spearman rank correlations of Part A candidate rankings between models, and Part A vs Part B gain
of each selected (own-best) prompt."""
from common import *
from scipy.stats import spearmanr

# ---- Spearman: 20 round-1 candidates per task, scored on 100 Part A items by each of the 5 models ----------
mats, long = {}, []
for t in TASKS:
    sc = pd.read_csv(f"results/{t}/{RUN}/partA_scores.csv"); sc = sc[sc["round"] != 0]
    m = pd.DataFrame(index=MODELS, columns=MODELS, dtype=float)
    for a, b in itertools.product(MODELS, MODELS):
        m.loc[a, b] = 1.0 if a == b else spearmanr(sc[a], sc[b])[0]   # NaN if a model scored every candidate identically
    mats[t] = m; m.round(3).to_csv(f"{OUT_T}/spearman_{t}.csv")
    for a, b in itertools.combinations(MODELS, 2):
        long.append(dict(task=t, model_a=a, model_b=b, spearman=m.loc[a, b], n_candidates=len(sc)))
mean_mat = sum(m.fillna(0) for m in mats.values()) / sum((~m.isna()).astype(int) for m in mats.values()).replace(0, np.nan)
mean_mat.round(3).to_csv(f"{OUT_T}/spearman_mean_across_tasks.csv")
pd.DataFrame(long).round(4).to_csv(f"{OUT_T}/spearman_pairs_long.csv", index=False)

# ---- Part A gain vs Part B gain for each model's own selected prompt -------------------------------------
rows = []
for t in TASKS:
    sc = pd.read_csv(f"results/{t}/{RUN}/partA_scores.csv").set_index("cand_idx")
    sel = json.load(open(f"results/{t}/{RUN}/selections.json"))
    arr, n = load_partB(t)
    for m in MODELS:
        k = int(sel[m])
        a_start, a_sel = sc.loc[0, m], sc.loc[k, m]
        b_start, b_sel = arr[(m, "start")].mean(), arr[(m, f"best_of_{m}")].mean()
        rows.append(dict(task=t, model=m, selected_candidate=k, partA_start=a_start, partA_selected=a_sel, gain_partA=a_sel - a_start,
                         partB_start=b_start, partB_own_best=b_sel, gain_partB=b_sel - b_start, shrinkage=(a_sel - a_start) - (b_sel - b_start)))
df = pd.DataFrame(rows); df.round(4).to_csv(f"{OUT_T}/partA_vs_partB_gain.csv", index=False)
print(df.round(3).to_string())
print("mean gain A %.3f  mean gain B %.3f  mean shrinkage %.3f" % (df.gain_partA.mean(), df.gain_partB.mean(), df.shrinkage.mean()))
print(mean_mat.round(2))

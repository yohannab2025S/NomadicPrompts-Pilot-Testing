"""Six charts for Part 4. Reads part4/tables/*.csv only (so every plotted number comes from the analysis scripts)."""
from common import *
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11, "xtick.labelsize": 10, "ytick.labelsize": 10,
                     "legend.fontsize": 9, "figure.dpi": 150, "savefig.bbox": "tight", "axes.spines.top": False, "axes.spines.right": False})
SHORT = {"gpt-oss-120b": "gpt-oss\n120B", "gpt-oss-20b": "gpt-oss\n20B", "deepseek-v4.1-flash": "DeepSeek\nV4.1-Flash", "qwen3.5-9b": "Qwen3.5\n9B", "qwen3.5-4b": "Qwen3.5\n4B"}
SH1 = {k: v.replace("\n", " ") for k, v in SHORT.items()}
T = lambda f: pd.read_csv(f"{OUT_T}/{f}")

# ---- Chart 1: grouped bars --------------------------------------------------------------------------------
acc = T("accuracy_by_condition.csv")
conds = ["start", "own_best", "borrowed (mean of 4 other models' best prompts)", "random", "one_size"]
labels = ["Start (no instruction)", "Own best", "Borrowed (mean of 4 others)", "Random", "One-size-fits-all"]
cols = ["#999999", "#0072B2", "#E69F00", "#CC79A7", "#009E73"]
fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), sharey=True)
for ax, t in zip(axes, TASKS):
    d = acc[acc.task == t]; w = 0.16
    for i, (c, l, col) in enumerate(zip(conds, labels, cols)):
        x = d[d.condition == c].set_index("model").loc[MODELS]
        pos = np.arange(len(MODELS)) + (i - 2) * w
        ax.bar(pos, x.accuracy * 100, w, color=col, label=l,
               yerr=[(x.accuracy - x.ci_lo) * 100, (x.ci_hi - x.accuracy) * 100], capsize=2, error_kw={"lw": 0.9})
    ax.set_xticks(range(len(MODELS))); ax.set_xticklabels([SHORT[m] for m in MODELS]); ax.set_title(TASK_LABEL[t]); ax.set_ylim(30, 105)
axes[0].set_ylabel("Part B accuracy (%) on held-out items\n(axis starts at 30%)")
h_, l_ = axes[0].get_legend_handles_labels(); fig.legend(h_, l_, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.06))
fig.suptitle("Chart 1. Held-out (Part B) accuracy by prompt condition, model and task (95% bootstrap CIs)", y=1.02, fontsize=13)
fig.savefig(f"{OUT_F}/chart1_accuracy_by_condition.png"); plt.close(fig)

# ---- Chart 2: heatmap of headroom capture (+ raw gain) ------------------------------------------------------
pm = T("pair_measures.csv"); pm = pm[pm.scope == "ALL_TASKS"]
hc = pm[pm.measure == "headroom_capture"]
M = hc.pivot(index="source", columns="target", values="ratio").reindex(index=MODELS, columns=MODELS)
G_ = hc.pivot(index="source", columns="target", values="borrowed_gain").reindex(index=MODELS, columns=MODELS) * 100 / len(TASKS)   # mean per task
ST = hc.pivot(index="source", columns="target", values="status").reindex(index=MODELS, columns=MODELS)
ratio_pt = (hc.borrowed_gain / hc.denominator)
RP = pd.DataFrame({"s": hc.source, "t": hc.target, "v": ratio_pt}).pivot(index="s", columns="t", values="v").reindex(index=MODELS, columns=MODELS)
fig, axes = plt.subplots(1, 2, figsize=(15, 5.2), gridspec_kw={"wspace": 0.55})
annot = RP.copy().astype(object)
for s in MODELS:
    for t in MODELS:
        annot.loc[s, t] = "" if s == t else (f"{RP.loc[s, t]:.2f}" + ("†" if str(ST.loc[s, t]).startswith("NA") else ""))
sns.heatmap(RP.clip(-1, 2), annot=annot, fmt="", cmap="RdBu_r", center=0.5, vmin=-1, vmax=2, ax=axes[0], cbar_kws={"label": "Headroom capture (clipped to [-1, 2])"}, linewidths=.5, linecolor="white")
sns.heatmap(G_, annot=True, fmt=".1f", cmap="RdBu_r", center=0, vmin=-8, vmax=8, ax=axes[1], cbar_kws={"label": "Mean gain over target's start (pp)"}, linewidths=.5, linecolor="white")
for ax, ttl in zip(axes, ["Headroom capture (point estimate)", "Raw gain from borrowed prompt (mean per task, pp)"]):
    ax.set_title(ttl); ax.set_xlabel("Target model (prompt used here)"); ax.set_ylabel("Source model (prompt written for)" if ax is axes[0] else "")
    ax.set_xticklabels([SH1[m.get_text()] for m in ax.get_xticklabels()], rotation=30, ha="right"); ax.set_yticklabels([SH1[m.get_text()] for m in ax.get_yticklabels()], rotation=0)
fig.suptitle("Chart 2. Transfer of each source model's best prompt to each target model, all three tasks summed", y=1.02, fontsize=13)
fig.savefig(f"{OUT_F}/chart2_headroom_heatmap.png"); plt.close(fig)

# ---- Chart 3: pooled ratios dot plot --------------------------------------------------------------------------
pg = T("pooled_groups.csv"); pg = pg[pg.scope == "ALL_TASKS"]
order = [("same_provider", "Same company"), ("cross_provider", "Different company"), ("large_to_small", "Larger → smaller"), ("small_to_large", "Smaller → larger")]
XL = (-4, 8)
fig, axes = plt.subplots(1, 2, figsize=(13, 3.9), sharey=True)
for ax, kind, ttl in zip(axes, ["headroom_capture", "retention"], ["Pooled headroom capture", "Pooled retention"]):
    for i, (g, lab) in enumerate(order[::-1]):
        r = pg[(pg.group == g) & (pg.measure == kind)].iloc[0]
        lo, hi = max(r.ci_lo, XL[0]), min(r.ci_hi, XL[1])
        ax.plot([lo, hi], [i, i], color="#0072B2", lw=2); ax.plot(r.pooled_ratio, i, "o", color="#0072B2", ms=8)
        if r.ci_hi > XL[1]: ax.annotate(f"CI to {r.ci_hi:.1f} →", (XL[1], i + .18), ha="right", fontsize=9)
        if r.ci_lo < XL[0]: ax.annotate(f"← CI to {r.ci_lo:.1f}", (XL[0], i + .18), ha="left", fontsize=9)
    ax.axvline(0, color="grey", lw=.8); ax.axvline(1, color="grey", lw=.8, ls="--"); ax.set_xlim(*XL)
    ax.set_yticks(range(4)); ax.set_yticklabels([l for _, l in order[::-1]]); ax.set_xlabel("Pooled ratio (0 = no gain kept, 1 = all of the target's own-best gain)"); ax.set_title(ttl)
fig.suptitle("Chart 3. Pooled transfer ratios by group, all tasks (95% paired bootstrap intervals)", y=1.04, fontsize=13)
fig.savefig(f"{OUT_F}/chart3_pooled_ratio_dotplot.png"); plt.close(fig)

# ---- Chart 4: slope chart Part A gain vs Part B gain --------------------------------------------------------
ab = T("partA_vs_partB_gain.csv"); tc = dict(zip(TASKS, ["#0072B2", "#D55E00", "#009E73"])); mk = dict(zip(MODELS, ["o", "s", "^", "D", "v"]))
fig, ax = plt.subplots(figsize=(8.4, 5.4))
for _, r in ab.iterrows():
    ax.plot([0, 1], [r.gain_partA * 100, r.gain_partB * 100], color=tc[r.task], alpha=.55, lw=1.4)
    ax.plot([0, 1], [r.gain_partA * 100, r.gain_partB * 100], mk[r.model], color=tc[r.task], ms=6)
ax.axhline(0, color="grey", lw=.8); ax.set_xticks([0, 1]); ax.set_xticklabels(["Part A\n(selection set, 100 items)", "Part B\n(held-out items)"]); ax.set_xlim(-.25, 1.25)
ax.set_ylabel("Gain of own-best prompt over start (percentage points)")
from matplotlib.lines import Line2D
h = [Line2D([0], [0], color=tc[t], lw=2, label=TASK_LABEL[t]) for t in TASKS] + [Line2D([0], [0], marker=mk[m], color="k", ls="", label=SH1[m]) for m in MODELS]
ax.legend(handles=h, fontsize=9, frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))
ax.set_title("Chart 4. Gain shrinks on new questions (15 selected prompts)", fontsize=12)
fig.savefig(f"{OUT_F}/chart4_partA_vs_partB_slope.png"); plt.close(fig)

# ---- Chart 5: Spearman matrix ---------------------------------------------------------------------------------
sm = pd.read_csv(f"{OUT_T}/spearman_mean_across_tasks.csv", index_col=0).loc[MODELS, MODELS]
fig, ax = plt.subplots(figsize=(6.4, 5.4))
sns.heatmap(sm, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, center=0, ax=ax, cbar_kws={"label": "Spearman ρ (mean over tasks)"}, linewidths=.5, linecolor="white")
ax.set_xticklabels([SH1[m.get_text()] for m in ax.get_xticklabels()], rotation=30, ha="right"); ax.set_yticklabels([SH1[m.get_text()] for m in ax.get_yticklabels()], rotation=0)
ax.set_xlabel("Model"); ax.set_ylabel("Model"); ax.set_title("Chart 5. Agreement between models on which of 20 candidate\nprompts is better (Part A rankings)", fontsize=12)
fig.savefig(f"{OUT_F}/chart5_spearman_matrix.png"); plt.close(fig)

# ---- Chart 6: share of transfers that kept most of the gain, by change type (needs human coding) -----------------
f6 = f"{OUT_T}/share_kept_by_change.csv"
if os.path.exists(f6):
    s = pd.read_csv(f6); d = s[s.group == "has change"].dropna(subset=["share_kept_most"])
    fig, ax = plt.subplots(figsize=(7.5, 4.4)); ax.bar(d.category.str.replace("_", " "), d.share_kept_most * 100, color="#0072B2")
    for i, (v, n) in enumerate(zip(d.share_kept_most * 100, d.n_transfers)): ax.text(i, v + 1.5, f"n={n}", ha="center", fontsize=9)
    ax.set_ylabel("Transfers keeping ≥50% of target's gain (%)"); ax.set_xlabel("Type of prompt change in the source prompt"); ax.set_ylim(0, 105)
    plt.xticks(rotation=25, ha="right"); ax.set_title("Chart 6. Share of transfers that kept most of the gain, by change type", fontsize=12)
    fig.savefig(f"{OUT_F}/chart6_kept_by_change.png"); plt.close(fig)
else:
    print("Chart 6 skipped: run coding_and_kappa.py after coding_sheet.csv exists")
print("charts done")

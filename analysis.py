"""Sections 5-6: Part B table, headroom capture / retention with paired bootstrap CIs, pooled group measures,
group-difference bootstrap, rank-correlation matrix. Usage: python analysis.py TASK
Rule (frozen): a ratio is reported only if the 95% bootstrap CI of its denominator excludes zero; otherwise 'NA (denominator not > 0)'."""
import sys, json, itertools
import numpy as np, pandas as pd
from scipy.stats import spearmanr, kendalltau
import config

B = 10000
rng = np.random.default_rng(config.SEED)

def load(task):
    df = pd.read_csv(f"results/{task}/partB_items.csv").sort_values("item_id")
    items = sorted(df.item_id.unique())
    arr = {(m, c): g.sort_values("item_id").correct.to_numpy() for (m, c), g in df.groupby(["model", "condition"])}
    return arr, len(items)

def groups_of(models):
    g = {"same_provider": [], "cross_provider": [], "large_to_small": [], "small_to_large": [], "same_size_cross_provider": []}
    for s, t in itertools.permutations(models, 2):
        ms, mt = config.MODELS[s], config.MODELS[t]
        (g["same_provider"] if ms["provider_group"] == mt["provider_group"] else g["cross_provider"]).append((s, t))
        if ms["size"] == "large" and mt["size"] == "small": g["large_to_small"].append((s, t))
        elif ms["size"] == "small" and mt["size"] == "large": g["small_to_large"].append((s, t))
        elif ms["provider_group"] != mt["provider_group"]: g["same_size_cross_provider"].append((s, t))
    return g

def run(task):
    arr, n = load(task)
    models = list(config.MODELS)
    idx = rng.integers(0, n, size=(B, n))                    # one shared resample for ALL conditions = paired bootstrap
    acc = lambda m, c: arr[(m, c)].mean()
    boot = lambda m, c: arr[(m, c)][idx].mean(axis=1)
    # --- table of Part B accuracies
    rows = []
    for m in models:
        r = dict(model=m, start=acc(m, "start"), own_best=acc(m, f"best_of_{m}"),
                 random=acc(m, "random"), one_size=acc(m, "one_size"))
        for s in models:
            if s != m: r[f"borrowed_from_{s}"] = acc(m, f"best_of_{s}")
        rows.append(r)
    pd.DataFrame(rows).round(3).to_csv(f"results/{task}/partB_table.csv", index=False)
    # --- per directed pair
    gain_own = {m: boot(m, f"best_of_{m}") - boot(m, "start") for m in models}
    pair_rows = []
    for s, t in itertools.permutations(models, 2):
        num = boot(t, f"best_of_{s}") - boot(t, "start")
        for name, den in (("headroom_capture", gain_own[t]), ("retention", gain_own[s])):
            point_den = (acc(s if name == "retention" else t, f"best_of_{s if name=='retention' else t}") -
                         acc(s if name == "retention" else t, "start"))
            point_num = acc(t, f"best_of_{s}") - acc(t, "start")
            lo_den = np.percentile(den, 2.5)
            row = dict(source=s, target=t, measure=name, borrowed_gain=round(point_num, 4), denominator=round(point_den, 4),
                       borrowed_gain_ci=f"[{np.percentile(num,2.5):.3f}, {np.percentile(num,97.5):.3f}]",
                       denominator_ci=f"[{lo_den:.3f}, {np.percentile(den,97.5):.3f}]")
            if lo_den > 0:
                ok = den > 0
                ratio = num[ok] / den[ok]
                row.update(ratio=round(point_num / point_den, 3), ratio_ci=f"[{np.percentile(ratio,2.5):.2f}, {np.percentile(ratio,97.5):.2f}]")
            else:
                row.update(ratio="NA (denominator CI includes 0)", ratio_ci="")
            pair_rows.append(row)
    pd.DataFrame(pair_rows).to_csv(f"results/{task}/pair_measures.csv", index=False)
    # --- pooled group measures + group differences (paired bootstrap)
    G = groups_of(models)
    def pooled(pairs, kind):
        num = sum(boot(t, f"best_of_{s}") - boot(t, "start") for s, t in pairs)
        den = sum(gain_own[t if kind == "headroom" else s] for s, t in pairs)
        return num, den
    pool_rows, draws = [], {}
    for g, pairs in G.items():
        for kind in ("headroom", "retention"):
            num, den = pooled(pairs, kind)
            ratio = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)
            draws[(g, kind)] = ratio
            pn = sum(acc(t, f"best_of_{s}") - acc(t, "start") for s, t in pairs)
            pd_ = sum((acc(t, f"best_of_{t}") - acc(t, "start")) if kind == "headroom" else (acc(s, f"best_of_{s}") - acc(s, "start")) for s, t in pairs)
            pool_rows.append(dict(group=g, measure=f"pooled_{kind}", n_pairs=len(pairs), total_borrowed_gain=round(pn, 4),
                                  total_denominator=round(pd_, 4), pooled_ratio=round(pn / pd_, 3) if pd_ > 0 else "NA",
                                  ci=f"[{np.nanpercentile(ratio,2.5):.2f}, {np.nanpercentile(ratio,97.5):.2f}]"))
    pd.DataFrame(pool_rows).to_csv(f"results/{task}/pooled_groups.csv", index=False)
    diffs = []
    for a, b in (("same_provider", "cross_provider"), ("large_to_small", "small_to_large")):
        for kind in ("headroom", "retention"):
            d = draws[(a, kind)] - draws[(b, kind)]
            diffs.append(dict(comparison=f"{a} minus {b}", measure=kind, ci=f"[{np.nanpercentile(d,2.5):.2f}, {np.nanpercentile(d,97.5):.2f}]",
                              excludes_zero=bool(np.nanpercentile(d, 2.5) > 0 or np.nanpercentile(d, 97.5) < 0)))
    pd.DataFrame(diffs).to_csv(f"results/{task}/group_differences.csv", index=False)
    # --- rank correlation of candidate rankings on Part A (40 generated candidates)
    sc = pd.read_csv(f"results/{task}/partA_scores.csv")
    sc = sc[sc["round"] != 0]
    sp = pd.DataFrame(index=models, columns=models, dtype=float); kt = sp.copy()
    for a, b in itertools.product(models, models):
        sp.loc[a, b] = spearmanr(sc[a], sc[b])[0]; kt.loc[a, b] = kendalltau(sc[a], sc[b])[0]
    sp.round(3).to_csv(f"results/{task}/rank_spearman.csv"); kt.round(3).to_csv(f"results/{task}/rank_kendall.csv")
    print(pd.DataFrame(rows).round(3).to_string()); print(pd.DataFrame(pool_rows).to_string()); print(pd.DataFrame(diffs).to_string()); print(sp.round(2))

if __name__ == "__main__":
    run(sys.argv[1])

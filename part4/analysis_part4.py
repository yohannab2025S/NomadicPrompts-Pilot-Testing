"""Part 4 calculations 1-3: accuracy tables, headroom capture / retention, pooled group ratios.
Frozen denominator rule (analysis.py): a ratio is reported only if the 95% bootstrap CI of its denominator
lies above 0; otherwise NA. Pooled ratios are ratio-of-sums (sum of borrowed gains / sum of denominators).
One resample of Part B item indices per task is shared by EVERY model and condition (paired bootstrap)."""
from common import *

rng = np.random.default_rng(config.SEED)
data, idx = {}, {}
for t in TASKS:
    data[t], n = load_partB(t)
    idx[t] = rng.integers(0, n, size=(B, n))
boot = lambda t, m, c: data[t][(m, c)][idx[t]].mean(axis=1)       # B-vector of resampled accuracy
point = lambda t, m, c: data[t][(m, c)].mean()
cond_own = lambda m: f"best_of_{m}"

# ---- 1. accuracy by model x task x condition -------------------------------------------------
rows = []
for t in TASKS:
    for m in MODELS:
        conds = {"start": boot(t, m, "start"), "own_best": boot(t, m, cond_own(m)),
                 "borrowed (mean of 4 other models' best prompts)": np.mean([boot(t, m, cond_own(s)) for s in MODELS if s != m], axis=0),
                 "random": boot(t, m, "random"), "one_size": boot(t, m, "one_size")}
        pts = {"start": point(t, m, "start"), "own_best": point(t, m, cond_own(m)),
               "borrowed (mean of 4 other models' best prompts)": np.mean([point(t, m, cond_own(s)) for s in MODELS if s != m]),
               "random": point(t, m, "random"), "one_size": point(t, m, "one_size")}
        for c in conds:
            lo, hi = pct(conds[c])
            rows.append(dict(task=t, model=m, condition=c, accuracy=pts[c], ci_lo=lo, ci_hi=hi, n_items=len(data[t][(m, "start")]), n_runs=1))
pd.DataFrame(rows).round(4).to_csv(f"{OUT_T}/accuracy_by_condition.csv", index=False)

# ---- 2. headroom capture + retention per directed pair (per task and all tasks pooled) ----------
def gain(t, m, c, b=True):   # borrowed/own gain over the model's own start prompt (B-vector or scalar)
    f = boot if b else point
    return f(t, m, c) - f(t, m, "start")

def ratio_row(num_b, den_b, num_p, den_p):
    lo_den, hi_den = pct(den_b)
    row = dict(borrowed_gain=num_p, borrowed_gain_lo=pct(num_b)[0], borrowed_gain_hi=pct(num_b)[1],
               denominator=den_p, denominator_lo=lo_den, denominator_hi=hi_den)
    if lo_den > 0:
        ok = den_b > 0
        r = num_b[ok] / den_b[ok]
        row.update(ratio=num_p / den_p, ratio_lo=pct(r)[0], ratio_hi=pct(r)[1], status="ok")
    else:
        row.update(ratio=np.nan, ratio_lo=np.nan, ratio_hi=np.nan, status="NA (denominator CI includes 0)")
    return row

pair_rows = []
for scope in TASKS + ["ALL_TASKS"]:
    ts = TASKS if scope == "ALL_TASKS" else [scope]
    for s, t_ in itertools.permutations(MODELS, 2):
        num_b = sum(gain(t, t_, cond_own(s)) for t in ts); num_p = sum(gain(t, t_, cond_own(s), False) for t in ts)
        for measure, dm in (("headroom_capture", t_), ("retention", s)):
            den_b = sum(gain(t, dm, cond_own(dm)) for t in ts); den_p = sum(gain(t, dm, cond_own(dm), False) for t in ts)
            pair_rows.append(dict(scope=scope, source=s, target=t_, measure=measure, **ratio_row(num_b, den_b, num_p, den_p)))
pd.DataFrame(pair_rows).round(4).to_csv(f"{OUT_T}/pair_measures.csv", index=False)

# ---- 3. pooled group ratios + differences -----------------------------------------------------
G = groups_of(MODELS)
pool_rows, draws = [], {}
for scope in TASKS + ["ALL_TASKS"]:
    ts = TASKS if scope == "ALL_TASKS" else [scope]
    for g, pairs in G.items():
        for kind in ("headroom_capture", "retention"):
            num_b = sum(gain(t, t_, cond_own(s)) for s, t_ in pairs for t in ts)
            num_p = sum(gain(t, t_, cond_own(s), False) for s, t_ in pairs for t in ts)
            dm = (lambda s, t_: t_) if kind == "headroom_capture" else (lambda s, t_: s)
            den_b = sum(gain(t, dm(s, t_), cond_own(dm(s, t_))) for s, t_ in pairs for t in ts)
            den_p = sum(gain(t, dm(s, t_), cond_own(dm(s, t_)), False) for s, t_ in pairs for t in ts)
            r = np.where(den_b > 0, num_b / np.where(den_b > 0, den_b, 1), np.nan)
            draws[(scope, g, kind)] = r
            lo, hi = pct(r); dlo, dhi = pct(den_b)
            pool_rows.append(dict(scope=scope, group=g, measure=kind, n_pairs=len(pairs), total_borrowed_gain=num_p,
                                  total_denominator=den_p, denominator_lo=dlo, denominator_hi=dhi,
                                  pooled_ratio=num_p / den_p if den_p > 0 else np.nan,
                                  ci_lo=lo, ci_hi=hi, frac_draws_den_positive=float(np.mean(den_b > 0)),
                                  status="ok" if dlo > 0 else "unstable (denominator CI includes 0)"))
pd.DataFrame(pool_rows).round(4).to_csv(f"{OUT_T}/pooled_groups.csv", index=False)
diff_rows = []
for scope in TASKS + ["ALL_TASKS"]:
    for a, b in (("same_provider", "cross_provider"), ("large_to_small", "small_to_large")):
        for kind in ("headroom_capture", "retention"):
            d = draws[(scope, a, kind)] - draws[(scope, b, kind)]
            pa = [r for r in pool_rows if (r["scope"], r["group"], r["measure"]) == (scope, a, kind)][0]["pooled_ratio"]
            pb = [r for r in pool_rows if (r["scope"], r["group"], r["measure"]) == (scope, b, kind)][0]["pooled_ratio"]
            lo, hi = pct(d)
            diff_rows.append(dict(scope=scope, comparison=f"{a} minus {b}", measure=kind, difference=pa - pb, ci_lo=lo, ci_hi=hi,
                                  excludes_zero=bool(lo > 0 or hi < 0)))
pd.DataFrame(diff_rows).round(4).to_csv(f"{OUT_T}/group_differences.csv", index=False)
print("done: accuracy, pair measures, pooled groups, differences")

# ---- 4. extra: own-best gain per task x model, and mean gain over start by condition (15 task x model cells) ----------
og = []
for t in TASKS:
    for m in MODELS:
        g = gain(t, m, cond_own(m)); lo, hi = pct(g)
        og.append(dict(task=t, model=m, own_best_gain=gain(t, m, cond_own(m), False), ci_lo=lo, ci_hi=hi, ci_excludes_zero=bool(lo > 0 or hi < 0)))
pd.DataFrame(og).round(4).to_csv(f"{OUT_T}/own_best_gain.csv", index=False)
cs_rows = []
for name, f in (("own_best", lambda m: cond_own(m)), ("random", lambda m: "random"), ("one_size", lambda m: "one_size")):
    g = sum(gain(t, m, f(m)) for t in TASKS for m in MODELS) / (len(TASKS) * len(MODELS))
    gp = sum(gain(t, m, f(m), False) for t in TASKS for m in MODELS) / (len(TASKS) * len(MODELS))
    cs_rows.append(dict(condition=name, mean_gain_over_start=gp, ci_lo=pct(g)[0], ci_hi=pct(g)[1]))
g = sum(gain(t, m, cond_own(s)) for t in TASKS for m in MODELS for s in MODELS if s != m) / (len(TASKS) * len(MODELS) * 4)
gp = sum(gain(t, m, cond_own(s), False) for t in TASKS for m in MODELS for s in MODELS if s != m) / (len(TASKS) * len(MODELS) * 4)
cs_rows.append(dict(condition="borrowed (all 60 task x pair cells)", mean_gain_over_start=gp, ci_lo=pct(g)[0], ci_hi=pct(g)[1]))
pd.DataFrame(cs_rows).round(6).to_csv(f"{OUT_T}/condition_summary.csv", index=False)
print(pd.DataFrame(cs_rows).round(4).to_string()); print(pd.DataFrame(og).round(3).ci_excludes_zero.sum(), "of 15 own-best gains have CI excluding 0")

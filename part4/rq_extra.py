"""Part 4 extra numbers for the RQ table, matched to the frozen Part 2 hypotheses.

Run from the repository root:  python part4/rq_extra.py
  - RQ1: pooled headroom capture over all 20 directed pairs (ratio of sums over pairs and tasks)
  - Size comparison using the within-company pairs promised in Part 3 Sec. 2.5
    (larger->smaller: gpt-oss-120b->20b, qwen3.5-9b->4b; smaller->larger: the reverse)
  - Prompt-preference agreement: mean Spearman within vs across provider
Paired bootstrap: 10,000 resamples of Part B items per task, one shared resample for all models
and conditions, seed 42. Percentile intervals; resamples with a non-positive denominator are kept.
"""
import csv
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

R = Path("results")
TASKS = ["word_sorting", "causal_judgement", "salient_translation_error_detection"]
M = ["gpt-oss-120b", "gpt-oss-20b", "deepseek-v4.1-flash", "qwen3.5-9b", "qwen3.5-4b"]
CO = {"gpt-oss-120b": "OpenAI", "gpt-oss-20b": "OpenAI", "deepseek-v4.1-flash": "DeepSeek",
      "qwen3.5-9b": "Alibaba", "qwen3.5-4b": "Alibaba"}
B = 10000
rng = np.random.default_rng(42)
data, idx = {}, {}
for t in TASKS:
    rows = list(csv.DictReader(open(R / t / "run1" / "partB_items.csv")))
    ids = sorted({r["item_id"] for r in rows}); pos = {i: k for k, i in enumerate(ids)}
    d = {}
    for r in rows:
        d.setdefault((r["model"], r["condition"]), np.zeros(len(ids)))[pos[r["item_id"]]] = int(r["correct"])
    data[t] = d; idx[t] = rng.integers(0, len(ids), size=(B, len(ids)))


def gain(t, target, prompt_of, boot):
    a = lambda c: (data[t][(target, c)][idx[t]].mean(1) if boot else data[t][(target, c)].mean())
    return (a(f"best_of_{prompt_of}") - a("start")) * 100


def pooled(pairs, kind):
    out = []
    for boot in (False, True):
        num = sum(gain(t, tg, s, boot) for s, tg in pairs for t in TASKS)
        den = sum(gain(t, tg, tg, boot) if kind == "headroom" else gain(t, s, s, boot) for s, tg in pairs for t in TASKS)
        out.append((num, den))
    (n, d), (nb, db) = out
    return n / d, nb / db


def fmt(point, boot):
    lo, hi = np.percentile(boot, [2.5, 97.5]); return f"{point:.2f} [{lo:.2f}, {hi:.2f}]"


allp = [(s, t) for s in M for t in M if s != t]
r, rb = pooled(allp, "headroom")
print("RQ1 pooled headroom capture, all 20 pairs:", fmt(r, rb))
nonpos = sum(1 for s, tg in allp if sum(gain(t, tg, s, False) for t in TASKS) <= 0)
print(f"RQ1 pairs whose borrowed gain (sum over tasks) <= 0: {nonpos} of 20")

ls = [("gpt-oss-120b", "gpt-oss-20b"), ("qwen3.5-9b", "qwen3.5-4b")]
sl = [(b, a) for a, b in ls]
for kind in ("headroom", "retention"):
    a, ab = pooled(ls, kind); b, bb = pooled(sl, kind)
    print(f"Size (within company) {kind}: larger->smaller {fmt(a, ab)}; smaller->larger {fmt(b, bb)}; "
          f"difference {fmt(a - b, ab - bb)}")

within, across = [], []
for i, a in enumerate(M):
    for b in M[i + 1:]:
        rs = []
        for t in TASKS:
            rows = [x for x in csv.DictReader(open(R / t / "run1" / "partA_scores.csv")) if x["round"] != "0"]
            rs.append(spearmanr([float(x[a]) for x in rows], [float(x[b]) for x in rows])[0])
        (within if CO[a] == CO[b] else across).append(np.mean(rs))
print(f"Spearman mean within provider {np.mean(within):.2f} (n={len(within)} pairs); "
      f"across provider {np.mean(across):.2f} (n={len(across)} pairs)")

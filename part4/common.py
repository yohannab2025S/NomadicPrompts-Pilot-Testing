"""Shared loading helpers for Part 4. Run all scripts from the repo root (see run_all.py)."""
import sys, os, itertools, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

TASKS = ["word_sorting", "causal_judgement", "salient_translation_error_detection"]
TASK_LABEL = {"word_sorting": "Word Sorting", "causal_judgement": "Causal Judgement",
              "salient_translation_error_detection": "Salient Translation Error"}
MODELS = list(config.ACTIVE_MODELS)            # gpt-oss-120b, gpt-oss-20b, deepseek-v4.1-flash, qwen3.5-9b, qwen3.5-4b
RUN = "run1"                                    # only run with complete Part A + Part B for all five models
B = 10000                                       # bootstrap draws
OUT_T, OUT_F = "part4/tables", "part4/figures"
os.makedirs(OUT_T, exist_ok=True); os.makedirs(OUT_F, exist_ok=True)

def load_partB(task):
    """Return {(model, condition): 0/1 array sorted by item_id}, n_items. Asserts identical item sets."""
    frames = [pd.read_csv(f"results/{task}/{RUN}/partB_{m}_items.csv") for m in MODELS]
    df = pd.concat(frames)
    items = None
    arr = {}
    for (m, c), g in df.groupby(["model", "condition"]):
        g = g.sort_values("item_id")
        ids = tuple(g.item_id)
        assert len(set(ids)) == len(ids), f"duplicate items {task} {m} {c}"
        items = items or ids
        assert ids == items, f"item mismatch {task} {m} {c}"
        arr[(m, c)] = g.correct.to_numpy().astype(float)
    return arr, len(items)

def groups_of(models):
    g = {"same_provider": [], "cross_provider": [], "large_to_small": [], "small_to_large": []}
    for s, t in itertools.permutations(models, 2):
        ms, mt = config.MODELS[s], config.MODELS[t]
        g["same_provider" if ms["provider_group"] == mt["provider_group"] else "cross_provider"].append((s, t))
        if ms["size"] == "large" and mt["size"] == "small": g["large_to_small"].append((s, t))
        elif ms["size"] == "small" and mt["size"] == "large": g["small_to_large"].append((s, t))
    return g

def pct(x, lo=2.5, hi=97.5):
    return float(np.nanpercentile(x, lo)), float(np.nanpercentile(x, hi))

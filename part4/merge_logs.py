"""Merge every exported call log into ONE de-duplicated file, show which model/task/phase logs are still missing, rebuild the
call/failure/token summary table (single source for Table 5.1), and draw a targeted 20-response hand-check sample.
Usage (repo root): python part4/merge_logs.py      Inputs: logs_export/*/calls.jsonl.gz (+ any new owner folders)
Outputs: logs_export/merged/calls_merged.jsonl.gz, part4/tables/call_log_summary.csv, log_coverage.csv, hand_check_targeted_20.csv"""
import glob, gzip, json, re, random, os, sys
import pandas as pd
sys.path.insert(0, ".")
import extraction
random.seed(42)
os.makedirs("logs_export/merged", exist_ok=True); os.makedirs("part4/tables", exist_ok=True)

rows = []
for f in sorted(glob.glob("logs_export/*/calls.jsonl.gz")):
    owner = f.split("/")[1]
    for line in gzip.open(f, "rt", encoding="utf-8"):
        try: d = json.loads(line)
        except Exception: continue
        d["owner"] = owner; rows.append(d)
df = pd.DataFrame(rows)
for c in ("prompt_tokens", "completion_tokens", "seconds", "ts"): df[c] = pd.to_numeric(df[c], errors="coerce")
df["failed"] = df.error.notna()
df["phase"] = df.experiment.fillna("").str.extract(r"(baseline_part3|baseline|smoke_test|-A$|-B$)", expand=False).fillna("other").str.strip("-")
df["task"] = df.item_id.fillna("").str.replace(r"_\d+$", "", regex=True)
print(f"raw log rows: {len(df)} from {df.owner.nunique()} owner folders")

# ---- de-duplicate by cache key: a successful call beats a failed one; later beats earlier -----------------------------
df = df.sort_values(["key", "failed", "ts"], ascending=[True, True, False])
first_fail = df[df.failed].groupby("key").size().rename("n_failed_attempts")
uniq = df.drop_duplicates("key", keep="first").merge(first_fail, left_on="key", right_index=True, how="left")
uniq["n_failed_attempts"] = uniq.n_failed_attempts.fillna(0).astype(int)
with gzip.open("logs_export/merged/calls_merged.jsonl.gz", "wt", encoding="utf-8") as g:
    for r in uniq.drop(columns=["failed"]).to_dict("records"): g.write(json.dumps(r, default=str) + "\n")
print(f"unique calls after merging: {len(uniq)}")

# ---- summary table: one row per model (raw failures, reruns, truncations, tokens) -------------------------------------
summ = []
for m, g in df.groupby("model_name"):
    u = uniq[uniq.model_name == m]; ok = u[~u.failed]
    summ.append(dict(model=m, unique_calls=len(u), successful=len(ok), failed_attempts_logged=int(df[(df.model_name == m) & df.failed].shape[0]),
                     failed_then_rerun_ok=int(((u.n_failed_attempts > 0) & ~u.failed).sum()), still_failed=int(u.failed.sum()),
                     truncated_finish_length=int((ok.finish_reason == "length").sum()), prompt_tokens=int(ok.prompt_tokens.sum()),
                     completion_tokens=int(ok.completion_tokens.sum()), mean_seconds_per_call=round(ok.seconds.mean(), 1)))
pd.DataFrame(summ).to_csv("part4/tables/call_log_summary.csv", index=False)
print(pd.DataFrame(summ).to_string(index=False))

# ---- coverage: which (model, task, phase) have logs vs which result files exist ----------------------------------------
MODELS = ["gpt-oss-120b", "gpt-oss-20b", "deepseek-v4.1-flash", "qwen3.5-9b", "qwen3.5-4b"]; TASKS = ["word_sorting", "causal_judgement", "salient_translation_error_detection"]
cov = []
for m in MODELS:
    for t in TASKS:
        for ph, fn in (("A", f"results/{t}/run1/partA_{m}_round1.csv"), ("B", f"results/{t}/run1/partB_{m}_items.csv")):
            n = int(((uniq.model_name == m) & (uniq.task == t) & (uniq.phase == ph) & ~uniq.failed & uniq.experiment.str.contains("run1")).sum())
            cov.append(dict(model=m, task=t, phase=ph, result_file_exists=os.path.exists(fn), run1_logged_calls=n))
cov = pd.DataFrame(cov); cov.to_csv("part4/tables/log_coverage.csv", index=False)
miss = cov[cov.result_file_exists & (cov.run1_logged_calls == 0)]
print("\nRun-1 results WITHOUT logs in the repo (hand check cannot see these responses; get the logs from whoever ran them):")
print(miss.to_string(index=False) if len(miss) else "none")

# ---- targeted hand check: wrong, truncated, unparseable and a few correct controls, spread over every model with logs -----------
tgt = {}
for t in TASKS:
    sp = json.load(open(f"data/{t}_split.json"))
    for part in ("part_a", "part_b"):
        for it in sp[part]: tgt[it["id"]] = (t, it["target"])
ok = uniq[~uniq.failed & uniq.item_id.isin(tgt) & uniq.phase.isin(["A", "B"])].copy()
ok["extracted"] = [extraction.extract_answer(tgt[i][0], r) for i, r in zip(ok.item_id, ok.response)]
ok["auto_correct"] = [extraction.is_correct(tgt[i][0], r, tgt[i][1]) for i, r in zip(ok.item_id, ok.response)]
ok["category"] = "correct (control)"
ok.loc[~ok.auto_correct, "category"] = "wrong answer"; ok.loc[ok.extracted.isna(), "category"] = "no answer extracted"; ok.loc[ok.finish_reason == "length", "category"] = "truncated"
have = [m for m in MODELS if (ok.model_name == m).any()]
plan = {"truncated": 5, "no answer extracted": 3, "wrong answer": 8, "correct (control)": 4}   # target mix across all models
pick = []
for cat, n in plan.items():
    pool = ok[ok.category == cat]
    # round-robin over models so every model with logs is represented
    by = {m: pool[pool.model_name == m].sample(frac=1, random_state=42) for m in have}
    k = 0
    while sum(len(b) for b in by.values()) and len([p for p in pick if p[0] == cat]) < n:
        m = have[k % len(have)]; k += 1
        if len(by[m]): pick.append((cat, by[m].iloc[0])); by[m] = by[m].iloc[1:]
        elif k > 50: break
hc = pd.DataFrame([dict(model=r.model_name, task=r.task, item_id=r.item_id, experiment=r.experiment, category=c, finish_reason=r.finish_reason,
                        completion_tokens=r.completion_tokens, auto_extracted=r.extracted, target=tgt[r.item_id][1], auto_correct=bool(r.auto_correct),
                        response_tail=str(r.response)[-500:], human_label_correct=None, extraction_error_Y_N=None, notes=None) for c, r in pick])
hc.to_csv("part4/tables/hand_check_targeted_20.csv", index=False)
print(f"\nhand-check sample: {len(hc)} responses; models with logs: {have}; models WITHOUT usable logs: {[m for m in MODELS if m not in have]}")
print(hc.groupby(["category", "model"]).size().unstack(fill_value=0))

"""Part A scoring for ONE person's models, without touching anyone else's columns.

Stand-in for the missing collect.py. Put this file in the repo root (next to ape_pipeline.py).

Score (writes results/<task>/<run>/partA_<model>_round1.csv, same format as Arsheya's files):
    python score_model.py partA all run1 --models qwen3.5-9b --workers 4
    python score_model.py partA causal_judgement run1 --models glm-5.3 --workers 7

Merge everyone's per-model files into results/<task>/<run>/partA_scores.csv:
    python score_model.py mergeA all run1

Show what is done / missing:
    python score_model.py status all run1

After mergeA: for each task run 'python ape_pipeline.py <task> run1 select', then COMMIT AND PUSH
results/<task>/run1/selections.json before anyone starts Part B.

Part B (writes results/<task>/<run>/partB_<model>_items.csv, one row per question):
    python score_model.py partB all run1 --models qwen3.5-9b --workers 4

Merge Part B into results/<task>/<run>/partB_items.csv and print an accuracy table:
    python score_model.py mergeB all run1

Copy this computer's logs into logs_export/<owner>/ so they can be committed:
    python score_model.py export-logs --owner nick

Safe to stop and restart: finished prompts are skipped, and answered questions are reused
from logs/calls.jsonl by arc_client's cache. Use a separate terminal per model.
"""
import argparse
import csv
import gzip
import os
import shutil
import subprocess
from pathlib import Path

import config
import ape_pipeline as ape

TASK_NAMES = ["word_sorting", "causal_judgement", "salient_translation_error_detection"]
ALL_MODELS = list(config.FINAL_SETTING)


def tasks_from(arg):
    return TASK_NAMES if arg == "all" else [arg]


def model_file(task, model):
    return ape.out(task, f"partA_{model}_round1.csv")


def read_model_scores(task, model):
    path = model_file(task, model)
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        return {r["cand_idx"]: r.get(model, "") for r in csv.DictReader(f)}


def write_model_scores(task, model, rows, scores):
    path = model_file(task, model)
    tmp = path.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["cand_idx", "round", "instruction", model])
        w.writeheader()
        for r in rows:
            w.writerow({**r, model: scores.get(r["cand_idx"], "")})
    os.replace(tmp, path)


def part_a(task, model, workers):
    rows = ape.candidate_rows(task)
    if any(r["round"] == "2" for r in rows):
        raise RuntimeError(f"{task}: candidates.json has Round 2 prompts; Run 1 uses Round 1 only.")
    items = ape.bbh_data.get_split(task)["part_a"]
    scores = read_model_scores(task, model)
    setting = config.FINAL_SETTING[model]
    for i, r in enumerate(rows, 1):
        idx = r["cand_idx"]
        if scores.get(idx, "") != "":
            continue
        print(f"[{task} {ape.RUN_ID}] {model} prompt {idx} ({i}/{len(rows)})", flush=True)
        correct = ape.evaluate_batch(task, model, setting, items, r["instruction"],
                                     experiment=f"ape-{task}-{ape.RUN_ID}-A")
        scores[idx] = f"{sum(correct) / len(correct):.4f}"
        write_model_scores(task, model, rows, scores)
        print(f"  accuracy {float(scores[idx]):.1%}  saved", flush=True)
    print(f"[{task}] {model} done: {model_file(task, model)}", flush=True)


def merge_a(task):
    rows = ape.candidate_rows(task)
    per_model = {m: read_model_scores(task, m) for m in ALL_MODELS}
    path = ape.out(task, "partA_scores.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["cand_idx", "round", "instruction"] + ALL_MODELS)
        w.writeheader()
        for r in rows:
            w.writerow({**r, **{m: per_model[m].get(r["cand_idx"], "") for m in ALL_MODELS}})
    status(task)
    print(f"Wrote {path}", flush=True)


def status(task):
    rows = ape.candidate_rows(task)
    print(f"{task} {ape.RUN_ID}: {len(rows)} prompts", flush=True)
    missing_total = 0
    for m in ALL_MODELS:
        s = read_model_scores(task, m)
        done = sum(1 for r in rows if s.get(r["cand_idx"], "") != "")
        missing_total += len(rows) - done
        print(f"  {m:22s} {done:2d}/{len(rows)}", flush=True)
    print(f"  Missing cells: {missing_total}", flush=True)


# ---------------------------------------------------------------- Part B

B_FIELDS = ["task", "model", "condition", "item_id", "correct"]


def b_conditions(task):
    """Same 9 conditions as ape_pipeline.evalB: start, best_of_<each model>, random, one_size."""
    path = ape.out(task, "selections.json")
    if not path.exists():
        raise RuntimeError(f"{task}: no selections.json. Run mergeA, then "
                           f"'python3 ape_pipeline.py {task} {ape.RUN_ID} select', commit, push, pull.")
    sel = ape.load_json(path)
    missing = [m for m in ALL_MODELS if m not in sel]
    if missing:
        raise RuntimeError(f"{task}: selections.json has no best prompt for {missing}. "
                           "It was made before all 6 models were scored; redo select.")
    rows = ape.load_scores(task)  # raises if any Part A cell is missing
    instr = {r["cand_idx"]: r["instruction"] for r in rows}
    cond = {"start": ""}
    for m in ALL_MODELS:
        cond[f"best_of_{m}"] = instr[str(sel[m])]
    cond["random"] = instr[str(sel["_random"])]
    cond["one_size"] = instr[str(sel["_one_size"])]
    return cond


def warn_if_uncommitted(path):
    """Selections must be committed before Part B so they provably did not use Part B."""
    try:
        dirty = subprocess.run(["git", "status", "--porcelain", str(path)],
                               capture_output=True, text=True).stdout.strip()
        tracked = subprocess.run(["git", "ls-files", str(path)],
                                 capture_output=True, text=True).stdout.strip()
    except OSError:
        return
    if dirty or not tracked:
        print(f"WARNING: {path} is not committed. Commit and push it before Part B.", flush=True)


def b_file(task, model):
    return ape.out(task, f"partB_{model}_items.csv")


def read_b(path):
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def part_b(task, model):
    cond = b_conditions(task)
    warn_if_uncommitted(ape.out(task, "selections.json"))
    items = ape.bbh_data.get_split(task)["part_b"]
    ids = {str(i["id"]) for i in items}
    path = b_file(task, model)
    have = read_b(path)
    setting = config.FINAL_SETTING[model]
    for n, (c, instruction) in enumerate(cond.items(), 1):
        got = {r["item_id"] for r in have if r["condition"] == c}
        if got == ids:
            continue
        if got:  # half-written condition (crash mid-write): drop it and redo; cache makes this cheap
            have = [r for r in have if r["condition"] != c]
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=B_FIELDS)
                w.writeheader()
                w.writerows(have)
        print(f"[{task} {ape.RUN_ID}] {model} Part B {c} ({n}/{len(cond)})", flush=True)
        correct = ape.evaluate_batch(task, model, setting, items, instruction,
                                     experiment=f"ape-{task}-{ape.RUN_ID}-B")
        new = [dict(task=task, model=model, condition=c, item_id=i["id"], correct=k)
               for i, k in zip(items, correct)]
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=B_FIELDS)
            if f.tell() == 0:
                w.writeheader()
            w.writerows(new)
        have += [{k: str(v) for k, v in r.items()} for r in new]
        print(f"  accuracy {sum(correct) / len(correct):.1%}  saved", flush=True)
    print(f"[{task}] {model} Part B done: {path}", flush=True)


def merge_b(task):
    cond = list(b_conditions(task))
    n = len(ape.bbh_data.get_split(task)["part_b"])
    rows, problems = [], []
    print(f"{task} {ape.RUN_ID} Part B accuracy (n={n} per cell)", flush=True)
    print("  " + "model".ljust(22) + "".join(c.replace("best_of_", "b:")[:13].rjust(14) for c in cond))
    for m in ALL_MODELS:
        mine = read_b(b_file(task, m))
        line = "  " + m.ljust(22)
        for c in cond:
            cell = [r for r in mine if r["condition"] == c]
            if len(cell) != n or len({r["item_id"] for r in cell}) != n:
                problems.append(f"{m}/{c}: {len(cell)}/{n}")
                line += "missing".rjust(14)
            else:
                line += f"{sum(int(r['correct']) for r in cell) / n:.1%}".rjust(14)
        print(line, flush=True)
        rows += mine
    path = ape.out(task, "partB_items.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=B_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {path} ({len(rows)} rows)", flush=True)
    print("Complete." if not problems else "INCOMPLETE: " + "; ".join(problems), flush=True)


# ---------------------------------------------------------------- logs

def export_logs(owner):
    """Copy this computer's call logs into logs_export/<owner>/ (gzipped) so they can be committed."""
    dest = Path("logs_export") / owner
    dest.mkdir(parents=True, exist_ok=True)
    for name in [config.LOG_PATH, "logs/writer_calls.jsonl"]:
        src = Path(name)
        if not src.exists():
            continue
        target = dest / (src.name + ".gz")
        with open(src, "rb") as fi, gzip.open(target, "wb") as fo:
            shutil.copyfileobj(fi, fo)
        lines = sum(1 for _ in open(src, encoding="utf-8"))
        mb = target.stat().st_size / 1e6
        print(f"{src} -> {target}: {lines} records, {mb:.1f} MB", flush=True)
        if mb > 95:
            print("  WARNING: over GitHub's 100 MB file limit; ask before pushing.", flush=True)
    print(f"Now: git add {dest} results && git commit -m 'logs + results ({owner})' && git push",
          flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("step", choices=["partA", "mergeA", "partB", "mergeB", "status", "export-logs"])
    p.add_argument("task", nargs="?", choices=TASK_NAMES + ["all"])
    p.add_argument("run", nargs="?", choices=["run1", "run2", "run3"])
    p.add_argument("--models", nargs="+", default=[])
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--owner", default="")
    a = p.parse_args()

    if a.step == "export-logs":
        if not a.owner:
            raise SystemExit("Use: python3 score_model.py export-logs --owner <yourname>")
        export_logs(a.owner)
        return
    if not a.task or not a.run:
        raise SystemExit("Give a task (or 'all') and a run, e.g.: status all run1")

    ape.RUN_ID = a.run
    ape.RUN_SEED = config.SEED + int(a.run[-1]) - 1
    print(f"Models in FINAL_SETTING: {ALL_MODELS}", flush=True)
    if len(ALL_MODELS) != 6:
        print("WARNING: FINAL_SETTING does not list 6 models; pull the latest config.py.", flush=True)

    if a.step in ("partA", "partB"):
        bad = [m for m in a.models if m not in ALL_MODELS]
        if not a.models or bad:
            raise SystemExit(f"--models must be from {ALL_MODELS}; got {a.models}")
        # Cap parallel requests per host (each terminal is its own process).
        config.MAX_CONCURRENT_ARC = a.workers
        config.MAX_CONCURRENT_OTHER = a.workers
        for t in tasks_from(a.task):
            for m in a.models:
                if a.step == "partA":
                    part_a(t, m, a.workers)
                else:
                    part_b(t, m)
    elif a.step == "mergeA":
        for t in tasks_from(a.task):
            merge_a(t)
    elif a.step == "mergeB":
        for t in tasks_from(a.task):
            merge_b(t)
    else:
        for t in tasks_from(a.task):
            status(t)


if __name__ == "__main__":
    main()

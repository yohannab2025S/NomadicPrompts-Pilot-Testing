
"""APE pilot for one BBH task.

Usage:
    python ape_pipeline.py word_sorting gen1
    python ape_pipeline.py word_sorting score
    python ape_pipeline.py word_sorting gen2
    python ape_pipeline.py word_sorting score
    python ape_pipeline.py word_sorting select
    python ape_pipeline.py word_sorting evalB
    python ape_pipeline.py word_sorting cost

Or run all steps:
    python ape_pipeline.py word_sorting all

IMPORTANT DEVIATION:
The required Claude Sonnet 4.6 writer was unavailable.
This pilot uses DeepSeek-V4.1-Flash through VT ARC instead.
The actual returned model name is recorded in writer_calls.jsonl.

Only models in config.FINAL_SETTING are evaluated.
"""

import csv
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import config
import bbh_data
import arc_client
from extraction import is_correct


MODELS = list(config.FINAL_SETTING)
WRITER_MODEL = "deepseek-v4.1-flash"
WRITER_SETTING = "thinking-low"
WRITER_TEMPERATURE = 0

RESULTS = Path("results")
LOGS = Path("logs")
LOGS.mkdir(parents=True, exist_ok=True)


def out(task, name):
    folder = RESULTS / task
    folder.mkdir(parents=True, exist_ok=True)
    return folder / name


def save_json(path, data):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    with open(temp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(temp, path)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def parse_list(txt, expected=None):
    """Parse a JSON array, tolerating Markdown code fences."""
    txt = txt.strip()

    try:
        data = json.loads(txt)
    except json.JSONDecodeError:
        match = re.search(r"\[[\s\S]*\]", txt)
        if not match:
            raise ValueError(
                "Writer response did not contain a JSON array. "
                f"Response begins: {txt[:300]!r}"
            )
        data = json.loads(match.group(0))

    if not isinstance(data, list):
        raise ValueError("Writer response is not a JSON list.")

    if not all(isinstance(item, str) and item.strip() for item in data):
        raise ValueError("Writer returned empty or non-string instructions.")

    data = [item.strip() for item in data]

    if expected is not None and len(data) != expected:
        raise ValueError(
            f"Expected {expected} instructions, received {len(data)}."
        )

    return data


def writer_call(prompt, tag):
    """Generate instructions using ARC, recording the actual model used.

    This is a documented substitution for the required Claude writer.
    """
    start = time.time()

    jobs = [(tag, prompt)]
    records = arc_client.run_batch(
        WRITER_MODEL,
        WRITER_SETTING,
        jobs,
        experiment=f"ape-writer-{tag}",
    )

    if len(records) != 1:
        raise RuntimeError(
            f"Writer returned {len(records)} records instead of one."
        )

    record = records[0]

    if record.get("error"):
        raise RuntimeError(f"Writer API error: {record['error']}")

    txt = record.get("response")
    if not txt or not isinstance(txt, str):
        raise RuntimeError(f"Writer returned no usable text: {record}")

    entry = {
        "tag": tag,
        "requested_model": WRITER_MODEL,
        "requested_setting": WRITER_SETTING,
        "returned_model": record.get("returned_model"),
        "required_writer": "claude-sonnet-4-6",
        "writer_deviation": "Claude unavailable; ARC DeepSeek substituted",
        "prompt": prompt,
        "response": txt,
        "input_tokens": record.get("prompt_tokens"),
        "output_tokens": record.get("completion_tokens"),
        "seconds": time.time() - start,
    }

    with open(LOGS / "writer_calls.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

    return txt


def gen1(task, k=5):
    """Generate 20 initial candidate instructions."""
    path = out(task, "candidates.json")

    if path.exists():
        existing = load_json(path)
        if len(existing.get("round1", [])) == 20:
            print("Round 1 already exists; skipping generation.", flush=True)
            return

    split = bbh_data.get_split(task)
    rng = random.Random(config.SEED)
    examples = rng.sample(split["part_a"], k)

    shots = "\n\n".join(
        f"Input:\n{e['input']}\nOutput:\n{e['target']}"
        for e in examples
    )

    prompt = (
        f"Below are example inputs and outputs for a task.\n\n{shots}\n\n"
        "Write 20 DIFFERENT instructions, each a short paragraph, "
        "that could be given to a language model before an input "
        "like these so that it produces the correct output. "
        "Vary the approach: clarify the task, reasoning steps, "
        "constraints, and formatting. "
        "Do not include the answer-format line; it is added separately. "
        "Reply with ONLY a valid JSON array of exactly 20 strings."
    )

    print("Generating 20 initial instructions...", flush=True)
    candidates = parse_list(
        writer_call(prompt, f"{task}-gen1"),
        expected=20,
    )

    save_json(path, {"round1": candidates, "round2": []})
    print(f"Saved {len(candidates)} candidates to {path}", flush=True)


def candidate_rows(task):
    candidates = load_json(out(task, "candidates.json"))
    prompts = [""] + candidates["round1"] + candidates["round2"]

    rows = []
    for idx, instruction in enumerate(prompts):
        rnd = (
            0 if idx == 0
            else 1 if idx <= len(candidates["round1"])
            else 2
        )
        rows.append({
            "cand_idx": str(idx),
            "round": str(rnd),
            "instruction": instruction,
        })

    return rows


def score_columns():
    return ["cand_idx", "round", "instruction"] + MODELS


def read_score_progress(task):
    path = out(task, "partA_scores.csv")
    progress = {}

    if not path.exists():
        return progress

    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            progress[row["cand_idx"]] = row

    return progress


def save_score_progress(task, rows, progress):
    """Checkpoint after each successfully evaluated model/prompt."""
    path = out(task, "partA_scores.csv")
    temp = path.with_suffix(".csv.tmp")

    with open(temp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=score_columns())
        writer.writeheader()

        for row in rows:
            idx = row["cand_idx"]
            saved = progress.get(idx, {})
            writer.writerow({
                **row,
                **{m: saved.get(m, "") for m in MODELS},
            })

    os.replace(temp, path)


def evaluate_batch(task, model, setting, items, instruction, experiment):
    jobs = [
        (item["id"], bbh_data.build_prompt(task, instruction, item))
        for item in items
    ]

    records = arc_client.run_batch(
        model,
        setting,
        jobs,
        experiment=experiment,
    )

    if len(records) != len(items):
        raise RuntimeError(
            f"{model}: expected {len(items)} responses, "
            f"received {len(records)}."
        )

    failed = [
        r for r in records
        if r.get("error") or r.get("response") is None
    ]

    if failed:
        raise RuntimeError(
            f"{model}: {len(failed)} failed API calls. "
            "Stopping rather than treating errors as incorrect answers."
        )

    return [
        int(bool(is_correct(task, record["response"], item["target"])))
        for record, item in zip(records, items)
    ]


def score(task):
    """Evaluate all candidates on Part A, saving after each batch."""
    items = bbh_data.get_split(task)["part_a"]
    rows = candidate_rows(task)
    progress = read_score_progress(task)

    save_score_progress(task, rows, progress)

    total = len(rows) * len(MODELS)
    done = 0

    for row in rows:
        idx = row["cand_idx"]
        progress.setdefault(idx, {})

        for model in MODELS:
            previous = progress[idx].get(model, "")

            if previous not in ("", None):
                done += 1
                continue

            print(
                f"[{done + 1}/{total}] "
                f"Part A: candidate {idx}, {model}",
                flush=True,
            )

            correctness = evaluate_batch(
                task,
                model,
                config.FINAL_SETTING[model],
                items,
                row["instruction"],
                experiment=f"ape-{task}-A",
            )

            accuracy = sum(correctness) / len(correctness)
            progress[idx][model] = f"{accuracy:.4f}"

            save_score_progress(task, rows, progress)

            done += 1
            print(f"  Accuracy: {accuracy:.1%} — saved", flush=True)

    print(f"Part A complete: {out(task, 'partA_scores.csv')}", flush=True)


def load_scores(task, require_complete=True):
    rows = list(read_score_progress(task).values())

    if not rows:
        raise RuntimeError("No Part A scores found. Run score first.")

    if require_complete:
        expected = len(candidate_rows(task))

        if len(rows) != expected:
            raise RuntimeError("Part A score table is incomplete.")

        for row in rows:
            for model in MODELS:
                if row.get(model, "") in ("", None):
                    raise RuntimeError(
                        f"Missing score for candidate "
                        f"{row['cand_idx']}, model {model}."
                    )

    return sorted(rows, key=lambda row: int(row["cand_idx"]))


def best_for(rows, model):
    """Highest accuracy, then shortest instruction, then lowest index."""
    return max(
        rows,
        key=lambda row: (
            float(row[model]),
            -len(row["instruction"]),
            -int(row["cand_idx"]),
        ),
    )


def gen2(task):
    """Generate 20 paraphrases of top Round 1 instructions."""
    path = out(task, "candidates.json")
    candidates = load_json(path)

    if len(candidates.get("round2", [])) == 20:
        print("Round 2 already exists; skipping.", flush=True)
        return

    rows = load_scores(task)
    pool_idx = []

    for model in MODELS:
        round1 = [row for row in rows if row["round"] == "1"]

        top = sorted(
            round1,
            key=lambda row: (
                -float(row[model]),
                len(row["instruction"]),
                int(row["cand_idx"]),
            ),
        )[:5]

        for row in top:
            if row["cand_idx"] not in pool_idx:
                pool_idx.append(row["cand_idx"])

    pool = [rows[int(idx)]["instruction"] for idx in pool_idx]

    if not pool:
        raise RuntimeError("No Round 1 instructions available.")

    rng = random.Random(config.SEED)
    allocations = [20 // len(pool)] * len(pool)

    for i in rng.sample(
        range(len(pool)),
        20 - sum(allocations),
    ):
        allocations[i] += 1

    progress_path = out(task, "round2_progress.json")

    if progress_path.exists():
        progress = load_json(progress_path)
    else:
        progress = {"completed": {}}

    for index, (source, count) in enumerate(zip(pool, allocations)):
        if count == 0:
            continue

        key = str(index)

        if key in progress["completed"]:
            continue

        prompt = (
            f"Paraphrase the following instruction in {count} "
            "different ways, preserving its meaning and approach.\n\n"
            f"Instruction:\n{source}\n\n"
            f"Reply with ONLY a valid JSON array of exactly "
            f"{count} strings."
        )

        print(
            f"Generating Round 2 paraphrases "
            f"for source {index + 1}/{len(pool)}...",
            flush=True,
        )

        generated = parse_list(
            writer_call(prompt, f"{task}-gen2-{index}"),
            expected=count,
        )

        progress["completed"][key] = generated
        save_json(progress_path, progress)

    new = []

    for index, count in enumerate(allocations):
        if count:
            new.extend(progress["completed"][str(index)])

    if len(new) != 20:
        raise RuntimeError(
            f"Expected 20 Round 2 instructions, got {len(new)}."
        )

    candidates["round2"] = new
    save_json(path, candidates)

    print(f"Saved 20 Round 2 instructions to {path}", flush=True)


def select(task):
    """Select own-best, random, and one-size-fits-all prompts."""
    rows = load_scores(task)
    generated = [row for row in rows if row["round"] != "0"]

    if not generated:
        raise RuntimeError("No generated instructions to select.")

    selections = {
        model: best_for(rows, model)["cand_idx"]
        for model in MODELS
    }

    rng = random.Random(config.SEED)
    selections["_random"] = rng.choice(generated)["cand_idx"]

    one_size = max(
        generated,
        key=lambda row: (
            sum(float(row[model]) for model in MODELS),
            -len(row["instruction"]),
            -int(row["cand_idx"]),
        ),
    )

    selections["_one_size"] = one_size["cand_idx"]

    save_json(out(task, "selections.json"), selections)

    print("Selected candidate indices:", flush=True)
    print(json.dumps(selections, indent=2), flush=True)

    return selections


def evalB(task):
    """Evaluate start, own-best, borrowed, random and one-size on Part B.

    Saves each completed model/condition batch and supports resuming.
    """
    items = bbh_data.get_split(task)["part_b"]
    rows = load_scores(task)
    selections = load_json(out(task, "selections.json"))

    instructions = {
        row["cand_idx"]: row["instruction"]
        for row in rows
    }

    conditions = {"start": ""}

    for model in MODELS:
        conditions[f"best_of_{model}"] = instructions[
            str(selections[model])
        ]

    conditions["random"] = instructions[
        str(selections["_random"])
    ]
    conditions["one_size"] = instructions[
        str(selections["_one_size"])
    ]

    path = out(task, "partB_items.csv")
    fieldnames = [
        "task", "model", "condition", "item_id", "correct"
    ]

    existing = []
    if path.exists():
        with open(path, newline="", encoding="utf-8") as f:
            existing = list(csv.DictReader(f))

    completed = set()
    expected_ids = {str(item["id"]) for item in items}

    for model in MODELS:
        for condition in conditions:
            matching = [
                row for row in existing
                if row["model"] == model
                and row["condition"] == condition
            ]

            matching_ids = {
                str(row["item_id"]) for row in matching
            }

            if len(matching) == len(items) and matching_ids == expected_ids:
                completed.add((model, condition))
            elif matching:
                raise RuntimeError(
                    f"Incomplete Part B batch for {model}/{condition}. "
                    f"Inspect {path} before resuming."
                )

    total = len(MODELS) * len(conditions)

    for model in MODELS:
        for condition, instruction in conditions.items():
            if (model, condition) in completed:
                continue

            print(
                f"[{len(completed) + 1}/{total}] "
                f"Part B: {model}, {condition}",
                flush=True,
            )

            correctness = evaluate_batch(
                task,
                model,
                config.FINAL_SETTING[model],
                items,
                instruction,
                experiment=f"ape-{task}-B",
            )

            with open(path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)

                if f.tell() == 0:
                    writer.writeheader()

                for item, correct in zip(items, correctness):
                    writer.writerow({
                        "task": task,
                        "model": model,
                        "condition": condition,
                        "item_id": item["id"],
                        "correct": correct,
                    })

                f.flush()
                os.fsync(f.fileno())

            completed.add((model, condition))

            print(
                f"  Accuracy: {sum(correctness) / len(items):.1%} "
                "— saved",
                flush=True,
            )

    print(f"Part B complete: {path}", flush=True)


def cost_report(task):
    """Report usage and time; do not invent dollar prices."""
    log_path = Path(config.LOG_PATH)

    calls = []

    if log_path.exists():
        with open(log_path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                record = json.loads(line)
                experiment = record.get("experiment", "")

                if (
                    experiment.startswith(f"ape-{task}-A")
                    or experiment.startswith(f"ape-{task}-B")
                    or experiment.startswith(f"ape-writer-{task}-")
                ):
                    calls.append(record)

    seen = set()
    unique = []

    for record in calls:
        key = record.get("key")

        if key is not None:
            if key in seen:
                continue
            seen.add(key)

        unique.append(record)

    successful = [
        record for record in unique
        if not record.get("error")
    ]

    writer_log = LOGS / "writer_calls.jsonl"
    writer_entries = []

    if writer_log.exists():
        with open(writer_log, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    if entry.get("tag", "").startswith(f"{task}-"):
                        writer_entries.append(entry)

    model_strings = sorted({
        f"{r.get('model_name')} -> {r.get('returned_model')}"
        for r in successful
    })

    report = {
        "task": task,
        "evaluation_models": MODELS,
        "fixed_settings": dict(config.FINAL_SETTING),
        "required_writer": "claude-sonnet-4-6",
        "substitute_writer_requested": WRITER_MODEL,
        "substitute_writer_setting": WRITER_SETTING,
        "writer_deviation": (
            "Claude Sonnet 4.6 unavailable; "
            "DeepSeek-V4.1-Flash used through ARC."
        ),
        "successful_api_calls": len(successful),
        "failed_api_calls": sum(
            bool(r.get("error")) for r in unique
        ),
        "prompt_tokens": sum(
            int(r.get("prompt_tokens") or 0)
            for r in successful
        ),
        "completion_tokens": sum(
            int(r.get("completion_tokens") or 0)
            for r in successful
        ),
        "sum_call_seconds": sum(
            float(r.get("seconds") or 0)
            for r in successful
        ),
        "writer_generation_calls": len(writer_entries),
        "writer_returned_models": sorted({
            str(r.get("returned_model"))
            for r in writer_entries
        }),
        "model_strings": model_strings,
        "estimated_usd": None,
        "cost_note": (
            "ARC pricing is unverified. Dollar cost is not "
            "estimated; zero-priced config placeholders "
            "must not be reported as actual costs."
        ),
        "wall_clock_note": (
            "Sum of call durations is not wall-clock time "
            "because ARC calls can run concurrently."
        ),
    }

    save_json(out(task, "run_cost.json"), report)
    print(json.dumps(report, indent=2), flush=True)

    return report


def main():
    if len(sys.argv) != 3:
        print(
            "Usage: python ape_pipeline.py TASK "
            "{gen1|score|gen2|select|evalB|cost|all}"
        )
        sys.exit(1)

    task, step = sys.argv[1], sys.argv[2]

    steps = {
        "gen1": gen1,
        "score": score,
        "gen2": gen2,
        "select": select,
        "evalB": evalB,
        "cost": cost_report,
    }

    if step == "all":
        gen1(task)
        score(task)
        gen2(task)
        score(task)
        select(task)
        evalB(task)
        cost_report(task)

    elif step in steps:
        steps[step](task)

    else:
        raise ValueError(f"Unknown step: {step}")


if __name__ == "__main__":
    main()

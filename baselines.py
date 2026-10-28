
"""Section 2: 50-item Part A baseline screening across accessible models."""

import csv
import time

import config
import bbh_data
import arc_client
from extraction import is_correct

SCREENING_MODELS = {"gpt-oss-120b", "deepseek-v4.1-flash", "glm-5.3"}

rows = []

for task in config.TASKS:
    split = bbh_data.get_split(task)
    items = split["part_a"][:50]

    for model_name, c in config.MODELS.items():
        if model_name not in SCREENING_MODELS:
            continue

        for setting in c["variants"]:
            jobs = [
                (it["id"], bbh_data.build_prompt(task, "", it))
                for it in items
            ]

            t0 = time.time()
            recs = arc_client.run_batch(
                model_name, setting, jobs, experiment="baseline"
            )

            errs = sum(1 for r in recs if r["error"])

            if errs:
                print(
                    f"WARNING: {errs} failed requests for "
                    f"{task} / {model_name} / {setting}"
                )
                continue

            if len(recs) != len(items):
                raise RuntimeError(
                    f"Expected {len(items)} responses, got {len(recs)}"
                )

            ok = sum(
                is_correct(task, r["response"], it["target"])
                for r, it in zip(recs, items)
            )

            acc = ok / len(items)

            row = dict(
                task=task,
                model=model_name,
                setting=setting,
                n=len(items),
                correct=ok,
                accuracy=round(acc, 3),
                in_50_80=0.5 <= acc <= 0.8,
                errors=errs,
                returned_model=recs[0]["returned_model"],
                minutes=round((time.time() - t0) / 60, 1),
                completion_tokens=sum(
                    r["completion_tokens"] or 0 for r in recs
                ),
            )

            rows.append(row)
            print(row, flush=True)
            with open("results/baselines.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)

if rows:
    with open("results/baselines.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print("Saved results/baselines.csv")
else:
    print("No successful baseline runs to save.")

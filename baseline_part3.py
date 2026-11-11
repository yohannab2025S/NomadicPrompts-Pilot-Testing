"""Part 3, Sec. 3: starting-prompt accuracy on 100 Part A questions."""
import csv, os, sys, time
import config, bbh_data, arc_client
from extraction import is_correct

model_name, task = sys.argv[1], sys.argv[2]
N, CHUNK, WORKERS = 100, 20, 4
OUT = "results/baselines_part3.csv"

part_a = bbh_data.get_split(task)["part_a"]
if len(part_a) < N:
    print(f"WARNING: Part A has only {len(part_a)} items; using all of them.")
items = part_a[:N]
c = config.MODELS[model_name]

for setting in c["variants"]:
    recs, t0 = [], time.time()
    for i in range(0, len(items), CHUNK):
        chunk = items[i:i + CHUNK]
        jobs = [(it["id"], bbh_data.build_prompt(task, "", it)) for it in chunk]
        recs += arc_client.run_batch(model_name, setting, jobs,
                                     experiment="baseline_part3", workers=WORKERS)
        print(f"{len(recs)}/{len(items)} done, {(time.time() - t0) / 60:.1f} min", flush=True)

    errs = sum(1 for r in recs if r["error"])
    truncated = sum(1 for r in recs if r.get("finish_reason") == "length")
    ok = sum((not r["error"]) and is_correct(task, r["response"], it["target"])
             for r, it in zip(recs, items))
    acc = ok / len(items)
    row = dict(task=task, model=model_name, setting=setting, n=len(items), correct=ok,
               accuracy=round(acc, 3), in_50_80=0.5 <= acc <= 0.8, errors=errs,
               truncated=truncated, returned_model=recs[0]["returned_model"],
               minutes=round((time.time() - t0) / 60, 1),
               sec_per_question=round((time.time() - t0) / len(items), 1),
               completion_tokens=sum(r["completion_tokens"] or 0 for r in recs))
    print(row, flush=True)

    new_file = not os.path.exists(OUT)
    with open(OUT, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new_file:
            w.writeheader()
        w.writerow(row)
    print(f"Added to {OUT}")

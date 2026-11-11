"""Part 3, Sec. 2: 5-question smoke test for a newly connected model."""
import csv, sys, time
import config, bbh_data, arc_client
from extraction import is_correct

model_name = sys.argv[1]                  # e.g. qwen3.5-4b
task = sys.argv[2] if len(sys.argv) > 2 else "word_sorting"
N = 5

items = bbh_data.get_split(task)["part_a"][:N]
c = config.MODELS[model_name]
rows = []
for setting in c["variants"]:
    jobs = [(it["id"], bbh_data.build_prompt(task, "", it)) for it in items]
    t0 = time.time()
    recs = arc_client.run_batch(model_name, setting, jobs, experiment="smoke_test", workers=1)
    secs = time.time() - t0
    print("Record fields logged:", sorted(recs[0].keys()))
    for r, it in zip(recs, items):
        ok = (not r["error"]) and is_correct(task, r["response"], it["target"])
        print("-" * 60)
        print("ID:", it["id"], "| correct:", ok, "| error:", r["error"])
        print("TARGET:  ", it["target"])
        print("RESPONSE:", (r["response"] or "")[-300:])
        rows.append(dict(task=task, model=model_name, setting=setting, item_id=it["id"],
                         correct=ok, error=r["error"], returned_model=r["returned_model"],
                         finish_reason=r.get("finish_reason"),
                         completion_tokens=r["completion_tokens"]))
    print("=" * 60)
    print(f"{model_name}/{setting}: {sum(x['correct'] for x in rows)}/{N} correct, "
          f"{secs/N:.1f} s per question")

with open(f"results/smoke_test_{model_name}.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(f"Saved results/smoke_test_{model_name}.csv")

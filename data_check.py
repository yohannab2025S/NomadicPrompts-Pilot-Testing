"""Part 3, Sec. 5: per-model call/failure/token totals and a 20-response random hand-check sheet."""
import csv, json, random, collections
import config, bbh_data
from extraction import extract_answer, is_correct

EXPS = {"smoke_test", "baseline_part3"}
recs = [json.loads(l) for l in open(config.LOG_PATH)]
recs = [r for r in recs if r.get("experiment") in EXPS]

by = collections.defaultdict(list)
for r in recs: by[r["model_name"]].append(r)
print(f"{'model':22}{'calls':>7}{'failed':>8}{'trunc':>7}{'prompt_tok':>12}{'compl_tok':>12}{'call_hours':>11}")
for m, rs in sorted(by.items()):
    fail = [r for r in rs if r["error"]]
    print(f"{m:22}{len(rs):>7}{len(fail):>8}{sum(r.get('finish_reason')=='length' for r in rs):>7}"
          f"{sum(r['prompt_tokens'] or 0 for r in rs):>12}{sum(r['completion_tokens'] or 0 for r in rs):>12}"
          f"{sum(r['seconds'] or 0 for r in rs)/3600:>11.1f}")
    for reason, n in collections.Counter(r["error"][:70] for r in fail).most_common():
        print(f"    failure x{n}: {reason}")

targets = {}
def target(item_id):
    task = max((t for t in config.TASKS if item_id.startswith(t)), key=len)
    if task not in targets:
        s = bbh_data.get_split(task); targets[task] = {it["id"]: it["target"] for it in s["part_a"] + s["part_b"]}
    return task, targets[task][item_id]

ok_recs = [r for r in recs if not r["error"]]
sample = random.Random(42).sample(ok_recs, 20)
with open("results/hand_check_20.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["model", "setting", "task", "item_id", "target", "extracted", "auto_correct",
                "response_end", "human_extraction_ok", "human_grade_ok", "notes"])
    for r in sample:
        task, tgt = target(r["item_id"])
        w.writerow([r["model_name"], r["setting"], task, r["item_id"], tgt, extract_answer(task, r["response"]),
                    is_correct(task, r["response"], tgt), (r["response"] or "")[-400:].replace("\n", " "), "", "", ""])
print("Wrote results/hand_check_20.csv")

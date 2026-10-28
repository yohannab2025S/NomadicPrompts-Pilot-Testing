"""Correct the Part 1 reasoning-token reading. hidden_reasoning ~= completion_tokens - tokens(visible response).
Cost uses COMPLETION tokens (which include hidden reasoning). Usage: python token_estimate.py logs/calls.jsonl"""
import sys, json, collections

def visible_tokens(text):
    try:
        import tiktoken
        return len(tiktoken.get_encoding("o200k_base").encode(text or ""))
    except Exception:
        return round(len(text or "") / 4)      # fallback ~4 chars/token (state this in the report)

agg = collections.defaultdict(lambda: dict(n=0, comp=0, vis=0, rep=0))
for l in open(sys.argv[1]):
    r = json.loads(l)
    if r.get("error") or r["completion_tokens"] is None: continue
    a = agg[(r["model_name"], r["setting"])]
    v = visible_tokens(r["response"])
    a["n"] += 1; a["comp"] += r["completion_tokens"]; a["vis"] += v; a["rep"] += r.get("reasoning_tokens_reported") or 0
print("model,setting,calls,mean_completion,mean_visible_est,mean_hidden_est,mean_reported_reasoning")
for (m, s), a in sorted(agg.items()):
    n = a["n"]
    print(f"{m},{s},{n},{a['comp']/n:.0f},{a['vis']/n:.0f},{max(a['comp']-a['vis'],0)/n:.0f},{a['rep']/n:.0f}")

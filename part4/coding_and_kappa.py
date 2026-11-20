"""Calculation 6: Cohen's kappa per category for the prompt-change coding + which kinds of changes go with better transfer.
Step 1 (automatic): writes part4/coding_template.csv, one row per distinct prompt used in Part B (own-best prompts, random, one-size).
Step 2 (HUMANS): write both coders' names in part4/coding/coders.json. Each coder fills ONLY their own file (coder_A.csv / coder_B.csv, 0/1 per category),
         independently. Then discuss disagreements and fill part4/coding/consensus.csv. Both coders' original files are kept (codebook.md).
Step 3 (automatic, if the sheet exists): kappa + raw agreement per category, and share of transfers that kept most of the gain."""
from common import *
from sklearn.metrics import cohen_kappa_score
CATS = ["semantic_clarification", "structural_change", "formatting", "reasoning_instruction", "examples", "output_constraint"]
USED_BY = {}
rows = []
for t in TASKS:
    sc = pd.read_csv(f"results/{t}/{RUN}/partA_scores.csv").set_index("cand_idx")
    sel = json.load(open(f"results/{t}/{RUN}/selections.json"))
    for key, k in sel.items():
        USED_BY.setdefault((t, int(k)), []).append(key)
    for (tt, k), users in USED_BY.items():
        pass
for (t, k), users in sorted(USED_BY.items()):
    sc = pd.read_csv(f"results/{t}/{RUN}/partA_scores.csv").set_index("cand_idx")
    rows.append(dict(prompt_id=f"{t}:{k}", task=t, candidate=k, used_as=";".join(users), change_text=sc.loc[k, "instruction"],
                     **{f"coder_A_{c}": "" for c in CATS}, **{f"coder_B_{c}": "" for c in CATS}, **{f"consensus_{c}": "" for c in CATS}))
tpl = pd.DataFrame(rows)
os.makedirs("part4/coding", exist_ok=True)
base = tpl[["prompt_id", "task", "candidate", "used_as", "change_text"]]
for who in ("A", "B"):                    # one blank file PER CODER so neither sees the other's labels
    out = base.copy()
    for c in CATS: out[f"coder_{who}_{c}"] = ""
    if not os.path.exists(f"part4/coding/coder_{who}.csv"): out.to_csv(f"part4/coding/coder_{who}.csv", index=False)
if not os.path.exists("part4/coding/coders.json"):
    json.dump({"A": "[name of coder A]", "B": "[name of coder B]"}, open("part4/coding/coders.json", "w"), indent=1)
cons = base.copy()
for c in CATS: cons[f"consensus_{c}"] = ""
if not os.path.exists("part4/coding/consensus.csv"): cons.to_csv("part4/coding/consensus.csv", index=False)
print(f"coding files ready in part4/coding/: {len(base)} distinct prompts (coder_A.csv, coder_B.csv, consensus.csv, coders.json)")

names = json.load(open("part4/coding/coders.json"))
A, Bf = pd.read_csv("part4/coding/coder_A.csv"), pd.read_csv("part4/coding/coder_B.csv")
if names["A"].startswith("[") or names["A"] == names["B"] or A[f"coder_A_{CATS[0]}"].isna().any() or Bf[f"coder_B_{CATS[0]}"].isna().any():
    print("Coding not finished (need two named, different coders and both label files filled) -> kappa and Chart 6 SKIPPED"); sys.exit(0)
cons_df = pd.read_csv("part4/coding/consensus.csv")
if cons_df[f"consensus_{CATS[0]}"].isna().any():
    print("Consensus labels missing (resolve disagreements by discussion, then fill consensus.csv) -> SKIPPED"); sys.exit(0)
cs = A.merge(Bf[["prompt_id"]+[f"coder_B_{c}" for c in CATS]], on="prompt_id").merge(cons_df[["prompt_id"]+[f"consensus_{c}" for c in CATS]], on="prompt_id")
cs.to_csv(f"{OUT_T}/coding_merged.csv", index=False)
kap = []
for c in CATS:
    a, b = cs[f"coder_A_{c}"].astype(int), cs[f"coder_B_{c}"].astype(int)
    kap.append(dict(category=c, n_prompts=len(cs), n_present_A=int(a.sum()), n_present_B=int(b.sum()), raw_agreement=float((a == b).mean()),
                    cohens_kappa=cohen_kappa_score(a, b) if len(set(a)) > 1 or len(set(b)) > 1 else np.nan))
pd.DataFrame(kap).assign(coder_A=names["A"], coder_B=names["B"]).round(3).to_csv(f"{OUT_T}/cohens_kappa.csv", index=False)
print(pd.DataFrame(kap).round(3).to_string())

# ---- which change types go with better transfer: share of transfers keeping >= 50% of the target's own-best gain ----
KEEP = 0.5          # "kept most of the gain" = borrowed gain >= 50% of target's own-best gain (point estimates, own-best gain > 0)
cons = cs.set_index("prompt_id")
out = []
for t in TASKS:
    sel = json.load(open(f"results/{t}/{RUN}/selections.json")); arr, n = load_partB(t)
    for s, tg in itertools.permutations(MODELS, 2):
        own = arr[(tg, f"best_of_{tg}")].mean() - arr[(tg, "start")].mean()
        if own <= 0: continue                      # no headroom to capture on this task/target
        bor = arr[(tg, f"best_of_{s}")].mean() - arr[(tg, "start")].mean()
        pid = f"{t}:{int(sel[s])}"
        row = dict(task=t, source=s, target=tg, kept_most=int(bor >= KEEP * own), capture=bor / own)
        for c in CATS: row[c] = int(cons.loc[pid, f"consensus_{c}"])
        out.append(row)
tr = pd.DataFrame(out); tr.to_csv(f"{OUT_T}/transfers_with_codes.csv", index=False)
summ = []
for c in CATS:
    for v, lab in ((1, "has change"), (0, "does not")):
        sub = tr[tr[c] == v]
        summ.append(dict(category=c, group=lab, n_transfers=len(sub), share_kept_most=sub.kept_most.mean() if len(sub) else np.nan))
pd.DataFrame(summ).round(3).to_csv(f"{OUT_T}/share_kept_by_change.csv", index=False)
print(pd.DataFrame(summ).round(3).to_string())

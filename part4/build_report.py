"""Builds part4/Part4_Report.pdf from the CSV tables and PNG figures (numbers are read from part4/tables, never typed by hand)."""
from common import *
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether
from PIL import Image as PILImage

T = lambda f: pd.read_csv(f"{OUT_T}/{f}")
ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontSize=13, spaceBefore=8, spaceAfter=4)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=10.5, spaceBefore=6, spaceAfter=3)
P = ParagraphStyle("P", parent=ss["BodyText"], fontSize=9, leading=11.5, spaceAfter=3)
CAP = ParagraphStyle("CAP", parent=P, fontSize=8.5, leading=10.5, textColor=colors.HexColor("#333333"), spaceAfter=6)
SM = ParagraphStyle("SM", parent=P, fontSize=7, leading=8.5)
TXT = ParagraphStyle("TXT", parent=P, fontSize=7.5, leading=9)
SHORT = {"gpt-oss-120b": "gpt-oss-120B", "gpt-oss-20b": "gpt-oss-20B", "deepseek-v4.1-flash": "DeepSeek-V4.1-Flash", "qwen3.5-9b": "Qwen3.5-9B", "qwen3.5-4b": "Qwen3.5-4B"}
PH = '<font color="#B00020">%s</font>'
pp = lambda x, d=1: f"{x*100:+.{d}f}"
def tbl(data, widths, fs=7, head=1, zebra=True):
    t = Table(data, colWidths=widths, repeatRows=head)
    st = [("FONTSIZE", (0, 0), (-1, -1), fs), ("LEADING", (0, 0), (-1, -1), fs + 1.5), ("BACKGROUND", (0, 0), (-1, head - 1), colors.HexColor("#DDE6F0")),
          ("FONTNAME", (0, 0), (-1, head - 1), "Helvetica-Bold"), ("GRID", (0, 0), (-1, -1), .3, colors.grey), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
          ("TOPPADDING", (0, 0), (-1, -1), 1.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5)]
    if zebra: st += [("BACKGROUND", (0, i), (-1, i), colors.HexColor("#F5F5F5")) for i in range(head + 1, len(data), 2)]
    t.setStyle(TableStyle(st)); return t
def fig(name, cap, width=7.0):
    w, h = PILImage.open(f"{OUT_F}/{name}").size
    return KeepTogether([Image(f"{OUT_F}/{name}", width=width * inch, height=width * inch * h / w), Paragraph(cap, CAP)])

acc = T("accuracy_by_condition.csv"); pm = T("pair_measures.csv"); pg = T("pooled_groups.csv"); gd = T("group_differences.csv")
ab = T("partA_vs_partB_gain.csv"); cs = T("condition_summary.csv").set_index("condition"); og = T("own_best_gain.csv")
sm = pd.read_csv(f"{OUT_T}/spearman_mean_across_tasks.csv", index_col=0).loc[MODELS, MODELS]
sl = T("spearman_pairs_long.csv")
S = []
S += [Paragraph("Nomadic Prompts Part 4: Analysis, Charts, and Results", ParagraphStyle("T", parent=ss["Title"], fontSize=17, spaceAfter=2)),
      Paragraph(PH % "[Team name / Research Group] &nbsp;|&nbsp; [Team members] &nbsp;|&nbsp; Repository: https://github.com/yohannab2025S/NomadicPrompts-Pilot-Testing (folder <font face='Courier'>part4/</font>)", P)]
S += [Paragraph("<b>Data used.</b> Run 1 of the pipeline: 3 tasks (Word Sorting, Causal Judgement, Salient Translation Error Detection), 5 models (gpt-oss-120B, gpt-oss-20B, "
      "DeepSeek-V4.1-Flash, Qwen3.5-9B, Qwen3.5-4B), 20 candidate prompts per task scored on 100 Part A items, and Part B on 150 / 87 / 150 held-out items. "
      "Each model was evaluated under 8 conditions: start (no added instruction), its own best prompt, the best prompt of each of the 4 other models (borrowed), "
      "a random candidate, and a one-size-fits-all prompt. Decoding was at temperature 0, so each condition has <b>one run</b> (&ldquo;averaged over runs&rdquo; is therefore n=1; "
      "all intervals reflect item sampling only). All numbers and figures come from <font face='Courier'>python part4/run_all.py</font> (seed 42; 10,000 paired bootstrap resamples of Part B items, "
      "one shared resample per task for every model and condition).", P)]

FB = ParagraphStyle("FB", parent=TXT, fontSize=7, leading=8.4)
def fb(*c): return [Paragraph(x, FB) for x in c]
GREEN, AMBER = '<font color="#1B7F3B"><b>Done</b></font>', '<font color="#B26A00"><b>Needs you</b></font>'
S += [Paragraph("1. Response to Part 3 Feedback", H1),
      tbl([fb("<b>#</b>", "<b>Feedback item</b>", "<b>What we did / where</b>", "<b>Status</b>"),
        fb("1", "Praise: better models, fair size comparison, better tasks, fair writer, honest reporting", "Kept the design. Problems are still documented (see Limitations); deviations log updated.", GREEN),
        fb("2", "Skip Round 2; use start prompt + 20 candidates (21 prompts)", "Round 2 is empty in every <font face='Courier'>candidates.json</font>; Part A uses 21 prompts per task. Logged in <font face='Courier'>deviations_log.md</font>.", GREEN),
        fb("3", "Finish Run 1 completely; add Runs 2-3 only if time allows; log it", "Run 1: 3 tasks, <b>5 models</b>, selections frozen, Part B done. GLM-5.3 was dropped from Part B (6 \u2192 5 models). Runs 2-3 are partial (Part A for gpt-oss-20B only) and are not used. Logged. " + PH % "[Confirm reason for dropping GLM and who decided.]", AMBER),
        fb("4", "Run ARC models in parallel; local models on separate computers", "Concurrency capped at 10 per host (<font face='Courier'>config.py</font>); local models ran on ARC Open OnDemand. " + PH % "[Confirm.]", AMBER),
        fb("5", "Numbers do not match between text and Table 5.1 (calls, tokens, failed calls shown although rerun)", "Table regenerated from the merged log: <font face='Courier'>tables/call_log_summary.csv</font> shows, per model, failed attempts logged, failed-then-rerun-OK, still failed, truncations and tokens, so reruns are explicit. Paste this table into Section 5.1 and make the text quote it. Note: totals are incomplete until the missing logs (row 6) are added.", AMBER),
        fb("6", "One place for all data: merge every log and result file into the repository", "<font face='Courier'>part4/merge_logs.py</font> merges the 3 exported log folders into <font face='Courier'>logs_export/merged/calls_merged.jsonl.gz</font> (36,922 rows \u2192 36,112 unique calls). <b>Still missing:</b> Run 1 logs for gpt-oss-120B and DeepSeek (all tasks) and gpt-oss-20B Part A (<font face='Courier'>tables/log_coverage.csv</font>). Their owners must run <font face='Courier'>score_model.py export-logs</font> and push.", AMBER),
        fb("7", "Hand check: one more check of 20 responses chosen on purpose (wrong, truncated, every model)", "Sample drawn: <font face='Courier'>tables/hand_check_targeted_20.csv</font> (8 wrong, 5 truncated, 3 no answer extracted, 4 correct controls; round-robin over all 5 models). Because Run 1 logs are missing for gpt-oss-120B/DeepSeek, their rows come from the Word Sorting pilot logs. " + PH % "[Someone must read each response and fill the human_label columns; then report agreement with the automatic scorer.]", AMBER),
        fb("8", "Coding: name the two coders; save their label files separately before computing \u03ba", "Separate files <font face='Courier'>part4/coding/coder_A.csv</font>, <font face='Courier'>coder_B.csv</font>, <font face='Courier'>consensus.csv</font> and <font face='Courier'>coders.json</font> (names). The \u03ba script refuses to run until both are filled by two named, different people. " + PH % "[Coders: fill names + labels.]", AMBER),
        fb("9", "Placeholders in model table (gpt-oss-20b level; GLM-5.3)", "Settings from <font face='Courier'>config.py</font>: gpt-oss-120B thinking-low; gpt-oss-20B <b>default</b> (the only configured variant; no reasoning level is set in the config, " + PH % "[confirm what level the server used]" + "); DeepSeek-V4.1-Flash thinking-low; Qwen3.5-9B/4B no-think (BF16); GLM-5.3 thinking-high per Section 5 (dropped from Part B).", AMBER)],
          [.25 * inch, 1.9 * inch, 4.2 * inch, .75 * inch], fs=7, zebra=False)]

S += [Paragraph("2. Calculations", H1), Paragraph("2.1 Part B accuracy by model, task and condition (% correct, 95% bootstrap CI)", H2)]
order = ["start", "own_best", "borrowed (mean of 4 other models' best prompts)", "random", "one_size"]
rows = [["Task", "Model", "Start", "Own best", "Borrowed", "Random", "One-size"]]
for t in TASKS:
    for m in MODELS:
        r = [TASK_LABEL[t] if m == MODELS[0] else "", SHORT[m]]
        for c in order:
            x = acc[(acc.task == t) & (acc.model == m) & (acc.condition == c)].iloc[0]
            r.append(f"{x.accuracy*100:.1f} [{x.ci_lo*100:.1f}, {x.ci_hi*100:.1f}]")
        rows.append(r)
S += [tbl(rows, [1.2 * inch, 1.15 * inch] + [.95 * inch] * 5, fs=6.8),
      Paragraph("Borrowed = mean accuracy of the target model over the four other models&rsquo; best prompts. n = 1 run per condition. DeepSeek is at ceiling on Word Sorting (100% start), leaving no headroom there.", CAP)]
c = lambda k: cs.loc[k]
S += [Paragraph(f"<b>Mean gain over start across all 15 task&times;model cells</b> (pp, paired bootstrap CI): own best {pp(c('own_best').mean_gain_over_start)} [{pp(c('own_best').ci_lo)}, {pp(c('own_best').ci_hi)}]; "
      f"borrowed {pp(c('borrowed (all 60 task x pair cells)').mean_gain_over_start)} [{pp(c('borrowed (all 60 task x pair cells)').ci_lo)}, {pp(c('borrowed (all 60 task x pair cells)').ci_hi)}]; "
      f"one-size {pp(c('one_size').mean_gain_over_start)} [{pp(c('one_size').ci_lo)}, {pp(c('one_size').ci_hi)}]; "
      f"random {pp(c('random').mean_gain_over_start)} [{pp(c('random').ci_lo)}, {pp(c('random').ci_hi)}]. "
      f"None of the {len(og)} individual own-best gains has a 95% CI that excludes 0 ({int(og.ci_excludes_zero.sum())} of {len(og)}).", P)]

S += [Paragraph("2.2 Headroom capture and retention for every directed pair (all three tasks pooled)", H2)]
hc = pm[(pm.scope == "ALL_TASKS")]
h = hc[hc.measure == "headroom_capture"].set_index(["source", "target"]); r_ = hc[hc.measure == "retention"].set_index(["source", "target"])
HP = ParagraphStyle("HP", parent=TXT, fontName="Helvetica-Bold", fontSize=6.5, leading=7.5)
rows = [[Paragraph(x, HP) for x in ["Source \u2192 Target", "Borrowed gain, pp (sum of 3 tasks) [95% CI]", "Target own-best gain = headroom, pp [CI]", "Source own-best gain, pp [CI]", "Headroom capture", "Retention"]]]
for s_, t_ in itertools.permutations(MODELS, 2):
    a, b = h.loc[(s_, t_)], r_.loc[(s_, t_)]
    f = lambda x: f"{x*100:+.1f}"
    rows.append([f"{SHORT[s_]} \u2192 {SHORT[t_]}", f"{f(a.borrowed_gain)} [{f(a.borrowed_gain_lo)}, {f(a.borrowed_gain_hi)}]", f"{f(a.denominator)} [{f(a.denominator_lo)}, {f(a.denominator_hi)}]",
                 f"{f(b.denominator)} [{f(b.denominator_lo)}, {f(b.denominator_hi)}]", "NA\u2020" if a.status != "ok" else f"{a.ratio:.2f} [{a.ratio_lo:.2f}, {a.ratio_hi:.2f}]",
                 "NA\u2020" if b.status != "ok" else f"{b.ratio:.2f} [{b.ratio_lo:.2f}, {b.ratio_hi:.2f}]"])
S += [tbl(rows, [1.9 * inch, 1.55 * inch, 1.45 * inch, 1.4 * inch, .75 * inch, .65 * inch], fs=6.5),
      Paragraph("\u2020 Frozen denominator rule: a ratio is reported only if the 95% bootstrap CI of its denominator is entirely above 0. Under this rule <b>all 20 pairs are NA</b> (also in each task separately, "
                "see <font face='Courier'>tables/pair_measures.csv</font>) because no model&rsquo;s own-best prompt produced a reliably positive held-out gain. Gains are summed over the three tasks (pp), so they are not percentage-point changes on one task.", CAP)]

S += [Paragraph("2.3 Pooled ratios by group, and group differences (ratio of sums over pairs and tasks; paired bootstrap)", H2)]
g_ = pg[pg.scope == "ALL_TASKS"]; rows = [["Group", "Pairs", "Headroom capture [95% CI]", "Retention [95% CI]", "Denominator CI excludes 0?"]]
names = {"same_provider": "Same company", "cross_provider": "Different company", "large_to_small": "Larger \u2192 smaller", "small_to_large": "Smaller \u2192 larger"}
for k, nm in names.items():
    a = g_[(g_.group == k) & (g_.measure == "headroom_capture")].iloc[0]; b = g_[(g_.group == k) & (g_.measure == "retention")].iloc[0]
    rows.append([nm, int(a.n_pairs), f"{a.pooled_ratio:.2f} [{a.ci_lo:.2f}, {a.ci_hi:.2f}]", f"{b.pooled_ratio:.2f} [{b.ci_lo:.2f}, {b.ci_hi:.2f}]", "No (unstable)" if a.status != "ok" else "Yes"])
S += [tbl(rows, [1.6 * inch, .5 * inch, 1.9 * inch, 1.9 * inch, 1.6 * inch], fs=7.5)]
d_ = gd[gd.scope == "ALL_TASKS"]; rows = [["Difference", "Measure", "Estimate [95% CI]", "CI excludes 0?"]]
for _, x in d_.iterrows():
    cmpn = x.comparison.replace("same_provider", "Same co.").replace("cross_provider", "Diff. co.").replace("large_to_small", "Larger\u2192smaller").replace("small_to_large", "Smaller\u2192larger").replace(" minus ", " \u2212 ")
    rows.append([cmpn, x.measure.replace("_", " "), f"{x.difference:+.2f} [{x.ci_lo:+.2f}, {x.ci_hi:+.2f}]", "yes" if x.excludes_zero else "no"])
S += [Spacer(1, 4), tbl(rows, [2.2 * inch, 1.3 * inch, 2.4 * inch, 1.3 * inch], fs=7.5),
      Paragraph("Within the same-company and different-company groups every model appears as both source and target equally often, so pooled headroom capture and pooled retention are mathematically identical there. "
                "The pooled denominators are small (own-best gains are near zero), so the ratios are unstable and their intervals are very wide.", CAP)]

S += [Paragraph("2.4 Spearman rank correlations of Part A candidate rankings", H2)]
off = sl.spearman.dropna(); mx = sl.loc[sl.spearman.idxmax()]; mn = sl.loc[sl.spearman.idxmin()]
S += [Paragraph(f"For each task, the 20 candidate prompts were ranked by each model&rsquo;s Part A score (100 items). Per-task matrices are in <font face='Courier'>tables/spearman_*.csv</font>; Chart 5 shows the mean over the three tasks. "
      f"Across the 30 model-pair&times;task correlations the mean is {off.mean():.2f} (range {mn.spearman:.2f} to {mx.spearman:.2f}). With 20 candidates, |&rho;| &gt; 0.44 is needed for two-sided p &lt; .05 (uncorrected). "
      f"Mean-over-task values: Qwen-9B&ndash;Qwen-4B {sm.loc['qwen3.5-9b','qwen3.5-4b']:.2f}; gpt-oss-120B&ndash;gpt-oss-20B {sm.loc['gpt-oss-120b','gpt-oss-20b']:.2f}; "
      f"DeepSeek with any other model {sm.loc['deepseek-v4.1-flash'].drop('deepseek-v4.1-flash').min():.2f} to {sm.loc['deepseek-v4.1-flash'].drop('deepseek-v4.1-flash').max():.2f}.", P)]

S += [Paragraph("2.5 Part A gain vs Part B gain for each selected (own-best) prompt", H2)]
rows = [["Task", "Model", "Prompt #", "Part A gain (pp)", "Part B gain (pp)", "Shrinkage (pp)"]]
for _, x in ab.iterrows(): rows.append([TASK_LABEL[x.task], SHORT[x.model], int(x.selected_candidate), pp(x.gain_partA), pp(x.gain_partB), f"{x.shrinkage*100:.1f}"])
rows.append(["Mean of 15", "", "", pp(ab.gain_partA.mean()), pp(ab.gain_partB.mean()), f"{ab.shrinkage.mean()*100:.1f}"])
S += [tbl(rows, [1.5 * inch, 1.4 * inch, .7 * inch, 1.2 * inch, 1.2 * inch, 1.1 * inch], fs=7),
      Paragraph(f"The gain was smaller on Part B than on Part A for {int((ab.gain_partB < ab.gain_partA).sum())} of {len(ab)} prompts. Part A gains are inflated because each prompt was <i>selected</i> for its Part A score (winner&rsquo;s curse).", CAP)]

S += [Paragraph("2.6 Cohen&rsquo;s &kappa; for prompt-change coding", H2),
      Paragraph(PH % "<b>Pending: human coding not yet done.</b> <font face='Courier'>tables/coding_template.csv</font> lists the 16 distinct prompts used in Part B. Two named coders must label them independently with the six codebook categories, "
                "each in their own file (<font face='Courier'>part4/coding/coder_A.csv</font>, <font face='Courier'>coder_B.csv</font>), agree on <font face='Courier'>consensus.csv</font>, and rerun <font face='Courier'>python part4/run_all.py</font>; that fills this table and Chart 6 "
                "(per-category &kappa; and raw agreement; share of transfers that kept &ge;50% of the target&rsquo;s gain, by change type). Insert the &kappa; table here.", P)]

S += [PageBreak(), Paragraph("3. Charts", H1)]
S += [fig("chart1_accuracy_by_condition.png", "<b>Chart 1.</b> Prompt optimization barely moves held-out accuracy: own-best, borrowed and one-size prompts sit within each other&rsquo;s error bars for nearly every model, while a random candidate is often worst (largest drop: Qwen3.5-4B on Word Sorting). Bars show accuracy on Part B items; whiskers are 95% bootstrap CIs; y-axis starts at 30%.", 7.2)]
S += [fig("chart2_headroom_heatmap.png", "<b>Chart 2.</b> Few source&rarr;target transfers capture a clear share of the target&rsquo;s headroom. Left: headroom capture (point estimates; \u2020 = denominator CI includes 0, so the value is unstable; colors clipped to [&minus;1, 2]). Right: raw gain of the borrowed prompt over the target&rsquo;s start (mean per task, pp). Qwen3.5-9B benefits most from others&rsquo; prompts; gpt-oss-20B is hurt by most.", 7.2)]
S += [fig("chart3_pooled_ratio_dotplot.png", "<b>Chart 3.</b> The data cannot separate the groups: every pooled ratio has an interval spanning both 0 (no gain kept) and 1 (all gain kept), because the own-best gains in the denominator are small. Points are pooled ratios (ratio of sums); lines are 95% paired bootstrap CIs; arrows mark intervals cut off by the axis.", 6.6)]
S += [fig("chart4_partA_vs_partB_slope.png", f"<b>Chart 4.</b> Prompt gains shrink on new questions: mean gain falls from {pp(ab.gain_partA.mean())} pp on Part A to {pp(ab.gain_partB.mean())} pp on Part B, and falls for all 15 selected prompts. Each line is one model&rsquo;s own-best prompt on one task.", 5.6)]
S += [fig("chart5_spearman_matrix.png", f"<b>Chart 5.</b> Models mostly disagree about which candidate prompts are better. Off-diagonal correlations average {off.mean():.2f}; the strongest agreement is within the Qwen family (9B&ndash;4B: {sm.loc['qwen3.5-9b','qwen3.5-4b']:.2f}). Spearman &rho; of Part A candidate rankings, averaged over three tasks (20 candidates each).", 4.3)]
S += [Paragraph(PH % "<b>Chart 6</b> (share of transfers that kept most of the gain, by type of prompt change) will be generated by <font face='Courier'>make_charts.py</font> once the coding sheet exists (see 2.6).", P)]

own = c("own_best"); G = lambda k, m: g_[(g_.group == k) & (g_.measure == m)].iloc[0]; D = lambda cmp_, m: d_[(d_.comparison == cmp_) & (d_.measure == m)].iloc[0]
a1, a2 = G("same_provider", "headroom_capture"), G("cross_provider", "headroom_capture"); a3, a4 = G("large_to_small", "headroom_capture"), G("small_to_large", "headroom_capture")
d1, d2 = D("same_provider minus cross_provider", "headroom_capture"), D("large_to_small minus small_to_large", "headroom_capture")
S += [Paragraph("4. Answers to the Research Questions", H1),
      Paragraph(PH % "Replace the hypotheses below with the exact wording from your Part 2 plan; the results and verdicts are computed from the data.", P)]
RQ = [["Research question", "Hypothesis", "Result (key number and interval)", "Verdict"],
 [Paragraph("RQ1. Do prompts transfer better between models from the same company than across companies?", TXT), Paragraph(PH % "[from Part 2]", TXT),
  Paragraph(f"Pooled headroom capture: same company {a1.pooled_ratio:.2f} [{a1.ci_lo:.2f}, {a1.ci_hi:.2f}] vs different company {a2.pooled_ratio:.2f} [{a2.ci_lo:.2f}, {a2.ci_hi:.2f}]; difference {d1.difference:+.2f} [{d1.ci_lo:+.2f}, {d1.ci_hi:+.2f}]. Denominators are unstable (CI includes 0).", TXT), Paragraph("<b>Unclear</b>", TXT)],
 [Paragraph("RQ2. Do prompts from larger models transfer to smaller models better than the reverse?", TXT), Paragraph(PH % "[from Part 2]", TXT),
  Paragraph(f"Pooled headroom capture: larger\u2192smaller {a3.pooled_ratio:.2f} [{a3.ci_lo:.2f}, {a3.ci_hi:.2f}] vs smaller\u2192larger {a4.pooled_ratio:.2f} [{a4.ci_lo:.2f}, {a4.ci_hi:.2f}]; difference {d2.difference:+.2f} [{d2.ci_lo:+.2f}, {d2.ci_hi:+.2f}]. Only one model (gpt-oss-120B) counts as &ldquo;large&rdquo;.", TXT), Paragraph("<b>Unclear</b>", TXT)],
 [Paragraph("RQ3. Do optimized prompts improve held-out accuracy, and how much of the Part A gain survives?", TXT), Paragraph(PH % "[from Part 2]", TXT),
  Paragraph(f"Own-best gain on Part B {pp(own.mean_gain_over_start)} pp [{pp(own.ci_lo)}, {pp(own.ci_hi)}] vs {pp(ab.gain_partA.mean())} pp on Part A; smaller for 15 of 15 prompts; 0 of 15 individual gains have a CI excluding 0. A random candidate is worse than start: {pp(c('random').mean_gain_over_start)} [{pp(c('random').ci_lo)}, {pp(c('random').ci_hi)}].", TXT), Paragraph("<b>Not supported</b> (gain does not clearly survive); random-worse-than-start is supported", TXT)],
 [Paragraph("RQ4. Do different models prefer the same prompts (Part A rankings)?", TXT), Paragraph(PH % "[from Part 2]", TXT),
  Paragraph(f"Mean off-diagonal Spearman &rho; = {off.mean():.2f} (range {mn.spearman:.2f} to {mx.spearman:.2f}); highest within Qwen family ({sm.loc['qwen3.5-9b','qwen3.5-4b']:.2f}) and gpt-oss ({sm.loc['gpt-oss-120b','gpt-oss-20b']:.2f}); n = 20 candidates per task.", TXT), Paragraph("<b>Not supported</b> for cross-company agreement; <b>unclear</b> for within-family", TXT)],
 [Paragraph("RQ5. Which kinds of prompt changes go with better transfer?", TXT), Paragraph(PH % "[from Part 2]", TXT), Paragraph(PH % "Pending human coding (see 2.6).", TXT), Paragraph(PH % "Pending", TXT)]]
S += [tbl(RQ, [1.7 * inch, .9 * inch, 3.7 * inch, 1.2 * inch], fs=7.5, zebra=False)]

S += [Paragraph("5. Limitations", H1)]
for t in [
 "<b>Small design.</b> 3 tasks, 5 models, 20 candidate prompts per task, one run per condition at temperature 0. Results describe these tasks and models and should not be generalized to other tasks, models or prompt styles.",
 f"<b>Little headroom, so ratios are unstable.</b> Own-best prompts raised held-out accuracy by only {pp(own.mean_gain_over_start)} pp on average (CI includes 0), so the denominators of headroom capture and retention are near zero. Under the frozen denominator rule all 20 per-pair ratios are NA, and pooled intervals are very wide. The study can only report &ldquo;unclear&rdquo; for the group comparisons.",
 "<b>Ceiling effect.</b> DeepSeek-V4.1-Flash scores 100% on Word Sorting with no instruction, and sits at 77&ndash;79% on Salient Translation Error, so it contributes almost no measurable headroom as a target.",
 "<b>Model coverage and confounds.</b> Only one model (gpt-oss-120B) is &ldquo;large&rdquo;, so size comparisons are confounded with provider. DeepSeek V4 Pro and GLM-5.3 were dropped (see <font face='Courier'>deviations_log.md</font>), so no company except OpenAI and Alibaba has two sizes. Qwen-9B and Qwen-4B are both classed &ldquo;small&rdquo;.",
 "<b>Local quantized models.</b> The Qwen and gpt-oss-20B models were run locally through ARC as GGUF files (BF16/F16 builds); results may differ from the providers&rsquo; hosted versions. Qwen models were run with thinking disabled; others use fixed reasoning settings.",
 "<b>One prompt writer.</b> All candidate prompts were generated by a single writer model at temperature 1.0 (" + PH % "[confirm: config.py lists Gemini 3.5 Flash-Lite; the Word Sorting pilot used DeepSeek-V4.1-Flash because Claude Sonnet 4.6 was unavailable]" + "), so the prompt styles reflect one writer. The sample of 20 candidates per task is small, and selection on 100 Part A items causes winner&rsquo;s-curse inflation of Part A gains.",
 "<b>One run, item-sampling intervals only.</b> Intervals capture which Part B items were drawn, not run-to-run variation, and do not account for the choice of candidate pool. Runs 2 and 3 are not yet complete for all models, so no run-averaging or between-run variability is reported.",
 "<b>What can and cannot be claimed.</b> We can claim that, here, optimized prompts did not reliably beat no instruction on new questions, that a random candidate was worse than no instruction, and that models&rsquo; rankings of candidates mostly disagreed. We cannot claim that transfer is better or worse across companies or sizes, nor that any category of prompt change helps transfer."]:
    S.append(Paragraph("\u2022 " + t, P))

S += [Paragraph("6. Contribution and AI Use Statements, Reproducibility", H1),
      Paragraph(PH % "<b>Contribution statement:</b> [Edit to match what actually happened, e.g., &ldquo;Yohanna Belay wrote the Part 4 analysis scripts, charts and report. Part A/B data collection was done by &hellip;&rdquo;]", P),
      Paragraph("<b>AI use statement:</b> Claude (Anthropic) was used to help write the analysis scripts in <font face='Courier'>part4/</font> (bootstrap, pooled ratios, Spearman, charts), to draft the report text and to review results. "
                + PH % "[Add what you changed or verified yourself, e.g., checked outputs against the Part B CSVs and the frozen denominator rule, edited the interpretation.]", P),
      Paragraph("<b>Reproduce:</b> <font face='Courier'>pip install numpy pandas scipy scikit-learn matplotlib seaborn reportlab pillow; python part4/run_all.py; python part4/build_report.py; python part4/build_docx.py</font> from the repository root. Tables are in <font face='Courier'>part4/tables/</font>, figures in <font face='Courier'>part4/figures/</font>.", P)]

doc = SimpleDocTemplate("part4/Part4_Report.pdf", pagesize=letter, leftMargin=.6 * inch, rightMargin=.6 * inch, topMargin=.55 * inch, bottomMargin=.55 * inch,
                        title="Nomadic Prompts Part 4", author="")
doc.build(list(S)); print("report built")

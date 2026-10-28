# NomadicPrompts-Pilot-Testing

Nomadic Prompts: cross-model transfer of optimized prompts

Run order (see comments at top of each script):
1. `export ARC_BASE_URL=... ARC_API_KEY=...` (and the other hosts' variables listed in `config.py`; `ANTHROPIC_API_KEY` for the writer). Edit the TODOs in `config.py` (exact model IDs, prices).
2. `python baselines.py`  -> `results/baselines.csv` (Part A start accuracy, task x model x setting). Pick final tasks and `FINAL_SETTING` (50-80% target).
3. `python ape_pipeline.py <task> all`  -> candidates, Part A score table, selections, Part B items, cost report.
4. `python analysis.py <task>`  -> Part B table, headroom/retention with paired bootstrap CIs, pooled group measures, group differences, rank-correlation matrices.
5. `python token_estimate.py logs/calls.jsonl`  -> hidden-reasoning estimate (completion minus visible tokens).

Every call is appended to `logs/calls.jsonl` with the requested model, the model string the server returned, settings, tokens and raw response.
Calls are cached by (model, setting, prompt), so re-running never re-pays for finished work. Concurrency is capped at 10 per host.
`codebook.md` = modification codebook; `deviations_log.md` = protocol changes after submission.

Status: these scripts were written without access to ARC and have NOT been run against the live APIs; `extraction.py` and `analysis.py` were tested on
synthetic data only. Check the first few raw responses and the OpenAI-compatible endpoint assumption in `arc_client.py` before launching full runs.


## Current status

The `word_sorting` pilot has completed Part A prompt scoring, prompt selection, and Part B evaluation through ARC using `gpt-oss-120b`, `deepseek-v4.1-flash`, and `glm-5.3`. Results are saved under `results/word_sorting/`.

DeepSeek-V4.1-Flash was used to generate candidate prompts because Claude Sonnet 4.6 was unavailable. This substitution is recorded in the run cost report.

Part B evaluation is complete; the statistical analysis step is still pending. DeepSeek achieved 100% baseline accuracy on the held-out items, leaving no room to measure improvement for that model on this task.

ARC pricing remains unverified, so no dollar cost is reported. Raw API logs and local credentials are excluded from Git.

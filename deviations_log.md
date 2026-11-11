# Deviations log (protocol is frozen at submission of Part 2)
Record every change made after submission: date, what changed, why, who decided.

| Date | Section changed | Change | Reason | Decided by |
|---|---|---|---|---|
| 2026-11-03 | Models | Removed DeepSeek-V4 Pro from this semester's plan. | Not accessible through ARC and no funding for an external API; Part 3 instructions direct its removal. | Instructor |
| 2026-11-03 | Models | Added gpt-oss-20b and one small model family in two sizes (Qwen 3.5 4B and 9B). | Pilot models were near ceiling (17 of 18 baselines at 84-100%) and no two models came from the same provider. | Instructor (Part 3 instructions) |
| 2026-11-03 | Models | Qwen 3.5 9B could not run on the planned local machine. | On a MacBook Pro (M2, 8 GB, macOS 15.7.4), Ollama reported the model needs 7.1 GiB with only 4.8 GiB available. | Nick Nguyen |
| 2026-11-09 | Models / hosting | Local models moved from team laptops to the ARC Open OnDemand "Large Language Model" app (1 GPU, 8 cores per session), reached from the pipeline through an SSH tunnel. | gpt-oss-20b froze on a 16 GB Mac and Qwen 9B would not load on an 8 GB Mac; ARC GPUs also run faster. | Instructor advice; team |
| 2026-11-09 | Models | Qwen models run at BF16 (unsloth/Qwen3.5-4B-GGUF/BF16 and unsloth/Qwen3.5-9B-GGUF/BF16). | Same precision for both sizes, so size is the only difference between them. | Nick Nguyen |
| 2026-11-09 | Task selection | Baseline screening uses 100 Part A questions per task and model (pilot used 50). | Part 3 instructions require 100. | Instructor (Part 3 instructions) |
# Codebook for prompt modifications (frozen with the protocol)
Unit of coding: each individual change between the starting prompt and the optimized prompt (not the whole prompt).
One change may receive more than one label. Two coders label independently; report Cohen's kappa and raw agreement per
category (category present / absent), then resolve disagreements by discussion and keep both versions of the labels.

| Category | Definition | Example |
|---|---|---|
| Semantic clarification | Restates or sharpens what the task is, defines terms, or states what counts as correct, without changing structure or format. | "An argument is valid only if the conclusion must be true whenever all premises are true; ignore whether premises are true in the real world." |
| Structural change | Reorders content or splits the instruction into stages, sections or numbered steps. | "First list the constraints. Then place each object. Finally pick the option." |
| Formatting | Changes surface presentation only: markdown, headings, delimiters, XML-style tags, capitalization, labels. | "### TASK" / "<question>...</question>" |
| Reasoning instruction | Tells the model how to think or check its work. | "Reason step by step and test for a counterexample before answering." |
| Examples | Adds worked or few-shot demonstrations of inputs and outputs. | "Example: Premises ... Answer: invalid" |
| Output constraint | Restricts what the response may contain or look like (length, allowed answers, no explanation). | "Reply with only the option letter." |

Coding sheet columns: prompt_id, change_text, then one 0/1 column per category, per coder (coder_A_*, coder_B_*).

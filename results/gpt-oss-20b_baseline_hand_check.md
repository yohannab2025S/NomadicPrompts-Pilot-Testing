# GPT-OSS-20B baseline hand check

Sample: 20 of the 300 original Part 3 baseline records, drawn without replacement using Python seed 42. Task order: Word Sorting, Formal Fallacies, Logical Deduction. Within each task, CSV row order was preserved.

I inspected the final answer lines against their saved targets and grades. All 20 matched their targets and were marked correct. The sample contained no truncated or incorrect responses.

Across all 300 records: 54,698 prompt tokens, 230,068 completion tokens, 284,766 combined. Finish reasons: {'stop': 295, 'length': 5}.

| Item ID | Final answer | Target | Saved grade | Review |
|---|---|---|---|---|
| word_sorting_0073 | convey decimate experiment fortieth incautious kudo marshall neoclassic rest whimper wiley xylem z's | convey decimate experiment fortieth incautious kudo marshall neoclassic rest whimper wiley xylem z's | Correct | Agrees |
| word_sorting_0015 | bare census exaltation gnomon humility infirm intrinsic manatee moth oblique paregoric patristic snagging sorrowful stressful timeout torch | bare census exaltation gnomon humility infirm intrinsic manatee moth oblique paregoric patristic snagging sorrowful stressful timeout torch | Correct | Agrees |
| formal_fallacies_0149 | invalid | invalid | Correct | Agrees |
| formal_fallacies_0222 | invalid | invalid | Correct | Agrees |
| formal_fallacies_0209 | invalid | invalid | Correct | Agrees |
| word_sorting_0159 | extempore gotten | extempore gotten | Correct | Agrees |
| word_sorting_0153 | cartilaginous no science spokane that'd | cartilaginous no science spokane that'd | Correct | Agrees |
| logical_deduction_five_objects_0218 | (E) | (E) | Correct | Agrees |
| word_sorting_0234 | covenant davenport densitometer noisy scoreboard sonorant thence | covenant davenport densitometer noisy scoreboard sonorant thence | Correct | Agrees |
| logical_deduction_seven_objects_0038 | (A) | (A) | Correct | Agrees |
| word_sorting_0115 | absorption aristocratic bermuda cesium cheerful congo diagram eucre ezra fallen juvenile musty nigeria nod quartile screechy slack testicle | absorption aristocratic bermuda cesium cheerful congo diagram eucre ezra fallen juvenile musty nigeria nod quartile screechy slack testicle | Correct | Agrees |
| word_sorting_0085 | besetting boyd counterweight detergent groove hide intangible menlo nv ovipositor sans spumoni | besetting boyd counterweight detergent groove hide intangible menlo nv ovipositor sans spumoni | Correct | Agrees |
| word_sorting_0124 | apprehension cashew ensemble | apprehension cashew ensemble | Correct | Agrees |
| formal_fallacies_0051 | invalid | invalid | Correct | Agrees |
| formal_fallacies_0105 | valid | valid | Correct | Agrees |
| logical_deduction_seven_objects_0089 | (D) | (D) | Correct | Agrees |
| word_sorting_0089 | ambient appropriable arroyo billion breccia coupon eardrum faze fivefold intimidate martinson o'connor perplex secretary social surtout terrestrial voltmeter | ambient appropriable arroyo billion breccia coupon eardrum faze fivefold intimidate martinson o'connor perplex secretary social surtout terrestrial voltmeter | Correct | Agrees |
| logical_deduction_five_objects_0065 | (E) | (E) | Correct | Agrees |
| formal_fallacies_0098 | invalid | invalid | Correct | Agrees |
| logical_deduction_three_objects_0086 | (C) | (C) | Correct | Agrees |

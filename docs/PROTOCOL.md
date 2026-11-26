# Research protocol v0.1 — planned, not completed

## Question and hypotheses

Does rewrite-based SFT improve correctness under meaning-preserving edits at the expense of correctness under meaning-changing edits? Can mixed augmentation improve both on held-out inputs?

- H1: original accuracy alone misses differences in grouped reliability.
- H2: rewrite-only augmentation can impair sensitivity to relevant fact changes.
- H3: mixed augmentation can improve the joint endpoint after controlling budget and data diversity.

H2/H3 are falsifiable. No observed trade-off or no benefit is an acceptable result. Never choose failure examples or exclude test items based on desired model behavior.

## Scope of the implementation

This is a mathematical-language pilot with nine authored templates. It is not evidence about general LLM reasoning, agents, alignment, or internal chain-of-thought faithfulness. The implementation edits input statements; cited trace-editing methods are related but different.

The software includes B/C/D/E data selection and answer-supervised LoRA SFT. It does not implement RLVR, new consistency losses, learned verifiers, automated semantic certification, quantization, or multi-device training. Keep these out of the project's claims.

## Dataset and review

Use all four variants as one statistical unit. The generator stores an arithmetic AST, deterministic answer, group ID, template ID, split, variant and review status. Each fact edit must produce a different answer. The same group/template ID cannot cross splits.

Review all pilot prompts before freezing. Check event order, antecedents, units, quantifiers, comparison direction, and whether all relevant facts remain. Record decisions in a separate review ledger with ID, reviewer, decision, reason and timestamp. Use a second reviewer where possible and report disagreements; do not invent reviewer agreement. Approved records must reflect actual review. Exclude a whole group if a variant is unresolved, to preserve the endpoint definition.

The current split holds out template IDs but shares broad families and some equivalent algebraic forms. A serious extension needs more independently authored templates, held-out language transformations, and an external task set with compatible labels. Adding numbers alone does not add structural diversity. Avoid claiming that synthetic data guarantees no pretraining contamination.

## Training controls

Start every arm from the same immutable model snapshot, identical adapters, optimizer settings, precision and prompts. Use seeds such as 17, 29 and 43, treating them as repeated training runs, not additional test examples.

The implementation caps total unpadded processed input+target tokens; one example may cross the cap. Report actual processed tokens, supervised tokens, steps, unique groups and variants. A step cap can stop a run before the token budget: inspect `budget_reached`.

Default data selection has different counts across arms. Thus it only supports an initial comparison. Before attributing causality to augmentation:

1. Add a B baseline with extra independent original questions to match data diversity/quantity as closely as possible.
2. Add equalized-example and repeated-original controls to separate repeated exposure from new information.
3. Compare both total processed-token and supervised-token matched runs; neither perfectly matches FLOPs.
4. Report natural-arm frequencies. In the current E, original/paraphrase/reorder/edit are each one quarter; this differs from C and D.
5. Keep evaluation data identical and locked. Tune choices only on development data. Test once after decisions are frozen; document later exploratory rounds.
6. Add a second model family before generalizing beyond one model; across families, parameter count or checkpoint differences prevent clean attribution to post-training.

Small models may show arithmetic or formatting errors; large models may saturate. Both outcomes can make this pilot non-diagnostic. Increase task diversity rather than selecting only examples your preferred model fails.

## Endpoints

Let o, p, r, e be binary correctness for original, paraphrase, reorder and edit in a group.

- Original accuracy: mean(o).
- Invariant all-correct: mean(o*p*r).
- Edit-pair correct: mean(o*e).
- Primary endpoint, group all-correct: mean(o*p*r*e).

Fix the number of variants across compared runs; all-correct rates decrease mechanically with group size. Raw agreement and answer-flip rates are optional diagnostics, not correctness metrics. If conditioning on original success, report the denominator and unconditional metric too.

Format failures count as incorrect in automatic scoring and are also reported separately. The parser accepts a bare integer or an integer on the final `FINAL:` marker line; it accepts commas and integral decimal notation. It intentionally does not guess free-form words, units or competing answers. Manually audit format failures; retain original automatic scores and version any adjudicated results separately.

## Statistics and figures

Built-in confidence intervals resample template clusters, preserving variants and instances within a sampled template. Three test templates are far too few for reliable population inference. Label intervals exploratory; more instances do not fix a lack of independent clusters. Training-seed uncertainty is separate and must be shown across runs.

For method comparisons, use paired deltas on the same groups and resample the same template clusters jointly across methods. The built-in per-run intervals alone do not establish significance of differences. Predeclare the main comparison (e.g. E versus C); treat multiple subgroup tests as exploratory or correct for multiplicity.

Recommended figures: (1) invariant correctness versus edit-pair correctness, (2) failures by template and variant, (3) primary endpoint across seeds and augmentation ratios, with real costs. No invented scores or target improvements.

## Failure analysis

Preserve complete raw outputs. Code initially labels only `wrong_answer` and `format_or_parser`. Humans may assign finer categories after audit, with disagreements recorded. A behavior is not proof of an internal mechanism. Correcting a prompt, giving a correct equation, or inspecting verbal explanations can suggest follow-up experiments but does not establish mechanistic causality.

Include counterexamples to your hypothesis. If improvements disappear under diversity controls, report that. If no trade-off is observed, restrict conclusions to tested conditions. See [FAILURE_CASES.md](FAILURE_CASES.md).

## Release criteria for real results

Release the exact code version, frozen dataset and review ledger, model/tokenizer revision, adapter identity, library versions, training budgets/seeds, prompts/decoding, raw predictions and analysis scripts. Do not publish private model outputs or data without permission. External datasets/model weights have their own licenses. Report limitations next to results, not only in an appendix.

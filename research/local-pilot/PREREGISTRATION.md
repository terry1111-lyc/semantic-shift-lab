# Local pilot preregistration

Plan written before observing model test outputs. This file and preregistration.json define an exploratory local-compute study, not a confirmatory publication experiment.

The frozen test set contains 60 four-variant groups from three templates. The primary endpoint is all-four correctness. Compare E minus C on the same groups separately for seeds 17 and 29, then report both seeds. Cluster bootstrap intervals are exploratory given only three templates.

Data are synthetic with agent-authored template inspection; no independent human review is claimed. The default arms have unequal unique variant counts. Even positive results cannot establish a method-specific causal benefit without further controls.

Development-only speed checks may alter batch size or CPU thread count for all arms, but not prompts, token caps, training budget, learning rate or held-out examples based on test outcomes. Any changes must be recorded.

## Amendment v2 — before test evaluation

The first development-only run produced valid-looking arithmetic in LaTeX and prose but failed the strict format parser, with some 96-token truncation. Before any test-set output was inspected, all arms were switched to the same answer-only instruction, integer-only SFT targets and a 16-token output cap. Original development outputs are preserved as a measurement failure diagnostic. This pilot therefore measures direct-answer behavior, not extended chain-of-thought reasoning.

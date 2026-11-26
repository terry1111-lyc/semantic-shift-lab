# Template review — assistant inspection, not independent human validation

All nine template definitions were inspected against their arithmetic programs before the study. Numbers satisfy a >= 30, b and c in [2,12], so intermediate card counts and comparisons remain nonnegative. Each edit changes the answer.

| Template | Original | Edit | Semantic check |
|---|---|---|---|
| transfer_0 | a+b-c | a-b-c | receive becomes give |
| transfer_1 | a-b-c | a-b+c | second give becomes receive |
| transfer_2 | a+b+c | a+b-c | second receive becomes give |
| comparison_0 | a+b | a-b | more becomes fewer; paraphrase reverses subject and relation together |
| comparison_1 | a+b+c | a+b-c | Ava more/fewer than Noah |
| comparison_2 | a+(a+b) | a+(a-b) | total includes both Mia and Noah |
| groups_0 | b*c+a | (b+1)*c+a | one additional box |
| groups_1 | b*(a-c) | b*(a+c) | removal/addition is per box |
| groups_2 | b*a-c | b*(a-c) | total removal becomes per-box removal; b>1,c>0 ensure changed answer |

Paraphrase and reorder versions preserve these relationships by inspection. Generated rows remain `unreviewed` to avoid presenting assistant inspection as human approval. Independent review remains required before treating this as a validated benchmark.

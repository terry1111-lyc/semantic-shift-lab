# Semantic Shift Lab — run report

**DEMONSTRATION ONLY**

No causal or publication claim is implied by this report.

Model: `label-aware-fault-injector`; backend: `demo`

| Metric | Value | Exploratory template-bootstrap 95% interval |
|---|---:|---:|
| original_accuracy | 1.000 | [1.000, 1.000] |
| invariant_all_correct | 0.333 | [0.333, 0.333] |
| edit_pair_correct | 0.889 | [0.667, 1.000] |
| group_all_correct | 0.222 | [0.000, 0.333] |

Format/parser failure rate: 0.083
Unreviewed records: 36

Exploratory percentile bootstrap over template IDs; few templates give unreliable population uncertainty. Not a training-seed CI.

See errors.jsonl for raw responses. Error categories are not mechanistic explanations.

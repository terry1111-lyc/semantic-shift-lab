# Semantic Shift Lab — run report

**MODEL RUN; HUMAN AUDIT REQUIRED**

No causal or publication claim is implied by this report.

Model: `Qwen/Qwen3-0.6B`; backend: `hf-cpu-batched`

| Metric | Value | Exploratory template-bootstrap 95% interval |
|---|---:|---:|
| original_accuracy | 0.267 | [0.000, 0.800] |
| invariant_all_correct | 0.033 | [0.000, 0.100] |
| edit_pair_correct | 0.183 | [0.000, 0.550] |
| group_all_correct | 0.033 | [0.000, 0.100] |

Format/parser failure rate: 0.004
Unreviewed records: 240

Exploratory percentile bootstrap over template IDs; few templates give unreliable population uncertainty. Not a training-seed CI.

See errors.jsonl for raw responses. Error categories are not mechanistic explanations.

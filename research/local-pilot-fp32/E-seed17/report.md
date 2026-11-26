# Semantic Shift Lab — run report

**MODEL RUN; HUMAN AUDIT REQUIRED**

No causal or publication claim is implied by this report.

Model: `Qwen/Qwen3-0.6B`; backend: `hf-cpu-batched`

| Metric | Value | Exploratory template-bootstrap 95% interval |
|---|---:|---:|
| original_accuracy | 0.283 | [0.000, 0.850] |
| invariant_all_correct | 0.033 | [0.000, 0.100] |
| edit_pair_correct | 0.017 | [0.000, 0.050] |
| group_all_correct | 0.000 | [0.000, 0.000] |

Format/parser failure rate: 0.000
Unreviewed records: 240

Exploratory percentile bootstrap over template IDs; few templates give unreliable population uncertainty. Not a training-seed CI.

See errors.jsonl for raw responses. Error categories are not mechanistic explanations.

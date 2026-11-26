# Changelog

## 0.2.0 — 2026-11-26

- Completed the real Qwen3-0.6B CPU pilot: six LoRA fits, seven evaluations, two training seeds and 1,680 saved test outputs.
- Added frozen data and protocols, raw predictions, loss histories, paired exploratory comparisons, Chinese report, figures, observed failures and reproduction scripts.
- Preserved earlier measurement failures and interrupted bfloat16 training as diagnostics; documented protocol amendments and explicit float32 rerun.
- Made CPU model loading explicitly float32. Full 32-test run passed after this change.
- Added evidence re-scoring verification and a separately exported, reload-checked six-adapter archive. No base model weights or paid cloud resources used.
- The pilot did not establish a reliable advantage of mixed augmentation. Independent human review, external evaluation and stronger controls remain future work.

## 0.1.0 — 2026-11-25

- Initial runnable project, synthetic demo fixtures, test suite and research protocol. No real pretrained-model research results in that release.

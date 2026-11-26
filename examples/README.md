# Synthetic fixture — not model results

`pilot.jsonl` contains 27 generated groups (108 records), with 9 groups in each split. The semantic review status is intentionally `unreviewed`.

`demo-run/` contains test-split predictions from the label-aware fault injector, with known wrong answers and format failures. Every output report is marked **DEMONSTRATION ONLY**. None of these predictions comes from an LLM.

Reproduce the fixture in a new output directory:

```bash
python -m semantic_shift run --data examples/pilot.jsonl --output runs/fixture-replay --backend demo --seed 17
python -m semantic_shift evaluate --data examples/pilot.jsonl --run runs/fixture-replay --bootstrap 500
```

Generated data and prediction values are deterministic for this seed. Metadata timestamps and Python versions depend on the execution environment.

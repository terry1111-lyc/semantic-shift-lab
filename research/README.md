# Actual-model research and reproduction

The completed study is in [`local-pilot-fp32/`](local-pilot-fp32/REPORT.zh-CN.md). It contains real Qwen3-0.6B predictions, frozen synthetic data, six training histories, two seeds, seven evaluation runs, observed failures and exploratory analysis. [`local-pilot/`](local-pilot/TRAINING_INTERRUPTION.md) preserves the earlier development diagnostics and interrupted bfloat16 attempt. Neither directory uses the synthetic predictions in `examples/`.

No server was rented and no inference API was used. The completed round used an Apple M1 with 16 GB memory, CPU float32, four intra-op threads and one inter-op thread. MPS was unavailable in this execution context. These settings are a small reproducible pilot, not a recommendation to train larger models on CPU.

## Inspect without loading a model

From the repository root, using Python 3.10+:

```bash
python research/verify_results.py
```

This re-scores all 1,680 predictions, checks saved summaries/errors and training histories, and verifies the source hashes frozen before the sweep. Fingerprints use canonical JSON, not raw file bytes. The full 720-item dataset fingerprint differs from each run's 240-item test-subset fingerprint by design.

For charts, install Matplotlib (the recorded run used the version in `release-validation.json`), then:

```bash
python research/analyze_local_study.py --study research/local-pilot-fp32
python research/write_observed_failures.py
python research/write_chinese_report.py
```

The report writers describe the shipped completed study. They are not general-purpose report generators for arbitrary new experiments. Analysis regenerates derived reports; raw prediction logs remain unchanged.

## Reproduce the training sweep

Use an isolated environment. `requirements-hf-tested.txt` records the main tested versions, not a complete lockfile. Availability on another platform can differ. Install this repository and those versions, then download the pinned public snapshot:

```bash
python -m pip install -r requirements-hf-tested.txt
python -m pip install -e .
python research/download_snapshot.py --output runs/models/qwen3-0.6b
mkdir -p runs/reproduction
cp research/local-pilot-fp32/data.jsonl runs/reproduction/data.jsonl
cp research/local-pilot-fp32/preregistration.json runs/reproduction/preregistration.json
python research/run_sweep.py --model-local runs/models/qwen3-0.6b --study runs/reproduction --scratch checkpoints/reproduction
python research/analyze_local_study.py --study runs/reproduction
```

Model: `Qwen/Qwen3-0.6B`, Hub commit `c1899de289a04d12100db370d81485cdf75e47ca`. The downloader fetches about 1.4 GB of base weights plus tokenizer files; it does not rent compute. Weights are excluded from the source archive. Model license information is in the downloaded snapshot.

The sweep runs A, then B/C/E at seeds 17 and 29. It refuses existing evaluation folders and stops on a failed process. Training saves separate LoRA adapters in the scratch path. Each process has a 15-minute watchdog; slower hardware can hit this limit. Preserve any partial run and document a watchdog change before retrying in a new directory. The general CLI has different default prompts/decoding: use the research scripts to reproduce this pilot.

Runs use identical batch size, ordering, prompt and greedy 16-token decoding. Floating-point libraries and devices can change predictions even with fixed seeds. Match the recorded environment for the closest comparison. No bitwise cross-platform reproducibility is promised.

## Trained adapters

The separate `semantic-shift-lab-adapters-v0.2.0.zip` release contains all six adapters, their training manifests, weight checksums and reload instructions. It is optional for source-only reproduction. Exported adapter configs replace the original local base-model path with the pinned public model ID; weight bytes are unchanged. The archive's release manifest records this packaging change. No base weights are included.

## Limits and protocol history

The completed float32 sweep is an engineering rerun after inspecting an earlier baseline. It is exploratory, not a newly blinded confirmatory test. The initial long-answer development check revealed correct mathematics rejected by the strict parser, leading to an answer-only protocol before the original baseline test. That protocol subsequently showed substantial floor effects. Both changes and all failed attempts are documented.

Only three test templates, two training seeds and a 6,144-token budget per fit were used. Data received assistant template review, not independent human review. The methods differ in unique-example coverage and original-question exposure. Arm D, external benchmarks, a second model family, RL and matched-diversity controls were not run. The small E−C difference does not establish a dependable benefit.

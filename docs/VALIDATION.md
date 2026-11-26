# Validation record

Validated locally on 2026-11-25–26. This record describes software checks, not substantive LLM research results.

**Latest full run: 32 tests passed, including one optional CPU integration test.**

## Core tests

The dependency-free suite covers deterministic generation, all nine arithmetic programs against manually specified answers, group integrity, split leakage, invalid labels, answer parsing, exact prediction coverage, dataset fingerprints, cluster bootstrap behavior, train-only selection, assistant-token masking, and end-to-end CLI reporting.

Run:

```bash
python -m unittest discover -s tests -v
```

## Real-library integration

A tiny randomly initialized GPT-2 model and local WordLevel tokenizer were created without downloading pretrained weights. The test completed inference, LoRA optimization, adapter save/reload, metadata identity checks and scoring on CPU.

Tested environment:

| Component | Version |
|---|---|
| Python | 3.13.13 |
| PyTorch | 2.15.0 |
| Transformers | 5.24.0 |
| PEFT | 0.22.0 |
| Accelerate | 1.16.0 |
| Tokenizers | 0.23.3 |
| Safetensors | 0.8.0 |

```bash
RUN_HF_INTEGRATION=1 python -m unittest discover -s tests -p test_hf_integration.py -v
```

PEFT was loaded from a separate temporary dependency directory; the supplied Python environment was not modified. MPS was compiled into PyTorch, but `torch.backends.mps.is_available()` returned false in this execution context. MPS and CUDA execution were not validated. This does not establish whether MPS works in a different terminal/session on the same computer.

Compatibility with arbitrary pretrained models, large-scale training, quantization, GPU memory limits has not been verified beyond the specific Qwen3-0.6B study below. The broad optional dependency ranges are compatibility intentions, not evidence that every version combination was tested. The reference pins in `requirements-hf-tested.txt` identify the tested main libraries, not a complete platform-independent lockfile.

The optional integration test is skipped in normal core CI. CI configuration is included, but GitHub-hosted runs cannot be claimed until the repository is published and Actions executes.

## Completed actual-model study

The separately documented [float32 pilot](../research/local-pilot-fp32/REPORT.zh-CN.md) ran the pinned public Qwen3-0.6B snapshot on CPU: six LoRA trainings and seven complete 240-item evaluations. The full 32-test suite passed after explicit float32 loading was introduced. Training and inference source hashes were captured before the completed sweep. The release verification script re-scores all saved predictions and checks frozen source/data fingerprints without downloading a model.

The real study is evidence that this specific path executes, not evidence of compatibility with every model. It also exposed strict-format false negatives, token-cap confounds, weak generalization and an interrupted bfloat16 training attempt. That attempt is retained as a diagnostic and is not counted among completed training runs. See [reproduction and evidence](../research/README.md).

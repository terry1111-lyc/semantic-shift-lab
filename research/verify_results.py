"""Recompute the released study's scores and verify its frozen evidence; no ML dependencies."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from semantic_shift.data import validate
from semantic_shift.evaluation import evaluate
from semantic_shift.io import fingerprint, read_jsonl


def verify():
    study = ROOT / 'research/local-pilot-fp32'
    plan = json.loads((study / 'preregistration.json').read_text())
    rows = read_jsonl(study / 'data.jsonl')
    validate(rows)
    assert fingerprint(rows) == plan['dataset_sha256'], 'Full dataset changed'
    for name, digest in json.loads((study / 'source-hashes.json').read_text()).items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    test = [r for r in rows if r['split'] == 'test']
    runs = ['A'] + [f'{arm}-seed{seed}' for seed in (17, 29) for arm in 'BCE']
    prompts = set()
    count = 0
    for name in runs:
        folder = study / name
        meta = json.loads((folder / 'metadata.json').read_text())
        assert meta['status'] == 'complete' and meta['is_demo'] is False
        assert meta['revision'] == plan['revision']
        assert meta['dtype'] == 'float32' and meta['batch_size'] == 8
        assert meta['prompt_instruction'] == plan['prompt_instruction']
        prompts.add(meta['prompt_sha256'])
        predictions = read_jsonl(folder / 'predictions.jsonl')
        summary, errors = evaluate(test, predictions, meta, 1000)
        assert summary == json.loads((folder / 'summary.json').read_text()), name
        assert errors == read_jsonl(folder / 'errors.jsonl'), name
        count += len(predictions)
        if name != 'A':
            manifest = json.loads((study / 'training' / f'{name}-manifest.json').read_text())
            history = json.loads((study / 'training' / f'{name}-losses.json').read_text())
            assert manifest['dataset_sha256'] == plan['dataset_sha256']
            assert manifest['hub_revision'] == plan['revision']
            assert manifest['config']['dtype'] == 'float32'
            assert len(history) == manifest['steps']
            assert history[-1]['processed_tokens'] == manifest['processed_tokens']
            assert manifest['budget_reached'] is True
    assert len(prompts) == 1 and count == 1680
    print(json.dumps({'verified_runs': len(runs), 'rescored_predictions': count,
                      'frozen_source_files': 5, 'training_histories': 6, 'status': 'passed'}))


if __name__ == '__main__':
    verify()

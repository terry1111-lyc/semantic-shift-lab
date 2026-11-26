import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from .data import generate, validate, expression
from .identity import validate_model_identity, directory_digest
from .evaluation import evaluate, markdown_report
from .io import read_jsonl, write_jsonl, save_json, fingerprint
from .models import demo_predictions, messages, HuggingFaceModel


def new_directory(path):
    path = Path(path)
    if path.exists():
        raise ValueError(f"Output already exists: {path}. Use a new directory to preserve provenance.")
    path.mkdir(parents=True)
    return path


def run(data_path, output, split="test", backend="demo", model=None, revision=None,
        adapter=None, device="cpu", max_new_tokens=256, seed=17):
    all_rows = read_jsonl(data_path)
    validate(all_rows)
    rows = [r for r in all_rows if r["split"] == split]
    validate(rows)
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    if backend not in {"demo", "hf"}:
        raise ValueError("Unknown backend")
    if backend == "hf":
        validate_model_identity(model, revision)
    adapter_manifest = None
    adapter_sha = None
    if adapter:
        if backend != "hf":
            raise ValueError("Adapters require the HF backend")
        manifest = Path(adapter) / "training_manifest.json"
        if not manifest.exists():
            raise ValueError("Adapter needs a training_manifest.json produced by this project")
        adapter_manifest = json.loads(manifest.read_text())
        trained = adapter_manifest["config"]
        if trained["model"] != model or trained["revision"] != revision:
            raise ValueError("Adapter and inference base-model identities differ")
        import hashlib
        adapter_files = sorted(Path(adapter).glob("adapter*"))
        if not adapter_files:
            raise ValueError("No adapter files found")
        hasher = hashlib.sha256()
        for file in adapter_files:
            hasher.update(file.name.encode())
            hasher.update(file.read_bytes())
        adapter_sha = hasher.hexdigest()
    output = new_directory(output)
    meta = dict(schema_version=1, is_demo=backend == "demo", backend=backend,
                model="label-aware-fault-injector" if backend == "demo" else model,
                revision=revision, adapter_sha256=adapter_sha, split=split,
                dataset_sha256=fingerprint(rows), source_sha256=fingerprint(all_rows),
                prompt_sha256=fingerprint([messages(r["prompt"]) for r in rows]),
                seed=seed, max_new_tokens=max_new_tokens, decoding="greedy" if backend == "hf" else "synthetic",
                python=platform.python_version(), device=device,
                created_utc=datetime.now(timezone.utc).isoformat(), status="running")
    save_json(output / "metadata.json", meta)
    if backend == "demo":
        predictions = demo_predictions(rows, seed)
        write_jsonl(output / "predictions.jsonl", predictions)
    else:
        import random
        import torch
        random.seed(seed)
        torch.manual_seed(seed)
        engine = HuggingFaceModel(model, revision, adapter, device)
        meta["versions"] = engine.versions
        meta["resolved_revision"] = engine.resolved_revision
        # Write one record at a time. Interrupted runs remain status=running and cannot be scored.
        with (output / "predictions.jsonl").open("w", encoding="utf-8") as stream:
            for row in rows:
                pred = dict(id=row["id"], **engine.predict(row["prompt"], max_new_tokens))
                stream.write(json.dumps(pred, ensure_ascii=False) + "\n")
                stream.flush()
    meta["status"] = "complete"
    save_json(output / "metadata.json", meta)
    return output


def score(data_path, run_path, repeats=1000):
    directory = Path(run_path)
    meta = json.loads((directory / "metadata.json").read_text())
    if meta.get("status") != "complete":
        raise ValueError("Incomplete run; do not score partial results")
    all_rows = read_jsonl(data_path)
    validate(all_rows)
    if fingerprint(all_rows) != meta.get("source_sha256"):
        raise ValueError("Source dataset changed since inference")
    rows = [r for r in all_rows if r["split"] == meta["split"]]
    summary, errors = evaluate(rows, read_jsonl(directory / "predictions.jsonl"), meta, repeats)
    save_json(directory / "summary.json", summary)
    write_jsonl(directory / "errors.jsonl", errors)
    (directory / "report.md").write_text(markdown_report(summary), encoding="utf-8")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="Semantic Shift Lab — research pilot, not published model results")
    sub = parser.add_subparsers(dest="command", required=True)
    identity = sub.add_parser("model-revision", help="Hash a local model snapshot")
    identity.add_argument("--model", required=True)
    g = sub.add_parser("generate", help="Create a deterministic pilot dataset")
    g.add_argument("--output", required=True)
    g.add_argument("--per-template", type=int, default=20)
    g.add_argument("--seed", type=int, default=17)
    v = sub.add_parser("validate", help="Check structural invariants; does not certify natural-language semantics")
    v.add_argument("--data", required=True)
    r = sub.add_parser("run", help="Run demo faults or a real HF model")
    r.add_argument("--data", required=True)
    r.add_argument("--output", required=True)
    r.add_argument("--split", choices=["train", "dev", "test"], default="test")
    r.add_argument("--backend", choices=["demo", "hf"], default="demo")
    r.add_argument("--model")
    r.add_argument("--revision")
    r.add_argument("--adapter")
    r.add_argument("--device", default="cpu")
    r.add_argument("--max-new-tokens", type=int, default=256)
    r.add_argument("--seed", type=int, default=17)
    e = sub.add_parser("evaluate", help="Score a complete run with exact coverage and provenance checks")
    e.add_argument("--data", required=True)
    e.add_argument("--run", required=True)
    e.add_argument("--bootstrap", type=int, default=1000)
    x = sub.add_parser("export-sft", help="Export only training rows for a specified condition")
    x.add_argument("--data", required=True)
    x.add_argument("--condition", choices=["B", "C", "D", "E"], required=True)
    x.add_argument("--output", required=True)
    t = sub.add_parser("train", help="Optional single-device LoRA SFT")
    t.add_argument("--config", required=True)
    t.add_argument("--data", required=True)
    t.add_argument("--output", required=True)
    d = sub.add_parser("demo", help="Run the entire pipeline offline; inject synthetic failures")
    d.add_argument("--output", default="runs/demo")
    args = parser.parse_args(argv)
    try:
        if args.command == "model-revision":
            print(directory_digest(args.model))
        elif args.command == "generate":
            if Path(args.output).exists():
                raise ValueError("Dataset output already exists")
            rows = generate(args.per_template, args.seed)
            write_jsonl(args.output, rows)
            save_json(str(args.output)+".manifest.json", dict(**validate(rows), sha256=fingerprint(rows),
                      seed=args.seed, per_template=args.per_template, version="0.1.0", license="MIT",
                      note="Synthetic pilot. Structural validation does not substitute for semantic review."))
            print(json.dumps(validate(rows)))
        elif args.command == "validate":
            print(json.dumps(validate(read_jsonl(args.data))))
        elif args.command == "run":
            print(run(args.data, args.output, args.split, args.backend, args.model, args.revision,
                      args.adapter, args.device, args.max_new_tokens, args.seed))
        elif args.command == "evaluate":
            print(json.dumps(score(args.data, args.run, args.bootstrap), indent=2))
        elif args.command == "export-sft":
            from .training import select_training
            if Path(args.output).exists():
                raise ValueError("SFT output already exists")
            rows = select_training(read_jsonl(args.data), args.condition)
            exported = [dict(id=r["id"], group_id=r["group_id"], condition=args.condition,
                             messages=messages(r["prompt"]) + [{"role": "assistant", "content":
                             f"{expression(r['program'])} = {r['answer']}\nFINAL: {r['answer']}"}]) for r in rows]
            write_jsonl(args.output, exported)
            print(f"Exported {len(exported)} training records")
        elif args.command == "train":
            from .training import train
            print(json.dumps(train(args.config, args.data, args.output)))
        elif args.command == "demo":
            dest = new_directory(args.output)
            data = dest / "data.jsonl"
            write_jsonl(data, generate())
            directory = run(data, dest / "fault-injector")
            score(data, directory)
            print(f"DEMONSTRATION ONLY. Report: {directory / 'report.md'}")
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()

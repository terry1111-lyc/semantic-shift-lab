"""Content identity for local model snapshots and pinned Hub models."""
import hashlib
from pathlib import Path


def directory_digest(path):
    path = Path(path)
    files = sorted(p for p in path.rglob("*") if p.is_file() and not any(part.startswith(".") for part in p.relative_to(path).parts))
    if not files:
        raise ValueError("Model directory is empty")
    digest = hashlib.sha256()
    for file in files:
        digest.update(str(file.relative_to(path)).encode())
        digest.update(b"\0")
        digest.update(str(file.stat().st_size).encode())
        digest.update(b"\0")
        with file.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def validate_model_identity(model, revision):
    if not model or not revision:
        raise ValueError("Model and immutable revision are required")
    if Path(model).is_dir():
        if directory_digest(model) != revision:
            raise ValueError("Local model revision must match its directory SHA256; use model-revision")
        return
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision.lower()):
        raise ValueError("Hub revision must be a 40-character model commit SHA")

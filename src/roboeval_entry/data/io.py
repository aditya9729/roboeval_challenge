"""Small artifact IO helpers: atomic JSON writes and file hashing."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def write_json_atomic(path: str | Path, obj) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    tmp.replace(path)
    return path


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            digest.update(block)
    return digest.hexdigest()

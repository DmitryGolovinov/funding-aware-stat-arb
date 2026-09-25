"""Run manifests: which code, protocol, data and environment produced an artifact."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path


def tree_hash(root: Path) -> str:
    h = hashlib.sha256()
    for p in sorted({*root.glob("src/**/*.py"), *root.glob("scripts/*.py")}):
        h.update(str(p.relative_to(root)).encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(path: Path, record: dict, outputs: list[Path]) -> None:
    rec = dict(record)
    rec["written_utc"] = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    rec["environment"] = {"python": sys.version.split()[0], "platform": platform.platform(),
                          "packages": {p: _v(p) for p in ("numpy", "pandas", "lightgbm", "pyarrow")}}
    rec["outputs"] = {p.name: sha256_file(p) for p in outputs if p.exists()}
    Path(path).write_text(json.dumps(rec, indent=1, sort_keys=True, default=str))


def _v(p):
    try:
        return metadata.version(p)
    except metadata.PackageNotFoundError:
        return None

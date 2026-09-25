"""Claim checker: every registered numeric claim is recomputed from current artifacts.

``docs/claims.yaml`` lists claims as {id, text, file, kind, selector, value, tol}. ``kind`` is
``csv`` (selector: {filter: {col: val}, column: name}), ``json`` (selector: dotted path), or
``external`` (a value quoted from a cited source, listed so it is not mistaken for our result).
The script also scans README.md for decimal numbers and reports any that is not the value (or
magnitude) of a registered claim, so unsupported numbers cannot slip into the public text;
generation labels such as "Generation 2.1" are identifiers, not numbers.

Every cited manifest must come from the current source tree, OR from an older tree that
``docs/release_lineage.json`` classifies for exactly this manifest (status ``unaffected``,
``reproduced`` or ``superseded``, with evidence) while recording the current tree. Historical
manifests are never rewritten; the lineage file states why their results still stand.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def lookup(c: dict) -> float:
    path = ROOT / c["file"]
    if c["kind"] == "csv":
        df = pd.read_csv(path)
        for k, v in c["selector"].get("filter", {}).items():
            df = df[df[k].astype(str) == str(v)]
        if len(df) != 1:
            raise ValueError(f"{c['id']}: selector matched {len(df)} rows")
        return float(df[c["selector"]["column"]].iloc[0])
    obj = json.loads(path.read_text())
    for part in c["selector"].split("."):
        obj = obj[int(part)] if isinstance(obj, list) else obj[part]
    return float(obj)


LINEAGE_OK = {"unaffected", "reproduced", "superseded"}


def lineage_entry(current: str, manifest: str, tree: str | None) -> dict | None:
    p = ROOT / "docs" / "release_lineage.json"
    if tree is None or not p.exists():
        return None
    lin = json.loads(p.read_text())
    if lin.get("current_source_tree_sha256") != current:
        return None  # the code changed after the lineage was written: it must be regenerated
    for a in lin.get("artifacts", []):
        if a["manifest"] == manifest and a["source_tree"] == tree and a["status"] in LINEAGE_OK:
            return a
    return None


def stale_manifests(spec: dict) -> int:
    """Fail if a cited manifest comes from other code than the current tree without lineage."""
    sys.path.insert(0, str(ROOT / "src"))
    pkg = next(p.name for p in (ROOT / "src").iterdir() if (p / "manifest.py").exists())
    tree_hash = __import__(f"{pkg}.manifest", fromlist=["tree_hash"]).tree_hash
    current, bad = tree_hash(ROOT), 0
    for m in spec.get("manifests", []):
        rec = json.loads((ROOT / m).read_text())
        # Generation-1 manifests use ``source_tree_sha256``, some later ones ``source_tree``.
        tree = rec.get("source_tree_sha256") or rec.get("source_tree")
        if tree == current:
            print(f"OK   manifest {m} source tree matches current code")
            continue
        entry = lineage_entry(current, m, tree)
        if entry:
            print(f"OK   manifest {m} historical tree {tree[:12]} ({entry['status']}, lineage)")
        else:
            print(f"FAIL manifest {m} source tree {str(tree)[:12]}: neither current nor in lineage")
            bad += 1
    return bad


def main() -> int:
    spec = yaml.safe_load((ROOT / "docs" / "claims.yaml").read_text()) or {}
    claims = spec.get("claims", [])
    bad, shown = stale_manifests(spec), set()
    for c in claims:
        if c["kind"] == "external":  # quoted from a cited source (e.g. a paper table), not ours
            shown.add(f"{abs(c['value']):.{c.get('decimals', 2)}f}")
            print(f"EXT  {c['id']}: {c['value']} quoted from {c['source']}")
            continue
        v = lookup(c) * c.get("scale", 1.0)
        ok = abs(v - c["value"]) <= c.get("tol", 1e-9) * max(1.0, abs(c["value"]))
        shown.add(f"{abs(c['value']):.{c.get('decimals', 2)}f}")
        print(f"{'OK  ' if ok else 'FAIL'} {c['id']}: artifact {v:.6g} vs claimed {c['value']}")
        bad += not ok
    readme = (ROOT / "README.md").read_text() if (ROOT / "README.md").exists() else ""
    body = re.sub(r"`[^`]*`|\([^)]*\.(?:md|png|csv|json|py)\)|https?://\S+", "", readme)  # code, file links, URLs
    body = re.sub(r"\bGeneration \d+(?:\.\d+)?", "", body)  # labels, not numbers
    nums = set(re.findall(r"(?<![\w.])-?\d+\.\d+(?![\d.])", body))
    unregistered = sorted(n for n in nums if n.lstrip("-") not in shown)
    for n in unregistered:
        print(f"WARN README number {n} is not a registered claim")
    return 1 if bad or unregistered else 0


if __name__ == "__main__":
    sys.exit(main())

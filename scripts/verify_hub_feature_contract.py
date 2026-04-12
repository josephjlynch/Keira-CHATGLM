#!/usr/bin/env python3
"""
Verify that config/hub_feature_names.json matches the built-in fallback tuples in
scentsation_hub.py and scentsation_ml/export.py (detect drift before a demo or CI).

Does not import those modules (avoids optional deps). Parses tuple literals with ast.

Run from repo root: python scripts/verify_hub_feature_contract.py
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
JSON_PATH = ROOT / "config" / "hub_feature_names.json"
HUB_PY = ROOT / "scentsation_hub.py"
EXPORT_PY = ROOT / "scentsation_ml" / "export.py"
EXPECTED_LEN = 6
TARGET = "_FALLBACK_HUB_FEATURE_NAMES"


def _tuple_from_assign(path: Path) -> tuple[str, ...]:
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src)

    def extract_from_value(val: ast.expr) -> tuple[str, ...]:
        if not isinstance(val, ast.Tuple):
            raise ValueError(f"{path}: {TARGET} is not a tuple")
        out: list[str] = []
        for elt in val.elts:
            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                out.append(elt.value)
            elif isinstance(elt, ast.Str):  # py<3.8 compat
                out.append(elt.s)
            else:
                raise ValueError(f"{path}: non-string element in {TARGET}")
        return tuple(out)

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == TARGET:
                    return extract_from_value(node.value)
        if isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == TARGET:
                if node.value is None:
                    raise ValueError(f"{path}: {TARGET} has no value")
                return extract_from_value(node.value)
    raise ValueError(f"{path}: {TARGET} not found")


def main() -> int:
    if not JSON_PATH.is_file():
        print(f"Missing {JSON_PATH}", file=sys.stderr)
        return 1
    with open(JSON_PATH, encoding="utf-8") as f:
        names = json.load(f)
    if not isinstance(names, list) or len(names) != EXPECTED_LEN:
        print(
            f"BAD: {JSON_PATH} must be a JSON array of length {EXPECTED_LEN}",
            file=sys.stderr,
        )
        return 1
    from_json = tuple(str(x) for x in names)

    try:
        hub_fb = _tuple_from_assign(HUB_PY)
        exp_fb = _tuple_from_assign(EXPORT_PY)
    except ValueError as e:
        print(e, file=sys.stderr)
        return 1

    if from_json != hub_fb:
        print("Mismatch: JSON vs scentsation_hub._FALLBACK_HUB_FEATURE_NAMES", file=sys.stderr)
        print(f"  json: {from_json!r}", file=sys.stderr)
        print(f"  hub:  {hub_fb!r}", file=sys.stderr)
        return 1
    if from_json != exp_fb:
        print(
            "Mismatch: JSON vs scentsation_ml/export._FALLBACK_HUB_FEATURE_NAMES",
            file=sys.stderr,
        )
        print(f"  json: {from_json!r}", file=sys.stderr)
        print(f"  export: {exp_fb!r}", file=sys.stderr)
        return 1
    print("OK: config/hub_feature_names.json matches hub and export fallback tuples.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Rebuild showcase/artifacts/model_registry.json from the scenarios.

Run after adding, removing or editing a scenario, or after retraining a
checkpoint — anything that changes which models exist or what their bytes are.
`run_scenario.py` also calls this at the end of every evaluation, so it is only
needed on its own for a model that has not been evaluated yet.

Usage:
    python scripts/build_model_registry.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from verifai.export.model_registry import write_registry  # noqa: E402


def main() -> None:
    path = write_registry(root=ROOT)
    reg = json.loads(path.read_text(encoding="utf-8"))
    print(f"✓ {len(reg['models'])} model(s) in {len(reg['projects'])} project(s) -> "
          f"{path.relative_to(ROOT)}")
    for m in reg["models"]:
        ident = m["sha256"] or m["identity"]
        trained = f"trained by {m['trained_by']}" if m["trained_by"] else "not trained here"
        print(f"   · {m['key']:<28} {len(m['configurations'])} config(s)  {ident:<40} {trained}")


if __name__ == "__main__":
    main()

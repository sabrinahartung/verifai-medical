#!/usr/bin/env python
"""Re-run every active scenario — what to do after a metric changes.

Archived scenarios are the record of an experiment and are never re-run here;
the showcase marks them as evaluated with the metrics of their day. Active ones
are kept current, and after a metric's version is bumped in
`verifai/core/suite.py` the showcase flags them as behind until this has run.

Usage:
    uv run python scripts/run_active.py            # run them
    uv run python scripts/run_active.py --list     # only say which would run
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def active_scenarios(scenarios_dir: Path = ROOT / "scenarios") -> list[Path]:
    return [p for p in sorted(scenarios_dir.glob("*.yaml"))
            if yaml.safe_load(p.read_text(encoding="utf-8")).get("status") == "active"]


def main(argv: list[str]) -> None:
    paths = active_scenarios()
    print(f"{len(paths)} active scenario(s): " + ", ".join(p.stem for p in paths))
    if "--list" in argv:
        return
    import os
    os.chdir(ROOT)               # run_scenario.py writes to paths relative to the repo root
    from scripts.run_scenario import main as run_one
    for p in paths:
        run_one(str(p.relative_to(ROOT)))


if __name__ == "__main__":
    main(sys.argv[1:])

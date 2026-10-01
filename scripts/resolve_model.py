#!/usr/bin/env python
"""Draft a scenario from a model reference — the resolver, from the command line.

Reads what a checkpoint can say about itself and prints a scenario marked
`draft: true`, with a TODO wherever it cannot. Nothing is run and no weights are
downloaded; `run_scenario` refuses the draft until every TODO is answered.

Usage:
    uv run python scripts/resolve_model.py hf:owner/repo[@revision]
    uv run python scripts/resolve_model.py path/to/weights.pt
    uv run python scripts/resolve_model.py hf:owner/repo --dataset data/manifests/ham10000_test.csv
    uv run python scripts/resolve_model.py hf:owner/repo -o scenarios/new_model.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from verifai.models.resolve import resolve


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("ref", help="hf:owner/repo[@revision], or a local weights file")
    ap.add_argument("--dataset", help="an evaluation manifest, to propose a label map against")
    ap.add_argument("-o", "--out", help="write the draft here instead of printing it")
    args = ap.parse_args(argv)

    draft = resolve(args.ref, dataset_manifest=args.dataset)
    text = draft.to_yaml()
    if args.out:
        out = Path(args.out)
        if out.exists():
            raise SystemExit(f"{out} exists; a draft never overwrites a scenario")
        out.write_text(text, encoding="utf-8")
        print(f"✓ draft with {len(draft.todos)} open TODO(s) -> {out}", file=sys.stderr)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()

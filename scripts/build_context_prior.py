#!/usr/bin/env python
"""Build a context prior: how much each class's odds shift given age and site.

Why this exists
---------------
Every lever tried so far changed the pixels pipeline — more images, a bigger
backbone, a different loss. The learning curve then showed that *distribution*
matters more than volume, which points somewhere else: at information that is not
in the pixels at all.

`age`, `sex` and `localization` sit on every manifest row (98% / 98% / 90%
populated in training, 100% in test) and feed no model. A dermatologist uses site
and age; an image-only classifier cannot. This turns them into a decision-time
prior, so the model is untouched and only the reading of its probabilities
changes — the one lever that has actually worked in this project.

The arithmetic
--------------
Bayes, with the model supplying the likelihood:

    P(class | image, context)  ∝  P(class | image) · P(class | context) / P(class)
                                  ^ the model          ^ this table, the "lift"

A lift of 1.0 is neutral. Measured on isic_train.csv, melanoma runs from 0.06 at
age 0-19 to 1.52 at 80-99, and reaches 2.72 on palms and soles — roughly a
twentyfold swing between the extremes, which is far too much signal to leave on
the floor.

What this deliberately does not include
---------------------------------------
`sex`. Its melanoma lift is 0.96 against 1.03 — nothing — while conditioning on
it would make the model's behaviour across sexes a design decision that has to be
defended. Signal that small is not worth that argument.

`dx_type` must never become a feature. It is 100% populated in test and 0% in
training, and "histo" records that a clinician already thought the lesion worth
cutting out. It is an outcome, not an input, and using it would look like a
spectacular result while being pure leakage that `audit_split` cannot see.

Computed from the TRAINING manifest only. Derived from test, this would be
fitting the decision rule to the answers.

Usage:
    python scripts/build_context_prior.py --manifest data/manifests/isic_train.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Bucketing lives in the engine and is imported by both sides. A second copy here
# would be free to drift, and a drifted bucket fails silently: every lookup misses,
# every lift falls back to 1.0, and the experiment reports "context does not help"
# without ever having applied context.
sys.path.insert(0, str(REPO))
from verifai.core.context import BUCKETERS, bucket_for   # noqa: E402


def _resolve(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (REPO / path)


def build(rows: list[dict], features: list[str], min_count: int,
          smoothing: float) -> dict:
    """Lift per (feature, bucket, class), with counts kept for inspection.

    Smoothed towards the base rate, and a bucket thinner than `min_count` is
    dropped to a lift of 1.0 rather than trusted. A cell with four cases can
    produce a lift of 4 that is entirely noise, and a decision rule is exactly
    the wrong place to let that through.
    """
    base = Counter(r["label"] for r in rows)
    total = sum(base.values())
    classes = sorted(base)
    base_rate = {c: base[c] / total for c in classes}

    out: dict[str, dict] = {}
    for feature in features:
        per_bucket: dict[str, Counter] = defaultdict(Counter)
        for r in rows:
            b = bucket_for(feature, r.get(feature, ""))
            if b is not None:
                per_bucket[b][r["label"]] += 1

        table: dict[str, dict] = {}
        for bucket, counts in sorted(per_bucket.items()):
            n = sum(counts.values())
            if n < min_count:
                continue
            lifts = {}
            for c in classes:
                # Smoothed towards the base rate with a fixed pseudo-count, so the
                # pull fades as a bucket fills. Scaling it with n instead would
                # damp the signal permanently: at 596 cases a well-measured lift of
                # 0.06 came back as 0.37, which is the metric being suppressed
                # rather than stabilised. A class absent from a bucket still gets a
                # small lift rather than zero, which would veto it outright.
                p = (counts[c] + smoothing * base_rate[c]) / (n + smoothing)
                lifts[c] = round(p / base_rate[c], 4)
            table[bucket] = {"n": n, "lift": lifts}
        out[feature] = table
    return {"classes": classes, "base_rate": {c: round(base_rate[c], 5) for c in classes},
            "n_train": total, "min_count": min_count, "smoothing": smoothing,
            "features": out}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifests/isic_train.csv")
    ap.add_argument("--out", default="data/priors/isic.json")
    ap.add_argument("--features", default="age,localization",
                    help="sex is excluded on purpose — see the module docstring")
    ap.add_argument("--min-count", type=int, default=50,
                    help="buckets thinner than this are treated as neutral")
    ap.add_argument("--smoothing", type=float, default=20.0,
                    help="pseudo-counts pulling towards the base rate; fades as a "
                         "bucket fills. 0 = raw counts")
    a = ap.parse_args()

    path = _resolve(a.manifest)
    if not path.exists():
        sys.exit(f"not found: {path}")
    if "test" in path.name:
        sys.exit(f"refusing to build a prior from {path.name} — this must come from "
                 f"training data, or the decision rule is fitted to the answers")

    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    features = [x.strip() for x in a.features.split(",") if x.strip()]
    unknown = [f for f in features if f not in BUCKETERS]
    if unknown:
        sys.exit(f"no bucketer for {unknown}; known: {sorted(BUCKETERS)}")

    prior = build(rows, features, a.min_count, a.smoothing)
    prior["source_manifest"] = str(path.relative_to(REPO))
    out = _resolve(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(prior, indent=2) + "\n", encoding="utf-8")

    print(f"▶ {prior['n_train']:,} training rows from {path.name}")
    for feature, table in prior["features"].items():
        print(f"\n  {feature}: {len(table)} buckets kept (>= {a.min_count} cases)")
        ranked = sorted(table.items(), key=lambda kv: -kv[1]["lift"].get("melanoma", 1))
        for bucket, cell in ranked[:4]:
            print(f"    {bucket:<20} n={cell['n']:>6,}  melanoma lift "
                  f"{cell['lift'].get('melanoma', 1):.2f}")
        if len(ranked) > 4:
            bucket, cell = ranked[-1]
            print(f"    {'... lowest: ' + bucket:<20} n={cell['n']:>6,}  melanoma lift "
                  f"{cell['lift'].get('melanoma', 1):.2f}")
    print(f"\n✓ wrote {out.relative_to(REPO)}")
    print("  A lift of 1.0 is neutral. Applied at decision time only — the model "
          "itself is unchanged.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Build nested training subsets for a learning curve.

Why this exists
---------------
Every experiment so far has asked "does this intervention help?". A learning
curve asks the question underneath them: **where does more data stop helping?**
If accuracy is already flat by a few thousand images, tripling the corpus was
never going to change it, and experiment 3's negative result stops being a
surprise and becomes a prediction.

It is also the question that decides whether to spend money. Labelling is the
expensive part of applied ML, and a curve answers "is another 10,000 images
worth it?" for the cost of some GPU time rather than an annotation budget.

Two properties that make the curve mean something
-------------------------------------------------
**Sampled by lesion, never by row.** The same lesion is photographed several
times, so drawing rows at random puts one lesion's photographs into both the
subset and the validation set. The small points would then look far better than
they are — the exact failure `build_splits.py` avoids for the main split, and it
bites hardest where the subsets are smallest.

**Nested.** Every subset is a superset of the one below it, so a point on the
curve differs from its neighbour only by *added* data. Drawing each size
independently would mix "more data" with "different data", and a dip could mean
either.

Stratified by diagnosis, so a 100-image subset is not simply 100 nevi.

Usage:
    python scripts/build_subsets.py --sizes 100,500,2000,7014
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

MANIFEST_FIELDS = ["filename", "image_id", "lesion_id", "label",
                   "dx_type", "age", "sex", "localization"]


def _resolve(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (REPO / path)


def read_manifest(path: Path) -> list[dict]:
    if not path.exists():
        sys.exit(f"not found: {path} — run scripts/build_isic_train.py first")
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def lesions_by_class(rows: list[dict]) -> dict[str, list[str]]:
    """{label: [lesion_id, ...]} — one entry per lesion, not per image.

    A lesion's diagnosis is taken from its first row; within these manifests a
    lesion carries a single diagnosis throughout.
    """
    seen: dict[str, str] = {}
    for r in rows:
        seen.setdefault(r["lesion_id"], r["label"])
    out: dict[str, list[str]] = defaultdict(list)
    for lesion, label in seen.items():
        out[label].append(lesion)
    return {k: sorted(v) for k, v in out.items()}


def nested_subsets(rows: list[dict], sizes: list[int], seed: int) -> dict[int, list[str]]:
    """Lesion ids per target size, each a superset of the smaller ones.

    Sizes are targets in *images*, not lesions: lesions are added class by class
    until the image count is reached, so the result lands near the target rather
    than exactly on it. Reporting the real count matters more than hitting a
    round number.
    """
    by_class = lesions_by_class(rows)
    images_per_lesion: dict[str, int] = defaultdict(int)
    for r in rows:
        images_per_lesion[r["lesion_id"]] += 1

    rng = random.Random(seed)
    order: dict[str, list[str]] = {}
    for label, lesions in by_class.items():
        shuffled = list(lesions)
        rng.shuffle(shuffled)                 # sorted upstream, so this is reproducible
        order[label] = shuffled

    # Class shares of the full corpus, so every subset keeps its shape.
    total_images = len(rows)
    share = {label: sum(images_per_lesion[l] for l in lesions) / total_images
             for label, lesions in by_class.items()}

    out: dict[int, list[str]] = {}
    for size in sorted(sizes):
        chosen: list[str] = []
        for label, lesions in order.items():
            want = max(1, round(size * share[label]))
            got = 0
            for lesion in lesions:
                if got >= want:
                    break
                chosen.append(lesion)
                got += images_per_lesion[lesion]
        out[size] = chosen
    # Nesting: because `order` is fixed per class and always consumed from the
    # front, a larger target takes everything a smaller one took and adds to it.
    for small, large in zip(sorted(sizes), sorted(sizes)[1:]):
        assert set(out[small]) <= set(out[large]), "subsets must nest"
    return out


def write_subset(rows: list[dict], lesions: set[str], path: Path) -> list[dict]:
    sel = [r for r in rows if r["lesion_id"] in lesions]
    sel.sort(key=lambda r: r["image_id"])          # deterministic
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS, lineterminator="\n")
        w.writeheader()
        for r in sel:
            w.writerow({k: r.get(k, "") for k in MANIFEST_FIELDS})
    return sel


def verify(subset_rows: list[dict], held_back: list[Path]) -> bool:
    """A subset inherits the parent's exclusions — but inherits is not verifies."""
    s_img = {r["image_id"] for r in subset_rows}
    s_les = {r["lesion_id"] for r in subset_rows}
    ok = True
    for p in held_back:
        with open(p, newline="", encoding="utf-8") as f:
            held = list(csv.DictReader(f))
        si = s_img & {r["image_id"] for r in held}
        sl = s_les & {r["lesion_id"] for r in held}
        ok &= not (si or sl)
        print(f"    [{'OK ' if not (si or sl) else 'LEAK'}] vs {p.name:<22} "
              f"{len(si)} shared images, {len(sl)} shared lesions")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifests/isic_train.csv")
    ap.add_argument("--sizes", default="100,500,2000,7014",
                    help="target image counts, comma separated")
    ap.add_argument("--prefix", default="isic")
    ap.add_argument("--out", default="data/manifests")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--held-back", nargs="*",
                    default=["data/manifests/ham10000_val.csv",
                             "data/manifests/ham10000_test.csv"])
    a = ap.parse_args()

    rows = read_manifest(_resolve(a.manifest))
    sizes = [int(x) for x in a.sizes.split(",")]
    held = [_resolve(p) for p in a.held_back]
    out_dir = _resolve(a.out)

    print(f"▶ {len(rows):,} images / {len({r['lesion_id'] for r in rows}):,} lesions "
          f"in {Path(a.manifest).name}")
    subsets = nested_subsets(rows, sizes, a.seed)

    clean = True
    for size in sorted(sizes):
        lesions = set(subsets[size])
        path = out_dir / f"{a.prefix}_n{size}_train.csv"
        sel = write_subset(rows, lesions, path)
        counts: dict[str, int] = defaultdict(int)
        for r in sel:
            counts[r["label"]] += 1
        mel = counts.get("melanoma", 0)
        print(f"\n  target {size:>6,}  ->  {len(sel):>6,} images / {len(lesions):>5,} lesions"
              f"   melanoma {mel:>5,}  -> {path.name}")
        clean &= verify(sel, held)

    # train_model.py reads `<prefix>_train.csv` AND `<prefix>_val.csv`, so every
    # subset prefix needs its own val file or training stops before it starts.
    # It is a byte copy, never a resampling: changing validation along with the
    # training set would score each point against a different bar, and the curve
    # would measure two things at once.
    val_src = out_dir / f"{a.prefix}_val.csv"
    if not val_src.exists():
        sys.exit(f"missing {val_src} — run scripts/build_isic_train.py first")
    val_bytes = val_src.read_bytes()
    for size in sorted(sizes):
        (out_dir / f"{a.prefix}_n{size}_val.csv").write_bytes(val_bytes)
    print(f"\n  validation copied unchanged to every subset prefix "
          f"({len(sizes)} x {val_src.name}, byte-identical)")
    if not clean:
        sys.exit("REFUSING: a subset overlaps held-back data.")
    print("✓ every subset is nested, stratified, and disjoint from val and test.")


if __name__ == "__main__":
    main()

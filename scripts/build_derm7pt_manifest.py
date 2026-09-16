#!/usr/bin/env python
"""Build an external evaluation manifest from Derm7pt [4].

Why this exists
---------------
Every number in this project so far was measured on HAM10000 — including the
models trained on ISIC, because the test split never moved. That answers "does
this configuration beat that one on these images" and cannot answer "does any of
it work somewhere else". Derm7pt is a different clinic, a different camera and a
different population, and no model here has ever trained on it.

The learning curve made this the obvious next question. It showed that
in-distribution data is worth roughly twice as much per image as
out-of-distribution data (+9.5 points at equal volume). The flip side is testable
only here: does a training corpus drawn from several archives buy robustness
against a *new* archive, or only against the ones it contains?

What this deliberately does not do
----------------------------------
It does not touch training. Once Derm7pt is the external test set, training on it
in any form destroys the only independent measurement available — and that
includes its seven-point concept annotations, tempting as they are for the
explainability pillar.

Mapping decisions, all of them judgement calls worth stating
------------------------------------------------------------
**Dermoscopic images only.** Each case ships a clinical photograph and a
dermoscopic one. These models were trained on dermoscopy; scoring them on
clinical photographs would measure a modality change and read as a domain gap.

**Twenty diagnoses onto seven classes.** 1,003 of 1,011 cases map cleanly; the
eight `miscellaneous` cases are dropped rather than forced into a class. All six
melanoma sub-labels (in situ, thickness bands, metastasis) collapse to
`melanoma`, which is what the models predict.

**`actinic_keratoses` does not occur here at all.** That is not an error to fix:
no two real populations share a class distribution, and the mismatch is part of
what an external test measures. It does mean the balanced accuracy is an average
over six classes rather than seven — `performance/classification.py` reports that
alongside the number so the two cannot be silently compared.

**Body site is translated, age is absent.** Derm7pt's vocabulary differs from
ISIC's (`back`, `acral`, `lower limbs` …) and is mapped to the ISIC terms so a
context prior could apply. `buttocks` has no clean equivalent and is left
untranslated, which makes it neutral rather than guessed. There is no age column
at all, so the age half of any context prior is inert here — worth knowing before
reading a prior-enabled run.

Licence [4]: CC BY-NC-ND 4.0, and the dataset's own README states the images may
not be redistributed. They stay in gitignored `data/raw/`, and no *derived* image
may be published either — so a Derm7pt scenario must leave `explainability.gradcam`
out of its metric list rather than render overlays and hide them.

Usage:
    python scripts/build_derm7pt_manifest.py
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

MANIFEST_FIELDS = ["filename", "image_id", "lesion_id", "label",
                   "dx_type", "age", "sex", "localization"]

# Derm7pt's twenty diagnoses -> the seven classes these models predict.
# Every melanoma variant collapses to `melanoma`: thickness and in-situ status
# are staging information the models were never asked to produce.
DIAGNOSIS_MAP = {
    "melanoma": "melanoma",
    "melanoma (in situ)": "melanoma",
    "melanoma (less than 0.76 mm)": "melanoma",
    "melanoma (0.76 to 1.5 mm)": "melanoma",
    "melanoma (more than 1.5 mm)": "melanoma",
    "melanoma metastasis": "melanoma",

    "clark nevus": "melanocytic_Nevi",
    "reed or spitz nevus": "melanocytic_Nevi",
    "dermal nevus": "melanocytic_Nevi",
    "blue nevus": "melanocytic_Nevi",
    "congenital nevus": "melanocytic_Nevi",
    "combined nevus": "melanocytic_Nevi",
    "recurrent nevus": "melanocytic_Nevi",

    "seborrheic keratosis": "benign_keratosis-like_lesions",
    "lentigo": "benign_keratosis-like_lesions",
    "melanosis": "benign_keratosis-like_lesions",

    "basal cell carcinoma": "basal_cell_carcinoma",
    "vascular lesion": "vascular_lesions",
    "dermatofibroma": "dermatofibroma",
    # "miscellaneous" is absent on purpose — see DROPPED below.
}
# Forcing these into a class would invent a label. Eight cases, dropped and counted.
DROPPED = {"miscellaneous"}

# Derm7pt body sites -> the ISIC vocabulary the context prior is keyed on.
# `buttocks` is left out: it is neither torso nor limb in ISIC's scheme, and a
# missing value is treated as no evidence, which is the honest answer.
SITE_MAP = {
    "back": "posterior torso",
    "chest": "anterior torso",
    "abdomen": "anterior torso",
    "upper limbs": "upper extremity",
    "lower limbs": "lower extremity",
    "head neck": "head/neck",
    "acral": "palms/soles",
    "genital areas": "oral/genital",
}


def _resolve(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (REPO / path)


def build(rows: list[dict]) -> tuple[list[dict], Counter]:
    out, dropped = [], Counter()
    for r in rows:
        diagnosis = (r.get("diagnosis") or "").strip().lower()
        label = DIAGNOSIS_MAP.get(diagnosis)
        if label is None:
            dropped[diagnosis or "(blank)"] += 1
            continue
        derm = (r.get("derm") or "").strip()
        if not derm:
            dropped["(no dermoscopic image)"] += 1
            continue
        out.append({
            "filename": derm,                      # e.g. "NEL/Nel026.jpg"
            "image_id": Path(derm).stem,
            # One dermoscopic image per case, so the case is the lesion. The
            # grouping key still has to exist: audit_split compares on it, and a
            # blank column would report as unverifiable rather than clean.
            "lesion_id": f"derm7pt_case_{r.get('case_num', '').strip()}",
            "label": label,
            "dx_type": "",                         # Derm7pt records no confirmation method
            "age": "",                             # no age column in this dataset
            "sex": (r.get("sex") or "").strip(),
            "localization": SITE_MAP.get((r.get("location") or "").strip().lower(), ""),
        })
    return out, dropped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default="data/raw/derm7pt_release_v0/meta/meta.csv")
    ap.add_argument("--images", default="data/raw/derm7pt_release_v0/images")
    ap.add_argument("--out", default="data/manifests/derm7pt_test.csv")
    ap.add_argument("--train-manifests", nargs="*",
                    default=["data/manifests/isic_train.csv",
                             "data/manifests/ham10000_train.csv"])
    a = ap.parse_args()

    meta = _resolve(a.meta)
    if not meta.exists():
        sys.exit(f"not found: {meta}\nDownload Derm7pt from http://derm.cs.sfu.ca/ "
                 f"(registration required) and unpack it under data/raw/.")

    with open(meta, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    built, dropped = build(rows)

    images = _resolve(a.images)
    missing = [r["filename"] for r in built if not (images / r["filename"]).exists()]
    if missing:
        sys.exit(f"{len(missing)} image(s) named by the metadata are absent, "
                 f"e.g. {missing[0]}")

    out = _resolve(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    built.sort(key=lambda r: r["image_id"])
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS, lineterminator="\n")
        w.writeheader()
        w.writerows(built)

    counts = Counter(r["label"] for r in built)
    print(f"▶ {len(rows):,} cases in {meta.name}  ->  {len(built):,} usable")
    for reason, n in dropped.most_common():
        print(f"  dropped {n:>4}  {reason}")
    print(f"\n  class distribution (evaluated as-is, no rebalancing):")
    for label, n in counts.most_common():
        print(f"    {label:<32} {n:>5}  ({100 * n / len(built):.1f}%)")

    # The models predict seven classes; this set contains six. That is a property
    # of a real external population, not a defect, but it has to be visible.
    seven = ["actinic_keratoses", "basal_cell_carcinoma", "benign_keratosis-like_lesions",
             "dermatofibroma", "melanocytic_Nevi", "melanoma", "vascular_lesions"]
    absent = [c for c in seven if c not in counts]
    if absent:
        print(f"\n  absent from this set: {', '.join(absent)}")
        print(f"    Balanced accuracy will average {len(seven) - len(absent)} classes, not "
              f"{len(seven)}, and is reported as such. Melanoma sensitivity and PPV do not")
        print(f"    depend on which other classes exist, which is why they stay comparable.")

    # Different archives use different identifier spaces, so this check can come
    # back unverifiable. Reporting that is the point: it is the difference between
    # "no overlap" and "no overlap that we could see".
    sys.path.insert(0, str(REPO))
    from verifai.core.integrity import audit_split
    print(f"\n▶ overlap against the training manifests")
    for m in a.train_manifests:
        p = _resolve(m)
        if not p.exists():
            print(f"  [skip] {p.name} not found")
            continue
        audit = audit_split(out, [p], group_key="lesion_id", id_key="image_id")
        flag = "OK " if audit.get("clean") else "LEAK"
        print(f"  [{flag}] vs {p.name:<24} {audit.get('shared_ids', 0)} shared images, "
              f"{audit.get('shared_groups', 0)} shared lesions"
              + ("" if audit.get("verifiable", True) else "   (UNVERIFIABLE)"))

    # What that check can and cannot establish. `ISIC_0024342` and `Nel026` come
    # from different archives with different naming schemes, so zero shared
    # identifiers is guaranteed by construction rather than measured. It rules
    # out the same *file* appearing twice; it cannot rule out the same physical
    # lesion having been photographed in both archives, which no identifier
    # comparison can see. That residual is stated rather than assumed away —
    # the same `verifiable` vs `clean` distinction audit_split draws elsewhere.
    print("    ^ these archives use disjoint identifier spaces, so zero overlap here is")
    print("      structural, not evidence. It rules out a shared file, not a shared lesion")
    print("      photographed in both — which no id comparison can detect. State it in the")
    print("      report rather than reading it as a verified clean split.")

    print(f"\n✓ wrote {out.relative_to(REPO)} — {len(built):,} images, "
          f"{counts.get('melanoma', 0)} melanoma")
    print("  Evaluation only. Training on this destroys the one independent measurement.")


if __name__ == "__main__":
    main()

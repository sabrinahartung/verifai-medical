"""Split-integrity auditing — is this evaluation worth believing at all?

HAM10000 photographs the same lesion repeatedly, so a split made on rows leaks
even when every image_id is unique. The dataset this project started from splits
that way: 80% of its test image_ids also appear in train. Every accuracy computed
on it was really a memorisation check.

This module is the single implementation of that check. `core/run.py` calls it as
a precondition (refusing to evaluate a contaminated split) and
`metrics/integrity/split_leakage.py` turns the same result into a published
finding — so the guard and the report can never disagree.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (REPO_ROOT / p)


def _read(manifest: str | Path, keys: tuple[str, ...]) -> list[dict[str, str]]:
    path = _resolve(manifest)
    if not path.exists():
        raise FileNotFoundError(f"manifest not found: {path}")
    with open(path, newline="", encoding="utf-8") as f:
        return [{k: (row.get(k) or "").strip() for k in keys} for row in csv.DictReader(f)]


def audit_split(test_manifest: str | Path,
                train_manifests: list[str | Path],
                group_key: str = "lesion_id",
                id_key: str = "image_id") -> dict[str, Any]:
    """Compare a test manifest against everything the model was trained on.

    Two kinds of contamination, in increasing severity:
      - `shared_groups`: the same lesion appears on both sides (a different photo
        of a lesion the model already learned),
      - `shared_ids`: literally the same image.

    Returns counts plus `contamination` — the share of test rows touched by either.
    """
    test = _read(test_manifest, (id_key, group_key))
    train: list[dict[str, str]] = []
    for m in train_manifests:
        train.extend(_read(m, (id_key, group_key)))

    train_groups = {r[group_key] for r in train if r[group_key]}
    train_ids = {r[id_key] for r in train if r[id_key]}

    shared_groups = sorted({r[group_key] for r in test
                            if r[group_key] and r[group_key] in train_groups})
    shared_ids = sorted({r[id_key] for r in test
                         if r[id_key] and r[id_key] in train_ids})
    affected = [r for r in test
                if (r[group_key] and r[group_key] in train_groups)
                or (r[id_key] and r[id_key] in train_ids)]

    n = len(test)
    has_groups = any(r[group_key] for r in test) and any(r[group_key] for r in train)
    has_ids = any(r[id_key] for r in test) and any(r[id_key] for r in train)
    # Without a usable identifier on both sides nothing was actually compared.
    # Reporting that as "clean" would publish an unverified split as a verified
    # one — the precise failure this pillar exists to catch.
    verifiable = has_groups or has_ids

    return {
        "n_test": n,
        "n_train": len(train),
        "group_key": group_key if has_groups else None,
        "id_key": id_key if has_ids else None,
        "shared_groups": len(shared_groups),
        "shared_ids": len(shared_ids),
        "affected_rows": len(affected),
        "contamination": round(len(affected) / n, 4) if n else 0.0,
        "verifiable": verifiable,
        "clean": (not shared_groups and not shared_ids) if verifiable else None,
        # a few examples, so a failure is actionable rather than just a number
        "example_shared_groups": shared_groups[:5],
        "example_shared_ids": shared_ids[:5],
    }


def train_manifests_from_scenario(scenario: dict[str, Any]) -> list[str]:
    """Where to look for 'what the model was trained on', by convention.

    Explicit `integrity.train_manifests` wins; otherwise derive from the
    `training:` block, which already names the manifest prefix and directory.
    """
    integ = scenario.get("integrity") or {}
    if integ.get("train_manifests"):
        return list(integ["train_manifests"])
    tcfg = scenario.get("training") or {}
    prefix = tcfg.get("manifest_prefix")
    if not prefix:
        return []
    d = tcfg.get("manifest_dir", "data/manifests")
    # validation counts as seen: the model was selected on it
    return [f"{d}/{prefix}_train.csv", f"{d}/{prefix}_val.csv"]


# --- corpus ancestry -----------------------------------------------------------
# The row-level check above needs both sides' manifests. For a model this project
# did not train there are none; what is left is whether its declared training
# *corpus* could contain the evaluation images at all. `data/corpora.yaml` holds
# the documented containment, and nothing outside it is assumed independent.
CORPORA_FILE = REPO_ROOT / "data" / "corpora.yaml"


def load_corpora(path: Path = CORPORA_FILE) -> dict[str, dict[str, Any]]:
    import yaml
    return (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("corpora") or {}


def descendants(corpus: str, corpora: dict[str, dict[str, Any]]) -> set[str]:
    """The corpus itself and everything distributed as part of it, transitively."""
    out, todo = set(), [corpus]
    while todo:
        c = todo.pop()
        if c in out:
            continue
        out.add(c)
        todo.extend((corpora.get(c) or {}).get("contains") or [])
    return out


def shared_corpora(trained_on: list[str], evaluated_on: str,
                   corpora: dict[str, dict[str, Any]]) -> list[str]:
    """Training corpora that could hold the evaluation images, or be held by them.

    Related means one contains the other, in either direction: a model trained on
    ISIC 2019 has seen HAM10000 images, and a model trained on HAM10000 has seen
    part of an ISIC 2019 test set.
    """
    ev = descendants(evaluated_on, corpora)
    return sorted(t for t in trained_on
                  if evaluated_on in descendants(t, corpora) or t in ev)


def declared_training(scenario: dict[str, Any]) -> dict[str, Any]:
    """What the scenario says the model was trained on: corpora, and on what basis.

    `model.trained_on` is either a list of corpus ids or
    `{corpora: [...], basis: "why this is believed"}`. The basis matters for a
    third-party model, whose training data is often only inferred.
    """
    raw = (scenario.get("model") or {}).get("trained_on")
    if isinstance(raw, dict):
        return {"corpora": list(raw.get("corpora") or []), "basis": raw.get("basis")}
    return {"corpora": list(raw or []), "basis": None}


def row_check(scenario: dict[str, Any], test_manifest: str | None) -> dict[str, Any] | None:
    """The row-level audit if it can run for this scenario, else None."""
    train = train_manifests_from_scenario(scenario)
    if not test_manifest or not train:
        return None
    cfg = scenario.get("integrity") or {}
    a = audit_split(test_manifest, train, group_key=cfg.get("group_key", "lesion_id"),
                    id_key=cfg.get("id_key", "image_id"))
    return a if a["verifiable"] else None


# --- label space ----------------------------------------------------------------
def label_space(model_classes: list[str], dataset_classes: list[str]) -> dict[str, Any]:
    """How the model's output classes relate to the classes the data is labelled with.

    identical        the same set
    dataset_subset   the data lacks some of the model's classes; those go unscored
    model_subset     the data holds classes the model cannot output; those images
                     can only ever be wrong
    partial          each has classes the other lacks
    disjoint         nothing in common — needs an explicit `label_map`, never a guess
    """
    m, d = set(model_classes), set(dataset_classes)
    if m == d:
        relation = "identical"
    elif not m & d:
        relation = "disjoint"
    elif d < m:
        relation = "dataset_subset"
    elif m < d:
        relation = "model_subset"
    else:
        relation = "partial"
    return {"relation": relation,
            "n_model_classes": len(m), "n_dataset_classes": len(d),
            "unscored": sorted(m - d),             # the model predicts, the data never holds
            "unpredictable": sorted(d - m)}        # the data holds, the model cannot output

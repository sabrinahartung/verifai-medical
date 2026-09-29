"""Integrity: where the model came from, and what that lets this report check.

Signature: run(model, dataset, ctx) -> Finding

For a model trained in this repository the training record sits beside the
weights and the training manifests are on disk, so the split can be audited row
by row. For a checkpoint someone else trained there is no manifest, and the
strongest check this project has cannot run. That is not a gap to hide — it is
the finding. A third-party model is never reported as having a clean split; it
is reported as having a split nobody can check.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from verifai.core.findings import Finding
from verifai.core.integrity import (REPO_ROOT, declared_training, load_corpora,
                                    train_manifests_from_scenario)

EXPLAIN = {
    "what": ("Every other number in this report assumes the model never saw these images. "
             "Whether that can be checked depends on where the model came from: a model "
             "trained here comes with the list of images it trained on, so the overlap can "
             "be counted. A model downloaded from someone else usually comes with no such "
             "list, and then the overlap cannot be counted at all."),
    "how": ("Read the source first — trained here, or downloaded — and then whether a "
            "training record was found. With a record, the split check above is the real "
            "answer. Without one, the corpus check beside it is the only leakage check left, "
            "and it can only say whether leakage is possible, never that it did not happen."),
    "limits": ("A training record says what its author declared. It cannot prove that "
               "nothing else was used — a model fine-tuned from another checkpoint inherits "
               "whatever that checkpoint saw, and no record here would show it."),
}


def _source(model_cfg: dict[str, Any]) -> tuple[str, str]:
    """(kind, a sentence naming where the weights came from)."""
    path = model_cfg.get("weights_path")
    if path and (Path(path) if Path(path).is_absolute() else REPO_ROOT / path).is_file():
        return "local", f"the local checkpoint `{Path(path).name}`"
    if model_cfg.get("repo_id"):
        rev = model_cfg.get("revision")
        where = f"`{model_cfg['repo_id']}`"
        return "hub", (f"the Hugging Face repository {where} at revision `{rev}`" if rev else
                       f"the Hugging Face repository {where}, revision unpinned")
    return "unknown", "an undeclared source"


def _training_record(model_cfg: dict[str, Any]) -> dict[str, Any] | None:
    path = model_cfg.get("weights_path")
    if not path:
        return None
    p = Path(path) if Path(path).is_absolute() else REPO_ROOT / path
    record = p.with_name(p.stem + "_training.json")
    return json.loads(record.read_text(encoding="utf-8")) if record.is_file() else None


def run(model, dataset, ctx: dict[str, Any]) -> Finding:
    scenario = ctx.get("scenario", {}) or {}
    model_cfg = scenario.get("model") or {}
    domain = scenario.get("domain", "image")
    n = len(dataset)
    kind, where = _source(model_cfg)
    record = _training_record(model_cfg)
    manifests = train_manifests_from_scenario(scenario)
    on_disk = [m for m in manifests if (REPO_ROOT / m).is_file() or Path(m).is_file()]
    declared = declared_training(scenario)
    names = [(load_corpora().get(c) or {}).get("name", c) for c in declared["corpora"]]

    value = {"source": kind, "training_record": record is not None,
             "training_manifests": len(on_disk), "trained_on": declared["corpora"],
             "trained_on_basis": declared["basis"]}
    checkable = bool(on_disk) and len(on_disk) == len(manifests)
    value["checkable"] = checkable

    if checkable:
        by = f" by scenario `{record['scenario']}`" if record and record.get("scenario") else ""
        summary = (f"Trained in this repository{by}, loaded from {where}. Its "
                   f"{len(on_disk)} training manifests are on disk, so every one of the "
                   f"{n:,} test images can be checked against what the model saw.")
        verdict = "measured"
    else:
        what = (f"Its training data is described as {', '.join(names)}"
                + (f" — {declared['basis']}" if declared["basis"] else "") + "."
                if names else "Its training data is not described at all.")
        summary = (f"Loaded from {where}, with no training manifests declared, so none of the "
                   f"{n:,} test images can be checked against what the model saw. {what} "
                   f"No split check is possible; that is not the same as a clean split.")
        verdict = "unavailable"

    return Finding(pillar="integrity", metric="provenance", domain=domain, value=value,
                   verdict=verdict, summary=summary, details={"explain": EXPLAIN})

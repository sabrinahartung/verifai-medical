"""Models and projects, read from the committed registry — status derived here.

The engine writes `artifacts/model_registry.json` from the scenarios (see
`verifai/export/model_registry.py`); this module only reads it. Whether a
configuration has been evaluated is *not* in that file. It is decided here, from
which artifact folders exist, so the registry never goes stale on a re-run and
an evaluation cannot claim to exist without its report.

Pure data, no Streamlit call, like `catalog.py`: the contract tests import it.

Artifacts that no registered configuration claims — the demo fixture, or an
evaluation whose scenario was since deleted — are returned separately rather
than dropped. Old shapes keep rendering; that is the rule the verdict
vocabulary already follows.
"""
from __future__ import annotations

import json

from catalog import ART

REGISTRY_FILE = ART / "model_registry.json"


def load_registry() -> dict | None:
    """The registry, or None when this deploy predates it."""
    if not REGISTRY_FILE.is_file():
        return None
    return json.loads(REGISTRY_FILE.read_text(encoding="utf-8"))


def evaluated_ids() -> set[str]:
    """Scenario ids with a complete artifact — the same test `load_catalog` applies."""
    if not ART.exists():
        return set()
    return {d.name for d in ART.iterdir()
            if (d / "card.json").exists() and (d / "report.json").exists()}


def model_status(model: dict, evaluated: set[str]) -> dict:
    """How many of a model's configurations have a report.

    Three states, stated as counts rather than a colour: *evaluated* means every
    declared configuration has a report, *partly* that some do, *not evaluated*
    that none do. It says what exists, not whether any of it is good.
    """
    total = len(model["configurations"])
    done = sum(1 for c in model["configurations"] if c["scenario"] in evaluated)
    state = "evaluated" if done == total else "partly" if done else "not_evaluated"
    return {"evaluated": done, "total": total, "state": state}


STATUS_LABEL = {
    "evaluated": "Evaluated",
    "partly": "Partly evaluated",
    "not_evaluated": "Not evaluated",
}


def find_model(registry: dict | None, key: str | None) -> dict | None:
    if not registry or not key:
        return None
    return next((m for m in registry["models"] if m["key"] == key), None)


def model_of_scenario(registry: dict | None, scenario: str) -> dict | None:
    """The model a configuration belongs to — the report's way back up."""
    if not registry:
        return None
    return next((m for m in registry["models"]
                 if any(c["scenario"] == scenario for c in m["configurations"])), None)


def models_in(registry: dict, project: str) -> list[dict]:
    keys = next((p["models"] for p in registry["projects"] if p["name"] == project), [])
    by_key = {m["key"]: m for m in registry["models"]}
    return [by_key[k] for k in keys if k in by_key]


def unclaimed(cards: list[dict], registry: dict | None) -> list[dict]:
    """Evaluations no registered configuration accounts for."""
    claimed = {c["scenario"] for m in (registry or {}).get("models", [])
               for c in m["configurations"]}
    return [c for c in cards if c["id"] not in claimed]


def describe(provenance: dict | None) -> str:
    """One line on what a checkpoint *is* — architecture, data, and how it was trained.

    Built from the trainer's own record, so it cannot drift from the weights.
    Empty when the model was not trained here, which the caller must say
    rather than leave blank.
    """
    if not provenance:
        return ""
    bits = [str(provenance.get("arch", "")).replace("resnet", "ResNet")]
    train = (provenance.get("manifests") or {}).get("train")
    if train:
        corpus = train.rsplit("/", 1)[-1].removesuffix(".csv").removesuffix("_train")
        n = provenance.get("train_images")
        bits.append(f"trained on {corpus}" + (f" ({n:,} images)" if n else ""))
    if provenance.get("freeze_backbone"):
        bits.append("linear probe — backbone frozen")
    if provenance.get("loss") and provenance["loss"] != "ce":
        g = provenance.get("focal_gamma")
        bits.append(f"{provenance['loss']} loss" + (f" (γ={g:g})" if g is not None else ""))
    if provenance.get("sampling") not in (None, "none"):
        bits.append(f"{provenance['sampling']} sampling")
    if provenance.get("class_weights"):
        bits.append("class-weighted")
    return " · ".join(b for b in bits if b)

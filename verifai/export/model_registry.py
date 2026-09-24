"""Which models exist, and which configurations of each — the model registry.

The showcase discovers evaluations by listing artifact folders, so a model that
has never been evaluated is invisible to it, and the gallery calls every
evaluation a model. The scenarios are the one place that knows about every
model whether or not it has run, so this module reads them and writes a single
committed JSON file the showcase can load. The showcase keeps its contract of
reading only precomputed artifacts; it never parses YAML or opens a checkpoint.

Three facts decide the shape, all read off the repository rather than assumed:

* **A model is a checkpoint, identified by its content hash.** `model.id` cannot
  do it: `skin-lesion-resnet18-clean` names three different checkpoints, and
  `skin_cancer_isic.pt` answers to five different ids. A name can be reused and a
  file can be regenerated; bytes cannot. Hashed the way `core/run.py` hashes an
  evaluation manifest, so both identities read alike.
* **The scenario whose `name` is the checkpoint's filename trained it.**
  `scripts/train_model.py` writes `<out_dir>/<name>.pt`. Every other scenario
  pointing at that file is a *configuration* of it — the same weights, read with
  a different decision rule or scored on a different set — even when it carries
  a copied `training:` block, as several do.
* **Evaluation status is not stored here.** It is derived by the showcase from
  which artifact folders exist, so this file changes only when a scenario or a
  checkpoint does, and re-running an evaluation never produces a diff in it.

Output is deterministic — sorted, no timestamp — so rebuilding an unchanged
repository is a no-op in git, and a contract test can rebuild it and demand
equality.

Checkpoints and training records are gitignored, so on a machine without them
the builder carries the previously recorded hash and provenance forward for the
same path rather than erasing them. A checkpoint that was never hashed anywhere
falls back to its path, and one with no declared weights at all is reported as
unidentified — never given a synthetic identity, which would let a comparison
treat two different models as one.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from verifai.core.suite import METRIC_VERSIONS

REGISTRY_PATH = Path("showcase/artifacts/model_registry.json")
SCHEMA = 1
UNASSIGNED = "Unassigned"
STATUSES = ("active", "archived")

# Copied from `<name>_training.json`. What a checkpoint *is*, never how well it
# did: `best_val_balanced_accuracy` and the per-epoch `history` are left out on
# purpose. They are validation numbers without an interval, and on a model page
# they would read as the model's score, competing with the test-set report that
# carries one.
PROVENANCE_KEYS = (
    "arch", "classes", "seed", "epochs", "image_size", "class_weights", "loss",
    "focal_gamma", "sampling", "freeze_backbone", "manifests", "train_images",
    "val_images", "note",
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _checkpoint(model: dict) -> dict:
    """Where a scenario's weights come from, as declared."""
    if model.get("weights_path"):
        return {"kind": "local", "path": model["weights_path"]}
    if model.get("repo_id"):
        return {"kind": "hub", "repo_id": model["repo_id"],
                "filename": model.get("filename"), "revision": model.get("revision")}
    return {"kind": "undeclared"}


def _identity(ck: dict, scenario: str, root: Path, known: dict[str, str]) -> tuple[str, str | None]:
    """(identity, sha256) — the key configurations are grouped by."""
    if ck["kind"] == "local":
        path = root / ck["path"]
        digest = _sha256(path) if path.is_file() else known.get(ck["path"])
        return (f"sha256:{digest}", digest) if digest else (f"path:{ck['path']}", None)
    if ck["kind"] == "hub":
        rev = ck.get("revision") or "unpinned"
        return f"hub:{ck['repo_id']}/{ck.get('filename')}@{rev}", None
    return f"unidentified:{scenario}", None


def _key(ck: dict, scenario: str) -> str:
    """A readable, stable id for URLs — the checkpoint's filename where there is one."""
    if ck["kind"] == "local":
        return Path(ck["path"]).stem
    if ck["kind"] == "hub":
        return Path(ck.get("filename") or ck["repo_id"]).stem
    return scenario


def _provenance(ck: dict, root: Path, previous: dict | None) -> dict | None:
    if ck["kind"] == "local":
        record = root / Path(ck["path"]).with_name(Path(ck["path"]).stem + "_training.json")
        if record.is_file():
            data = json.loads(record.read_text(encoding="utf-8"))
            return {k: data[k] for k in PROVENANCE_KEYS if k in data}
    return (previous or {}).get("provenance")


def build_registry(scenarios_dir: str | Path = "scenarios", root: str | Path = ".",
                   previous: dict | None = None) -> dict:
    """Every declared model, its provenance and its configurations.

    `previous` is an earlier registry, used only to carry hashes and provenance
    forward when the checkpoint files are not on this machine.
    """
    root = Path(root)
    prev_models = {m["key"]: m for m in (previous or {}).get("models", [])}
    known_hashes = {m["checkpoint"]["path"]: m["sha256"]
                    for m in prev_models.values()
                    if m.get("sha256") and m["checkpoint"].get("kind") == "local"}

    groups: dict[str, dict[str, Any]] = {}
    for p in sorted(Path(scenarios_dir).glob("*.yaml")):
        sc = yaml.safe_load(p.read_text(encoding="utf-8"))
        name, model = sc["name"], sc.get("model") or {}
        ck = _checkpoint(model)
        identity, digest = _identity(ck, name, root, known_hashes)
        g = groups.setdefault(identity, {
            "identity": identity, "sha256": digest, "checkpoint": ck,
            "key": _key(ck, name), "configurations": [], "_projects": {}, "_trainer": None,
        })
        if ck["kind"] == "local" and Path(ck["path"]).stem == name and sc.get("training"):
            g["_trainer"] = sc
        g["_projects"][name] = sc.get("project") or UNASSIGNED
        status = sc.get("status", "archived")
        if status not in STATUSES:
            raise ValueError(f"{name}: status must be one of {STATUSES}, not {status!r}")
        g["configurations"].append({
            "scenario": name,
            "label": sc.get("label") or name,
            # Active configurations are re-run when a metric changes; archived
            # ones are kept as the record of what was measured, when.
            "status": status,
            "decision_weights": model.get("decision_weights") or None,
            "eval_manifest": (sc.get("dataset") or {}).get("manifest"),
            "dataset_id": (sc.get("dataset") or {}).get("id"),
            "metrics": list(sc.get("metrics") or []),
        })

    models = []
    for g in groups.values():
        projects = set(g.pop("_projects").values())
        if len(projects) > 1:
            raise ValueError(
                f"checkpoint {g['key']} is declared under {len(projects)} projects "
                f"({', '.join(sorted(projects))}). A model belongs to one project; "
                f"its configurations must agree.")
        trainer = g.pop("_trainer")
        g["project"] = projects.pop()
        g["trained_by"] = trainer["name"] if trainer else None
        # The trainer's label names the checkpoint; failing that, the only or
        # first configuration's, which is what the gallery called it before.
        g["name"] = (trainer.get("label") if trainer else None) or g["configurations"][0]["label"]
        g["provenance"] = _provenance(g["checkpoint"], root, prev_models.get(g["key"]))
        g["identified"] = not g["identity"].startswith("unidentified:")
        g["active"] = any(c["status"] == "active" for c in g["configurations"])
        models.append(g)

    # Two checkpoints can share a filename in different folders; the key must
    # still be unique, so the loser of a collision is qualified by its hash.
    seen: dict[str, int] = {}
    for m in sorted(models, key=lambda m: m["identity"]):
        seen[m["key"]] = seen.get(m["key"], 0) + 1
        if seen[m["key"]] > 1:
            m["key"] = f"{m['key']}-{(m['sha256'] or m['identity'])[-6:]}"

    models.sort(key=lambda m: (m["project"], m["key"]))
    projects: dict[str, list[str]] = {}
    for m in models:
        projects.setdefault(m["project"], []).append(m["key"])
    return {
        "schema": SCHEMA,
        # The current version of every metric, so the showcase — which never
        # imports the engine — can tell a report produced by an older one.
        "metric_versions": dict(sorted(METRIC_VERSIONS.items())),
        "projects": [{"name": n, "models": ks} for n, ks in sorted(projects.items())],
        "models": models,
    }


def write_registry(scenarios_dir: str | Path = "scenarios", root: str | Path = ".",
                   path: str | Path = REGISTRY_PATH) -> Path:
    """Rebuild the registry file in place, carrying forward what this machine lacks."""
    path = Path(root) / path
    previous = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
    registry = build_registry(scenarios_dir=Path(root) / scenarios_dir, root=root,
                              previous=previous)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path

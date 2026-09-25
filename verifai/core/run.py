"""Scenario runner + tiny registry.

A scenario (YAML) declares: domain, model, dataset, sample_size, seed, and which
metrics (pillars) to run. `run_scenario` builds the pieces, runs each metric, and
collects the results into a single `Report`.

Design goals: no web server, no DB — pure functions in, `Report` out.
"""
from __future__ import annotations

import importlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[2]

from verifai.core.findings import Report, Finding
from verifai.core.integrity import audit_split, train_manifests_from_scenario
from verifai.core.suite import suite_for
from verifai.metrics._baseline import attach as attach_baseline
from verifai.models.base import DEFAULT_TASK, WHY_UNREACHABLE, reaches


class SplitLeakageError(RuntimeError):
    """Raised instead of producing a flattering number from a leaking split."""

@dataclass(frozen=True)
class MetricSpec:
    """What a metric is, and what it needs before it can run.

    `requires` is the lowest access level the metric needs (see
    `verifai.models.base.ACCESS_LEVELS`); `modalities` of `None` means any.
    `pillar` and `finding` name the finding it produces, so a metric that cannot
    run still gets a row — in its own pillar, under its own name — saying why.
    """
    target: str                                  # "module_path:function_name"
    pillar: str
    finding: str
    tasks: tuple[str, ...] = ("classification",)
    modalities: tuple[str, ...] | None = None
    requires: str = "labels"


# metric id -> MetricSpec. A plain "module:function" string is still accepted
# (see `spec_of`), so a registry entry written before the contracts keeps working.
# Each metric fn has signature: fn(model, dataset, ctx: dict) -> Finding | list[Finding]
METRIC_REGISTRY: dict[str, MetricSpec | str] = {
    # needs the manifests, not the model at all
    "integrity.split_leakage": MetricSpec(
        "verifai.metrics.integrity.split_leakage:run", "integrity", "split_leakage"),
    "performance.classification": MetricSpec(
        "verifai.metrics.performance.classification:run", "performance", "top1_accuracy",
        requires="probs"),
    "explainability.gradcam": MetricSpec(
        "verifai.metrics.explainability.gradcam:run", "explainability", "gradcam_faithfulness",
        modalities=("pixels",), requires="gradients"),
    "robustness.corruption": MetricSpec(
        "verifai.metrics.robustness.corruption:run", "robustness", "corruption_stability",
        modalities=("pixels",), requires="probs"),
    "fairness.skin_tone": MetricSpec(
        "verifai.metrics.fairness.skin_tone_ita:run", "fairness", "skin_tone_ita",
        modalities=("pixels",), requires="probs"),
    # the attacker's signal is confidence, but telling members from non-members
    # needs to know who the members were
    "privacy.mia": MetricSpec(
        "verifai.metrics.privacy.mia:run", "privacy", "membership_inference_auc",
        requires="training_data"),
}


def spec_of(metric_id: str) -> MetricSpec:
    """The registry entry as a MetricSpec, whichever form it was written in.

    A bare string declares nothing, so it is taken at its most permissive —
    every task, every modality, the lowest access — which is exactly how every
    metric ran before the contracts existed.
    """
    entry = METRIC_REGISTRY[metric_id]
    if isinstance(entry, MetricSpec):
        return entry
    return MetricSpec(entry, pillar=metric_id.split(".", 1)[0], finding=metric_id,
                      tasks=(), modalities=None, requires="labels")


def _load(target: str | MetricSpec) -> Callable:
    if isinstance(target, MetricSpec):
        target = target.target
    module_path, func = target.split(":")
    return getattr(importlib.import_module(module_path), func)


def applies(spec: MetricSpec, task: str, modality: str | None) -> bool:
    """Whether a metric belongs in an evaluation of this task and payload at all.

    Not the same question as whether it *can run*: a metric that applies but
    needs more access than the model gives is unavailable, and says why; one
    that does not apply is left out of the denominator entirely.
    """
    task_ok = not spec.tasks or task in spec.tasks
    modality_ok = spec.modalities is None or modality in spec.modalities
    return task_ok and modality_ok


def model_access(model, scenario: dict[str, Any]) -> str:
    """How much of the model this evaluation can reach.

    The model declares what it exposes. Whether its training data is known is
    a fact about the scenario, so a model that can be opened (`weights`) is
    raised to `training_data` when training manifests are declared. A model
    that declares nothing is taken at the lowest level rather than trusted.
    """
    access = getattr(model, "access", None) or "labels"
    if reaches(access, "weights") and train_manifests_from_scenario(scenario):
        return "training_data"
    return access


def unavailable_finding(spec: MetricSpec, access: str, domain: str) -> Finding:
    """The row a metric gets when the model cannot support it — never a silent gap."""
    why = WHY_UNREACHABLE.get(spec.requires, f"it needs access level {spec.requires!r}")
    return Finding(
        pillar=spec.pillar, metric=spec.finding, domain=domain, value=None,
        verdict="unavailable",
        summary=(f"Not run: {why}. This model's access level is {access}; the metric "
                 f"needs {spec.requires}. This says what could be measured, not how "
                 f"the model behaves."),
        details={"requires": spec.requires, "access": access},
    )


def _build_model(spec: dict[str, Any]):
    # spec: {domain, loader: "verifai.models.image:load", ...}
    return _load(spec["loader"])(spec)


def _build_dataset(spec: dict[str, Any]):
    return _load(spec["loader"])(spec)


def _eval_set_fingerprint(dataset) -> dict[str, Any]:
    """Identify *what was evaluated on*, precisely enough to refuse bad comparisons.

    Two runs are only comparable if they were scored on the same rows. A path is
    not enough — a manifest can be regenerated with a different seed and keep its
    name — so the file's content hash is what actually decides.
    """
    import hashlib
    meta = getattr(dataset, "meta", None) or {}
    manifest = meta.get("manifest")
    digest = None
    if manifest:
        path = Path(manifest)
        if not path.is_absolute():
            path = REPO_ROOT / path
        if path.exists():
            digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    return {"manifest": str(manifest) if manifest else None,
            "sha256": digest,
            "n": _safe_len(dataset)}


def _safe_len(dataset) -> int | None:
    try:
        return len(dataset)
    except TypeError:
        return None


def _enforce_split_integrity(scenario: dict[str, Any], dataset) -> None:
    """Refuse to evaluate a test set the model was trained on.

    The failure mode this exists for is silent: a contaminated split does not
    crash, it just reports a high number. Set `integrity.enforce: false` to
    downgrade this to a reported finding instead of a hard stop.
    """
    cfg = scenario.get("integrity") or {}
    if not cfg.get("enforce", True):
        return
    test_manifest = (getattr(dataset, "meta", None) or {}).get("manifest")
    train_manifests = train_manifests_from_scenario(scenario)
    if not test_manifest or not train_manifests:
        return                      # nothing declared to check against; the metric says so
    a = audit_split(test_manifest, train_manifests,
                    group_key=cfg.get("group_key", "lesion_id"),
                    id_key=cfg.get("id_key", "image_id"))
    if not a["verifiable"]:
        return                      # nothing comparable in the manifests; the metric says so
    if not a["clean"]:
        raise SplitLeakageError(
            f"{a['affected_rows']} of {a['n_test']} test images ({a['contamination']*100:.1f}%) "
            f"were seen during training: {a['shared_ids']} identical images, "
            f"{a['shared_groups']} shared lesions (e.g. {a['example_shared_groups'][:3]}). "
            f"Refusing to evaluate — the result would be a memorisation check. "
            f"Rebuild the split with scripts/build_splits.py, or set integrity.enforce: false "
            f"to report it as a finding instead."
        )


def _checkpoint_fingerprint(model_cfg: dict) -> dict:
    """The weights a report was scored with, hashed as the model registry hashes them."""
    path = model_cfg.get("weights_path")
    if path and Path(path).is_file():
        from verifai.export.model_registry import _sha256
        return {"path": path, "sha256": _sha256(Path(path))}
    if model_cfg.get("repo_id"):
        return {"repo_id": model_cfg["repo_id"], "filename": model_cfg.get("filename"),
                "revision": model_cfg.get("revision")}
    return {}


def run_scenario(scenario: dict[str, Any]) -> Report:
    seed = scenario.get("seed", 42)
    random.seed(seed)
    try:
        import numpy as np; np.random.seed(seed)
        import torch; torch.manual_seed(seed)
    except Exception:
        pass

    task = scenario.get("task") or DEFAULT_TASK
    unknown = [m for m in scenario["metrics"] if m not in METRIC_REGISTRY]
    if unknown:
        raise KeyError(f"unknown metric(s) {unknown}; registered: {sorted(METRIC_REGISTRY)}")

    model = _build_model(scenario["model"])
    dataset = _build_dataset(scenario["dataset"])
    modality = getattr(model, "modality", None)
    # A metric that does not apply to this task or payload is a scenario error,
    # caught before any metric runs rather than reported as a finding: it was
    # never part of what this evaluation could mean.
    misplaced = [m for m in scenario["metrics"] if not applies(spec_of(m), task, modality)]
    if misplaced:
        raise ValueError(f"metric(s) {misplaced} do not apply to task {task!r} on "
                         f"{modality!r} payloads; remove them from the scenario")
    access = model_access(model, scenario)
    _enforce_split_integrity(scenario, dataset)

    report = Report(
        scenario=scenario["name"],
        domain=scenario["domain"],
        model_id=scenario["model"].get("id", "unknown"),
        dataset_id=scenario["dataset"].get("id", "unknown"),
        meta={"seed": seed,
              # what was actually evaluated, not merely what the YAML asked for
              "sample_size": scenario.get("sample_size") or _safe_len(dataset),
              "device": str(getattr(model, "device", "cpu")),
              "eval_set": _eval_set_fingerprint(dataset),
              "label": scenario.get("label") or scenario["model"].get("id", scenario["name"]),
              # Which metric versions and which weights produced this report —
              # what lets the showcase tell a current report from one produced
              # before a metric changed or the checkpoint was retrained.
              "metric_versions": suite_for(scenario["metrics"]),
              "checkpoint": _checkpoint_fingerprint(scenario["model"]),
              # What this evaluation could reach, and what it covered. Kept
              # beside the findings so the showcase can tell "not applicable to
              # this task" from "not requested" from "unavailable, because …".
              "task": task, "modality": modality, "access": access},
    )

    ctx = {"scenario": scenario, "seed": seed, "plot_dir": scenario.get("_plot_dir", "plots")}
    outcome: dict[str, list[str]] = {}
    for metric_id in scenario["metrics"]:
        spec = spec_of(metric_id)
        # Checked before the call, so a metric never has to discover mid-run
        # that the model cannot give it what it needs.
        if not reaches(access, spec.requires):
            result = unavailable_finding(spec, access, scenario["domain"])
        else:
            result = _load(spec)(model, dataset, ctx)
        for f in (result if isinstance(result, list) else [result]):
            attach_baseline(f)
            report.add(f)
            outcome.setdefault(metric_id, []).append(f.verdict)
    report.meta["coverage"] = coverage(task, modality, access, scenario["metrics"], outcome)
    return report


def coverage(task: str, modality: str | None, access: str, requested: list[str],
             outcome: dict[str, list[str]]) -> list[dict[str, Any]]:
    """Every registered metric, and where it stands in this evaluation.

    A completeness statement, never a quality one: it counts what was measured
    and what could not be, and why — never what passed. `status` is one of
    `not_applicable` (wrong task or payload), `not_requested` (applies, but this
    scenario did not ask for it), or the verdict(s) its findings came back with.
    """
    rows = []
    for metric_id in METRIC_REGISTRY:
        spec = spec_of(metric_id)
        if not applies(spec, task, modality):
            status = "not_applicable"
        elif metric_id not in requested:
            status = "not_requested"
        else:
            verdicts = outcome.get(metric_id) or ["unavailable"]
            status = verdicts[0] if len(set(verdicts)) == 1 else "mixed"
        rows.append({"metric": metric_id, "pillar": spec.pillar, "finding": spec.finding,
                     "requires": spec.requires, "status": status,
                     "reachable": reaches(access, spec.requires)})
    return rows

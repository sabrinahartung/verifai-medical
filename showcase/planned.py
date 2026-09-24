"""Components that are planned and not built — the skeleton's contents.

Pure data, no imports beyond the standard library, for the same reason
`verifai/core/glossary.py` is: the showcase must stay torch-free and must not
grow a dependency to describe something it does not yet do.

Each entry answers the three questions a placeholder has to answer to be worth
drawing at all: what would be here, what is stopping it, and where the plan for
it lives. `phase` is a path into `docs/`, and a contract test asserts every one
of them resolves to a real heading — a placeholder pointing at a plan that no
longer exists is worse than no placeholder.

Rendering is off unless VERIFAI_SKELETON=1. A public deploy must never advertise
capability it does not have; see `render.placeholder`.

Filling one in means deleting its entry here and adding the real component.
"""
from __future__ import annotations

PLANNED: dict[str, dict[str, str]] = {
    # ---- model page ---------------------------------------------------------
    "stale_status": {
        "title": "Evaluated on an older checkpoint",
        "shows": "which reports were scored against weights that have since been "
                 "retrained — evaluated, but no longer of this model",
        "blocked_by": "a report does not record the checkpoint hash it was scored "
                      "against; one field in report.json's meta, written by the runner",
        "phase": "ui-ux-design.md#what-step-2-landed",
    },
    # ---- report -------------------------------------------------------------
    "provenance_strip": {
        "title": "Provenance strip",
        "shows": "task, access level, intended-use profile, checkpoint revision "
                 "and the preprocessing fingerprint the model was scored under",
        "blocked_by": "Phase A (access and task declarations) and Phase B "
                      "(preprocessing fingerprint)",
        "phase": "ROADMAP.md#phase-a-the-two-contracts-and-capability-gating",
    },
    "integrity_gate": {
        "title": "The rest of the integrity gate",
        "shows": "provenance, corpus ancestry and label-space compatibility beside the "
                 "split-leakage check — the parts of the gate that need more than a "
                 "training manifest to run",
        "blocked_by": "Phase B. Split leakage exists, and the banner that holds the page "
                      "when it fails is built; the other three checks are not",
        "phase": "ROADMAP.md#phase-b-what-you-can-and-cannot-verify-about-someone-elses-model",
    },
    "coverage_map": {
        "title": "Coverage map",
        "shows": "how many applicable metrics were measured, how many came back "
                 "inconclusive or not assessed, and why",
        "blocked_by": "Phase A: nothing knows the *applicable* set until tasks, "
                      "modalities and access are declared, so the denominator "
                      "would have to be invented",
        "phase": "ROADMAP.md#what-takes-its-place",
    },
    "findings_strip": {
        "title": "What this evaluation established",
        "shows": "the findings whose interval clears a stated reference, ordered "
                 "by strength of evidence and never by how good the number is",
        "blocked_by": "details[\"baseline\"] on each metric — `measured` currently "
                      "means three different things across the six",
        "phase": "ui-ux-design.md#findings-are-ordered-by-strength-of-evidence-never-by-how-good-the-number-is",
    },
    "impact": {
        "title": "Why it matters",
        "shows": "who is affected by this number being what it is, in this clinical "
                 "context — the third of the info box's five questions, and the only one "
                 "it cannot answer yet",
        "blocked_by": "no metric ships explain.impact; it is the fourth key beside "
                      "what, how and limits",
        "phase": "extending.md#planned-the-info-box-a-metric-must-be-able-to-fill",
    },
    "criterion_card": {
        "title": "Criterion",
        "shows": "the threshold applied to this number, its rationale, its owner "
                 "and its references — a judgement you can argue with",
        "blocked_by": "Phase G: the versioned policy file does not exist",
        "phase": "ROADMAP.md#phase-g-the-findings-layer-measurement-judgement-and-the-line-between-them",
    },
    "case_link": {
        "title": "Reach the cases behind this number",
        "shows": "the individual cases that produced a finding, and what they "
                 "share with each other",
        "blocked_by": "details[\"per_example\"] is written by every metric and "
                      "read by nothing; at scale it needs a cap first",
        "phase": "case-view.md#screen-2-the-failure-cluster",
    },
    "tensions": {
        "title": "Declared tensions",
        "shows": "what this evaluation cannot simultaneously optimise — "
                 "sensitivity against PPV, privacy against utility",
        "blocked_by": "the glossary carries `tension` per metric; cross-metric "
                      "statements are not modelled yet",
        "phase": "ROADMAP.md#what-takes-its-place",
    },
    "export_card": {
        "title": "Export as an evaluation card",
        "shows": "the report as a readable document — coverage, every metric with "
                 "its interval, and what the evaluation does not tell you",
        "blocked_by": "the coverage map and the info box it would be built from",
        "phase": "ROADMAP.md#what-takes-its-place",
    },
    # ---- compare ------------------------------------------------------------
    "access_statement": {
        "title": "Access levels in this group",
        "shows": "which runs could be opened and which could only be queried, and "
                 "which rows are therefore compared at the weaker level",
        "blocked_by": "Phase A: every model here is a local checkpoint, so the "
                      "question has never had to be asked",
        "phase": "ROADMAP.md#comparing-models-that-were-not-evaluated-under-the-same-suite",
    },
    "delta_view": {
        "title": "Against the previous version",
        "shows": "what changed since the version you shipped, on the same images "
                 "— a comparison that needs no authored threshold at all",
        "blocked_by": "nothing in the engine; it needs `supersedes:` to be "
                      "declared, and nothing in this repository is a successor yet",
        "phase": "direction.md#the-judgement-that-needs-no-threshold-at-all",
    },
    # ---- whole pages --------------------------------------------------------
    "metric_catalogue": {
        "title": "The metric catalogue",
        "shows": "every metric in the taxonomy with the tier, access level and "
                 "task it needs — including the ones that did not run here",
        "blocked_by": "Phase F: the catalogue is prose in docs/pillars.md, not data",
        "phase": "ROADMAP.md#phase-f-the-metric-catalogue-the-taxonomy-becomes-data",
    },
    "policy_catalogue": {
        "title": "The policy catalogue",
        "shows": "every criterion the active profile applies, with its rationale, "
                 "its source and the person accountable for it",
        "blocked_by": "Phase G: there is no policy file",
        "phase": "ROADMAP.md#phase-g-the-findings-layer-measurement-judgement-and-the-line-between-them",
    },
    "studio": {
        "title": "New evaluation",
        "shows": "resolve a checkpoint or a Hub link, review what was resolved, "
                 "pick a test set, preflight it, run it",
        "blocked_by": "Phases C and D. Local only by construction — the public "
                      "deploy has no torch and no checkpoints",
        "phase": "ROADMAP.md#phase-d-the-two-entry-tracks-one-command",
    },
    "preflight": {
        "title": "Preflight",
        "shows": "provenance, label space and split integrity with no metric run "
                 "— plus what the evaluation will cost before it starts",
        "blocked_by": "Phase D, and the per-metric `cost` declaration from Phase A",
        "phase": "ROADMAP.md#phase-d-the-two-entry-tracks-one-command",
    },
}

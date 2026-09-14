"""Findings data model — the single result format for the whole pipeline.

Ported/simplified from verifai_2_0's findings layer. The engine produces
`Finding`s, the exporter serializes them to JSON, the Streamlit showcase renders
them. One data model from end to end.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Literal

Pillar = Literal["integrity", "fairness", "robustness", "explainability",
                 "privacy", "performance"]
# Deliberately epistemic, never evaluative. These say what is *known* about a
# number, not whether the number is good — because "good" is a claim this project
# has no basis for. A robustness of 0.85 or a fairness gap of 0.08 are thresholds
# somebody would have to justify, and nothing here can: what counts as robust
# enough depends on where the model is deployed and what the cost of being wrong
# is, which is a decision for the reader, not a constant in a metric module.
#
# Measured on the real runs, the old accuracy threshold did worse than nothing.
# It marked the configuration that catches 159 of 163 melanomas as a WARNING and
# the one that misses 82 of them as a PASS, because under-calling the rare class
# raises overall accuracy. A reader trusting the badges would have picked the
# worst detector in the set.
#
#   measured      computed, and the sample supports reporting it
#   insufficient  computed, but too few cases to support any claim
#   unavailable   could not be computed; `summary` says why
#   invalid       a precondition failed, so the number does not mean what it
#                 appears to (today: a contaminated split)
#
# The field is still called `verdict` because it is the serialized key in every
# artifact already written; renaming it would break them for no reader-visible
# gain. The vocabulary is what carries the meaning.
Verdict = Literal["measured", "insufficient", "unavailable", "invalid"]
Domain = Literal["image", "text", "tabular", "llm"]


@dataclass
class Finding:
    pillar: Pillar
    metric: str                     # e.g. "subgroup_sensitivity_gap"
    domain: Domain
    value: Any                      # scalar or dict of numbers
    verdict: Verdict = "measured"
    summary: str = ""               # one human-readable sentence
    details: dict[str, Any] = field(default_factory=dict)
    plots: list[str] = field(default_factory=list)  # relative paths under the report's plot dir


@dataclass
class Report:
    scenario: str
    domain: Domain
    model_id: str
    dataset_id: str
    findings: list[Finding] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)   # sample_size, seed, versions, timing
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def add(self, finding: Finding) -> None:
        self.findings.append(finding)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

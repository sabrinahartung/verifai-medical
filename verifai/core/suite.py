"""Which version of each metric is current — the metric suite.

A report records the versions that produced it, and the model registry
records the current ones, so the showcase can say when an *active* report was
produced by a metric that has since changed. That is what lets archived runs
stay frozen: nobody has to re-run twenty-three evaluations because one metric
changed, only the active ones — and the app says which are behind.

**Bump a metric's version whenever what it reports changes** — a new sample, a
new field, a fixed bug, different wording in a verdict. Not for a refactor
that produces byte-identical output. A version that is not bumped makes an
out-of-date report look current, which is the one failure this module exists
to prevent.

Deliberately free of imports: the registry exporter and the tests read it
without loading torch.
"""
from __future__ import annotations

# Version 2 (2026-09-25): every finding carries `details["baseline"]` — what its
# number is compared with, and whether the interval clears it. Grad-CAM's 2 is
# also a new measurement: every test image, against a random-region control.
METRIC_VERSIONS: dict[str, int] = {
    "integrity.split_leakage": 2,
    "performance.classification": 2,
    "explainability.gradcam": 2,        # 2: every test image, against a random control
    "robustness.corruption": 2,
    "fairness.skin_tone": 2,
    "privacy.mia": 2,
}


def suite_for(metric_ids: list[str]) -> dict[str, int]:
    """The versions of the metrics a scenario runs, as a report records them."""
    return {m: METRIC_VERSIONS[m] for m in metric_ids}

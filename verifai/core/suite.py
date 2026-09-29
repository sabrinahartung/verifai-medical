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
#
# Version 3 (2026-09-25): summaries a first-time reader can read — class names in
# words, three decimals, thousands separated, en-dash intervals. Robustness also
# gains an interval on its mean and states n.
METRIC_VERSIONS: dict[str, int] = {
    "integrity.split_leakage": 2,
    "integrity.provenance": 1,
    "integrity.corpus_ancestry": 1,
    "integrity.label_space": 1,
    "performance.classification": 3,
    "explainability.gradcam": 3,        # 2: every test image, against a random control
    "robustness.corruption": 3,         # 3: an interval on the mean, and n
    "fairness.skin_tone": 4,           # 4: no dataset named in its text; intervals in the summary
    "privacy.mia": 3,
}


def suite_for(metric_ids: list[str]) -> dict[str, int]:
    """The versions of the metrics a scenario runs, as a report records them."""
    return {m: METRIC_VERSIONS[m] for m in metric_ids}

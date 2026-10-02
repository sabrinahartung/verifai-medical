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

**Two numbers per metric** (step F1, 2026-10-02). `METRIC_VERSIONS` says when a
report is behind: any change to what a metric reports, its wording included.
`MEASUREMENT_VERSIONS` says when two numbers stop being comparable: bumped only
when what the number *means* changes — a different sample, a different method,
a fixed bug that moves values. A new field or a new sentence leaves it alone.
The comparison view refuses to set a value beside one measured under another
measurement version, so a wording change never empties a comparison and a
method change never hides inside one.

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
# Version +1 for every metric (2026-10-02, step F2): each finding carries its
# sub-aspect. Measurement versions are unchanged — no number moved.
METRIC_VERSIONS: dict[str, int] = {
    "integrity.split_leakage": 3,
    "integrity.provenance": 3,          # 2: declares that none of its counts is ranked
    "integrity.corpus_ancestry": 3,     # 2: likewise
    "integrity.label_space": 3,         # 2: likewise
    "integrity.preprocessing": 3,       # C4: the preprocessing against the model's own
                                        # 2: an unchecked run's zeros are not ranked
    "performance.classification": 4,
    "explainability.gradcam": 5,        # 2: every test image, against a random control
                                        # 4: per-image scores in the run's case table
    "performance.calibration": 1,       # F3
    "robustness.corruption": 4,         # 3: an interval on the mean, and n
    "fairness.skin_tone": 5,           # 4: no dataset named in its text; intervals in the summary
    "privacy.mia": 4,
}


# What each metric's numbers mean. Every metric is still on the measurement it
# shipped with, except Grad-CAM, whose deletion test moved from seven images to
# every test image against a random-region control (ee46d25, 2026-09-25). Read
# off the history of each metric module 2026-10-02: the other changes since the
# first snapshot (2026-09-08) added fields or rewrote sentences, and batching
# (C3) reproduced every value exactly.
MEASUREMENT_VERSIONS: dict[str, int] = {m: 1 for m in METRIC_VERSIONS}
MEASUREMENT_VERSIONS["explainability.gradcam"] = 2

# When each measurement version took effect, for snapshots written before
# snapshots recorded versions (before step F1). A version not listed here was in
# force from the metric's first run. Commit time, so a snapshot from the same day
# is placed on the right side of it.
MEASUREMENT_SINCE: dict[str, dict[int, str]] = {
    "explainability.gradcam": {2: "2026-09-25T00:14:03+02:00"},
}


def measurement_at(metric_id: str, created_at: str) -> int:
    """The measurement version a metric ran under at a moment in the past.

    For a snapshot that does not say. ISO timestamps with offsets compare
    correctly as datetimes, never as strings.
    """
    from datetime import datetime
    when = datetime.fromisoformat(created_at)
    version = 1
    for v, since in sorted(MEASUREMENT_SINCE.get(metric_id, {}).items()):
        if when >= datetime.fromisoformat(since):
            version = v
    return version


def suite_for(metric_ids: list[str]) -> dict[str, int]:
    """The versions of the metrics a scenario runs, as a report records them."""
    return {m: METRIC_VERSIONS[m] for m in metric_ids}

"""Export a Report to static artifacts the Streamlit showcase reads.

Writes:
  <out_dir>/<scenario>/report.json          (the full Report — always "latest")
  <out_dir>/<scenario>/card.json            (tile metadata)
  <out_dir>/<scenario>/plots/*.png          (referenced by Finding.plots)
  <out_dir>/<scenario>/history/<ts>.json    (one snapshot per run, kept)

`report.json` is overwritten every run so the dashboard is unchanged. Snapshots
accumulate beside it so a model can be compared against its own earlier versions —
without which each training experiment would silently overwrite the evidence of
the last one.

No DB, no server — just files that get committed into the showcase.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

from verifai.core.findings import Report
from verifai.core.suite import MEASUREMENT_VERSIONS

# Worst last. A verdict outside the vocabulary (a legacy word) ranks with `measured`,
# so on its own it still travels unchanged.
_INTEGRITY_ORDER = {"measured": 0, "insufficient": 1, "unavailable": 2, "invalid": 3}


def write_report(report: Report, out_dir: str = "showcase/artifacts",
                 card: dict | None = None) -> Path:
    """Write report.json (+ card.json so the showcase auto-lists it as a tile).

    `card` overrides/extends the auto-generated tile metadata, e.g.
    {"name": "...", "emoji": "🔬", "description": "...", "hf_url": "...", "dataset": "..."}.
    Adding a new model = one more run() -> one more folder -> one more tile.
    """
    base = Path(out_dir) / report.scenario
    (base / "plots").mkdir(parents=True, exist_ok=True)

    data = report.to_dict()
    write_cases(data, base)
    with open(base / "report.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    card_out = {
        "id": report.scenario,
        "name": report.model_id,
        "emoji": "🧠",
        "domain": report.domain,
        "dataset": report.dataset_id,
        "description": "",
        "sample": False,
        **(card or {}),
    }
    with open(base / "card.json", "w", encoding="utf-8") as f:
        json.dump(card_out, f, ensure_ascii=False, indent=2)

    write_snapshot(report, base)
    return base / "report.json"


CASES_FILE = "cases.csv"


def _cell(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def write_cases(data: dict, base: Path) -> Path | None:
    """Move every finding's per-case rows into one table per run: `cases.csv`.

    A metric still returns `details["per_example"]` — one dict per case, with
    its `id` — and keeps no copy in `report.json`, where the rows of every
    metric for every case would make a report grow with metrics × cases. In
    their place the finding says which columns are its own:
    `details["cases"] = {"file": "cases.csv", "columns": [...]}`.

    One row per case, in the order the cases were first seen; a column per
    metric field, named `<finding>.<field>`, so two metrics never overwrite each
    other's `pred`. Booleans as `true`/`false`, a missing value empty. Written
    once per run and replaced by the next, never edited. A file, never a
    database: what the planned case view indexes.
    """
    rows: dict[str, dict[str, str]] = {}
    columns: list[str] = []
    for f in data.get("findings", []):
        details = f.get("details") or {}
        per = details.pop("per_example", None)
        if not per:
            continue
        own: list[str] = []
        for entry in per:
            row = rows.setdefault(str(entry["id"]), {})
            for k, v in entry.items():
                if k == "id":
                    continue
                col = f"{f['metric']}.{k}"
                if col not in own:
                    own.append(col)
                row[col] = _cell(v)
        columns += [c for c in own if c not in columns]
        details["cases"] = {"file": CASES_FILE, "columns": own}
    path = base / CASES_FILE
    if not rows:
        path.unlink(missing_ok=True)   # a run with no per-case rows leaves none behind
        return None
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")   # as git stores it; \r\n would show as a change
        w.writerow(["id", *columns])
        for case_id, row in rows.items():
            w.writerow([case_id, *(row.get(c, "") for c in columns)])
    return path


# Deep enough to reach value["per_class"]["melanoma"]["sensitivity"] — the number
# a cost-sensitive experiment is actually about. At depth 2 it was invisible, so
# the comparison could show accuracy moving while the metric that motivated the
# change was missing from the table.
MAX_FLATTEN_DEPTH = 3


def _flatten(value: Any, prefix: str, out: dict[str, float], depth: int = 0) -> None:
    """Collect the numeric leaves of a finding's `value` as dotted keys.

    Generic on purpose: a new metric becomes comparable without this module
    learning anything about it.
    """
    if isinstance(value, bool):            # bool is an int; not a metric
        return
    if isinstance(value, (int, float)):
        out[prefix] = float(value)
        return
    if isinstance(value, dict) and depth < MAX_FLATTEN_DEPTH:
        for k, v in value.items():
            _flatten(v, f"{prefix}.{k}" if prefix else str(k), out, depth + 1)


def snapshot_metrics(report: Report) -> dict[str, float]:
    """Every numeric result in the report, keyed as `<pillar>.<path>`."""
    flat: dict[str, float] = {}
    for f in report.findings:
        _flatten(f.value, f.pillar, flat)
    return flat


def snapshot_key_metric(report: Report) -> dict[str, str]:
    """Which registered metric each flattened key comes from.

    Keys are grouped by pillar, so five integrity checks share `integrity.*`;
    this is what tells them apart again. Empty for a report whose runner did not
    record `meta["metric_ids"]` (before F1).
    """
    ids = (report.meta or {}).get("metric_ids") or {}
    out: dict[str, str] = {}
    for f in report.findings:
        if f.metric not in ids:
            continue
        keys: dict[str, float] = {}
        _flatten(f.value, f.pillar, keys)
        out.update({k: ids[f.metric] for k in keys})
    return out


def snapshot_directions(report: Report) -> dict[str, str]:
    """Which way is an improvement, as declared by each metric.

    The app must not infer this: "auc" means higher-is-better for a classifier and
    lower-is-better for membership inference, so only the metric knows. Patterns
    may contain `*` (e.g. `per_class.*.sensitivity`). Anything undeclared is left
    unranked rather than guessed at.
    """
    out: dict[str, str] = {}
    for f in report.findings:
        for pattern, direction in ((f.details or {}).get("better") or {}).items():
            out[f"{f.pillar}.{pattern}"] = direction
    return out


def write_snapshot(report: Report, base: Path) -> Path:
    """One immutable record per run, carrying what makes it comparable (or not)."""
    hist = base / "history"
    hist.mkdir(parents=True, exist_ok=True)
    meta = report.meta or {}

    # The worst of the integrity checks, as the report page's banner takes it, so the
    # comparison and the report cannot disagree about whether a run is usable. Before
    # C4 this was the first check alone — the split — and a run whose images were
    # prepared differently from the model's own would still have been plotted.
    checks = [f for f in report.findings if f.pillar == "integrity"]
    decided = max(checks, key=lambda f: _INTEGRITY_ORDER.get(f.verdict, 0), default=None)
    snap = {
        "created_at": report.created_at,
        "scenario": report.scenario,
        "label": meta.get("label") or report.model_id,
        "model_id": report.model_id,
        "dataset_id": report.dataset_id,
        # the comparability key: same rows, and an evaluation worth believing
        "eval_set": meta.get("eval_set") or {},
        "integrity": decided.verdict if decided else None,
        # which check decided it, so a blocked run can say why
        "integrity_by": decided.metric if decided else None,
        "device": meta.get("device"),
        "seed": meta.get("seed"),
        # how much of the model this run could reach: a comparison between a
        # model that was opened and one that was only queried says so
        "access": meta.get("access"),
        "task": meta.get("task"),
        "verdicts": {f.pillar: f.verdict for f in report.findings},
        "metrics": snapshot_metrics(report),
        "directions": snapshot_directions(report),
        # Since F1: which metric each number came from, and the measurement version
        # it was taken under. The comparison sets a value only beside values of the
        # same measurement version. Older snapshots carry neither; the showcase
        # dates them from the registry's measurement history instead.
        "measurement_versions": {m: MEASUREMENT_VERSIONS[m]
                                 for m in (meta.get("metric_versions") or {})
                                 if m in MEASUREMENT_VERSIONS},
        "key_metric": snapshot_key_metric(report),
    }
    name = re.sub(r"[^0-9A-Za-z]", "-", report.created_at) + ".json"
    path = hist / name
    with open(path, "w", encoding="utf-8") as f:
        json.dump(snap, f, ensure_ascii=False, indent=2)
    return path

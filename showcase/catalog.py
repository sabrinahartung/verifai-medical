"""Artifacts, snapshots and the verdict vocabulary — the showcase's data layer.

Everything here reads files and returns plain dicts. No Streamlit call belongs in
this module: it is what the contract tests import, and what `views/` renders.

Split out of the single-file `app.py` (roadmap Phase E). The public names stay
importable from `showcase.app`, so nothing outside had to change.
"""
from __future__ import annotations

import json
from pathlib import Path

ART = Path(__file__).parent / "artifacts"

# integrity comes first on purpose: every other pillar is conditional on it.
PILLARS = ["integrity", "performance", "fairness", "robustness", "explainability", "privacy"]

# The plain-language question each pillar answers, for readers who have never
# seen a Responsible-AI report.
PILLAR_QUESTION = {
    "integrity":      "Can these results be trusted at all?",
    "performance":    "Does the model get the answer right?",
    "fairness":       "Does it work equally well for everyone?",
    "robustness":     "Does it stay reliable when the image is imperfect?",
    "explainability": "Can we see why it decided what it decided?",
    "privacy":        "Could the model leak the data it was trained on?",
}

# icon, short label, what the status actually means.
#
# Epistemic, never evaluative: these say what is known about a number, not
# whether it is good. There is no pass and no fail, because "good enough" is a
# threshold somebody would have to justify and nothing here can — what counts as
# robust or fair enough depends on where the model is used and what being wrong
# costs. `invalid` is the one hard signal, and it judges the *measurement*, not
# the model: a contaminated split does not measure generalisation at all.
# Icons must be real emoji, not geometric symbols: `st.info(icon=...)` validates
# them and raises on anything else, so a tidy-looking ◐ or ∅ takes the whole
# finding down at render time. Asserted in tests.
#
# `insufficient` is deliberately ❔ and not ⚠️. A warning sign was the icon for the
# retired `warn`, and bringing it back would restore the visual language of
# grading even though the word behind it changed — worse, it would blame the
# model for a limit of the evaluation. On the 7-image run four of six pillars are
# `insufficient`; in warning triangles that reads as four problems with the
# model, when the model has no findings at all and the sample is simply too
# small. A question mark says the one true thing: we cannot say. It is also not
# ⏳, which suggests a measurement still running rather than one already finished
# and inconclusive, and not ❔, whose outline glyph renders as pale grey and is
# nearly invisible on the light background. 🔍 reads as "looked at, and cannot
# say" once the label beside it says so.
VERDICT = {
    "measured":     ("📊", "Measured",        "Computed, and the sample supports reporting "
                                              "it. Whether the value is good enough is a "
                                              "judgement this report does not make."),
    "insufficient": ("🔍", "Not enough data",  "Computed, but too few cases to support any "
                                              "claim — the interval is too wide to "
                                              "distinguish this from chance."),
    "unavailable":  ("➖", "Not computable",   "Could not be computed. The result line says "
                                              "what was missing; no number is invented."),
    "invalid":      ("⛔", "Not usable",      "A precondition failed — the split was "
                                              "contaminated, so these numbers measure "
                                              "memory rather than generalisation."),
}
VERDICT_ORDER = {"measured": 0, "insufficient": 1, "unavailable": 2, "invalid": 3}

# Artifacts written before the vocabulary changed still carry pass/warn/fail.
# Mapped on read so old reports render in today's language rather than breaking,
# and so a stale badge cannot go on claiming a verdict this project withdrew.
# Integrity is mapped separately because its old `fail`/`warn` meant a
# contaminated split — a fact worth keeping — while elsewhere they were only a
# threshold nobody could justify.
_LEGACY_VERDICT = {"pass": "measured", "warn": "measured", "fail": "measured",
                   "info": "insufficient"}
_LEGACY_INTEGRITY = {"pass": "measured", "warn": "invalid", "fail": "invalid",
                     "info": "unavailable"}
# Grad-CAM emitted `info` unconditionally, whatever the sample size — for that
# metric it meant "no verdict is defined here", never "not enough evidence".
# Mapping it like the rest reported "Not enough data" next to overlays that had
# been rendered and a faithfulness score that had been computed.
_LEGACY_EXPLAINABILITY = {**_LEGACY_VERDICT, "info": "measured"}

_LEGACY_BY_PILLAR = {"integrity": _LEGACY_INTEGRITY,
                     "explainability": _LEGACY_EXPLAINABILITY}


def normalise_verdict(value: str | None, pillar: str | None = None) -> str:
    """Today's status for a finding, mapping the retired pass/warn/fail words.

    Pillar-aware, because the retired `info` was overloaded: for most metrics it
    gated a claim on the evidence, for integrity it meant the split could not be
    checked, and for explainability it was simply the only value that metric ever
    emitted. One table cannot say all three.
    """
    if value in VERDICT:
        return value
    table = _LEGACY_BY_PILLAR.get(pillar or "", _LEGACY_VERDICT)
    return table.get(value or "", "unavailable")


# ---------- catalog ----------
def load_catalog() -> list[dict]:
    cards = []
    if not ART.exists():
        return cards
    for d in sorted(ART.iterdir()):
        card_f, report_f = d / "card.json", d / "report.json"
        if card_f.exists() and report_f.exists():
            card = json.loads(card_f.read_text(encoding="utf-8"))
            card["_dir"] = d
            cards.append(card)
    # run_label falls back to these for artifacts written before scenarios
    # declared a `label:`. Primed here rather than by the caller because every
    # page needs it and only this function knows when the catalog was read.
    _CARD_NAMES.update({c["id"]: c["name"] for c in cards if c.get("name")})
    return cards


# ---------- snapshots: comparing runs, and refusing to ----------
def load_snapshots() -> list[dict]:
    """Every recorded run across every artifact folder, newest last."""
    snaps = []
    if not ART.exists():
        return snaps
    for d in sorted(ART.iterdir()):
        for f in sorted((d / "history").glob("*.json")) if (d / "history").is_dir() else []:
            try:
                s = json.loads(f.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            s["_file"] = f.name
            snaps.append(s)
    return sorted(snaps, key=lambda s: s.get("created_at", ""))


# Scenario id -> the card's human name, filled once per render from the catalog.
# A module-level cache rather than a parameter because run_label is called from
# eight places, several of them deep inside the comparison view.
_CARD_NAMES: dict[str, str] = {}


def run_label(snap: dict) -> str:
    """What to call this run in a table, a chart legend or a warning.

    A scenario declares `label:` and the exporter writes it into the snapshot;
    without one it falls back to `model_id`, and a table headed
    `external-derm7pt-isic` against `external-derm7pt-isic-masked` is one a reader
    has to decode rather than read — the two differ by a single decision weight
    and nothing in those strings says which is which.

    Every scenario carries a label now, but artifacts written before that do not,
    and re-running eighteen evaluations to change a caption would be absurd. So a
    snapshot whose label is merely its model id falls back to the gallery card's
    name, which was always human-readable.
    """
    label = snap.get("label")
    if label and label != snap.get("model_id"):
        return label
    return _CARD_NAMES.get(snap.get("scenario", ""), label or snap.get("scenario", "?"))


def comparability_key(snap: dict) -> tuple:
    """Runs are comparable only when scored on exactly the same rows.

    The manifest's content hash, not its path — a manifest can be regenerated
    with a different seed and keep its name, and the same bytes can be read from
    a different path (a clone elsewhere, or a renamed checkout). The hash is
    recorded over the file's contents alone, so it already answers the only
    question that matters: were these runs scored on the same rows?

    A snapshot with no hash is the exception. Absent a hash there is nothing to
    compare on, so those fall back to the path and never merge with each other
    on the strength of being equally unidentified.
    """
    ev = snap.get("eval_set") or {}
    digest = ev.get("sha256")
    if not digest:
        return (None, ev.get("manifest"))
    return (digest,)


def group_snapshots(snaps: list[dict]) -> dict[tuple, list[dict]]:
    groups: dict[tuple, list[dict]] = {}
    for s in snaps:
        groups.setdefault(comparability_key(s), []).append(s)
    return groups


def direction_for(key: str, snaps: list[dict]) -> str | None:
    """'higher' | 'lower' | None — as declared by the metric, never inferred.

    Patterns may contain `*` (e.g. `performance.per_class.*.sensitivity`). An
    undeclared metric stays unranked: showing a value is honest, calling it better
    is not.
    """
    import fnmatch
    for s in snaps:
        for pattern, d in (s.get("directions") or {}).items():
            if key == pattern or fnmatch.fnmatch(key, pattern):
                return d
    return None


def best_run(key: str, runs: list[dict], direction: str | None) -> str | None:
    """Label of the run leading on this metric, or None if the metric is unranked."""
    if not direction:
        return None
    vals = [(r, r["metrics"].get(key)) for r in runs]
    vals = [(r, v) for r, v in vals if v is not None]
    if not vals:
        return None
    pick = max(vals, key=lambda rv: rv[1]) if direction == "higher" else min(vals, key=lambda rv: rv[1])
    return run_label(pick[0])


def dominated_by(runs: list[dict], keys: list[str], dirs: dict[str, str | None]) -> dict[str, str]:
    """Which runs are beaten on *every* ranked metric by some other run.

    A dominated run can be dismissed on the evidence alone. Anything left over is
    a genuine trade-off, where choosing requires saying what the model is *for* —
    which no amount of charting can decide.
    """
    ranked = [k for k in keys if dirs.get(k)]
    if not ranked:
        return {}
    out: dict[str, str] = {}
    for a in runs:
        la = run_label(a)
        for b in runs:
            if a is b:
                continue
            lb = run_label(b)
            better_somewhere = False
            worse_somewhere = False
            for k in ranked:
                va, vb = a["metrics"].get(k), b["metrics"].get(k)
                if va is None or vb is None:
                    worse_somewhere = True      # cannot claim dominance on missing data
                    break
                if va == vb:
                    continue
                a_wins = (va > vb) if dirs[k] == "higher" else (va < vb)
                better_somewhere |= a_wins
                worse_somewhere |= not a_wins
            if not worse_somewhere and better_somewhere:
                out[lb] = la                     # b is dominated by a
    return out


def _blocked_reason(snap: dict) -> str | None:
    """Why this run must not be plotted alongside the others."""
    if (snap.get("eval_set") or {}).get("sha256") is None:
        return "no evaluation manifest recorded, so there is nothing to match against"
    integrity = normalise_verdict(snap.get("integrity"), "integrity")
    if integrity == "invalid":
        return "its split was contaminated — the numbers are inflated by an unknown amount"
    if integrity != "measured":
        return "split integrity was never verified, so the numbers rest on an unchecked assumption"
    return None

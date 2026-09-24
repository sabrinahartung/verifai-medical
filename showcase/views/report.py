"""Report — one configuration's evaluation, pillar by pillar.

Laid out for the reader this project settled on: someone who has never seen a
Responsible-AI report and will spend twenty minutes, not five. Two rules follow
from that (docs/ui-ux-design.md, "Two settled decisions"):

* **The first screen says what was found.** The "At a glance" list is every
  pillar's result in the engine's own words, in fixed pillar order, each a link
  down to its section. Above it, *What this evaluation established* lists only
  the findings whose interval clears a stated reference, also in pillar order.
* **Depth by scrolling, never by clicking.** Each finding's five-question box is
  rendered open; only the reference definitions sit behind a click.

Integrity is not one pillar among six. Every other number is conditional on the
split being clean, so when it could not be verified — or was contaminated — the
report says so in a banner above everything else, in the finding's own words.
"""
from __future__ import annotations

import json

import streamlit as st

from catalog import PILLARS, PILLAR_QUESTION, VERDICT, VERDICT_ORDER, normalise_verdict
from registry import decision_rule, is_archived, load_registry, model_of_scenario, out_of_date
from render import breadcrumb, placeholder, render_finding
from routing import go_to_model, go_to_overview, go_to_project


def integrity_state(findings: list[dict]) -> tuple[str, dict | None]:
    """The report's integrity status, and the finding that decided it.

    No integrity finding at all is `unavailable` — an evaluation that never
    checked its split has exactly the standing of one that could not.
    """
    items = [f for f in findings if f.get("pillar") == "integrity"]
    if not items:
        return "unavailable", None
    worst = max(items, key=lambda f: VERDICT_ORDER.get(
        normalise_verdict(f.get("verdict"), "integrity"), 1))
    return normalise_verdict(worst.get("verdict"), "integrity"), worst


def _integrity_banner(state: str, finding: dict | None):
    """Hold the page when the measurement itself is unusable or unverified.

    This judges the *measurement*, not the model — the one kind of judgement this
    tool makes. The words are the finding's own, so the banner cannot drift from
    what the integrity check actually reported.
    """
    if state == "measured":
        return
    icon, label, _ = VERDICT[state]
    said = (finding or {}).get("summary") or "No split-integrity check was run for this evaluation."
    if state == "invalid":
        st.error(f"**{label}: the split is contaminated.** {said} Everything below is shown "
                 f"for the record and should not be read as a result.", icon=icon)
    else:
        st.warning(f"**Read everything below as provisional.** {said}", icon=icon)


def _status_banner(card: dict, report: dict, registry: dict | None, model: dict | None):
    """Say whether this report is current, before the reader reads a number.

    Archived is not a warning — the numbers were measured correctly with the
    metrics of their day — so it gets a neutral note, said once. An *active*
    report that a metric or a retrain has overtaken gets a warning, with the
    reasons, because it claims to be current and no longer is.
    """
    when = (report.get("created_at") or "")[:10] or "an earlier date"
    if is_archived(registry, card["id"]):
        st.info(f"**Archived.** Evaluated on {when} with the metrics of that time and kept as "
                f"the record of what was measured then. It is not re-run as the metrics change, "
                f"so it may lack metrics added or improved since.", icon="📦")
        return
    reasons = out_of_date(report.get("meta") or {}, registry, model) if model else []
    if reasons:
        st.warning(f"**Behind the current metrics** — evaluated on {when}, and "
                   + "; ".join(reasons)
                   + ". Re-run the active configurations to bring it up to date: "
                     "`uv run python scripts/run_active.py`.", icon="🔄")


def _identity(card: dict, report: dict, model: dict | None, config: dict | None):
    meta = report.get("meta") or {}
    ev = meta.get("eval_set") or {}
    manifest = (ev.get("manifest") or "").rsplit("/", 1)[-1]
    n = ev.get("n") or meta.get("sample_size")
    if card.get("description"):
        st.caption(card["description"])
    if model and config:
        st.markdown(
            f"A configuration of **{model['name']}** · decision rule: "
            f"{decision_rule(config['decision_weights'])} · scored on `{manifest or '?'}`"
            + (f" ({n:,} images)" if n else ""))
    else:
        # An evaluation no registered model claims keeps the identifiers it has.
        st.markdown(f"**Model:** `{report['model_id']}` · **Dataset:** `{report['dataset_id']}`"
                    + (f" ({n:,} images)" if n else ""))
    if card.get("hf_url"):
        st.markdown(f"[🤗 Model on Hugging Face]({card['hf_url']})")


def established(findings: list[dict], integrity: str) -> list[dict] | None:
    """The findings this evaluation established, in fixed pillar order.

    Established means two things at once: the metric's own verdict is
    `measured` — the sample supports reporting it — and its baseline's interval
    clears the stated reference. Ordered by pillar, never by how good or bad
    the number is: a clean split and a 33-point fairness gap stand side by side
    because both are supported, not because one is worse.

    `None` when no finding carries a baseline at all — a report from before
    baselines existed, which has nothing to say here rather than nothing
    established. And nothing is established on a contaminated split.
    """
    if integrity == "invalid":
        return []
    if not any("baseline" in (f.get("details") or {}) for f in findings):
        return None
    order = {p: i for i, p in enumerate(PILLARS)}
    hits = [f for f in findings
            if normalise_verdict(f.get("verdict"), f.get("pillar")) == "measured"
            and ((f.get("details") or {}).get("baseline") or {}).get("cleared")]
    return sorted(hits, key=lambda f: order.get(f.get("pillar"), len(order)))


REFERENCE_KIND = {"ideal": "against its ideal", "chance": "against chance",
                  "control": "against a control measured in the same run"}


def _findings_strip(findings: list[dict], integrity: str):
    hits = established(findings, integrity)
    if hits is None:
        return                 # predates baselines; the archived banner already says so
    st.subheader("What this evaluation established")
    st.caption("Only results whose interval clears a stated reference — an ideal, chance, or a "
               "control measured in the same run. Ordered by pillar, never by how good or bad "
               "the number is.")
    if not hits:
        st.markdown(":gray[Nothing in this report clears its reference at this sample size.]")
        return
    with st.container(border=True):
        for f in hits:
            b = f["details"]["baseline"]
            p = f["pillar"]
            st.markdown(f"**[{p.capitalize()}](#{p})** — {b['claim']}")
            st.caption(f"Compared {REFERENCE_KIND.get(b['kind'], '')}: {b['basis']}.")


def _at_a_glance(by_pillar: dict[str, list]):
    st.subheader("At a glance", anchor="at-a-glance")
    st.caption("Every pillar's result in the engine's own words, in fixed order. A status says "
               "what is known about a number — never whether it is good. Hover a status for "
               "what it means; select a pillar to jump to its section.")
    for p in PILLARS:
        items = by_pillar.get(p, [])
        with st.container(border=True):
            name, result = st.columns([1.3, 5])
            with name:
                st.markdown(f"**[{p.capitalize()}](#{p})**")
                st.caption(PILLAR_QUESTION[p])
            with result:
                if not items:
                    st.markdown(":gray[Not evaluated in this run.]")
                for f in items:
                    icon, label, meaning = VERDICT[normalise_verdict(f.get("verdict"), p)]
                    st.markdown(f"{icon} **{label}** — {f.get('summary', '')}", help=meaning)


def dashboard(card: dict):
    base = card["_dir"]
    report = json.loads((base / "report.json").read_text(encoding="utf-8"))

    # The way back up is the hierarchy the reader came down: project, model,
    # configuration. An evaluation no declared model accounts for has no
    # hierarchy above it, and gets the plain way back.
    registry = load_registry()
    model = model_of_scenario(registry, card["id"])
    config = next((c for c in model["configurations"] if c["scenario"] == card["id"]),
                  None) if model else None
    if model and config:
        breadcrumb([("Overview", go_to_overview),
                    (model["project"], lambda: go_to_project(model["project"])),
                    (model["name"], lambda: go_to_model(model["key"]))],
                   # A model with one configuration usually shares its name, and
                   # "… › Focal loss (γ=2) › Focal loss (γ=2)" reads as a glitch.
                   here=config["label"] if config["label"] != model["name"] else "Report")
    elif st.button("← Back to overview"):
        go_to_overview()

    # The configuration's label is its name everywhere else — the model page,
    # the comparison table, the breadcrumb just above — so it is the title too.
    st.title(f"{card.get('emoji', '🧠')} {config['label'] if config else card['name']}")
    _identity(card, report, model, config)
    _status_banner(card, report, registry, model)
    if card.get("sample"):
        st.warning("This view shows SAMPLE data — placeholder numbers, not an evaluation.")
    placeholder("export_card", compact=True)
    placeholder("provenance_strip")

    findings = report["findings"]
    state, decided_by = integrity_state(findings)
    _integrity_banner(state, decided_by)
    placeholder("integrity_gate")

    n = (report.get("meta") or {}).get("sample_size")
    if n:
        st.caption(f"Everything below was computed on {n:,} image(s). Small samples are marked as "
                   f"such. Nothing here is scored against a threshold: the numbers and their "
                   f"intervals are reported, and what counts as good enough is your call.")

    placeholder("coverage_map")
    _findings_strip(findings, state)

    by_pillar: dict[str, list] = {p: [] for p in PILLARS}
    for f in findings:
        by_pillar.setdefault(f["pillar"], []).append(f)
    _at_a_glance(by_pillar)

    # Per-finding slots that no metric can fill yet, drawn once rather than
    # under every finding — six identical boxes would be a wall, not a skeleton.
    if placeholder("criterion_card", compact=True):
        placeholder("case_link", compact=True)

    for p in PILLARS:
        items = by_pillar.get(p, [])
        if not items:
            continue
        st.divider()
        st.header(p.capitalize(), anchor=p)
        # Every section links back up: with each explanation open the page runs
        # to eleven screens, and the glance list is where a reader re-chooses.
        st.caption(f"{PILLAR_QUESTION[p]} · [↑ At a glance](#at-a-glance)")
        for f in items:
            render_finding(f, base)

    st.divider()
    placeholder("tensions")


def page():
    """Route entry: one run's report, or an empty state when none is selected."""
    from catalog import load_catalog
    from routing import current

    run_id = current("run")
    cards = load_catalog()
    card = next((c for c in cards if c["id"] == run_id), None)
    if card is None:
        st.title("No run selected")
        st.caption("Pick a model on the overview to see its analysis. "
                   "A report has its own link, so one can be shared or bookmarked.")
        if st.button("← Back to overview"):
            go_to_overview()
        return
    dashboard(card)

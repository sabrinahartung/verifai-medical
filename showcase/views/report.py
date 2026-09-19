"""Report — one run, grouped by pillar."""
from __future__ import annotations

import json

import streamlit as st

from catalog import PILLARS, PILLAR_QUESTION, VERDICT, VERDICT_ORDER, normalise_verdict
from render import (entries_for, metric_keys, placeholder, render_caveats,
                    render_chart, render_explain, render_metric_explanations)
from routing import go_to_overview

def dashboard(card: dict):
    base = card["_dir"]
    report = json.loads((base / "report.json").read_text(encoding="utf-8"))

    if st.button("← Back to overview"):
        go_to_overview()

    st.title(f"{card.get('emoji','🧠')} {card['name']}")
    st.write(f"**Domain:** {report['domain']}  ·  **Model:** `{report['model_id']}`  ·  "
             f"**Dataset:** `{report['dataset_id']}`")
    if card.get("hf_url"):
        st.markdown(f"[🤗 Model on Hugging Face]({card['hf_url']})")
    if card.get("sample"):
        st.warning("This view shows SAMPLE data — a placeholder until the real engine run produces the artifacts.")

    n = (report.get("meta") or {}).get("sample_size")
    if n:
        st.caption(f"Everything below was computed on {n} image(s). Small samples are marked as "
                   f"such. Nothing here is scored against a threshold: the numbers and their "
                   f"intervals are reported, and what counts as good enough is your call.")

    placeholder("provenance_strip")
    placeholder("integrity_gate")
    placeholder("coverage_map")
    placeholder("findings_strip")

    by_pillar: dict[str, list] = {p: [] for p in PILLARS}
    for f in report["findings"]:
        by_pillar.setdefault(f["pillar"], []).append(f)

    # ---- pillar overview row: icon + what the icon means ----
    st.subheader("At a glance")
    cols = st.columns(len(PILLARS))
    for c, p in zip(cols, PILLARS):
        items = by_pillar.get(p, [])
        if not items:
            c.metric(p.capitalize(), "–", help="Not evaluated in this run.")
            continue
        worst = max((normalise_verdict(f["verdict"], f.get("pillar")) for f in items),
                    key=lambda v: VERDICT_ORDER.get(v, 1))
        icon, label, meaning = VERDICT[worst]
        c.metric(p.capitalize(), icon, help=f"{PILLAR_QUESTION[p]}\n\n**{label}** — {meaning}")
        c.caption(label)

    with st.expander("What do the icons mean?"):
        for v in ("measured", "insufficient", "unavailable", "invalid"):
            icon, label, meaning = VERDICT[v]
            st.markdown(f"{icon} **{label}** — {meaning}")

    # Rendered once rather than under all six findings: the slot is per finding,
    # but eighteen identical boxes would be a wall instead of a skeleton.
    if placeholder("info_box", compact=True):
        placeholder("criterion_card", compact=True)
        placeholder("case_link", compact=True)

    st.divider()
    for p in PILLARS:
        items = by_pillar.get(p, [])
        if not items:
            continue
        st.header(p.capitalize())
        st.caption(PILLAR_QUESTION[p])
        for f in items:
            icon, label, _ = VERDICT[normalise_verdict(f["verdict"], f.get("pillar"))]
            st.markdown(f"##### {icon} `{f['metric']}` · {label}")
            ex = render_explain(f)
            # One accordion per finding rather than one switch for the page, so
            # the reader opens definitions only where they actually want them —
            # and closed, it costs a single line instead of pushing the chart
            # down. The finding's own `explain` says what this chart shows;
            # these cards define the quantities it is drawn from. Keys are
            # derived the same way the comparison view's are, so both views
            # explain a metric with the same words.
            cards = metric_keys(f.get("value"), f["pillar"])
            if entries_for(cards):
                with st.expander("Metric explanations"):
                    render_metric_explanations(cards, compact=True)
            details = f.get("details") or {}
            chart = details.get("chart")
            if chart:
                render_chart(chart, base)
            elif f.get("plots"):
                render_chart({"kind": "images", "paths": f["plots"]}, base)
            # optional secondary chart (e.g. a faithfulness scale or subgroup gap)
            if details.get("chart2"):
                render_chart(details["chart2"], base)
            render_caveats(ex)
            st.divider()


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
    placeholder("tensions")
    placeholder("export_card")

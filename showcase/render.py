"""Charts, explanation cards and the placeholder — everything that draws.

`render_chart` is the extensibility trick: a metric ships a chart spec inside its
finding and this module draws it, so a new metric needs no change here. The
placeholder at the bottom is the same idea for something that does *not* exist
yet.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st
import plotly.graph_objects as go

from catalog import PILLARS, PILLAR_QUESTION, VERDICT, normalise_verdict

# Plain-language metric explanations. They live in the engine
# (verifai/core/glossary.py) for the same reason a finding's own `explain` text
# does: adding a metric should mean adding its wording there, never editing this
# file. The module is pure data with no heavy imports, so requiring it does not
# drag torch into the light showcase deployment.
#
# Degrading instead of crashing is deliberate: the showcase's contract is that it
# renders precomputed artifacts. If it is ever deployed without the engine
# alongside, the explanations simply disappear and every number still renders.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    from verifai.core.glossary import (entries_for, explain_metric,
                                       metric_keys, pillar_of)
except ImportError:                                           # pragma: no cover
    entries_for = lambda keys: []                             # noqa: E731
    metric_keys = lambda value, prefix: []                    # noqa: E731
    explain_metric = lambda key: None                         # noqa: E731
    pillar_of = lambda key: None                              # noqa: E731


# Planned components render only when explicitly asked for. A public deploy that
# drew empty "Coverage map" cards would be advertising capability it does not
# have — the same failure `card.sample` exists to prevent, one level up.
SKELETON = os.getenv("VERIFAI_SKELETON") == "1"


def breadcrumb(trail: list[tuple[str, object]], here: str) -> None:
    """Where the reader is, and one click back to any level above it.

    The drill-down pages are hidden from the sidebar, so this is what locates
    them. Tertiary buttons read as links and stay on one line; the last element
    is the current page and is plain text, because a link to where you already
    are is noise.
    """
    with st.container(horizontal=True, gap="small", vertical_alignment="center"):
        for i, (label, go) in enumerate(trail):
            if st.button(label, type="tertiary", key=f"crumb_{i}_{label}"):
                go()
            st.markdown(":gray[›]")
        st.markdown(f"**{here}**")


def placeholder(key: str, compact: bool = False) -> bool:
    """Draw the slot a planned component will occupy. Returns whether it drew.

    One visual language for absence, deliberately: an *authored* gap (not built
    yet) and a *computed* one (this model exposes no gradients, so the metric
    cannot run) are both the app saying "this is not here, and here is why".
    Two different treatments would teach a reader that one kind of gap matters
    and the other does not.
    """
    if not SKELETON:
        return False
    from planned import PLANNED
    entry = PLANNED.get(key)
    if not entry:
        return False
    with st.container(border=True):
        st.markdown(f"**:gray[PLANNED — {entry['title']}]**")
        st.caption(entry["shows"])
        if not compact:
            st.caption(f"Blocked by: {entry['blocked_by']}  \n"
                       f"Plan: `docs/{entry['phase']}`")
    return True


# ---------- generic Plotly renderer (the extensibility trick) ----------
def _scale(spec: dict):
    """A value placed on a labeled band scale.

    Replaces the old dial gauge: a dial shows a number, this shows whether the
    number is a *good* number. Bands come from the metric, so each metric defines
    its own good/bad semantics (e.g. for MIA-AUC, low is good).
    """
    lo, hi = float(spec.get("min", 0.0)), float(spec.get("max", 1.0))
    val = spec.get("value")
    bands = spec.get("bands") or [{"to": hi, "label": "", "color": "#D9E2EC"}]

    fig = go.Figure()
    # invisible trace so the axes exist for the shapes/annotations below
    fig.add_trace(go.Scatter(x=[lo, hi], y=[0.5, 0.5], mode="markers",
                             marker=dict(opacity=0), hoverinfo="skip", showlegend=False))
    start = lo
    for b in bands:
        end = float(b.get("to", hi))
        fig.add_shape(type="rect", x0=start, x1=end, y0=0, y1=1,
                      fillcolor=b.get("color", "#D9E2EC"), line_width=0, layer="below")
        if b.get("label"):
            fig.add_annotation(x=(start + end) / 2, y=0.5, text=b["label"], showarrow=False,
                               font=dict(size=13, color="#243B53"))
        start = end

    if val is not None:
        val = float(val)
        fig.add_shape(type="line", x0=val, x1=val, y0=-0.08, y1=1.08,
                      line=dict(color="#102A43", width=4))
        # no explicit colour: this sits on the plot background, so the Streamlit
        # template must pick it or it goes invisible in dark mode. Same below.
        fig.add_annotation(x=val, y=1.42, text=f"<b>{val:g}</b>", showarrow=False,
                           font=dict(size=20))
    if spec.get("value_label"):
        fig.add_annotation(x=(lo + hi) / 2, y=-0.75, text=spec["value_label"], showarrow=False,
                           font=dict(size=12), xanchor="center", align="center")

    fig.update_xaxes(range=[lo, hi], showgrid=False, zeroline=False,
                     tickvals=spec.get("ticks", [lo, hi]),
                     ticktext=spec.get("tick_labels"))
    fig.update_yaxes(range=[-1.0, 1.8], visible=False)
    # transparent plot area so only the bands carry colour, in either theme.
    # The band labels keep an explicit dark colour because they always sit on a
    # light pastel band.
    fig.update_layout(title=spec.get("title", ""), height=210, showlegend=False,
                      plot_bgcolor="rgba(0,0,0,0)",
                      margin=dict(l=30, r=30, t=60, b=20))
    st.plotly_chart(fig, width="stretch")


def render_chart(spec: dict, base: Path):
    kind = spec.get("kind")
    title = spec.get("title", "")
    if kind == "bar":
        marker_color = spec.get("colors", spec.get("color", "#5B3FD6"))  # list = per-bar
        bar = go.Bar(x=spec["x"], y=spec["y"], marker_color=marker_color)
        if spec.get("y_lo") and spec.get("y_hi"):
            # Wilson intervals are asymmetric, so plus/minus arms differ.
            bar.error_y = dict(
                type="data", symmetric=False,
                array=[hi - y for y, hi in zip(spec["y"], spec["y_hi"])],
                arrayminus=[y - lo for y, lo in zip(spec["y"], spec["y_lo"])],
                thickness=1.4, width=6, color="#455A64")
        if spec.get("hover"):
            bar.text = spec["hover"]
            bar.hovertemplate = "%{text}<br>%{y}<extra></extra>"
        fig = go.Figure(bar)
        fig.update_layout(title=title, xaxis_title=spec.get("x_title", ""), yaxis_title=spec.get("y_title", ""))
        st.plotly_chart(fig, width="stretch")
    elif kind == "line":
        fig = go.Figure(go.Scatter(x=spec["x"], y=spec["y"], mode="lines+markers"))
        fig.update_layout(title=title, xaxis_title=spec.get("x_title", ""), yaxis_title=spec.get("y_title", ""))
        st.plotly_chart(fig, width="stretch")
    elif kind == "heatmap":
        hm = go.Heatmap(z=spec["z"], x=spec.get("x"), y=spec.get("y"), colorscale="Blues",
                        zmin=spec.get("zmin"), zmax=spec.get("zmax"))
        if spec.get("text"):
            hm.text = spec["text"]
            hm.hovertemplate = "%{text}<extra></extra>"
        fig = go.Figure(hm)
        fig.update_layout(title=title, xaxis_title=spec.get("x_title", ""),
                          yaxis_title=spec.get("y_title", ""),
                          # true classes read top-to-bottom, like a printed matrix
                          yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, width="stretch")
    elif kind in ("scale", "gauge"):   # "gauge" kept as an alias so older reports still render
        _scale(spec)
    elif kind == "images":
        paths = spec.get("paths", [])
        captions = spec.get("captions") or []
        cols = st.columns(min(3, max(1, len(paths))))
        for i, rel in enumerate(paths):
            p = base / rel
            if p.exists():
                cols[i % len(cols)].image(str(p), caption=captions[i] if i < len(captions) else None)
    else:
        st.json(spec)


# ---------- the info box: five questions, open, in the same order every time ----------
# The order is the contract (docs/extending.md, "the info box a metric must be
# able to fill"): what was measured, what came out, why it matters, how to read
# the chart, what it does not tell you. The chart sits between the fourth and the
# fifth, so the reading instructions come right before the thing they explain.
#
# Everything is rendered open. The reader this is written for has never seen a
# Responsible-AI report, and the settled rule is progressive disclosure by depth
# on the page, never by click: an expander is a decision the reader has to make
# before knowing what is inside, and it hid exactly the two answers — how to
# read it, what it cannot tell you — a non-specialist most needs.
INFO_BOX_ORDER = ("what", "summary", "impact", "how", "limits")
INFO_BOX_LABEL = {
    "what": "What was measured",
    "summary": "What came out",
    "impact": "Why it matters",
    "how": "How to read the chart",
    "limits": "What it does not tell you",
}


def _label(key: str):
    st.markdown(f"**:gray[{INFO_BOX_LABEL[key]}]**")


def render_finding(finding: dict, base: Path):
    """One finding: its status, its five-question box, its charts, its definitions."""
    verdict = normalise_verdict(finding.get("verdict"), finding.get("pillar"))
    icon, label, meaning = VERDICT[verdict]
    st.markdown(f"##### {icon} `{finding['metric']}` · {label}", help=meaning)

    ex = (finding.get("details") or {}).get("explain") or {}
    details = finding.get("details") or {}

    if ex.get("what"):
        _label("what")
        st.markdown(ex["what"])
    if finding.get("summary"):
        _label("summary")
        # The result carries the finding's own status icon rather than a generic
        # info glyph: standing description above, what this run produced here.
        st.info(finding["summary"], icon=icon)
    if ex.get("impact"):
        _label("impact")
        st.markdown(ex["impact"])
    else:
        placeholder("impact", compact=True)
    if ex.get("how"):
        _label("how")
        st.markdown(ex["how"])

    chart = details.get("chart")
    if chart:
        render_chart(chart, base)
    elif finding.get("plots"):
        render_chart({"kind": "images", "paths": finding["plots"]}, base)
    if details.get("chart2"):      # e.g. a faithfulness scale or a subgroup gap
        render_chart(details["chart2"], base)

    if ex.get("limits"):
        _label("limits")
        st.markdown(ex["limits"])

    # Definitions stay one click away, and only these. They are reference — what
    # a quantity *is*, the same for any model — where the box above is what
    # *this* run found. Keys are derived the way the comparison view derives
    # them, so both views define a metric with the same words.
    keys = metric_keys(finding.get("value"), finding["pillar"])
    if entries_for(keys):
        with st.expander("Definitions of the terms above"):
            render_metric_explanations(keys, compact=True)


# ---------- metric explanations ----------

# Streamlit's own colour names, one per pillar. Native markdown colours are used
# rather than CSS: `st.markdown` is sanitised with `FORBID_TAGS: ['style']`, so an
# injected <style> block is stripped and a card built on CSS classes renders as
# unstyled text — which is exactly how the first two attempts at this ended up
# invisible. Anything drawn with `:colour[...]` and `st.container(border=True)`
# cannot be sanitised away.
PILLAR_COLOR = {
    "integrity": "red",           # first, and gates everything below it
    "performance": "violet",
    "fairness": "orange",
    "robustness": "blue",
    "explainability": "gray",
    "privacy": "green",
}
_DEFAULT_COLOR = "gray"


def explanation_markdown(entry: dict, covered: list[str], color: str,
                         compact: bool = False) -> str:
    """One explanation as Streamlit markdown.

    Pure so it can be tested without a Streamlit runtime. The glossary's own
    `**bold**` markers are left alone: `st.markdown` renders them natively, so
    no escaping or HTML conversion is needed anywhere in this path.
    """
    keys = "  ".join(f"`{k}`" for k in covered)
    lines = [f"**:{color}[{entry['term']}]**  {keys}", ""]
    rows = [("Measures", entry.get("measures")),
            ("Ideal value", entry.get("ideal"))]
    if not compact:
        rows.append(("Reading it", entry.get("reading")))
    rows.append(("Trades against", entry.get("tension")))
    for label, text in rows:
        if text:
            lines.append(f"**{label}** — {text}  ")
    return "\n".join(lines)


def render_metric_explanations(keys: list[str], compact: bool = False) -> int:
    """Bordered, colour-coded cards for `keys`. Returns how many were rendered.

    Concepts are deduplicated first, so sensitivity selected for three classes
    is one card naming all three columns, not the same paragraph three times.

    `compact` drops the worked "Reading it" line. In a single report the
    finding's own `explain.how` already says how to read the number in front of
    you, so repeating a general worked example there is length without
    information. The comparison view has no finding to lean on and keeps it.
    """
    shown = 0
    for entry, covered in entries_for(keys):
        color = PILLAR_COLOR.get(pillar_of(covered[0]) or "", _DEFAULT_COLOR)
        with st.container(border=True):
            st.markdown(explanation_markdown(entry, covered, color, compact=compact))
        shown += 1
    return shown


def render_metric_legend(keys: list[str], columns: int = 2) -> int:
    """Explanations for `keys` as a legend, grouped by pillar.

    Grouped rather than listed flat because the comparison table mixes pillars,
    and a reader scanning a row is asking "what is this?" — the pillar is half
    that answer.

    Laid out in columns because a full-width card runs to roughly 200 characters
    a line, and prose stops being comfortably readable somewhere around 90. Two
    columns also halve how far the table scrolls away while the legend is open.
    """
    shown = 0
    for pillar in PILLARS:
        here = [k for k in keys if pillar_of(k) == pillar]
        grouped = entries_for(here)
        if not grouped:
            continue
        color = PILLAR_COLOR.get(pillar, _DEFAULT_COLOR)
        st.markdown(f"**:{color}[{pillar.upper()}]** · {PILLAR_QUESTION[pillar]}")
        cols = st.columns(columns)
        for i, (entry, covered) in enumerate(grouped):
            with cols[i % columns]:
                with st.container(border=True):
                    st.markdown(explanation_markdown(entry, covered, color))
            shown += 1
    return shown

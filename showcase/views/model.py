"""Model — one checkpoint: what it is, and every configuration it was evaluated in.

This is the page the gallery never had. It answers the three questions a reader
arrives with: *what is this model*, *has it been evaluated*, and *how do its
configurations differ* — and it states the model's identity rather than its
name, because a name can be reused and bytes cannot.
"""
from __future__ import annotations

import streamlit as st

from model_card import evaluation_line, reference_report, render_card, reports_of
from render import breadcrumb, placeholder
from registry import dataset_name, decision_rule, find_model, load_registry
from routing import (current, go_to_compare, go_to_overview, go_to_project,
                     go_to_report)


def by_evaluation_set(model: dict) -> list[tuple[str | None, list[dict]]]:
    """Configurations grouped by the images they were scored on.

    The model's own test set first — the one its trainer was evaluated on —
    then any other, such as an external clinic. Within a set, `argmax` before
    any weighted rule, so the model as trained reads before its variants.
    """
    home = next((c["eval_manifest"] for c in model["configurations"]
                 if c["scenario"] == model.get("trained_by")), None)
    groups: dict[str | None, list[dict]] = {}
    for c in model["configurations"]:
        groups.setdefault(c["eval_manifest"], []).append(c)
    order = sorted(groups, key=lambda m: (m != home, m or ""))
    return [(m, sorted(groups[m], key=lambda c: (bool(c["decision_weights"]), c["label"])))
            for m in order]


def _configuration_row(c: dict, report: dict | None, key: str):
    with st.container(border=True):
        left, right = st.columns([5, 1.6], vertical_alignment="center")
        with left:
            st.markdown(f"**{c['label']}**")
            st.caption(f"Decision rule: {decision_rule(c['decision_weights'])}"
                       + ("  \n:gray[Archived — kept as the record, not re-run]"
                          if c.get("status") == "archived" else "  \nActive — re-run as the metrics change"))
            # Whether the evaluation can be read, and how much of it was measured —
            # never one of its results, which would read as the model's score.
            if report is not None:
                st.caption(evaluation_line(report))
        with right:
            if report is not None:
                if st.button("Open report →", key=key):
                    go_to_report(c["scenario"])
            else:
                # A computed absence, not a placeholder: it is always shown,
                # because a configuration that exists and was never run is a fact.
                st.markdown(":gray[Not evaluated]")
                st.caption("Evaluation runs locally.")


def page():
    registry = load_registry()
    model = find_model(registry, current("model"))
    if model is None:
        st.title("No model selected")
        st.caption("Pick a model from a project on the overview.")
        if st.button("← Back to overview"):
            go_to_overview()
        return

    breadcrumb([("Overview", go_to_overview),
                (model["project"], lambda: go_to_project(model["project"]))],
               here=model["name"])
    st.title(model["name"])
    reports = reports_of(model)
    if not model.get("active"):
        st.info("**Archived model.** None of its configurations is re-run as the metrics change; "
                "its reports are the record of what was measured when they were made.", icon="📦")

    st.subheader("Model card")
    render_card(model, reference_report(model, reports))

    done = [c for c in model["configurations"] if c["scenario"] in reports]

    st.subheader("Configurations")
    st.caption("The same weights, read with a different decision rule or scored on a "
               "different set of images. Each configuration has its own report. Only "
               "configurations under the same heading were scored on the same images, so "
               "only those can be compared directly.")
    for manifest, group in by_evaluation_set(model):
        st.markdown(f"**Scored on the {dataset_name(manifest)}**",
                    help=f"Manifest: `{(manifest or 'unknown').rsplit('/', 1)[-1]}`")
        for c in group:
            _configuration_row(c, reports.get(c["scenario"]), key=f"cfg_{c['scenario']}")

    if len(done) >= 2:
        st.caption("Configurations scored on the same images can be compared directly; "
                   "those scored on different images are shown side by side but never "
                   "ranked against each other.")
        if st.button(f"Compare its {len(done)} evaluated configurations →"):
            go_to_compare(model=model["key"])

    placeholder("delta_view")

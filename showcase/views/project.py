"""Project — the models of one problem, what each is, and which are evaluated.

One row per trained checkpoint. The gallery this replaces showed evaluations and
called them models, so the same weights appeared four times under four names,
and two different checkpoints appeared under one heading. Here a row is a model,
its configurations are counted beside it, and they are one click further down.

Investigations — the learning curve, the external validation — cut *across*
models: the ISIC checkpoint has configurations in both the internal and the
external set. So they cannot be sections of this list. They are a filter.
"""
from __future__ import annotations

import re

import streamlit as st

from render import breadcrumb
from registry import (STATUS_LABEL, describe, evaluated_ids, load_registry,
                      model_status, models_in)
from routing import current, go_to_model, go_to_overview

ALL = "All"


def _investigation(group: str | None) -> str | None:
    """The short name of a gallery group: its text before the dash."""
    return group.split(" — ")[0].strip() if group else None


def natural_key(text: str) -> list:
    """Sort numbers as numbers: n=100, n=500, n=2,000 — not n=100, n=2,000, n=500."""
    parts = re.split(r"(\d[\d,]*)", text.lower())
    return [int(t.replace(",", "")) if t[:1].isdigit() else t for t in parts]


def investigations_of(model: dict, cards_by_id: dict) -> list[str]:
    found = {_investigation(cards_by_id[c["scenario"]].get("group"))
             for c in model["configurations"] if c["scenario"] in cards_by_id}
    return sorted(i for i in found if i)


def visible_models(models: list[dict], show_archived: bool) -> list[dict]:
    """Active models always; archived ones only when asked for.

    An archived model is the record of an experiment, not something to choose
    between today — so it is one switch away rather than gone, and never
    silently mixed in with the models kept current.
    """
    return [m for m in models if m.get("active") or show_archived]


def _model_row(model: dict, status: dict, invs: list[str], key: str):
    with st.container(border=True):
        left, mid, right = st.columns([5, 2, 1.4], vertical_alignment="center")
        with left:
            st.markdown(f"**{model['name']}**")
            st.caption(describe(model["provenance"])
                       or "Not trained in this repository — no training record declared.")
        with mid:
            if status["state"] == "not_evaluated":
                st.markdown(":gray[Not evaluated]")
            else:
                st.markdown(f"{status['evaluated']} of {status['total']} evaluated")
            tags = ([] if model.get("active") else ["Archived"]) + invs
            if tags:
                st.caption(" · ".join(tags))
        with right:
            if st.button("Open →", key=key):
                go_to_model(model["key"])


def page():
    from catalog import load_catalog

    registry = load_registry()
    name = current("project")
    known = [p["name"] for p in (registry or {}).get("projects", [])]
    if registry is None or name not in known:
        st.title("No project selected")
        st.caption("Pick a project on the overview.")
        if st.button("← Back to overview"):
            go_to_overview()
        return

    breadcrumb([("Overview", go_to_overview)], here=name)
    st.title(name)

    models = models_in(registry, name)
    cards_by_id = {c["id"]: c for c in load_catalog()}
    evaluated = evaluated_ids()
    configs = sum(len(m["configurations"]) for m in models)
    active = [m for m in models if m.get("active")]
    st.caption(
        f"{len(active)} active and {len(models) - len(active)} archived models · {configs} "
        f"configurations. A **model** is one trained checkpoint. A **configuration** is that "
        f"model read with one decision rule and scored on one set of images — the same weights "
        f"can have several, and each has its own report. **Active** models are re-evaluated "
        f"when the metrics change; **archived** ones are kept as the record of an experiment.")

    show_archived = st.toggle(f"Show archived models ({len(models) - len(active)})",
                              value=False, key="show_archived")
    candidates = visible_models(models, show_archived)
    invs = {m["key"]: investigations_of(m, cards_by_id) for m in models}
    options = [ALL] + sorted({i for m in candidates for i in invs[m["key"]]})
    choice = st.pills("Investigation", options, default=ALL, key="investigation_filter")
    shown = [m for m in candidates if choice in (None, ALL) or choice in invs[m["key"]]]
    if not shown:
        st.caption("No active model in this project — the archived ones are one switch away.")

    for i, model in enumerate(sorted(shown, key=lambda m: natural_key(m["name"]))):
        _model_row(model, model_status(model, evaluated), invs[model["key"]], key=f"model_{i}")

    states = [model_status(m, evaluated)["state"] for m in models]
    st.caption(" · ".join(f"{states.count(s)} {STATUS_LABEL[s].lower()}"
                          for s in ("evaluated", "partly", "not_evaluated") if states.count(s))
               + f" — of {len(models)} models in this project.")

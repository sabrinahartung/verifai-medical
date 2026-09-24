"""Overview — the projects, and the evaluations no model accounts for.

The landing page lists *problems*, not runs. Twenty-four evaluation tiles on
the front page expected a reader to know what a lineage was; one card per
project says what exists and how much of it has been evaluated, and everything
else is one click down.

`gallery` is the previous front page, kept for a deploy that has no model
registry yet — the same read-the-old-shape rule `normalise_verdict` follows.
"""
from __future__ import annotations

import streamlit as st

from catalog import PILLARS, PILLAR_QUESTION, load_snapshots
from registry import (STATUS_LABEL, evaluated_ids, load_registry, model_status,
                      models_in, unclaimed)
from routing import go_to_compare, go_to_project, go_to_report

# ---------- views ----------
DEFAULT_GROUP = "Models"


def _tile(card: dict, key: str):
    """One evaluated configuration."""
    with st.container(border=True):
        st.markdown(f"### {card.get('emoji','🧠')} {card['name']}")
        st.caption(f"Domain: {card.get('domain','?')}  ·  {card.get('dataset','')}")
        st.write(card.get("description", ""))
        if card.get("sample"):
            st.warning("SAMPLE data (placeholder until the real run lands)")
        if st.button("View analysis →", key=key):
            go_to_report(card["id"])


def _lineage_card(lineage: str, members: list[dict], key: str):
    """Several configurations of one investigation, collapsed into a single card.

    Five tiles for five decision rules on one checkpoint is a wall, not a gallery.
    The card leads with the comparison, because these exist to be read against each
    other — a single one of them in isolation is the least useful view of the set.
    """
    first = members[0]
    with st.container(border=True):
        st.markdown(f"### {first.get('emoji','🧠')} {lineage}")
        st.caption(f"Domain: {first.get('domain','?')}  ·  {first.get('dataset','')}  ·  "
                   f"**{len(members)} configurations**")
        st.write("  ·  ".join(m["name"] for m in members))
        if st.button(f"Compare {len(members)} configurations →", key=f"{key}_cmp"):
            go_to_compare(lineage)
        pick = st.selectbox("or open one", [m["name"] for m in members],
                            key=f"{key}_sel", label_visibility="collapsed")
        if st.button("View analysis →", key=f"{key}_one"):
            go_to_report(next(m["id"] for m in members if m["name"] == pick))


def gallery(cards: list[dict]):
    st.title("VERIFAI Medical — Responsible-AI Evaluation")
    # Derived from PILLARS rather than written out: this line claimed "five pillars"
    # and then listed five of the six for as long as integrity had existed.
    st.caption("Pick a model — and see its analysis across every pillar: "
               + ", ".join(p.capitalize() for p in PILLARS) + ".")

    with st.expander("What am I looking at?"):
        st.markdown(
            "Every model here has been put through the same battery of checks. Instead of a "
            "single accuracy number, each run asks five separate questions:\n\n"
            + "\n".join(f"- **{p.capitalize()}** — {PILLAR_QUESTION[p]}" for p in PILLARS)
            + "\n\nThe results were computed once by the engine in this repo and stored as "
              "files, so this page is just a reader — nothing is recomputed when you click. "
              "Every metric states what it measures and, just as importantly, when the sample "
              "is too small to support a claim."
        )

    snaps = load_snapshots()
    if snaps:
        if st.button(f"Compare all runs ({len(snaps)} recorded) →"):
            go_to_compare()

    if not cards:
        st.info("No models yet. Create one with `python scripts/run_scenario.py scenarios/skin_cancer.yaml`.")
        return

    # sections in a stable order, with placeholder data pushed to the end
    groups: dict[str, list[dict]] = {}
    for c in cards:
        groups.setdefault(c.get("group") or DEFAULT_GROUP, []).append(c)
    ordered = sorted(groups.items(), key=lambda kv: (kv[0].startswith("Demo"), kv[0]))

    for gi, (group, members) in enumerate(ordered):
        st.divider()
        st.subheader(group)

        # collapse configurations of one investigation into a single card
        lineages: dict[str, list[dict]] = {}
        for c in members:
            lineages.setdefault(c.get("lineage") or c["id"], []).append(c)

        cols = st.columns(3)
        for i, (lineage, ms) in enumerate(sorted(lineages.items())):
            with cols[i % 3]:
                if len(ms) == 1:
                    _tile(ms[0], key=f"btn_{gi}_{i}")
                else:
                    _lineage_card(lineage, sorted(ms, key=lambda m: m["name"]),
                                  key=f"lin_{gi}_{i}")


def _what_am_i_looking_at():
    with st.expander("What am I looking at?"):
        st.markdown(
            "Each **project** is one problem a set of models tries to solve. Inside it, each "
            "**model** is one trained checkpoint, and each model can be evaluated in several "
            "**configurations** — read with a different decision rule, or scored on a "
            "different set of images. Every evaluation asks the same separate questions "
            "instead of reporting a single accuracy number:\n\n"
            + "\n".join(f"- **{p.capitalize()}** — {PILLAR_QUESTION[p]}" for p in PILLARS)
            + "\n\nThe results were computed once by the engine in this repo and stored as "
              "files, so this page is just a reader — nothing is recomputed when you click. "
              "Every metric states what it measures and, just as importantly, when the sample "
              "is too small to support a claim."
        )


def _project_card(project: dict, registry: dict, evaluated: set[str], cards_by_id: dict,
                  key: str):
    models = models_in(registry, project["name"])
    states = [model_status(m, evaluated)["state"] for m in models]
    configs = sum(len(m["configurations"]) for m in models)
    # The project's most common emoji, not its first: the first model
    # alphabetically is a learning-curve run, and its 📈 is not the project's.
    emojis = [cards_by_id[c["scenario"]].get("emoji") for m in models
              for c in m["configurations"] if c["scenario"] in cards_by_id]
    emojis = [e for e in emojis if e]
    emoji = max(set(emojis), key=emojis.count) if emojis else "🧠"
    with st.container(border=True):
        st.markdown(f"#### {emoji} {project['name']}")
        st.caption(f"{len(models)} models · {configs} configurations")
        # Counts, not a colour: this says what exists, never whether it is good.
        counts = [f"**{states.count(s)}** {STATUS_LABEL[s].lower()}"
                  for s in ("evaluated", "partly", "not_evaluated") if states.count(s)]
        st.markdown(" · ".join(counts))
        if st.button("Open project →", key=key):
            go_to_project(project["name"])


def projects_overview(registry: dict, cards: list[dict]):
    st.title("VERIFAI Medical — Responsible-AI Evaluation")
    st.caption("Your models, grouped by the problem they solve. Open a project to see which "
               "models exist, which have been evaluated, and what each evaluation found.")
    _what_am_i_looking_at()

    evaluated = evaluated_ids()
    cards_by_id = {c["id"]: c for c in cards}
    # Two columns, not three: a project name is a phrase, and a third of a
    # narrow window broke "classification" mid-word.
    cols = st.columns(2)
    for i, project in enumerate(registry["projects"]):
        with cols[i % 2]:
            _project_card(project, registry, evaluated, cards_by_id, key=f"proj_{i}")

    snaps = load_snapshots()
    if snaps and st.button(f"Compare all runs ({len(snaps)} recorded) →", type="tertiary"):
        go_to_compare()

    # Evaluations no declared model accounts for are shown, never dropped: the
    # a report whose scenario was since deleted, or a placeholder fixture.
    others = unclaimed(cards, registry)
    if others:
        st.divider()
        st.subheader("Other evaluations")
        st.caption("Reports that no declared model accounts for.")
        cols = st.columns(3)
        for i, card in enumerate(sorted(others, key=lambda c: c["name"])):
            with cols[i % 3]:
                _tile(card, key=f"other_{i}")


def page():
    """Route entry: projects when a model registry exists, the old gallery otherwise."""
    from catalog import load_catalog
    cards = load_catalog()
    registry = load_registry()
    if registry is None:
        gallery(cards)
        return
    projects_overview(registry, cards)

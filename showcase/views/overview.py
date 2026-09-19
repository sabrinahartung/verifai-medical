"""Overview — which investigation, and which run inside it."""
from __future__ import annotations

import streamlit as st

from catalog import PILLARS, PILLAR_QUESTION, load_snapshots
from render import placeholder
from routing import go_to_compare, go_to_report

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
    st.title("VERIFAI — Responsible-AI Showcase")
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

    placeholder("projects")
    placeholder("portfolio_coverage")

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


def page():
    """Route entry: the gallery of everything that has been evaluated."""
    from catalog import load_catalog
    gallery(load_catalog())

"""Compare — every recorded run, grouped by the images it was scored on."""
from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go

from catalog import (_blocked_reason, comparability_key, direction_for,
                     best_run, dominated_by, group_snapshots, run_label)
from render import explain_metric, placeholder, render_metric_legend
from routing import current, go_to_overview

def comparison(snaps: list[dict], cards: list[dict] | None = None):
    lineage = current("lineage")
    st.title("Comparing runs" + (f" — {lineage}" if lineage else ""))
    st.caption("Every recorded evaluation, grouped by the exact set of images it was scored on.")

    if st.button("← Back to overview"):
        go_to_overview()

    # A lineage narrows *what is shown*; it never widens what may be compared.
    # Grouping stays keyed on the evaluation set, so two runs of one lineage scored
    # on different manifests still land in different groups.
    hidden_comparable: dict[tuple, set[str]] = {}
    if lineage and cards:
        ids = {c["id"] for c in cards if (c.get("lineage") or c["id"]) == lineage}
        kept = [s for s in snaps if s.get("scenario") in ids]
        # A filtered view still prints a `best` column, and that column ranks only
        # what is on screen. If a run scored on the *same images* is hidden, the
        # word "best" becomes false without anything saying so — which is exactly
        # how a reader concludes that the strongest configuration does not exist.
        kept_keys = {comparability_key(s) for s in kept}
        for s in snaps:
            if s.get("scenario") in ids:
                continue
            key = comparability_key(s)
            if key in kept_keys:
                hidden_comparable.setdefault(key, set()).add(
                    run_label(s))
        snaps = kept
        st.caption(f"Filtered to the {len(ids)} configuration(s) in this lineage. "
                   f"Comparability is still decided by the evaluation set, not the lineage.")

    with st.expander("Why runs are grouped, and when a comparison is refused"):
        st.markdown(
            "A difference between two numbers only means something if everything else was "
            "held equal. Two rules decide that here:\n\n"
            "1. **Same images.** Runs are grouped by the *content hash* of the evaluation "
            "manifest, not its filename — a manifest can be regenerated with a different "
            "seed and keep its name. Runs in different groups are never plotted together.\n"
            "2. **A verified split.** A run whose split was contaminated, or never checked, "
            "is excluded from the chart and listed with the reason. Its accuracy is inflated "
            "by an unknown amount, so plotting it beside an honest run would manufacture a "
            "comparison rather than report one.\n\n"
            "This is deliberately stricter than most dashboards. A green *+12 points* against "
            "a leaked baseline is exactly the claim this project exists to catch."
        )

    with st.expander("Why several runs, and how to read them"):
        st.markdown(
            "Each run is one configuration — a model, plus the decision rule that reads "
            "its probabilities. They are not competitors in a race with a winner; "
            "together they map a **trade-off**.\n\n"
            "Two things here *are* decidable from the data:\n\n"
            "- **Which run leads on a given metric** — shown in the `best` column.\n"
            "- **Whether a run is beaten on everything.** If another run is at least as "
            "good on every selected metric, the loser can be dropped with no judgement "
            "call at all.\n\n"
            "What is *not* decidable: which of the surviving runs is best overall. A "
            "configuration catching 94% of melanomas while misreading a third of moles "
            "is better or worse than the reverse **depending entirely on what the tool "
            "is for**. Collapsing that into one score would not resolve the question, "
            "it would only hide it."
        )

    placeholder("delta_view")

    groups = group_snapshots(snaps)
    if not groups:
        st.info("No runs recorded yet. Every `run_scenario.py` invocation writes one.")
        return

    order = sorted(groups.items(), key=lambda kv: -len(kv[1]))
    for gi, (key, runs) in enumerate(order):
        ev = runs[0].get("eval_set") or {}
        name = (ev.get("manifest") or "unknown evaluation set").split("/")[-1]
        st.subheader(f"{name}  ·  {ev.get('n') or '?'} images")
        st.caption(f"content hash `{ev.get('sha256') or 'none'}`  ·  {len(runs)} run(s)")

        also = sorted(hidden_comparable.get(key, ()))
        if also:
            st.warning(
                f"**{len(also)} further run(s) were scored on these same images** and are "
                f"hidden by the lineage filter: {', '.join(f'*{a}*' for a in also)}. "
                f"The **best** column below ranks only what is shown, so it may not name "
                f"the strongest configuration you have. Use **Compare all runs** from the "
                f"overview to see them together.", icon="🔎")

        placeholder("access_statement", compact=True)

        usable = [r for r in runs if _blocked_reason(r) is None]
        blocked = [(r, _blocked_reason(r)) for r in runs if _blocked_reason(r) is not None]

        # Re-running the same configuration records another snapshot, which is the
        # point of a history — but three identical rows help nobody read a table.
        # Collapse to the newest per configuration, with the full record a click away.
        by_label: dict[str, list[dict]] = {}
        for r in sorted(usable, key=lambda r: r.get("created_at", "")):
            by_label.setdefault(run_label(r), []).append(r)
        repeats = sum(len(v) - 1 for v in by_label.values())
        if repeats and not st.checkbox(
                f"Show every recorded run ({repeats} repeat(s) of a configuration hidden)",
                key=f"all_{gi}"):
            usable = [v[-1] for v in by_label.values()]

        for r, why in blocked:
            st.warning(f"**{r.get('label', r.get('scenario'))}** is excluded — {why}.")

        if len(usable) < 2:
            st.info(
                "Nothing to compare yet in this group: "
                + ("no run here has a verified split." if not usable
                   else "only one comparable run so far. Train a variant and re-run it "
                        "against this same manifest.")
            )
            st.divider(); continue

        keys = sorted({k for r in usable for k in r["metrics"]})
        default = [k for k in ("performance.per_class.melanoma.sensitivity",
                               "performance.per_class.melanoma.ppv_test_prevalence",
                               "performance.accuracy", "performance.balanced_accuracy",
                               "performance.top3_accuracy", "robustness.mean_stability",
                               "fairness.accuracy_gap", "privacy.mia_auc") if k in keys]
        chosen = st.multiselect("Metrics to compare", keys, default=default or keys[:6],
                                key=f"ms_{gi}")
        # Right above the table it annotates, so the reader can see what it does
        # without hunting for it. Off by default: someone who knows the terms
        # should not have to scroll past their definitions.
        explaining = st.toggle(
            "Explain these metrics", value=False, key=f"exp_{gi}",
            help="Adds a colour-coded legend under the table: what each metric "
                 "measures, its ideal value, how to read it, what it trades against.")
        if not chosen:
            st.divider(); continue

        dirs = {k: direction_for(k, usable) for k in chosen}
        labels = [run_label(r) for r in usable]

        # --- is there an outright winner, or is this a trade-off? ---
        dom = dominated_by(usable, chosen, dirs)
        survivors = [l for l in labels if l not in dom]
        if len(survivors) == 1 and len(labels) > 1:
            st.success(f"**{survivors[0]}** is at least as good as every other run on all "
                       f"selected metrics. On this evidence it is the one to keep.")
        else:
            st.info(
                f"**No single best run.** {len(survivors)} of {len(labels)} runs trade off "
                "against each other: each is better on some selected metric and worse on "
                "another. Which one is *right* depends on what the model is for — a triage "
                "tool and a rule-out tool want opposite ends of this table. That is a "
                "decision about intended use, not one the data can settle."
            )
        for loser, winner in dom.items():
            st.caption(f"↳ **{loser}** is beaten by *{winner}* on every selected metric, "
                       f"so it can be dismissed without a value judgement.")

        # --- table: value, direction, and who leads each metric ---
        rows = []
        for k in chosen:
            d = dirs[k]
            arrow = {"higher": "↑ better", "lower": "↓ better"}.get(d, "—")
            leader = best_run(k, usable, d)
            # The key alone is a developer string: reading
            # `..._ppv_test_prevalence` as "how many did it catch" is the exact
            # misreading this table invites. When explanations are on, lead with
            # the name the concept is actually known by.
            row = {"metric": k, "good": arrow}
            if explaining:
                entry = explain_metric(k)
                row = {"term": entry["term"] if entry else "—", **row}
            for r in usable:
                lab = run_label(r)
                v = r["metrics"].get(k)
                row[lab] = None if v is None else round(v, 4)
            row["best"] = leader or "—"
            rows.append(row)
        st.dataframe(rows, width="stretch")
        st.caption("**best** is only filled in where the metric declared which direction is "
                   "an improvement. An undeclared metric is shown but not ranked.")

        if explaining:
            st.markdown("###### Legend — what the rows above mean")
            st.caption("General definitions: the same wording applies to any model on any "
                       "dataset. Colour marks the pillar each metric belongs to.")
            if not render_metric_legend(chosen):
                st.caption("No explanations available for the selected metrics.")

        # --- the trade-off, seen directly ---
        ranked_keys = [k for k in chosen if dirs.get(k)]
        if len(ranked_keys) >= 2:
            c1, c2 = st.columns(2)
            xk = c1.selectbox("Trade-off: x", ranked_keys, index=0, key=f"x_{gi}")
            yk = c2.selectbox("Trade-off: y", ranked_keys,
                              index=min(1, len(ranked_keys) - 1), key=f"y_{gi}")
            fig = go.Figure()
            for r in usable:
                lab = run_label(r)
                fig.add_trace(go.Scatter(
                    x=[r["metrics"].get(xk)], y=[r["metrics"].get(yk)], mode="markers+text",
                    text=[lab], textposition="top center", name=lab, marker=dict(size=14)))
            fig.update_layout(
                title=f"{yk} against {xk} — each point is one run",
                xaxis_title=f"{xk} ({dirs.get(xk, '')} is better)",
                yaxis_title=f"{yk} ({dirs.get(yk, '')} is better)", showlegend=False)
            st.plotly_chart(fig, width="stretch")

        metric = st.selectbox("Bar chart", chosen, key=f"sb_{gi}")
        leader = best_run(metric, usable, dirs.get(metric))
        fig = go.Figure(go.Bar(
            x=labels, y=[r["metrics"].get(metric) for r in usable],
            marker_color=["#2E9E5B" if l == leader else "#5B3FD6" for l in labels]))
        fig.update_layout(title=f"{metric}"
                                + (f"  ·  best: {leader}" if leader else "  ·  unranked"),
                          xaxis_title="Run", yaxis_title=metric)
        st.plotly_chart(fig, width="stretch")
        st.divider()

    if len(order) > 1:
        st.error(
            f"**{len(order)} groups above were not compared with each other.** They were "
            "scored on different sets of images, so a difference between them would measure "
            "the datasets, not the models."
        )


def page():
    """Route entry: every recorded run, grouped by the images it was scored on."""
    from catalog import load_catalog, load_snapshots
    comparison(load_snapshots(), load_catalog())

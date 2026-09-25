"""Compare — every recorded run, grouped by the images it was scored on."""
from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go

from catalog import (_blocked_reason, comparability_key, direction_for,
                     best_run, dominated_by, group_snapshots, run_label)
from registry import dataset_name, find_model, load_registry
from render import GLOSSARY, breadcrumb, explain_metric, placeholder, render_metric_legend
from routing import current, go_to_compare, go_to_model, go_to_overview, go_to_project


def plural(n: int, word: str, many: str | None = None) -> str:
    """`1 run`, `3 runs` — never `3 run(s)`."""
    return f"{n:,} {word if n == 1 else (many or word + 's')}"


def scope_ids(cards: list[dict] | None, registry: dict | None,
              lineage: str | None = None, model: str | None = None
              ) -> tuple[set[str] | None, str | None]:
    """Which scenarios a filtered comparison shows, and what to call the filter.

    `(None, None)` is unfiltered. A model's scope is its configurations — the
    same weights, read and scored differently — and a lineage's is whatever its
    cards declare. Either way the filter only narrows what is *shown*: the
    groups below stay keyed on the evaluation set, and runs the filter hides
    from a group are disclosed.
    """
    if model:
        m = find_model(registry, model)
        if m:
            return {c["scenario"] for c in m["configurations"]}, m["name"]
    if lineage and cards:
        return {c["id"] for c in cards if (c.get("lineage") or c["id"]) == lineage}, lineage
    return None, None


def comparison(snaps: list[dict], cards: list[dict] | None = None):
    registry = load_registry()
    statuses = {c["scenario"]: c.get("status", "archived") for m in (registry or {}).get("models", [])
                for c in m["configurations"]} if registry else None
    model_key = current("model")
    ids, scope = scope_ids(cards, registry, lineage=current("lineage"), model=model_key)
    model = find_model(registry, model_key) if model_key else None

    if model and ids is not None:
        breadcrumb([("Overview", go_to_overview),
                    (model["project"], lambda: go_to_project(model["project"])),
                    (model["name"], lambda: go_to_model(model["key"]))],
                   here="Compare")
    st.title("Comparing runs" + (f" — {scope}" if scope else ""))
    st.caption("Every recorded evaluation, grouped by the exact set of images it was scored on.")

    if not model and st.button("← Back to overview"):
        go_to_overview()

    # A filter narrows *what is shown*; it never widens what may be compared.
    # Grouping stays keyed on the evaluation set, so two runs of one lineage or
    # one model scored on different manifests still land in different groups.
    hidden_comparable: dict[tuple, set[str]] = {}
    if ids is not None:
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
        # A filter can outlive the click that set it — the sidebar reopens this
        # page with the last one still applied — so it is always stated, and
        # always one click from undone.
        st.caption(f"Filtered to the {plural(len(ids), 'configuration')} of *{scope}*. "
                   f"Comparability is still decided by the evaluation set, not by this filter.")
        if st.button("Show all runs"):
            go_to_compare()

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
            "- **Which run leads on a given metric** — its cell is bold and tinted.\n"
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
        n_images = ev.get("n")
        st.subheader(f"{dataset_name(ev.get('manifest'))}  ·  "
                     + (plural(n_images, "image") if n_images else "? images"),
                     help=f"Manifest `{(ev.get('manifest') or 'unknown').split('/')[-1]}` · "
                          f"fingerprint `{ev.get('sha256') or 'none'}`. Runs are grouped by the "
                          f"fingerprint, a hash of the manifest's content, not by its name.")

        also = sorted(hidden_comparable.get(key, ()))
        if also:
            st.warning(
                f"**{plural(len(also), 'further run')} "
                f"{'was' if len(also) == 1 else 'were'} scored on these same images** and "
                f"{'is' if len(also) == 1 else 'are'} hidden by this filter: "
                f"{', '.join(f'*{a}*' for a in also)}. "
                f"The leading cells below rank only what is shown, so they may not name "
                f"the strongest configuration you have. **Show all runs**, above, puts "
                f"them back.", icon="🔎")

        placeholder("access_statement", compact=True)

        usable = [r for r in runs if _blocked_reason(r) is None]
        blocked = [(r, _blocked_reason(r)) for r in runs if _blocked_reason(r) is not None]
        usable_scenarios = {r.get("scenario") for r in usable}

        # Re-running the same configuration records another snapshot, which is the
        # point of a history — but three identical rows help nobody read a table.
        # Collapse to the newest per configuration, with the full record a click away.
        by_label: dict[str, list[dict]] = {}
        for r in sorted(usable, key=lambda r: r.get("created_at", "")):
            by_label.setdefault(run_label(r), []).append(r)
        repeats = sum(len(v) - 1 for v in by_label.values())
        # One sentence for the counts, so "31 runs", "14 repeats" and "17 runs"
        # are visibly the same runs counted three ways, not three claims.
        st.caption(f"{plural(len(runs), 'recorded run')} of "
                   f"{plural(len({r.get('scenario') for r in runs}), 'configuration')}"
                   + (f". The table shows the newest run of each; "
                      f"{plural(repeats, 'earlier repeat')} {'is' if repeats == 1 else 'are'} "
                      f"hidden unless you ask for them." if repeats else "."))
        if repeats and not st.checkbox(
                f"Also show the {plural(repeats, 'earlier repeat')}", key=f"all_{gi}"):
            usable = [v[-1] for v in by_label.values()]

        # Excluded runs are named the way the table names runs, and dated. Most
        # are earlier snapshots of a configuration that has a newer, verified run
        # on screen; naming those by raw model id, undated, read as though the
        # charted run itself had been thrown out.
        by_scenario: dict[str, list[tuple[dict, str]]] = {}
        for r, why in blocked:
            by_scenario.setdefault(r.get("scenario", "?"), []).append((r, why))
        for scen, entries in by_scenario.items():
            rs = [r for r, _ in entries]
            why = entries[-1][1]
            dates = sorted({(r.get("created_at") or "")[:10] for r in rs if r.get("created_at")})
            when = (dates[0] if len(dates) == 1 else f"{dates[0]} to {dates[-1]}") if dates else "undated"
            label = run_label(rs[-1])
            if scen in usable_scenarios:
                st.caption(f"↳ {plural(len(rs), 'earlier run')} of **{label}** ({when}) left out — {why}. "
                           f"A later, verified run of the same configuration is included.")
            else:
                st.warning(f"**{label}** ({when}) is excluded — {why}.")

        if len(usable) < 2:
            st.info(
                "Nothing to compare yet in this group: "
                + ("no run here has a verified split." if not usable
                   else "only one comparable run so far. Train a variant and re-run it "
                        "against this same manifest.")
            )
            st.divider(); continue

        keys = sorted({k for r in usable for k in r["metrics"]})
        names = metric_labels(keys)
        default = [k for k in ("performance.per_class.melanoma.sensitivity",
                               "performance.per_class.melanoma.ppv_test_prevalence",
                               "performance.accuracy", "performance.balanced_accuracy",
                               "performance.top3_accuracy", "robustness.mean_stability",
                               "fairness.accuracy_gap", "privacy.mia_auc") if k in keys]
        chosen = st.multiselect("Metrics to compare", keys, default=default or keys[:6],
                                format_func=lambda k: names[k], key=f"ms_{gi}")
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
                f"**No single best run.** {len(survivors)} of the {len(labels)} runs shown trade off "
                "against each other: each is better on some selected metric and worse on "
                "another. Which one is *right* depends on what the model is for — a triage "
                "tool and a rule-out tool want opposite ends of this table. That is a "
                "decision about intended use, not one the data can settle."
            )

        # --- table: one row per run, one column per metric ---
        # Runs are rows because runs are what grows: seventeen of them as columns
        # pushed the `best` column off the right edge, and the metrics — the axis
        # the reader chooses — became the one that could not scroll.
        table, leaders = comparison_table(usable, chosen, dirs, dom,
                                          dated=bool(repeats) and len(usable) > len(by_label),
                                          statuses=statuses)
        st.dataframe(style_leaders(table, leaders), hide_index=True, width="stretch",
                     column_config=table_columns(chosen, names, dirs))
        st.caption("One row per run. A **bold, tinted** cell leads its column — only where the "
                   "metric declares which direction is better (↑ higher, ↓ lower); a column "
                   "without an arrow is shown but not ranked. The last column names a run "
                   "that beats this one on every selected metric: such a run can be dismissed "
                   "without a value judgement. Hover a column name for its definition; click "
                   "it to sort.")

        if explaining:
            st.markdown("###### Legend — what the columns above mean")
            st.caption("General definitions: the same wording applies to any model on any "
                       "dataset. Colour marks the pillar each metric belongs to.")
            if not render_metric_legend(chosen):
                st.caption("No explanations available for the selected metrics.")

        # --- the trade-off, seen directly ---
        ranked_keys = [k for k in chosen if dirs.get(k)]
        if len(ranked_keys) >= 2:
            c1, c2 = st.columns(2)
            xk = c1.selectbox("Trade-off: x", ranked_keys, index=0,
                              format_func=lambda k: names[k], key=f"x_{gi}")
            yk = c2.selectbox("Trade-off: y", ranked_keys, index=min(1, len(ranked_keys) - 1),
                              format_func=lambda k: names[k], key=f"y_{gi}")
            st.plotly_chart(tradeoff_figure(usable, xk, yk, dirs, names), width="stretch")
            st.caption("Labelled points are the runs no other run beats on both axes at once, "
                       "joined by the line. A grey point is beaten on both by at least one "
                       "other run — hover it for its name. Which labelled point is right "
                       "depends on what the model is for.")

        metric = st.selectbox("Bar chart", chosen, format_func=lambda k: names[k],
                              key=f"sb_{gi}")
        leader = best_run(metric, usable, dirs.get(metric))
        fig = go.Figure(go.Bar(
            x=labels, y=[r["metrics"].get(metric) for r in usable],
            marker_color=["#2E9E5B" if l == leader else "#5B3FD6" for l in labels]))
        fig.update_layout(title=names[metric]
                                + (f"  ·  {'highest' if dirs.get(metric) == 'higher' else 'lowest'}: "
                                   f"{leader}" if leader else "  ·  not ranked"),
                          xaxis_title="Run", yaxis_title=names[metric])
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


# ---------- the table and the trade-off, as pure functions ----------
ARROW = {"higher": "↑", "lower": "↓"}
LEADER_STYLE = "font-weight: 700; background-color: rgba(46, 158, 91, 0.22)"
BEATEN_BY = "Beaten on every selected metric by"


def metric_labels(keys: list[str]) -> dict[str, str]:
    """The name each flattened key is known by, unique across `keys`.

    The glossary's term, plus whatever its pattern's wildcards matched — the
    class in `performance.per_class.*.sensitivity` — so melanoma's sensitivity
    and a mole's do not share a header. Two keys that still collide (the metric
    records class size under two paths) keep their raw key beside the name.
    """
    import fnmatch
    out = {}
    for key in keys:
        name = key
        for pattern, entry in GLOSSARY:
            if key == pattern or fnmatch.fnmatch(key, pattern):
                ks, ps = key.split("."), pattern.split(".")
                wild = [k for k, p in zip(ks, ps) if "*" in p] if len(ks) == len(ps) else []
                name = entry["term"] + ("".join(f" · {w}" for w in wild))
                break
        out[key] = name
    counts: dict[str, int] = {}
    for name in out.values():
        counts[name] = counts.get(name, 0) + 1
    return {k: (n if counts[n] == 1 else f"{n} ({k})") for k, n in out.items()}


def comparison_table(runs: list[dict], keys: list[str], dirs: dict[str, str | None],
                     dominated: dict[str, str], dated: bool = False,
                     statuses: dict[str, str] | None = None):
    """(DataFrame with one row per run, {key: leading run's label or None})."""
    import pandas as pd
    data: dict[str, list] = {"Run": [run_label(r) for r in runs]}
    if statuses is not None:
        # An archived run is measured with the metrics of its day; shown beside
        # an active one it says so, rather than passing for a current result.
        data["Status"] = [statuses.get(r.get("scenario"), "—").capitalize() for r in runs]
    if dated:
        data["Recorded"] = [(r.get("created_at") or "")[:10] for r in runs]
    for k in keys:
        data[k] = [None if r["metrics"].get(k) is None else round(r["metrics"][k], 4)
                   for r in runs]
    data[BEATEN_BY] = [dominated.get(lab, "") for lab in data["Run"]]
    leaders = {k: best_run(k, runs, dirs.get(k)) for k in keys}
    return pd.DataFrame(data), leaders


def style_leaders(table, leaders: dict[str, str | None]):
    """Bold and tint the leading cell of every ranked column; never an unranked one."""
    def mark(col):
        lead = leaders.get(col.name)
        return [LEADER_STYLE if lead and run == lead else "" for run in table["Run"]]
    def fmt(k):
        # Three decimals, like every other number in the showcase; counts print
        # as counts rather than as 1493.000.
        vals = table[k].dropna()
        return "{:,.0f}" if len(vals) and (vals == vals.round()).all() and vals.abs().max() > 1 \
            else "{:.3f}"
    return (table.style.apply(mark, axis=0)
            .format({k: fmt(k) for k in leaders}, na_rep="—"))


def table_columns(keys: list[str], names: dict[str, str], dirs: dict[str, str | None]) -> dict:
    cols = {"Run": st.column_config.TextColumn("Run", pinned=True),
            "Status": st.column_config.TextColumn(
                "Status", help="Active: re-run as the metrics change. Archived: the record of an "
                               "experiment, evaluated with the metrics of its day."),
            BEATEN_BY: st.column_config.TextColumn(
                BEATEN_BY, help="Another run at least as good on every selected metric and "
                                "better on one. Empty means this run is part of the trade-off.")}
    for k in keys:
        entry = explain_metric(k) or {}
        d = dirs.get(k)
        rank = (f"{d.capitalize()} is better." if d else
                "Not ranked: this metric declares no direction, and none is inferred from "
                "its name.")
        # The arrow leads: a long name is cut off at the column's edge, and the
        # direction is the part the reader needs to rank by.
        cols[k] = st.column_config.NumberColumn(
            f"{ARROW.get(d, '')} {names[k]}".strip(),
            help=f"`{k}` — {entry.get('measures', 'No definition recorded.')} {rank}")
    return cols


def pareto_front(points: list[tuple[str, float | None, float | None]],
                 dx: str, dy: str) -> set[str]:
    """Labels of the points no other point beats on both axes.

    A point is beaten when another is at least as good on x and on y and better
    on one, in each axis's declared direction. That is decidable from the data;
    choosing among the points that are left is not.
    """
    def better_or_equal(a, b, d):
        return a >= b if d == "higher" else a <= b
    pts = [(l, x, y) for l, x, y in points if x is not None and y is not None]
    front = set()
    for la, xa, ya in pts:
        beaten = any(better_or_equal(xb, xa, dx) and better_or_equal(yb, ya, dy)
                     and (xb, yb) != (xa, ya)
                     for lb, xb, yb in pts if lb != la)
        if not beaten:
            front.add(la)
    return front


def label_offsets(labelled: list[tuple[str, float, float]], dx: str, dy: str,
                  span: float) -> list[tuple[int, int]]:
    """Pixel offset (ax, ay) of each frontier label from its point.

    Labels go into the quadrant *better* on both axes. On a trade-off frontier
    that quadrant is empty by definition — no run beats a frontier point on both
    — so a label there can cover no point. Only labels can still collide, when
    neighbours sit close in x (three runs within 0.05 on the real data): those
    are pushed one step further out along the diagonal, alternately.
    """
    sx = 1 if dx == "higher" else -1               # plotly: positive ax is right
    sy = -1 if dy == "higher" else 1               # plotly: negative ay is up
    out, level, prev_x = [], 0, None
    for _, x, _y in labelled:
        close = prev_x is not None and abs(x - prev_x) / (span or 1.0) < 0.08
        level = (level + 1) % 3 if close else 0
        d = 18 + 22 * level
        out.append((sx * d, sy * d))
        prev_x = x
    return out


def tradeoff_figure(runs, xk, yk, dirs, names):
    pts = [(run_label(r), r["metrics"].get(xk), r["metrics"].get(yk)) for r in runs]
    front = pareto_front(pts, dirs[xk], dirs[yk])
    on = sorted((p for p in pts if p[0] in front and p[1] is not None), key=lambda p: p[1])
    off = [p for p in pts if p[0] not in front and p[1] is not None and p[2] is not None]
    fig = go.Figure()
    if off:
        fig.add_trace(go.Scatter(
            x=[p[1] for p in off], y=[p[2] for p in off], mode="markers",
            marker=dict(size=10, color="rgba(130, 130, 130, 0.55)"),
            hovertext=[p[0] for p in off], hoverinfo="text", name="beaten on both axes"))
    fig.add_trace(go.Scatter(
        x=[p[1] for p in on], y=[p[2] for p in on], mode="lines+markers",
        line=dict(dash="dot", width=1.5, color="#2E9E5B"),
        marker=dict(size=14, color="#2E9E5B"),
        hovertext=[p[0] for p in on], hoverinfo="text",
        name="the trade-off"))
    xs = [p[1] for p in pts if p[1] is not None]
    span = (max(xs) - min(xs)) if xs else 1.0
    for (label, x, y), (ax, ay) in zip(on, label_offsets(on, dirs[xk], dirs[yk], span)):
        fig.add_annotation(x=x, y=y, text=label, showarrow=True, arrowhead=0,
                           arrowwidth=1, arrowcolor="rgba(46, 158, 91, 0.6)",
                           ax=ax, ay=ay, xanchor="left" if ax > 0 else "right",
                           font=dict(size=12))
    fig.update_layout(
        title=f"{names[yk]} against {names[xk]} — each point is one run",
        xaxis_title=f"{names[xk]} ({dirs[xk]} is better)",
        yaxis_title=f"{names[yk]} ({dirs[yk]} is better)", showlegend=False,
        # room for the labels that lean out past the last point
        height=520, margin=dict(r=170, t=90))
    return fig

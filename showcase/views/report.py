"""Report — one configuration's evaluation, pillar by pillar.

Laid out for the reader this project settled on: someone who has never seen a
Responsible-AI report and will spend twenty minutes, not five. Two rules follow
from that (docs/ui-ux-design.md, "Two settled decisions"):

* **The first screen says what was found.** "At a glance" is one card per
  pillar, in fixed order: its status, its result in the engine's own words, and
  what it was compared with — marked **established** where the interval clears
  that reference. Each card links down to its section.
* **Depth by scrolling, never by clicking.** Each finding's five-question box is
  rendered open; only the reference definitions sit behind a click.

Integrity is not one pillar among six. Every other number is conditional on the
split being clean, so when it could not be verified — or was contaminated — the
report says so in a banner above everything else, in the finding's own words.
"""
from __future__ import annotations

import json

import streamlit as st

from catalog import (ACCESS_LABEL, COVERAGE, PILLARS, PILLAR_QUESTION, VERDICT, VERDICT_ORDER,
                     normalise_verdict)
from registry import (dataset_name, decision_rule, is_archived, load_registry,
                      model_of_scenario, out_of_date)
from render import breadcrumb, metric_name, placeholder, render_finding
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
    n = ev.get("n") or meta.get("sample_size")
    if card.get("description"):
        st.caption(card["description"])
    if model and config:
        st.markdown(
            f"A configuration of **{model['name']}** · decision rule: "
            f"{decision_rule(config['decision_weights'])} · scored on the "
            f"{dataset_name(ev.get('manifest'))}" + (f" ({n:,} images)" if n else ""))
    else:
        # An evaluation no registered model claims keeps the identifiers it has.
        st.markdown(f"**Model:** `{report['model_id']}` · **Dataset:** `{report['dataset_id']}`"
                    + (f" ({n:,} images)" if n else ""))
    if meta.get("access"):
        _provenance_strip(meta)
    if card.get("hf_url"):
        st.markdown(f"[🤗 Model on Hugging Face]({card['hf_url']})")


def weights_line(checkpoint: dict) -> str | None:
    """Which weights produced this report, in the form a reader can look up."""
    if checkpoint.get("sha256"):
        return f"weights `{checkpoint['path'].rsplit('/', 1)[-1]}` · fingerprint `{checkpoint['sha256']}`"
    if checkpoint.get("repo_id"):
        rev = checkpoint.get("revision")
        return (f"weights from the Hub, `{checkpoint['repo_id']}`"
                + (f" at `{rev}`" if rev else " — revision **unpinned**"))
    return None


def _provenance_strip(meta: dict):
    """One line on what was evaluated and how much of it the evaluation could reach."""
    bits = [f"Task: {meta.get('task', 'classification')}",
            f"the evaluation could reach the model's "
            f"**{ACCESS_LABEL.get(meta['access'], meta['access'])}**"]
    w = weights_line(meta.get("checkpoint") or {})
    if w:
        bits.append(w)
    pre = (meta.get("model") or {}).get("preprocessing_sha256")
    if pre:
        bits.append(f"preprocessing `{pre}`")
    st.caption(" · ".join(bits),
               help="How much of the model an evaluation can reach decides which metrics can "
                    "run at all: a model that can only be queried has no gradients, so Grad-CAM "
                    "cannot run on it — and says so rather than disappearing. An unpinned Hub "
                    "revision can change under the same name. The preprocessing fingerprint is "
                    "a hash of how images were resized and normalised; a different one means "
                    "the model was not scored the way it was trained.")


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


def reference_line(f: dict, hits: list[dict] | None, integrity: str) -> str | None:
    """One line under a pillar's result: what it was compared with, and the outcome.

    `None` where there is nothing to say — no baseline (a report from before
    baselines), or an integrity finding on a split the banner already calls
    unusable. The wording never grades: established or not is a statement about
    the evidence, in either direction.
    """
    b = (f.get("details") or {}).get("baseline")
    if not b:
        return None
    against = REFERENCE_KIND.get(b.get("kind"), "")
    if hits is not None and any(h is f for h in hits):
        # A claim the result sentence already contains (a clean split says the
        # same thing twice) is replaced by what it was compared with.
        repeats = b["claim"].rstrip(".").lower() in (f.get("summary") or "").lower()
        said = b["basis"] if repeats else b["claim"]
        # On a split nobody could check, the claim is still true of the
        # measurement, but the measurement may include memory. Mark and caveat
        # are one phrase, so they cannot be read apart. Integrity's own checks
        # do not depend on the split and keep a plain mark.
        caveat = ("" if integrity == "measured" or f.get("pillar") == "integrity"
                  else ", on a split that could not be checked")
        return f"✓ **Established {against}{caveat}** — {said}"
    if f.get("pillar") == "integrity":
        return None
    if normalise_verdict(f.get("verdict"), f.get("pillar")) != "measured":
        return (f"– Compared {against}: {b['basis']}. The sample does not support a claim "
                f"yet, so nothing is established.")
    if b.get("kind") == "ideal":
        return (f"– Compared {against}: {b['basis']}. Every model falls short of an ideal, "
                f"so the gap is reported and nothing is claimed.")
    return (f"– Compared {against}: {b['basis']}. The interval does not clear it, so "
            f"nothing is established.")


MODALITY_WORD = {"pixels": "images", "tokens": "text", "rows": "tables", "audio": "audio"}


def coverage_counts(rows: list[dict]) -> dict[str, int]:
    """How many registered metrics stand where. Counts completeness, never quality."""
    out: dict[str, int] = {}
    for r in rows:
        out[r["status"]] = out.get(r["status"], 0) + 1
    return out


def _coverage_map(meta: dict):
    """What this evaluation could measure, and what it did not, with the reason.

    A completeness statement: it counts what was measured and what was not —
    never what passed, which would be a rating by another name.
    """
    rows = meta.get("coverage")
    if not rows:
        return                 # predates coverage records; the pillar cards say what exists
    counts = coverage_counts(rows)
    applicable = [r for r in rows if r["status"] != "not_applicable"]
    order = ("measured", "insufficient", "unavailable", "invalid", "mixed", "not_requested")
    label = {**{k: v[1] for k, v in VERDICT.items()}, **{k: v[1] for k, v in COVERAGE.items()},
             "mixed": "Mixed"}
    with st.container(border=True):
        st.markdown(
            f"**Coverage** — {len(applicable)} of the {len(rows)} metrics this framework has "
            f"apply to {meta.get('task', 'classification')} on "
            f"{MODALITY_WORD.get(meta.get('modality'), 'this data')}: "
            + " · ".join(f"**{counts[k]}** {label[k].lower()}" for k in order if counts.get(k)),
            help="How much of what could be measured was measured. It says nothing about "
                 "whether any result is good.")
        missing = [r for r in applicable if r["status"] not in ("measured",)]
        for r in missing:
            name = metric_name(r["finding"])
            if r["status"] == "not_requested" and r.get("reason"):
                why = f"not requested — {r['reason']}"
            elif r["status"] == "not_requested":
                why = "not requested in this run" + (
                    "" if r.get("reachable", True) else
                    f"; it could not have run anyway, as it needs {r['requires']}")
            else:
                why = label.get(r["status"], r["status"]).lower() + " — see its section below"
            st.caption(f"{r['pillar'].capitalize()} · {name}: {why}.")


def _empty_pillar(p: str, rows: list[dict] | None) -> str:
    """What to say for a pillar with no finding, from the coverage record if there is one."""
    mine = [r for r in (rows or []) if r["pillar"] == p]
    if not mine:
        return ":gray[Not evaluated in this run.]"
    if all(r["status"] == "not_applicable" for r in mine):
        return f":gray[{COVERAGE['not_applicable'][0]} Not applicable to this task.]"
    skipped = [r for r in mine if r["status"] == "not_requested"]
    reasons = [r["reason"] for r in skipped if r.get("reason")]
    if reasons:
        # The scenario said why, and that is the sentence the reader needs.
        return (f":gray[{COVERAGE['not_requested'][0]} Not requested: "
                + "; ".join(reasons) + ".]")
    names = ", ".join(metric_name(r["finding"]) for r in skipped)
    return (f":gray[{COVERAGE['not_requested'][0]} Not requested in this run"
            + (f", though {names} would apply." if names else ".") + "]")


def _pillar_cards(by_pillar: dict[str, list], findings: list[dict], integrity: str,
                  coverage_rows: list[dict] | None = None):
    """Every pillar once, in fixed order: its status, its result, its reference.

    This replaces two lists that named every pillar one after the other — what
    was established, then every result — and read as the same list twice. What
    was established is now a mark on the pillar's own card, so a clean split and
    a 33-point fairness gap still stand side by side, in pillar order, never
    ranked by how good or bad they are.
    """
    hits = established(findings, integrity)
    st.subheader("At a glance", anchor="at-a-glance")
    st.caption("Every pillar's result in the engine's own words, in fixed order. A status says "
               "what is known about a number — never whether it is good. **✓ Established** "
               "marks a result whose interval clears a stated reference: an ideal, chance, or a "
               "control measured in the same run. Select a pillar to jump to its section.")
    for p in PILLARS:
        items = by_pillar.get(p, [])
        with st.container(border=True):
            name, result = st.columns([1.3, 5])
            with name:
                st.markdown(f"**[{p.capitalize()}](#{p})**")
                st.caption(PILLAR_QUESTION[p])
            with result:
                if not items:
                    st.markdown(_empty_pillar(p, coverage_rows))
                for f in items:
                    icon, label, meaning = VERDICT[normalise_verdict(f.get("verdict"), p)]
                    st.markdown(f"{icon} **{label}** — {f.get('summary', '')}", help=meaning)
                    line = reference_line(f, hits, integrity)
                    if line:
                        st.caption(line)


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
    placeholder("intended_use", compact=True)

    findings = report["findings"]
    state, decided_by = integrity_state(findings)
    _integrity_banner(state, decided_by)

    n = (report.get("meta") or {}).get("sample_size")
    if n:
        st.caption(f"Everything below was computed on {n:,} {'image' if n == 1 else 'images'}. Small samples are marked as "
                   f"such. Nothing here is scored against a threshold: the numbers and their "
                   f"intervals are reported, and what counts as good enough is your call.")

    _coverage_map(report.get("meta") or {})

    by_pillar: dict[str, list] = {p: [] for p in PILLARS}
    for f in findings:
        by_pillar.setdefault(f["pillar"], []).append(f)
    _pillar_cards(by_pillar, findings, state, (report.get("meta") or {}).get("coverage"))

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

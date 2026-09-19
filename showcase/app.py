"""VERIFAI Showcase — Streamlit entry point.

UX: an overview of everything evaluated -> click a model -> a Responsible-AI
report that a non-specialist can actually read: every metric states what it
measures, what the chart shows, and what it cannot tell you.

Extensible by design: every run is a folder under ./artifacts/<id>/ with
  card.json    -> tile metadata (name, domain, description, hf_url, sample?)
  report.json  -> the Findings produced by the engine (verifai.export.artifacts)
  plots/       -> optional images (e.g. Grad-CAM overlays)
Add a folder -> a new tile appears. No code change needed.

The explanatory text travels *with* the finding (details["explain"]), so a new
metric brings its own wording and still needs no change here.

Results are PRECOMPUTED (run once by the engine) so this app stays free &
always-on.

This file is now routing and re-exports only; the code lives in `catalog.py`
(data), `render.py` (drawing) and `views/` (the pages). The public names stay
importable from here because the contract tests import them, and because the
split is meant to be invisible from outside.

Set VERIFAI_SKELETON=1 to render the planned-but-unbuilt components in place:

    VERIFAI_SKELETON=1 .venv/bin/streamlit run showcase/app.py
"""
from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="VERIFAI Showcase — Responsible AI", layout="wide")

# Re-exported so `import app` keeps working for anything outside this package.
# noqa: F401 throughout — these are the module's public surface, not dead code.
from catalog import (ART, PILLARS, PILLAR_QUESTION, VERDICT,  # noqa: F401,E402
                     VERDICT_ORDER, _CARD_NAMES, _blocked_reason, best_run,
                     comparability_key, direction_for, dominated_by,
                     group_snapshots, load_catalog, load_snapshots,
                     normalise_verdict, run_label)
from render import (PILLAR_COLOR, SKELETON, entries_for,  # noqa: F401,E402
                    explain_metric, explanation_markdown, metric_keys,
                    pillar_of, placeholder, render_caveats, render_chart,
                    render_explain, render_metric_explanations,
                    render_metric_legend)
from views.compare import comparison  # noqa: F401,E402
from views.overview import gallery  # noqa: F401,E402
from views.report import dashboard  # noqa: F401,E402

import routing  # noqa: E402
from views import compare as _compare, overview as _overview, report as _report  # noqa: E402


def _build_navigation():
    """The pages, and the sidebar that lists them.

    Report is listed even with nothing selected — it then says so. The
    alternative, hiding it until a run is picked, makes the navigation change
    shape under the reader, which is worse than one honest empty state.
    """
    # No url_path on the default page: Streamlit serves the default at "/" and
    # 404s on an explicit path for it, so naming one would break the very link
    # a reader is most likely to type.
    overview = st.Page(_overview.page, title="Overview", icon=":material/grid_view:",
                       default=True)
    report = st.Page(_report.page, title="Report", icon=":material/description:",
                     url_path="report")
    compare = st.Page(_compare.page, title="Compare runs", icon=":material/compare_arrows:",
                      url_path="compare")
    routing.register(overview=overview, report=report, compare=compare)

    sections: dict[str, list] = {"Evaluations": [overview, report, compare]}

    if SKELETON:
        # Planned pages appear only in skeleton mode, so the public navigation
        # lists exactly the pages that do something.
        from views import planned_pages
        sections["Reference (planned)"] = [
            st.Page(planned_pages.catalogue, title="Metric catalogue",
                    icon=":material/list_alt:", url_path="catalogue"),
            st.Page(planned_pages.policy, title="Policy",
                    icon=":material/gavel:", url_path="policy"),
        ]
        sections["Run (planned, local only)"] = [
            st.Page(planned_pages.studio, title="New evaluation",
                    icon=":material/play_circle:", url_path="studio"),
        ]
    return st.navigation(sections)


if __name__ == "__main__":
    if SKELETON:
        st.sidebar.caption(":gray[Skeleton mode — planned components are shown "
                           "as empty slots. Unset VERIFAI_SKELETON to hide them.]")
    _build_navigation().run()

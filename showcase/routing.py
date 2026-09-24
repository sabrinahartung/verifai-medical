"""Where the app can go, and how it gets there.

Routes carry an opaque id rather than an artifact path. That is deliberate: the
root object is going to change — models grouped under projects, with runs
beneath them — and a router that already treats the id as opaque survives that
without being rewritten. See docs/ui-ux-design.md.

Both a query parameter and session state are written on every hop. The query
parameter is what makes a report linkable and the back button work; session
state is the fallback, so a lost or stripped parameter degrades to a working
app rather than an empty page.
"""
from __future__ import annotations

import streamlit as st

# Filled once by app.py, which owns the st.Page objects. Kept here rather than
# imported from app.py so the views never import their own entry point.
_PAGES: dict[str, object] = {}


def register(**pages) -> None:
    _PAGES.update(pages)


def _hop(page: str, **params) -> None:
    for key, value in params.items():
        if value is None:
            st.query_params.pop(key, None)
            st.session_state.pop(key, None)
        else:
            st.query_params[key] = value
            st.session_state[key] = value
    target = _PAGES.get(page)
    if target is None:                       # no runtime (tests, bare import)
        return
    st.switch_page(target)


def go_to_overview() -> None:
    _hop("overview", run=None, lineage=None, model=None, project=None)


def go_to_project(name: str) -> None:
    _hop("project", project=name, model=None, run=None, lineage=None)


def go_to_model(key: str) -> None:
    _hop("model", model=key, run=None, lineage=None)


def go_to_report(run_id: str) -> None:
    _hop("report", run=run_id, lineage=None)


def go_to_compare(lineage: str | None = None, model: str | None = None) -> None:
    """All runs, or the runs of one lineage or one model. Never both filters."""
    _hop("compare", lineage=lineage, model=model, run=None)


def current(key: str) -> str | None:
    """The route parameter, preferring the URL so a pasted link wins.

    Known limit, measured on Streamlit 1.63: `st.switch_page` clears the query
    string, and it cannot be put back within that run — writing it before the
    render, after the render, and on a forced rerun were all undone. So a
    *pasted* link carries its parameter (`/report?run=<id>` renders that run)
    while an *in-app* click lands on `/report` with the run held in session
    state. The page is right either way; only the address bar is short.

    Worth revisiting when Streamlit's navigation gains parameterised pages.
    """
    value = st.query_params.get(key)
    if value:
        st.session_state[key] = value
        return value
    return st.session_state.get(key)

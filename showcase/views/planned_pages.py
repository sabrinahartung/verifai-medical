"""Pages that are planned and not built.

They exist only in skeleton mode (VERIFAI_SKELETON=1), so the public deploy's
navigation lists exactly the pages that do something. A nav entry leading to an
empty page is a promise the app cannot keep.
"""
from __future__ import annotations

import streamlit as st

from render import placeholder


def _stub(key: str, title: str, lead: str):
    def page():
        st.title(title)
        st.caption(lead)
        placeholder(key)
    return page


catalogue = _stub(
    "metric_catalogue", "Metric catalogue",
    "Every metric the framework knows about — including the ones that did not "
    "run here, and why they could not.")

policy = _stub(
    "policy_catalogue", "Policy",
    "The criteria applied to these numbers, who authored them, and on what "
    "grounds. A judgement you can argue with is the only kind worth making.")


def studio():
    st.title("New evaluation")
    st.caption("Point it at a checkpoint or a Hub link, review what was resolved, "
               "check it before spending the compute, then run it.")
    placeholder("studio")
    placeholder("preflight")

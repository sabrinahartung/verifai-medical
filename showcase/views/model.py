"""Model — one checkpoint: what it is, and every configuration it was evaluated in.

This is the page the gallery never had. It answers the three questions a reader
arrives with: *what is this model*, *has it been evaluated*, and *how do its
configurations differ* — and it states the model's identity rather than its
name, because a name can be reused and bytes cannot.
"""
from __future__ import annotations

import streamlit as st

from render import breadcrumb, placeholder
from registry import describe, evaluated_ids, find_model, load_registry
from routing import (current, go_to_compare, go_to_overview, go_to_project,
                     go_to_report)


def decision_rule(weights: dict | None) -> str:
    """A configuration's decision rule in words — `argmax` unless weights say otherwise."""
    if not weights:
        return "argmax — the most probable class"
    return "weighted — " + ", ".join(f"{cls} ×{w:g}" for cls, w in weights.items())


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


def _identity(model: dict):
    ck = model["checkpoint"]
    if ck["kind"] == "local" and model.get("sha256"):
        st.caption(f"Checkpoint `{ck['path']}` · content hash `{model['sha256']}`")
    elif ck["kind"] == "local":
        st.caption(f"Checkpoint `{ck['path']}` · never hashed — no machine this registry was "
                   f"built on had the file, so it is identified by its path only.")
    elif ck["kind"] == "hub":
        rev = ck.get("revision")
        where = f"`{ck['repo_id']}/{ck.get('filename')}`"
        st.caption(f"Hugging Face {where} · "
                   + (f"revision `{rev}`" if rev else
                      "revision **unpinned** — the Hub copy can change under this name, so "
                      "these reports may not describe what the link serves today."))
    else:
        st.warning("No weights are declared for this model, so it has no identity: it cannot "
                   "be told apart from another model and is never compared as the same one.")


def _training_record(model: dict):
    prov = model.get("provenance")
    if not prov:
        st.markdown(":gray[Not trained in this repository. No training record is declared, "
                    "so what it was trained on is not known here.]")
        return
    st.markdown(describe(prov))
    m = prov.get("manifests") or {}
    bits = []
    if m.get("train"):
        bits.append(f"train `{m['train'].rsplit('/', 1)[-1]}`"
                    + (f" ({prov['train_images']:,})" if prov.get("train_images") else ""))
    if m.get("val"):
        bits.append(f"validation `{m['val'].rsplit('/', 1)[-1]}`"
                    + (f" ({prov['val_images']:,})" if prov.get("val_images") else ""))
    for k, label in (("epochs", "epochs"), ("image_size", "px"), ("seed", "seed")):
        if prov.get(k) is not None:
            bits.append(f"{prov[k]} {label}" if k != "seed" else f"seed {prov[k]}")
    if bits:
        st.caption(" · ".join(bits))
    if prov.get("note"):
        st.caption(f"Trainer's note: *{prov['note']}*")
    if model.get("trained_by"):
        st.caption(f"Trained by scenario `{model['trained_by']}`.")


def _configuration_row(c: dict, done: bool, key: str):
    with st.container(border=True):
        left, right = st.columns([5, 1.6], vertical_alignment="center")
        with left:
            st.markdown(f"**{c['label']}**")
            st.caption(f"Decision rule: {decision_rule(c['decision_weights'])}")
        with right:
            if done:
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
    _identity(model)

    st.subheader("What it is")
    _training_record(model)

    evaluated = evaluated_ids()
    configs = model["configurations"]
    done = [c for c in configs if c["scenario"] in evaluated]

    st.subheader("Configurations")
    st.caption("The same weights, read with a different decision rule or scored on a "
               "different set of images. Each configuration has its own report. Only "
               "configurations under the same heading were scored on the same images, so "
               "only those can be compared directly.")
    for manifest, group in by_evaluation_set(model):
        st.markdown(f"**Scored on** `{(manifest or 'unknown').rsplit('/', 1)[-1]}`")
        for c in group:
            _configuration_row(c, c["scenario"] in evaluated, key=f"cfg_{c['scenario']}")

    if len(done) >= 2:
        st.caption("Configurations scored on the same images can be compared directly; "
                   "those scored on different images are shown side by side but never "
                   "ranked against each other.")
        if st.button(f"Compare its {len(done)} evaluated configurations →"):
            go_to_compare(model=model["key"])

    placeholder("stale_status")
    placeholder("delta_view")

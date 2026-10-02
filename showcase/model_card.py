"""The model card — what a checkpoint *is*, every line read from a file.

Shown at the top of the model page and, from any report, in a pop-up. It holds
only facts a machine can check, each with the place it was read from, shown on
hover: the model registry (built from the scenarios), the training record beside
the weights, the report's own metadata, and the Hub card's header as it was at
the evaluated commit. What an author wrote in prose — intended use, limitations,
how it was trained — is linked to, never restated.

It names no result. How the model did is the report's business, with every
pillar beside every other; a card that carried one number would make that number
the model's score. What it may say about an evaluation is what the coverage map
says: whether its integrity checks hold, and how much of it was measured.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import streamlit as st

from catalog import ACCESS_LABEL, ART, VERDICT, coverage_counts, integrity_state
from registry import ARCHIVE_NAMES, dataset_name
from render import readable
from routing import go_to_model


@dataclass(frozen=True)
class Fact:
    label: str
    value: str
    source: str          # where the value was read from, shown on hover


LICENCES = {"cc-by-nc-4.0": "CC BY-NC 4.0", "cc-by-4.0": "CC BY 4.0",
            "cc-by-sa-4.0": "CC BY-SA 4.0", "cc-by-nc-sa-4.0": "CC BY-NC-SA 4.0",
            "apache-2.0": "Apache 2.0", "mit": "MIT", "bsd-3-clause": "BSD 3-Clause",
            "openrail": "OpenRAIL", "other": "other (see the card)"}
IMAGENET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
NOT_YET = "read on first evaluation"
NOT_YET_WHY = ("Read from the model when an evaluation runs. No configuration of this model "
               "has been evaluated yet.")


# ---------- which report the run-time facts come from ----------
def load_report(scenario: str) -> dict | None:
    f = ART / scenario / "report.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.is_file() else None


def reports_of(model: dict) -> dict[str, dict]:
    """Every configuration's report that exists, by scenario id."""
    out = {}
    for c in model["configurations"]:
        r = load_report(c["scenario"])
        if r is not None:
            out[c["scenario"]] = r
    return out


def reference_report(model: dict, reports: dict[str, dict]) -> dict | None:
    """The report the card's run-time facts are read from.

    The newest report of an active configuration, else the newest of any. All
    of them loaded the same weights, so they agree on what the model is; an
    active one is the one kept current, and so the one that records the newest
    things a run reads (the Hub card's header arrived with the model card view).
    """
    status = {c["scenario"]: c["status"] for c in model["configurations"]}
    ranked = sorted(reports.items(),
                    key=lambda kv: (status.get(kv[0]) == "active", kv[1].get("created_at", "")))
    return ranked[-1][1] if ranked else None


# ---------- the facts ----------
def _rev7(ck: dict) -> str:
    return (ck.get("revision") or "")[:7]


def _arch_name(arch: str) -> str:
    return arch.replace("resnet", "ResNet").replace("convnext", "ConvNeXt")


def _general(model: dict, meta: dict, evaluated: bool) -> list[Fact]:
    ck = model["checkpoint"]
    out: list[Fact] = []
    if ck.get("kind") == "hub":
        out.append(Fact("Source", f"Hugging Face · `{ck['repo_id']}`",
                        "The scenario's `repo_id`. The run downloads the weights from there."))
        out.append(
            Fact("Version", f"commit `{_rev7(ck)}`, pinned",
                 "The scenario's `revision`. The run loads exactly this commit, so the card "
                 "and every report describe the same bytes.") if ck.get("revision") else
            Fact("Version", "unpinned — the default branch",
                 "The scenario declares no `revision`, so the Hub copy can change under this "
                 "name and a report may not describe what the link serves today."))
    elif ck.get("kind") == "local":
        out.append(Fact("Source", f"local file `{Path(ck['path']).name}`",
                        "The scenario's weights path."))
        out.append(Fact("Version", f"fingerprint `{model['sha256']}`",
                        "A hash of the weights file's bytes. Two reports with the same "
                        "fingerprint were made from exactly the same weights.")
                   if model.get("sha256") else
                   Fact("Version", "not hashed",
                        "No machine this registry was built on had the file, so it is "
                        "identified by its path only."))
    else:
        out.append(Fact("Source", "not declared",
                        "The scenarios name no weights, so this model cannot be told apart "
                        "from another."))
    out.append(_licence(ck, meta, evaluated))
    out.append(Fact("Trained", f"here, by `scenarios/{model['trained_by']}.yaml`",
                    "The scenario whose name is the checkpoint's filename trained it.")
               if model.get("trained_by") else
               Fact("Trained", "elsewhere",
                    "No scenario in this repository trained these weights."))
    out.append(Fact("Evaluation reach", ACCESS_LABEL.get(meta["access"], meta["access"]),
                    "How much of the model its adapter exposes. It decides which metrics can "
                    "run at all.") if meta.get("access") else
               Fact("Evaluation reach", NOT_YET, NOT_YET_WHY))
    return out


def _licence(ck: dict, meta: dict, evaluated: bool) -> Fact:
    if ck.get("kind") != "hub":
        return Fact("Licence", "not declared",
                    "A local weights file carries no licence, and none is declared for it.")
    if not evaluated:
        return Fact("Licence", NOT_YET, NOT_YET_WHY)
    card = meta.get("hub_card")
    if card is None:
        return Fact("Licence", "not recorded yet",
                    "This model's reports predate recording the Hub card; the next "
                    "evaluation reads it.")
    if not card.get("present"):
        return Fact("Licence", "not stated — the repository has no model card",
                    f"No `README.md` in `{ck['repo_id']}` at commit `{_rev7(ck)}`.")
    lic = card.get("license")
    if not lic:
        return Fact("Licence", "not stated in the model card",
                    f"The model card's header at commit `{_rev7(ck)}` has no `license` field.")
    return Fact("Licence", LICENCES.get(str(lic).lower(), f"`{lic}`"),
                f"The `license` field of the model card's header at commit `{_rev7(ck)}`.")


def _model(model: dict, meta: dict, report: dict | None) -> list[Fact]:
    prov, dec = model.get("provenance") or {}, model.get("declared") or {}
    out: list[Fact] = []
    if prov.get("arch"):
        out.append(Fact("Architecture", _arch_name(prov["arch"]),
                        "The training record written beside the weights."))
    elif dec.get("architecture"):
        out.append(Fact("Architecture", f"`{dec['architecture']}`",
                        "The repository's `config.json`. The loader refuses a scenario that "
                        "contradicts it."))
    elif dec.get("arch"):
        out.append(Fact("Architecture", f"{_arch_name(dec['arch'])} (declared)",
                        "Declared in the scenario. A bare `state_dict` does not record its "
                        "architecture, so this is the scenario's word, not the file's."))
    else:
        out.append(Fact("Architecture", "not stated", "Nothing on file states it."))

    m = meta.get("model") or {}
    classes = m.get("classes")
    if classes:
        names = [readable(c) for c in classes]
        names = [n[:1].upper() + n[1:] for n in names]
        out.append(Fact("Classes", f"{len(classes)} — " + ", ".join(names),
                        "The class list the evaluation loaded the model with, in the order of "
                        "its outputs."))
    else:
        out.append(Fact("Classes", NOT_YET, NOT_YET_WHY))

    pp = m.get("preprocessing")
    if pp:
        how = "How the evaluation prepared each image, as its report records it."
        out.append(Fact("Input", _input_words(pp), how))
        out.append(Fact("Normalisation", _normalisation_words(pp), how))
    else:
        out.append(Fact("Input", NOT_YET, NOT_YET_WHY))
    out.append(_preprocessing_check(report))
    return out


def _size(v) -> str:
    if isinstance(v, dict) and set(v) == {"height", "width"}:
        return f"{v['height']}×{v['width']}"
    if isinstance(v, dict):
        return ", ".join(f"{k.replace('_', ' ')} {x}" for k, x in v.items())
    if isinstance(v, (list, tuple)):
        return "×".join(str(x) for x in v)
    return str(v)


def _input_words(pp: dict) -> str:
    bits = [f"resized to {_size(pp['resize'])}" if pp.get("resize") else "not resized"]
    if pp.get("center_crop"):
        bits.append(f"centre crop {_size(pp['center_crop'])}")
    if pp.get("resample"):
        bits.append(f"{pp['resample']} interpolation")
    return ", ".join(bits)


def _normalisation_words(pp: dict) -> str:
    mean, std = pp.get("mean"), pp.get("std")
    if not mean or not std:
        return "none"
    if all(abs(a - b) < 1e-3 for a, b in zip([*mean, *std], [*IMAGENET[0], *IMAGENET[1]])):
        return "ImageNet mean and std"
    return (f"mean [{', '.join(f'{x:.3f}' for x in mean)}], "
            f"std [{', '.join(f'{x:.3f}' for x in std)}]")


def _preprocessing_check(report: dict | None) -> Fact:
    f = next((f for f in (report or {}).get("findings", [])
              if f.get("metric") == "preprocessing"), None)
    if report is None:
        return Fact("Preparation check", NOT_YET, NOT_YET_WHY)
    if f is None:
        return Fact("Preparation check", "not run",
                    "This model's reports predate the check (step C4).")
    words = {"measured": "agrees with " + ((f.get("details") or {}).get("reference_source")
                                           or "the model's own"),
             "unavailable": "nothing to check against",
             "invalid": "differs from the model's own — see the report"}
    return Fact("Preparation check", words.get(f.get("verdict"), f.get("verdict", "")),
                f["summary"])


def _training(model: dict, meta: dict, evaluated: bool) -> list[Fact]:
    prov, dec = model.get("provenance") or {}, model.get("declared") or {}
    ck = model["checkpoint"]
    out: list[Fact] = []
    train = (prov.get("manifests") or {}).get("train")
    if train:
        n = prov.get("train_images")
        out.append(Fact("Corpus", dataset_name(train) + (f" · {n:,} images" if n else ""),
                        "The training record, which lists the images the weights were "
                        "trained on."))
    elif dec.get("trained_on"):
        t = dec["trained_on"]
        corpora = t.get("corpora", []) if isinstance(t, dict) else t
        basis = t.get("basis") if isinstance(t, dict) else None
        out.append(Fact("Corpus", ", ".join(ARCHIVE_NAMES.get(c, c) for c in corpora),
                        "Declared in the scenario"
                        + (f", {basis}." if basis else ".")))
    else:
        out.append(Fact("Corpus", "unknown",
                        "Nothing declares the training data. An unknown corpus is never "
                        "treated as independent of the test images."))

    if ck.get("kind") == "hub":
        card = meta.get("hub_card") or {}
        if card.get("datasets"):
            out.append(Fact("Named by the card", ", ".join(f"`{d}`" for d in card["datasets"]),
                            "The `datasets` field of the model card's header."))
        elif not evaluated:
            out.append(Fact("Named by the card", NOT_YET, NOT_YET_WHY))

    why = ("Whether each test image can be compared with the training images. Without "
           "that list, overlap can at best be ruled out archive by archive, and the reports "
           "say so.")
    out.append(Fact("Image-by-image check",
                    "possible — the training image list is on file" if train else
                    "not possible — no list of training images", why))
    if prov:
        # Architecture and corpus have their own lines; this says how it was trained.
        bits = []
        if prov.get("freeze_backbone"):
            bits.append("linear probe, backbone frozen")
        if prov.get("loss") and prov["loss"] != "ce":
            g = prov.get("focal_gamma")
            bits.append(f"{prov['loss']} loss" + (f" (γ={g:g})" if g is not None else ""))
        if prov.get("sampling") not in (None, "none"):
            bits.append(f"{prov['sampling']} sampling")
        if prov.get("class_weights"):
            bits.append("class-weighted")
        if prov.get("epochs") is not None:
            bits.append(f"{prov['epochs']} epochs")
        if prov.get("seed") is not None:
            bits.append(f"seed {prov['seed']}")
        out.append(Fact("Training run", " · ".join(bits) or "not stated",
                        "The training record written beside the weights."))
    elif ck.get("kind") == "hub":
        out.append(Fact("Training run", "in the model card's text",
                        "Prose cannot be checked by a machine, so the card links to it "
                        "rather than restating it."))
    return out


def facts(model: dict, report: dict | None) -> dict[str, list[Fact]]:
    """The card's three boxes, each a list of facts with their sources."""
    meta = (report or {}).get("meta") or {}
    evaluated = report is not None
    return {"General": _general(model, meta, evaluated),
            "Model": _model(model, meta, report),
            "Training data": _training(model, meta, evaluated)}


# ---------- one evaluation, without a result ----------
# The report banner's three states, in its words: clear, provisional, unusable.
INTEGRITY_WORDS = {"measured": "✓ integrity checks hold",
                   "insufficient": "⚠️ provisional — integrity not fully checked",
                   "unavailable": "⚠️ provisional — integrity not fully checked",
                   "invalid": "⛔ not usable — a precondition failed"}


def evaluation_line(report: dict) -> str:
    """Integrity and coverage of one report, in words. Never a metric's value."""
    state, _ = integrity_state(report.get("findings", []))
    rows = (report.get("meta") or {}).get("coverage") or []
    counts = coverage_counts(rows)
    done = " · ".join(f"{counts[k]} {VERDICT[k][1].lower()}"
                      for k in ("measured", "insufficient", "unavailable", "invalid")
                      if counts.get(k))
    return INTEGRITY_WORDS[state] + (f" · {done}" if done else "")


# ---------- drawing ----------
def hub_card_url(model: dict, report: dict | None) -> str | None:
    """The model card on the Hub at the evaluated commit, or the repository."""
    ck = model["checkpoint"]
    if ck.get("kind") != "hub":
        return None
    card = ((report or {}).get("meta") or {}).get("hub_card") or {}
    base = f"https://huggingface.co/{ck['repo_id']}"
    if card.get("present") and ck.get("revision"):
        return f"{base}/blob/{ck['revision']}/README.md"
    return base


def render_card(model: dict, report: dict | None) -> None:
    url = hub_card_url(model, report)
    st.caption("Every line is read from a file — the scenario, the training record, the "
               "report or the model card on the Hub — never written by hand. The ? beside "
               "each line says which."
               + (f" [🤗 The model card on the Hub]({url})" if url else ""))
    for col, (section, rows) in zip(st.columns(3), facts(model, report).items()):
        with col, st.container(border=True):
            st.markdown(f"**{section}**")
            for f in rows:
                st.markdown(f":gray[{f.label}]  \n{f.value}", help=f.source)


@st.dialog("Model card", width="large")
def card_dialog(model: dict, scenario: str | None = None) -> None:
    """The card from a report: the model this configuration belongs to."""
    reports = reports_of(model)
    st.subheader(model["name"])
    render_card(model, reference_report(model, reports))
    st.markdown("**Evaluations**")
    for c in model["configurations"]:
        r = reports.get(c["scenario"])
        if r is None or c["status"] != "active":
            continue
        here = " · *this report*" if c["scenario"] == scenario else ""
        st.markdown(f"{c['label']}{here}  \n:gray[{dataset_name(c['eval_manifest'])} · "
                    f"{evaluation_line(r)}]")
    archived = sum(1 for c in model["configurations"]
                   if c["status"] == "archived" and c["scenario"] in reports)
    if archived:
        st.caption(f"And {archived} archived configuration{'s' if archived > 1 else ''}, "
                   f"listed on the model page.")
    if st.button("Open the model page →"):
        go_to_model(model["key"])

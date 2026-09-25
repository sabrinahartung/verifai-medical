"""Integrity: do the model and the data name the same classes?

Signature: run(model, dataset, ctx) -> Finding

A model scored on data labelled with other classes measures something else
without saying so. Derm7pt has no actinic keratoses, so one of the model's
seven classes can never be scored there; a dataset with a class the model
cannot output makes every image of it a guaranteed error. Both are facts about
the evaluation, stated before any number is read.

The runner checks the same relation before any metric runs and refuses a
disjoint pair unless the scenario gives an explicit `dataset.label_map` — a
mapping is a claim about medicine, and it is never guessed.
"""
from __future__ import annotations

from typing import Any

from verifai.core.findings import Finding
from verifai.core.integrity import label_space
from verifai.metrics._common import class_name

EXPLAIN = {
    "what": ("The model answers with one of a fixed list of classes, and the test images are "
             "labelled with a list of their own. If the two lists differ, some of the model's "
             "classes can never be tested here, or some images can never be answered "
             "correctly. Either changes what every other number means, so it is stated first."),
    "how": ("Identical lists are the ordinary case. Classes the data lacks are left out of "
            "per-class results and of balanced accuracy, which is then averaged over fewer "
            "classes. Classes the model lacks turn every image of them into an error. A label "
            "map, when there is one, says which data label was read as which model class."),
    "limits": ("It compares names, not meanings. Two archives can use the same word for "
               "slightly different diagnostic criteria, and a label map records a decision "
               "someone made about that — it does not verify it."),
}

_SENTENCE = {
    "identical": "The model and the {n} test images use the same {k} classes.",
    "dataset_subset": ("The {n} test images hold {d} of the model's {m} classes: {unscored} "
                       "never occur here, so they go unscored and averages over classes use "
                       "{d} rather than {m}."),
    "model_subset": ("The {n} test images hold {d} classes and the model can output only {m} of "
                     "them: every image of {unpredictable} is necessarily an error."),
    "partial": ("The model and the {n} test images share {both} classes. {unscored} occur only "
                "in the model and go unscored; {unpredictable} occur only in the data and are "
                "necessarily errors."),
    "disjoint": ("The model and the {n} test images share no class name; the data was read "
                 "through the scenario's label map."),
}


def run(model, dataset, ctx: dict[str, Any]) -> Finding:
    scenario = ctx.get("scenario", {}) or {}
    ls = label_space(list(model.classes), list(dataset.classes))
    label_map = (scenario.get("dataset") or {}).get("label_map") or {}
    value = {**ls, "label_map": dict(label_map)}
    words = lambda cs: ", ".join(class_name(c) for c in cs) or "none"   # noqa: E731
    summary = _SENTENCE[ls["relation"]].format(
        n=f"{len(dataset):,}", k=ls["n_model_classes"], m=ls["n_model_classes"],
        d=ls["n_dataset_classes"], both=len(set(model.classes) & set(dataset.classes)),
        unscored=words(ls["unscored"]), unpredictable=words(ls["unpredictable"]))
    if label_map:
        k = len(label_map)
        summary += (f" {k} data {'label was' if k == 1 else 'labels were'} mapped onto model "
                    f"classes by the scenario.")
    return Finding(pillar="integrity", metric="label_space",
                   domain=scenario.get("domain", "image"), value=value, verdict="measured",
                   summary=summary, details={"explain": EXPLAIN})

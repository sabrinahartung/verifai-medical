"""Integrity: was each image prepared the way the model expects?

Signature: run(model, dataset, ctx) -> Finding

A model only ever sees tensors. Resize an image differently, crop it, or normalise
it with other numbers, and the same weights give different answers — every metric
in the report then measures a model that exists nowhere but in this evaluation.
Nothing crashes when that happens, which is why it is checked rather than trusted.

The preprocessing the evaluation used is compared, field by field, with what the
model's own side states, taken from the first of these that exists:

  1. the model's own image processor (a Hugging Face model uses it as is);
  2. a `preprocessor_config.json` in the model's Hub repository, at the pinned revision;
  3. this project's training record beside the weights.

With none of them there is nothing to compare against, and the finding is
`unavailable`: the preprocessing is the scenario's declaration, unchecked. A field
that differs makes the finding `invalid`, because a mismatch there is a broken
precondition of every other number, not a property of the model.
"""
from __future__ import annotations

from typing import Any

from verifai.core.findings import Finding
from verifai.metrics._baseline import NO_DIRECTION
from verifai.metrics.integrity.provenance import _training_record
from verifai.models.preprocessing import compare, record_spec

EXPLAIN = {
    "what": ("Before a model sees an image, the image is resized, perhaps cropped, and its "
             "colours are rescaled to the numbers the model was trained on. If the evaluation "
             "prepares images differently from training, the same model gives different "
             "answers, and every other number in this report describes that different model. "
             "So the preparation used here is compared with what the model's own side says it "
             "should be."),
    "how": ("Each field the model's side states — resize, crop, interpolation, rescaling and "
            "the colour normalisation — is compared with what this evaluation used. Any field "
            "that differs is listed with both values, and makes the report unusable as a "
            "measurement of that model. Fields nobody stated cannot be checked, and are listed "
            "as such."),
    "limits": ("It compares settings, not pixels: two libraries given the same settings can "
               "still interpolate very slightly differently. And the model's side can only say "
               "what its author wrote down; a processor file that was itself wrong would agree "
               "with a wrong evaluation."),
}

_FIELD_WORDS = {"resize": "resize", "center_crop": "centre crop", "resample": "interpolation",
                "rescale_factor": "pixel rescaling", "mean": "normalisation mean",
                "std": "normalisation std", "crop_pct": "crop percentage"}


def _words(fields: list[str]) -> str:
    """Field names as a phrase: `the resize`, or `the resize, interpolation and centre crop`."""
    ws = [_FIELD_WORDS.get(f, f) for f in fields]
    return "the " + (ws[0] if len(ws) == 1 else ", ".join(ws[:-1]) + " and " + ws[-1])


def _show(v: Any) -> str:
    if isinstance(v, dict):
        return "×".join(str(x) for x in v.values()) if set(v) == {"height", "width"} else \
            ", ".join(f"{k.replace('_', ' ')} {x}" for k, x in v.items())
    if isinstance(v, list):
        return "[" + ", ".join(f"{x:.3f}" if isinstance(x, float) else str(x) for x in v) + "]"
    if isinstance(v, float):
        return f"{v:.5f}"
    return "none" if v is None else str(v)


def run(model, dataset, ctx: dict[str, Any]) -> Finding:
    scenario = ctx.get("scenario", {}) or {}
    domain = scenario.get("domain", "image")
    n = len(dataset)
    used = dict((getattr(model, "metadata", None) or {}).get("preprocessing") or {})
    used.pop("processor", None)

    reference = getattr(model, "reference_preprocessing", None)
    if reference is None:
        record = _training_record(scenario.get("model") or {})
        spec = record_spec(record) if record else {}
        if spec:
            reference = {"source": "training_record",
                         "where": f"this project's training record for `{record.get('scenario')}`",
                         "spec": spec}

    if reference is None:
        value = {"source": None, "fields_compared": 0, "fields_differing": 0,
                 "differences": [], "not_stated": []}
        summary = (f"The {n:,} test images were prepared as the scenario declares, and nothing "
                   f"on the model's side states its preprocessing: no processor file in its "
                   f"repository and no training record. So it cannot be checked; that is not "
                   f"the same as a match.")
        return Finding(pillar="integrity", metric="preprocessing", domain=domain, value=value,
                       verdict="unavailable", summary=summary,
                       # its zeros mean "nothing was compared", not "nothing differed",
                       # so none of them is ranked beside a run that was checked
                       details={"explain": EXPLAIN, "used": used, "better": NO_DIRECTION})

    c = compare(used, reference["spec"])
    value = {"source": reference["source"], "fields_compared": len(c["compared"]),
             "fields_differing": len(c["differences"]), "differences": c["differences"],
             "not_stated": c["not_stated"]}
    if c["differences"]:
        diffs = "; ".join(f"{_FIELD_WORDS.get(d['field'], d['field'])} {_show(d['used'])} here "
                          f"against {_show(d['reference'])}" for d in c["differences"])
        summary = (f"The {n:,} test images were prepared differently from what "
                   f"{reference['where']} states — {diffs}. Every other number in this report "
                   f"therefore describes a different model from the one named.")
        verdict = "invalid"
    else:
        own = reference["source"] == "own_processor"
        summary = (f"The {n:,} test images were prepared by {reference['where']}."
                   if own else
                   f"The {n:,} test images were prepared as {reference['where']} states: "
                   f"{_words(c['compared'])} {'agrees' if len(c['compared']) == 1 else 'agree'}.")
        if c["not_stated"] and not own:
            summary += (f" It does not state {_words(c['not_stated'])}, so "
                        f"{'that was' if len(c['not_stated']) == 1 else 'those were'} not checked.")
        verdict = "measured"
    # The two specs sit in details, not value: value's numbers become comparison
    # keys, and a resize height is not a result.
    return Finding(pillar="integrity", metric="preprocessing", domain=domain, value=value,
                   verdict=verdict, summary=summary,
                   details={"explain": EXPLAIN, "used": used, "reference": reference["spec"],
                            "reference_source": reference["where"],
                            "better": {"fields_differing": "lower"}})

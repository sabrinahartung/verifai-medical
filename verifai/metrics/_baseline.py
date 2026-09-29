"""What each metric's number is compared with — `details["baseline"]`.

A number says nothing to a reader who does not know what it *should* be. Each
metric publishes a reference beside its value, and whether the value's interval
clears it. The report's findings strip shows only what cleared — what the
evaluation *established* — and it is ordered by pillar, never by how good the
number is.

Three kinds of reference, as in the roadmap's Phase G, with different weight:

* `control` — measured in the same run under the same conditions (the other
  skin-tone groups; a random highlight of the same size). The strongest: data,
  not opinion.
* `chance` — what a trivial rule scores (always answering the most common class;
  an attacker who guesses). Follows from the data, not from a choice.
* `ideal` — what the quantity is by definition at its best (nothing shared
  between the splits; a prediction no distortion changes).

A fourth kind is never used here: a *criterion*, an authored threshold for
"good enough". That belongs in a versioned policy file with a rationale and an
owner, not in a metric.

**Symmetric in good and bad news.** A claim is made whenever the interval
excludes the reference, in whichever direction — accuracy below always-nevus is
as established as accuracy above it. A strip that only promoted bad (or good)
news would be a rating by another name.

Pure functions over a finding's `value`, free of torch, so the tests exercise
them directly and a metric cannot publish a claim its own numbers do not carry.
"""
from __future__ import annotations

from typing import Any

from verifai.metrics._common import class_name


def _reference(kind: str, value: Any, basis: str, cleared: bool,
               claim: str | None = None, gap: float | None = None) -> dict:
    return {"kind": kind, "value": value, "basis": basis, "cleared": bool(cleared),
            "claim": claim if cleared else None, "gap": gap}


def _versus(interval, reference: float) -> str | None:
    """'above' / 'below' when the interval excludes the reference, else None."""
    if not interval or len(interval) != 2 or None in interval:
        return None
    lo, hi = interval
    return "above" if lo > reference else "below" if hi < reference else None


def split_leakage(value: dict) -> dict | None:
    if "contamination" not in value:
        return None
    clean = value.get("shared_groups", 1) == 0 and value.get("shared_ids", 1) == 0
    return _reference(
        "ideal", 0, "nothing shared between the test set and the training data", clean,
        f"None of the {value.get('n_test', 0):,} test images shares a lesion or an image "
        f"with the {value.get('n_train', 0):,} the model trained on.")


def provenance(value: dict) -> dict | None:
    if "checkable" not in value:
        return None
    return _reference("ideal", True, "a declared training record the split can be checked against",
                      value["checkable"],
                      "The model's training data is on record, so the split could be checked "
                      "image by image.")


def corpus_ancestry(value: dict) -> dict | None:
    if not value.get("trained_on") or not value.get("evaluated_on") or value.get("unknown"):
        return None
    separate = not value.get("shared")
    held_back = value.get("held_back_row_by_row") is True
    return _reference("ideal", "separate",
                      "no archive shared with the training data, or one held back image by image",
                      separate or held_back,
                      "The test images come from an archive the model did not train on."
                      if separate else
                      "The archive the test images come from is part of the training corpus, "
                      "and it was held back image by image.")


def label_space_(value: dict) -> dict | None:
    if "relation" not in value:
        return None
    return _reference("ideal", "identical", "the model and the data naming the same classes",
                      value["relation"] == "identical",
                      "The model and the test images use the same classes.")


def classification(value: dict) -> dict | None:
    per_class, n, acc = value.get("per_class") or {}, value.get("n") or 0, value.get("accuracy")
    supports = {c: v.get("support") or 0 for c, v in per_class.items()}
    if not supports or not n or acc is None:
        return None
    majority_class, count = max(supports.items(), key=lambda kv: kv[1])
    majority = round(count / n, 4)
    side = _versus(value.get("accuracy_ci"), majority)
    lo, hi = value.get("accuracy_ci") or (None, None)
    claim = side and (f"Accuracy {acc:.3f} [{lo:.2f}–{hi:.2f}] is {side} what always answering "
                      f"the most common class scores ({class_name(majority_class)}, "
                      f"{majority:.3f}).")
    return _reference("chance", majority,
                      f"always answering the most common class, {class_name(majority_class)} "
                      f"({count:,} of {n:,})", bool(side), claim, gap=round(acc - majority, 4))


def skin_tone(value: dict) -> dict | None:
    if "gap_is_separated" not in value:
        return None
    acc = value.get("subgroup_accuracy") or {}
    if len(acc) < 2:
        return None
    (low_g, low), (high_g, high) = min(acc.items(), key=lambda kv: kv[1]), max(acc.items(), key=lambda kv: kv[1])
    gap = value.get("accuracy_gap")
    return _reference(
        "control", "the other skin-tone groups",
        "accuracy in the other skin-tone groups, each with its interval",
        value["gap_is_separated"],
        f"Accuracy differs between skin-tone groups: {gap * 100:.0f} points between "
        f"{low_g} ({low:.2f}) and {high_g} ({high:.2f}), with intervals that do not overlap.",
        gap=gap)


def corruption(value: dict) -> dict | None:
    mean = value.get("mean_stability")
    if mean is None:
        return None
    # An ideal nobody reaches is not a finding: every model changes some answers
    # under noise, so "less than perfectly stable" would be established for all
    # of them and say nothing. The gap is published; no claim is made.
    return _reference("ideal", 1.0, "a prediction no ordinary distortion changes",
                      False, gap=round(1.0 - mean, 4))


def membership_inference(value: dict) -> dict | None:
    auc = value.get("mia_auc")
    if auc is None:
        return None
    side = _versus(value.get("mia_auc_ci"), 0.5)
    lo, hi = value.get("mia_auc_ci") or (None, None)
    claim = side and ((f"Training membership is detectable: attack AUC {auc:.3f} "
                       f"[{lo:.2f}–{hi:.2f}] lies above chance (0.5).") if side == "above" else
                      (f"The attack scores below chance: AUC {auc:.3f} [{lo:.2f}–{hi:.2f}]."))
    return _reference("chance", 0.5, "an attacker who guesses", bool(side), claim,
                      gap=round(auc - 0.5, 4))


def gradcam(value: dict) -> dict | None:
    gain, ci = value.get("faithfulness_gain"), value.get("faithfulness_gain_ci")
    if gain is None:
        return None
    side = _versus(ci, 0.0)
    claim = side and (
        (f"Grad-CAM's highlight matters more than a random one of the same size: masking it "
         f"lowers confidence by {gain:.3f} [{ci[0]:.3f}–{ci[1]:.3f}] more.") if side == "above" else
        (f"Grad-CAM's highlight matters less than a random one of the same size "
         f"({gain:.3f} [{ci[0]:.3f}–{ci[1]:.3f}])."))
    return _reference("control", value.get("mean_random_control"),
                      "masking a random region of the same size, in the same images",
                      bool(side), claim, gap=gain)


# Keyed by the *finding's* metric name, since that is what a report stores.
BY_FINDING = {
    "split_leakage": split_leakage,
    "provenance": provenance,
    "corpus_ancestry": corpus_ancestry,
    "label_space": label_space_,
    "top1_accuracy": classification,
    "skin_tone_ita": skin_tone,
    "corruption_stability": corruption,
    "membership_inference_auc": membership_inference,
    "gradcam_faithfulness": gradcam,
}


def attach(finding) -> None:
    """Give a finding its baseline, computed from its own value.

    Called by the runner on every finding, so no metric can forget it and none
    can publish a claim its numbers do not carry. A metric with no entry here
    publishes `None` — no reference, and therefore nothing it can claim.
    """
    fn = BY_FINDING.get(finding.metric)
    finding.details.setdefault(
        "baseline", fn(finding.value) if fn and isinstance(finding.value, dict) else None)

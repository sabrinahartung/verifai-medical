"""Fairness (image): skin-tone coverage & subgroup accuracy via ITA.

Signature: run(model, dataset, ctx) -> Finding

HAM10000 has no skin-type labels, so we estimate the Individual Typology Angle
(ITA) from each image and bin it into coarse skin-tone groups — a label-free way
to probe fairness. Two things are reported:

  - COVERAGE: how the sample distributes across skin tones. HAM10000 is known to
    skew light; a skewed sample is itself a documented fairness limitation.
  - SUBGROUP ACCURACY: top-1 accuracy per skin-tone bin *when the sample is big
    enough*. With only a few images per bin this is not claimed as a result —
    the summary says so. Run the full subset (notebook) for a real gap.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from verifai.core.findings import Finding
from verifai.metrics._common import estimate_ita, ita_bin, ITA_BINS
from verifai.metrics._stats import fmt, wilson


def run(model, dataset, ctx: dict[str, Any]) -> Finding:
    bins_order = [b[0] for b in ITA_BINS]
    per_bin_total = defaultdict(int)
    per_bin_correct = defaultdict(int)
    per_example = []

    for s in dataset:
        img = dataset.load(s)
        ita = estimate_ita(img)
        b = ita_bin(ita)
        per_bin_total[b] += 1
        entry = {"id": s.id, "ita": round(ita, 1), "bin": b, "true": s.label}
        if s.label is not None:
            probs = model.predict_probs(img)
            top = model.decide(probs, getattr(s, "meta", None))
            ok = (top == s.label)
            per_bin_correct[b] += int(ok)
            entry["pred"] = top
            entry["correct"] = ok
        per_example.append(entry)

    coverage = [per_bin_total.get(b, 0) for b in bins_order]
    n = len(per_example)
    # A subgroup *gap* needs at least two groups to compare, each big enough to
    # mean something. One populated bin would otherwise yield a gap of 0.0 and a
    # green "pass" — precisely backwards on a sample that contains only one skin
    # tone, which is HAM10000's documented skew.
    populated = [c for c in coverage if c > 0]
    enough_per_bin = len(populated) >= 2 and min(populated) >= 10

    # Coverage skew is stated in the summary rather than judged. A thin bin is a
    # fact about the data; whether it disqualifies the model is not something
    # this metric can know.
    dark_share = per_bin_total.get(bins_order[-1], 0) / n if n else 0
    verdict = "measured" if enough_per_bin else "insufficient"

    # Said about this sample, whichever archive it came from: the reason no gap
    # is reported is the bin counts, and those are right here.
    summary = (f"Skin-tone coverage (ITA-estimated) over {n:,} images: "
               + ", ".join(f"{b.split(' ')[0]} {per_bin_total.get(b, 0):,}" for b in bins_order)
               + (". Only one skin-tone group is present, so there is no second group to "
                  "compare accuracy with." if len(populated) < 2 else
                  ". At least one group holds fewer than 10 images, so no accuracy gap "
                  "between groups is reported."))

    chart_bar = {
        "kind": "bar", "title": "Skin-tone coverage of the sample (ITA bins)",
        "x": [b for b in bins_order], "y": coverage, "color": "#C77700",
        "x_title": "Estimated skin type", "y_title": "Number of images",
    }

    explain = {
        "what": ("A skin-lesion model that was only ever tested on light skin cannot be "
                 "trusted on dark skin. Skin-type labels are rarely recorded with "
                 "dermoscopy images, so the skin tone is estimated from the image itself: "
                 "the ITA (Individual Typology "
                 "Angle) is measured on the healthy skin around the lesion and sorted into "
                 "light, medium and dark bins."),
        "how": ("The bars show how many images fall into each skin-tone bin. What matters "
                "is the shape: a tall light bar next to a near-empty dark bar means the "
                "model is barely being tested on darker skin, so any headline accuracy "
                "mostly describes light skin. Once every populated bin holds at least 10 "
                "images, a second chart adds the accuracy per bin and the gap between "
                "them."),
        "limits": ("Coverage is not performance — this chart shows who is in the sample, "
                   "not how well the model serves them. ITA is also an estimate from "
                   "pixels, not a clinical Fitzpatrick assessment: lighting, vignetting "
                   "and dermatoscope settings all shift it."),
    }
    details: dict[str, Any] = {"explain": explain,
                               "better": {"accuracy_gap": "lower",
                                          "subgroup_accuracy.*": "higher"},
                               "per_example": per_example,
                               "chart": chart_bar, "enough_per_bin": bool(enough_per_bin)}
    value: dict[str, Any] = {"coverage": dict(zip(bins_order, coverage)), "n": n}

    if enough_per_bin:
        populated_bins = [b for b in bins_order if per_bin_total.get(b)]
        acc = {b: round(per_bin_correct[b] / per_bin_total[b], 3) for b in populated_bins}
        ci = {b: wilson(per_bin_correct[b], per_bin_total[b]) for b in populated_bins}
        vals = list(acc.values())
        gap = round(max(vals) - min(vals), 3) if len(vals) > 1 else 0.0
        value["subgroup_accuracy"] = acc
        value["subgroup_accuracy_ci"] = ci
        value["accuracy_gap"] = gap

        # A gap only counts as evidence if the groups' intervals do not overlap.
        # Small bins produce wide intervals and therefore large but unsupported gaps.
        best = max(acc, key=acc.get)
        worst = min(acc, key=acc.get)
        separated = bool(ci[best] and ci[worst] and ci[worst][1] < ci[best][0])
        value["gap_is_separated"] = separated

        # Whether the intervals separate is a statement about evidence, not a
        # value judgement, so it is the one thing worth encoding here. How large
        # a gap is *acceptable* is a policy question with no answer in the data.
        verdict = "measured" if separated else "insufficient"

        details["chart2"] = {
            "kind": "bar", "title": "Accuracy by skin type (ITA), with 95% intervals",
            "x": populated_bins, "y": [acc[b] for b in populated_bins], "color": "#5B3FD6",
            "y_lo": [ci[b][0] for b in populated_bins],
            "y_hi": [ci[b][1] for b in populated_bins],
            "x_title": "Estimated skin type", "y_title": "Top-1 accuracy",
            "hover": [f"{acc[b]:.3f} [{ci[b][0]:.2f}–{ci[b][1]:.2f}] on "
                      f"{per_bin_total[b]:,} images" for b in populated_bins],
        }
        summary = (
            f"Accuracy by ITA-estimated skin type over {n:,} images; the largest gap is "
            f"{gap*100:.0f} points, between {worst} ({fmt(acc[worst], ci[worst])} on "
            f"{per_bin_total[worst]:,} images) and {best} ({fmt(acc[best], ci[best])} on "
            f"{per_bin_total[best]:,}). "
            + ("Their 95% intervals do not overlap, so the difference is supported."
               if separated else
               "Their 95% intervals overlap, so this gap is not yet distinguishable from "
               "sampling noise — the smaller groups are too small to conclude from."))

    return Finding(
        pillar="fairness", metric="skin_tone_ita", domain="image",
        value=value, verdict=verdict, summary=summary, details=details,
    )

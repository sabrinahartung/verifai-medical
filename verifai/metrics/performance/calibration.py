"""Performance: does the model's stated confidence match how often it is right?

Signature: run(model, dataset, ctx) -> Finding

A model that says "melanoma, 90%" should be right about nine times in ten when it
says so. Accuracy cannot tell: two models with the same accuracy can differ
completely in whether their percentages mean anything — and a percentage is what
a reader acts on. Calibration is also the first thing expected to break when the
clinic changes, which is why it is measured on every evaluation set.

Three numbers, all over the decided class (`model.decide`, never `argmax`, so a
scenario's decision weights apply):

* **Expected calibration error** (ECE) — the images are sorted into ten bins by
  stated confidence; in each, stated confidence is compared with the share
  actually right, and the gaps are averaged, weighted by bin size [Naeini et al.
  2015; Guo et al. 2017 — docs/references.md [53], [15]]. With a 95% bootstrap interval.
* **The same error for a perfectly calibrated model**, as a control measured in
  the same run: the outcomes are redrawn from the model's own confidences, so the
  model is calibrated by construction, and the error is recomputed — consistency
  resampling [Bröcker & Smith 2007 — [54]]. ECE is biased upwards on a finite sample; a
  calibrated model never scores 0. Miscalibration is claimed only when the real
  error exceeds what a calibrated model reaches 95% of the time.
* **Brier score** — the squared distance between the probabilities and what
  happened, over all classes [Brier 1950 — [55]]. 0 is perfect, 2 the worst possible.

Per class, calibration in the large: the average probability the model gives a
class against how often the class actually occurs.
"""
from __future__ import annotations

import random
from typing import Any

from verifai.core.findings import Finding
from verifai.metrics._common import chunks, class_name, predict_many
from verifai.metrics._stats import fmt, mean_ci, wilson

BINS = 10
RESAMPLES = 1000
VERDICT_MIN_N = 30

_BETTER = {"ece": "lower", "brier": "lower"}

_EXPLAIN = {
    "what": ("When the model calls an image a melanoma with 90% confidence, is it right about "
             "nine times in ten? That is calibration: whether the model's percentages mean what "
             "they say. Two models can have the same accuracy while one says 90% when it is "
             "right half the time — and a reader, or a doctor, acts on the percentage. The "
             "expected calibration error averages the gap between stated confidence and actual "
             "accuracy across the images; the Brier score measures how far the probabilities sit "
             "from what happened."),
    "how": ("Each point is a group of images the model was similarly sure about: how sure it "
            "said it was (across) against how often it was right (up), with a 95% interval. On "
            "the dashed diagonal, confidence and accuracy agree. Points below it mean the model "
            "was overconfident there; above it, too cautious. Points from few images have wide "
            "intervals, and a gap inside the interval is no evidence of anything. The result "
            "above also gives the error a perfectly calibrated model would show on this many "
            "images by chance — the real error only means something above that."),
    "limits": ("Calibration is about the model's percentages, not its decisions: a well-calibrated "
               "model can still miss most melanomas, and a badly calibrated one can rank cases "
               "well. It is measured on these images, at this mix of diagnoses; at a clinic with "
               "another mix the same model can be calibrated differently. And the ten groups are a "
               "convention — other groupings give somewhat different numbers, which is why the "
               "error is compared with a control measured the same way, never with zero."),
}


def _ece(conf: list[float], hit: list[int]) -> float:
    """Expected calibration error over equal-width confidence bins."""
    sums = [[0.0, 0, 0] for _ in range(BINS)]          # confidence sum, hits, count
    for c, h in zip(conf, hit):
        b = min(int(c * BINS), BINS - 1)
        sums[b][0] += c
        sums[b][1] += h
        sums[b][2] += 1
    n = len(conf)
    return sum(abs(s / k - h / k) * k / n for s, h, k in sums if k)


def _bins(conf: list[float], hit: list[int]) -> list[dict[str, Any]]:
    """The reliability diagram's points: one per non-empty bin, with a Wilson interval."""
    out = []
    for b in range(BINS):
        lo, hi = b / BINS, (b + 1) / BINS
        idx = [i for i, c in enumerate(conf) if lo <= c < hi or (b == BINS - 1 and c == 1.0)]
        if not idx:
            continue
        k, hits = len(idx), sum(hit[i] for i in idx)
        out.append({"from": lo, "to": hi, "n": k, "confidence": sum(conf[i] for i in idx) / k,
                    "accuracy": hits / k, "accuracy_ci": wilson(hits, k)})
    return out


def run(model, dataset, ctx: dict[str, Any]) -> Finding:
    classes = list(getattr(model, "classes", []))
    labeled = [s for s in dataset if s.label is not None]
    n = len(labeled)
    if not n:
        return Finding(pillar="performance", metric="calibration", domain="image", value=None,
                       verdict="unavailable", summary="No labeled examples, so nothing to calibrate against.",
                       details={"reason": "no_labels"})

    conf: list[float] = []
    hit: list[int] = []
    brier: list[float] = []
    prob_sum = {c: 0.0 for c in classes}
    per_example: list[dict[str, Any]] = []
    scored = ((s, p) for batch in chunks(labeled)
              for s, p in zip(batch, predict_many(model, [dataset.load(s) for s in batch])))
    for s, probs in scored:
        top = model.decide(probs, getattr(s, "meta", None))
        c = float(probs[top])
        b = sum((float(probs.get(k, 0.0)) - (1.0 if k == s.label else 0.0)) ** 2 for k in classes)
        conf.append(c)
        hit.append(int(top == s.label))
        brier.append(b)
        for k in classes:
            prob_sum[k] += float(probs.get(k, 0.0))
        per_example.append({"id": s.id, "confidence": round(c, 4), "brier": round(b, 4)})

    ece = _ece(conf, hit)
    rng = random.Random(int(ctx.get("seed", 42)))
    idx = range(n)
    boot = sorted(_ece([conf[i] for i in s], [hit[i] for i in s])
                  for s in ([rng.randrange(n) for _ in idx] for _ in range(RESAMPLES)))
    ece_ci = (round(boot[int(0.025 * RESAMPLES)], 4), round(boot[int(0.975 * RESAMPLES) - 1], 4))
    # A calibrated model, by construction: each outcome drawn with the probability
    # the model stated for it. Its error is what chance alone produces at this n.
    null = sorted(_ece(conf, [int(rng.random() < c) for c in conf]) for _ in range(RESAMPLES))
    ece_null95 = null[int(0.95 * RESAMPLES) - 1]

    accuracy = sum(hit) / n
    mean_conf = sum(conf) / n
    gap = mean_conf - accuracy
    brier_mean = sum(brier) / n
    counts = {k: sum(1 for s in labeled if s.label == k) for k in classes}
    per_class = {k: {"predicted": round(prob_sum[k] / n, 4), "observed": round(counts[k] / n, 4),
                     "observed_ci": wilson(counts[k], n)} for k in classes}
    bins = _bins(conf, hit)

    verdict = "measured" if n >= VERDICT_MIN_N else "insufficient"
    direction = "overconfident" if gap > 0 else "underconfident"
    summary = (f"Expected calibration error {fmt(ece, ece_ci)} over {n:,} images: the model's "
               f"stated confidence averages {mean_conf:.3f} against an accuracy of {accuracy:.3f}, "
               f"{direction} by {abs(gap) * 100:.1f} points. A perfectly calibrated model would reach "
               f"up to {ece_null95:.3f} on this many images by chance alone.")
    worst = max(classes, key=lambda k: abs(per_class[k]["predicted"] - per_class[k]["observed"]))
    w = per_class[worst]
    summary += (f" Furthest off on average: {class_name(worst)}, given {w['predicted']:.3f} on average "
                f"and present in {w['observed']:.3f} of the images.")
    if verdict == "insufficient":
        summary += f" With {n} images this is a plausibility check, not a measurement."

    return Finding(
        pillar="performance", metric="calibration", domain="image",
        value={"ece": round(ece, 4), "ece_ci": ece_ci, "ece_if_calibrated": round(ece_null95, 4),
               "brier": round(brier_mean, 4), "brier_ci": mean_ci(brier),
               "mean_confidence": round(mean_conf, 4), "confidence_gap": round(gap, 4),
               "n_calibration": n, "bins": BINS, "per_class_calibration": per_class},
        verdict=verdict, summary=summary,
        details={
            "explain": _EXPLAIN, "better": _BETTER, "per_example": per_example,
            "reliability": bins,
            "chart": {
                "kind": "line", "diagonal": True,
                "title": f"Stated confidence against accuracy — {n:,} images in {len(bins)} groups",
                "x": [round(b["confidence"], 4) for b in bins],
                "y": [round(b["accuracy"], 4) for b in bins],
                "y_lo": [b["accuracy_ci"][0] for b in bins],
                "y_hi": [b["accuracy_ci"][1] for b in bins],
                "hover": [f"stated {b['confidence']:.2f}, right {b['accuracy']:.2f} "
                          f"on {b['n']:,} images" for b in bins],
                "x_title": "Stated confidence", "y_title": "Share actually right",
            },
        },
    )

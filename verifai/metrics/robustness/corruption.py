"""Robustness (image): stability under common corruptions.

Signature: run(model, dataset, ctx) -> Finding

For each image we compare the clean top-1 prediction to the prediction under
each corruption (noise, blur, brightness, JPEG). We report:
  - PREDICTION STABILITY: share of images whose top-1 class is unchanged,
  - mean confidence of the clean-top class after each corruption.

A model that flips its call under mild, clinically-irrelevant perturbations is
fragile — a real concern for a decision-support tool.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from verifai.core.findings import Finding
from verifai.metrics._common import CORRUPTIONS
from verifai.metrics._stats import fmt, mean_ci, wilson


def run(model, dataset, ctx: dict[str, Any]) -> Finding:
    seed = ctx.get("seed", 42)
    rng = np.random.default_rng(seed)

    names = list(CORRUPTIONS)
    stable = {c: 0 for c in names}
    conf_after = {c: [] for c in names}
    per_image: list[float] = []      # share of corruptions each image survived
    n = 0

    for s in dataset:
        img = dataset.load(s)
        clean = model.predict_probs(img)
        clean_top = model.decide(clean, getattr(s, "meta", None))
        n += 1
        kept = 0
        for c in names:
            corrupted = CORRUPTIONS[c](img, rng=rng) if c == "noise" else CORRUPTIONS[c](img)
            probs = model.predict_probs(corrupted)
            if model.decide(probs, getattr(s, "meta", None)) == clean_top:
                stable[c] += 1
                kept += 1
            conf_after[c].append(probs[clean_top])
        per_image.append(kept / len(names))

    stability = {c: round(stable[c] / n, 3) for c in names} if n else {}
    stability_ci = {c: wilson(stable[c], n) for c in names} if n else {}
    mean_stability = round(float(np.mean(list(stability.values()))), 3) if stability else None
    # The mean over corruptions equals the mean over images of the share each
    # survived, so the interval is taken over images — the unit that varies.
    mean_stability_ci = mean_ci(per_image)
    weakest = min(names, key=lambda c: stability[c]) if stability else None

    # No stability threshold. What counts as robust enough depends on the
    # deployment, and this metric cannot see it — a model that is confidently and
    # consistently wrong scores a perfect 1.0 here, so a "pass" would have been
    # actively misleading. The share of predictions that flip is the finding.
    verdict = "insufficient" if (n < 20 or mean_stability is None) else "measured"

    note = "" if n >= 20 else f" Small sample (n={n}) — illustrative only."
    return Finding(
        pillar="robustness", metric="corruption_stability", domain="image",
        value={"prediction_stability": stability, "stability_ci": stability_ci,
               "mean_stability": mean_stability, "mean_stability_ci": mean_stability_ci,
               "n": n},
        verdict=verdict,
        summary=(f"The top-1 prediction stays the same under a corruption in "
                 f"{fmt(mean_stability, mean_stability_ci)} of cases on average, over "
                 f"{n:,} images and {len(names)} corruptions ({', '.join(names)}). "
                 f"Least stable under {weakest}: "
                 f"{fmt(stability[weakest], stability_ci[weakest])}.{note}"
                 if mean_stability is not None else "No images evaluated."),
        details={
            "better": {"mean_stability": "higher", "prediction_stability.*": "higher"},
            "explain": {
                "what": ("Real images are never clean: sensor noise, a slightly out-of-"
                         "focus shot, harsh lighting, heavy JPEG compression. None of that "
                         "changes the diagnosis, so the model's answer should not change "
                         "either. This applies each distortion and checks whether it does."),
                "how": ("Each bar is one kind of distortion. The height is the share of "
                        "images whose top-1 class stayed the same after it was applied — "
                        "1.0 means the model never changed its mind, 0.5 means it flipped "
                        "on half the images. Short bars point at the distortion this model "
                        "is most brittle against."),
                "limits": ("Stability is not correctness: a model that is confidently wrong "
                           "both before and after a distortion scores a perfect 1.0 here. "
                           "Read this next to the performance pillar, never on its own."),
            },
            "chart": {
                "kind": "bar", "title": "Prediction stability per corruption",
                "x": names, "y": [stability[c] for c in names], "color": "#1F8A70",
                "y_lo": [stability_ci[c][0] for c in names],
                "y_hi": [stability_ci[c][1] for c in names],
                "hover": [f"{stability[c]:.3f} "
                          f"[{stability_ci[c][0]:.2f}–{stability_ci[c][1]:.2f}] of {n:,}"
                          for c in names],
                "x_title": "Corruption", "y_title": "Share of unchanged top-1",
            },
        },
    )

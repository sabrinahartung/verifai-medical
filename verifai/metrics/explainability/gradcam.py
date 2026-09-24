"""Explainability (image): Grad-CAM overlays + a deletion-faithfulness score.

Signature: run(model, dataset, ctx) -> Finding

The Grad-CAM itself is ported verbatim (semantics-wise) from
ML_Training_Dojo/streamlit_app.py: last residual block, gradient-weighted
activations, ReLU (evidence *for* the class only), no per-map normalisation.

Added here (so the dashboard can quantify, not just illustrate):
  - overlays saved as PNGs for the first few images (pictures are what make an
    artifact heavy, so only they are capped),
  - a *deletion faithfulness* score over EVERY test image: grey out the
    most-attended region and measure how far the class probability drops,
  - a *random-region control* under identical conditions: the same region moved
    to a random place. The claim is the difference between the two — whether the
    highlight matters more than chance — with a 95% interval over the images.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from verifai.core.findings import Finding

FLAT_EPS = 1e-6


def _gradcam(torch_model, layer, x, class_idx):
    """Raw (un-normalised) Grad-CAM tensor for `class_idx`. See module docstring.

    `layer` comes from the model adapter (model.cam_layer), so this works for
    any architecture rather than assuming a ResNet. Returns a CPU tensor so the
    numpy/PIL code below stays device-agnostic.
    """
    activations, gradients = {}, {}
    handles = [
        layer.register_forward_hook(lambda m, i, o: activations.update(v=o)),
        layer.register_full_backward_hook(lambda m, gi, go: gradients.update(v=go[0])),
    ]
    try:
        logits = torch_model(x)
        torch_model.zero_grad(set_to_none=True)
        logits[0, class_idx].backward()
    finally:
        for h in handles:
            h.remove()
    acts, grads = activations["v"][0], gradients["v"][0]  # [C,h,w]
    weights = grads.mean(dim=(1, 2), keepdim=True)
    return (weights * acts).sum(dim=0).relu().detach().cpu()


def _overlay(img, cam, scale, alpha=0.5):
    import numpy as np
    import matplotlib
    from PIL import Image
    cam = (cam / scale).clamp(0, 1)
    img = img.convert("RGB")
    cam_img = Image.fromarray((cam.numpy() * 255).astype("uint8")).resize(img.size, Image.BICUBIC)
    heat = matplotlib.colormaps["jet"](np.asarray(cam_img) / 255.0)[..., :3]
    return Image.blend(img, Image.fromarray((heat * 255).astype("uint8")), alpha)


def _cam_mask(cam, size, frac=0.2):
    """The top-`frac` most-attended pixels, at image resolution, or None if flat."""
    import numpy as np
    from PIL import Image
    cam_img = np.asarray(
        Image.fromarray(cam.numpy().astype("float32")).resize(size, Image.BICUBIC),
        dtype=np.float32)
    if cam_img.max() <= FLAT_EPS:
        return None
    return cam_img >= np.quantile(cam_img, 1.0 - frac)


def _shifted(mask, rng):
    """The same region, moved to a random place in the image.

    The control has to differ from the highlight in *where* it is and in nothing
    else — same size, same shape. Scattering the same number of random pixels
    would hide fine texture everywhere at once and measure something else.
    """
    import numpy as np
    h, w = mask.shape
    return np.roll(mask, (int(rng.integers(h)), int(rng.integers(w))), axis=(0, 1))


def _drop(model, img, mask, p0, cls):
    """How far the class probability falls when `mask` is greyed out, in [0, 1]."""
    import numpy as np
    from PIL import Image
    arr = np.asarray(img.convert("RGB")).copy()
    arr[mask] = 128
    p1 = model.predict_probs(Image.fromarray(arr))[cls]
    return float(max(0.0, min(1.0, p0 - p1)))


def run(model, dataset, ctx: dict[str, Any]) -> Finding:
    import numpy as np
    from verifai.metrics._stats import mean_ci
    from verifai.metrics.performance.classification import VERDICT_MIN_N

    plot_dir = Path(ctx.get("plot_dir", "plots"))
    plot_dir.mkdir(parents=True, exist_ok=True)
    classes = model.classes
    tm = model.torch_module
    cam_layer = model.cam_layer
    rng = np.random.default_rng(ctx.get("seed", 42))

    # Every test image is scored; only the first few are drawn. The cap used to
    # limit both, so the published faithfulness was a mean over the first seven
    # filenames of a sorted manifest — six of them nevi. Pictures are what make
    # an artifact heavy; scores are not.
    max_overlays = int(ctx.get("scenario", {}).get("gradcam_max_images", 7))
    rel_plots: list[str] = []
    captions: list[str] = []
    faith: list[float] = []
    control: list[float] = []

    for i, s in enumerate(dataset):
        img = dataset.load(s)
        probs = model.predict_probs(img)
        top = model.decide(probs, getattr(s, "meta", None))
        ci = classes.index(top)

        x = model.to_tensor(img)
        x.requires_grad_(True)
        cam = _gradcam(tm, cam_layer, x, ci)

        if i < max_overlays:
            out = _overlay(img, cam, max(FLAT_EPS, float(cam.max())))
            fname = f"gradcam_{s.id}.png"
            out.save(plot_dir / fname)
            rel_plots.append(f"plots/{fname}")
            captions.append(f"{s.id}: {top} ({probs[top]*100:.0f}%)")

        mask = _cam_mask(cam, img.size)
        if mask is None:          # a flat map highlights nothing: no region to test
            continue
        faith.append(_drop(model, img, mask, probs[top], top))
        control.append(_drop(model, img, _shifted(mask, rng), probs[top], top))

    n = len(faith)
    gains = [f - c for f, c in zip(faith, control)]
    mean = (lambda xs: round(sum(xs) / len(xs), 4) if xs else None)
    mf, mr, gain = mean(faith), mean(control), mean(gains)
    gain_ci = mean_ci(gains)
    verdict = "measured" if n >= VERDICT_MIN_N else "insufficient"
    small = (f" Small sample (n={n}) — a plausibility check, not a benchmark."
             if n < VERDICT_MIN_N else "")

    summary = (f"Masking the region Grad-CAM highlights lowers the model's confidence by "
               f"{mf:.3f} on average over {n:,} images, against {mr:.3f} for a region of the "
               f"same size placed at random — a difference of {gain:.3f}"
               + (f" [{gain_ci[0]:.3f}–{gain_ci[1]:.3f}]" if gain_ci else "") + "." + small
               if n else "Grad-CAM produced no non-flat map, so no region could be tested.")

    def bar_ci(xs):
        c = mean_ci(xs)
        return c or (None, None)
    (f_lo, f_hi), (r_lo, r_hi) = bar_ci(faith), bar_ci(control)

    return Finding(
        pillar="explainability",
        metric="gradcam_faithfulness",
        domain="image",
        value={"n": n, "n_overlays": len(rel_plots),
               "mean_deletion_faithfulness": mf, "mean_random_control": mr,
               "faithfulness_gain": gain, "faithfulness_gain_ci": list(gain_ci) if gain_ci else None},
        verdict=verdict,
        summary=summary,
        details={
            "explain": {
                "what": ("Grad-CAM highlights the part of the image that pushed the model towards "
                         "its answer. To check the highlight is honest, that region is greyed out "
                         "and we measure how far the model's confidence falls — and then do the "
                         "same with a region of the same size and shape placed at random. A "
                         "highlight that matters should hurt the prediction more than a random "
                         "one; if it does not, it is decoration."),
                "how": ("The overlays show where the model looked for a few example images: warm "
                        "colours drove the decision, blue barely mattered. The bars compare the "
                        "average confidence lost when the highlighted region is hidden with the "
                        "loss when a random region of the same size is hidden, each with its 95% "
                        "interval, over every test image. The gap between the two bars is the "
                        "finding; the height of either bar alone says little."),
                "limits": ("A highlight that clearly beats chance shows the model relies on that "
                           "region — not that relying on it is medically right. It says where the "
                           "model looked, never whether it looked for the right reason. And greying "
                           "out pixels produces images the model never saw in training, which can "
                           "lower confidence for reasons of its own; the random control shares that "
                           "effect, which is why the comparison, not the raw drop, is reported."),
            },
            "better": {"mean_deletion_faithfulness": "higher", "faithfulness_gain": "higher"},
            "target_layer": getattr(model, "cam_layer_path", "layer4[-1]"),
            "faithfulness_per_image": [round(f, 3) for f in faith],
            "random_control_per_image": [round(c, 3) for c in control],
            "chart": {"kind": "images", "title": "Where the model looks (Grad-CAM)",
                      "paths": rel_plots, "captions": captions},
            "chart2": {
                "kind": "bar",
                "title": f"Confidence lost when a region is masked out — {n:,} images",
                "x": ["Grad-CAM's highlighted region", "A random region of the same size"],
                "y": [mf or 0.0, mr or 0.0],
                "y_lo": [f_lo, r_lo] if f_lo is not None else None,
                "y_hi": [f_hi, r_hi] if f_hi is not None else None,
                "colors": ["#5B3FD6", "#9AA5B1"],
                "x_title": "Region masked", "y_title": "Mean drop in confidence",
            },
        },
        plots=rel_plots,
    )

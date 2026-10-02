"""One vocabulary for how an image becomes a tensor, and a field-by-field comparison.

A torchvision pipeline, a Hugging Face image processor and a training record each
describe preprocessing in their own words. Before two of them can be compared they
are written in the same six fields:

    resize          {"height", "width"}, or {"shortest_edge"[, "longest_edge"]}
    center_crop     {"height", "width"}, or None when there is no crop
    resample        the interpolation filter, by name
    rescale_factor  what raw pixel values are multiplied by (1/255 for ToTensor)
    mean, std       the normalisation, per channel

`crop_pct` is carried when a processor states it (ConvNeXt resizes to
`shortest_edge / crop_pct` before cropping), because it changes the pixels too.

This compares parameters, not pixels. Two libraries given the same parameters can
still interpolate slightly differently; that is below what this check can see.

Pure Python, no torch: the metric, both adapters and the tests use it.
"""
from __future__ import annotations

from typing import Any

FIELDS = ("resize", "center_crop", "resample", "rescale_factor", "mean", "std", "crop_pct")

# PIL's resampling enum, which Hugging Face processors store as an integer.
RESAMPLE = {0: "nearest", 1: "lanczos", 2: "bilinear", 3: "bicubic", 4: "box", 5: "hamming"}


def _hw(size: Any) -> dict[str, int] | None:
    if size is None:
        return None
    if isinstance(size, int):
        return {"height": size, "width": size}
    if isinstance(size, (list, tuple)) and len(size) == 2:
        return {"height": int(size[0]), "width": int(size[1])}
    return {k: int(v) for k, v in dict(size).items() if v is not None}


def torchvision_spec(size: int, mean, std) -> dict[str, Any]:
    """What `verifai.models.image.build_preprocess` does, in the shared vocabulary.

    `transforms.Resize((s, s))` with its default bilinear filter, no crop,
    `ToTensor()` (which divides by 255), then `Normalize(mean, std)`.
    """
    return {"resize": {"height": int(size), "width": int(size)}, "center_crop": None,
            "resample": "bilinear", "rescale_factor": 1 / 255,
            "mean": [float(x) for x in mean], "std": [float(x) for x in std]}


def processor_spec(pp: dict[str, Any]) -> dict[str, Any]:
    """A `preprocessor_config.json` (or `processor.to_dict()`) in the shared vocabulary.

    A step a processor switches off (`do_resize: false`, …) is recorded as None,
    which is different from a field the processor does not state at all.
    """
    out: dict[str, Any] = {}
    if "size" in pp or "do_resize" in pp:
        out["resize"] = _hw(pp.get("size")) if pp.get("do_resize", True) else None
    if "do_center_crop" in pp or "crop_size" in pp:
        out["center_crop"] = _hw(pp.get("crop_size")) if pp.get("do_center_crop") else None
    if "resample" in pp:
        r = pp["resample"]
        out["resample"] = RESAMPLE.get(r, str(r)) if isinstance(r, int) else str(r).lower()
    if "rescale_factor" in pp or "do_rescale" in pp:
        out["rescale_factor"] = (float(pp.get("rescale_factor", 1 / 255))
                                 if pp.get("do_rescale", True) else None)
    if "image_mean" in pp or "image_std" in pp or "do_normalize" in pp:
        normalise = pp.get("do_normalize", True)
        out["mean"] = [float(x) for x in pp["image_mean"]] if normalise and pp.get("image_mean") else None
        out["std"] = [float(x) for x in pp["image_std"]] if normalise and pp.get("image_std") else None
    if pp.get("crop_pct") is not None:
        out["crop_pct"] = float(pp["crop_pct"])
    return out


def record_spec(record: dict[str, Any]) -> dict[str, Any]:
    """What a `<stem>_training.json` states about preprocessing.

    Records written since C4 carry the full spec under `preprocessing`. Older ones
    state only `image_size`, so only the resize can be compared against them.
    """
    if record.get("preprocessing"):
        return dict(record["preprocessing"])
    if record.get("image_size"):
        s = int(record["image_size"])
        return {"resize": {"height": s, "width": s}}
    return {}


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        try:
            return abs(float(a) - float(b)) < 1e-6
        except (TypeError, ValueError):
            return False
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    return a == b


def compare(used: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    """Each field the reference states, checked against what the evaluation used."""
    compared = [f for f in FIELDS if f in reference]
    differing = [{"field": f, "used": used.get(f), "reference": reference[f]}
                 for f in compared if not _same(used.get(f), reference[f])]
    return {"compared": compared, "not_stated": [f for f in FIELDS[:6] if f not in reference],
            "differences": differing}

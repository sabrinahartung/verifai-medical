"""Bucketing for context priors — one implementation, used by both sides.

The prior is built by `scripts/build_context_prior.py` and applied by
`verifai/models/image.py::ImageClassifier.context_lift`. Those two must bucket a
raw manifest value identically: if the builder files a 63-year-old under `60-79`
and the adapter looks up `60-69`, every lookup misses, every lift silently falls
back to 1.0, and the experiment reports "context does not help" while never
having applied any context at all.

A failure like that leaves no trace — no error, no warning, just a neutral
result — so the defence is structural rather than a test. There is one
implementation and both sides import it.

Kept free of heavy imports: `showcase/app.py` reaches into `verifai.core`, and
the showcase must not need torch.
"""
from __future__ import annotations

# 20-year bands. A per-year prior would put a few dozen cases in most cells,
# which is noise dressed as precision; a band this wide keeps thousands per cell
# while still separating the ends, where the real signal is (melanoma runs ~0.09
# lift under 20 against ~1.5 over 80).
AGE_BAND = 20


def age_bucket(raw) -> str | None:
    """`"63.0"` -> `"60-79"`. Unparseable or absent -> None, meaning no evidence."""
    try:
        age = float(raw)
    except (TypeError, ValueError):
        return None
    if age < 0:
        return None
    lo = int(age // AGE_BAND) * AGE_BAND
    return f"{lo}-{lo + AGE_BAND - 1}"


def site_bucket(raw) -> str | None:
    """Body site, normalised. Already categorical, so only case and spacing."""
    if raw is None:
        return None
    text = str(raw).strip().lower()
    return text or None


BUCKETERS = {"age": age_bucket, "localization": site_bucket}


def bucket_for(feature: str, value) -> str | None:
    """Bucket `value` for `feature`, or None when there is nothing to go on.

    None is meaningful: it means this case contributes no context evidence, and
    the caller must treat that as a lift of exactly 1.0 rather than guessing a
    bucket. Ten percent of training rows carry no body site, and inventing one
    for them would manufacture evidence out of a blank field.
    """
    bucketer = BUCKETERS.get(feature)
    return bucketer(value) if bucketer else None

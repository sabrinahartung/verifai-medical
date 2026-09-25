"""Integrity: could the model's training corpus contain these test images?

Signature: run(model, dataset, ctx) -> Finding

The split check compares rows. This compares archives: HAM10000 is part of ISIC
2019, so a model trained on ISIC 2019 has seen HAM10000 images unless someone
held them back row by row. For a model trained here that is checkable, and this
finding defers to the row-level result. For a downloaded model it is not, and
this is the only leakage check left — it can say leakage is *possible*, never
that it did not happen.

The ancestry is data, in `data/corpora.yaml`, each containment with its
reference. An archive missing from the table is unknown, never independent.
"""
from __future__ import annotations

from typing import Any

from verifai.core.findings import Finding
from verifai.core.integrity import (declared_training, load_corpora, row_check,
                                    shared_corpora)

EXPLAIN = {
    "what": ("Public image archives are often built out of each other: the ISIC 2019 "
             "collection includes all of HAM10000. A model trained on the bigger archive has "
             "therefore seen images from the smaller one, and a test on the smaller one may "
             "ask it questions it has already answered. This checks whether the archive the "
             "model trained on and the archive these test images come from overlap at all."),
    "how": ("No shared archive means this kind of leakage is impossible. A shared archive "
            "means it is possible — and then the question is whether the overlap was removed "
            "image by image, which only the split check above can say. When that check cannot "
            "run, leakage cannot be ruled out, and every number below should be read as an "
            "upper bound on how well the model generalises."),
    "limits": ("It knows only the containment listed in the project's archive table, each "
               "with its source. Two archives that share images without saying so are not "
               "caught, and an archive missing from the table is reported as unknown rather "
               "than assumed to be separate."),
}


def _finding(verdict: str, summary: str, value: dict[str, Any], domain: str) -> Finding:
    return Finding(pillar="integrity", metric="corpus_ancestry", domain=domain, value=value,
                   verdict=verdict, summary=summary, details={"explain": EXPLAIN})


def run(model, dataset, ctx: dict[str, Any]) -> Finding:
    scenario = ctx.get("scenario", {}) or {}
    domain = scenario.get("domain", "image")
    corpora = load_corpora()
    n = len(dataset)
    declared = declared_training(scenario)
    evaluated_on = (scenario.get("dataset") or {}).get("corpus")
    name = lambda c: (corpora.get(c) or {}).get("name", c)             # noqa: E731
    value: dict[str, Any] = {"trained_on": declared["corpora"], "evaluated_on": evaluated_on,
                             "shared": [], "held_back_row_by_row": None}

    if not declared["corpora"] or not evaluated_on:
        missing = ("the model's training corpus" if not declared["corpora"]
                   else "the corpus these test images come from")
        return _finding("unavailable",
                        f"Not checked for the {n:,} test images: {missing} is not declared, "
                        f"so whether the two archives overlap is unknown.", value, domain)
    unknown = [c for c in declared["corpora"] + [evaluated_on] if c not in corpora]
    if unknown:
        value["unknown"] = unknown
        return _finding("unavailable",
                        f"Not checked for the {n:,} test images: {', '.join(unknown)} "
                        f"{'is' if len(unknown) == 1 else 'are'} not in the project's archive "
                        f"table, and an unlisted archive is never assumed to be separate.",
                        value, domain)

    shared = shared_corpora(declared["corpora"], evaluated_on, corpora)
    value["shared"] = shared
    trained = ", ".join(name(c) for c in declared["corpora"])
    if not shared:
        return _finding("measured",
                        f"No shared archive: the model trained on {trained}, and the "
                        f"{n:,} test images come from {name(evaluated_on)}, which neither "
                        f"contains nor is contained in it.", value, domain)

    audit = row_check(scenario, (dataset.meta or {}).get("manifest"))
    overlap = _overlap_sentence(shared[0], evaluated_on, corpora, n)
    if audit is None:
        basis = f" ({declared['basis']})" if declared["basis"] else ""
        return _finding("insufficient",
                        f"{overlap}{basis}. No image-by-image check is possible, so leakage "
                        f"cannot be ruled out: read every result below as possibly inflated.",
                        value, domain)
    value["held_back_row_by_row"] = bool(audit["clean"])
    if audit["clean"]:
        return _finding("measured",
                        f"{overlap}. The overlap was held back: the image-by-image check finds "
                        f"none of them among the {audit['n_train']:,} the model trained on.",
                        value, domain)
    return _finding("invalid",
                    f"{overlap}, and the overlap was not removed: {audit['affected_rows']:,} "
                    f"of them were seen in training.", value, domain)


def _overlap_sentence(trained: str, evaluated: str, corpora: dict[str, dict[str, Any]],
                      n: int) -> str:
    """How the two archives relate, in the direction that is true."""
    from verifai.core.integrity import descendants
    name = lambda c: (corpora.get(c) or {}).get("name", c)             # noqa: E731
    if trained == evaluated:
        how = f"the archive the {n:,} test images come from"
    elif evaluated in descendants(trained, corpora):
        how = f"which contains {name(evaluated)}, where the {n:,} test images come from"
    else:
        how = f"which is part of {name(evaluated)}, where the {n:,} test images come from"
    return f"The model trained on {name(trained)}, {how}"

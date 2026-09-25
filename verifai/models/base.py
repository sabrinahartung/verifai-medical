"""The model contract, and how much of a model an evaluation can reach.

The contract was already domain-neutral before it was written down: four of the
six metrics need only `predict_probs`, `decide` and `rank`. What was missing is
the second half — a model saying *how much of itself it exposes*, so a metric
that cannot run against it is reported as unavailable, with the reason, instead
of crashing or silently vanishing from the report.

Deliberately free of torch: the runner, the registry exporter and the tests
import it, and so may the showcase.
"""
from __future__ import annotations

from typing import Any, Literal, Protocol, runtime_checkable

# From least to most. Each level includes every level before it: whoever can
# take gradients can read probabilities, and whoever can read probabilities can
# read the predicted label. A total order, so gating is one comparison.
#
#   labels         the predicted class, nothing else
#   probs          a score for every class (a hosted classifier)
#   logits         raw pre-softmax outputs
#   gradients      backward passes through the model
#   weights        permission to modify the parameters and restore them
#   training_data  the manifests it was trained on are declared and checkable
ACCESS_LEVELS: tuple[str, ...] = ("labels", "probs", "logits", "gradients", "weights",
                                  "training_data")
Access = Literal["labels", "probs", "logits", "gradients", "weights", "training_data"]

# What `dataset.load(sample)` hands a model.
Modality = Literal["pixels", "tokens", "audio", "rows"]

# What decides which metrics apply. Separate from the scenario's `domain:`, which
# names the payload type — the old `Domain` literal listed "llm" beside "image",
# a task beside a data type.
Task = Literal["classification", "generation"]
DEFAULT_TASK: Task = "classification"


def access_rank(level: str) -> int:
    """Position on the ladder; an unknown level is a configuration error, not a guess."""
    try:
        return ACCESS_LEVELS.index(level)
    except ValueError:
        raise ValueError(f"unknown access level {level!r}; expected one of "
                         f"{', '.join(ACCESS_LEVELS)}") from None


def reaches(has: str, needs: str) -> bool:
    """Whether a model at access level `has` supports a metric that needs `needs`."""
    return access_rank(has) >= access_rank(needs)


# Why a level is out of reach, in the words a reader needs. Keyed by the level
# the metric needs; the sentence is completed with the level the model has.
WHY_UNREACHABLE: dict[str, str] = {
    "probs": "it needs a score for every class, and this model returns only its predicted label",
    "logits": "it needs the model's raw outputs, and this model returns only normalised scores",
    "gradients": "it needs gradients, and this model can only be queried, not opened, so "
                 "gradients do not exist for it",
    "weights": "it needs to modify the model's parameters and restore them, and this model's "
               "weights cannot be changed",
    "training_data": "it needs to know what the model was trained on, and no training data "
                     "is declared for this model",
}


@runtime_checkable
class ModelAdapter(Protocol):
    """What every model must offer, whatever it is and wherever it runs.

    `decide` and `rank` exist so no metric hardcodes argmax: a scenario's
    decision weights apply everywhere at once. `meta` is the sample's metadata
    and may be None.
    """

    classes: list[str]
    access: str
    modality: str

    def predict_probs(self, payload: Any) -> dict[str, float]: ...

    def decide(self, probs: dict[str, float], meta: dict | None = None) -> str: ...

    def rank(self, probs: dict[str, float], meta: dict | None = None) -> list[str]: ...

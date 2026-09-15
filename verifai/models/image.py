"""Image-domain model adapter for torchvision classifiers.

Loads an already-trained model (from Hugging Face, or a local .pt) and exposes
the exact preprocessing + prediction used at training time, so every metric sees
the model behave identically to the original demo.

Nothing here is skin-specific any more: the class list, the architecture, the
Grad-CAM target layer and the device all come from the scenario's `model:` block.
The HAM10000 defaults below keep existing scenarios working unchanged.

Originally mirrored ML_Training_Dojo/streamlit_app.py.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

# --- Default classes: exact sorted order used when training the HAM10000 model.
# A scenario may override this with `model.classes`; the order must match the
# checkpoint's output layer, so it is never derived from the data.
CLASSES = [
    "actinic_keratoses",
    "basal_cell_carcinoma",
    "benign_keratosis-like_lesions",
    "dermatofibroma",
    "melanocytic_Nevi",
    "melanoma",
    "vascular_lesions",
]

# ImageNet normalization + 224x224, exactly as in the original training/demo.
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Sensible Grad-CAM target per family: the last conv block before pooling.
DEFAULT_CAM_LAYER = {
    "resnet18": "layer4[-1]", "resnet34": "layer4[-1]",
    "resnet50": "layer4[-1]", "densenet121": "features",
    "efficientnet_b0": "features[-1]", "mobilenet_v3_small": "features[-1]",
}


def build_preprocess(size: int = 224, mean=None, std=None):
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean or MEAN, std or STD),
    ])


def resolve_device(name: str | None = "auto"):
    """'auto' picks the fastest available backend; an explicit name wins.

    The old code loaded to CPU and never moved the model, so the free-GPU
    notebook silently ran on CPU and the Mac's MPS was never used.
    """
    import torch
    if name and name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _resolve_module(root, path: str):
    """Resolve a layer path like 'layer4[-1]' or 'features[-1].conv' on a module."""
    obj = root
    for part in path.split("."):
        m = re.fullmatch(r"([A-Za-z_]\w*)(?:\[(-?\d+)\])?", part)
        if not m:
            raise ValueError(f"cannot parse layer path segment {part!r} in {path!r}")
        obj = getattr(obj, m.group(1))
        if m.group(2) is not None:
            obj = obj[int(m.group(2))]
    return obj


# Shared with scripts/build_context_prior.py — see verifai/core/context.py for
# why there is exactly one implementation of this.
from verifai.core.context import bucket_for as _bucket_for  # noqa: E402


class ImageClassifier:
    """Thin wrapper around a torch module + the metadata metrics need.

    Metrics use `.torch_module` and `.cam_layer` (for hooks / Grad-CAM) and the
    convenience `.predict_probs(pil_image)` / `.to_tensor(pil_image)` helpers.
    """

    def __init__(self, model, classes: list[str], device=None,
                 cam_layer: str = "layer4[-1]", preprocess=None,
                 decision_weights: dict[str, float] | None = None,
                 context_prior: dict | None = None,
                 prior_strength: float = 1.0):
        import torch
        self.device = device if device is not None else torch.device("cpu")
        self.model = model.to(self.device)
        self.classes = classes
        self.cam_layer_path = cam_layer
        self._pre = preprocess or build_preprocess()
        self.decision_weights = dict(decision_weights or {})
        self.context_prior = context_prior or None
        self.prior_strength = float(prior_strength)

    @property
    def torch_module(self):
        return self.model

    @property
    def cam_layer(self):
        """The module Grad-CAM hooks onto — configurable per architecture."""
        return _resolve_module(self.model, self.cam_layer_path)

    def to_tensor(self, img):
        return self._pre(img.convert("RGB")).unsqueeze(0).to(self.device)  # [1,3,H,W]

    def context_lift(self, meta: dict | None) -> dict[str, float]:
        """Per-class multiplier from this case's non-image context.

        Bayes with the model as the likelihood: multiplying by
        `P(class|context)/P(class)` turns `P(class|image)` into
        `P(class|image, context)`. The table comes from the *training* manifest
        (`scripts/build_context_prior.py`), so nothing here has seen test labels.

        Features are combined as if independent, which they are not — age and
        body site correlate. Naive Bayes over two features is a mild
        approximation and `prior_strength` is tuned on validation partly to
        absorb it.

        An absent or unknown value contributes exactly 1.0. Missing context must
        never push a decision: 10% of training rows have no site recorded, and
        guessing one for them would invent evidence.
        """
        if not self.context_prior or self.prior_strength == 0:
            return {c: 1.0 for c in self.classes}
        lift = {c: 1.0 for c in self.classes}
        features = (self.context_prior.get("features") or {})
        for feature, table in features.items():
            bucket = _bucket_for(feature, (meta or {}).get(feature))
            cell = table.get(bucket) if bucket else None
            if not cell:
                continue
            for c in self.classes:
                lift[c] *= float(cell["lift"].get(c, 1.0)) ** self.prior_strength
        return lift

    def _score(self, probs: dict[str, float], meta: dict | None):
        w = self.decision_weights
        lift = self.context_lift(meta)
        return lambda c: probs[c] * w.get(c, 1.0) * lift.get(c, 1.0)

    def decide(self, probs: dict[str, float], meta: dict | None = None) -> str:
        """Turn a probability vector into an answer.

        `argmax` is the default, but it is a *choice*, not a law — it maximises
        expected accuracy, which on imbalanced data means systematically
        under-calling rare classes. `decision_weights` scales each class by the
        cost of missing it, and `context_prior` scales it by what this patient's
        age and lesion site say before the image is even looked at. In both cases
        the model itself is untouched; only the reading of its output changes.

        `meta` is optional so every existing caller keeps working unchanged —
        without it the context prior is simply inert.

        Every metric routes its decision through here, so the rule is configured
        once per scenario rather than reimplemented per metric.
        """
        if not self.decision_weights and not self.context_prior:
            return max(probs, key=probs.get)
        return max(probs, key=self._score(probs, meta))

    def rank(self, probs: dict[str, float], meta: dict | None = None) -> list[str]:
        """Classes best-first under the same rule — for top-k differential metrics."""
        if not self.decision_weights and not self.context_prior:
            return sorted(probs, key=probs.get, reverse=True)
        return sorted(probs, key=self._score(probs, meta), reverse=True)

    def predict_probs(self, img) -> dict[str, float]:
        import torch
        x = self.to_tensor(img)
        with torch.no_grad():
            probs = self.model(x).softmax(dim=1)[0].cpu()
        return {self.classes[i]: float(probs[i]) for i in range(len(self.classes))}


# Kept so older imports/pickles keep resolving.
SkinLesionModel = ImageClassifier


def load(spec: dict[str, Any]) -> ImageClassifier:
    """spec example:
    {loader: "verifai.models.image:load",
     id: "skin-lesion-resnet18",
     repo_id: "sabrinahartung1010/skin-lesion-resnet18",
     filename: "resnet18_ham10000_classweights.pt",
     weights_path: "/optional/local/override.pt",  # skips the HF download
     arch: "resnet18",          # any torchvision classifier factory
     classes: [...],            # must match the checkpoint's output order
     cam_layer: "layer4[-1]",   # Grad-CAM target
     device: "auto",            # auto | cpu | cuda | mps
     decision_weights: {melanoma: 2.5},   # cost-sensitive rule; default is argmax
     context_prior: "data/priors/isic.json",  # age/site lifts, built from TRAINING
     prior_strength: 1.0}                 # 0 disables it; tuned on validation
    """
    import torch
    import torchvision.models as tvm

    classes = list(spec.get("classes") or CLASSES)
    arch = spec.get("arch", "resnet18")
    device = resolve_device(spec.get("device", "auto"))

    # A path rather than an inline table: the prior is a measured artifact of one
    # training manifest, so it belongs on disk where it can be read, diffed and
    # pointed at, next to the manifests it was derived from.
    prior = None
    prior_path = spec.get("context_prior")
    if prior_path:
        import json as _json
        path_obj = Path(prior_path)
        if not path_obj.is_absolute():
            path_obj = Path(__file__).resolve().parents[2] / path_obj
        if not path_obj.exists():
            raise FileNotFoundError(
                f"context_prior {prior_path} not found — build it with "
                f"scripts/build_context_prior.py")
        prior = _json.loads(path_obj.read_text(encoding="utf-8"))

    weights_path = spec.get("weights_path")
    if weights_path and Path(weights_path).exists():
        path = weights_path
    else:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id=spec["repo_id"], filename=spec["filename"])

    factory = getattr(tvm, arch, None)
    if factory is None:
        raise ValueError(f"unknown torchvision architecture {arch!r}")
    model = factory(weights=None)

    # swap the classifier head for our class count, wherever this family keeps it
    if hasattr(model, "fc"):                       # resnet family
        model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
    elif hasattr(model, "classifier"):             # densenet / efficientnet / mobilenet
        head = model.classifier
        if isinstance(head, torch.nn.Sequential):
            last = len(head) - 1
            head[last] = torch.nn.Linear(head[last].in_features, len(classes))
        else:
            model.classifier = torch.nn.Linear(head.in_features, len(classes))
    else:
        raise ValueError(f"{arch!r} has no recognised classifier head to resize")

    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    size = int(spec.get("image_size", 224))
    return ImageClassifier(
        model, classes, device=device,
        cam_layer=spec.get("cam_layer") or DEFAULT_CAM_LAYER.get(arch, "layer4[-1]"),
        preprocess=build_preprocess(size, spec.get("mean"), spec.get("std")),
        decision_weights=spec.get("decision_weights"),
        context_prior=prior,
        prior_strength=float(spec.get("prior_strength", 1.0)),
    )

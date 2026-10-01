"""Image-domain adapter for self-describing Hugging Face classifiers.

A `transformers` image classifier ships what a torchvision `state_dict` cannot:
its class order (`config.json` → `id2label`) and its preprocessing
(`preprocessor_config.json`). This adapter reads both from the repository at the
pinned revision, so neither has to be copied into the scenario by hand — and when
the scenario does state them, a disagreement is refused rather than resolved in
either direction.

Everything a metric uses is inherited from `ImageClassifier`: `decide` and `rank`
(with decision weights and the context prior), `to_tensor`, `predict_probs` and
`predict_probs_batch`. What differs is only how the module is built, how an image
becomes a tensor, and where Grad-CAM hooks on.

Grad-CAM needs a convolutional feature map. For the CNN families below the layer is
known; a scenario may name one with `cam_layer`. A Vision Transformer has none, so
`cam_layer` is None and the Grad-CAM finding reads `unavailable`, with the reason.
"""
from __future__ import annotations

from typing import Any

from verifai.models.image import (ImageClassifier, _resolve_module, load_context_prior,
                                  resolve_device)
from verifai.models.preprocessing import processor_spec

# The last convolutional stage before pooling, per `config.model_type`. Each path
# was checked against a randomly initialised model of the family (2026-09-30).
# A family missing here gets no Grad-CAM rather than a guessed layer.
CAM_LAYERS = {
    "resnet": "resnet.encoder.stages[-1]",
    "convnext": "convnext.encoder.stages[-1]",
    "convnextv2": "convnextv2.encoder.stages[-1]",
    "regnet": "regnet.encoder.stages[-1]",
    "mobilenet_v2": "mobilenet_v2.conv_1x1",
}


def _logits_module(hf_model):
    """The HF model as a module whose forward returns a logits tensor.

    `transformers` returns a `ModelOutput`; every metric here, Grad-CAM included,
    expects the tensor a torchvision classifier returns. Wrapping once keeps the
    metrics unaware of where a model came from.
    """
    import torch

    class Logits(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, pixel_values):
            return self.inner(pixel_values=pixel_values).logits

    return Logits(hf_model)


class HFImageClassifier(ImageClassifier):
    """A `transformers` image classifier behind the same contract as a torchvision one."""

    access = "weights"
    modality = "pixels"

    def __init__(self, hf_model, processor, device=None, cam_layer: str | None = None,
                 **kw):
        config = hf_model.config
        classes = [config.id2label[i] for i in range(config.num_labels)]
        self.hf_model = hf_model.eval()
        self.processor = processor
        self.model_type = config.model_type
        super().__init__(_logits_module(self.hf_model).eval(), classes, device=device,
                         cam_layer=cam_layer or "", preprocess=self._pixels, **kw)
        self.cam_layer_path = cam_layer or CAM_LAYERS.get(self.model_type)
        # The processor is the model's own and is used as is, so the reference and
        # what was used are one object. The finding still says so, rather than
        # leaving the one kind of model with no preprocessing question unanswered.
        self.reference_preprocessing = {
            "source": "own_processor", "where": "the model's own image processor, used as is",
            "spec": processor_spec(self.processor.to_dict())}

    def _pixels(self, img):
        """One RGB image -> [3,H,W], exactly as the model's own processor makes it."""
        return self.processor(images=img, return_tensors="pt")["pixel_values"][0]

    @property
    def cam_layer(self):
        """The layer Grad-CAM hooks onto, or None when this architecture has none."""
        if not self.cam_layer_path:
            return None
        return _resolve_module(self.hf_model, self.cam_layer_path)

    @property
    def metadata(self) -> dict:
        import hashlib
        import json as _json
        spec = {"processor": type(self.processor).__name__,
                **processor_spec(self.processor.to_dict())}
        blob = _json.dumps(spec, sort_keys=True, default=str).encode()
        return {"classes": list(self.classes), "preprocessing": spec,
                "preprocessing_sha256": hashlib.sha256(blob).hexdigest()[:16],
                "architecture": type(self.hf_model).__name__}


def load(spec: dict[str, Any]) -> HFImageClassifier:
    """spec example (what `verifai.models.resolve` drafts):
    {loader: "verifai.models.hf_image:load",
     id: "derm-convnext",
     repo_id: "owner/derm-convnext",       # or weights_path: a local save_pretrained directory
     revision: "<commit sha>",            # pins config, processor and weights together
     architecture: "ConvNextForImageClassification",  # optional; checked, never applied
     classes: [...],                      # optional; checked against id2label, never applied
     cam_layer: "convnext.encoder.stages[-1]",        # optional; defaults per family
     device: "auto",
     decision_weights: {...}, context_prior: "...", prior_strength: 1.0}
    """
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    source = spec.get("weights_path") or spec["repo_id"]
    rev = None if spec.get("weights_path") else spec.get("revision")
    model = AutoModelForImageClassification.from_pretrained(source, revision=rev)
    processor = AutoImageProcessor.from_pretrained(source, revision=rev)

    # The repository is the authority on its own class order and architecture.
    # A scenario that says otherwise is a mistake somewhere, and picking either
    # side would silently score a different model than the one described.
    config = model.config
    own = [config.id2label[i] for i in range(config.num_labels)]
    if spec.get("classes") and list(spec["classes"]) != own:
        raise ValueError(f"scenario classes {list(spec['classes'])} differ from the "
                         f"repository's id2label order {own}; the repository's is the "
                         f"output layer's, so fix or delete model.classes")
    archs = list(getattr(config, "architectures", None) or [type(model).__name__])
    if spec.get("architecture") and spec["architecture"] not in archs:
        raise ValueError(f"scenario architecture {spec['architecture']!r} is not the "
                         f"repository's ({', '.join(archs)})")

    return HFImageClassifier(
        model, processor, device=resolve_device(spec.get("device", "auto")),
        cam_layer=spec.get("cam_layer"),
        decision_weights=spec.get("decision_weights"),
        context_prior=load_context_prior(spec.get("context_prior")),
        prior_strength=float(spec.get("prior_strength", 1.0)),
    )

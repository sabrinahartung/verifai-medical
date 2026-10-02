"""Draft a scenario from a model reference — what a checkpoint can say about itself.

`hf:owner/repo[@rev]` or a local path goes in; a scenario marked `draft: true`
comes out. The resolver drafts and never runs: every value the checkpoint cannot
supply is written as a `TODO: …` string saying what has to be declared and why,
and `run_scenario` refuses a draft, or any scenario still holding a TODO. A person
reads what was resolved, answers the TODOs and deletes the flag.

What it reads from the Hub, without downloading the weights:
  - `model_info`: the commit sha (pinned as `revision`), the file list, the card's
    `datasets` and `license`, and whether the repository is gated;
  - `config.json`: `id2label` (the class order) and `architectures`;
  - `preprocessor_config.json`: the preprocessing the model expects.

A torchvision `state_dict` carries no metadata at all — not its architecture,
not its class order, not its preprocessing — so for a bare `.pt` the draft says
which fields must be declared instead of guessing them. The one exception is a
checkpoint this project trained: `scripts/train_model.py` writes a
`<stem>_training.json` record beside it, and that record is read.

Two rules carried over from ADR 0002:
  - a card dataset resolves to a corpus only through `hub_aliases:` in
    `data/corpora.yaml`; an unlisted dataset id stays unknown, never independent;
  - a label map is proposed only where two class names are identical after
    normalising case and punctuation. Every other pairing is a claim somebody has
    to make, so it is left as a TODO.
"""
from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from verifai.core.integrity import REPO_ROOT, load_corpora
from verifai.models.hub_card import parse_card

TODO = "TODO"
WEIGHT_SUFFIXES = (".pt", ".pth", ".bin", ".safetensors")
DEFAULT_METRICS = [
    "integrity.split_leakage", "integrity.provenance", "integrity.corpus_ancestry",
    "integrity.label_space", "integrity.preprocessing", "performance.classification", "explainability.gradcam",
    "robustness.corruption", "fairness.skin_tone", "privacy.mia",
]


def todo(why: str) -> str:
    return f"{TODO}: {why}"


def is_todo(value: Any) -> bool:
    return isinstance(value, str) and value.startswith(TODO)


def open_todos(node: Any, path: str = "") -> list[str]:
    """Every dotted path in a scenario whose value is still a TODO."""
    if isinstance(node, dict):
        return [p for k, v in node.items() for p in open_todos(v, f"{path}.{k}" if path else str(k))]
    if isinstance(node, list):
        return [p for i, v in enumerate(node) for p in open_todos(v, f"{path}[{i}]")]
    return [path] if is_todo(node) else []


@dataclass
class Draft:
    scenario: dict[str, Any]
    source: str
    notes: list[str] = field(default_factory=list)   # what was resolved, and from where

    @property
    def todos(self) -> list[str]:
        return open_todos(self.scenario)

    def to_yaml(self) -> str:
        import yaml
        head = [f"# DRAFT scenario, resolved from {self.source} by verifai.models.resolve.",
                "#",
                "# Nothing here has run. Answer every TODO, then delete `draft: true`;",
                "# run_scenario refuses a draft, and any scenario still holding a TODO.",
                "#"]
        if self.notes:
            head += ["# Resolved:"] + [f"#   - {n}" for n in self.notes] + ["#"]
        todos = self.todos
        head += [f"# Open: {len(todos)} TODO(s)" + (" — " + ", ".join(todos) if todos else "")]
        body = yaml.safe_dump(self.scenario, sort_keys=False, allow_unicode=True, width=100)
        return "\n".join(head) + "\n" + body


# --- references -----------------------------------------------------------------
def parse_ref(ref: str) -> tuple[str, str, str | None]:
    """`hf:owner/repo[@rev]` -> ("hub", repo, rev); anything else is a local path."""
    if ref.startswith("hf:"):
        repo, _, rev = ref[3:].partition("@")
        if repo.count("/") != 1 or not all(repo.split("/")):
            raise ValueError(f"expected hf:owner/repo[@revision], got {ref!r}")
        return "hub", repo, rev or None
    return "local", ref, None


# --- corpora --------------------------------------------------------------------
def corpus_for_dataset(dataset_id: str, corpora: dict[str, dict[str, Any]]) -> str | None:
    """The corpus a Hub dataset id is a copy of, if `data/corpora.yaml` says so."""
    for cid, entry in corpora.items():
        if dataset_id in (entry.get("hub_aliases") or []):
            return cid
    return None


def _trained_on(card_datasets: list[str], corpora: dict[str, dict[str, Any]]) -> tuple[Any, list[str]]:
    if not card_datasets:
        return (todo("the model card names no training data. Declare the corpora if another "
                     "source documents them, or delete this key: an undeclared corpus is "
                     "reported as unknown"), ["model card: no `datasets`"])
    resolved = {d: corpus_for_dataset(d, corpora) for d in card_datasets}
    unknown = sorted(d for d, c in resolved.items() if c is None)
    note = "model card datasets: " + ", ".join(
        f"{d} -> {c or 'unknown'}" for d, c in resolved.items())
    if unknown:
        return (todo(f"the card names {', '.join(unknown)}, which data/corpora.yaml does not "
                     f"list. Add it there with a `hub_aliases` entry and a reference, or "
                     f"delete this key: an unlisted corpus stays unknown"), [note])
    corpora_ids = sorted(set(resolved.values()))
    return ({"corpora": corpora_ids,
             "basis": "model card metadata (datasets: " + ", ".join(card_datasets) + ")"},
            [note])


# --- label map ------------------------------------------------------------------
def _norm(name: str) -> str:
    return re.sub(r"[^0-9a-z]", "", name.lower())


def propose_label_map(model_classes: list[str], data_classes: list[str]) -> dict[str, str]:
    """data label -> model class, only where the names agree after normalising.

    A data label whose name is already a model class needs no entry. Every other
    data label gets a TODO: `MEL` and `melanoma` may well be the same class, but
    saying so is a claim, and the resolver does not make claims.
    """
    by_norm: dict[str, list[str]] = {}
    for c in model_classes:
        by_norm.setdefault(_norm(c), []).append(c)
    out: dict[str, str] = {}
    for d in data_classes:
        if d in model_classes:
            continue
        match = by_norm.get(_norm(d)) or []
        if len(match) == 1:
            out[d] = match[0]
        else:
            out[d] = todo("which model class is this data label, if any? Delete the line if "
                          "none: its images are then scored as always wrong")
    return out


def manifest_classes(manifest: str | Path) -> list[str]:
    path = Path(manifest)
    path = path if path.is_absolute() else REPO_ROOT / path
    with open(path, newline="", encoding="utf-8") as f:
        return sorted({(r.get("label") or "").strip() for r in csv.DictReader(f)} - {""})


# --- preprocessing ----------------------------------------------------------------
def _preprocessing(pp: dict[str, Any] | None) -> tuple[dict[str, Any], list[str]]:
    """Map a `preprocessor_config.json` onto the torchvision loader's fields.

    That loader knows a square resize, a mean and a std. A shortest-edge resize or
    a centre crop is something it cannot reproduce, so it becomes a TODO rather
    than a silently different preprocessing.
    """
    if not pp:
        why = "a state_dict does not record its preprocessing; declare the training-time value"
        return {"image_size": todo(why), "mean": todo(why), "std": todo(why)}, []
    out: dict[str, Any] = {}
    size = pp.get("size")
    if isinstance(size, int):
        size = {"height": size, "width": size}
    if isinstance(size, dict) and size.get("height") and size.get("height") == size.get("width") \
            and not pp.get("do_center_crop"):
        out["image_size"] = int(size["height"])
    else:
        out["image_size"] = todo(f"the processor resizes with {size!r}"
                                 + (f" and centre-crops to {pp.get('crop_size')!r}"
                                    if pp.get("do_center_crop") else "")
                                 + "; the torchvision loader only resizes to a square")
    out["mean"] = pp.get("image_mean") or todo("the processor states no image_mean")
    out["std"] = pp.get("image_std") or todo("the processor states no image_std")
    return out, ["preprocessing from preprocessor_config.json"]


# --- resolving ------------------------------------------------------------------
def _hub_fetch_json(repo: str, filename: str, revision: str) -> dict[str, Any]:
    from huggingface_hub import hf_hub_download
    return json.loads(Path(hf_hub_download(repo_id=repo, filename=filename,
                                           revision=revision)).read_text(encoding="utf-8"))


def _hub_fetch_text(repo: str, filename: str, revision: str) -> str:
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(repo_id=repo, filename=filename,
                                revision=revision)).read_text(encoding="utf-8")


def _card_field(card: Any, key: str) -> Any:
    if card is None:
        return None
    if isinstance(card, dict):
        return card.get(key)
    return getattr(card, key, None)


def _as_list(v: Any) -> list[str]:
    if not v:
        return []
    return [v] if isinstance(v, str) else list(v)


def _classes_from_config(config: dict[str, Any]) -> list[str] | None:
    id2label = config.get("id2label") or {}
    if not id2label:
        return None
    return [id2label[k] for k in sorted(id2label, key=lambda k: int(k))]


def resolve(ref: str, *, dataset_manifest: str | None = None,
            corpora: dict[str, dict[str, Any]] | None = None,
            api: Any = None,
            fetch_json: Callable[[str, str, str], dict[str, Any]] | None = None,
            fetch_text: Callable[[str, str, str], str] | None = None) -> Draft:
    """Resolve a model reference into a draft scenario.

    `api` and `fetch_json` exist for the tests, which run offline: `api` stands in
    for `huggingface_hub.HfApi()`, `fetch_json(repo, filename, revision)` for
    reading a small JSON file from the repository, `fetch_text` for its README.
    """
    corpora = load_corpora() if corpora is None else corpora
    kind, where, rev = parse_ref(ref)
    if kind == "hub":
        model, notes = _resolve_hub(where, rev, corpora, api, fetch_json or _hub_fetch_json,
                                    fetch_text or _hub_fetch_text)
    else:
        model, notes = _resolve_local(where)
    classes = model.get("classes")
    name = re.sub(r"[^0-9a-z]+", "_", where.rsplit("/", 1)[-1].lower()).strip("_")
    scenario: dict[str, Any] = {
        "draft": True,
        "name": name,
        "label": todo("a short human name for this configuration, as the comparison table "
                      "should read it"),
        "project": todo("the problem this model belongs to, e.g. \"Skin lesion classification\""),
        "status": "active",
        "domain": "image",
        "seed": 42,
        "sample_size": None,
        "model": model,
        "dataset": _dataset_block(classes, dataset_manifest),
        "metrics": list(DEFAULT_METRICS),
    }
    if kind == "hub":
        scenario["card"] = {"hf_url": f"https://huggingface.co/{where}"}
    return Draft(scenario=scenario, source=ref, notes=notes)


def _resolve_hub(repo: str, rev: str | None, corpora, api, fetch_json,
                 fetch_text) -> tuple[dict, list[str]]:
    if api is None:
        from huggingface_hub import HfApi
        api = HfApi()
    info = api.model_info(repo, revision=rev)
    sha = info.sha
    files = [s.rfilename for s in (info.siblings or [])]
    notes = [f"revision pinned to commit {sha}"
             + (f" (what {rev!r} pointed at when resolved)" if rev and rev != sha else "")]
    if getattr(info, "gated", False):
        notes.append(f"the repository is gated ({info.gated}): the run needs an accepted licence "
                     f"and a token")

    card = getattr(info, "card_data", None)
    if card is None:
        notes.append("the repository has no model card: nothing states its training data "
                     "or its licence")
    trained_on, tnotes = _trained_on(_as_list(_card_field(card, "datasets")), corpora)
    notes += tnotes
    licence = _card_field(card, "license")
    if licence:
        notes.append(f"licence from the model card: {licence}")

    model: dict[str, Any]
    if "config.json" in files:
        # A self-describing model: the class order and the preprocessing travel with
        # it, so the adapter (C3) reads them from the repository at this commit.
        config = fetch_json(repo, "config.json", sha) or {}
        archs = config.get("architectures") or []
        classes = _classes_from_config(config)
        model = {"loader": "verifai.models.hf_image:load",
                 "id": repo.rsplit("/", 1)[-1], "repo_id": repo, "revision": sha,
                 "architecture": archs[0] if archs else todo("config.json names no architecture"),
                 "classes": classes or todo("config.json has no id2label; declare the class "
                                            "order of the output layer"),
                 "device": "auto"}
        notes.append("class order from config.json id2label" if classes
                      else "config.json has no id2label")
        if "preprocessor_config.json" in files:
            pp = fetch_json(repo, "preprocessor_config.json", sha) or {}
            notes.append("preprocessor_config.json present: the adapter reads it at this commit ("
                         + ", ".join(f"{k}={pp[k]!r}" for k in
                                     ("size", "crop_size", "image_mean", "image_std")
                                     if k in pp) + ")")
        else:
            notes.append("no preprocessor_config.json: the preprocessing cannot be checked "
                         "against the model's own")
            model["preprocessing"] = todo("the repository ships no preprocessor_config.json; "
                                          "declare the training-time preprocessing")
    else:
        weights = [f for f in files if f.endswith(WEIGHT_SUFFIXES)]
        pp = (fetch_json(repo, "preprocessor_config.json", sha)
              if "preprocessor_config.json" in files else None)
        prep, pnotes = _preprocessing(pp)
        notes += pnotes
        model = {"loader": "verifai.models.image:load",
                 "id": repo.rsplit("/", 1)[-1], "repo_id": repo,
                 "filename": weights[0] if len(weights) == 1 else todo(
                     "which file holds the weights? " + (", ".join(weights) or "none found")),
                 "revision": sha,
                 **_state_dict_fields(), **prep, "device": "auto"}
        notes.append("no config.json: a bare state_dict, which records no architecture, class "
                     "order or preprocessing")
    # The model page is headed by `model.name`. The card's own title is the
    # author's name for the model; without one the page falls back to the
    # repository name until somebody declares it.
    title = (parse_card(fetch_text(repo, "README.md", sha)).get("title")
             if "README.md" in files else None)
    if title:
        model = {"name": title, **model}
        notes.append(f"name from the model card's title: {title}")
    model["trained_on"] = trained_on
    model["licence"] = licence or todo(
        "the model card states no licence. Find out whether evaluating the model and "
        "publishing its scores is allowed before this runs")
    return model, notes


def _state_dict_fields() -> dict[str, Any]:
    return {"arch": todo("a state_dict does not record its architecture; name the torchvision "
                         "factory, e.g. resnet18"),
            "classes": todo("a state_dict does not record its class order; list the output "
                            "layer's classes in order"),
            "cam_layer": todo("the Grad-CAM target layer; resnets use layer4[-1]")}


def _resolve_local(path: str) -> tuple[dict, list[str]]:
    p = Path(path)
    record = p.with_name(p.stem + "_training.json")
    model: dict[str, Any] = {"loader": "verifai.models.image:load", "id": p.stem,
                             "weights_path": path}
    if record.is_file():
        # This project trained it, and train_model.py wrote down how.
        data = json.loads(record.read_text(encoding="utf-8"))
        model["arch"] = data.get("arch") or _state_dict_fields()["arch"]
        model["classes"] = data.get("classes") or _state_dict_fields()["classes"]
        if data.get("image_size"):
            model["image_size"] = int(data["image_size"])
        return ({**model, "device": "auto"},
                [f"arch, class order and image size from the training record {record.name}",
                 "trained here: declare `training:` or `integrity.train_manifests` so the split "
                 "is checked row by row"])
    prep, _ = _preprocessing(None)
    model.update(_state_dict_fields())
    model.update(prep)
    model["device"] = "auto"
    model["trained_on"] = todo("a local state_dict names no training data; declare its corpora, "
                               "or delete this key: an undeclared corpus is reported as unknown")
    return model, ["a local state_dict with no training record: every field it cannot "
                   "carry is marked TODO"]


def _dataset_block(model_classes: Any, manifest: str | None) -> dict[str, Any]:
    block: dict[str, Any] = {
        "loader": "verifai.datasets.loaders:load_image_manifest",
        "id": todo("a short id for the evaluation set"),
        "manifest": manifest or todo("the evaluation manifest, e.g. data/manifests/ham10000_test.csv"),
        "images_dir": todo("where the manifest's images live"),
        "corpus": todo("the data/corpora.yaml id the evaluation images come from"),
    }
    if not manifest:
        return block
    data_classes = manifest_classes(manifest)
    if isinstance(model_classes, list):
        mapping = propose_label_map(model_classes, data_classes)
        if mapping:
            block["label_map"] = mapping
    else:
        block["label_map"] = todo("the model's classes are not known yet; once model.classes "
                                  "is declared, map each data label that differs: "
                                  + ", ".join(data_classes))
    return block

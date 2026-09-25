"""Contract tests for the pieces a second model has to plug into.

Deliberately cheap: no checkpoint download, no HF network call. They cover the
seams that broke when the engine was generalised from one hardcoded model.
"""
from __future__ import annotations

import csv
import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

def _showcase_source() -> str:
    """Every line of the showcase package, concatenated.

    The assertions below are about the *rendering path*, which used to be one
    file. It is now a package, and reading `app.py` alone would quietly stop
    checking the code that actually draws anything — a test that passes because
    it is looking in the wrong place is worse than no test.
    """
    files = sorted((REPO / "showcase").rglob("*.py"))
    assert files, "the showcase package must not be empty"
    return "\n".join(f.read_text(encoding="utf-8") for f in files)


sys.path.insert(0, str(REPO))

from verifai.core.run import METRIC_REGISTRY, _load          # noqa: E402
from verifai.datasets import loaders                          # noqa: E402
from verifai.models.image import _resolve_module, resolve_device  # noqa: E402


# --- the registry actually resolves ------------------------------------------
@pytest.mark.parametrize("metric_id", sorted(METRIC_REGISTRY))
def test_every_registered_metric_is_importable(metric_id):
    assert callable(_load(METRIC_REGISTRY[metric_id]))


# --- layer paths, so Grad-CAM is not ResNet-only -----------------------------
def test_resolve_module_handles_index_and_dots():
    torch = pytest.importorskip("torch")
    m = torch.nn.Module()
    m.layer4 = torch.nn.Sequential(torch.nn.Conv2d(1, 1, 1), torch.nn.Conv2d(1, 2, 1))
    assert _resolve_module(m, "layer4[-1]").out_channels == 2
    assert _resolve_module(m, "layer4[0]").out_channels == 1
    with pytest.raises(ValueError):
        _resolve_module(m, "layer4[oops]")


def test_resolve_device_honours_explicit_name():
    pytest.importorskip("torch")
    assert resolve_device("cpu").type == "cpu"
    assert resolve_device("auto").type in {"cpu", "cuda", "mps"}


# --- a dataset must not learn its classes from a model -----------------------
def test_loader_does_not_import_the_model_package():
    src = (REPO / "verifai" / "datasets" / "loaders.py").read_text(encoding="utf-8")
    assert "verifai.models" not in src, "dataset loader must not depend on a model"


def _manifest(tmp_path: Path, rows: list[dict]) -> Path:
    imgs = tmp_path / "img"
    imgs.mkdir()
    from PIL import Image
    for r in rows:
        Image.new("RGB", (8, 8)).save(imgs / r["filename"])
    man = tmp_path / "m.csv"
    with open(man, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    return man


def test_classes_are_derived_from_the_data(tmp_path):
    pytest.importorskip("PIL")
    man = _manifest(tmp_path, [
        {"filename": "a.jpg", "label": "zebra", "lesion_id": "L1"},
        {"filename": "b.jpg", "label": "apple", "lesion_id": "L2"},
    ])
    ds = loaders.load_image_manifest({"manifest": str(man), "images_dir": str(man.parent / "img")})
    assert ds.classes == ["apple", "zebra"]          # sorted, from the labels
    assert ds.has_labels and len(ds) == 2


def test_extra_manifest_columns_survive_on_the_sample(tmp_path):
    pytest.importorskip("PIL")
    man = _manifest(tmp_path, [
        {"filename": "a.jpg", "label": "mel", "lesion_id": "HAM_1", "sex": "female"},
    ])
    ds = loaders.load_image_manifest({"manifest": str(man), "images_dir": str(man.parent / "img")})
    s = list(ds)[0]
    assert s.meta["lesion_id"] == "HAM_1" and s.meta["sex"] == "female"


def test_explicit_classes_win_over_derived(tmp_path):
    pytest.importorskip("PIL")
    man = _manifest(tmp_path, [{"filename": "a.jpg", "label": "mel"}])
    ds = loaders.load_image_manifest({"manifest": str(man), "images_dir": str(man.parent / "img"),
                                      "classes": ["nv", "mel"]})
    assert ds.classes == ["nv", "mel"]              # checkpoint order, not sorted


def test_old_loader_name_still_resolves():
    assert loaders.load_ham10000 is loaders.load_image_manifest


# --- the adapter works for an arbitrary class count / architecture -----------
def test_adapter_is_not_tied_to_seven_skin_classes():
    torch = pytest.importorskip("torch")
    tvm = pytest.importorskip("torchvision.models")
    from PIL import Image
    from verifai.models.image import ImageClassifier

    net = tvm.resnet18(weights=None)
    net.fc = torch.nn.Linear(net.fc.in_features, 3)
    clf = ImageClassifier(net.eval(), ["a", "b", "c"], device=torch.device("cpu"))

    probs = clf.predict_probs(Image.new("RGB", (64, 64)))
    assert set(probs) == {"a", "b", "c"}
    assert abs(sum(probs.values()) - 1.0) < 1e-5
    assert clf.cam_layer is net.layer4[-1]


# --- honesty gates: a status must not outrun the evidence --------------------
def test_single_skin_tone_bin_is_reported_as_insufficient_evidence():
    """One populated ITA bin means one group — a gap of 0 there says nothing.

    Regression: HAM10000 skews so heavily towards light skin that a full run can
    put nearly every image in one bin. The old gate accepted that and emitted a
    green 'pass' on fairness for the most skewed sample imaginable.
    """
    pytest.importorskip("numpy")
    from PIL import Image
    from verifai.metrics.fairness import skin_tone_ita as f

    class _DS:
        # 12 identical pale images -> all land in the same ITA bin
        samples = [type("S", (), {"id": f"i{i}", "label": "mel", "meta": {}})()
                   for i in range(12)]
        def __iter__(self): return iter(self.samples)
        def load(self, s): return Image.new("RGB", (64, 64), (245, 224, 210))

    class _M:
        classes = ["mel", "nv"]
        def predict_probs(self, img): return {"mel": 0.9, "nv": 0.1}
        def decide(self, probs, meta=None): return max(probs, key=probs.get)

    finding = f.run(_M(), _DS(), {})
    populated = [c for c in finding.value["coverage"].values() if c > 0]
    assert len(populated) == 1, "test setup should produce exactly one populated bin"
    assert finding.verdict == "insufficient", \
        "one populated bin cannot support a subgroup claim, and must say so"
    assert "accuracy_gap" not in finding.value, "no gap should be claimed from one group"


# --- split integrity: the check that guards every other number ---------------
def _split_manifest(tmp_path: Path, name: str, rows: list[tuple[str, str]]) -> Path:
    p = tmp_path / f"{name}.csv"
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["filename", "image_id", "lesion_id", "label"])
        for img, les in rows:
            w.writerow([f"{img}.jpg", img, les, "mel"])
    return p


def test_audit_split_passes_a_disjoint_split(tmp_path):
    from verifai.core.integrity import audit_split
    tr = _split_manifest(tmp_path, "tr", [("i1", "L1"), ("i2", "L2")])
    te = _split_manifest(tmp_path, "te", [("i3", "L3"), ("i4", "L4")])
    a = audit_split(te, [tr])
    assert a["clean"] and a["contamination"] == 0.0


def test_audit_split_catches_a_shared_lesion_even_with_new_images(tmp_path):
    """The failure the original dataset had: distinct image_ids, same lesion."""
    from verifai.core.integrity import audit_split
    tr = _split_manifest(tmp_path, "tr", [("i1", "L1")])
    te = _split_manifest(tmp_path, "te", [("i9", "L1"), ("i8", "L2")])   # i9 is a new photo of L1
    a = audit_split(te, [tr])
    assert not a["clean"]
    assert a["shared_ids"] == 0 and a["shared_groups"] == 1
    assert a["contamination"] == 0.5


def test_audit_split_catches_identical_images(tmp_path):
    from verifai.core.integrity import audit_split
    tr = _split_manifest(tmp_path, "tr", [("i1", "L1")])
    te = _split_manifest(tmp_path, "te", [("i1", "L1")])
    a = audit_split(te, [tr])
    assert a["shared_ids"] == 1 and a["contamination"] == 1.0


def test_our_committed_manifests_are_leak_free():
    """The real split this repo ships. If this ever fails, do not publish a number."""
    from verifai.core.integrity import audit_split
    man = REPO / "data" / "manifests"
    if not (man / "ham10000_test.csv").exists():
        pytest.skip("run scripts/build_splits.py first")
    a = audit_split(man / "ham10000_test.csv",
                    [man / "ham10000_train.csv", man / "ham10000_val.csv"])
    assert a["clean"], f"committed split leaks: {a}"
    assert a["n_test"] == 1493


def test_train_manifests_are_derived_from_the_training_block():
    from verifai.core.integrity import train_manifests_from_scenario
    got = train_manifests_from_scenario(
        {"training": {"manifest_prefix": "ham10000", "manifest_dir": "data/manifests"}})
    assert got == ["data/manifests/ham10000_train.csv", "data/manifests/ham10000_val.csv"], \
        "validation counts as seen — the model was selected on it"
    assert train_manifests_from_scenario({}) == []


def test_unverifiable_split_is_never_reported_as_clean(tmp_path):
    """A manifest without identifiers answers nothing — that is not a pass.

    Regression: manifests carrying only filename+label compared as 0 shared
    lesions and 0 shared images, which scored a green 'clean split'.
    """
    from verifai.core.integrity import audit_split
    def bare(name, rows):
        p = tmp_path / f"{name}.csv"
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(["filename", "label"])
            w.writerows(rows)
        return p
    a = audit_split(bare("tr", [["a.jpg", "mel"]]), [bare("te", [["a.jpg", "mel"]])])
    assert a["verifiable"] is False
    assert a["clean"] is not True, "unverifiable must never read as clean"


# --- snapshots: the evidence trail that makes comparison possible -----------
def _report_with(value, pillar="performance", verdict="pass"):
    from verifai.core.findings import Report, Finding
    r = Report(scenario="s", domain="image", model_id="m", dataset_id="d")
    r.add(Finding(pillar=pillar, metric="m", domain="image", value=value, verdict=verdict))
    return r


def test_snapshot_metrics_flattens_numeric_leaves_only():
    from verifai.export.artifacts import snapshot_metrics
    m = snapshot_metrics(_report_with(
        {"accuracy": 0.8, "n": 100, "correct": True,          # bool is not a metric
         "per_class_recall": {"melanoma": 0.64, "nevi": None},
         "note": "text"}))
    assert m["performance.accuracy"] == 0.8
    assert m["performance.n"] == 100.0
    assert m["performance.per_class_recall.melanoma"] == 0.64
    assert "performance.correct" not in m, "bool must not be recorded as a metric"
    assert "performance.note" not in m
    assert "performance.per_class_recall.nevi" not in m


def test_snapshot_records_what_makes_a_run_comparable(tmp_path):
    from verifai.export.artifacts import write_snapshot
    r = _report_with({"accuracy": 0.8})
    r.add.__self__.findings.append(
        __import__("verifai.core.findings", fromlist=["Finding"]).Finding(
            pillar="integrity", metric="split_leakage", domain="image",
            value={"contamination": 0.0}, verdict="pass"))
    r.meta = {"eval_set": {"manifest": "m.csv", "sha256": "deadbeef", "n": 10},
              "label": "baseline", "device": "cpu", "seed": 42}
    path = write_snapshot(r, tmp_path)
    snap = json.loads(path.read_text(encoding="utf-8"))
    assert snap["eval_set"]["sha256"] == "deadbeef"
    assert snap["integrity"] == "pass", "the integrity verdict must travel with the snapshot"
    assert snap["label"] == "baseline"
    assert snap["metrics"]["performance.accuracy"] == 0.8


def test_eval_set_fingerprint_uses_content_not_filename(tmp_path):
    """A manifest can be regenerated with a different seed and keep its name."""
    from verifai.core.run import _eval_set_fingerprint

    class _DS:
        def __init__(self, p): self.meta = {"manifest": str(p)}
        def __len__(self): return 2

    a = tmp_path / "same_name.csv"; a.write_text("filename,label\nx.jpg,mel\n")
    first = _eval_set_fingerprint(_DS(a))["sha256"]
    a.write_text("filename,label\ny.jpg,nv\n")            # same path, different rows
    second = _eval_set_fingerprint(_DS(a))["sha256"]
    assert first and second and first != second, "different rows must not look comparable"


# --- uncertainty: a number without an interval is not a claim ---------------
def test_wilson_interval_brackets_the_estimate_and_narrows_with_n():
    from verifai.metrics._stats import wilson
    lo, hi = wilson(104, 163)                       # melanoma in the real test set
    assert lo < 104 / 163 < hi
    wide = wilson(10, 13)                           # dermatofibroma: 13 images
    narrow = wilson(776, 1009)                      # nevi: 1,009 images
    assert (wide[1] - wide[0]) > 3 * (narrow[1] - narrow[0]), \
        "a class with 13 images must not look as certain as one with 1,009"


def test_wilson_stays_inside_zero_one_at_the_extremes():
    """Where the normal approximation fails: 0/n and n/n."""
    from verifai.metrics._stats import wilson
    lo, hi = wilson(0, 10)
    assert lo == 0.0 and 0.0 < hi < 1.0, "0/10 is not certainty"
    lo, hi = wilson(10, 10)
    assert hi == 1.0 and 0.0 < lo < 1.0, "10/10 is not certainty either"
    assert wilson(5, 0) is None


def test_ppv_moves_with_prevalence_even_though_sensitivity_does_not():
    """The clinical point: precision on a test set is not PPV at deployment."""
    from verifai.metrics._stats import ppv_at_prevalence
    high = ppv_at_prevalence(0.85, 0.90, 0.10)
    low = ppv_at_prevalence(0.85, 0.90, 0.01)
    assert high > 5 * low, "PPV must fall sharply as prevalence falls"
    assert ppv_at_prevalence(0.85, 0.90, 0.0) is None


def test_per_class_metrics_from_a_known_confusion_matrix():
    from verifai.metrics.performance.classification import _per_class
    #            pred A  B
    cm = [[8, 2],      # true A: 10
          [3, 7]]      # true B: 10
    pc = _per_class(cm, ["A", "B"])
    assert pc["A"]["sensitivity"] == 0.8 and pc["A"]["support"] == 10
    assert pc["A"]["specificity"] == 0.7          # B correctly not called A: 7/10
    assert pc["A"]["ppv_test_prevalence"] == round(8 / 11, 4)
    assert pc["A"]["sensitivity_ci"][0] < 0.8 < pc["A"]["sensitivity_ci"][1]


def test_fairness_gap_is_not_claimed_when_group_intervals_overlap():
    """A large gap between two small groups is not evidence of a gap."""
    pytest.importorskip("numpy")
    from PIL import Image
    from verifai.metrics.fairness import skin_tone_ita as f

    # two bins, 10 images each, one scoring 0.6 and one 0.9 -> wide, overlapping CIs
    light = [(245, 224, 210)] * 10
    dark = [(90, 62, 48)] * 10

    class _S:
        def __init__(self, i, rgb, ok): self.id, self.rgb, self.label = f"i{i}", rgb, ("a" if ok else "b")

    samples = [_S(i, c, i < 6) for i, c in enumerate(light)] + \
              [_S(100 + i, c, i < 9) for i, c in enumerate(dark)]

    class _DS:
        def __iter__(self): return iter(samples)
        def load(self, s): return Image.new("RGB", (64, 64), s.rgb)

    class _M:
        classes = ["a", "b"]
        def predict_probs(self, img): return {"a": 0.9, "b": 0.1}   # always predicts "a"
        def decide(self, probs, meta=None): return max(probs, key=probs.get)

    finding = f.run(_M(), _DS(), {})
    if "accuracy_gap" in finding.value:              # only if both bins were populated
        assert finding.value["gap_is_separated"] is False
        assert finding.verdict == "insufficient", \
            "overlapping intervals mean the gap is not established, and must read that way"


# --- the decision rule: argmax is a choice, not a law -----------------------
def _clf(weights=None):
    torch = pytest.importorskip("torch")
    tvm = pytest.importorskip("torchvision.models")
    from verifai.models.image import ImageClassifier
    net = tvm.resnet18(weights=None)
    net.fc = torch.nn.Linear(net.fc.in_features, 3)
    return ImageClassifier(net.eval(), ["nevus", "melanoma", "other"],
                           device=torch.device("cpu"), decision_weights=weights)


def test_default_decision_rule_is_plain_argmax():
    clf = _clf()
    assert clf.decide({"nevus": 0.5, "melanoma": 0.3, "other": 0.2}) == "nevus"


def test_weights_let_a_rare_class_clear_a_lower_bar():
    """The whole point: melanoma wins on 0.3 vs 0.5 once missing it costs more."""
    probs = {"nevus": 0.5, "melanoma": 0.3, "other": 0.2}
    assert _clf({"melanoma": 2.0}).decide(probs) == "melanoma"   # 0.6 > 0.5
    assert _clf({"melanoma": 1.5}).decide(probs) == "nevus"      # 0.45 < 0.5, not enough
    assert _clf().decide(probs) == "nevus"                       # unweighted


def test_ranking_follows_the_same_rule_so_top_k_stays_consistent():
    probs = {"nevus": 0.5, "melanoma": 0.3, "other": 0.2}
    assert _clf().rank(probs)[0] == "nevus"
    assert _clf({"melanoma": 2.0}).rank(probs)[0] == "melanoma"


def test_unlisted_classes_keep_weight_one():
    probs = {"nevus": 0.5, "melanoma": 0.3, "other": 0.2}
    assert _clf({"other": 10.0}).decide(probs) == "other"        # 2.0 beats 0.5


# --- comparison: what is decidable from data, and what is not --------------
def test_metrics_declare_their_own_direction_rather_than_the_app_guessing():
    """'auc' is higher-better for a classifier and lower-better for membership
    inference. Only the metric can say which."""
    from verifai.export.artifacts import snapshot_directions
    from verifai.core.findings import Report, Finding
    r = Report(scenario="s", domain="image", model_id="m", dataset_id="d")
    r.add(Finding(pillar="privacy", metric="m", domain="image", value={"mia_auc": 0.6},
                  details={"better": {"mia_auc": "lower"}}))
    r.add(Finding(pillar="performance", metric="m", domain="image", value={"accuracy": 0.8},
                  details={"better": {"accuracy": "higher", "per_class.*.sensitivity": "higher"}}))
    d = snapshot_directions(r)
    assert d["privacy.mia_auc"] == "lower"
    assert d["performance.accuracy"] == "higher"
    assert d["performance.per_class.*.sensitivity"] == "higher"


# --- training variants: focal loss and balanced sampling -------------------
def test_focal_loss_reduces_to_cross_entropy_at_gamma_zero():
    torch = pytest.importorskip("torch")
    sys.path.insert(0, str(REPO / "scripts"))
    from train_model import FocalLoss
    logits = torch.tensor([[3.0, 0.0, 0.0], [0.4, 0.3, 0.3]])
    target = torch.tensor([0, 0])
    ce = torch.nn.CrossEntropyLoss()(logits, target)
    assert torch.allclose(FocalLoss(gamma=0.0)(logits, target), ce)


def test_focal_loss_shifts_weight_from_easy_examples_to_hard_ones():
    """The whole point: a confidently-correct nevus should stop contributing."""
    torch = pytest.importorskip("torch")
    sys.path.insert(0, str(REPO / "scripts"))
    from train_model import FocalLoss
    easy = (torch.tensor([[6.0, 0.0, 0.0]]), torch.tensor([0]))    # already right
    hard = (torch.tensor([[0.4, 0.3, 0.3]]), torch.tensor([0]))    # barely right
    ce_ratio = (torch.nn.CrossEntropyLoss()(*hard) / torch.nn.CrossEntropyLoss()(*easy))
    fl = FocalLoss(gamma=2.0)
    focal_ratio = fl(*hard) / fl(*easy)
    assert focal_ratio > 10 * ce_ratio, "focal must concentrate far harder on hard cases"


def test_focal_loss_accepts_class_weights_so_it_composes_with_alpha():
    torch = pytest.importorskip("torch")
    sys.path.insert(0, str(REPO / "scripts"))
    from train_model import FocalLoss
    logits = torch.tensor([[0.4, 0.3, 0.3]])
    target = torch.tensor([0])
    plain = FocalLoss(gamma=2.0)(logits, target)
    weighted = FocalLoss(gamma=2.0, weight=torch.tensor([5.0, 1.0, 1.0]))(logits, target)
    assert weighted > plain, "a class weight of 5 must scale that class's loss up"


def test_lineage_never_widens_what_may_be_compared():
    """Presentation grouping must not leak into the comparability rule.

    Two configurations of one lineage that were scored on different manifests
    still have to land in different comparability groups — the content hash
    decides, not the label they share in the gallery.
    """
    pytest.importorskip("streamlit")
    sys.path.insert(0, str(REPO / "showcase"))
    import app
    same_lineage_different_rows = [
        {"scenario": "a", "label": "a", "integrity": "pass", "created_at": "1",
         "eval_set": {"sha256": "aaa", "manifest": "test.csv", "n": 100}, "metrics": {}},
        {"scenario": "b", "label": "b", "integrity": "pass", "created_at": "2",
         "eval_set": {"sha256": "bbb", "manifest": "test.csv", "n": 100}, "metrics": {}},
    ]
    assert len(app.group_snapshots(same_lineage_different_rows)) == 2


def test_the_same_rows_at_a_different_path_stay_comparable():
    """A moved or renamed checkout must not split a comparison group.

    The hash is taken over the manifest's contents alone, so the same rows read
    from `/old/repo/test.csv` and `/new/repo/test.csv` are the same evaluation
    set. Keying on the path too would have quietly filed every run made before a
    directory rename apart from every run made after it — same rows, two groups,
    each looking like it held fewer runs than it did.
    """
    pytest.importorskip("streamlit")
    sys.path.insert(0, str(REPO / "showcase"))
    import app
    same_rows_moved_repo = [
        {"scenario": "a", "label": "a", "integrity": "pass", "created_at": "1",
         "eval_set": {"sha256": "aaa", "manifest": "/old/repo/test.csv", "n": 100},
         "metrics": {}},
        {"scenario": "b", "label": "b", "integrity": "pass", "created_at": "2",
         "eval_set": {"sha256": "aaa", "manifest": "/new/repo/test.csv", "n": 100},
         "metrics": {}},
    ]
    assert len(app.group_snapshots(same_rows_moved_repo)) == 1


def test_unhashed_snapshots_do_not_merge_on_being_equally_unidentified():
    """No hash means nothing to compare on — those must not pool together."""
    pytest.importorskip("streamlit")
    sys.path.insert(0, str(REPO / "showcase"))
    import app
    no_hash = [
        {"scenario": "a", "label": "a", "integrity": "pass", "created_at": "1",
         "eval_set": {"sha256": None, "manifest": "a.csv", "n": 7}, "metrics": {}},
        {"scenario": "b", "label": "b", "integrity": "pass", "created_at": "2",
         "eval_set": {"sha256": None, "manifest": "b.csv", "n": 9}, "metrics": {}},
    ]
    assert len(app.group_snapshots(no_hash)) == 2


# --- the ISIC exclusion, which is what keeps the new model comparable --------
def _isic_builder():
    sys.path.insert(0, str(REPO / "scripts"))
    import build_isic_train
    return build_isic_train


def _write_gt(path: Path, rows: list[tuple[str, str]]) -> None:
    """rows = [(image_id, ISIC code)] -> one-hot ground truth, as published."""
    codes = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC", "UNK"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image"] + codes)
        for image_id, code in rows:
            w.writerow([image_id] + ["1.0" if c == code else "0.0" for c in codes])


def _write_md(path: Path, rows: list[tuple[str, str]]) -> None:
    """rows = [(image_id, lesion_id)]; an empty lesion_id is left empty."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image", "age_approx",
                                          "anatom_site_general", "lesion_id", "sex"])
        w.writeheader()
        for image_id, lesion_id in rows:
            w.writerow({"image": image_id, "age_approx": "45",
                        "anatom_site_general": "trunk",
                        "lesion_id": lesion_id, "sex": "female"})


def _write_manifest(path: Path, rows: list[tuple[str, str, str]]) -> None:
    """rows = [(image_id, lesion_id, label)] in the repo's manifest schema."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["filename", "image_id", "lesion_id", "label",
                                          "dx_type", "age", "sex", "localization"])
        w.writeheader()
        for image_id, lesion_id, label in rows:
            w.writerow({"filename": f"{image_id}.jpg", "image_id": image_id,
                        "lesion_id": lesion_id, "label": label,
                        "dx_type": "", "age": "", "sex": "", "localization": ""})


def test_held_back_images_never_reach_the_isic_training_manifest():
    """ISIC 2019 contains HAM10000, so this anti-join is the whole safeguard."""
    m = _isic_builder()
    labels = {"ISIC_0000001": "melanoma", "ISIC_0000002": "melanocytic_Nevi"}
    meta = {"ISIC_0000001": {"lesion_id": "HAM_1"}, "ISIC_0000002": {"lesion_id": "HAM_2"}}
    rows, why = m.build_rows(labels, meta, {"ISIC_0000001"}, set())
    assert [r["image_id"] for r in rows] == ["ISIC_0000002"]
    assert why["excluded by image_id"] == 1


def test_a_second_photo_of_a_held_back_lesion_is_excluded_too():
    """The trap image_id matching alone would miss.

    A held-back lesion can appear in ISIC under a completely different image id.
    Keeping it would mean testing on a lesion the model had memorised, which is
    the same mistake `audit_split` grouping by lesion_id exists to prevent.
    """
    m = _isic_builder()
    labels = {"ISIC_9999001": "melanoma"}
    meta = {"ISIC_9999001": {"lesion_id": "HAM_0001981"}}
    rows, why = m.build_rows(labels, meta, set(), {"HAM_0001981"})
    assert rows == []
    assert why["excluded by lesion_id"] == 1


def test_a_missing_lesion_id_is_absence_of_evidence_not_a_match():
    """An empty lesion_id must not collide with a held-back one and drop the row."""
    m = _isic_builder()
    labels = {"ISIC_9999002": "basal_cell_carcinoma"}
    rows, _ = m.build_rows(labels, {"ISIC_9999002": {"lesion_id": ""}}, set(), {""})
    assert len(rows) == 1
    # It still needs a grouping key, and falls back to its own image id.
    assert rows[0]["lesion_id"] == "ISIC_9999002"


def test_scc_is_dropped_by_default_and_unk_is_never_a_label(tmp_path):
    """The frozen test set has 7 classes; an 8th head could never be scored on it."""
    m = _isic_builder()
    gt = tmp_path / "gt.csv"
    _write_gt(gt, [("ISIC_1", "MEL"), ("ISIC_2", "SCC"), ("ISIC_3", "UNK")])

    labels, dropped = m.read_groundtruth(gt, keep_scc=False)
    assert labels == {"ISIC_1": "melanoma"}
    assert sum(dropped.values()) == 2

    kept, _ = m.read_groundtruth(gt, keep_scc=True)
    assert kept["ISIC_2"] == "squamous_cell_carcinoma"
    assert "ISIC_3" not in kept, "UNK is 'none of the above', never a diagnosis"


def test_the_isic_manifest_matches_the_schema_the_loader_already_reads(tmp_path):
    """A new manifest must need no loader change: same columns, same meaning."""
    m = _isic_builder()
    with open(REPO / "data" / "manifests" / "ham10000_test.csv", encoding="utf-8") as f:
        existing = csv.DictReader(f).fieldnames
    assert m.MANIFEST_FIELDS == list(existing)


def test_the_builder_refuses_to_leave_a_leaking_manifest_on_disk(tmp_path):
    """verify() reads the written file back, so a bug upstream still gets caught."""
    m = _isic_builder()
    train = tmp_path / "isic_train.csv"
    held = tmp_path / "ham10000_test.csv"
    _write_manifest(train, [("ISIC_0000001", "HAM_1", "melanoma")])
    _write_manifest(held, [("ISIC_0000001", "HAM_1", "melanoma")])
    assert m.verify(train, [held]) is False

    _write_manifest(train, [("ISIC_0000009", "HAM_9", "melanoma")])
    assert m.verify(train, [held]) is True


def test_building_isic_manifests_leaves_the_frozen_test_set_untouched(tmp_path):
    """Comparability is the content hash of the eval manifest — it must not move.

    If building the new training set altered ham10000_test.csv by even a byte,
    the new run would land in a different comparability group and could not be
    ranked against the five existing configurations.
    """
    import hashlib
    m = _isic_builder()
    test_manifest = REPO / "data" / "manifests" / "ham10000_test.csv"
    before = hashlib.sha256(test_manifest.read_bytes()).hexdigest()

    gt, md = tmp_path / "gt.csv", tmp_path / "md.csv"
    _write_gt(gt, [("ISIC_8880001", "MEL"), ("ISIC_8880002", "NV")])
    _write_md(md, [("ISIC_8880001", "LES_A"), ("ISIC_8880002", "LES_B")])
    labels, _ = m.read_groundtruth(gt, keep_scc=False)
    meta = m.read_metadata(md)
    excl_i, excl_l = m.read_exclusions([test_manifest])
    rows, _ = m.build_rows(labels, meta, excl_i, excl_l)
    m.write_manifest(rows, tmp_path / "isic_train.csv")

    assert hashlib.sha256(test_manifest.read_bytes()).hexdigest() == before


def test_training_images_dir_does_not_change_the_images_we_score_on():
    """A scenario may train on one corpus and be scored on a frozen other one.

    `skin_cancer_isic` trains on ISIC 2019 and is evaluated on the same HAM10000
    files as every earlier run. If training and evaluation shared one
    `images_dir`, growing the training set would swap the evaluation images for
    re-encoded copies — identical manifest, identical content hash, different
    pixels — and the comparison would look legitimate while being meaningless.
    Neither the content hash nor `audit_split` can catch that: both read
    manifests, not images.
    """
    sys.path.insert(0, str(REPO / "scripts"))
    from train_model import split_spec
    base = {"images_dir": "data/raw/ham10000", "manifest": "eval.csv"}

    train = split_spec(base, {"images_dir": "data/raw/isic"}, "data/manifests", "isic", "train")
    assert train["images_dir"] == "data/raw/isic"
    assert train["manifest"] == "data/manifests/isic_train.csv"
    # The evaluation spec the metrics read is untouched by the override.
    assert base["images_dir"] == "data/raw/ham10000"
    assert base["manifest"] == "eval.csv"

    # Absent the override, training keeps reading the dataset's own directory.
    plain = split_spec(base, {}, "data/manifests", "ham10000", "train")
    assert plain["images_dir"] == "data/raw/ham10000"


def test_the_isic_scenario_scores_on_the_same_images_as_the_runs_it_compares_to():
    """Guards the specific wiring, not just the mechanism."""
    yaml = pytest.importorskip("yaml")
    isic = yaml.safe_load((REPO / "scenarios" / "skin_cancer_isic.yaml").read_text())
    clean = yaml.safe_load((REPO / "scenarios" / "skin_cancer_clean.yaml").read_text())

    # Same evaluation manifest AND same pixels as the configuration it ranks against.
    assert isic["dataset"]["manifest"] == clean["dataset"]["manifest"]
    assert isic["dataset"]["images_dir"] == clean["dataset"]["images_dir"]
    # Training reads somewhere else, or the point above is lost.
    assert isic["training"]["images_dir"] != isic["dataset"]["images_dir"]
    # Membership inference must compare like with like: both sides from one corpus.
    assert isic["privacy"]["members"] == clean["privacy"]["members"]
    assert isic["privacy"]["non_members"] == clean["privacy"]["non_members"]


# --- materializing a second image corpus ------------------------------------
def _isic_mat():
    sys.path.insert(0, str(REPO / "scripts"))
    import materialize_isic
    return materialize_isic


def test_the_zip_is_indexed_by_basename_not_by_nested_path():
    """The challenge zip nests under ISIC_2019_Training_Input/; a repack may not."""
    import io
    import zipfile
    m = _isic_mat()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("ISIC_2019_Training_Input/ISIC_0000001.jpg", b"x")
        zf.writestr("ISIC_0000002.jpg", b"y")
        zf.writestr("ISIC_2019_Training_Input/", b"")       # directory entry
    with zipfile.ZipFile(buf) as zf:
        index = m.zip_index(zf)
    assert index["ISIC_0000001"] == "ISIC_2019_Training_Input/ISIC_0000001.jpg"
    assert index["ISIC_0000002"] == "ISIC_0000002.jpg"
    assert "" not in index, "a directory entry must not become an image key"


def test_materializing_refuses_to_mix_two_encodings_in_one_directory():
    """One directory, one encoding.

    A blur radius or a JPEG quality means something different at a different
    resolution, so a directory holding two encodings is unusable for a
    comparison — and nothing in the files themselves would reveal it.
    """
    import json
    m = _isic_mat()
    out = Path(tempfile.mkdtemp())
    (out / "_materialize.json").write_text(
        json.dumps({"max_size": 320, "quality": 90, "source": "x"}), encoding="utf-8")

    m.check_stamp(out, 320, 90)                    # same encoding: fine
    with pytest.raises(SystemExit):
        m.check_stamp(out, 224, 90)                # different size
    with pytest.raises(SystemExit):
        m.check_stamp(out, 320, 75)                # different quality


def test_the_shrink_matches_what_data_raw_ham10000_already_holds():
    """Both materializers must apply the same transform, or the corpora disagree."""
    pytest.importorskip("PIL")
    import io
    from PIL import Image
    m = _isic_mat()
    buf = io.BytesIO()
    Image.new("RGB", (600, 450), (10, 20, 30)).save(buf, "JPEG", quality=95)

    out = Image.open(io.BytesIO(m._shrink(buf.getvalue(), 320, 90)))
    assert max(out.size) == 320, "longest side is capped"
    assert out.size == (320, 240), "aspect ratio is preserved"
    # data/raw/ham10000 was written at exactly this setting; if that ever changes
    # the two directories stop being comparable.
    stamp = REPO / "data" / "raw" / "ham10000" / "_materialize.json"
    if stamp.exists():
        have = json.loads(stamp.read_text(encoding="utf-8"))
        assert (have["max_size"], have["quality"]) == (320, 90)


def test_the_manifests_are_the_source_of_truth_for_which_images_are_needed():
    """No manifest, no guessing: it exits rather than materialize an unknown set."""
    m = _isic_mat()
    empty = Path(tempfile.mkdtemp())
    with pytest.raises(SystemExit):
        m.wanted(empty, "isic", ["train"])

    want = m.wanted(REPO / "data" / "manifests", "isic", ["train", "val"])
    # Training reads both splits, so both have to be materialized.
    assert len(want) > 20000
    assert all(f.endswith(".jpg") for f in want.values())


def test_pretrained_weight_downloads_get_a_ca_bundle():
    """python.org macOS builds have no root certs, so torch.hub cannot fetch.

    ResNet18 only worked because its weights were already cached; the first
    uncached architecture died with CERTIFICATE_VERIFY_FAILED. build_splits.py
    already solves this for its own downloads via certifi, but torch.hub reads
    the environment, so train_model.py has to set it.
    """
    import os
    sys.path.insert(0, str(REPO / "scripts"))
    import importlib.util
    spec = importlib.util.spec_from_file_location("_tm", REPO / "scripts" / "train_model.py")
    tm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tm)

    pytest.importorskip("certifi")
    assert os.environ.get("SSL_CERT_FILE"), "importing train_model must set a CA bundle"
    assert Path(os.environ["SSL_CERT_FILE"]).exists()

    # An explicit choice must win over the fallback.
    prior = os.environ["SSL_CERT_FILE"]
    try:
        os.environ["SSL_CERT_FILE"] = "/tmp/explicit-choice.pem"
        tm._trust_certifi()
        assert os.environ["SSL_CERT_FILE"] == "/tmp/explicit-choice.pem"
    finally:
        os.environ["SSL_CERT_FILE"] = prior


# --- plain-language metric explanations --------------------------------------
def test_every_metric_in_every_artifact_has_an_explanation():
    """The contract: a metric that reaches the comparison view explains itself.

    Snapshot keys are produced generically by `snapshot_metrics`, so a new metric
    appears in the comparison table whether or not anyone wrote wording for it.
    This is what catches that omission.
    """
    from verifai.core.glossary import explain_metric
    keys = set()
    for f in (REPO / "showcase" / "artifacts").glob("*/history/*.json"):
        keys |= set((json.loads(f.read_text(encoding="utf-8")).get("metrics") or {}))
    assert keys, "no snapshots found — this test would pass vacuously"
    missing = sorted(k for k in keys if explain_metric(k) is None)
    assert not missing, f"metric keys with no glossary entry: {missing}"


def test_every_metric_in_every_artifact_has_a_human_name():
    """A report heading names its metric in words, never by its identifier."""
    from verifai.core.glossary import METRIC_NAMES
    ids = set()
    for f in (REPO / "showcase" / "artifacts").glob("*/report.json"):
        ids |= {x["metric"] for x in json.loads(f.read_text(encoding="utf-8"))["findings"]}
    assert ids, "no reports found — this test would pass vacuously"
    missing = sorted(ids - set(METRIC_NAMES))
    assert not missing, f"metric ids with no human name in METRIC_NAMES: {missing}"


def test_an_unknown_metric_gets_no_explanation_rather_than_a_guess():
    """A confident explanation of the wrong quantity is worse than none."""
    from verifai.core.glossary import explain_metric
    assert explain_metric("performance.some_future_metric") is None
    assert explain_metric("") is None


def test_specific_glossary_patterns_are_matched_before_general_ones():
    """Ordering is load-bearing, exactly as it is for metric directions.

    `fnmatch` has no notion of specificity, so a broad pattern placed early would
    silently swallow the precise entries that follow it.
    """
    from verifai.core.glossary import explain_metric
    sens = explain_metric("performance.per_class.melanoma.sensitivity")
    ppv = explain_metric("performance.per_class.melanoma.ppv_test_prevalence")
    assert sens is not None and ppv is not None
    assert sens != ppv, "sensitivity and PPV must not resolve to the same entry"
    assert "recall" in sens["measures"].lower()
    assert "precision" in ppv["measures"].lower()


def test_every_glossary_entry_teaches_the_same_four_things():
    """Uniform shape is the point: the reader learns where to look once."""
    from verifai.core.glossary import GLOSSARY
    for pattern, entry in GLOSSARY:
        for field in ("measures", "ideal", "reading"):
            assert entry.get(field), f"{pattern} is missing '{field}'"
            assert len(entry[field]) > 40, f"{pattern}.{field} is too terse to help"
        # `term` is what the reader sees as the card's heading. Without it the
        # card falls back to a raw key like
        # `performance.per_class.melanoma.ppv_test_prevalence`, which is a
        # developer string, not a name anyone knows the concept by.
        assert entry.get("term"), f"{pattern} is missing a human-readable 'term'"
        assert "." not in entry["term"], f"{pattern}.term looks like a key, not a name"
        assert set(entry) <= {"term", "measures", "ideal", "reading", "tension"}, \
            f"{pattern} has unexpected fields"


def test_the_metric_whose_direction_is_counterintuitive_says_so():
    """`mia_auc` is lower-is-better while an AUC normally is not.

    The comparison table already refuses to infer direction from the name; the
    explanation has to carry the same warning in words, or a reader sees 0.54
    next to 'AUC' and reads it as a poor score rather than a good one.
    """
    from verifai.core.glossary import explain_metric
    entry = explain_metric("privacy.mia_auc")
    assert "0.5" in entry["ideal"]
    assert "lower is better" in entry["ideal"].lower()


def test_the_glossary_stays_light_enough_for_the_showcase():
    """showcase/requirements.txt has no torch — importing it must not need one."""
    import subprocess
    code = ("import sys; before=set(sys.modules);"
            "import verifai.core.glossary;"
            "heavy=[m for m in set(sys.modules)-before "
            "if m.split('.')[0] in ('torch','torchvision','numpy','PIL','pandas')];"
            "print(','.join(heavy))")
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO,
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert not out.stdout.strip(), f"glossary pulled in heavy modules: {out.stdout}"


def test_explanation_cards_survive_streamlits_html_sanitiser():
    """The cards must not be built from raw HTML or CSS.

    `st.markdown` is sanitised with `FORBID_TAGS: ['style']`, so an injected
    <style> block is removed outright and anything styled through CSS classes
    renders as unstyled text — twice this feature shipped invisible for exactly
    that reason. Native markdown colours and `st.container(border=True)` cannot
    be sanitised away, so the rendering path is asserted to contain no markup.
    """
    pytest.importorskip("streamlit")
    sys.path.insert(0, str(REPO / "showcase"))
    import app
    from verifai.core.glossary import explain_metric

    md = app.explanation_markdown(explain_metric("privacy.mia_auc"),
                                  ["privacy.mia_auc"], "green")
    assert ":green[" in md, "colour must come from Streamlit's own markdown"
    assert "<" not in md and "style=" not in md, "no raw HTML in the card"
    assert "Membership-inference AUC" in md and "Measures" in md

    source = _showcase_source()
    assert "unsafe_allow_html" not in source, (
        "the explanation path must not depend on HTML Streamlit may strip")


def test_the_app_degrades_instead_of_crashing_without_the_engine():
    """The showcase's contract is that it renders artifacts. Explanations are a
    bonus, so losing them must not take the numbers down with them."""
    pytest.importorskip("streamlit")
    sys.path.insert(0, str(REPO / "showcase"))
    import app
    assert callable(app.explain_metric) and callable(app.pillar_of)
    assert app.render_metric_explanations(["performance.not_a_real_metric"]) == 0


def test_one_concept_is_explained_once_however_many_columns_use_it():
    """Sensitivity for three classes is one idea, not three paragraphs.

    Without this the cards become the wall of repeated text that made the first
    version of this feature unusable.
    """
    from verifai.core.glossary import entries_for
    got = entries_for(["performance.per_class.melanoma.sensitivity",
                       "performance.per_class.melanocytic_Nevi.sensitivity",
                       "performance.accuracy"])
    assert len(got) == 2, "the two sensitivities must collapse into one card"
    entry, covered = got[0]
    assert entry["term"] == "Sensitivity (recall)"
    assert len(covered) == 2, "the card must name every column it covers"
    # Unknown keys are dropped, never padded out with a placeholder.
    assert entries_for(["performance.not_a_real_metric"]) == []


def test_the_two_views_explain_a_metric_with_the_same_words():
    """`metric_keys` mirrors the export layer's flattening, so it can drift.

    The report view derives its keys with glossary.metric_keys while the
    comparison view reads keys produced by artifacts._flatten. If those two ever
    disagree, one view silently explains a metric the other cannot, which is
    precisely the kind of divergence this project keeps single-sourcing to avoid.
    """
    from verifai.core.glossary import metric_keys
    from verifai.export.artifacts import _flatten

    report = json.loads(
        (REPO / "showcase" / "artifacts" / "skin_cancer_isic" / "report.json")
        .read_text(encoding="utf-8"))
    for finding in report["findings"]:
        expected: dict[str, float] = {}
        _flatten(finding["value"], finding["pillar"], expected)
        assert sorted(metric_keys(finding["value"], finding["pillar"])) == sorted(expected), \
            f"flattening disagrees for {finding['metric']}"


# --- the status vocabulary is epistemic, never evaluative --------------------
def test_no_metric_can_score_a_model_against_an_invented_threshold():
    """The vocabulary has no pass and no fail, and that is the point.

    Measured on the real runs, the old accuracy threshold ranked the
    configurations almost exactly against the clinical goal: it marked the one
    catching 159 of 163 melanomas a WARNING and the one missing 82 of them a
    PASS, because under-calling a rare class raises overall accuracy. A reader
    trusting the badges would have picked the worst detector in the set.
    """
    from verifai.core.findings import Verdict
    import typing
    allowed = set(typing.get_args(Verdict))
    assert allowed == {"measured", "insufficient", "unavailable", "invalid"}
    assert not allowed & {"pass", "warn", "fail"}, "no evaluative word may return"

    import re
    for path in (REPO / "verifai" / "metrics").rglob("*.py"):
        src = path.read_text(encoding="utf-8")
        for word in ('"pass"', '"warn"', '"fail"'):
            assert not re.search(rf"verdict\s*=\s*{re.escape(word)}", src), \
                f"{path.name} still assigns an evaluative verdict"


def test_any_overlap_at_all_invalidates_the_split():
    """Not a percentage bar — the old 1% cut was as arbitrary as the rest.

    A split sharing one lesion is already not measuring generalisation; how far
    it is from measuring it is exactly what nobody can quantify.
    """
    pytest.importorskip("numpy")
    from verifai.metrics.integrity import split_leakage as sl
    import inspect
    src = inspect.getsource(sl)
    assert "pct >= 1" not in src, "the arbitrary 1% threshold must be gone"
    assert '"invalid"' in src, "any contamination must mark the measurement unusable"


def test_the_app_maps_retired_words_so_old_artifacts_still_read():
    """Artifacts written before the change carry pass/warn/fail.

    They must render in today's language — and integrity is mapped apart, because
    its old `fail` recorded a contaminated split, which is a fact worth keeping,
    while elsewhere the same word was only a threshold nobody could justify.
    """
    pytest.importorskip("streamlit")
    sys.path.insert(0, str(REPO / "showcase"))
    import app

    assert app.normalise_verdict("pass", "performance") == "measured"
    assert app.normalise_verdict("fail", "performance") == "measured", \
        "a retired quality threshold must not keep condemning a model"
    assert app.normalise_verdict("fail", "integrity") == "invalid", \
        "a contaminated split is a fact, not a threshold — it must survive the mapping"
    assert app.normalise_verdict("info", "integrity") == "unavailable"
    # Grad-CAM emitted `info` unconditionally, so for explainability it never
    # meant "not enough evidence" — reading it that way reported missing data
    # next to overlays that had actually been rendered.
    assert app.normalise_verdict("info", "explainability") == "measured"
    assert app.normalise_verdict("info", "performance") == "insufficient"
    # Today's words pass through untouched.
    for v in ("measured", "insufficient", "unavailable", "invalid"):
        assert app.normalise_verdict(v, "performance") == v
    # Every status the app can produce must be renderable.
    for v in app.VERDICT:
        assert len(app.VERDICT[v]) == 3 and app.VERDICT[v][1]


def test_every_status_icon_is_an_emoji_streamlit_will_accept():
    """`st.info(icon=...)` validates its icon and raises on anything else.

    A geometric symbol like ◐ or ∅ looks right in a source file and takes the
    whole finding down when rendered, which is how the first version of this
    vocabulary shipped broken.
    """
    pytest.importorskip("streamlit")
    sys.path.insert(0, str(REPO / "showcase"))
    import app
    from streamlit.string_util import validate_icon_or_emoji as validate
    for status, (icon, _, _) in app.VERDICT.items():
        validate(icon)          # raises StreamlitAPIException if unusable

    # The guard has to be able to fail, or it guards nothing.
    from streamlit.errors import StreamlitAPIException
    with pytest.raises(StreamlitAPIException):
        validate("\u25d0")     # ◐ — a geometric symbol, not an emoji


def test_a_lineage_filter_must_disclose_comparable_runs_it_hides():
    """`best` ranks what is on screen, so hiding a comparable run makes it false.

    The clean-split lineage shows five configurations and names High-sensitivity
    (w=50) best on melanoma sensitivity at 0.945. Three further runs were scored
    on the identical images — same manifest, same content hash — and one reaches
    0.976. Filtered, the table asserted a superlative that the full evidence
    contradicts, with nothing on screen saying runs were missing.
    """
    import json as _json
    pytest.importorskip("streamlit")
    sys.path.insert(0, str(REPO / "showcase"))
    import app

    snaps = [_json.loads(f.read_text(encoding="utf-8"))
             for f in (REPO / "showcase" / "artifacts").glob("*/history/*.json")]
    cards = []
    for d in (REPO / "showcase" / "artifacts").iterdir():
        card = d / "card.json"
        if card.is_file():
            cards.append({**_json.loads(card.read_text(encoding="utf-8")), "id": d.name})

    lineage = "ResNet18 · clean split"
    ids = {c["id"] for c in cards if (c.get("lineage") or c["id"]) == lineage}
    assert ids, "fixture check: that lineage must exist in the artifacts"

    kept = [s for s in snaps if s.get("scenario") in ids]
    kept_keys = {app.comparability_key(s) for s in kept}
    hidden = [s for s in snaps
              if s.get("scenario") not in ids and app.comparability_key(s) in kept_keys]
    assert hidden, (
        "fixture check: some run outside this lineage shares its evaluation set — "
        "that overlap is the whole reason the disclosure is needed")

    key = "performance.per_class.melanoma.sensitivity"
    best_shown = max((s["metrics"].get(key) or 0) for s in kept)
    best_hidden = max((s["metrics"].get(key) or 0) for s in hidden)
    assert best_hidden > best_shown, (
        "fixture check: a hidden run must actually beat the shown ones, or this "
        "test would pass even with the disclosure removed")

    source = _showcase_source()
    assert "hidden_comparable" in source, \
        "the comparison view must track runs the lineage filter hides"
    assert "scored on these same images" in source, \
        "and must say so on screen, next to the leading cells it undermines"


# --- linear probing and the learning curve -----------------------------------
def test_freezing_the_backbone_leaves_only_the_head_trainable():
    """A probe that silently trains everything is a fine-tune with extra steps."""
    torch = pytest.importorskip("torch")
    import torchvision.models as tvm

    net = tvm.resnet18(weights=None)
    net.fc = torch.nn.Linear(net.fc.in_features, 7)
    for p in net.parameters():
        p.requires_grad = False
    for p in net.fc.parameters():
        p.requires_grad = True

    trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    total = sum(p.numel() for p in net.parameters())
    assert trainable == sum(p.numel() for p in net.fc.parameters())
    assert trainable / total < 0.001, "a probe must train a vanishing share of the model"


def test_learning_curve_subsets_nest_and_keep_their_class_shape():
    """Each point must differ from the one below it only by *added* data.

    Drawing every size independently would confound "more data" with "different
    data", and a dip in the curve could mean either. Stratification matters just
    as much: an unstratified 100-image subset of this corpus is mostly nevi, and
    the curve would then measure class balance rather than size.
    """
    import csv as _csv
    man = REPO / "data" / "manifests"
    sizes = [100, 500, 2000, 7014]
    paths = [man / f"isic_n{n}_train.csv" for n in sizes]
    if not all(p.exists() for p in paths):
        pytest.skip("subsets not built yet (scripts/build_subsets.py)")

    lesions, mel_share = [], []
    for p in paths:
        rows = list(_csv.DictReader(p.open(encoding="utf-8")))
        lesions.append({r["lesion_id"] for r in rows})
        mel_share.append(sum(r["label"] == "melanoma" for r in rows) / len(rows))

    for small, large, n_s, n_l in zip(lesions, lesions[1:], sizes, sizes[1:]):
        assert small <= large, f"n={n_s} must nest inside n={n_l}"

    full = list(_csv.DictReader((man / "isic_train.csv").open(encoding="utf-8")))
    full_share = sum(r["label"] == "melanoma" for r in full) / len(full)
    for n, share in zip(sizes, mel_share):
        assert abs(share - full_share) < 0.06, \
            f"n={n} melanoma share {share:.3f} drifts from the corpus {full_share:.3f}"


def test_every_curve_point_is_scored_against_the_same_validation_set():
    """Validation must not move with the training set.

    `train_model.py` reads `<prefix>_val.csv`, so each subset prefix needs its
    own copy — and it has to be a byte copy. Resampling it per size would score
    each point against a different bar, and the curve would measure two things
    at once while looking like it measured one.
    """
    import hashlib
    man = REPO / "data" / "manifests"
    base = man / "isic_val.csv"
    if not base.exists():
        pytest.skip("isic manifests not built yet")
    want = hashlib.sha256(base.read_bytes()).hexdigest()
    for n in (100, 500, 2000, 7014):
        p = man / f"isic_n{n}_val.csv"
        if not p.exists():
            pytest.skip("subsets not built yet (scripts/build_subsets.py)")
        assert hashlib.sha256(p.read_bytes()).hexdigest() == want, \
            f"isic_n{n}_val.csv differs from isic_val.csv"


def test_curve_scenarios_differ_only_in_size_and_regime():
    """Generated, so that "everything else is held fixed" is code, not a comment."""
    yaml = pytest.importorskip("yaml")
    scen = REPO / "scenarios"
    points = sorted(scen.glob("curve_*.yaml"))
    if not points:
        pytest.skip("curve scenarios not generated yet")

    def flat(d, prefix=""):
        out = {}
        for k, v in d.items():
            out.update(flat(v, f"{prefix}{k}.")) if isinstance(v, dict) \
                else out.update({f"{prefix}{k}": v})
        return out

    loaded = {p.name: flat(yaml.safe_load(p.read_text(encoding="utf-8"))) for p in points}
    allowed = {"training.manifest_prefix", "training.freeze_backbone", "training.lr",
               "name", "label", "model.id", "model.weights_path"}
    names = sorted(loaded)
    ref = loaded[names[0]]
    for other in names[1:]:
        differing = {k for k in set(ref) | set(loaded[other])
                     if ref.get(k) != loaded[other].get(k)}
        stray = {k for k in differing if not (k in allowed or k.startswith("card."))}
        assert not stray, f"{names[0]} vs {other} differ in {stray}, which voids the curve"

    # Every point is scored on the untouched evaluation set.
    for name, sc in loaded.items():
        assert sc["dataset.manifest"] == "data/manifests/ham10000_test.csv", name


# --- context priors: information that is not in the pixels -------------------
def test_missing_context_is_neutral_rather_than_guessed():
    """A blank field must not move a decision.

    Ten percent of training rows carry no body site. Inventing one for them —
    or letting a missed lookup fall through to something other than 1.0 — would
    manufacture evidence out of an empty cell.
    """
    import json as _json
    from verifai.models.image import ImageClassifier

    prior_path = REPO / "data" / "priors" / "isic.json"
    if not prior_path.exists():
        pytest.skip("prior not built (scripts/build_context_prior.py)")

    m = ImageClassifier.__new__(ImageClassifier)
    m.classes = ["melanoma", "melanocytic_Nevi"]
    m.decision_weights = {}
    m.context_prior = _json.loads(prior_path.read_text(encoding="utf-8"))
    m.prior_strength = 1.0

    for meta in ({}, None, {"age": "", "localization": ""},
                 {"age": "not-a-number", "localization": "nowhere-in-the-table"}):
        lift = m.context_lift(meta)
        assert all(abs(v - 1.0) < 1e-9 for v in lift.values()), \
            f"absent or unknown context must be exactly neutral, got {lift}"


def test_the_prior_moves_decisions_in_the_direction_the_data_says():
    """Old patient on an acral site up, young patient on the back down."""
    import json as _json
    from verifai.models.image import ImageClassifier

    prior_path = REPO / "data" / "priors" / "isic.json"
    if not prior_path.exists():
        pytest.skip("prior not built")

    m = ImageClassifier.__new__(ImageClassifier)
    m.classes = ["melanoma", "melanocytic_Nevi"]
    m.decision_weights = {}
    m.context_prior = _json.loads(prior_path.read_text(encoding="utf-8"))
    m.prior_strength = 1.0

    probs = {"melanoma": 0.30, "melanocytic_Nevi": 0.45}      # argmax says nevus
    old_acral = {"age": "85", "localization": "palms/soles"}
    young_back = {"age": "15", "localization": "posterior torso"}

    assert m.context_lift(old_acral)["melanoma"] > 2.0
    assert m.context_lift(young_back)["melanoma"] < 0.5
    assert m.decide(probs, old_acral) == "melanoma", "context should overturn argmax here"
    assert m.decide(probs, young_back) == "melanocytic_Nevi"
    assert m.decide(probs, None) == "melanocytic_Nevi", "no context, no change"

    # Strength 0 must reproduce plain argmax exactly.
    m.prior_strength = 0.0
    assert m.decide(probs, old_acral) == "melanocytic_Nevi"


def test_one_bucketing_implementation_shared_by_builder_and_adapter():
    """Two copies could drift, and a drifted bucket fails silently.

    If the builder files a 63-year-old under `60-79` and the adapter looks up
    `60-69`, every lookup misses, every lift falls back to 1.0, and the run
    reports "context does not help" without ever having applied context. No
    error, no warning — so the defence is that there is only one implementation.
    """
    from verifai.core import context
    builder = (REPO / "scripts" / "build_context_prior.py").read_text(encoding="utf-8")
    adapter = (REPO / "verifai" / "models" / "image.py").read_text(encoding="utf-8")
    assert "from verifai.core.context import" in builder
    assert "from verifai.core.context import" in adapter
    assert "def age_bucket" not in builder, "the builder must not keep its own copy"
    assert "def age_bucket" not in adapter, "the adapter must not keep its own copy"

    assert context.bucket_for("age", "63.0") == "60-79"
    assert context.bucket_for("age", "") is None
    assert context.bucket_for("localization", " Palms/Soles ") == "palms/soles"
    assert context.bucket_for("unknown_feature", "x") is None


def test_a_prior_is_never_built_from_the_test_manifest():
    """Deriving it from test would fit the decision rule to the answers."""
    import subprocess
    out = subprocess.run(
        [sys.executable, "scripts/build_context_prior.py",
         "--manifest", "data/manifests/ham10000_test.csv"],
        cwd=REPO, capture_output=True, text=True)
    assert out.returncode != 0, "building a prior from test data must fail"
    assert "refusing" in (out.stdout + out.stderr).lower()


def test_dx_type_is_not_available_as_a_context_feature():
    """It is 100% populated in test and 0% in training, and it is an outcome.

    `histo` records that a clinician already thought the lesion worth cutting
    out. As a feature it would look like a spectacular result and be pure
    leakage — and `audit_split` could not see it, because no image is shared.
    """
    from verifai.core.context import BUCKETERS
    assert "dx_type" not in BUCKETERS
    assert "sex" not in BUCKETERS, "measured lift 0.96 vs 1.03 — not worth the argument"


def test_a_class_absent_from_the_test_set_is_undefined_not_zero():
    """No cases to catch is not the same as catching none of them.

    A sensitivity of 0.0 for a class with no support would read as total failure
    on that class and would drag the balanced average down for a reason that has
    nothing to do with the model.
    """
    pytest.importorskip("numpy")
    from verifai.metrics.performance.classification import _per_class

    classes = ["a", "b", "c"]
    #            predicted a, b, c
    cm = [[8, 2, 0],        # 10 true "a"
          [3, 7, 0],        # 10 true "b"
          [0, 0, 0]]        # class "c" never occurs in this evaluation set
    pc = _per_class(cm, classes)

    assert pc["c"]["support"] == 0
    assert pc["c"]["sensitivity"] is None, "undefined, never 0.0"
    assert pc["a"]["sensitivity"] == 0.8 and pc["b"]["sensitivity"] == 0.7

    contributing = [v["sensitivity"] for v in pc.values() if v["sensitivity"] is not None]
    assert len(contributing) == 2, "the absent class must drop out of the average"
    assert abs(sum(contributing) / len(contributing) - 0.75) < 1e-9


def test_balanced_accuracy_says_how_many_classes_it_averaged():
    """A mean over six classes and a mean over seven share a name.

    They are different quantities, and an external evaluation set is exactly
    where that happens — Derm7pt contains no actinic keratoses. Without the count
    travelling alongside, two numbers from different sets look directly
    comparable on the page and are not.
    """
    import inspect
    from verifai.metrics.performance import classification
    src = inspect.getsource(classification)
    assert '"balanced_accuracy_n_classes"' in src
    assert '"classes_absent_from_test"' in src
    assert "not comparable" in src, \
        "the summary has to say it in words, not only in a field nobody reads"


def test_every_scenario_declares_a_human_label():
    """Without one the comparison table heads its columns with model ids.

    `external-derm7pt-isic` against `external-derm7pt-isic-masked` differ by a
    single decision weight, and neither string says which is which — a reader has
    to decode the table instead of reading it.
    """
    yaml = pytest.importorskip("yaml")
    missing = []
    for path in sorted((REPO / "scenarios").glob("*.yaml")):
        sc = yaml.safe_load(path.read_text(encoding="utf-8"))
        label = sc.get("label")
        if not label:
            missing.append(path.name)
            continue
        assert label != sc.get("model", {}).get("id"), \
            f"{path.name}: label must not simply repeat the model id"
        assert "_" not in label, f"{path.name}: {label!r} reads like an identifier"
    assert not missing, f"scenarios without a label: {missing}"


def test_older_artifacts_fall_back_to_the_card_name():
    """Snapshots written before scenarios declared a label carry the model id.

    Re-running eighteen evaluations to change a caption would be absurd, so the
    app resolves it from the gallery card, which was always human-readable. An
    explicit label must still win — the fallback may not override a real one.
    """
    pytest.importorskip("streamlit")
    sys.path.insert(0, str(REPO / "showcase"))
    import app

    app._CARD_NAMES["some_scenario"] = "A readable name"
    legacy = {"scenario": "some_scenario", "label": "model-id-x", "model_id": "model-id-x"}
    assert app.run_label(legacy) == "A readable name"

    explicit = {"scenario": "some_scenario", "label": "Focal loss (γ=2)",
                "model_id": "model-id-x"}
    assert app.run_label(explicit) == "Focal loss (γ=2)", \
        "an explicit label must survive the fallback"

    # No card either: the model id is still more informative than the scenario
    # directory name, so it stands rather than being replaced by something worse.
    unknown = {"scenario": "never_seen", "label": "mid", "model_id": "mid"}
    assert app.run_label(unknown) == "mid"
    assert app.run_label({"scenario": "never_seen"}) == "never_seen"


# --- the skeleton: placeholders for what is planned and not built -------------
def _heading_ids(path) -> set:
    """Approximate python-markdown's toc slugifier for one file's headings."""
    import re as _re
    import unicodedata as _ud
    ids = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        m = _re.match(r"^#{1,6}\s+(.*)", line)
        if not m:
            continue
        t = _re.sub(r"`([^`]*)`", r"\1", m.group(1).strip().lower())
        t = _re.sub(r"\[\[?([^\]]*)\]\]?\([^)]*\)", r"\1", t)
        t = _re.sub(r"[*_]", "", t)
        t = _ud.normalize("NFKD", t)
        t = _re.sub(r"[^\w\s-]", "", t)
        ids.add(_re.sub(r"\s+", "-", t.strip()))
    return ids


def test_every_placeholder_points_at_a_plan_that_still_exists():
    """A placeholder is a promise. An unresolvable one is a stale promise.

    Each entry in `planned.py` names the document section that says what the
    component is for and what is blocking it. If that section is renamed or
    deleted, the placeholder has to be updated or removed in the same change —
    otherwise the app goes on advertising a plan nobody can read.
    """
    sys.path.insert(0, str(REPO / "showcase"))
    from planned import PLANNED

    assert PLANNED, "the skeleton must not be empty while components are unbuilt"
    for key, entry in PLANNED.items():
        for field in ("title", "shows", "blocked_by", "phase"):
            assert entry.get(field), f"{key} is missing {field}"
        doc, _, anchor = entry["phase"].partition("#")
        path = REPO / "docs" / doc
        assert path.exists(), f"{key} points at a document that does not exist: {doc}"
        assert anchor, f"{key} must name a section, not just a file"
        assert anchor in _heading_ids(path), (
            f"{key} points at docs/{doc}#{anchor}, which is not a heading there")


def test_planned_components_are_hidden_unless_asked_for():
    """The public deploy must never advertise capability it does not have.

    Same rule as the `sample: true` banner, one level up: an empty "Coverage
    map" card on a live page is a claim about the future rendered in the
    present. Skeleton mode is opt-in through the environment, which is why the
    torch-free Streamlit Cloud deploy is safe by default.
    """
    pytest.importorskip("streamlit")
    sys.path.insert(0, str(REPO / "showcase"))
    import render

    assert render.SKELETON == (os.getenv("VERIFAI_SKELETON") == "1")
    if not render.SKELETON:
        assert render.placeholder("coverage_map") is False, \
            "placeholders must not draw unless VERIFAI_SKELETON=1"
    assert render.placeholder("not_a_planned_component") is False


def test_the_showcase_package_stays_free_of_heavy_imports():
    """`showcase/requirements.txt` has no torch and the skeleton must not add one."""
    source = _showcase_source()
    for heavy in ("import torch", "import torchvision", "from torch",
                  "import transformers"):
        assert heavy not in source, f"the showcase must not need {heavy!r}"


# --- the model registry: which models exist, whether or not they have run -----
def _write_scenario(dir_, name, weights=None, project="P", training=False, **extra):
    """A minimal scenario file, enough for the registry builder to read."""
    import yaml as _yaml
    sc = {"name": name, "label": name.replace("_", " "), "project": project,
          "model": {"weights_path": weights} if weights else {},
          "dataset": {"manifest": "m.csv", "id": "d"}, "metrics": ["x.y"], **extra}
    if training:
        sc["training"] = {"arch": "resnet18"}
    (dir_ / f"{name}.yaml").write_text(_yaml.safe_dump(sc), encoding="utf-8")


def test_every_scenario_declares_a_project():
    """The overview groups models by project, and a model without one lands in
    'Unassigned' — a bucket that grows quietly until it is the whole page."""
    yaml = pytest.importorskip("yaml")
    missing = [p.name for p in sorted((REPO / "scenarios").glob("*.yaml"))
               if not yaml.safe_load(p.read_text(encoding="utf-8")).get("project")]
    assert not missing, f"scenarios without a project: {missing}"


def test_the_committed_model_registry_is_in_step_with_the_scenarios():
    """Rebuilding must reproduce the committed file exactly.

    Checkpoints are gitignored, so where they are absent the builder carries
    hashes and provenance forward from the committed file — and equality then
    means the *scenarios* have not moved on without it. Where they are present,
    it also means no checkpoint was retrained without the registry noticing.
    Fix: `python scripts/build_model_registry.py`.
    """
    pytest.importorskip("yaml")
    from verifai.export.model_registry import REGISTRY_PATH, build_registry
    committed = json.loads((REPO / REGISTRY_PATH).read_text(encoding="utf-8"))
    rebuilt = build_registry(REPO / "scenarios", root=REPO, previous=committed)
    assert rebuilt == committed, (
        "showcase/artifacts/model_registry.json is out of date — "
        "run scripts/build_model_registry.py")


def test_a_model_is_its_checkpoint_not_its_model_id():
    """`model.id` names three checkpoints in one direction and one checkpoint
    answers to five ids in the other, so grouping on it would merge different
    models and split one. Every configuration of a model must point at the same
    weights, and no two models may share them."""
    from verifai.export.model_registry import REGISTRY_PATH
    reg = json.loads((REPO / REGISTRY_PATH).read_text(encoding="utf-8"))
    yaml = pytest.importorskip("yaml")
    weights_of = {}
    for p in (REPO / "scenarios").glob("*.yaml"):
        sc = yaml.safe_load(p.read_text(encoding="utf-8"))
        m = sc.get("model") or {}
        weights_of[sc["name"]] = m.get("weights_path") or m.get("repo_id")
    seen = {}
    for model in reg["models"]:
        refs = {weights_of[c["scenario"]] for c in model["configurations"]}
        assert len(refs) == 1, f"{model['key']} mixes weights: {refs}"
        ref = refs.pop()
        assert ref not in seen, f"{model['key']} and {seen.get(ref)} share {ref}"
        seen[ref] = model["key"]
    # and the configurations are all accounted for, once
    listed = [c["scenario"] for m in reg["models"] for c in m["configurations"]]
    assert sorted(listed) == sorted(weights_of), "every scenario is one configuration"


def test_only_the_scenario_named_after_a_checkpoint_trained_it(tmp_path):
    """Several configurations carry a copied `training:` block while evaluating
    someone else's weights. `train_model.py` writes `<out_dir>/<name>.pt`, so
    that name — not the presence of a block — says who trained it."""
    pytest.importorskip("yaml")
    from verifai.export.model_registry import build_registry
    sc = tmp_path / "scenarios"; sc.mkdir()
    _write_scenario(sc, "base", weights="ck/base.pt", training=True)
    _write_scenario(sc, "base_tuned", weights="ck/base.pt", training=True,
                    )   # a copied block, and not the trainer
    reg = build_registry(sc, root=tmp_path)
    (model,) = reg["models"]
    assert model["trained_by"] == "base"
    assert [c["scenario"] for c in model["configurations"]] == ["base", "base_tuned"]


def test_model_provenance_never_carries_a_score(tmp_path):
    """A validation accuracy on the model page would read as the model's result,
    with no interval, beside the test-set report that carries one."""
    pytest.importorskip("yaml")
    from verifai.export.model_registry import build_registry
    sc = tmp_path / "scenarios"; sc.mkdir()
    ck = tmp_path / "ck"; ck.mkdir()
    (ck / "m.pt").write_bytes(b"weights")
    (ck / "m_training.json").write_text(json.dumps({
        "arch": "resnet18", "train_images": 10, "best_val_balanced_accuracy": 0.72,
        "history": [{"epoch": 1, "val_accuracy": 0.6}]}), encoding="utf-8")
    _write_scenario(sc, "m", weights="ck/m.pt", training=True)
    prov = build_registry(sc, root=tmp_path)["models"][0]["provenance"]
    assert prov == {"arch": "resnet18", "train_images": 10}


def test_a_model_belongs_to_one_project(tmp_path):
    pytest.importorskip("yaml")
    from verifai.export.model_registry import build_registry
    sc = tmp_path / "scenarios"; sc.mkdir()
    _write_scenario(sc, "a", weights="ck/a.pt", project="Skin")
    _write_scenario(sc, "a_other", weights="ck/a.pt", project="Chest")
    with pytest.raises(ValueError, match="2 projects"):
        build_registry(sc, root=tmp_path)


def test_models_without_declared_weights_are_never_merged(tmp_path):
    """Two scenarios that name no weights are not therefore the same model.
    An invented shared identity would let a comparison treat them as one."""
    pytest.importorskip("yaml")
    from verifai.export.model_registry import build_registry
    sc = tmp_path / "scenarios"; sc.mkdir()
    _write_scenario(sc, "x"); _write_scenario(sc, "y")
    models = build_registry(sc, root=tmp_path)["models"]
    assert len(models) == 2 and not any(m["identified"] for m in models)


def test_a_missing_checkpoint_keeps_the_identity_it_had(tmp_path):
    """Checkpoints are gitignored. Rebuilding on a machine without them must not
    erase what is known about them — it would silently turn every model's
    content hash back into a path."""
    pytest.importorskip("yaml")
    from verifai.export.model_registry import build_registry
    sc = tmp_path / "scenarios"; sc.mkdir()
    ck = tmp_path / "ck"; ck.mkdir()
    (ck / "m.pt").write_bytes(b"weights")
    (ck / "m_training.json").write_text('{"arch": "resnet18"}', encoding="utf-8")
    _write_scenario(sc, "m", weights="ck/m.pt", training=True)
    first = build_registry(sc, root=tmp_path)
    (ck / "m.pt").unlink(); (ck / "m_training.json").unlink()
    assert build_registry(sc, root=tmp_path, previous=first) == first
    orphaned = build_registry(sc, root=tmp_path)["models"][0]
    assert orphaned["identity"] == "path:ck/m.pt" and orphaned["sha256"] is None


def test_the_showcase_derives_status_and_keeps_unclaimed_evaluations():
    """Status is counted from which reports exist, never stored; and an
    evaluation no model claims is returned, not dropped."""
    pytest.importorskip("streamlit")
    sys.path.insert(0, str(REPO / "showcase"))
    import registry as reg_mod
    model = {"configurations": [{"scenario": "a"}, {"scenario": "b"}]}
    assert reg_mod.model_status(model, {"a", "b"})["state"] == "evaluated"
    assert reg_mod.model_status(model, {"a"}) == {"evaluated": 1, "total": 2, "state": "partly"}
    assert reg_mod.model_status(model, set())["state"] == "not_evaluated"
    fake = {"models": [{"key": "k", "configurations": [{"scenario": "a"}]}]}
    left = reg_mod.unclaimed([{"id": "a"}, {"id": "fixture"}], fake)
    assert [c["id"] for c in left] == ["fixture"]


# --- step 3: projects, the model page, and the model-scoped comparison --------
def test_every_planned_component_has_a_place_on_some_page():
    """A planned entry nothing draws is a promise with no slot — the skeleton
    would claim a component is coming without showing where it goes."""
    import re as _re
    sys.path.insert(0, str(REPO / "showcase"))
    from planned import PLANNED
    # A slot inside a page, or a whole planned page (`planned_pages._stub`).
    placed = set(_re.findall(r'(?:placeholder|_stub)\(\s*"([a-z_]+)"', _showcase_source()))
    assert set(PLANNED) <= placed, f"planned but never placed: {sorted(set(PLANNED) - placed)}"
    assert placed <= set(PLANNED), f"placed but not planned: {sorted(placed - set(PLANNED))}"


def test_a_model_scoped_comparison_shows_exactly_its_configurations():
    """The model page's 'compare' narrows to that checkpoint's configurations —
    and goes through the same scope as the lineage filter, so it inherits the
    same disclosure of hidden runs scored on the same images."""
    pytest.importorskip("streamlit")
    sys.path.insert(0, str(REPO / "showcase"))
    from views.compare import scope_ids
    reg = {"models": [{"key": "m", "name": "Model M",
                       "configurations": [{"scenario": "a"}, {"scenario": "b"}]}]}
    cards = [{"id": "a", "lineage": "L"}, {"id": "c", "lineage": "L"}]
    assert scope_ids(cards, reg, model="m") == ({"a", "b"}, "Model M")
    assert scope_ids(cards, reg, lineage="L") == ({"a", "c"}, "L")
    assert scope_ids(cards, reg) == (None, None)
    # an unknown model falls back to unfiltered rather than to an empty table
    assert scope_ids(cards, reg, model="gone") == (None, None)


def test_a_models_configurations_are_grouped_by_the_images_they_were_scored_on():
    """Only configurations under one heading can be compared directly, so the
    page groups by evaluation set: the model's own test set first, argmax before
    any weighted rule."""
    pytest.importorskip("streamlit")
    sys.path.insert(0, str(REPO / "showcase"))
    from views.model import by_evaluation_set
    model = {"trained_by": "own", "configurations": [
        {"scenario": "ext", "label": "external", "eval_manifest": "a_external.csv",
         "decision_weights": None},
        {"scenario": "own_w", "label": "weighted", "eval_manifest": "z_test.csv",
         "decision_weights": {"melanoma": 5}},
        {"scenario": "own", "label": "as trained", "eval_manifest": "z_test.csv",
         "decision_weights": None},
    ]}
    groups = by_evaluation_set(model)
    assert [m for m, _ in groups] == ["z_test.csv", "a_external.csv"], "home set first"
    assert [c["scenario"] for c in groups[0][1]] == ["own", "own_w"], "argmax first"


def test_model_names_sort_their_numbers_as_numbers():
    pytest.importorskip("streamlit")
    sys.path.insert(0, str(REPO / "showcase"))
    from views.project import natural_key
    names = ["n=2,000", "n=100", "n=7,014", "n=500"]
    assert sorted(names, key=natural_key) == ["n=100", "n=500", "n=2,000", "n=7,014"]


# --- step 5: the report layout ------------------------------------------------
def test_the_info_box_asks_its_five_questions_in_the_documented_order():
    """'Same order every time' is the contract docs/extending.md states. A
    reader who has learned where 'what it does not tell you' sits must find it
    there on every finding."""
    pytest.importorskip("streamlit")
    sys.path.insert(0, str(REPO / "showcase"))
    import render
    assert render.INFO_BOX_ORDER == ("what", "summary", "impact", "how", "limits")
    doc = (REPO / "docs" / "extending.md").read_text(encoding="utf-8")
    table = doc[doc.index("| The reader asks |"):]
    positions = [table.index(k) for k in ("explain.what", "`summary`", "explain.impact",
                                           "explain.how", "explain.limits")]
    assert positions == sorted(positions), "the documented order and the rendered one differ"


def test_a_findings_explanation_is_never_behind_a_click():
    """Progressive disclosure by depth on the page, never by click — the
    settled rule for the primary reader. The expander that hid 'how to read
    this chart' and 'what it does not tell you' must not come back; only the
    reference definitions may sit behind one."""
    source = _showcase_source()
    for label in ("How to read this chart", "What it does not tell you",
                  "What was measured", "What came out"):
        assert f'expander("{label}' not in source, f"{label!r} is behind a click again"


def test_the_report_holds_its_page_when_integrity_is_not_verified():
    """Every other number is conditional on the split. A report whose split was
    never checked — or has no integrity finding at all — must say so above
    everything, not as one sixth of a row."""
    pytest.importorskip("streamlit")
    sys.path.insert(0, str(REPO / "showcase"))
    from views.report import integrity_state
    clean = {"pillar": "integrity", "verdict": "measured", "summary": "clean"}
    unverified = {"pillar": "integrity", "verdict": "unavailable", "summary": "not checked"}
    leaked = {"pillar": "integrity", "verdict": "fail", "summary": "leaked"}   # retired word
    other = {"pillar": "performance", "verdict": "measured"}
    assert integrity_state([clean, other])[0] == "measured"
    assert integrity_state([unverified, other]) == ("unavailable", unverified)
    assert integrity_state([leaked])[0] == "invalid", "a retired `fail` still gates the page"
    assert integrity_state([other]) == ("unavailable", None), "no check is not a clean check"


def test_a_configuration_goes_by_one_name_on_every_page():
    """The comparison table named 18 of 23 configurations differently from the
    model page, because it preferred the label a snapshot recorded at run time.
    The label the scenario declares today wins; the recorded one only names a
    run whose scenario is gone; a card name never overrides either."""
    pytest.importorskip("streamlit")
    sys.path.insert(0, str(REPO / "showcase"))
    import app
    app._SCENARIO_LABELS["renamed"] = "Current name"
    app._CARD_NAMES["renamed"] = "Gallery name"
    recorded = {"scenario": "renamed", "label": "Name at run time", "model_id": "mid"}
    assert app.run_label(recorded) == "Current name"
    gone = {"scenario": "deleted_since", "label": "Name at run time", "model_id": "mid"}
    assert app.run_label(gone) == "Name at run time"
    # and on the real artifacts: every snapshot reads as its model page does
    app.load_catalog()
    import glob as _glob
    reg = json.loads((REPO / "showcase/artifacts/model_registry.json").read_text("utf-8"))
    current = {c["scenario"]: c["label"] for m in reg["models"] for c in m["configurations"]}
    for f in _glob.glob(str(REPO / "showcase/artifacts/*/history/*.json")):
        snap = json.loads(Path(f).read_text(encoding="utf-8"))
        if snap["scenario"] in current:
            assert app.run_label(snap) == current[snap["scenario"]], snap["scenario"]


# --- step 6: the comparison page ----------------------------------------------
def test_comparison_columns_are_named_as_readers_know_them():
    """A header reading `performance.per_class.melanoma.ppv_test_prevalence`
    invites exactly the misreading the glossary exists to prevent. Each column
    gets its term, plus what the pattern's wildcard matched, so melanoma's
    sensitivity and a mole's do not share a name — and names stay unique."""
    pytest.importorskip("streamlit")
    sys.path.insert(0, str(REPO / "showcase"))
    from views.compare import metric_labels
    keys = ["performance.per_class.melanoma.sensitivity",
            "performance.per_class.melanocytic_Nevi.sensitivity",
            "performance.per_class.melanoma.support", "performance.support.melanoma",
            "privacy.mia_auc", "not.in.the.glossary"]
    names = metric_labels(keys)
    assert names["performance.per_class.melanoma.sensitivity"].endswith("· melanoma")
    assert names["privacy.mia_auc"] == "Membership-inference AUC"
    assert names["not.in.the.glossary"] == "not.in.the.glossary", "unknown keys stay raw"
    assert len(set(names.values())) == len(keys), f"names collide: {names}"


def test_the_tradeoff_frontier_respects_each_axis_direction():
    """Grey on the scatter means beaten on both axes by another run — a fact.
    With privacy's AUC lower-is-better, a run that is worse on that axis is not
    'beaten' merely for having the larger number."""
    pytest.importorskip("streamlit")
    sys.path.insert(0, str(REPO / "showcase"))
    from views.compare import pareto_front
    pts = [("a", 0.9, 0.6), ("b", 0.8, 0.5), ("c", 0.7, 0.7), ("d", None, 0.1)]
    assert pareto_front(pts, "higher", "lower") == {"a", "b"}   # c is beaten by b on both
    assert pareto_front(pts, "higher", "higher") == {"a", "c"}  # b is beaten by a on both
    assert pareto_front([("x", 1, 1), ("y", 1, 1)], "higher", "higher") == {"x", "y"}, \
        "identical runs do not beat each other"


def test_the_comparison_table_has_a_row_per_run_and_marks_only_ranked_leaders():
    pytest.importorskip("streamlit")
    sys.path.insert(0, str(REPO / "showcase"))
    from views.compare import BEATEN_BY, comparison_table
    runs = [{"scenario": "p", "label": "P", "model_id": "p", "metrics": {"m.up": 0.9, "m.free": 3}},
            {"scenario": "q", "label": "Q", "model_id": "q", "metrics": {"m.up": 0.5, "m.free": 7}}]
    table, leaders = comparison_table(runs, ["m.up", "m.free"],
                                      {"m.up": "higher", "m.free": None}, {"Q": "P"})
    assert list(table["Run"]) == ["P", "Q"], "runs are rows"
    assert leaders == {"m.up": "P", "m.free": None}, "an undeclared metric is never ranked"
    assert list(table[BEATEN_BY]) == ["", "P"]


# --- dependencies: one source of truth, two files exported from it ------------
def _group(name: str) -> list[str]:
    import re as _re
    import tomllib
    groups = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))["dependency-groups"]
    return sorted(_re.split(r"[<>=!~;\[ ]", d, maxsplit=1)[0].lower().replace("_", "-")
                  for d in groups[name])


def _requirement_names(path) -> list[str]:
    import re as _re
    names = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith(("#", "-")):
            names.append(_re.split(r"[<>=!~;\[ ]", line, maxsplit=1)[0].lower().replace("_", "-"))
    return names


def test_the_public_showcase_never_installs_torch():
    """The free tier of Streamlit Community Cloud is what keeps the demo
    always-on, and it holds only while the showcase's install list stays light.
    Checked at the source (the `showcase` group) and in the file Cloud actually
    reads — the second is exported, and an export can pick up anything."""
    heavy = {"torch", "torchvision", "torchaudio", "transformers"}
    assert not heavy & set(_group("showcase")), "the showcase group must stay torch-free"
    installed = set(_requirement_names(REPO / "showcase" / "requirements.txt"))
    assert not heavy & installed, f"showcase/requirements.txt pulls in {heavy & installed}"
    assert {"streamlit", "plotly", "pandas", "pillow"} <= installed, \
        "showcase/requirements.txt must hold what the showcase imports"


def test_the_notebooks_engine_list_names_what_the_engine_group_names():
    """requirements-engine.txt is unpinned on purpose — on Colab a pinned torch
    would replace the platform's GPU build — so versions may float, but which
    packages it installs may not drift from pyproject.toml."""
    assert sorted(_requirement_names(REPO / "requirements-engine.txt")) == _group("engine")


# --- active and archived: re-run what is current, keep the rest as the record --
def test_every_scenario_declares_whether_it_is_active():
    """Active scenarios are re-run when a metric changes; archived ones never are.
    A scenario that says neither would be silently one or the other."""
    yaml = pytest.importorskip("yaml")
    bad = {p.name: yaml.safe_load(p.read_text(encoding="utf-8")).get("status")
           for p in sorted((REPO / "scenarios").glob("*.yaml"))}
    bad = {k: v for k, v in bad.items() if v not in ("active", "archived")}
    assert not bad, f"scenarios without status: active|archived: {bad}"


def test_every_registered_metric_has_a_version():
    """A report records the version of each metric that produced it. A metric
    with no version could change without any report ever looking behind."""
    from verifai.core.run import METRIC_REGISTRY
    from verifai.core.suite import METRIC_VERSIONS
    assert set(METRIC_REGISTRY) == set(METRIC_VERSIONS)
    assert all(isinstance(v, int) and v >= 1 for v in METRIC_VERSIONS.values())


def test_the_registry_tells_the_showcase_the_current_metric_versions():
    from verifai.core.suite import METRIC_VERSIONS
    from verifai.export.model_registry import REGISTRY_PATH
    reg = json.loads((REPO / REGISTRY_PATH).read_text(encoding="utf-8"))
    assert reg["metric_versions"] == METRIC_VERSIONS
    for m in reg["models"]:
        assert m["active"] == any(c["status"] == "active" for c in m["configurations"])


def test_an_active_report_says_why_it_is_behind():
    """Only three things make an active report out of date: it predates
    versioning, a metric it ran has a newer version, or its weights changed."""
    pytest.importorskip("streamlit")
    sys.path.insert(0, str(REPO / "showcase"))
    from registry import out_of_date
    reg = {"metric_versions": {"a.metric": 2, "b.metric": 1}}
    model = {"sha256": "abc"}
    current = {"metric_versions": {"a.metric": 2, "b.metric": 1}, "checkpoint": {"sha256": "abc"}}
    assert out_of_date(current, reg, model) == []
    older = {**current, "metric_versions": {"a.metric": 1, "b.metric": 1}}
    assert len(out_of_date(older, reg, model)) == 1 and "version 1 here, 2 now" in out_of_date(older, reg, model)[0]
    retrained = {**current, "checkpoint": {"sha256": "def"}}
    assert out_of_date(retrained, reg, model) == ["the checkpoint has been retrained since"]
    assert len(out_of_date({}, reg, model)) == 1, "a report predating versioning is behind, once"


def test_archived_models_are_one_switch_away_never_gone():
    pytest.importorskip("streamlit")
    sys.path.insert(0, str(REPO / "showcase"))
    from views.project import visible_models
    models = [{"key": "a", "active": True}, {"key": "b", "active": False}]
    assert [m["key"] for m in visible_models(models, False)] == ["a"]
    assert [m["key"] for m in visible_models(models, True)] == ["a", "b"]


def test_run_active_runs_exactly_the_active_scenarios():
    pytest.importorskip("yaml")
    sys.path.insert(0, str(REPO / "scripts"))
    from run_active import active_scenarios
    import yaml as _yaml
    expected = sorted(p.name for p in (REPO / "scenarios").glob("*.yaml")
                      if _yaml.safe_load(p.read_text(encoding="utf-8")).get("status") == "active")
    assert sorted(p.name for p in active_scenarios(REPO / "scenarios")) == expected
    assert expected, "at least one scenario must stay active"


# --- every page renders -------------------------------------------------------
def _render_page(params):
    """Run one page function the way st.navigation would, with its route set."""
    import sys as _sys
    import importlib as _il
    import streamlit as _st
    _sys.path.insert(0, "showcase")
    page = params.pop("_page")
    for key, value in params.items():
        _st.session_state[key] = value
    _il.import_module(f"views.{page}").page()


@pytest.mark.parametrize("params", [
    {"_page": "overview"},
    {"_page": "project", "project": "Skin lesion classification"},
    {"_page": "model", "model": "skin_cancer_isic"},
    {"_page": "model", "model": "skin_cancer_focal"},
    {"_page": "report", "run": "skin_cancer_isic"},
    {"_page": "report", "run": "skin_cancer"},
    {"_page": "compare"},
    {"_page": "compare", "model": "skin_cancer_isic"},
    {"_page": "compare", "lineage": "ResNet18 · clean split"},
], ids=lambda p: "-".join(str(v) for v in p.values()))
def test_every_page_renders_without_an_exception(params, monkeypatch):
    """The unit tests cover the helpers; this covers the pages that call them.
    A broken call inside a page — an argument slipped into the wrong bracket —
    passed every other test and crashed the whole comparison page."""
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    monkeypatch.chdir(REPO)
    at = AppTest.from_function(_render_page, args=(dict(params),), default_timeout=120)
    at.run()
    assert not at.exception, [e.value for e in at.exception]


# --- M2: baselines and the findings strip ---------------------------------------
def test_a_claim_is_made_in_whichever_direction_the_interval_excludes_the_reference():
    """Symmetric in good and bad news: below chance is as established as above it."""
    from verifai.metrics import _baseline as B
    above = B.membership_inference({"mia_auc": 0.6, "mia_auc_ci": [0.55, 0.65]})
    below = B.membership_inference({"mia_auc": 0.4, "mia_auc_ci": [0.35, 0.45]})
    spans = B.membership_inference({"mia_auc": 0.52, "mia_auc_ci": [0.48, 0.56]})
    assert above["cleared"] and "above chance" in above["claim"]
    assert below["cleared"] and "below chance" in below["claim"]
    assert not spans["cleared"] and spans["claim"] is None, "an interval spanning 0.5 claims nothing"


def test_accuracy_is_compared_with_always_answering_the_most_common_class():
    from verifai.metrics import _baseline as B
    value = {"accuracy": 0.8, "accuracy_ci": [0.78, 0.82], "n": 100,
             "per_class": {"common": {"support": 70}, "rare": {"support": 30}}}
    b = B.classification(value)
    assert b["kind"] == "chance" and b["value"] == 0.7 and b["cleared"]
    assert "common" in b["basis"]


def test_an_unreached_ideal_is_published_as_a_gap_never_as_a_claim():
    """Every model changes some answers under noise; 'less than perfect' would be
    established for all of them and say nothing."""
    from verifai.metrics import _baseline as B
    b = B.corruption({"mean_stability": 0.76})
    assert not b["cleared"] and b["claim"] is None and b["gap"] == 0.24


def test_the_strip_holds_only_measured_findings_that_clear_in_pillar_order():
    pytest.importorskip("streamlit")
    sys.path.insert(0, str(REPO / "showcase"))
    from views.report import established

    def f(pillar, verdict, cleared):
        return {"pillar": pillar, "verdict": verdict,
                "details": {"baseline": {"cleared": cleared, "claim": "x", "kind": "chance", "basis": "b"}}}
    findings = [f("privacy", "measured", True), f("integrity", "measured", True),
                f("fairness", "insufficient", True), f("robustness", "measured", False)]
    assert [x["pillar"] for x in established(findings, "measured")] == ["integrity", "privacy"]
    assert established(findings, "invalid") == [], "nothing is established on a contaminated split"
    assert established([{"pillar": "performance", "verdict": "measured", "details": {}}], "measured") is None, \
        "a report predating baselines has no strip, rather than an empty one"


def test_each_pillar_card_says_what_it_was_compared_with_and_never_grades():
    """The strip and the glance list became one card per pillar. What was
    established is a mark on the card; what was not says why, in words that
    speak about the evidence rather than the model."""
    pytest.importorskip("streamlit")
    sys.path.insert(0, str(REPO / "showcase"))
    from views.report import established, reference_line

    def f(pillar, verdict, cleared, kind="chance"):
        return {"pillar": pillar, "verdict": verdict,
                "details": {"baseline": {"cleared": cleared, "claim": "the claim" if cleared else None,
                                         "kind": kind, "basis": "the basis"}}}
    perf, robust, thin = (f("performance", "measured", True), f("robustness", "measured", False, "ideal"),
                          f("fairness", "insufficient", True, "control"))
    hits = established([perf, robust, thin], "measured")
    assert reference_line(perf, hits, "measured").startswith("✓ **Established against chance**")
    assert "falls short of an ideal" in reference_line(robust, hits, "measured"), \
        "an unreachable ideal must say why nothing is claimed, not vanish"
    assert "does not support a claim" in reference_line(thin, hits, "measured")
    assert reference_line({"pillar": "privacy", "verdict": "measured", "details": {}}, hits,
                          "measured") is None, "no baseline, no line"
    for word in ("pass", "fail", "good", "bad"):
        for x in (perf, robust, thin):
            assert word not in reference_line(x, hits, "measured").lower().split()

    source = (REPO / "showcase" / "views" / "report.py").read_text(encoding="utf-8")
    assert "What this evaluation established" not in source, \
        "the separate strip is merged into the pillar cards; it must not come back as a second list"


def test_every_measured_summary_states_n_and_an_interval():
    """The summary is a template in the metric, so its rules can be checked: a
    reader must learn how many images a number rests on, and how uncertain it is.

    Integrity is exempt from the interval — a count of shared identifiers is
    exact, not a sample estimate.
    """
    import re
    from verifai.core.suite import METRIC_VERSIONS  # noqa: F401  (current reports only)
    registry = json.loads((REPO / "showcase" / "artifacts" / "model_registry.json").read_text())
    active = {c["scenario"] for m in registry["models"] for c in m["configurations"]
              if c.get("status") == "active"}
    assert active, "no active configuration — this test would pass vacuously"
    n_images = re.compile(r"\d[\d,]*\s+(?:[\w-]+\s+)?images")
    interval = re.compile(r"\[-?\d+\.\d+–-?\d+\.\d+\]")
    for scenario in active:
        report = json.loads((REPO / "showcase" / "artifacts" / scenario / "report.json").read_text())
        for f in report["findings"]:
            if f["verdict"] not in ("measured", "insufficient") or not f.get("value"):
                continue
            s_ = f["summary"]
            assert n_images.search(s_), f"{scenario}/{f['metric']}: no image count in {s_!r}"
            if f["pillar"] != "integrity" and (f["details"] or {}).get("enough_per_bin", True):
                assert interval.search(s_), f"{scenario}/{f['metric']}: no interval in {s_!r}"
            assert "_" not in re.sub(r"`[^`]*`", "", s_), \
                f"{scenario}/{f['metric']}: an identifier leaked into {s_!r}"


def test_every_finding_in_a_current_active_report_carries_a_baseline():
    """The runner attaches one to every finding; a current active report without
    one means a metric is missing from BY_FINDING."""
    from verifai.core.suite import METRIC_VERSIONS
    from verifai.metrics._baseline import BY_FINDING
    reg = json.loads((REPO / "showcase/artifacts/model_registry.json").read_text("utf-8"))
    active = [c["scenario"] for m in reg["models"] for c in m["configurations"] if c["status"] == "active"]
    for scen in active:
        report = json.loads((REPO / "showcase/artifacts" / scen / "report.json").read_text("utf-8"))
        if report["meta"].get("metric_versions") != {k: METRIC_VERSIONS[k] for k in report["meta"].get("metric_versions", {})}:
            continue                     # behind the current metrics: the app says so
        for finding in report["findings"]:
            assert finding["metric"] in BY_FINDING, f"{scen}: no baseline function for {finding['metric']}"
            assert "baseline" in finding["details"], f"{scen}: {finding['metric']} has no baseline"


def test_the_random_control_moves_the_region_without_changing_it():
    """The control may differ from the highlight in where it is and nothing else."""
    np = pytest.importorskip("numpy")
    from verifai.metrics.explainability.gradcam import _shifted
    mask = np.zeros((40, 60), dtype=bool); mask[5:15, 10:30] = True
    moved = _shifted(mask, np.random.default_rng(0))
    assert moved.shape == mask.shape and moved.sum() == mask.sum()

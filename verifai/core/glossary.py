"""Plain-language explanations for the metric keys used in comparisons.

Why this lives in the engine
----------------------------
`Finding.details["explain"]` explains one metric's *chart* inside one report.
The comparison view does not read findings — it reads the flattened snapshot
keys (`<pillar>.<path>`, produced generically by `snapshot_metrics`), so a chart
explanation cannot reach it. This module fills that gap, and stays engine-side
for the same reason the per-finding text does: adding a metric should mean
adding its wording here, never editing the app.

Not in the snapshot, on purpose. `directions` travels per run because it is a
property of that run's metric. These explanations are static reference text
about the *concept*, identical for every run — copying them into every snapshot
would duplicate the same paragraphs across every artifact and still leave older
artifacts without them.

Shape of an entry
-----------------
Deliberately uniform, so the reader learns the same four things every time:

    measures  what the number counts, and why anyone cares
    ideal     the value you would see if it were perfect
    reading   what a given value actually means, with a worked fraction
    tension   (optional) the metric it trades against — the reason a single
              number is never the whole answer

Written for someone meeting these terms for the first time, in general terms
rather than about any one dataset: a reader should be able to apply the same
sentence to a different model on a different problem.

Keys are `fnmatch` patterns and are matched **in order**, so a specific pattern
must precede the general one it would otherwise be swallowed by. That mirrors
`showcase/app.py::direction_for`, which resolves metric directions the same way.
"""
from __future__ import annotations

import fnmatch

# (pattern, entry) — order matters: first match wins.
GLOSSARY: list[tuple[str, dict[str, str]]] = [
    # ---------- performance -------------------------------------------------
    ("performance.per_class.*.sensitivity", {
        "term": "Sensitivity (recall)",
        "measures": "Of all the cases that truly belong to a class, the share the "
                    "model actually found. Also called recall or the true-positive "
                    "rate. This is the metric that answers \"how much does it miss?\"",
        "ideal": "1.0 — every real case caught. Nothing is missed.",
        "reading": "0.80 means one in five real cases was missed. When missing a case "
                   "is the expensive mistake — a disease left undetected, a fraud let "
                   "through — this is the number to read first, because overall "
                   "accuracy can stay high while a rare class is missed almost "
                   "entirely.",
        "tension": "Precision (PPV). Any change that catches more real cases also "
                   "flags more things that turn out to be nothing.",
    }),
    ("performance.per_class.*.ppv_test_prevalence", {
        "term": "Precision (PPV)",
        "measures": "Of everything the model *labelled* as a class, the share that "
                    "really was. Also called precision or positive predictive value. "
                    "It answers \"when it says yes, can I believe it?\"",
        "ideal": "1.0 — every alarm is a real case, no false alarms.",
        "reading": "0.40 means three of every five alarms are false. A model can reach "
                   "a high PPV simply by being reluctant to ever say yes, so a strong "
                   "PPV alongside a weak sensitivity usually describes a cautious "
                   "model rather than a good one.",
        "tension": "Sensitivity. Reading these two apart is the most common way to "
                   "misjudge a classifier.",
    }),
    ("performance.per_class.*.specificity", {
        "term": "Specificity",
        "measures": "Of all the cases that do *not* belong to a class, the share the "
                    "model correctly left alone. The true-negative rate.",
        "ideal": "1.0 — nothing irrelevant is ever flagged.",
        "reading": "0.95 sounds excellent, but on a rare class it can still mean a lot "
                   "of false alarms in absolute terms: 5% of a large majority class is "
                   "a bigger number than 100% of a small one. Always read it next to "
                   "the class sizes.",
    }),
    ("performance.per_class.*.support", {
        "term": "Class size (support)",
        "measures": "How many test cases of this class exist. Not a score — the "
                    "sample size every other number for this class rests on.",
        "ideal": "There is no good or bad value; more is simply more certain.",
        "reading": "A rate computed on 13 cases moves by 7.7 points if a single one "
                   "flips, so its confidence interval is wide and small differences "
                   "between models cannot be resolved at all. Check this before "
                   "believing any per-class comparison.",
    }),
    ("performance.per_class_recall.*", {
        "term": "Per-class recall",
        "measures": "The same quantity as per-class sensitivity: of all the real cases "
                    "of a class, the share the model found.",
        "ideal": "1.0 for every class — no class is systematically missed.",
        "reading": "Read the values across classes rather than one at a time. A wide "
                   "spread means the model is strong on common classes and weak on "
                   "rare ones, which a single overall accuracy figure hides completely.",
    }),
    ("performance.support.*", {
        "term": "Class size (support)",
        "measures": "How many test cases each class contributes.",
        "ideal": "No ideal value — this is context, not performance.",
        "reading": "Imbalance here explains most surprises elsewhere. If one class is "
                   "two thirds of the data, a model that only ever predicts that class "
                   "already scores two thirds accuracy while being useless.",
    }),
    ("performance.accuracy", {
        "term": "Accuracy (top-1)",
        "measures": "The share of all predictions that were correct, counted over "
                    "every case equally.",
        "ideal": "1.0 — every case classified correctly, across all classes.",
        "reading": "On imbalanced data this is the most misleading number available. "
                   "Always compare it against the largest class's share: if 67% of the "
                   "data is one class, then 67% accuracy is what guessing that class "
                   "every time would score.",
        "tension": "Balanced accuracy, which weights every class equally instead.",
    }),
    ("performance.balanced_accuracy_n_classes", {
        "term": "Classes in the balanced average",
        "measures": "How many classes actually contributed to the balanced accuracy "
                    "beside it — that is, how many occur in this evaluation set at all.",
        "ideal": "Equal to the number of classes the model can predict; anything lower "
                 "means the evaluation set does not contain them all.",
        "reading": "This is what makes two balanced accuracies comparable or not. A mean "
                   "over six classes and a mean over seven share a name and are different "
                   "quantities, and an external evaluation set is exactly where they "
                   "diverge. Check this before reading any difference between two runs as "
                   "a difference in the models.",
    }),
    ("performance.classes_absent_from_test", {
        "term": "Classes absent from the test set",
        "measures": "How many classes the model can predict that never appear in the "
                    "evaluation data.",
        "ideal": "0 — every class the model knows is represented and can be scored.",
        "reading": "Their sensitivity is undefined rather than zero: there is nothing to "
                   "catch, so failing to catch it means nothing. The model can still "
                   "predict them, and every such prediction is automatically wrong, which "
                   "shows up in overall accuracy but in no per-class recall.",
    }),
    ("performance.balanced_accuracy", {
        "term": "Balanced accuracy",
        "measures": "The average of the per-class sensitivities — each class counts "
                    "the same regardless of how many cases it has.",
        "ideal": "1.0. Chance level is 1 divided by the number of classes (0.14 for "
                 "seven classes, 0.5 for two).",
        "reading": "Much lower than plain accuracy means the model is carried by the "
                   "common classes and weak on the rare ones. Because it averages, it "
                   "can also hide the opposite trade: one class improving while "
                   "another degrades leaves the average unchanged.",
    }),
    ("performance.top3_accuracy", {
        "term": "Top-3 accuracy",
        "measures": "How often the correct answer is among the model's three highest-"
                    "ranked guesses, rather than only its first.",
        "ideal": "1.0 — the right answer is always in the shortlist.",
        "reading": "The gap between this and top-1 separates two very different "
                   "problems. High top-3 with low top-1 means the model *has* the "
                   "information and is committing badly — fixable by changing the "
                   "decision rule, with no retraining. Low top-3 means the information "
                   "is not there, and only better inputs, features or models help.",
    }),
    ("performance.correct", {
        "term": "Correct predictions",
        "measures": "The plain count of correct predictions, before any rate is "
                    "computed from it.",
        "ideal": "Equal to the number of cases evaluated — nothing got wrong.",
        "reading": "Useful as a sanity check against the accuracy rate, and a reminder "
                   "of scale: the same percentage means something different over 7 "
                   "cases than over 7,000.",
    }),
    ("performance.n", {
        "term": "Sample size",
        "measures": "How many cases the performance numbers were computed over.",
        "ideal": "No ideal value; larger means narrower confidence intervals.",
        "reading": "This sets the resolution of every comparison. Below roughly 30 "
                   "cases most differences cannot be distinguished from chance at all, "
                   "which is why small evaluations are reported as plausibility checks "
                   "rather than results.",
    }),

    # ---------- fairness ----------------------------------------------------
    ("fairness.accuracy_gap", {
        "term": "Subgroup accuracy gap",
        "measures": "The distance between the best-served and worst-served subgroup, "
                    "in accuracy. A single number for \"does this work equally well "
                    "for everyone?\"",
        "ideal": "0.0 — every subgroup served equally well.",
        "reading": "0.20 means twenty points separate the group the model treats best "
                   "from the group it treats worst. A gap is only worth claiming when "
                   "the groups' confidence intervals actually separate; on small "
                   "subgroups an apparent gap is often just noise.",
    }),
    ("fairness.gap", {
        "term": "Subgroup gap",
        "measures": "The spread between the strongest and weakest subgroup result.",
        "ideal": "0.0 — every subgroup sees the same result, none disadvantaged.",
        "reading": "Read it together with the subgroup sizes. A large gap resting on a "
                   "handful of cases is a reason to collect more data, not yet a "
                   "finding about the model.",
    }),
    ("fairness.subgroup_accuracy.*", {
        "term": "Subgroup accuracy",
        "measures": "How well the model performs within one subgroup, measured "
                    "separately rather than blended into the overall average.",
        "ideal": "The same value for every subgroup — equality matters more here than "
                 "the level.",
        "reading": "An overall score is an average weighted by group size, so a large "
                   "group can carry it while a small group is served badly. That is "
                   "exactly what splitting by subgroup is for. Beware: a model can "
                   "improve on average by getting better where the data already was "
                   "and worse everywhere else.",
    }),
    ("fairness.coverage.*", {
        "term": "Subgroup coverage",
        "measures": "How many cases fall into one subgroup — the representation of "
                    "that group in the evaluation data.",
        "ideal": "Enough in every group to support a claim, ideally matching the "
                 "population the model will be used on.",
        "reading": "Thin coverage is itself a finding: a model cannot be shown to work "
                   "for a group that is barely present in the test set. Absence of "
                   "evidence here is not evidence of fairness.",
    }),
    ("fairness.n", {
        "term": "Sample size (fairness)",
        "measures": "Total cases across all subgroups in the fairness analysis.",
        "ideal": "No ideal value; it bounds how finely the data can be split.",
        "reading": "Dividing a small evaluation set into subgroups makes each one "
                   "smaller still, so subgroup conclusions are always less certain "
                   "than the overall number they come from.",
    }),

    # ---------- robustness --------------------------------------------------
    ("robustness.prediction_stability.*", {
        "term": "Prediction stability (per corruption)",
        "measures": "How often the prediction stays the same after one specific, "
                    "harmless distortion — noise, blur, a brightness shift, "
                    "recompression. The label should not depend on such things.",
        "ideal": "1.0 — the decision never changes, because the content did not.",
        "reading": "0.70 means three in ten predictions flip on an input a human would "
                   "call unchanged. Comparing the distortions tells you *what* the "
                   "model is sensitive to, which is more actionable than one average.",
    }),
    ("robustness.mean_stability", {
        "term": "Mean prediction stability",
        "measures": "Prediction stability averaged over all the distortions tested. A "
                    "summary of how easily the model can be knocked off its answer.",
        "ideal": "1.0 — completely insensitive to changes that carry no meaning.",
        "reading": "Notably: this says nothing about whether the predictions are "
                   "*right*. A model that is confidently and consistently wrong scores "
                   "a perfect 1.0. Read it strictly alongside accuracy.",
    }),
    ("robustness.clean", {
        "term": "Clean accuracy",
        "measures": "Accuracy on the untouched images — the reference point the "
                    "corrupted versions are compared against.",
        "ideal": "As high as possible; it is the ceiling the corrupted scores fall from.",
        "reading": "On its own it is just accuracy. Its role here is the size of the "
                   "drop when distortion is applied.",
    }),
    ("robustness.mean_corrupted", {
        "term": "Accuracy under corruption",
        "measures": "Average accuracy once the distortions are applied.",
        "ideal": "Equal to the clean accuracy — meaning distortion cost nothing.",
        "reading": "The distance between this and the clean score is the real result. "
                   "A large drop means the model leans on details that survive in "
                   "clean data but not in ordinary real-world variation.",
    }),
    ("robustness.n", {
        "term": "Sample size (robustness)",
        "measures": "How many cases the robustness check was run over.",
        "ideal": "No ideal value; larger means more reliable stability rates.",
        "reading": "Stability is a proportion like any other, so small samples give it "
                   "wide intervals and small differences between models stop being "
                   "resolvable.",
    }),

    # ---------- privacy -----------------------------------------------------
    ("privacy.mia_auc", {
        "term": "Membership-inference AUC",
        "measures": "Whether an attacker could tell that a particular case was in the "
                    "training data, judged from the model's confidence alone. Models "
                    "tend to be more certain about what they memorised.",
        "ideal": "0.5 — pure chance, meaning training members are indistinguishable "
                 "from strangers. **Lower is better here**, which is the opposite of "
                 "what an AUC usually means.",
        "reading": "0.5 is a coin flip and is the good case. 0.65 means the attacker "
                   "wins noticeably more often than chance, so the model leaks "
                   "membership — a genuine privacy problem when training on personal "
                   "data. A verdict should be taken on the interval's upper bound, not "
                   "the point estimate.",
    }),
    ("privacy.mean_confidence_members", {
        "term": "Confidence on training data",
        "measures": "How confident the model is, on average, about cases it was "
                    "trained on.",
        "ideal": "No ideal in isolation — it matters only next to the non-member value.",
        "reading": "This is one half of the membership signal. Read it as a pair with "
                   "the non-member confidence.",
    }),
    ("privacy.mean_confidence_non_members", {
        "term": "Confidence on unseen data",
        "measures": "How confident the model is, on average, about cases it has never "
                    "seen.",
        "ideal": "As close as possible to the members' confidence.",
        "reading": "A large gap between members and non-members *is* the leak: it means "
                   "confidence alone reveals who was in the training set. Equal "
                   "confidence on both sides is the healthy result.",
    }),
    ("privacy.n_members", {
        "term": "Members tested",
        "measures": "How many training cases the privacy attack was evaluated on.",
        "ideal": "No ideal value; enough to make the AUC's interval meaningful.",
        "reading": "Too few on either side and the attack's AUC is too uncertain to "
                   "clear or condemn the model.",
    }),
    ("privacy.n_non_members", {
        "term": "Non-members tested",
        "measures": "How many unseen cases the privacy attack was evaluated on.",
        "ideal": "No ideal value; ideally comparable to the member count.",
        "reading": "Both sides should be drawn the same way apart from membership "
                   "itself. If members and non-members differ in some other respect, "
                   "the attack measures that difference instead of memorisation.",
    }),

    # ---------- integrity ---------------------------------------------------
    ("integrity.contamination", {
        "term": "Contamination rate",
        "measures": "The share of the test set that the model already saw while "
                    "training. The precondition every other number depends on.",
        "ideal": "0.0 — strictly nothing shared between training and test.",
        "reading": "Anything above zero means the evaluation is partly a memory test, "
                   "and scores are inflated by an unknown amount. This is not a metric "
                   "to trade off against others: if it is not zero, the remaining "
                   "numbers do not describe how the model handles new cases.",
    }),
    ("integrity.shared_ids", {
        "term": "Shared items",
        "measures": "How many individual test items also appear in the training data.",
        "ideal": "0 — not one item appears on both the training and test side.",
        "reading": "Even a few shared items inflate scores. This is the most direct "
                   "form of leakage and also the easiest to check.",
    }),
    ("integrity.shared_groups", {
        "term": "Shared groups",
        "measures": "How many *groups* span both sides — several records of the same "
                    "underlying subject, patient or object.",
        "ideal": "0 — no subject appears in both training and test.",
        "reading": "Subtler than shared ids and just as damaging: a second photograph "
                   "of a memorised subject is not a fair test question. Whenever the "
                   "data has a natural grouping, splits must be made on the group, "
                   "never on individual rows.",
    }),
    ("integrity.n_test", {
        "term": "Test items checked",
        "measures": "How many cases were checked for leakage on the test side.",
        "ideal": "Should equal the evaluation set size — everything gets checked.",
        "reading": "If this is lower than the number of cases scored, part of the test "
                   "set went unverified.",
    }),
    ("integrity.n_train", {
        "term": "Training items compared",
        "measures": "How many training cases the test set was compared against.",
        "ideal": "The complete training set, so nothing escapes the comparison.",
        "reading": "A leakage check is only as strong as the training manifest it "
                   "compares to. Unlisted training data cannot be detected.",
    }),
    ("integrity.training_manifests", {
        "term": "Training manifests on record",
        "measures": "How many lists of training images were found for this model. Without at "
                    "least one, nothing can check whether a test image was seen in training.",
        "ideal": "Every manifest the model was trained on — so the split can be checked image "
                 "by image.",
        "reading": "Zero is the ordinary case for a downloaded model: its author published the "
                   "weights, not the list of images. It means the split cannot be checked, "
                   "not that it is clean.",
    }),
    ("integrity.n_model_classes", {
        "term": "Classes the model can output",
        "measures": "How many classes the model can answer with — the size of its fixed "
                    "answer list, set when it was trained.",
        "ideal": "Exactly the classes the test images are labelled with — no more, no fewer.",
        "reading": "More than the data holds means some of the model's classes are never "
                   "tested; fewer means some images can never be answered correctly.",
    }),
    ("integrity.n_dataset_classes", {
        "term": "Classes in the test data",
        "measures": "How many distinct labels the test images carry.",
        "ideal": "Exactly the classes the model can output — no more, no fewer.",
        "reading": "Read beside the model's count: an average over classes, such as balanced "
                   "accuracy, is taken over the classes both have, so two runs with different "
                   "counts are not averaging the same thing.",
    }),
    ("integrity.affected_rows", {
        "term": "Affected test cases",
        "measures": "How many test cases are implicated by the overlap found.",
        "ideal": "0 — no test case is compromised by what the model already saw.",
        "reading": "Translates the contamination rate into a count. Usually the "
                   "clearest way to state the problem: 'this many of the test answers "
                   "may have been memorised.'",
    }),

    # ---------- explainability ----------------------------------------------
    ("explainability.mean_deletion_faithfulness", {
        "term": "Explanation faithfulness (deletion)",
        "measures": "Whether the region the explanation highlights is really what the "
                    "decision rested on, tested by deleting that region and watching "
                    "the confidence fall.",
        "ideal": "Higher is better: hiding the highlighted area should sharply reduce "
                 "confidence, proving the highlight was honest.",
        "reading": "A low value means the picture is decorative — it marks a plausible "
                   "region that the model was not in fact using. An explanation nobody "
                   "has tested is a claim, not evidence, and a convincing-looking "
                   "heatmap is exactly how that goes unnoticed.",
    }),
    ("explainability.faithfulness_gain", {
        "term": "Faithfulness beyond chance",
        "measures": "How much more confidence is lost when the explanation's highlighted region "
                    "is hidden than when a region of the same size, placed at random, is hidden "
                    "in the same image.",
        "ideal": "Higher is better; 0 means the highlight matters no more than a random patch.",
        "reading": "This, not the raw drop, is the claim: hiding any part of an image can lower "
                   "confidence, so a highlight only earns trust by beating a random one under "
                   "identical conditions. Read it with its interval — above 0 means the model "
                   "does rely on the region shown, not that relying on it is right.",
        "tension": "A faithful explanation of a wrong decision is still faithful.",
    }),
    ("explainability.mean_random_control", {
        "term": "Random-region control",
        "measures": "The confidence lost when a region of the same size and shape as the "
                    "highlight, moved to a random place, is hidden — the reference the "
                    "explanation is measured against.",
        "ideal": "No ideal value; it is the baseline, not a result.",
        "reading": "Measured in the same run, on the same images, so it carries every side "
                   "effect of masking itself. What the highlighted region loses beyond it is "
                   "what the explanation can take credit for.",
    }),
    ("explainability.n", {
        "term": "Sample size (explainability)",
        "measures": "How many test images the explanation was scored on.",
        "ideal": "No ideal value; larger means more reliable statements.",
        "reading": "Since version 2 this is every test image with a non-flat map; earlier "
                   "reports scored the first seven only.",
    }),
    ("explainability.faithfulness", {
        "term": "Explanation faithfulness",
        "measures": "How well the explanation matches what the model actually did.",
        "ideal": "Higher is better — the explanation and the decision agree.",
        "reading": "Faithfulness is about the explanation, not about the model being "
                   "right. A faithful explanation of a wrong decision is still "
                   "faithful, and considerably more useful than a flattering one.",
    }),
    ("explainability.n_overlays", {
        "term": "Examples shown",
        "measures": "How many examples were rendered with an explanation overlay.",
        "ideal": "No ideal value; enough to be representative rather than cherry-picked.",
        "reading": "A handful of overlays illustrates, it does not establish. Ask how "
                   "the examples were chosen: if only successes are shown, the pictures "
                   "say nothing about the failures.",
    }),
]


# What a finding's heading calls it, keyed by the finding's `metric` id. The id
# is an identifier, and a reader who has never seen one should not have to decode
# `split_leakage` to know which section they are in. Keyed by the finding rather
# than a flattened key because a heading names the whole finding, not one of the
# numbers inside it.
METRIC_NAMES: dict[str, str] = {
    "split_leakage": "Overlap between test and training data",
    "provenance": "Where the model came from",
    "corpus_ancestry": "Overlap between the archives",
    "label_space": "Classes the model and the data share",
    "top1_accuracy": "Classification accuracy",
    "skin_tone_ita": "Accuracy across skin tones",
    "corruption_stability": "Stability under image corruption",
    "gradcam_faithfulness": "Grad-CAM faithfulness",
    "membership_inference_auc": "Membership inference",
}


def metric_name(metric_id: str) -> str:
    """The human name for a finding's metric id, or the id itself if it has none."""
    return METRIC_NAMES.get(metric_id, metric_id)


def explain_metric(key: str) -> dict[str, str] | None:
    """Plain-language entry for a flattened metric key, or None if unknown.

    Patterns are tried in order, so `performance.per_class.*.sensitivity` is
    reached before any broader pattern that would also match it. Returning None
    rather than a guess is deliberate: showing a confident explanation of the
    wrong quantity is worse than showing none, and the app simply omits it.
    """
    for pattern, entry in GLOSSARY:
        if key == pattern or fnmatch.fnmatch(key, pattern):
            return entry
    return None


def entries_for(keys: list[str]) -> list[tuple[dict[str, str], list[str]]]:
    """Deduplicated explanations for `keys`, in first-seen order.

    Several keys routinely share one concept — melanoma sensitivity and nevus
    sensitivity are the same idea measured on different classes — so a naive
    loop prints the identical paragraph several times and buries the reader in
    a wall of repeats. Each concept is returned once, together with every key it
    covers, so the caller can say which columns it applies to.

    Keys with no entry are dropped rather than given a placeholder: see
    `explain_metric`.
    """
    out: list[tuple[dict[str, str], list[str]]] = []
    index: dict[int, int] = {}
    for key in keys:
        entry = explain_metric(key)
        if entry is None:
            continue
        at = index.get(id(entry))
        if at is None:
            index[id(entry)] = len(out)
            out.append((entry, [key]))
        else:
            out[at][1].append(key)
    return out


def metric_keys(value, prefix: str, depth: int = 0) -> list[str]:
    """Flattened keys a finding's `value` contributes, as `<pillar>.<path>`.

    Mirrors `verifai.export.artifacts._flatten`, which produces the keys the
    comparison view sees. Duplicated here on purpose: the showcase imports this
    module to explain a single report's findings, and should not have to pull in
    the export layer to do it. A contract test asserts the two agree, so the
    duplication cannot drift silently.
    """
    if isinstance(value, bool):            # bool is an int; not a metric
        return []
    if isinstance(value, (int, float)):
        return [prefix]
    if isinstance(value, dict) and depth < 6:
        found: list[str] = []
        for k, v in value.items():
            found += metric_keys(v, f"{prefix}.{k}" if prefix else str(k), depth + 1)
        return found
    return []


def pillar_of(key: str) -> str | None:
    """Leading `<pillar>` segment of a flattened key, when it names a real one."""
    head = key.split(".", 1)[0]
    return head if head in {"integrity", "performance", "fairness",
                            "robustness", "explainability", "privacy"} else None

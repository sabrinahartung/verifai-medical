# Glossary

Every term this project uses, in plain language, with a real example from the
[current results](results.md) — the 1,493-image held-out run.

!!! tip "The same definitions are inside the dashboard"
    The comparison view has a **Show metric explanations** switch that prints a card
    for each selected metric — what it measures, its ideal value, how to read the
    number, and which metric it trades against. Those cards live in
    `verifai/core/glossary.py` and are deliberately *general*: they describe the
    concept for any model on any dataset, so they stay true as the numbers change.

    This page is the other half — the same terms, but pinned to the real numbers of
    one concrete run. Update both when a metric is added: the engine module is what
    the app reads, this page is what a reader browsing the docs finds.

---

## Getting the answer right

**The three questions people confuse.** All three are about melanoma, all three are different:

| Question | Metric | This model |
|---|---|---:|
| Of the real melanomas, how many did we **catch**? | sensitivity | 0.638 |
| Of everything that wasn't melanoma, how often did we correctly **not** say melanoma? | specificity | 0.920 |
| When the model **says** melanoma, how often is it right? | PPV | 0.495 |

You could score a perfect sensitivity of 1.0 by calling *every* image melanoma — you'd catch
all of them. But then PPV collapses and every patient gets an unnecessary biopsy. Sensitivity
counts **missed cancers**; PPV counts **false alarms**. They fail in opposite directions.

Accuracy (top-1)
:   How often the model's single best guess is correct. **0.796** here. Simple, and misleading
    on unbalanced data: 1,009 of the 1,493 test images are nevi, so this mostly measures how
    well the model recognises common moles.

Balanced accuracy
:   The same idea, but averaged over *classes* instead of images, so melanoma counts as much as
    nevi. **0.728** — seven points below the headline. That gap *is* the class imbalance,
    made visible.

Sensitivity (recall, true positive rate)
:   Of the cases that really are X, how many did the model find? **Melanoma: 0.638** — it finds
    64 of every 100 melanomas and misses 36.

Specificity (true negative rate)
:   Of the cases that are *not* X, how many did the model correctly not call X?
    **Melanoma: 0.920.**

PPV (positive predictive value, precision)
:   When the model says X, how often is it actually X? **Melanoma: 0.495** — right about half
    the time. Depends heavily on **prevalence** (below), so a PPV measured on one population
    does not transfer to another.

NPV (negative predictive value)
:   When the model says *not* X, how often is that right? The mirror of PPV.

Top-3 (differential) accuracy
:   Is the correct diagnosis among the model's three best guesses? **0.976.** This matches how a
    dermatologist actually uses a suggestion — as a ranked shortlist, not a verdict — and it is
    why a seven-class model is worth keeping. Top-1 alone *understates* this system.

Support
:   How many test images a class has. **melanoma 163, dermatofibroma 13.** Always read a score
    next to its support.

Confusion matrix
:   A grid: rows are the true diagnosis, columns are what the model said. A perfect model is a
    bright diagonal. The bright cells *off* the diagonal are the confusions that matter — read
    the melanoma row to see what melanomas get mistaken for.

Prevalence
:   How common a condition is in the population you are testing. A screening clinic and a
    referral centre differ enormously. **Why it matters:** at sensitivity 0.85 and specificity
    0.90, PPV is 0.49 when prevalence is 10% and only 0.08 when it is 1% — same model, same
    sensitivity, wildly different usefulness.

---

## How sure are we?

Confidence interval (CI)
:   The range the true value plausibly sits in. Flip a coin ten times, get seven heads — you
    don't conclude the coin is 70% heads. Ten flips isn't enough to know, and the interval says
    so.

    **The comparison that makes this concrete:**

    | Class | Score | 95% CI | Images |
    |---|---:|:--:|---:|
    | melanocytic_Nevi | 0.862 | [0.84, 0.88] | 1,009 |
    | dermatofibroma | 0.769 | [0.50, 0.92] | **13** |

    Both look respectable in a plain table. But dermatofibroma rests on 13 images, so the truth
    could be anywhere from a coin flip to excellent. **That number is not a result, it's a
    shrug.**

The overlap rule
:   **If two intervals overlap, you have not shown the two things differ.** This changed a
    conclusion here: everyone assumes the fairness story is "worse on dark skin", but light
    [0.76, 0.81] and dark [0.63, 0.84] **overlap** — so on this data that claim is *not
    demonstrated*. The gap that is real runs between medium and dark.

Wilson score interval
:   The specific method used for proportions. The textbook "normal approximation" misbehaves
    exactly where these metrics live — small samples, and scores near 0 or 1, where it can
    produce impossible bounds below 0 or above 1. Wilson doesn't. **n=7 with 7 correct** reads
    `1.000 [0.65–1.00]` — perfect on paper, deeply uncertain in fact.

AUC (area under the ROC curve)
:   How well a score separates two groups. **0.5 = no better than guessing; 1.0 = perfect
    separation.** Used here for membership inference, where 0.5 is the *good* outcome.

---

## The data

Data leakage
:   When information from the test set was also available during training, so the model is
    partly *recalling* rather than generalising. The score comes out high and nothing crashes.
    **In the dataset this project started from: 80% of test images also appeared in training.**

Lesion-level grouping
:   HAM10000 photographs the same lesion several times. Splitting on *images* still puts a
    second photo of a memorised lesion into the test set — which is not a fair question. So the
    split is made on `lesion_id`: every photo of one lesion lands on the same side.

Stratified split
:   Keeping each class in the same proportion across train/val/test, so rare classes don't
    vanish from the test set. Here: 11–16% of every class landed in test, including
    **163 melanomas**.

Manifest
:   A small CSV listing exactly which images belong to which split, with labels and metadata.
    Committed to git, so "what did the model see" is a file you can diff rather than a claim
    you have to trust.

Train / validation / test
:   **Train** = learned from. **Validation** = used to choose between checkpoints (so the model
    is tuned to it — it counts as *seen*). **Test** = touched once, at the end. Here:
    7,014 / 1,508 / 1,493 images.

Distribution shift
:   When deployment data differs from training data — different scanner, clinic, or population.
    The commonest reason a medical model that scored well in one hospital fails in another.

Class imbalance
:   When some classes are far commoner than others. 67% of this data is nevi. Untreated, a
    model learns to say "nevi" and scores well doing it — which is why training uses **class
    weights** and why *balanced* accuracy is reported.

---

## How the model decides

Decision rule
:   How probabilities become an answer. Not a law of nature — a choice.

argmax
:   The default rule: pick whichever class has the highest probability. It maximises expected
    accuracy, which is *not* the same as being clinically useful, and it systematically
    under-calls rare classes.

Operating point / threshold
:   Choosing a different bar — e.g. flag melanoma when its probability exceeds τ, even if
    another class scores higher. Trades sensitivity against PPV, and the right trade depends on
    intended use. This is the next planned experiment.

Cost-sensitive decision
:   Weighting classes by the cost of missing them: `argmax(p × w)`. Still returns one of seven
    classes; melanoma simply clears a lower bar.

Calibration
:   Whether stated confidence matches reality — of all the cases called "80% melanoma", are
    about 80% melanoma? A model can be accurate and badly calibrated, which is dangerous in
    deployment.

Class weights · oversampling · focal loss
:   Three ways to stop a model ignoring rare classes: penalise their mistakes more, show them
    more often, or focus the loss on hard examples. All reshape the loss; **none add
    information** — which is why more real minority images has a higher ceiling than any of
    them.

---

## The pillars

Grad-CAM
:   A heatmap of which image regions drove the decision. Warm colours = influential. You want
    the heat on the lesion, not on hair, rulers, or the image border.

Deletion faithfulness
:   Checks whether the heatmap is *honest*: mask the highlighted region and measure how far
    confidence falls. A big drop means the highlight really mattered. **0.45 here** — partly
    faithful. A pretty heatmap is not proof of correct reasoning.

ITA (Individual Typology Angle)
:   A label-free estimate of skin tone computed from the healthy skin *around* the lesion, since
    HAM10000 carries no skin-type labels. Binned into light / medium / dark. An estimate from
    pixels, not a clinical assessment.

Corruption stability
:   Whether the prediction survives distortions that shouldn't change a diagnosis — noise, blur,
    brightness, JPEG. **71.7% here.** Note that stability is not correctness: a model that is
    confidently wrong both before and after scores a perfect 1.0.

Membership inference
:   Could an attacker tell whether a specific patient's image was in the training set, from the
    model's confidence alone? Reported as an AUC where **0.5 is ideal**. **0.558 here** — low
    risk.

Split integrity
:   The precondition check: does the test set overlap what the model trained on? Compared by
    lesion as well as by image. If this fails, every other number in the report is void — which
    is why it is not really a "pillar" at all, but a gate in front of them.

---

## This project's own vocabulary

Pillar
:   One dimension of the evaluation: performance, fairness, robustness, explainability, privacy —
    plus integrity, which is a precondition rather than a property, and safety, which asks what
    happens if someone *acts* on the output and is defined only for generative systems.

Finding
:   One metric's result: a value, a verdict, a one-sentence summary, explanatory text, and a
    chart specification. The single unit that flows from engine to dashboard.

Verdict
:   `measured` · `insufficient` · `unavailable` · `invalid`. Deliberately **epistemic, never
    evaluative**: it says what is *known* about a number, not whether the number is good.
    There is no pass and no fail, because what counts as accurate, fair or robust enough
    depends on where the model runs and what being wrong costs — the reader's call, not a
    constant in a metric module. `invalid` is the one hard signal, and it judges the
    *measurement* rather than the model: a contaminated split does not measure generalisation
    at all.

Evidence gate
:   The minimum evidence before a metric may state a result. Accuracy stays `insufficient`
    below n=30; robustness below n=20; a fairness gap needs two groups of ≥10 whose intervals
    separate; membership inference needs 50 per side.

Access level
:   How much of the model this evaluation actually has: `labels` · `probs` · `logits` ·
    `gradients` · `weights` · `training_data`, in increasing order, each level including the ones
    before it. It decides which metrics **can exist** for a run, and it is a fact about the
    evaluation setup, never a judgement about the model. See
    [The pillars](pillars.md#how-much-of-the-model-do-you-have).

White-box / black-box
:   White-box means the evaluation can reach inside the model — gradients, and sometimes the
    parameters themselves. Black-box means it can only send inputs and read outputs. The
    distinction is not a detail: a white-box attack is *stronger*, so a model that can be
    inspected scores worse on robustness than an opaque one evaluated the same day. Comparisons
    are therefore levelled down to the weakest access present, or the tool would reward opacity.

Coverage
:   The only thing this project aggregates: how many applicable metrics were measured, how many
    came back `insufficient` or `unavailable`, and why. A completeness statement, never a
    quality one — there is no composite score, and there will not be one.

Snapshot
:   One immutable record per run, in `history/`. Carries the evaluation manifest's **content
    hash** and the integrity verdict — the two things that decide whether a later comparison is
    legitimate.

Comparability
:   Two runs may only be compared if they scored the same rows (identical manifest hash) and
    both had a verified split. Otherwise the tool refuses and says why, rather than drawing a
    chart of a difference it cannot support.

Project
:   The problem a set of models is trying to solve — here, *Skin lesion classification*: seven
    classes of dermoscopy image. Every scenario names its project, and the overview groups models
    by it. It is a way of finding things, never a licence to compare them: two models in one
    project are only compared when they were scored on the same images.

Model
:   One trained checkpoint — a specific file of weights — identified by a hash of its contents
    rather than by its name. This repository has 15. `skin_cancer_clean.pt` is one model even
    though it appears in four evaluations, and `focal loss` and `balanced oversampling` are two
    different models even though an earlier version of the gallery filed them under one heading.

Configuration
:   One model, read one way, scored on one set of images: the model plus its decision rule and
    its evaluation set. The same `skin_cancer_clean.pt` weights read with `argmax` catch 104 of
    163 melanomas; read with `melanoma ×50` they catch 154. Two configurations, one model, no
    retraining — which is why the difference belongs to the rule and not the weights.

Investigation
:   A question several runs were made to answer together — the *learning curve* (how much data is
    enough?) or the *external validation* (does it hold up at an unseen clinic?). It is a filter
    on a project's models, not a folder: the ISIC model has configurations in both the internal
    test and the external validation, so it belongs to both.

Active and archived
:   An **active** configuration is kept current: when a metric changes it is evaluated again. An
    **archived** one is the record of an experiment — kept, still readable, never re-run. Archived
    does not mean wrong: its numbers were measured correctly with the metrics of their day; it
    only lacks what was added since. Here, the ISIC model is active on its internal test and on
    Derm7pt; the learning curve, focal loss and the other experiments are archived.

Metric version
:   A number each metric carries, raised whenever what it reports changes. Every report records
    the versions that produced it, so an active report produced by an older version can say it is
    behind — without anyone re-running the twenty-one archived experiments.

Model registry
:   The list of every declared model and its configurations, whether or not any of them has been
    evaluated yet, written from the scenarios into `model_registry.json`. It records what each
    checkpoint *is* — architecture, training data, loss — and deliberately no score of any kind.

---

## Planned — terms the expanded catalogue introduces

Defined ahead of the metrics that will use them (see [The pillars](pillars.md) for which
tier each lands in). Worked examples here are **illustrative**, not measurements from this
project's runs — where a number is real, it says so.

### Calibration

Calibration
:   Do the probabilities mean what they say? Among the cases a model calls 90% likely, about
    90% should actually be that class. A model can rank cases perfectly and still be badly
    calibrated — discrimination and calibration are independent properties, and the second is
    the first to break when the model meets a new population.

Reliability curve
:   Calibration, drawn. Predicted probability on the x-axis, observed frequency on the y-axis.
    The diagonal is perfect. Above it, the model is underconfident; below it, overconfident —
    which in a triage tool means false reassurance.

ECE (expected calibration error)
:   The average gap between predicted and observed, weighted by how many cases fall in each
    bin. **0 is ideal.** If a bin of 100 cases is given 0.9 confidence and only 70 are right,
    that bin contributes a gap of 0.2.

Brier score
:   Mean squared error of the probabilities themselves. **0 is ideal.** It moves with both
    discrimination and calibration at once, which makes it a good summary and a poor diagnosis —
    read it with the reliability curve, not instead of it.

Calibration intercept and slope
:   The two numbers external-validation studies report. Intercept says whether the model is
    systematically over- or under-predicting on the new population (**0 is ideal**); slope says
    whether its confidence is too extreme or too timid (**1 is ideal**). A slope below 1 is the
    classic signature of a model meeting a population it was not trained on.

### Decision quality

ROC-AUC
:   The probability that a randomly chosen positive case is ranked above a randomly chosen
    negative one. **0.5 is guessing, 1.0 is perfect.** It ignores prevalence entirely, which is
    exactly why it flatters a model on a rare class.

PR-AUC (average precision)
:   The same ranking question asked in terms of precision and recall. Unlike ROC-AUC it *does*
    move with prevalence, so on an imbalanced problem it is the honest one. Its baseline is the
    positive rate, not 0.5.

Youden's J
:   Sensitivity + specificity − 1. **0 is guessing.** One number that cannot be gamed by moving
    the operating point, which is why this project reads external validation on it — a model can
    always raise sensitivity by calling the positive class more often, and J does not reward that.

Net benefit (decision curve analysis)
:   Asks whether using the model beats the two trivial strategies — treat everybody, treat
    nobody — across a range of thresholds. The threshold encodes how many false alarms one
    missed case is worth, which is a clinical value judgement, so the metric reports the whole
    **curve** and never picks a point on it.

Selective prediction
:   What happens when the model is allowed to say "I don't know". Accuracy plotted against
    coverage: if accuracy climbs steeply as the least confident cases are set aside, the
    confidence signal is useful for triage. If it stays flat, the model does not know what it
    does not know.

### Fairness definitions

Demographic parity
:   Equal positive rates across groups. Assumes the groups have equal base rates — where they
    genuinely differ, enforcing this makes the model worse for everybody. Rarely the right
    question in medicine, and included so it can be read against the others.

Equalised odds
:   Equal sensitivity **and** equal specificity across groups. Usually the fairness definition
    that matches a clinical goal: the same chance of being caught, whoever you are.

Equal opportunity
:   The weaker half of equalised odds — equal sensitivity only. Appropriate when a missed case
    is the harm that matters and a false alarm is cheap.

Predictive parity
:   Equal PPV across groups: "when it says melanoma, it is right equally often for everyone".

The impossibility result
:   When base rates genuinely differ between groups, equal calibration and equalised odds
    **cannot both hold**. This is arithmetic, not a design failure. It is why the fairness
    pillar reports several definitions side by side and names the tension, rather than
    reporting one number called *fairness*.

Individual fairness
:   Similar cases get similar decisions. Measured as a flip rate: change only a sensitive
    attribute and count how often the answer changes.

### Adversarial robustness

Threat model
:   The assumptions an attack is measured under: white-box (the attacker has the weights) or
    black-box, which norm bounds the perturbation, how large it may be, and how many iterations
    the attacker gets. **An attack success rate without these four is not a measurement**, so
    each adversarial metric carries them in its result.

FGSM · PGD · DeepFool
:   Three standard white-box attacks of increasing strength: one gradient step, many projected
    steps, and a search for the nearest decision boundary. Reported against an additive-noise
    baseline, so the reader can see how much of the damage needed an *adversary* rather than
    just bad luck.

Adversarial versus natural robustness
:   Different questions. Natural asks whether a clinically irrelevant distortion flips the
    call; adversarial asks whether a deliberate one can be constructed. They are never merged
    into one "robustness" number.

### Explanation quality

Insertion and deletion
:   Deletion masks the most-attended evidence and watches confidence fall; insertion starts from
    nothing and adds the most-attended evidence back. Deletion alone is gameable, so both are
    reported.

Complexity (sparseness, entropy)
:   How concentrated an attribution is. An explanation that highlights everything explains
    nothing, and this is the number that says so.

Randomisation check
:   Randomise the model's parameters layer by layer and recompute the explanation. If it barely
    changes, the method is an edge detector reacting to the image, not an explanation of the
    model. The cheapest way to catch explainability theatre, and almost nobody runs it.

Max-sensitivity
:   How much the explanation moves when the input is nudged imperceptibly. A method whose
    heatmap is rewritten by invisible noise is not describing a stable reason.

### Privacy

Population attack
:   The cheap membership attack this project already runs: compare the model's confidence on
    members against non-members, with no extra training.

Shadow attack
:   The stronger membership attack — train models on data drawn from the same distribution to
    learn what "a member" looks like. It needs that distribution, so for a checkpoint whose
    training data is unknown it is not merely unimplemented but **impossible**, and the report
    says so rather than leaving a blank.

Canary
:   A deliberately meaningless, unique sequence planted in the training data *before* training —
    for example a record reading `patient id 481-90-2231`, where the digits are drawn at random
    from a known space. It stands in for the real thing you are afraid of leaking: an identifier,
    a rare diagnosis string, a line of a free-text note.

Canary exposure
:   How strongly the trained model prefers the canary it actually saw over every other sequence
    it *could* have seen from the same random space. Roughly, the number of bits by which the
    true canary outranks the field: if it is the single most likely of a billion candidates,
    exposure is about 30 bits and the string is effectively extractable; if its rank is no better
    than chance, exposure is about 0 and no memorisation has been shown. **0 is ideal.**

    Why it is stronger evidence than a membership attack: you planted it, so you know it is in
    there and you know exactly what it looks like. That turns an average-case guess into a
    calibrated worst-case measurement with a known ground truth. The price is that **you must
    control training** — you cannot plant a canary in a checkpoint someone else trained, which
    is why it sits beside shadow MIA as permanently unavailable for third-party models.

    It suits sequence models best. For an image classifier the analogue — a distinctive planted
    image — is weaker, because there is no ranked vocabulary to measure exposure against.

Attribute inference
:   Whether a sensitive attribute the model was never asked to predict can be recovered from its
    outputs or its internal representations.

### Generative

Benchmark contamination
:   Split leakage, one level up: was the evaluation benchmark inside the training corpus? The
    single most important unasked question behind most published LLM scores, and the direct
    analogue of the audit that started this project.

Groundedness
:   Are the claims in an answer supported by the sources it was given? Distinct from correctness:
    an answer can be true and ungrounded, which is still a failure for a system that is supposed
    to cite.

Hallucination rate
:   Unsupported claims per answer. Counted, not scored — the number means nothing without the
    rubric that defined "unsupported", so the rubric travels with it.

Chain-of-thought faithfulness
:   Does the stated reasoning actually *cause* the answer? Tested by intervening on the
    reasoning and seeing whether the answer follows. Reading the reasoning and finding it
    plausible tests nothing.

Sycophancy
:   Whether the answer changes when the user pushes back without new evidence. In a clinical
    setting, a model that folds is worse than one that was wrong confidently, because the
    disagreement was the signal.

LLM judge
:   A model used to grade another model's output. It is itself a model under evaluation, so any
    judge-based finding records the judge's identity and version, the exact prompt, and its
    agreement with human labels on a sample. Without that agreement number the verdict stays
    `insufficient`.

### Safety

Scope compliance
:   Does the system stay inside the indication it claims, and refrain from asserting authority
    it does not have — the machine-checkable form of "this is not a medical device".

Escalation
:   On a red-flag presentation, does it send the user to a clinician? Measured on vignettes
    written for the purpose.

Refusal appropriateness
:   Measured in **both** directions. Answering what it should refuse is the obvious failure;
    refusing what it should answer is the one that gets ignored, and it has a real cost to real
    users.

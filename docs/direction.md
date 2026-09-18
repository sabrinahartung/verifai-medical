# Direction — what this tool is for

!!! abstract "Design reasoning, not a specification"
    This page holds the *why* behind the plan: what this framework adds that a metric library
    cannot, where the line between measuring and judging sits, and how deep the reporting
    should go. The [roadmap](ROADMAP.md) says what gets built and in what order; this says
    what it is for. [A worked example](case-view.md) shows the destination in pictures.

    Nothing here is settled. Several of these questions are open and the last section says
    which.

---

## The tool is not a metric library

A fair objection, and one worth taking seriously: **executing and displaying metrics is not
new.** Quantus [[27]](references.md#ref-27) runs explanation-quality metrics. ART
[[35]](references.md#ref-35) runs attacks and membership inference. TextAttack
[[24]](references.md#ref-24) perturbs text. If this framework is a nicer front end for those,
it has no reason to exist.

The distinction is one of *layer*. Those libraries are **instruments**. A hospital does not
lack thermometers. What it has that a thermometer does not is a protocol saying which patients
to measure, when, and what counts as evidence.

Four things a single-metric library structurally cannot do, because it has no concept of a
run, a dataset, or a claim:

**It cannot refuse to measure.** Quantus will score a model on a contaminated split without
complaint, because it has never heard of your split. The audit behind this project found 99.5%
of HAM10000 inside the original checkpoint's training data, and no toolkit would have said so.
Refusing to produce a number is not a feature a metric library can have.

**It cannot carry provenance.** `max_sensitivity = 0.34` — of which checkpoint revision, which
attribution method, which preprocessing, which images? A float in a notebook is uninterpretable
six months later and uncheckable by anyone else. The artifact, the manifest content hash and
the snapshot history are what make a number survive contact with time.

**It cannot refuse a comparison.** Two numbers are only comparable if they came from the same
rows, under the same access level, with the same method configuration. Declining to plot a
difference it cannot support is the most unusual behaviour in this repository, and no
instrument does it.

**It cannot tell you what you did not measure.** The coverage map — *how many applicable
metrics were measured, how many came back inconclusive, and why* — has no equivalent in any of
these libraries, because none of them knows what the full set would have been.

> **The libraries answer "what is the number?". This answers "should anyone believe it?"**

That is an evidence-management layer, closer to a lab notebook with a protocol than to a
toolkit. It also sets the design constraint: every feature should be judged on whether it
serves *believability*. A metric this framework merely runs faster than Quantus is not a
feature.

---

## Judging: the line, and who is on each side of it

The instinct that a tool should not judge a model is right but slightly misaimed. Judging is
not the problem. **Unattributable judging is.**

The retired accuracy threshold was not wrong because it was a threshold. It was wrong because
it had no author, no rationale and no stated context of use — and, measured on the real runs,
it marked the configuration catching 159 of 163 melanomas a *warning* and the one missing 82 of
them a *pass*. A criterion carrying a required rationale, a named owner and a version is a
different object entirely: it is a judgement you can argue with, which is what makes it
legitimate.

### The line

> **The tool may judge the measurement. It may never judge the model.**

| The tool decides | The tool declines |
|---|---|
| this split is contaminated | 0.72 balanced accuracy is good |
| n=13 cannot support a claim | a 21-point fairness gap is unacceptable |
| the intervals separate, so the gap is real | this model is safe to deploy |
| this adapter cannot supply gradients | this model is better than that one |

The left column is epistemic — facts about what is known. The right is evaluative — facts about
what matters, which depend on where the model runs and what being wrong costs.

### Who has the right to judge?

Whoever bears the consequence. In a medical setting that is the regulator for market access,
the deploying institution for clinical governance, the clinician for the individual patient,
and increasingly the law. Never the tool.

The tool's obligation is to make that judgement **possible and cheap** — not to pre-empt it,
and not to abdicate so completely that nobody can make it. Refusing to judge does not remove
the judgement; it relocates it onto a reader who is usually less equipped than the person who
built the evaluation. That is a transfer of risk dressed as humility.

### So give a reference, and say what kind it is

*"Flip rate 14.5%"* means nothing to a reader who does not already know what good looks like.
Every finding should carry what the value **should** be, with its epistemic weight visible:

| Kind | Example | Weight |
|---|---|---|
| **control** — measured in this run | attribution faithfulness against a *random-attribution control* under identical conditions | strongest: data, not opinion |
| **chance / ideal** — definitional | membership AUC against 0.5; flip rate against 0% | strong: follows from the statistic |
| **criterion** — from the policy | flip rate ≤ 5% for medical use | weakest: authored, versioned, arguable |

A fourth kind is refused: **external norms**. *"Good dermoscopy models achieve ≤3% flip rate"*
would need a reference corpus that does not exist, and inventing one is worse than having no
baseline.

### The judgement that needs no threshold at all

Comparing a model to **its own previous version** is objective. *"Melanoma sensitivity fell 8
points against the run you shipped, on the same 1,493 images"* requires no authored criterion,
no intended-use profile and no argument. The snapshot history and the frozen manifest already
make this possible, and it dissolves a large part of the judging problem by never raising it.

---

## Configuration: does letting users set things add bias?

Yes, in one predictable direction: people configure until they pass.

But the alternative is worse, because a fixed threshold makes the same choice and hides it
behind an appearance of objectivity. The question is not configurable versus fixed. It is
configurable versus **configurable and accountable**.

> **You may configure what the model is for. You may not configure what was measured.**

| Configurable | Never configurable |
|---|---|
| intended-use profile (medical decision support vs research) | the split-integrity gate |
| the harm ratio a decision curve is read at | interval gating — a criterion decides only when the interval clears it |
| deployment prevalence for PPV | the sample-size floor |
| which classes matter clinically | the evaluation manifest's content hash |

Three rules turn configuration into evidence rather than a loophole:

1. **Every configuration is a versioned artifact with an owner**, not a slider whose state
   vanishes when the page reloads.
2. **The report states which profile ran and diffs it against the default.** You do not prevent
   gaming; you make it leave a trail.
3. **The defaults are argued for in prose**, so departing from them is a visible act rather
   than a quiet one.

---

## What SonarQube does, and what actually transfers

SonarQube is the closest thing in another field to what this could become. An issue there
carries a rule, a location, a *"Where is the issue?"* view with the offending code, secondary
locations, an execution flow from source to sink, a *"Why is this an issue?"* explanation, and
an activity trail. It is specific, actionable and never a score.

**Three mechanisms are worth taking.**

*The rule catalogue.* Sonar's judgements are not a number but a versioned, inspectable set of
rules, each with a rationale and worked examples. That is exactly the policy file in
[Phase G](ROADMAP.md#phase-g-the-findings-layer-measurement-judgement-and-the-line-between-them)
— the harder half is already designed.

*Judge the delta, not the whole.* Sonar's strongest idea is "Clean as You Code": it does not
grade your entire codebase, it grades what changed. Ported here, that means **judging a model
against its previous version rather than against an absolute standard** — objective, requiring
no authored threshold, and buildable today from the snapshot history.

*Always say what to do.* Every Sonar issue ends in a fix. The `remedy` field in the policy
schema is currently the least-developed part of that design and deserves to be the most.

**One mechanism does not transfer.** Sonar can answer *"why is this an issue?"* because its
rules are **decidable** — a null dereference is a null dereference, and the execution flow from
source to sink is a proof. Machine-learning failures are not decidable. Why did the model miss
*this* melanoma? Nobody knows, and any per-case causal story would be precisely the confident,
unprovenanced claim this project exists to catch.

---

## How deep should the report go?

Deep enough to reach individual cases — and no deeper than the evidence supports. The
substitute for *"why did this fail"* is three things that can be defended:

- **what happened** — prediction, truth, confidence, and what each metric observed on this case;
- **what it shares with other failures** — because a single case has no explanation and a
  counted pattern does;
- **what would change it** — which of the configurations scored on these same images gets this
  case right, and what that costs everywhere else.

[A worked example](case-view.md) shows all three as screens.

**The honest limits.** Report size becomes binding — `details["per_example"]` is written by
every metric and, at fifty-one metrics over 1,493 cases, cannot stay a committed JSON file. The
prose cannot be authored per case, so it has to come from rule templates, which means the case
view and the findings layer are the *same feature*. And Derm7pt's licence forbids
redistributing its images [[4]](references.md#ref-4), so the project's only external validation
set is precisely the one that cannot be illustrated this way.

**And a question about purpose.** Sonar's per-issue depth pays off because a developer fixes
issues one at a time. Nobody fixes a model one image at a time — you change the data, the loss,
or the decision rule. So the case view's job is not remediation. It is **conviction**: it turns
"a 21.3-point fairness gap" from a number a reader nods at into twenty-six cases they cannot
unsee. That is a serious purpose, and the feature should be built for it rather than for
Sonar's.

---

## Still open

Written down so they are not mistaken for settled:

- **Does the delta judgement need a floor?** "No worse than last time" rewards a model that was
  bad and stayed bad. Regression gating may need an absolute companion, which reopens the
  threshold question it was meant to avoid.
- **Who authors the default policy?** The current design requires a named owner per criterion.
  For a portfolio project that is one person; for anything real it is a governance question the
  tool cannot answer.
- **How much prose can be generated before it stops being honest?** Claim templates scale;
  hand-written analysis does not. The boundary between "generated from a counted pattern" and
  "plausible-sounding text" is not yet drawn.
- **Does the case view survive its own scaling?** Twenty-six cases is a cluster a person can
  read. Two thousand is a database, and a database with no interface is not conviction.

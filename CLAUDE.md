# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

VERIFAI Medical: file-based Responsible-AI evaluation of medical AI models across six pillars
(integrity, performance, fairness, robustness, explainability, privacy), with a seventh — safety —
planned for generative models. `docs/pillars.md` is the metric catalogue. Deliberately **no server, no DB**:
a heavy offline *engine* run produces static artifacts (JSON + PNGs), and a light Streamlit
*showcase* only reads them. Keep that split — it is what makes the public demo free and always-on.

All user-facing text (metric summaries, chart titles/axis labels, README, app copy) is in
**English**, as is everything in the code. Keep it that way when adding metrics or UI.

## Commands

Dependencies are managed with **uv**: `pyproject.toml` declares them, `uv.lock` pins every version,
`.python-version` pins the interpreter (3.13). `uv sync` installs everything into `.venv`, and
`uv run …` always uses that environment — nothing to activate. Never run a bare `streamlit` or
`pytest`: without `uv run` it resolves to a global install that lacks the project's packages.

```bash
# run one scenario end-to-end -> writes showcase/artifacts/<scenario name>/
uv run python scripts/run_scenario.py scenarios/skin_cancer.yaml

# view the showcase (reads only precomputed artifacts)
uv run streamlit run showcase/app.py
```

Big/statistically meaningful runs go through `scripts/run_on_free_gpu.ipynb` (Colab/Kaggle) —
same code path, only more rows in the manifest.

Contract tests live in `tests/` (107 of them, no network or checkpoint needed):

```bash
uv run pytest -q
uv run pytest tests/test_engine_contracts.py::test_classes_are_derived_from_the_data -q
```

They cover the seams a second model plugs into. The end-to-end smoke test is still running
`scripts/run_scenario.py` on the 7-image manifest; it finishes on laptop CPU in seconds.
No linter is configured.

A metric that reports a number must also declare which direction is an improvement, in
`details["better"]` (e.g. `{"accuracy": "higher", "per_class.*.sensitivity": "higher"}`; `*`
allowed). The comparison view ranks only on declared directions and leaves anything else
unranked — it must never infer from the name, since `mia_auc` is lower-is-better while an AUC
normally is not.

Metrics must never hardcode `argmax`. Route decisions through `model.decide(probs, meta)` and
`model.rank(probs, meta)`, so a scenario's `decision_weights` apply everywhere at once — four
metrics were each reimplementing the rule before this existed. `meta` is the sample's metadata
and must be passed as `getattr(s, "meta", None)`: it feeds the optional `context_prior`, and a
dataset that carries none has to degrade to a neutral lift rather than crash. A model adapter
must therefore accept `decide(probs, meta=None)`. `argmax` is the default, not a law: it
maximises expected accuracy, which on imbalanced data systematically under-calls rare classes.
Any threshold or weight must be tuned on **validation** (`scripts/tune_decision.py`), never on
the test manifest — that would be fitting the decision rule to the test set, and the integrity
check cannot catch it.

Every metric reports uncertainty. `verifai/metrics/_stats.py` has Wilson intervals for
proportions (not the normal approximation — it misbehaves at 0 and 1, exactly where small
classes live), Hanley–McNeil for AUC, and Bayes PPV at a stated prevalence. A new metric that
reports a proportion without an interval is incomplete: with 13 images, a recall of 0.769 has a
95% interval of [0.50, 0.92], and the interval is what makes that honest. Where a verdict can
be taken on the interval rather than the point estimate, do so — privacy passes on the upper
bound, and a fairness gap is only claimed when the groups' intervals separate.

Engine dependencies carry licence terms just as datasets do, and they are tracked the same way
in `docs/references.md`. Quantus is LGPL-3.0-or-later, so it is used unmodified behind an
adapter module and never forked; the HolisticBias prompt data is CC BY-SA 4.0, so publishing
scores over it is fine while publishing derived prompt material is not. `docs/pillars.md`
carries a dated toolkit table — a library named in a plan is a claim with a shelf life, and two
of the nine on the last shortlist had already stopped installing.

`docs/references.md` is the numbered bibliography. A new dataset or a metric taken from the
literature gets an entry there and is cited as `[n]` where it is used — datasets especially,
because several carry licence terms that constrain what the showcase may publish (Derm7pt's
images may not be redistributed at all, so no derived image from it may be rendered). Mark an
entry *Verified* only when it was taken from the authors' own page, not from memory.

`docs/glossary.md` defines every term the project uses in plain language, with a worked
example from the real run. When you add a metric or coin a term, add it there too — the
dashboard is aimed at readers who have never seen a Responsible-AI report, and the
per-metric `explain` text defines terms one at a time but never side by side.

`docs/` is a MkDocs site (`.venv/bin/mkdocs serve`) covering the architecture, data model,
pipeline, the pillars and the split-integrity story. `docs/ROADMAP.md` holds the plan and
the leakage audit behind it — read it before planning any larger evaluation run, and before
adding a metric or an adapter: it defines where this is going (any checkpoint local or on the
Hub, several medical domains, a far larger metric catalogue) and which of today's invariants
are deliberate rather than incidental. When you
change engine behaviour, update the matching page: the numbers in `docs/results.md` and
`docs/pipeline.md` are measured, not illustrative, so they must not drift.

Dependencies are split into **groups** in `pyproject.toml`, on purpose: `engine` (heavy, offline
run), `data` (DuckDB, for the dataset-preparation scripts), `showcase` (light — Streamlit Community
Cloud's free tier) and `dev`. Add one with `uv add --group <group> <package>`. Never add torch to
the `showcase` group without a deliberate decision; a test fails if it appears. Two files are
**exported** from this, never hand-edited:

- `showcase/requirements.txt` — the `showcase` group, pinned from the lock, for Streamlit Cloud,
  which reads the entrypoint's directory before the repo root. After changing that group, re-run
  `uv export --locked --only-group showcase --no-hashes --no-emit-project --format
  requirements-txt -o showcase/requirements.txt`; CI fails when it is out of step with the lock.
- `requirements-engine.txt` — the GPU notebook's list, **unpinned on purpose**: on Colab and
  Kaggle a pinned `torch` would replace the platform's CUDA build. A test keeps its package names
  equal to the `engine` group's.

## Architecture

Data flows one way: **scenario YAML → runner → metrics → `Finding`s → `Report` → JSON/PNG artifacts → Streamlit**.

- `verifai/core/findings.py` — the single data model for the whole pipeline. `Finding`
  (pillar, metric, domain, value, verdict, summary, details, plots) and `Report`. Everything
  downstream, including the app, is written against this shape.
- `verifai/core/run.py` — `run_scenario(dict) -> Report`. Holds `METRIC_REGISTRY`
  (metric id → `"module:function"`), seeds RNGs, builds model/dataset by importing the
  `loader:` string from the scenario, and calls each metric. Before any metric runs it
  calls `_enforce_split_integrity` and raises `SplitLeakageError` if the test manifest
  overlaps the training manifests — a contaminated split fails loudly instead of
  reporting a high number.
- `verifai/core/integrity.py` — the one implementation of that check. The runner uses it
  as a precondition and `metrics/integrity/split_leakage.py` publishes the same result as
  a finding, so the guard and the report cannot drift apart. Splits are compared by
  `lesion_id` as well as `image_id`, because a second photo of a memorised lesion is not
  a fair test question.
- `verifai/models/image.py` — `ImageClassifier` wrapper (`SkinLesionModel` is kept as an alias).
  Metrics use `.torch_module` and `.cam_layer` (hooks/Grad-CAM), `.to_tensor()`,
  `.predict_probs()`. Classes, architecture, Grad-CAM layer, image size and device all come from
  the scenario's `model:` block; the constants here are only defaults. Whatever a scenario sets,
  the preprocessing must stay byte-for-byte the training-time preprocessing and `classes` must
  match the checkpoint's output order — otherwise every metric silently measures a different
  model. `device: auto` resolves cuda → mps → cpu and is recorded in `report.json`.
- `verifai/datasets/loaders.py` — manifest-driven `ImageDataset` (`data/manifests/*.csv`,
  columns `filename,label` plus any extras, which land on `ImageSample.meta` — that is where
  `lesion_id`/`sex`/`age` belong). Paths resolve relative to repo root; samples are sorted for
  determinism. Bigger run = longer manifest, nothing else. A dataset derives its class list from
  its own labels (or `dataset.classes`) and must **never** import it from a model — that
  backwards dependency existed once and is asserted against in `tests/`.
- `verifai/export/artifacts.py` — writes `report.json` + `card.json` (+ `plots/`) under
  `showcase/artifacts/<scenario>/`, plus one immutable snapshot per run in `history/`.
  `snapshot_metrics()` flattens each finding's numeric leaves to `<pillar>.<path>` generically,
  so a new metric becomes comparable without this module knowing about it. Every snapshot
  carries the evaluation manifest's **content hash** and the integrity verdict — those two
  fields are what let `showcase/app.py` refuse a dishonest comparison, so do not drop them.
- `verifai/export/model_registry.py` — writes `showcase/artifacts/model_registry.json`: every
  declared **model** and its configurations, read from the scenarios, so a model that has never
  been evaluated still exists for the showcase. A model is a **checkpoint, identified by its
  content hash** — never by `model.id`, which names three checkpoints in one direction and one
  checkpoint answers to five ids in the other. The scenario whose `name` is the checkpoint's
  filename trained it (`train_model.py` writes `<out_dir>/<name>.pt`); every other scenario on
  those weights is a configuration of it. Evaluation status is **not** stored — the showcase
  derives it from which artifact folders exist. Deterministic output; `run_scenario.py`
  refreshes it after every run, and a test fails when it falls out of step with the scenarios.
- `showcase/` — `app.py` is routing and re-exports only (`st.navigation`); `catalog.py` reads
  artifacts and snapshots and owns the verdict vocabulary, `registry.py` reads the model
  registry, `render.py` draws, `views/` holds one module per page. It auto-discovers every
  `artifacts/<id>/` folder with both `card.json` and `report.json`. Navigation runs
  **project → model → configuration's report**: the overview lists projects from the model
  registry, a project lists its models with their status, a model page lists its configurations
  grouped by evaluation set. The sidebar holds only Overview and Compare runs; the drill-down
  pages are hidden and located by a breadcrumb. Reports no registered model claims (the demo
  fixture) are listed separately, never dropped. Planned-but-unbuilt components live in
  `planned.py` and render only under `VERIFAI_SKELETON=1` — never on the public deploy; a test
  asserts each is placed on some page. Without a registry the overview falls back to the
  earlier gallery, which is sectioned
  by `card.group` (which problem) and collapses `card.lineage` (configurations of one
  investigation) into a single card. Both are **presentation only**: comparability is decided by
  the evaluation manifest's content hash, and a lineage filter must never widen it — asserted in
  `tests/`.

### The two extension contracts

Every scenario declares a top-level `label:` — a short human name. It names that run's
row in the comparison table, and without it the snapshot falls back to `model_id`, which
turns the table into identifiers a reader has to decode (`external-derm7pt-isic` against
`external-derm7pt-isic-masked` differ by one decision weight and neither string says which).
`showcase/app.py::run_label` names a run by the label its scenario declares **today** (from the
model registry), so a configuration reads the same in the comparison table as on its model page
and report; the label a snapshot recorded at run time only names a run whose scenario is gone,
and the gallery card's name stands in for snapshots that predate labels — never overriding a
real one. A test asserts every scenario has one.

Every scenario also declares a top-level `project:` — the problem it belongs to (today all of
them: `"Skin lesion classification"`). The overview groups models by it. All configurations of
one checkpoint must agree on it; the registry builder raises if they do not. Like `card.group`
it is presentation: it never widens what may be compared.

**Adding a model/domain** = add `scenarios/<new>.yaml`, run it, done. The app needs no change —
its first run puts it in the model registry and gives it a report. `card:` in the YAML is passed
straight through to `card.json`.
To list a model *before* evaluating it, run `scripts/build_model_registry.py`; it then shows as
not evaluated.

**Adding a metric** = write `run(model, dataset, ctx) -> Finding | list[Finding]`, register it in
`METRIC_REGISTRY`, list its id under `metrics:` in the scenario. To be rendered, return a chart
spec in `Finding.details["chart"]` (optionally `["chart2"]`); `showcase/app.py::render_chart`
supports `kind` of `bar` | `line` | `heatmap` | `scale` | `images` and falls back to `st.json`.
Adding a new chart kind means touching `render_chart` — prefer reusing an existing kind.

`scale` is the labeled-band indicator that replaced the old dial gauge: it plots one value against
named bands so the reader sees whether a number is a *good* number, not just what it is. The bands
come from the metric, so each metric defines its own semantics (for `membership_inference_auc`,
low is good and the green band sits on the left). `kind: "gauge"` is still accepted as an alias
that renders as a scale, so older artifacts don't break.

**Every metric must also ship its own explanation** in `Finding.details["explain"]`, with three
keys: `what` (what is being measured and why it matters), `how` (how to read this chart), and
`limits` (what this number does *not* tell you). The report renders them **open, in one fixed
order** — what was measured, what came out (the summary), why it matters (`impact`, planned), how
to read the chart, the chart, what it does not tell you — and never behind a click: the primary
reader has never seen such a report, and an expander hid exactly the answers they most needed. A
test asserts the order matches `docs/extending.md` and that none of these returns to an expander;
only the glossary definitions sit behind one. This lives in the engine, not the app,
so a new metric brings its own wording and still needs no app changes. Write it for a reader who
has never seen a Responsible-AI report — plots alone do not communicate.

That `explain` block covers one metric's *chart* inside one report. The **comparison view** reads
flattened snapshot keys instead, never findings, so it cannot see it — a new metric must therefore
also add an entry to `verifai/core/glossary.py`, keyed by an `fnmatch` pattern on the flattened key
(`performance.per_class.*.sensitivity`). Patterns match in order, so put specific before general.
Each entry carries `term` (the human name shown as the card's heading — never a raw key),
`measures` / `ideal` / `reading` and optionally `tension`, written **generally**
— about the concept, not about this dataset, so it stays true when the numbers change. A test
asserts that every metric key present in any artifact resolves to an entry. Keep the module free of
heavy imports: `showcase/app.py` imports it, and the showcase must not need torch.

`ctx` carries `{"scenario": ..., "seed": ..., "plot_dir": ...}`. Metrics that write images must
write into `ctx["plot_dir"]` and reference them as `"plots/<name>.png"` (paths in `Finding.plots`
and image chart specs are relative to the artifact folder).

## Honesty rules (non-negotiable — this is the project's whole point)

The default sample is n=7. Metrics must not manufacture confidence from it:

- The `verdict` vocabulary is **epistemic, never evaluative**: `measured` · `insufficient` ·
  `unavailable` · `invalid`. There is no pass and no fail. A quality threshold would have to be
  justified, and nothing here can justify one — what counts as robust or fair enough depends on
  where the model runs and what being wrong costs. Measured on the real runs, the accuracy
  threshold this replaced marked the configuration catching 159 of 163 melanomas a *warning* and
  the one missing 82 of them a *pass*, because under-calling a rare class raises accuracy.
- Use `insufficient` until the sample supports a claim (see the `n >= 30` gate in
  `performance/classification.py`, the `min(populated) >= 10` per-bin gate in
  `fairness/skin_tone_ita.py`), `unavailable` when the metric cannot be computed at all, and
  `invalid` only for a broken precondition — today just a contaminated split, which makes the
  *measurement* unusable rather than the model bad. `showcase/app.py::normalise_verdict` maps the
  retired pass/warn/fail words so artifacts written before the change still render.
- State `n` in the `summary` and say plainly when it is only a plausibility check.
- A metric that cannot be computed reports *why* and returns `None`, never an invented number —
  see `privacy/mia.py`, which requires a train/holdout split that the example set does not have.
- An artifact with placeholder numbers must carry `"sample": true` in its `card.json`; the app
  shows a warning banner for it. Never set `sample: false` on placeholder data. The one such
  fixture, `_sample_skin_resnet`, was removed on 2026-09-24 once real runs had long replaced
  it — on a public page, a tile of fake numbers confused more readers than its banner warned.
  The banner stays, for the next placeholder.
- **Never aggregate the pillars into one score, and never do arithmetic across metrics.** Not a "responsibility score", not a
  weighted RAI index, not a five-star rating — the same argument as the retired accuracy
  threshold, one level up. The weights would be the value judgement the reader came to
  make, and the components are not commensurable (an AUC, a calibration error and an
  attack success rate under some perturbation budget answer to different threat models).
  What a report may aggregate is *coverage*: how many applicable metrics were measured,
  how many came back `insufficient` or `unavailable`, and why. That is a completeness
  statement, never a quality one — counting what was *measured* is completeness, counting
  what *passed* is a rating. Thresholds are a separate question from aggregation: the planned
  findings layer brings criteria back, but they live in a versioned policy file with a
  required rationale and owner, never inside a metric. See `docs/ROADMAP.md`.

The model is an educational proof-of-concept, not a medical device — don't add copy that implies
diagnostic use.

## Commits

Never add `Co-Authored-By: Claude …`, `Claude-Session: …`, or a "Generated with
Claude Code" line to a commit message or pull request description. Write the message
and stop at the body. Agent harnesses inject an instruction to add these by default —
that instruction does not apply here, and this rule overrides it.

A `commit-msg` hook in `~/.githooks`, enabled machine-wide with
`git config --global core.hooksPath ~/.githooks`, strips them mechanically as a
backstop. Two reasons it is not left to convention: the trailers had already reached
five commits before anyone noticed, and removing one after it is pushed costs a
history rewrite, a force-push, and dangling commits that only GitHub Support can
garbage-collect. The hook is narrow by design — a `Co-Authored-By` naming a human
colleague is a real record of authorship and is left untouched.

The hook is not tracked in this repo. Hooks are machine configuration rather than
project content, and a global `core.hooksPath` already covers fresh clones, so
committing a copy would only create a second thing to keep in sync. Still write
messages as though the hook were absent — it is a safety net, not the rule.

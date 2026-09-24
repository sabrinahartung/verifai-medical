# Data model

One shape flows end to end: a metric returns a `Finding`, the runner collects them into a
`Report`, the exporter serialises it, and the app renders it. Nothing translates between
formats along the way.

```mermaid
classDiagram
    class Report {
        +str scenario
        +Domain domain
        +str model_id
        +str dataset_id
        +list~Finding~ findings
        +dict meta
        +str created_at
        +add(finding)
        +to_dict() dict
    }
    class Finding {
        +Pillar pillar
        +str subaspect
        +str metric
        +Domain domain
        +Any value
        +Verdict verdict
        +str summary
        +dict details
        +list~str~ plots
    }
    class Details {
        +dict explain
        +dict chart
        +dict chart2
        +Any ...metric specific
    }
    class Explain {
        +str what
        +str how
        +str limits
    }
    class ChartSpec {
        +str kind
        +str title
        +... kind specific
    }
    Report "1" *-- "many" Finding
    Finding "1" *-- "1" Details
    Details "1" o-- "0..1" Explain
    Details "1" o-- "0..2" ChartSpec
```

## Vocabularies

The literals below are what the engine writes today. Where an addition is planned it says so
inline — nothing here is aspirational unless it is labelled.


`Pillar`
: `integrity` · `performance` · `fairness` · `robustness` · `explainability` · `privacy`

    `safety` joins them with the first generative metric — see
    [The pillars](pillars.md#safety-what-happens-if-someone-acts-on-this). It is not in the
    literal yet on purpose: adding it before anything can fill it would put a permanently
    empty column on every published dashboard, which teaches readers to ignore a pillar
    before it has ever said anything.

`Finding.subaspect`
: Optional, and the grouping level between pillar and metric — `calibration` under
    performance, `adversarial` under robustness, `randomisation` under explainability. At six
    metrics a flat list per pillar was fine; at the [catalogue's](pillars.md) fifty-one it is a
    wall, so the dashboard groups pillar → sub-aspect → metric.

`Verdict`
: `measured` · `insufficient` · `unavailable` · `invalid`

`Domain`
: `image` · `text` · `tabular` · `llm`

!!! tip "There is no `pass`, and that is deliberate"
    The vocabulary says what is *known* about a number, never whether it is good. A pass
    would need a threshold, and a threshold would need justifying: what counts as robust or
    fair enough depends on where the model runs and what being wrong costs, which is the
    reader's decision and not a constant in a metric module.

    Measured on the real runs, the accuracy threshold this replaced did worse than nothing.
    It marked the configuration catching 159 of 163 melanomas a **warning** and the one
    missing 82 of them a **pass**, because under-calling a rare class raises overall
    accuracy. A reader trusting the badges would have picked the worst detector in the set.

    `invalid` is the single hard signal, and it judges the *measurement* rather than the
    model: a contaminated split does not measure generalisation at all, so nothing computed
    on it means what it appears to.

`Report.meta` records `seed`, `sample_size` (what was actually evaluated, not what the YAML
declared) and `device` — the last because results are bit-identical *per device*, not across
devices. It will also record the **access level** the run had over the model
(`labels` … `training_data`), because that decides which metrics could exist at all and is what
lets the comparison view level two runs down to the weakest access they share rather than
reading a missing row as a worse model.

## The `explain` contract

Every metric ships its own explanatory text inside `details["explain"]`. This lives in the
engine, not the app, so a new metric brings its own wording and the app needs no change.

| Key | Answers | Rendered |
|---|---|---|
| `what` | What is measured, and why it matters | inline, always visible |
| `how` | How to read this particular chart | in the "How to read this chart" expander |
| `limits` | What this number does **not** tell you | same expander, under "What it does *not* tell you" |
| `impact` | Who is affected by the number being what it is, in this clinical context | same expander |

`impact` is what replaces a composite score. A reader who cannot be handed "7.4 out of 10"
still needs to know what a 21-point subgroup gap *means* for the people on the wrong side of
it, and that sentence belongs with the metric, in the engine, rather than in a dashboard the
metric knows nothing about.

`limits` is the one that earns its place. It is where each metric names the trap it sets:

> **Robustness** — "Stability is not correctness: a model that is confidently wrong both
> before and after a distortion scores a perfect 1.0 here."

> **Fairness** — "Coverage is not performance — this chart shows who is in the sample, not
> how well the model serves them."

> **Explainability** — "A convincing heatmap is not proof of medically correct reasoning —
> it shows where the model looked, not whether it looked for the right reason."

## Chart specifications

A metric describes a chart as data; `showcase/app.py::render_chart` draws it. Unknown kinds
fall back to `st.json`, so a malformed spec degrades rather than crashes.

```mermaid
flowchart LR
    F["Finding.details"]
    F --> C1["chart"]
    F --> C2["chart2 (optional)"]
    C1 & C2 --> R{{"render_chart(spec)"}}
    R -->|bar| B["grouped / coloured bars"]
    R -->|line| L["line + markers"]
    R -->|heatmap| H["confusion matrix"]
    R -->|scale| S["value on labelled bands"]
    R -->|images| I["PNG grid with captions"]
    R -->|unknown| J["st.json fallback"]
    style R fill:#EDE9FB,stroke:#5B3FD6,color:#1a1a2e
```

| `kind` | Required keys | Used by |
|---|---|---|
| `bar` | `x`, `y` (+ `color`/`colors`, `hover`) | performance, robustness, fairness |
| `line` | `x`, `y` | — |
| `heatmap` | `z` (+ `x`, `y`, `text`, `zmin`, `zmax`) | performance at n > 50 |
| `scale` | `value`, `min`, `max`, `bands` | integrity, explainability, privacy |
| `images` | `paths` (+ `captions`) | explainability |

### Why `scale` replaced the dial gauge

A dial shows a number. A `scale` shows whether it is a *good* number, because the bands are
named and supplied by the metric — so each metric defines its own semantics. For deletion
faithfulness, higher is better; for membership-inference AUC, the green band sits on the left.

```mermaid
flowchart LR
    subgraph faith["explainability: higher is better"]
        direction LR
        f1["0 — decorative"] --> f2["0.2 — partly faithful"] --> f3["0.5 — faithful → 1.0"]
    end
    subgraph mia["privacy: lower is better"]
        direction LR
        m1["0.5 — low risk"] --> m2["0.6 — moderate"] --> m3["0.75 — high risk → 1.0"]
    end
    style f3 fill:#CDE8D5,color:#1a1a2e
    style f1 fill:#F5D3CE,color:#1a1a2e
    style m1 fill:#CDE8D5,color:#1a1a2e
    style m3 fill:#F5D3CE,color:#1a1a2e
```

`kind: "gauge"` is still accepted as an alias that renders as a scale, so artifacts written
before the change keep rendering.

## Comparing runs

Every evaluation writes a snapshot to `history/`. The comparison view groups them by the
evaluation set's content hash and separates three questions:

```mermaid
flowchart TB
    Q1{"same rows, and<br/>both splits verified?"}
    Q1 -->|no| R1["✋ not compared —<br/>reason shown"]
    Q1 -->|yes| Q2{"is one run better or equal<br/>on EVERY chosen metric?"}
    Q2 -->|yes| R2["that run dominates —<br/>drop the others, no judgement needed"]
    Q2 -->|no| R3["a genuine trade-off —<br/>choose by <b>intended use</b>,<br/>which the data cannot settle"]
    style R1 fill:#F5D3CE,stroke:#C0392B,color:#1a1a2e
    style R2 fill:#CDE8D5,stroke:#2E9E5B,color:#1a1a2e
    style R3 fill:#FFF6E0,stroke:#C77700,color:#1a1a2e
```

Ranking needs to know which direction is an improvement, and **the metric declares that**, in
`details["better"]` (patterns may use `*`):

```python
"better": {"accuracy": "higher", "per_class.*.sensitivity": "higher"}   # performance
"better": {"mia_auc": "lower"}                                          # privacy
```

The app must not infer it from the name: an AUC is normally higher-is-better, but for
membership inference 0.5 is the good end. Anything undeclared is displayed and left unranked —
showing a value is honest, calling it "best" without knowing which way is good is not.

!!! note "What this shows on the current runs"
    Across the three configurations, the baseline leads on six of eight metrics — and loses
    the one that matters most clinically, melanoma sensitivity. **No run dominates any other.**
    A weighted average would have crowned the baseline and shipped a model that misses a third
    of melanomas. That is the argument against aggregate scoring, produced by the tool itself.

## Artifact layout

```mermaid
flowchart TB
    A["showcase/artifacts/"]
    A --> R["model_registry.json<br/><i>every declared model</i>"]
    A --> C["skin_cancer/<br/><i>7 examples, integrity unverified</i>"]
    A --> D["skin_cancer_clean/<br/><i>1,493 held-out images</i>"]
    D --> D1["card.json<br/><i>tile metadata</i>"]
    D --> D2["report.json<br/><i>the Findings</i>"]
    D --> D3["plots/<br/><i>Grad-CAM overlays</i>"]
    style D fill:#E3F2E7,stroke:#2E9E5B,color:#1a1a2e
```

`model_registry.json` is the one file at the top level. It lists every **model** the scenarios
declare — a checkpoint, identified by its content hash — with its provenance from the trainer's
record and its configurations, whether or not any of them has run. It stores no evaluation
status and no score: whether a configuration is evaluated is decided by whether its folder
below exists. Written by `verifai/export/model_registry.py`. Each configuration carries its
`status` (`active` or `archived`), each model whether any configuration is active, and the file
carries the current version of every metric.

Every `report.json` records, in `meta`, the `metric_versions` and the `checkpoint` (path and
content hash) it was produced with. Set against the registry, that is how the app tells a current
report from one a metric change or a retrain has overtaken. Reports from before 2026-09-24 record
neither, and say so.

The app lists any directory containing **both** `card.json` and `report.json`. `card.json`
carries the tile metadata (`name`, `emoji`, `domain`, `dataset`, `description`, `hf_url`,
`sample`) and comes straight from the scenario's `card:` block.

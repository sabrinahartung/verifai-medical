# VERIFAI Showcase — Responsible-AI Evaluation

[![tests & docs](https://github.com/sabrinahartung/verifai-medical/actions/workflows/ci.yml/badge.svg)](https://github.com/sabrinahartung/verifai-medical/actions/workflows/ci.yml)

**Systematic, reproducible evaluation of medical AI models along the Responsible-AI pillars** —
integrity, performance, fairness, robustness, explainability, privacy — with every number
carrying the sample size and interval that produced it, and none of them scored against a
threshold.

Today this evaluates image classifiers, in dermatology. The engine seams are built so a model
arrives as a scenario file rather than a code change; making that true for *any* checkpoint
(local or Hugging Face) and for text, speech and generated text is the plan in
[`docs/ROADMAP.md`](docs/ROADMAP.md).

> This version turns the larger VERIFAI framework into a **file-based, reproducible, free-to-run**
> form: the (potentially heavy) evaluation runs *once* — locally on CPU for small samples, or on a
> free GPU for large ones — and produces static **artifacts** (JSON + plots). A small Streamlit app
> shows them interactively: **tiles → click → dashboard**. No server, no database, no running costs.

**Full documentation:** `mkdocs serve` (or `docs/`) — architecture, the data model, the
pipeline, the full metric catalogue across the pillars, and the split-integrity story.

---

## Quickstart

**Engine — produce a real run** (in an environment with `torch` — e.g. your
`ML_Training_Dojo/.venv`, where everything is already installed):

```bash
pip install -r requirements-engine.txt          # only if torch is missing
python scripts/run_scenario.py scenarios/skin_cancer.yaml
```

This uses the **7 real, labeled HAM10000 example images** in `data/examples/` and your
**real ResNet18 from Hugging Face**, and writes `showcase/artifacts/skin_cancer/` (report.json,
card.json, plots/ with Grad-CAM overlays). Runs on a laptop in seconds.

> Tip: point `weights_path:` in `scenarios/skin_cancer.yaml` at your local `.pt` and even the
> Hugging Face download goes away.

**Showcase — take a look:**

```bash
pip install -r showcase/requirements.txt
streamlit run showcase/app.py
```

**Big run without GPU worries:** `scripts/run_on_free_gpu.ipynb` (Colab/Kaggle) — exactly so your
Mac does **not** have to compute the full subset.

---

## Architecture at a glance

```
verifai/            ← ENGINE (offline: local / Kaggle / Colab)
  core/             findings data model + runner + metric registry
  models/           domain adapters (image: ResNet18 from Hugging Face)
  datasets/         small, pinned subsets via manifests (reproducible)
  metrics/          the pillars: performance / fairness / robustness / explainability / privacy
  export/           findings -> static artifacts (JSON + plots)

data/
  examples/         7 real, labeled HAM10000 images (MVP sample)
  manifests/        which images + labels (CSV, versioned)

scenarios/          one run = one YAML (e.g. skin_cancer.yaml)
scripts/            run_scenario.py (CLI) + run_on_free_gpu.ipynb

showcase/           SHOP WINDOW (deploys for free on Streamlit Community Cloud)
  app.py            tile gallery -> click a model -> Plotly dashboard
  artifacts/<id>/   one folder = one tile (card.json + report.json + plots/)
  requirements.txt  deliberately LIGHT (free-tier friendly)
```

## What the `skin_cancer` run measures (all really computed)

| Pillar | Metric | What it does |
|---|---|---|
| Performance | `top1_accuracy` | Top-1 hits + confidence per example (green=correct / red=wrong) |
| Explainability | `gradcam_faithfulness` | Grad-CAM overlays (ported from your `streamlit_app.py`) + deletion faithfulness |
| Robustness | `corruption_stability` | Does the prediction stay stable under noise/blur/brightness/JPEG? |
| Fairness | `skin_tone_ita` | Skin-type coverage via ITA (label-free); subgroup gap on the big run |
| Privacy | `membership_inference_auc` | **honest:** needs a train/holdout split → in the GPU run, no invented number |

**Honesty about the sample:** n=7 is a *plausibility check*, not a benchmark. Every metric states
its `n` and stays at `insufficient` until the sample supports a claim. The status vocabulary is
deliberately **epistemic, never evaluative** — `measured` · `insufficient` · `unavailable` ·
`invalid` — because what counts as accurate, fair or robust *enough* depends on where the model
runs and what being wrong costs, which is the reader's call and not a constant in a metric module.
There is no pass, no fail, and no composite score. For solid numbers, run the larger subset —
**same code path**, just more rows in the manifest.

## Extensible: adding a new model / a new domain

1. Create a scenario (`scenarios/<new>.yaml`) with model + dataset + metrics.
2. `python scripts/run_scenario.py scenarios/<new>.yaml` → produces `showcase/artifacts/<new>/`.
3. Done — the next time you open the app, **a new tile** appears automatically. No app code changes.

A new metric? It returns a small chart specification in `Finding.details["chart"]` (optionally
`["chart2"]`) — `{"kind": "bar"|"line"|"heatmap"|"scale"|"images", ...}` — and the app renders it
generically with Plotly. The metric signature is the same everywhere:
`run(model, dataset, ctx) -> Finding`.

**No MongoDB, no forced Docker, no backend server.** Results are files.

---

## Transparency (important)

The public Streamlit demo shows **precomputed** results so it stays free and always reachable.
That is a deliberate design decision, not obfuscation:

- **Full code:** this repo — including every metric.
- **Model:** public on Hugging Face ([`sabrinahartung1010/skin-lesion-resnet18`](https://huggingface.co/sabrinahartung1010/skin-lesion-resnet18)).
- **Data:** the real example images are in the repo (`data/examples/`), labels in the manifest.
- **Reproduce it yourself:** `scenarios/*.yaml` + `run_scenario.py` — every result is recomputable
  (the GPU notebook is included).
- **Real run on video:** see the portfolio.

> Every artifact in `showcase/artifacts/` is a real evaluation. An artifact with placeholder
> numbers would carry `"sample": true` in its `card.json` and be shown with a warning banner;
> the one such fixture was removed once the real runs had replaced it.

---

## Status

- [x] Engine + findings data model + runner + registry
- [x] Image metrics implemented across **all pillars** (performance, fairness, robustness, explainability; privacy honestly marked as "needs the full run")
- [x] Streamlit showcase: tile gallery → Plotly dashboard, auto-extensible
- [x] Reproducible example sample (7 real HAM10000 images + manifest)
- [x] **First real run** executed (`run_scenario.py`) → replace the SAMPLE tile with the real one (fixture removed 2026-09-24)
- [x] Larger subset on a free GPU (solid fairness/privacy numbers)
- [x] Deploy to Streamlit Community Cloud + short video
- [x] Lesion-grouped split, leakage guard, uncertainty intervals, cost-sensitive decisions
- [x] External validation on a second archive (Derm7pt) — the internal ranking inverts
- [ ] Adapter contract + capability gating, so a metric skips with a reason instead of assuming pixels
- [ ] Provenance & label-space preflight, for models whose training data we cannot inspect
- [ ] Hugging Face model resolution (`hf:owner/repo`) and the `transformers` adapters
- [ ] A far larger metric catalogue: calibration, subgroup fairness, adversarial robustness, XAI evaluation
- [ ] A second domain (chest X-ray), then text
- [ ] Generative AI: contamination, groundedness, extraction — see the roadmap

## Data / license

The example images come from **HAM10000** (Tschandl et al., 2018; CC BY-NC 4.0) and serve
demonstration purposes only. The model is an **educational proof of concept — not a medical
device, not for diagnostic use.**

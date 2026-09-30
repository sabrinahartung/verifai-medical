# Model cards

A model card is the document that travels with a published model and says what the weights
alone cannot: what the model is, what it was trained on, how to feed it, what it is for and
what it is not for. The idea comes from Mitchell et al. [[49]](references.md#ref-49). On the
Hugging Face Hub a model card is the repository's `README.md`, with a YAML header that machines
read and a body that people read.

This page matters here for two reasons. The resolver (step C2 of the
[roadmap](ROADMAP.md#phase-c-resolving-a-model-and-the-adapter-catalogue)) reads a card to draft
a scenario, so a card decides how much of an evaluation can start from the checkpoint rather
than from a person. And the project's own Hub checkpoint turned out to have no card at all.

## What the resolver reads

| Card field | What it becomes in the draft scenario |
|---|---|
| `datasets:` | `model.trained_on`, but only for a dataset id that `data/corpora.yaml` lists under `hub_aliases:`. An unlisted id stays a TODO, because an archive nobody has checked is *unknown*, never independent |
| `license:` | `model.licence`. With none, the draft carries a TODO: whether scores may be published has to be answered before anything runs |
| the body | nothing. Prose is for people; the resolver never parses it |

The YAML header is the part a machine can rely on, so it should be complete even when the body
says the same thing in more words. Everything else the resolver needs (the class order, the
preprocessing) comes from `config.json` and `preprocessor_config.json`, which only a
`transformers` model ships. For a PyTorch `state_dict`, the card's body is the only place those
facts can live.

!!! warning "Do not add a `config.json` to a `state_dict` repository"
    The resolver takes a `config.json` to mean a self-describing `transformers` model, and drafts
    it for the `hf_image` adapter. Written by hand beside a torchvision `state_dict`, it would
    send the draft to the wrong loader. Put the class order and preprocessing in the card body.

## What a good card contains

The Hugging Face template and Mitchell et al. agree on the same core. For a medical image
classifier, each section has one question it must answer:

| Section | The question it answers |
|---|---|
| **Model details** | What architecture, what file format, who made it, under which licence? |
| **Output classes** | Which output index is which class? A `state_dict` does not know |
| **Preprocessing** | Resize, crop, scaling and normalisation, exactly as at training time. A different one gives a different model |
| **How to load it** | A few lines of code that reproduce the model from the file |
| **Training data** | Which dataset, which split, and any known problem with it |
| **Training procedure** | Epochs, optimiser, loss, augmentation, frozen layers |
| **Evaluation** | Where trustworthy numbers are, and why any others are not |
| **Intended use and limits** | What it is for, what it is explicitly not for, known weaknesses, the population it was trained on |
| **Licence** | What may be done with the weights, and why that licence |

Two rules from the rest of this project apply to cards too. Do not copy numbers that will
change into the card; link to where they are measured. And do not write anything that implies
diagnostic use: *not a medical device* belongs at the top, not in a footnote.

## The original checkpoint's case

Resolving [`sabrinahartung1010/skin-lesion-resnet18`](https://huggingface.co/sabrinahartung1010/skin-lesion-resnet18)
on 2026-09-30 found `.gitattributes` and one `.pt` file at commit `f96683d`, and no `README.md`.
Until then the scenarios on this checkpoint, and several pages here, said its training data was
inferred *from the model card*. There was never a card. The real source is the training
notebook [[50]](references.md#ref-50), which calls `load_dataset("marmal88/skin_cancer")` and
trains on its `train` split [[48]](references.md#ref-48). The `.pt` that notebook saved is
byte-identical to the file on the Hub (sha256 `58a63e41…`). The inference was right, but it
cited the wrong source, and a report has to cite a source that exists.

The corrections, in order:

1. **The wording.** *Done 2026-09-30.* `trained_on.basis` in the two active scenarios now names
   the training notebook, and the pages that said *model card* say where the claim really comes
   from. [ADR 0002](adr/0002-verifying-a-foreign-model.md) keeps its original text as the record,
   with a dated correction.
2. **The card.** *Done 2026-09-30.* Uploaded as commit
   [`d6f877a`](https://huggingface.co/sabrinahartung1010/skin-lesion-resnet18/commit/d6f877a88a282d2fa491a43b8bdd2bcd5bc8d506),
   byte-identical to
   [`model_cards/skin-lesion-resnet18/README.md`](https://github.com/sabrinahartung/verifai-medical/blob/main/model_cards/skin-lesion-resnet18/README.md),
   which stays in this repository as the source of the card. A change to the card is made there
   first and uploaded again.
3. **Re-pin and re-run.** *Done 2026-09-30.* The weights file in `d6f877a` is the same LFS object
   as in `f96683d` (sha256 `58a63e41…`), so the three scenarios on the checkpoint are pinned to
   the card's commit and the two active reports were re-run. Only the provenance sentence, the
   quoted basis and the recorded revision changed; every measured value is identical. The
   archived report is left as it was recorded. Resolved again, the repository now yields
   `trained_on: {corpora: [ham10000], basis: "model card metadata"}` and the licence from the
   card's header; architecture, class order and preprocessing remain TODO, since only the
   card's prose states them.

What the card changes in a report is the `basis`: the training data becomes *declared by the
author* rather than *inferred*. The integrity verdicts do not change. Corpus ancestry still finds
HAM10000 on both sides with no image-by-image check, so leakage still *cannot be ruled out*.

!!! question "A row-level check is now possible — deferred 2026-09-30"
    The notebook names the exact split, and `marmal88/skin_cancer` carries `image_id` and
    `lesion_id` for every row. So a training manifest could be built from its `train` split
    and declared under `integrity.train_manifests`, and the split check could then count
    shared images instead of stopping at *cannot be ruled out*. Given the audit (99.5% of
    HAM10000 is in `train` and `validation`), the result would almost certainly be contamination.
    The runner would then refuse the HAM10000 run, and the verdict would move from
    `insufficient` to `invalid`. That is more honest, but it ends the configuration's role as the
    example of what cannot be verified about someone else's model. Deferred on 2026-09-30:
    the configuration stays the example for now, and the check is revisited when these
    scenarios are re-run with more metrics.

## Uploading the card

A step-by-step for the first time. Everything happens in the browser; no code is needed.

1. **Read the draft** at `model_cards/skin-lesion-resnet18/README.md` and change anything that
   is not true or not how you want it said. Check these three first:
    - the **licence**: `cc-by-nc-4.0` follows HAM10000's own licence (CC BY-NC 4.0 [[1]](references.md#ref-1)).
      Whether weights trained on a dataset count as adapted material is not settled law, so
      matching the dataset's licence is the cautious choice rather than a legal requirement.
      It is still your decision;
    - the **author** line, and whether you want your name there;
    - the **known weakness** and **population** notes, which come from the training notebook
      and the HAM10000 paper.
2. **Open the repository** on the Hub, signed in:
   `https://huggingface.co/sabrinahartung1010/skin-lesion-resnet18`. Because there is no card,
   the page offers **Create model card**. Alternatively, open *Files and versions*, then
   *Add file*, then *Create a new file*, and name it `README.md`.
3. **Paste the whole draft**, including the `---` header at the top. The header is the
   machine-readable metadata; without it the licence and dataset fields stay empty.
4. **Check the preview.** The Hub shows the licence and the dataset as tags on the right of
   the model page. If they are missing, the header did not parse, usually because of a missing
   `---` or wrong indentation.
5. **Commit** with a message such as *Add model card*. This creates a new commit on `main`;
   the weights are untouched.
6. **Tell me the new commit hash.** It is on *Files and versions*, then *History*. I then
   re-pin the three scenarios, re-run the two active ones and check that the resolver reads
   the card.

The same repository on GitHub, where the notebook lives, has no licence file either. It is
worth adding one there too, so that the code and the weights both say what may be done with
them.

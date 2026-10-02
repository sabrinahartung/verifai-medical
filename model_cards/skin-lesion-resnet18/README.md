---
license: cc-by-nc-4.0
datasets:
- marmal88/skin_cancer
library_name: pytorch
pipeline_tag: image-classification
tags:
- dermatology
- skin-lesion
- ham10000
- resnet18
- torchvision
- education
---

# Skin-Lesion ResNet18 (HAM10000, class-weighted)

A torchvision ResNet18, pretrained on ImageNet and fine-tuned to sort dermatoscopic images into
the seven HAM10000 lesion classes. It was trained as a learning exercise and is the first
model evaluated by [VERIFAI Medical](https://github.com/sabrinahartung/verifai-medical).

> **Not a medical device.** This is an educational proof of concept. It has not been validated
> clinically and must not be used to diagnose, triage or reassure anyone about a skin lesion.

## Model details

| | |
|---|---|
| Architecture | torchvision `resnet18`, ImageNet weights (`ResNet18_Weights.DEFAULT`), final layer replaced by `Linear(512, 7)` |
| File | `resnet18_ham10000_classweights.pt` — a PyTorch `state_dict`, not a pickled model |
| Framework | PyTorch / torchvision |
| Trained with | [`SkinLesions.ipynb`](https://github.com/sabrinahartung/ham10000-skin-lesion-classification/blob/main/SkinLesions.ipynb) |
| Author | Sabrina Hartung |
| Licence | CC BY-NC 4.0, following the training data's licence (see below) |

A `state_dict` stores numbers only. It does not record the architecture, the order of the
output classes or the preprocessing, so all three are stated here, and a different choice for
any of them silently gives a different model.

### Output classes, in output order

| Index | Class |
|---|---|
| 0 | `actinic_keratoses` |
| 1 | `basal_cell_carcinoma` |
| 2 | `benign_keratosis-like_lesions` |
| 3 | `dermatofibroma` |
| 4 | `melanocytic_Nevi` |
| 5 | `melanoma` |
| 6 | `vascular_lesions` |

The order is the sorted list of the dataset's `dx` labels.

### Preprocessing at inference

1. Convert to RGB.
2. Resize to 224 × 224 (`transforms.Resize((224, 224))`, no crop).
3. `ToTensor()`, which scales pixels to [0, 1].
4. Normalise with the ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`.

## How to load it

```python
import torch
import torchvision
from huggingface_hub import hf_hub_download

path = hf_hub_download("sabrinahartung1010/skin-lesion-resnet18",
                       "resnet18_ham10000_classweights.pt")
model = torchvision.models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 7)
model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
model.eval()
```

## Training data

The `train` split of [`marmal88/skin_cancer`](https://huggingface.co/datasets/marmal88/skin_cancer),
a repackaging of HAM10000 (Tschandl et al., 2018, doi:10.1038/sdata.2018.161). The `validation`
split was used to watch the training, not to pick a checkpoint: the file is the state after the
last epoch.

**That dataset's splits share lesions.** An audit by image and lesion id found that 80% of
the images in its `test` split also appear in `train`, and that `train` and `validation`
together cover 9,964 of HAM10000's 10,015 images. Two consequences:

- any accuracy measured on `marmal88/skin_cancer`'s own `validation` or `test` split, including
  the one in the training notebook, mostly measures memory, not generalisation;
- almost every HAM10000 image, whatever split a later project puts it in, was seen in training.

The audit and its method are in the VERIFAI Medical
[roadmap](https://github.com/sabrinahartung/verifai-medical/blob/main/docs/ROADMAP.md#why-this-exists).

## Training procedure

| | |
|---|---|
| Epochs | 5 |
| Optimiser | Adam, learning rate 1e-4 |
| Batch size | 32 |
| Loss | cross-entropy with class weights `total / (n_classes × count)`, because about two thirds of the images are benign nevi |
| Augmentation (train only) | random horizontal and vertical flips, colour jitter (brightness 0.1, contrast 0.1) |
| Frozen layers | none; the whole network was fine-tuned |
| Hardware | Apple Silicon (MPS) |

## Evaluation

Because of the leakage above, a trustworthy number needs images from outside HAM10000. The
independent evaluation, with intervals, sample sizes and what each number cannot tell you, is
the [VERIFAI Medical showcase](https://verifai-medical.streamlit.app/) and its
[current results](https://github.com/sabrinahartung/verifai-medical/blob/main/docs/results.md).
The Derm7pt run there is the one no leakage can inflate. The numbers are not copied into this
card so that they cannot go out of date here.

## Intended use and limits

- **Intended:** learning, teaching and testing evaluation tooling.
- **Out of scope:** any clinical, screening or consumer-health use.
- **Known weakness:** melanoma recall. In the training notebook, 14% of melanomas were called
  benign nevi, the clinically most dangerous error.
- **Population:** HAM10000 comes from two sites, in Austria and Australia, and mostly shows
  lighter skin. Performance on other populations, cameras or clinics is not established by
  this training data.

## Licence

HAM10000 is published under CC BY-NC 4.0 (Harvard Dataverse, doi:10.7910/DVN/DBW86T). These
weights are released under the same licence: attribution required, no commercial use.

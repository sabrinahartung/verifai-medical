# References

Everything this project builds on, numbered so the rest of the docs can point at it:
write `[1]` in a page and link it here.

!!! info "How to read the status column"
    **Verified** means the citation was taken from the authors' own page or the dataset's own
    README during this work, not from memory. **Compiled** means it is a standard reference
    written out here and worth re-checking against the publisher before it goes into a thesis
    or a paper. The distinction is kept because a plausible-looking DOI is the easiest thing
    in a bibliography to get wrong and the hardest to notice.

---

## Datasets

**[1] HAM10000** · *Verified*
:   Tschandl, P., Rosendahl, C. & Kittler, H. **The HAM10000 dataset, a large collection of
    multi-source dermatoscopic images of common pigmented skin lesions.** *Scientific Data*
    **5**, 180161 (2018).
    [doi:10.1038/sdata.2018.161](https://doi.org/10.1038/sdata.2018.161) ·
    [paper](https://www.nature.com/articles/sdata2018161)

    Used as: the evaluation set for every internal run (`ham10000_test.csv`, 1,493 images),
    and as part of the ISIC 2019 training corpus. The ISIC organisers ask specifically for
    this published version rather than the earlier preprint.

**[2] BCN20000** · *Verified*
:   Combalia, M., Codella, N. C. F., Rotemberg, V., Helba, B., Vilaplana, V., Reiter, O.,
    Halpern, A. C., Puig, S. & Malvehy, J. **BCN20000: Dermoscopic Lesions in the Wild.**
    (2019). [arXiv:1908.02288](https://arxiv.org/abs/1908.02288)

    Used as: part of the ISIC 2019 corpus this project trains on. It is a required citation
    for anyone using the 2019 challenge data, which is easy to miss because only HAM10000 is
    named in the filenames.

**[3] ISIC Challenge** · *Verified*
:   Codella, N. C. F., Gutman, D., Celebi, M. E., Helba, B., Marchetti, M. A., Dusza, S. W.,
    Kalloo, A., Liopyris, K., Mishra, N., Kittler, H. & Halpern, A. **Skin Lesion Analysis
    Toward Melanoma Detection: A Challenge at the 2017 International Symposium on Biomedical
    Imaging (ISBI), Hosted by the International Skin Imaging Collaboration (ISIC).** (2017).
    [arXiv:1710.05006](https://arxiv.org/abs/1710.05006)

    Used as: the challenge framework the 2019 training data is distributed under.

**[4] Derm7pt — the 7-point criteria dataset** · *Verified*
:   Kawahara, J., Daneshvar, S., Argenziano, G. & Hamarneh, G. **Seven-point checklist and
    skin lesion classification using multitask multimodal neural nets.** *IEEE Journal of
    Biomedical and Health Informatics* **23**(2), 538–546 (2019).
    [doi:10.1109/JBHI.2018.2824327](https://doi.org/10.1109/JBHI.2018.2824327) ·
    [dataset](http://derm.cs.sfu.ca/)

    Used as: the **external** evaluation set — a different clinic, camera and population, and
    data no model here has ever trained on.

    !!! warning "Licence: CC BY-NC-ND 4.0, and the images may not be redistributed"
        The dataset's own README states it plainly: *"The images may not be redistributed"*,
        academic research use only, and Dr. Giuseppe Argenziano retains all rights.

        Two consequences for this repo. The images stay in gitignored `data/raw/` and are
        never committed. And **no derived image may be published** — a Grad-CAM overlay is an
        adaptation, so `explainability.gradcam` is left out of any Derm7pt scenario rather
        than rendered and then hidden. The metrics themselves are numbers, not adaptations,
        and are unaffected.

**[5] Derm7pt+ (concept-consistent re-release)** · *Verified*
:   Nápoles, G., Grau, I. & Salgueiro, Y. **Concept Inconsistency in Dermoscopic Concept
    Bottleneck Models: A Rough-Set Analysis of the Derm7pt Dataset.** (2026).
    [dataset & code](https://github.com/gnapoles/Consistent-Derm7pt) ·
    [record](https://research.tue.nl/en/datasets/derm7pt-a-concept-consistent-dermoscopy-benchmark/)

    **Not used here**, and recorded so the reasoning is not lost. It ships corrected concept
    annotations and filtered splits for concept bottleneck models, not images. Its two
    variants would both bias a melanoma measurement: the symmetric one removes 136 melanoma
    cases (54% of them), and the asymmetric one removes the concept-inconsistent
    *non*-melanoma cases, which are the confusable benign ones — that would flatter precision
    without the model improving.

---

## Methods

**[6] ResNet** · *Compiled*
:   He, K., Zhang, X., Ren, S. & Sun, J. **Deep Residual Learning for Image Recognition.**
    (2015). [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)

    Used as: every backbone in this project (ResNet18, ResNet50), ImageNet-pretrained.

**[7] Grad-CAM** · *Compiled*
:   Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D. & Batra, D.
    **Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.**
    (2016). [arXiv:1610.02391](https://arxiv.org/abs/1610.02391)

    Used as: the explainability pillar. Requires a convolutional feature map, which is why a
    ViT backbone would report `unavailable` rather than a substitute.

**[8] Deletion / insertion faithfulness (RISE)** · *Compiled*
:   Petsiuk, V., Das, A. & Saenko, K. **RISE: Randomized Input Sampling for Explanation of
    Black-box Models.** (2018). [arXiv:1806.07421](https://arxiv.org/abs/1806.07421)

    Used as: the deletion check behind `mean_deletion_faithfulness` — hide what the heatmap
    highlights and see whether confidence actually falls. An explanation nobody has tested is
    a claim, not evidence.

**[9] Focal loss** · *Compiled*
:   Lin, T.-Y., Goyal, P., Girshick, R., He, K. & Dollár, P. **Focal Loss for Dense Object
    Detection.** (2017). [arXiv:1708.02002](https://arxiv.org/abs/1708.02002)

    Used as: one of the two class-imbalance interventions in experiment 2. Reported as a
    negative result.

**[10] Membership inference** · *Compiled*
:   Shokri, R., Stronati, M., Song, C. & Shmatikov, V. **Membership Inference Attacks Against
    Machine Learning Models.** (2016). [arXiv:1610.05820](https://arxiv.org/abs/1610.05820)

    Used as: the privacy pillar — can an attacker tell a training image from an unseen one
    using confidence alone?

---

## Statistics

**[11] Wilson score interval** · *Compiled*
:   Wilson, E. B. **Probable Inference, the Law of Succession, and Statistical Inference.**
    *Journal of the American Statistical Association* **22**(158), 209–212 (1927).

    Used as: the interval on every proportion in this project. Chosen over the normal
    approximation because that one misbehaves at 0 and 1 — exactly where the small classes
    live.

**[12] AUC standard error** · *Compiled*
:   Hanley, J. A. & McNeil, B. J. **The meaning and use of the area under a receiver operating
    characteristic (ROC) curve.** *Radiology* **143**(1), 29–36 (1982).
    [doi:10.1148/radiology.143.1.7063747](https://doi.org/10.1148/radiology.143.1.7063747)

    Used as: the interval on the membership-inference AUC, which the privacy verdict is read
    from rather than from the point estimate.

**[13] ITA and skin tone in dermatology datasets** · *Compiled*
:   Kinyanjui, N. M., Odonga, T., Cintas, C., Codella, N. C. F., Panda, R., Sattigeri, P. &
    Varshney, K. R. **Fairness of Classifiers Across Skin Tones in Dermatology.** (2020).
    [arXiv:2006.09865](https://arxiv.org/abs/2006.09865)

    Used as: the basis for the ITA-estimated skin-tone bins in the fairness pillar. HAM10000
    carries no skin-type labels, so the tone is estimated from the image itself — a proxy,
    and reported as one.

---

## Citing this project

Nothing here is a medical device or validated for clinical use. It is an educational
Responsible-AI evaluation framework, and any number it reports carries the sample size and
interval that produced it.

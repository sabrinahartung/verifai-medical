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

**[1] HAM10000** · *Verified* {#ref-1}
:   Tschandl, P., Rosendahl, C. & Kittler, H. **The HAM10000 dataset, a large collection of
    multi-source dermatoscopic images of common pigmented skin lesions.** *Scientific Data*
    **5**, 180161 (2018).
    [doi:10.1038/sdata.2018.161](https://doi.org/10.1038/sdata.2018.161) ·
    [paper](https://www.nature.com/articles/sdata2018161)

    Used as: the evaluation set for every internal run (`ham10000_test.csv`, 1,493 images),
    and as part of the ISIC 2019 training corpus. The ISIC organisers ask specifically for
    this published version rather than the earlier preprint.

**[2] BCN20000** · *Verified* {#ref-2}
:   Combalia, M., Codella, N. C. F., Rotemberg, V., Helba, B., Vilaplana, V., Reiter, O.,
    Halpern, A. C., Puig, S. & Malvehy, J. **BCN20000: Dermoscopic Lesions in the Wild.**
    (2019). [arXiv:1908.02288](https://arxiv.org/abs/1908.02288)

    Used as: part of the ISIC 2019 corpus this project trains on. It is a required citation
    for anyone using the 2019 challenge data, which is easy to miss because only HAM10000 is
    named in the filenames.

**[3] ISIC Challenge** · *Verified* {#ref-3}
:   Codella, N. C. F., Gutman, D., Celebi, M. E., Helba, B., Marchetti, M. A., Dusza, S. W.,
    Kalloo, A., Liopyris, K., Mishra, N., Kittler, H. & Halpern, A. **Skin Lesion Analysis
    Toward Melanoma Detection: A Challenge at the 2017 International Symposium on Biomedical
    Imaging (ISBI), Hosted by the International Skin Imaging Collaboration (ISIC).** (2017).
    [arXiv:1710.05006](https://arxiv.org/abs/1710.05006)

    Used as: the challenge framework the 2019 training data is distributed under.

**[4] Derm7pt — the 7-point criteria dataset** · *Verified* {#ref-4}
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

**[5] Derm7pt+ (concept-consistent re-release)** · *Verified* {#ref-5}
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

**[6] ResNet** · *Compiled* {#ref-6}
:   He, K., Zhang, X., Ren, S. & Sun, J. **Deep Residual Learning for Image Recognition.**
    (2015). [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)

    Used as: every backbone in this project (ResNet18, ResNet50), ImageNet-pretrained.

**[7] Grad-CAM** · *Compiled* {#ref-7}
:   Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D. & Batra, D.
    **Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.**
    (2016). [arXiv:1610.02391](https://arxiv.org/abs/1610.02391)

    Used as: the explainability pillar. Requires a convolutional feature map, which is why a
    ViT backbone would report `unavailable` rather than a substitute.

**[8] Deletion / insertion faithfulness (RISE)** · *Compiled* {#ref-8}
:   Petsiuk, V., Das, A. & Saenko, K. **RISE: Randomized Input Sampling for Explanation of
    Black-box Models.** (2018). [arXiv:1806.07421](https://arxiv.org/abs/1806.07421)

    Used as: the deletion check behind `mean_deletion_faithfulness` — hide what the heatmap
    highlights and see whether confidence actually falls. An explanation nobody has tested is
    a claim, not evidence.

**[9] Focal loss** · *Compiled* {#ref-9}
:   Lin, T.-Y., Goyal, P., Girshick, R., He, K. & Dollár, P. **Focal Loss for Dense Object
    Detection.** (2017). [arXiv:1708.02002](https://arxiv.org/abs/1708.02002)

    Used as: one of the two class-imbalance interventions in experiment 2. Reported as a
    negative result.

**[10] Membership inference** · *Compiled* {#ref-10}
:   Shokri, R., Stronati, M., Song, C. & Shmatikov, V. **Membership Inference Attacks Against
    Machine Learning Models.** (2016). [arXiv:1610.05820](https://arxiv.org/abs/1610.05820)

    Used as: the privacy pillar — can an attacker tell a training image from an unseen one
    using confidence alone?

---

## Statistics

**[11] Wilson score interval** · *Compiled* {#ref-11}
:   Wilson, E. B. **Probable Inference, the Law of Succession, and Statistical Inference.**
    *Journal of the American Statistical Association* **22**(158), 209–212 (1927).

    Used as: the interval on every proportion in this project. Chosen over the normal
    approximation because that one misbehaves at 0 and 1 — exactly where the small classes
    live.

**[12] AUC standard error** · *Compiled* {#ref-12}
:   Hanley, J. A. & McNeil, B. J. **The meaning and use of the area under a receiver operating
    characteristic (ROC) curve.** *Radiology* **143**(1), 29–36 (1982).
    [doi:10.1148/radiology.143.1.7063747](https://doi.org/10.1148/radiology.143.1.7063747)

    Used as: the interval on the membership-inference AUC, which the privacy verdict is read
    from rather than from the point estimate.

**[13] ITA and skin tone in dermatology datasets** · *Compiled* {#ref-13}
:   Kinyanjui, N. M., Odonga, T., Cintas, C., Codella, N. C. F., Panda, R., Sattigeri, P. &
    Varshney, K. R. **Fairness of Classifiers Across Skin Tones in Dermatology.** (2020).
    [arXiv:2006.09865](https://arxiv.org/abs/2006.09865)

    Used as: the basis for the ITA-estimated skin-tone bins in the fairness pillar. HAM10000
    carries no skin-type labels, so the tone is estimated from the image itself — a proxy,
    and reported as one.


## Reporting and evaluation methodology

**[14] TRIPOD+AI** · *Verified* {#ref-14}
:   Collins, G. S., Moons, K. G. M., Dhiman, P., Riley, R. D., Beam, A. L., Van Calster, B.,
    Ghassemi, M., Liu, X., Reitsma, J. B., van Smeden, M. *et al.* **TRIPOD+AI statement:
    updated guidance for reporting clinical prediction models that use regression or machine
    learning methods.** *BMJ* **385**, e078378 (2024).
    [doi:10.1136/bmj-2023-078378](https://doi.org/10.1136/bmj-2023-078378) ·
    [PMID 38626948](https://pubmed.ncbi.nlm.nih.gov/38626948/)

    Used as: the reference standard for what an evaluation of a trained model on a new dataset
    has to report — the external-validation discipline behind the roadmap's provenance and
    label-space preflight, and behind reporting calibration next to discrimination rather
    than discrimination alone.


## The expanded metric catalogue

The sources behind the metrics planned in [The pillars](pillars.md) and scheduled in the
[roadmap](ROADMAP.md). **All are marked *Compiled*:** they are standard references written
out from working knowledge, and none was checked against the authors' own page in this pass.
Re-verify each one before it goes into a thesis or a paper — and mark it *Verified* here when
you do, which is the whole point of the status column.

### Calibration and decision quality

**[15] Calibration of modern neural networks** · *Compiled* {#ref-15}
:   Guo, C., Pleiss, G., Sun, Y. & Weinberger, K. Q. **On Calibration of Modern Neural
    Networks.** *ICML* (2017). [arXiv:1706.04599](https://arxiv.org/abs/1706.04599)

    Used as: the basis for expected calibration error and the reliability curve. Also the
    source of the finding that modern networks are systematically overconfident, which is why
    calibration is a pillar-level metric here rather than a footnote to accuracy.

**[16] The calibration hierarchy** · *Compiled* {#ref-16}
:   Van Calster, B., McLernon, D. J., van Smeden, M., Wynants, L. & Steyerberg, E. W.
    **Calibration: the Achilles heel of predictive analytics.** *BMC Medicine* **17**, 230 (2019).

    Used as: the clinical framing — calibration intercept and slope, and why a model that
    discriminates well can still be unusable on a new population.

**[17] Decision curve analysis** · *Compiled* {#ref-17}
:   Vickers, A. J. & Elkin, E. B. **Decision curve analysis: a novel method for evaluating
    prediction models.** *Medical Decision Making* **26**(6), 565–574 (2006).

    Used as: net benefit across a range of harm ratios, reported as a curve. It is the
    standard way to ask whether a model is worth using at all, and it makes the value
    judgement explicit instead of hiding it in a chosen threshold.

**[18] Selective prediction** · *Compiled* {#ref-18}
:   Geifman, Y. & El-Yaniv, R. **Selective Classification for Deep Neural Networks.** *NeurIPS*
    (2017). [arXiv:1705.08500](https://arxiv.org/abs/1705.08500)

    Used as: the accuracy-against-coverage curve for a model allowed to abstain — the shape
    that matters for a triage tool.

### Fairness

**[19] Equality of opportunity** · *Compiled* {#ref-19}
:   Hardt, M., Price, E. & Srebro, N. **Equality of Opportunity in Supervised Learning.**
    *NeurIPS* (2016). [arXiv:1610.02413](https://arxiv.org/abs/1610.02413)

    Used as: the definitions of equalised odds and equal opportunity in the group-fairness
    metric.

**[20] Individual fairness** · *Compiled* {#ref-20}
:   Dwork, C., Hardt, M., Pitassi, T., Reingold, O. & Zemel, R. **Fairness Through Awareness.**
    *ITCS* (2012). [arXiv:1104.3913](https://arxiv.org/abs/1104.3913)

    Used as: "similar cases get similar decisions" — the basis of the consistency / flip-rate
    metric.

**[21] The impossibility result** · *Compiled* {#ref-21}
:   Chouldechova, A. **Fair prediction with disparate impact.** (2017).
    [arXiv:1703.00056](https://arxiv.org/abs/1703.00056) · Kleinberg, J., Mullainathan, S. &
    Raghavan, M. **Inherent Trade-Offs in the Fair Determination of Risk Scores.** (2016).
    [arXiv:1609.05807](https://arxiv.org/abs/1609.05807)

    Used as: the reason the fairness pillar reports several definitions side by side and names
    the tension between them. When base rates genuinely differ between groups, equal
    calibration and equalised odds cannot both hold — arithmetic, not a design failure, and
    exactly the kind of statement that replaces a composite score here.

### Robustness

**[22] Common corruptions** · *Compiled* {#ref-22}
:   Hendrycks, D. & Dietterich, T. **Benchmarking Neural Network Robustness to Common
    Corruptions and Perturbations.** *ICLR* (2019).
    [arXiv:1903.12261](https://arxiv.org/abs/1903.12261)

    Used as: the severity-sweep shape for the natural-corruption metric. The existing
    single-severity check is a simplification of this.

**[23] Adversarial attacks** · *Compiled* {#ref-23}
:   Goodfellow, I. J., Shlens, J. & Szegedy, C. **Explaining and Harnessing Adversarial
    Examples.** (2014). [arXiv:1412.6572](https://arxiv.org/abs/1412.6572) ·
    Madry, A., Makelov, A., Schmidt, L., Tsipras, D. & Vladu, A. **Towards Deep Learning Models
    Resistant to Adversarial Attacks.** (2017). [arXiv:1706.06083](https://arxiv.org/abs/1706.06083) ·
    Moosavi-Dezfooli, S.-M., Fawzi, A. & Frossard, P. **DeepFool.** *CVPR* (2016).
    [arXiv:1511.04599](https://arxiv.org/abs/1511.04599)

    Used as: FGSM, PGD and DeepFool in the adversarial metric. Each paper states its threat
    model explicitly, which is the convention this project adopts — the norm, the budget and
    the iteration count travel with the number.

**[24] Text attacks** · *Compiled* {#ref-24}
:   Morris, J. X., Lifland, E., Yoo, J. Y., Grigsby, J., Jin, D. & Qi, Y. **TextAttack: A
    Framework for Adversarial Attacks, Data Augmentation, and Adversarial Training in NLP.**
    *EMNLP* (2020). [arXiv:2005.05909](https://arxiv.org/abs/2005.05909) ·
    [GitHub](https://github.com/QData/TextAttack)

    Used as: the intended dependency for word-substitution attacks in the text domain — the
    one place in the catalogue where re-implementing is more error-prone than importing.

### Explanation quality

**[25] Sanity checks for saliency maps** · *Compiled* {#ref-25}
:   Adebayo, J., Gilmer, J., Muelly, M., Goodfellow, I., Hardt, M. & Kim, B. **Sanity Checks
    for Saliency Maps.** *NeurIPS* (2018). [arXiv:1810.03292](https://arxiv.org/abs/1810.03292)

    Used as: the parameter-randomisation check. It is the paper showing that several popular
    attribution methods produce near-identical maps for a trained and a randomised model —
    i.e. they were describing the image, not the model. Almost nobody runs this, which is
    precisely why it is in the catalogue.

**[26] Explanation infidelity and sensitivity** · *Compiled* {#ref-26}
:   Yeh, C.-K., Hsieh, C.-Y., Suggala, A. S., Inouye, D. I. & Ravikumar, P. **On the (In)fidelity
    and Sensitivity of Explanations.** *NeurIPS* (2019).
    [arXiv:1901.09392](https://arxiv.org/abs/1901.09392)

    Used as: max-sensitivity, the explanation-stability metric — does an imperceptible change
    to the input rewrite the heatmap.

**[27] Quantus** · *Compiled* {#ref-27}
:   Hedström, A., Weber, L., Bareeva, D., Krakowczyk, D., Motzkus, F., Samek, W., Lapuschkin, S.
    & Höhne, M. M.-C. **Quantus: An Explainable AI Toolkit for Responsible Evaluation of Neural
    Network Explanations.** *JMLR* (2023). [arXiv:2202.06861](https://arxiv.org/abs/2202.06861) ·
    [GitHub](https://github.com/understandable-machine-intelligence-lab/Quantus) ·
    [PyPI](https://pypi.org/project/quantus/)

    Used as: the implementation behind five of the eight explainability rows, reached through a
    single adapter module. **Licensed LGPL-3.0-or-later**, unlike everything else in this
    section: importing it unmodified imposes nothing on this repository, but modifying it would
    put those modifications under the LGPL. That is a second, independent reason to wrap it and
    never fork it. Verified 2026-09-18: v0.6.0 (2025-07-21), actively maintained, resolves
    against this stack.

### Privacy

**[28] Memorisation and canary exposure** · *Compiled* {#ref-28}
:   Carlini, N., Liu, C., Erlingsson, Ú., Kos, J. & Song, D. **The Secret Sharer: Evaluating and
    Testing Unintended Memorization in Neural Networks.** *USENIX Security* (2019).
    [arXiv:1802.08232](https://arxiv.org/abs/1802.08232)

    Used as: canary exposure — the memorisation metric for models we train ourselves.

**[29] Training-data extraction** · *Compiled* {#ref-29}
:   Carlini, N., Tramèr, F., Wallace, E., Jagielski, M., Herbert-Voss, A., Lee, K., Roberts, A.,
    Brown, T., Song, D., Erlingsson, Ú., Oprea, A. & Raffel, C. **Extracting Training Data from
    Large Language Models.** *USENIX Security* (2021).
    [arXiv:2012.07805](https://arxiv.org/abs/2012.07805)

    Used as: the generative privacy metric — verbatim recall and PII regurgitation under
    prompting.

### Generative evaluation

**[30] Benchmark contamination** · *Compiled* {#ref-30}
:   Sainz, O., Campos, J. A., García-Ferrero, I., Etxaniz, J., de Lacalle, O. L. & Agirre, E.
    **NLP Evaluation in Trouble: On the Need to Measure LLM Data Contamination for each
    Benchmark.** *Findings of EMNLP* (2023).

    Used as: `integrity.benchmark_contamination` — the generative analogue of this project's
    split-leakage check, and the same argument one level up.

**[31] Clinical knowledge in large language models** · *Compiled* {#ref-31}
:   Singhal, K., Azizi, S., Tu, T., Mahdavi, S. S., Wei, J., Chung, H. W., Scales, N.,
    Tanwani, A., Cole-Lewis, H., Pfohl, S. *et al.* **Large language models encode clinical
    knowledge.** *Nature* **620**, 172–180 (2023).

    Used as: the rubric-based framing of the safety and generative-performance pillars —
    graded axes such as harm, scope and appropriateness rather than a single accuracy figure.

**[32] TRIPOD-LLM** · *Compiled* {#ref-32}
:   Gallifant, J., Afshar, M., Ameen, S., Aphinyanaphongs, Y., Chen, S., Cacciamani, G.,
    Demner-Fushman, D., Dligach, D., Daneshjou, R. *et al.* **The TRIPOD-LLM Statement: A
    Targeted Guideline For Reporting Large Language Models Use.** *medRxiv* (2024).
    [doi:10.1101/2024.07.24.24310930](https://doi.org/10.1101/2024.07.24.24310930)

    Used as: the reporting standard for the generative phase, as [[14]](#ref-14) is for the classifier
    phase. Relevant here mainly for what it requires to be disclosed — prompts, model versions,
    and how any judge was validated.


## Prior work in this project's own lineage

VERIFAI has been built three times. These entries exist so the current repository can cite
what it inherits instead of silently reinventing it — and so a reader can find the working
implementation of a metric this catalogue still lists as planned.

**[33] VERIFAI prototype (master's project)** · *Verified* {#ref-33}
:   Göllner, S. **VERIFAI — Prototype Implementation.** Master's project, Department of
    Computer Science, Faculty of Engineering and Computer Science, Hamburg University of
    Applied Sciences (HAW Hamburg). Local: `~/Development/VERIFAI_PROTOTYPE`.

    Used as: the origin of the aspect → sub-aspect → per-modality taxonomy this catalogue
    follows, and a working implementation of the security, privacy and XAI-evaluation
    metrics across image, text and tabular data. Its skin-cancer image case is the direct
    ancestor of this repository's use case. Its `components/responsibility/` layer — a 0–10
    score per pillar rendered as `danger` / `warning` / `success` — is what
    [the roadmap](ROADMAP.md#no-composite-score) deliberately does not carry forward.

**[34] VERIFAI 2.0 / Test Lab** · *Verified* {#ref-34}
:   Hartung, S. **VERIFAI 2.0 — evaluation backend and findings layer.** Local:
    `~/Development/verifai_2_0` (`verifai_test_lab` is an earlier copy of the same work).

    Used as: the source of the findings-layer design this project adopts — the separation of
    **Indicator** (what was measured, owned by the metric) from **Criterion** (whether that is
    acceptable, owned by a versioned policy file) from **Finding** (the two, joined, owned by
    the engine), together with intended-use profiles, interval-gated criteria, and the rule
    that every gap is expressed in the indicator's own unit. Also the working implementation
    of the generative fairness, privacy and security metrics listed in
    [The pillars](pillars.md).

### Tools these metrics are built on

**[35] Adversarial Robustness Toolbox (ART)** · *Compiled* {#ref-35}
:   Nicolae, M.-I., Sinn, M., Tran, M. N., Buesser, B., Rawat, A., Wistuba, M., Zantedeschi, V.,
    Baracaldo, N., Chen, B., Ludwig, H., Molloy, I. M. & Edwards, B. **Adversarial Robustness
    Toolbox v1.0.0.** (2018). [arXiv:1807.01069](https://arxiv.org/abs/1807.01069) ·
    [GitHub](https://github.com/Trusted-AI/adversarial-robustness-toolbox)

    Used as: black-box membership inference and the tabular `ZooAttack` in the prototype [[33]](#ref-33).
    Its `worst_case_mia_score` (TPR at a low fixed FPR) is the reporting convention the
    privacy pillar should adopt — an average-case AUC understates a leak that is severe for a
    few records.

**[36] Foolbox** · *Compiled* {#ref-36}
:   Rauber, J., Zimmermann, R., Bethge, M. & Brendel, W. **Foolbox Native: Fast adversarial
    attacks to benchmark the robustness of machine learning models.** *Journal of Open Source
    Software* (2020). [GitHub](https://github.com/bethgelab/foolbox)

    Used as: `FGSM`, `LinfPGD`, `LinfDeepFoolAttack` and `LinfAdditiveUniformNoiseAttack` in
    the prototype [[33]](#ref-33) — including the additive-noise baseline that separates "an adversary
    did this" from "bad luck did this", and the epsilon sweep that makes the perturbation
    budget part of the result rather than a hidden constant.

**[37] ML Privacy Meter** · *Compiled* {#ref-37}
:   Murakonda, S. K. & Shokri, R. **ML Privacy Meter: Aiding Regulatory Compliance by
    Quantifying the Privacy Risks of Machine Learning.** (2020).
    [arXiv:2007.09339](https://arxiv.org/abs/2007.09339) ·
    [GitHub](https://github.com/privacytrustlab/ml_privacy_meter)

    Used as: the population and shadow membership attacks in the prototype [[33]](#ref-33). It is also
    where the shadow attack's precondition is explicit — reference data from the training
    distribution — which is why `privacy.mia_shadow` is permanently unavailable for a
    third-party checkpoint.

    Verified 2026-09-18: **install from git, not PyPI.** The published `privacy-meter` 1.0.1
    (2023-07-28) declares *zero* dependencies, which is a packaging defect rather than a lean
    library — it will import and then fail on whatever it actually needs.

**[38] MIMIR** · *Compiled* {#ref-38}
:   Duan, M., Suri, A., Mireshghallah, N., Min, S., Shi, W., Zettlemoyer, L., Tsvetkov, Y.,
    Choi, Y., Evans, D. & Hajishirzi, H. **Do Membership Inference Attacks Work on Large
    Language Models?** (2024). [GitHub](https://github.com/iamgroot42/mimir)

    Used as: the neighbourhood-comparison membership attack implemented in [[34]](#ref-34) for generative
    models. Its headline finding — that most MIAs barely beat chance on LLMs — is the reason
    the generative privacy metric must report its interval rather than a bare AUC.

    Verified 2026-09-18. Note the repository moved: `privacytrustlab/mimir`, the link carried in
    the older notes [[34]](#ref-34), is a 404. `iamgroot42/mimir` above is the live one.

**[39] FairMLHealth** · *Compiled* {#ref-39}
:   Center for Open Source Data and AI Technologies. **FairMLHealth: Tools and tutorials for
    variation analysis in healthcare machine learning.**
    [GitHub](https://github.com/KenSciResearch/fairMLHealth)

    Used as: **the source of the approach, not a live dependency.** It is healthcare-specific,
    and it reports a panel of fairness definitions side by side rather than choosing one —
    which is the behaviour `fairness.group` copies. The prototype [[33]](#ref-33) vendored it locally.

    Verified 2026-09-18: last release 1.0.2 (2021-09-15), last commit 2022-12-22, and it no
    longer builds on a current Python. `fairness.group` is therefore written here, against
    `_stats.py`, rather than imported.

**[40] ROBBIE and HolisticBias** · *Compiled* {#ref-40}
:   Esiobu, D., Tan, X., Hosseini, S., Ung, M., Zhang, Y., Fernandes, J., Dwivedi-Yu, J.,
    Presani, E., Williams, A. & Smith, E. M. **ROBBIE: Robust Bias Evaluation of Large
    Generative Language Models.** *EMNLP* (2023).
    [arXiv:2311.18140](https://arxiv.org/abs/2311.18140)

    Used as: the generative fairness suite implemented in [[34]](#ref-34) — adversarial bias, toxicity,
    regard, demographic bias, hate speech and holistic bias, each over a prompt corpus rather
    than a labelled test set. Code lives in
    [facebookresearch/ResponsibleNLP](https://github.com/facebookresearch/ResponsibleNLP).

    **Two licences, and the second one binds.** The HolisticBias and AdvPromptSet *code* is MIT;
    the HolisticBias *dataset* is **CC BY-SA 4.0**. Publishing scores computed over it is fine;
    publishing derived prompt material carries the ShareAlike obligation onto whatever it is
    published in — the same class of constraint as Derm7pt's [[4]](#ref-4) no-redistribution terms.
    Verified 2026-09-18.

### Reporting conventions for the metrics themselves

**[41] Membership inference, done properly** · *Compiled* {#ref-41}
:   Carlini, N., Chien, S., Nasr, M., Song, S., Terzis, A. & Tramèr, F. **Membership Inference
    Attacks From First Principles.** *IEEE S&P* (2022).
    [arXiv:2112.03570](https://arxiv.org/abs/2112.03570) · Song, L. & Mittal, P. **Systematic
    Evaluation of Privacy Risks of Machine Learning Models.** *USENIX Security* (2021).
    [arXiv:2003.10595](https://arxiv.org/abs/2003.10595)

    Used as: the argument that an average-case AUC is the wrong summary for a privacy attack.
    What matters is the true-positive rate at a low false-positive rate — whether a *few*
    records can be identified with confidence, which is the realistic threat. Cited in the
    policy rule behind `privacy.mia`.

**[42] ROAR and the critique of deletion metrics** · *Compiled* {#ref-42}
:   Hooker, S., Erhan, D., Kindermans, P.-J. & Kim, B. **A Benchmark for Interpretability
    Methods in Deep Neural Networks (ROAR).** *NeurIPS* (2019).
    [arXiv:1806.10758](https://arxiv.org/abs/1806.10758) · Rong, Y., Leemann, T., Borisov, V.,
    Kasneci, G. & Kasneci, E. **A Consistent and Efficient Evaluation Strategy for Attribution
    Methods.** *ICML* (2022). [arXiv:2202.00449](https://arxiv.org/abs/2202.00449)

    Used as: the reason deletion faithfulness needs a **measured random-attribution control**
    rather than a threshold. Masking pixels moves the input off the data distribution, so some
    of the confidence drop is distribution shift rather than evidence removal — and a random
    attribution suffers the same shift. The gap between the two is the part that means
    something. This is the `control` reference kind in the findings layer.

**[43] TorchAttack** · *Compiled* {#ref-43}
:   Wu, S. **torchattack: A curated collection of adversarial attacks in PyTorch.**
    [GitHub](https://github.com/spencerwooo/torchattack)

    Used as: the intended implementation of `robustness.adversarial` for images, replacing the
    prototype's [[33]](#ref-33) foolbox. Verified 2026-09-18: v1.7.2 (2025-12-22), MIT, and the only
    toolkit in this catalogue that adds **nothing** to the dependency tree — it declares
    `torch`, `torchvision` and `numpy`, all of which the engine already has.

**[44] Adversarial Robustness Toolbox reporting** · *Compiled* {#ref-44}
:   See [[35]](#ref-35) for the citation ·
    [GitHub](https://github.com/Trusted-AI/adversarial-robustness-toolbox) ·
    [PyPI](https://pypi.org/project/adversarial-robustness-toolbox/). Verified 2026-09-18: v1.20.1 (2025-07-07), MIT, resolves against
    this stack, adding six packages that Quantus [[27]](#ref-27) largely brings anyway.

    Used as: the black-box membership-inference attack, and specifically its
    `worst_case_mia_score` — the true-positive rate at a low fixed false-positive rate that
    [[41]](#ref-41) argues for, which the shipped `membership_inference_auc` does not yet report.

**[45] ferret** · *Compiled* {#ref-45}
:   Attanasio, G., Pastor, E., Di Bonaventura, C. & Nozza, D. **ferret: a Framework for
    Benchmarking Explainers on Transformers.** *EACL demo track* (2023).
    [GitHub](https://github.com/g8a9/ferret) · [PyPI](https://pypi.org/project/ferret-xai/)

    Named in the older shortlist [[34]](#ref-34) as the text-explainability toolkit. **Not used.** Verified
    2026-09-18: v0.4.2 (2024-01-08), no commits for 23 months, and it pins `numpy<2.0` alongside
    `scikit-image<0.22`, `opencv-python<5` and `shap<0.45` — on Python 3.13 there is no numpy-1
    wheel, so installation tries to compile numpy from source and fails. The pins cannot be
    relaxed without forking. Recorded here so the next person does not rediscover it.

**[46] AutoDAN** · *Compiled* {#ref-46}
:   Liu, X., Xu, N., Chen, M. & Xiao, C. **AutoDAN: Generating Stealthy Jailbreak Prompts on
    Aligned Large Language Models.** *ICLR* (2024).
    [arXiv:2310.04451](https://arxiv.org/abs/2310.04451) ·
    [GitHub](https://github.com/SheltonLiu-N/AutoDAN)

    Used as: one of the two intended implementations of `robustness.jailbreak` (tier 5).
    Research code rather than a package — cloned and adapted, not installed. Verified
    2026-09-18: MIT, last commit 2025-01-22.

**[47] GCG (llm-attacks)** · *Compiled* {#ref-47}
:   Zou, A., Wang, Z., Carlini, N., Nasr, M., Kolter, J. Z. & Fredrikson, M. **Universal and
    Transferable Adversarial Attacks on Aligned Language Models.** (2023).
    [arXiv:2307.15043](https://arxiv.org/abs/2307.15043) ·
    [GitHub](https://github.com/llm-attacks/llm-attacks)

    Used as: the other half of `robustness.jailbreak`, and the implementation behind the GCG
    metric already written in the 2.0 backend [[34]](#ref-34). Still the reference gradient-based jailbreak
    attack, and it needs `gradients` — so it is white-box only and cannot be run against a
    hosted model at all. Verified 2026-09-18: MIT, last commit 2024-08-02, i.e. two years stale;
    the attack remains the standard baseline but the repository is not maintained.

---

## Citing this project

Nothing here is a medical device or validated for clinical use. It is an educational
Responsible-AI evaluation framework, and any number it reports carries the sample size and
interval that produced it.

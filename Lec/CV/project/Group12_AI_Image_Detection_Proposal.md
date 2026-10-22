# AI Generated Image Detection Across Generators and Image Transformations

**MAIE5421 / ARIN5201 · Fall 2026 · Project Proposal · Group 12**

## 1 Project Background and Objectives

Realistic AI-generated images create a need for reliable authenticity assessment. Detectors may fail on unseen generators or after compression and resizing. The project will therefore examine how model choice and training augmentations affect detection across image sources and common transformations.

We propose a system that accepts a JPEG or PNG image and returns **Real** or **AI-Generated**, with an estimated probability of AI generation. We will compare two practical pretrained detection pipelines and investigate whether realistic augmentations improve detection on unseen generators while preserving performance on familiar sources.

## 2 Group Members and Preliminary Responsibilities

| Member | Full name | Student ID | Preliminary responsibility |
|---|---|---|---|
| A | GAO, Ge | 21337511 | Coordination, dataset construction and split management |
| B | SHI, Jingyi | 21386263 | CNN baseline and reproducible training pipeline |
| C | LUO, Xiaoyu | 21341457 | Pretrained feature comparison and ablation experiments |
| D | WANG, Ziyang | 21313656 | Unseen-generator and image-transformation experiments |
| E | YANG, Sibo | 21321809 | Evaluation pipeline, statistical analysis and data-bias checks |
| F | ZHANG, Boxiang | 21344801 | Inference interface, system integration and presentation coordination |

Responsibilities are adjustable. All members will contribute technical work, report content and presentation material, using weekly meetings, Git and a shared experiment log.

## 3 Preliminary Methodology

**Data construction and separation.** We will use the official ProGAN training and validation data released by the CNNDetection project for development. Final testing will cover held-out ProGAN images and unseen GAN and diffusion generators, such as StyleGAN2 and GLIDE, using the public test datasets released by CNNDetection and UniversalFakeDetect. We will verify access and usage terms before fixing the dataset manifest. Mixed-generator training will be an optional extension only if independent training data are available; public test sets will remain reserved for evaluation.

We target approximately 20,000 training images and 4,000 validation images, each balanced between real and generated images. Each test domain will contain up to 2,000 images in total: 1,000 real and 1,000 generated, adjusted to availability. Budgets may be reduced for one GPU or Colab. We will preserve official partitions, remove exact and near duplicates across splits, and keep derivatives of each source image together. We will match semantic categories where possible, document source, class and resolution distributions, and use label-independent preprocessing to reduce dataset shortcuts. Final test images will not inform training, model selection or threshold tuning.

**Models and training.** Our baseline will fine-tune an ImageNet-pretrained ResNet-18 with a binary classification head. We will compare it with a frozen CLIP ViT-B/32 image encoder and a trainable linear head. For CLIP, we will cache features for original images and a fixed set of augmented training variants separately. Both models will use binary cross-entropy, the same data splits and 224 × 224 inputs, with model-appropriate normalization. Learning rate, training duration and decision threshold will be selected using validation data only. Output probabilities are model estimates, not guarantees of authenticity.

**Improvement and ablation.** We will test label-independent JPEG compression, resizing and mild blur during training. Within each pipeline, we will compare training with and without the same predefined augmentations while keeping other settings fixed. This isolates the augmentation effect within that pipeline. The comparison between pipelines is practical rather than a controlled test of architecture alone, since pretraining and fine-tuning also differ. We will measure both unseen-generator performance and any loss of clean-image accuracy, prioritizing a small, reproducible experiment set.

**Evaluation and demonstration.** We will evaluate three settings: held-out ProGAN images, generators unseen during detector training and validation, and transformed versions of test images. Perturbations will include JPEG compression, blur, noise and cropping at predefined severity levels, applied equally to real and synthetic images. We will report accuracy, precision, recall, F1, ROC-AUC and confusion matrices, with AI-generated images as the positive class. Per-generator results and their macro-average will prevent large sources from dominating the summary. We will report clean-to-perturbed performance drops, variation over three training seeds for the main comparison, and representative false positives and false negatives. A lightweight upload interface will display the predicted class and AI-generation probability.

## 4 Project Timeline and Expected Outcomes

| Course week | Main activities and milestone |
|---|---|
| 7 | Finalize scope and responsibilities; submit proposal by 17 October 2026 at 23:59. |
| 8 | Verify dataset access, audit sources and freeze development and test partitions. |
| 9 | Complete CNN and frozen-feature baselines with validation results. |
| 10 | Run augmentation comparisons and ablations; freeze model choices and evaluation settings. |
| 11 | Complete unseen-generator and perturbation tests; analyze failures and integrate the demo. |
| 12–13 | Have the demo and approximately 10-minute presentation ready for the assigned slot; finalize the 6-page CVPR-style report by the course deadline. |

Expected outcomes are a working detector and reproducible evidence of generalization and robustness relative to the baseline. If resources are constrained, we will reduce dataset size while retaining unseen-generator tests. The report will document limitations, member contributions and failure cases.

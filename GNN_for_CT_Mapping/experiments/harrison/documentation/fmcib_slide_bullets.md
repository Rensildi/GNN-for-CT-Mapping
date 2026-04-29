# FMCIB — One-Slide Reference

Suggested slide title: **FMCIB — Foundation Model for Cancer Imaging Biomarkers**

---

## Background

- **Paper:** Pai et al. 2024, *Nature Machine Intelligence* — "Foundation model for cancer imaging biomarkers." AIM-Harvard / Aerts lab.
- **Code + weights:** [AIM-Harvard/foundation-cancer-image-biomarker](https://github.com/AIM-Harvard/foundation-cancer-image-biomarker). Pretrained weights on Zenodo (~738 MB), freely downloadable.
- **Pretraining objective:** **SimCLR** — contrastive self-supervised learning. No labels used during pretraining; the model learns to place similar nodule patches near each other in feature space.
- **Pretraining data:** ~11,000 lesion-containing CT scans across multiple tumor types and institutions — cancer-specific, unlike generic 3D CNNs trained on segmentation data.

## Architecture

- **Backbone:** 3D ResNet-50 with `widen_factor = 2` ("wide 3D ResNet-50"), single-channel input, `conv1_t_stride = 2`. The segmentation / classification head is stripped — only the trunk is released.
- **Input:** **50³-voxel** patches at isotropic 1 mm³ spacing, centered on the lesion coordinate, HU-normalized as `(HU + 1024) / 3072`.
- **Output:** **4096-D** global-pooled feature vector per patch (twice Med3D's 2048-D because of `widen_factor = 2`).
- **Deployment:** distributed as a frozen feature extractor via `fmcib.models.fmcib_model()`; patch preprocessing via `fmcib.preprocessing.get_transforms()`.

## Why it matters for this project

- **Cancer-specific pretraining + contrastive objective** give FMCIB clustered nodule embeddings out of the box — exactly the inductive bias a similarity-graph GNN wants.
- **In our Experiment 3:** FMCIB × image-only reaches **AUC 0.885** on radiologist-consensus labels and **0.887** on the pathology-confirmed subset — **+0.27 AUC over Med3D image-only** (p = 0.0312, 5/5 folds). It is the *only* image encoder in our grid whose image-only performance generalizes between radiologist and pathology labels.

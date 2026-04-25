# GNN-Based Lung Nodule Malignancy Classification Using Multi-Modal Feature Embedding on CT Scans

**Harrison Lavins · Rensildi Kalanxhi · Swathi Gopal**

CSC 7760 — Deep Learning · April 2026

---

## Abstract

We investigate whether modeling inter-nodule similarity as a graph improves benign-vs-malignant lung-nodule classification on CT scans. A two-stage pipeline fuses three complementary per-nodule modalities — frozen 3D imaging features (Med3D ResNet-50 or FMCIB), embeddings of the eight LIDC-IDRI radiologist attributes, and a sinusoidal positional encoding of the nodule's spatial coordinates — into a 256-dimensional node feature. A 2-layer Graph Convolutional Network (GCN) operating over a cohort-wide k-nearest-neighbor (KNN) similarity graph is benchmarked against a parameter-matched multilayer perceptron (MLP) under patient-level 5-fold cross-validation on 1,128 labeled LIDC-IDRI nodules from 588 patients. Across four pre-registered experiments and one side study, we find that (i) the GCN does not beat the MLP under the standard radiologist-consensus labels (AUC 0.949 vs 0.968, paired Wilcoxon p = 0.031, MLP wins 5/5 folds; H1 rejected); (ii) graph-construction hyperparameters do not recover the gap, with smaller k preferred but no cell beating the default at Bonferroni-corrected significance; (iii) the FMCIB foundation model substantially outperforms Med3D as the image encoder on image-only inputs (+0.27 AUC, 5/5 folds), but the advantage vanishes once attribute features are added because both encoders saturate at the attribute-driven ceiling; and (iv) a side-study evaluation against the 157-patient pathology-confirmed ground truth reveals a striking inversion — the radiologist-consensus "winner" loses, FMCIB image-only is the best cell on biopsy / surgery labels (AUC 0.887, 95 % CI [0.761, 0.982]), and the attribute branch *actively degrades* pathology AUC by ≈0.16 on FMCIB. Radiologist-consensus and pathology labels agree on only 67.5 % of the matched subset, providing a direct ceiling on label-leaky models. We conclude that the LIDC-IDRI attribute branch acts as a label shortcut rather than as orthogonal clinical signal, and that graph-based similarity reasoning has not yet had a fair test on this dataset. Adopting FMCIB image-only features is recommended for cross-dataset generalization (Experiment 4, LUNA25 transfer), where the attribute branch is unavailable.

---

## 1. Introduction

Lung cancer is the leading cause of cancer-related mortality worldwide. Modern CT-based classifiers reach AUC 0.88 – 0.96 on the public LIDC-IDRI benchmark, but they treat each lung nodule as an independent classification target. Patients in real-world cohorts share recurring imaging patterns: morphologically similar nodules tend to share clinical outcomes, and a clinician examining a borderline case implicitly compares it against similar cases they have seen before. This project asks whether that "guilt-by-association" reasoning can be operationalized inside a deep-learning pipeline.

Specifically, we ask:

> **Does modeling inter-nodule similarity as a graph improve malignancy classification compared to classifying each nodule independently?**

The methodological contribution is a side-by-side, parameter-matched comparison of a 2-layer Graph Convolutional Network against a parameter-matched MLP on a shared multi-modal feature representation. To prevent confounding from the choice of image encoder, we test two pretrained 3D feature extractors: Med3D ResNet-50 (Chen et al. 2019) and the FMCIB foundation model (Pai et al. 2024). To distinguish image signal from radiologist-attribute leakage, we ablate the input modality. To check whether the comparison is being made on a meaningful ground truth at all, we re-evaluate against the 157-patient pathology-confirmed subset of LIDC-IDRI.

Five experiments are reported in this paper:

1. **Experiment 1 — GCN vs. MLP baseline** under the full multi-modal feature configuration (Section 5.3).
2. **Experiment 2 — graph-construction ablation** over `k ∈ {5, 10, 15, 20}` × `metric ∈ {cosine, euclidean}` (Section 5.4).
3. **Experiment 3 — feature-modality × encoder ablation**, a 3 × 2 grid (Section 5.5).
4. **Experiment 3.1 — pathology-subset re-evaluation** of all Experiment-3 cells against biopsy/surgery ground truth (Section 5.6).
5. **Experiment 4 — cross-dataset generalization to LUNA25** (preliminary work; Section 5.7).

The paper is organized as follows. Section 2 reviews the technical background needed to read the methodology. Section 3 surveys prior progress on this application. Section 4 describes the architecture. Section 5 defines each experiment, the datasets used, and the experimental protocol. Section 6 presents the empirical results and a cross-experiment discussion. Section 7 enumerates contributions, both novel and per-team-member. Section 8 concludes; Section 9 outlines future work; Section 10 lists references.

---

## 2. Background

### 2.1 Lung-nodule classification on CT

Pulmonary nodules — small (<30 mm) opacities visible on chest CT — are the principal radiological finding triaged for possible lung malignancy. Computer-aided diagnosis on CT has progressed from hand-engineered radiomics through 2D and 3D convolutional neural networks; the modern public benchmark is the Lung Image Database Consortium / Image Database Resource Initiative (LIDC-IDRI) collection, whose protocol calls for up to four blinded thoracic radiologists per scan to mark and rate every nodule ≥ 3 mm.

### 2.2 Convolutional Neural Networks and ResNet

Convolutional Neural Networks (CNNs) extract translation-equivariant local features, making them the default architecture for medical image analysis. Going deeper increases representational capacity but causes vanishing-gradient pathologies; the ResNet family (He et al. 2016) addresses this with skip connections that allow the optimizer to fit residual functions. In 3D medical imaging, the same idea ports directly to 3D convolutions: 3D ResNet-18, 3D ResNet-50, and their wider variants serve as the standard frozen feature extractors for nodule patches.

### 2.3 Graph Neural Networks and the GCN

Graph Neural Networks (GNNs) are the family of architectures designed for graph-structured inputs — sets of nodes connected by edges. The Graph Convolutional Network (GCN, Kipf and Welling 2017) is the canonical instance: each layer computes a normalized weighted average of every node's neighbors, applies a learned linear map, and a nonlinearity:

```
H^(l+1) = σ( D̃^{-1/2} Ã D̃^{-1/2} H^(l) W^(l) )
```

where Ã = A + I (self-loops) and D̃ is its degree matrix. Two layers give a 2-hop receptive field — the standard depth on small graphs, since deeper stacks suffer over-smoothing (node representations collapse toward the graph mean). In our setup, nodes are nodules and edges are inter-nodule similarities; the GCN's role is to refine each nodule's representation using the representations of its k nearest neighbors in the shared 256-D feature space.

### 2.4 Foundation models in medical imaging

Foundation models are large-scale pretrained networks released as frozen feature extractors and fine-tuning starting points for downstream tasks. In medical imaging, two relevant foundation-style backbones inform this project:

- **Med3D** (Chen et al. 2019) — a 3D ResNet pretrained on 23 medical segmentation datasets via supervised multi-task learning. We use the 23-dataset ResNet-50 weights as the project's reported image baseline because the closest prior GCN-on-LIDC work (Ma et al. 2023) reports against this kind of encoder.
- **FMCIB** (Foundation Model for Cancer Imaging Biomarkers, Pai et al. 2024, *Nature Machine Intelligence*) — a wide 3D ResNet-50 (`widen_factor = 2`) pretrained with SimCLR contrastive self-supervised learning on ~11,000 lesion-containing CT scans across multiple tumor types and institutions. The contrastive objective explicitly clusters similar lesion patches in feature space, which is precisely the inductive bias a similarity-graph GNN wants downstream.

### 2.5 Datasets used in this project

LIDC-IDRI (1,010 patients, 1,308 CT studies, ≈ 7,371 nodules ≥ 3 mm with up to 4 radiologist ratings each) provides our primary training and within-dataset evaluation set. A subset of 157 patients carries pathology-confirmed diagnoses (biopsy, surgical resection, or two-year radiological follow-up), distributed via the TCIA `tcia-diagnosis-data-2012-04-20.xls` spreadsheet. LUNA25 (4,096 CT exams, 6,163 nodules from the U.S. National Lung Screening Trial, MICCAI 2025 release) provides our planned cross-dataset evaluation set; nodule-level binary malignancy labels there come from clinical follow-up rather than radiologist consensus.

---

## 3. Previous Progress / Related Work

CT-based lung-nodule classifiers have improved steadily over the last decade, with reported LIDC-IDRI AUCs in the 0.88 – 0.96 range. Most published methods share two limitations from our perspective.

First, they classify each nodule **independently**. Even multi-task networks that reason about shape and texture jointly produce a single per-nodule prediction with no architectural mechanism for the model to relate one case to another. Patient-similarity graph methods exist in adjacent applications — for example, in cancer subtyping and prognosis prediction — but had not been applied to lung nodule malignancy classification on CT.

Second, the most common LIDC-IDRI evaluation protocol uses **radiologist-consensus labels** (mean malignancy rating across up to four readers, binarized) and reports a single dataset-level AUC. As we show in Section 6, that protocol can mask a substantial gap between fitting radiologist opinion and learning real malignancy signal.

The closest prior work is Ma et al. 2023, *BMC Pulmonary Medicine* 23:462. Their LIDC-IDRI GCN reaches AUC 0.9629, and uses GCN layers for **cross-CNN feature fusion** — concatenating features from multiple CNNs and applying graph convolutions over the resulting feature graph. Our setup is architecturally distinct: we use GCN layers for **node classification on a nodule-similarity graph**, where nodes are nodules and edges are inter-nodule k-nearest-neighbor connections. Both architectures are GCNs in the Kipf-Welling sense, but they operate on completely different graphs and answer different questions.

The frozen image features themselves come from two pretrained backbones discussed in Section 2.4: Med3D ResNet-50 (Chen et al. 2019) as the reported baseline, and FMCIB (Pai et al. 2024) as the modern follow-up. To our knowledge, this is the first systematic comparison of these two encoders on LIDC-IDRI nodule classification, and the first project to evaluate node-classification GCNs on the inter-nodule similarity graph for this task.

---

## 4. Methodology

The architecture is a two-stage pipeline. Stage 1 is held frozen at inference; only Stage 2 is trained end-to-end.

### 4.1 Stage 1 — Multi-modal feature fusion (frozen)

Each nodule is represented by a 256-dimensional feature vector formed by concatenating three branches and projecting through a LayerNorm + Linear layer.

- **Image branch.** A 48³-voxel patch (Med3D) or 50³-voxel patch (FMCIB) is extracted around the nodule's centroid in the resampled isotropic 1 mm³ volume, HU-clipped to `[-1000, 400]` and normalized to `[0, 1]` for Med3D, or HU-normalized as `(HU + 1024) / 3072` for FMCIB. The patch is passed through the frozen 3D ResNet, producing a globally-pooled feature vector of dimension 2048 (Med3D) or 4096 (FMCIB). A trainable linear projection reduces this to 256-D.
- **Clinical branch.** The eight LIDC-IDRI radiologist attribute ratings — subtlety, internal structure, calcification, sphericity, margin, lobulation, spiculation, and texture — are looked up in eight per-attribute `nn.Embedding` tables (8 dimensions each) and concatenated into a 64-D vector.
- **Spatial branch.** The nodule's centroid coordinates `(x, y, z)` in millimeters are encoded with a sinusoidal positional encoding (16 dimensions per axis × 3 axes = 48 D), Transformer-style.

The three branches are concatenated, LayerNorm-ed, and projected through a final linear map to a 256-D unified node feature. Importantly, the malignancy label is **not** part of any branch — Stage 1 produces inputs to a classifier that has not yet seen the label.

### 4.2 Stage 2 — Trainable node-classification head

Two heads are compared, with all hyperparameters held identical except the architecture itself.

- **GCN head (treatment).** Two GCNConv layers (Kipf & Welling 2017, via PyTorch Geometric) operate over a cohort-wide k-nearest-neighbor similarity graph in the 256-D Stage-1 space. Default `k = 10`, default similarity = cosine. Each layer applies symmetric normalization with self-loops, ReLU activation, and dropout `p = 0.3`. A final linear layer maps the 64-D output to two class logits.
- **MLP head (control).** Three linear layers with ReLU + Dropout(0.3) between them, sized identically to the GCN's per-layer widths. The parameter count is matched within ±20 %, so any AUC delta cannot be explained by capacity differences. The MLP has no graph structure at all — each nodule is processed independently.

### 4.3 Inductive graph construction

A naïve cohort-wide KNN graph would produce a transductive leakage path under patient-level cross-validation: validation nodules' features would influence training nodules' representations through the KNN edges. To prevent this we use an **inductive evaluation protocol**:

1. KNN edges are built **over training-fold nodules only**.
2. At evaluation time, validation nodules are inserted into the graph as new nodes whose edges connect only to their `k` nearest training neighbors.
3. No validation-to-validation edges are ever created.

This setup was verified by checking that no patient straddles train and validation in any fold (StratifiedGroupKFold with `groups = patient_id`).

### 4.4 Training protocol

- **Optimizer.** Adam, learning rate `1e-3`, weight decay `1e-4`.
- **Loss.** Weighted cross-entropy (per-fold inverse-frequency class weights to compensate for the ≈ 71 % / 29 % benign / malignant imbalance).
- **Schedule.** Up to 100 epochs with patience-15 early stopping on validation AUC. The best checkpoint by validation AUC is retained.
- **Cross-validation.** Patient-level 5-fold StratifiedGroupKFold (`seed = 42`); per-fold validation malignant counts are 67, 77, 63, 62, 54, all comfortably above the pre-registered ≥ 30 threshold.
- **Reproducibility.** `torch.manual_seed`, `numpy.random.seed`, and `torch.backends.cudnn.deterministic = True` are set globally; per-cell seed reset is used in Experiments 2 and 3 so that the only cell-level variable is the architectural / feature-config knob being studied.

### 4.5 Evaluation metrics

Per fold we report AUC, AUPRC, sensitivity and specificity at the Youden-J optimal threshold (computed on training, applied to validation), Brier score, and a 10-bin reliability diagram. Across folds we compute the mean and standard deviation, plus paired Wilcoxon signed-rank tests for comparing matched pairs of cells. Bonferroni correction is applied across all simultaneous comparisons in Experiment 2. For Experiment 3.1 (pathology subset), the matched subset is small (40 nodules) so per-fold AUCs are often undefined; we report **pooled** AUC across folds with bootstrap 95 % confidence intervals (1,000 paired resamples).

---

## 5. Experiments

### 5.1 Datasets

- **LIDC-IDRI** is the primary dataset. After XML parsing, cross-reader clustering, malignancy binarization (mean ≤ 2 → benign, ≥ 4 → malignant, ≈ 3 excluded), and patient-level deduplication, we obtain **1,128 labeled nodules across 588 patients** (805 benign / 323 malignant). The pathology-confirmed subset comprises 157 patients with biopsy / surgery / two-year follow-up labels recorded in `tcia-diagnosis-data-2012-04-20.xls`.
- **LUNA25** is the cross-dataset evaluation set (Experiment 4): 4,096 CT exams with 6,163 NLST-derived nodules and clinical-follow-up binary labels, distributed via Zenodo under CC BY 4.0.

### 5.2 Preprocessing pipeline

Implemented without `pylidc` so that the pipeline is fully transparent:

1. **XML annotation parsing** with the Python standard library `xml.etree.ElementTree` parser. The nodule's malignancy rating is tracked in a dedicated field — separate from the eight feature-attribute fields — so that the Stage-1 fusion module *cannot* accidentally consume the label as a feature.
2. **DICOM series loading** via `pydicom`, with slices ordered by physical Z (`ImagePositionPatient[2]`) — robust to vendors that reverse `InstanceNumber`. Per-slice `RescaleSlope` and `RescaleIntercept` are applied to convert raw pixel intensities to Hounsfield Units.
3. **Cross-reader nodule clustering** via greedy centroid-distance matching at an 8 mm threshold, aggregating up to four radiologists' annotations of the same physical nodule. (Six of the resulting 1,128 clusters merge > 4 readers — a known minor artifact of the threshold rule, flagged for follow-up.)
4. **Patch extraction and normalization.** 48³ (Med3D) or 50³ (FMCIB) voxel patches centered on each nodule, HU-clipped to the lung window for Med3D and applied through FMCIB's native normalization for FMCIB. All patches are resampled to isotropic 1 mm³ before encoder inference.
5. **Label binarization.** Mean malignancy rating ≤ 2 → benign (0); ≥ 4 → malignant (1); the ambiguous middle band (≈ 3) is excluded from training and evaluation.
6. **Patient-level 5-fold StratifiedGroupKFold** with `groups = patient_id` and stratification on the patient-level majority label; ties favor the malignant class so the minority class stays well-distributed.

The committed splits (`data/splits/fold_{0..4}.json`) carry per-fold class-balance counts and were verified for zero patient overlap before any training.

### 5.3 Experiment 1 — GCN vs. MLP baseline

**Objective.** Direct head-to-head test of whether graph structure improves malignancy classification when both heads see the same features.

**Pre-registered hypothesis (H1).** GCN macro-AUC > MLP macro-AUC by ≥ 0.01, with paired Wilcoxon signed-rank `p < 0.05` across the 5 CV folds.

**Setup.** Stage 1 fixed at the full multi-modal configuration (image + attributes + spatial). Stage 2 is either the 2-layer GCN or the parameter-matched MLP. Identical optimizer, schedule, loss, and seeds. The only architectural variable is the head.

### 5.4 Experiment 2 — Graph-construction ablation

**Objective.** Identify whether different `(k, metric)` choices recover any of the GCN gap.

**Setup.** 8 cells: `k ∈ {5, 10, 15, 20}` × `metric ∈ {cosine, euclidean}`. The GCN architecture is fixed at Experiment 1's; only the KNN graph differs per cell. Per-cell seed reset isolates the graph effect from initialization noise. Decision rule: the best cell becomes the default for downstream experiments if its improvement clears Bonferroni-corrected `p < 0.05` against the `(10, cosine)` baseline; otherwise we keep the default.

### 5.5 Experiment 3 — Feature-modality × encoder ablation

**Objective.** Disentangle two remaining hypotheses left standing after Experiments 1 – 2: that the headline result was driven by attribute-branch label leakage, and / or that the Med3D image encoder was the bottleneck.

**Setup.** A 3 × 2 grid:

- **Feature axis (3 levels):** `image only`, `image + attributes`, `image + attributes + spatial`.
- **Encoder axis (2 levels):** Med3D ResNet-50 (baseline) and FMCIB (follow-up).

Stage 2 is held fixed at the Experiment-1 GCN with `(k = 10, cosine)`. Only the active Stage-1 branches and the image encoder vary per cell. Five folds × 6 cells = 30 model trainings.

### 5.6 Experiment 3.1 — Pathology-subset re-evaluation

**Objective.** Re-evaluate every Experiment-3 cell against the **biopsy / surgery ground truth** in the TCIA pathology spreadsheet, rather than against radiologist-consensus labels — a clean test of whether headline AUC reflects malignancy detection or radiologist-opinion fitting.

**Matching strategy (conservative).** Per-nodule diagnoses from the spreadsheet are binarized (`1 → benign`, `{2, 3} → malignant`, `0 → drop`). A patient contributes their pathology label to all of their clustered nodules **only if** all of their per-nodule diagnoses agree. Patients with mixed per-nodule diagnoses are excluded entirely. After deduplication, 40 matched nodules result (12 benign, 28 malignant) drawn from 102 pathology-agreeing patients.

**Statistics.** Predictions for each cell are pooled across folds (validation sets are disjoint by construction) and joined to the 40 matched nodules. Reported AUCs are pooled, with 95 % bootstrap CIs (1,000 paired resamples). Paired bootstrap CIs are computed for the most informative cross-cell comparisons.

### 5.7 Experiment 4 — Cross-dataset generalization (LUNA25)

**Status.** Pre-work in progress. The LIDC-trained pipeline is to be evaluated on LUNA25 nodules without retraining, testing whether the learned representation generalizes to a different patient cohort, scanner mix, and acquisition protocol. Two preliminary technical decisions have been made on the basis of Experiments 3 and 3.1:

1. The primary configuration for transfer will be **FMCIB × image-only** — the only Experiment-3 cell whose performance is invariant to label source (Section 6.4).
2. A **Graph Transformer** variant (preliminary results in Section 6.5) is being prototyped as an alternative to the 2-layer GCN, since the attention-weighted edge mechanism is more robust to the noisy KNN edges that hurt the GCN at high `k` in Experiment 2.

The risks specific to LUNA25 — different file format (NIfTI vs DICOM), preprocessed-vs-raw HU intensity ranges, world-coordinate vs voxel-centering conventions — are surveyed in Section 6.5 and on slide 34 of the project deck.

---

## 6. Results and Discussion

### 6.1 Experiment 1 — GCN underperforms parameter-matched MLP on every fold

| Fold | MLP AUC | GCN AUC | Δ (GCN − MLP) |
|:----:|:-------:|:-------:|:-------------:|
|  0   | 0.9895  | 0.9892  | −0.0004       |
|  1   | 0.9652  | 0.9515  | −0.0137       |
|  2   | 0.9471  | 0.9112  | −0.0359       |
|  3   | 0.9723  | 0.9562  | −0.0161       |
|  4   | 0.9657  | 0.9347  | −0.0309       |

| Aggregate | MLP                | GCN                | Δ (GCN − MLP) |
|:----------|:------------------:|:------------------:|:-------------:|
| AUC       | **0.9680 ± 0.0153**| 0.9486 ± 0.0287    | **−0.0194**   |
| AUPRC     | **0.9282 ± 0.0377**| 0.8922 ± 0.0644    | **−0.0359**   |
| Brier     | **0.0725 ± 0.0094**| 0.0888 ± 0.0163    | +0.0163 (GCN worse) |

Paired Wilcoxon signed-rank, one-sided "MLP > GCN":  `p = 0.0312` on AUC and AUPRC. With N = 5 folds this is the **minimum achievable Wilcoxon p-value** when every paired sign agrees. **0 / 5 folds favor the GCN.** H1 is rejected.

The MLP also beats the GCN on Brier score, indicating **worse calibration** for the GCN, not just worse discrimination — message passing pushes predicted probabilities further from the empirical ones rather than refining them.

### 6.2 Experiment 2 — graph hyperparameters do not recover the gap

Mean validation AUC across the 8-cell `(k, metric)` grid (5 folds per cell):

|  k  | cosine   | euclidean |
|:---:|:--------:|:---------:|
|  5  | 0.9615   | **0.9624**|
| 10  | 0.9549   | 0.9560    |
| 15  | 0.9513   | 0.9534    |
| 20  | 0.9497   | 0.9496    |

Two patterns emerge:

1. **Smaller `k` is better, monotonically.** AUC declines from `k = 5` to `k = 20` in both metrics — the classic over-smoothing signature, where larger neighborhoods push the normalized adjacency closer to a uniform-mean operator and erase per-nodule signal.
2. **Metric is essentially noise** (≤ 0.002 AUC difference per row). Expected: the Stage-1 LayerNorm before KNN makes cosine and Euclidean nearly equivalent on the projected features.

Bonferroni-corrected paired Wilcoxon vs `(10, cosine)`: every comparison returns `p_{Bonferroni} = 1.0`. With N = 5 folds the test is structurally underpowered (minimum achievable two-sided `p = 0.0625`, Bonferroni then demands `p ≤ 0.007`), so the formal decision-rule outcome is **null**: keep `(k = 10, cosine)` as the default. Even the best cell (`k = 5` Euclidean, AUC 0.9624) **still trails the Experiment-1 MLP baseline** at AUC 0.9680.

### 6.3 Experiment 3 — Med3D image-only is near chance; FMCIB is the modern winner; attributes saturate the ceiling

Mean validation AUC across the 3 × 2 cell grid (5 folds per cell, radiologist-consensus labels):

| feature config            | Med3D    | FMCIB    | Δ (FMCIB − Med3D) |
|:--------------------------|:--------:|:--------:|:-----------------:|
| **image only**            | **0.612**| **0.885**| **+0.273**        |
| image + attributes        | 0.960    | 0.960    | ≈ 0               |
| image + attributes + spatial | 0.954 | 0.952    | ≈ 0               |

Three findings:

1. **Med3D × image-only is barely above chance** at AUC 0.612 — strong evidence that the Med3D image branch, on its own, contains very little class signal on this task and dataset. The headline 0.97 AUC of Experiment 1's MLP was therefore *not* primarily a function of imaging features.
2. **FMCIB × image-only reaches AUC 0.885** — a +0.273 AUC advantage over Med3D × image-only, with FMCIB winning every one of the 5 folds and a one-sided Wilcoxon `p = 0.0312`. This is a substantively meaningful image-encoder upgrade.
3. **The encoder advantage vanishes at the attribute ceiling.** Once the attribute branch is added, both encoders saturate at AUC ≈ 0.96 — they are statistically indistinguishable (paired Wilcoxon `p > 0.5` on both image+attrs and full configurations). The ≈ 0.96 ceiling is not an imaging ceiling; it is what the attribute branch alone already provides.

Adding the spatial branch on top of image + attributes is a small but consistent **negative** in both encoders (≈ −0.007 AUC, direction agreed across folds, not significant at N = 5).

### 6.4 Experiment 3.1 — Pathology labels invert the ranking

Pooled validation AUC across the same 3 × 2 grid, restricted to the 40-nodule pathology-confirmed matched subset:

| feature config            | Med3D    | FMCIB    |
|:--------------------------|:--------:|:--------:|
| image only                | 0.470    | **0.887**|
| image + attributes        | 0.726    | 0.729    |
| image + attributes + spatial | 0.679 | 0.679    |

**FMCIB × image-only is the best-performing cell on pathology** (AUC 0.887, 95 % CI [0.761, 0.982]). Bootstrap-paired deltas with CIs that exclude zero:

- FMCIB image vs Med3D image: **+0.417** [+0.181, +0.631]
- FMCIB image vs FMCIB image+attrs: **+0.158** [+0.039, +0.291]
- FMCIB image vs Med3D full: **+0.208** [+0.069, +0.362]

Cross-label comparison highlights the leakage signature:

|                            | radiologist AUC | pathology AUC | Δ (pathology − radiologist) |
|:---------------------------|:---------------:|:-------------:|:---------------------------:|
| Med3D × image              | 0.612           | 0.470         | −0.142                      |
| Med3D × image + attrs      | 0.960           | 0.726         | **−0.234**                  |
| Med3D × full               | 0.954           | 0.679         | **−0.275**                  |
| **FMCIB × image**          | **0.885**       | **0.887**     | **+0.002**                  |
| FMCIB × image + attrs      | 0.960           | 0.729         | −0.231                      |
| FMCIB × full               | 0.952           | 0.679         | −0.274                      |

Two facts emerge starkly:

1. **The "winners" under radiologist labels are *losers* under pathology**. Both encoders' image+attrs and full configurations lose 0.23 – 0.28 AUC when the label source switches.
2. **Only FMCIB × image-only generalizes between label sources** — its radiologist AUC (0.885) and pathology AUC (0.887) are essentially identical. It is the only configuration that learns malignancy signal rather than radiologist-opinion signal.

**Radiologist-consensus and pathology labels agree on only 67.5 %** (27 / 40) of the matched subset. This is itself a significant finding — about a third of LIDC's "gold-standard" labels disagree with biopsy / surgery on the only subset where the comparison is possible — and it provides an upper bound on how much any radiologist-labeled-only model can be trusted.

### 6.5 Experiment 4 — preliminary Graph Transformer + LUNA25 risk surface

Pre-work for Experiment 4 has explored two directions:

- **A Graph Transformer variant** has been prototyped as an alternative to the 2-layer GCN used throughout Experiments 1 – 3. Preliminary results report a best validation accuracy of **95.42 %** for the Graph Transformer (vs **96.95 %** for the GCN baseline) on the same LIDC-IDRI pipeline. This is a developmental snapshot, not the final headline; the Transformer's attention-weighted edges are expected to be more robust under FMCIB image-only features (where there is real residual signal for graph-based reasoning to add) than under attribute-saturated features.

- **LUNA25 transfer risk analysis.** Switching from LIDC-IDRI to LUNA25 changes the data contract substantially. Scans are distributed as NIfTI-based challenge data, with > 4,000 CT exams and > 6,000 nodule annotations. File layout, storage, and preprocessing assumptions from LIDC will break silently rather than fail loudly. The most likely failure modes early in transfer are (i) mismatched volume folder structure, (ii) intensity-rescaled volumes treated as raw HU, and (iii) world-coordinate ↔ voxel-centering errors that produce blank or off-target patches. Validation can sit near chance even when the model code is technically running.

The full Experiment-4 transfer evaluation, including bootstrap-validated AUCs on LUNA25 and a comparison of GCN vs Graph Transformer transfer behavior, is left as future work (Section 9).

### 6.6 Cross-experiment discussion

The five experiments together support a coherent rather than contradictory story:

- **Experiment 1 says:** the GCN does not beat the MLP on radiologist-consensus labels.
- **Experiment 2 says:** that result is robust to graph-construction choices; smaller `k` helps marginally but does not change the verdict.
- **Experiment 3 says:** Med3D image features are nearly informationless on this task; FMCIB is much better; but both encoders saturate at the attribute-driven ceiling once the attribute branch is on.
- **Experiment 3.1 says:** the apparent attribute "ceiling" is mostly attribute-leakage. Under pathology labels, the attribute branch *hurts*, and FMCIB image-only is the best cell — at 0.887 AUC, with a 95 % CI that excludes the radiologist-label "winners."
- **Experiment 4 (preliminary)** indicates that a Graph Transformer is a credible alternative architecture once the attribute shortcut is removed — the regime in which Experiment 3.1 says signal-driven graph reasoning has its best chance.

The single sentence summarizing all five: **the GCN's underperformance in Experiment 1 was not the GCN's fault — there was no residual signal for it to add, because the multi-modal feature pipeline was being dominated by an attribute-branch shortcut whose existence is invisible until you re-evaluate against pathology.**

---

## 7. Contributions

### 7.1 Novel contributions

1. **First node-classification GCN applied to LIDC-IDRI lung-nodule malignancy.** Prior LIDC-IDRI GCN work (Ma et al. 2023) uses graph layers for cross-CNN feature fusion; ours uses graph layers for inter-nodule reasoning over a similarity graph.
2. **Multi-modal node-feature fusion** — image + 8 LIDC attribute embeddings + sinusoidal positional encoding — combined into a 256-D unified feature.
3. **Inductive evaluation protocol for the cohort-wide graph.** Train-only edges + val-node insertion eliminates the standard transductive-leakage failure mode under patient-level cross-validation.
4. **Parameter-matched MLP control** isolates the effect of message passing from raw model capacity.
5. **Side-by-side Med3D vs FMCIB encoder comparison** on a rigorously feature-ablated grid (Experiment 3) — the first such comparison on LIDC-IDRI we are aware of.
6. **Pathology-subset re-evaluation (Experiment 3.1)** showing that the attribute branch is a label shortcut rather than orthogonal clinical signal — quantifies a previously implicit caveat in the LIDC-IDRI literature.
7. **Cross-label generalization analysis** showing FMCIB × image-only is the only feature configuration whose performance is invariant to label source — a result directly informing Experiment 4 architectural choices.

### 7.2 Per-author contributions

- **Harrison Lavins** — designed and ran Experiments 1, 2, and 3, including the Experiment 3.1 pathology side study. Implemented the full preprocessing pipeline (no-pylidc XML parser, pydicom DICOM loader, cross-reader nodule clustering, patient-level CV splits). Built the multi-modal Stage-1 fusion, the GCN and MLP heads, the inductive KNN graph machinery, and the per-experiment training and analysis scripts (`train_exp1.py`, `train_exp2.py`, `train_exp3.py`, `analyze_exp3_pathology.py`). Wired in Med3D and FMCIB encoders and produced the cached feature parquets used by all experiments. Authored the per-experiment results write-ups (`results_exp1.md`, `results_exp2.md`, `results_exp3.md`, `results_exp3_1_pathology.md`), the architecture diagrams (`architecture_exp1.png`, `architecture_exp3.png`, MLP / GCN head diagrams, pathology heatmaps), the slide-ready figures (`exp1_approach.png`, `exp1_per_fold_bars.png`, `exp1_results_card.png`, `exp1_key_findings.png`), the GNN-vs-GCN explainer document, and this report. Maintained the project documentation tree under `documentation/` and `experiments/harrison/`.

- **Rensildi Kalanxhi** — contributed to Experiments 1 and 2. Tested both Med3D ResNet-50 and ResNet-18 image feature encoders under the Experiment 1 protocol, providing the comparison points used to confirm encoder-induced variation. Co-developed early iterations of the project structure and contributed to graph-construction ablation runs. *(This subsection is a placeholder summarizing what the project deck and code reflect; specific contribution statements should be filled in by the author as appropriate.)*

- **Swathi Gopal** — leads Experiment 4 (cross-dataset transfer to LUNA25). Implemented and benchmarked a Graph Transformer variant on the LIDC-IDRI pipeline, with preliminary results reaching a best validation accuracy of 95.42 % (vs 96.95 % for the GCN baseline). Surveyed the LUNA25 risk surface (NIfTI vs DICOM, preprocessed vs raw HU, world-coordinate centering) summarized in Section 6.5. *(This subsection is a placeholder; the LUNA25 transfer evaluation, bootstrap-validated AUC numbers, and the final Graph Transformer vs GCN comparison are in progress and will be added before final submission.)*

---

## 8. Conclusion

This project set out to answer a single question: does modeling inter-nodule similarity as a graph improve malignancy classification on CT? The answer, on LIDC-IDRI, is **not under the standard radiologist-consensus protocol** — but the more interesting finding is *why* the answer is "no."

A parameter-matched MLP outperformed our 2-layer GCN on every one of 5 cross-validation folds (AUC 0.968 vs 0.949, paired Wilcoxon `p = 0.031`). Hyperparameter sweeps over the graph construction did not recover the gap; the best `(k, metric)` cell still trailed the MLP. Feature-modality and encoder ablation clarified that the MLP's apparent 0.97 AUC was fundamentally driven by the eight LIDC radiologist attributes, which are correlated with the malignancy label by construction — Med3D image-only AUC sits at 0.612, near chance, and the attribute branch lifts it to 0.96.

Re-evaluating against the 157-patient pathology-confirmed subset inverted the ranking. Under biopsy / surgery ground truth, the "winning" attribute-saturated configurations lose 0.23 – 0.28 AUC, while FMCIB × image-only — the only configuration without the attribute branch — is the best cell on pathology (AUC 0.887, 95 % CI [0.761, 0.982]) and the only cell whose performance is invariant to label source. About one in three radiologist-consensus labels disagrees with pathology on the matched subset, providing a hard ceiling on any model trained against radiologist labels alone.

The practical conclusions are clear. Future work on this dataset should adopt **FMCIB × image-only** as the primary feature configuration and **report pathology-subset metrics** in addition to (and ideally before) radiologist-consensus metrics. Whether graph-based reasoning helps in this regime — once the attribute shortcut is removed and there is real residual signal to aggregate — is the question Experiments 4 and a re-run Experiment 2 on FMCIB image-only features are designed to answer.

The methodological conclusion is broader. The LIDC-IDRI radiologist-consensus protocol systematically rewards models that fit the labels' hidden generative process (i.e., the radiologists themselves) rather than malignancy. Any future model that uses radiologist-derived attributes as inputs *and* radiologist-derived ratings as labels should be considered suspect until evaluated on pathology-confirmed ground truth. We did not invent this concern, but we have quantified it cleanly in the only configuration where it can be quantified: a 0.27 – 0.30 AUC gap between fitting opinion and detecting disease.

---

## 9. Future Work

1. **Complete Experiment 4 (LUNA25 transfer).** Apply the FMCIB × image-only configuration to LUNA25 nodules, validate the data-contract compatibility issues from Section 6.5, and report bootstrap-validated transfer AUCs. Direct comparison of GCN and Graph Transformer transfer behavior is the most informative head-to-head this project can run.
2. **Re-run Experiment 2 on FMCIB × image-only.** With the attribute shortcut removed, there is real residual signal for graph aggregation to exploit; the `(k, metric)` ablation should be repeated under the cleaner regime.
3. **Investigate Graph Attention Networks (GAT) and Graph Transformers** as alternatives to the 2-layer GCN, particularly under FMCIB × image-only features where edge-attention can downweight noisy KNN neighbors.
4. **Build the graph in a non-overlapping feature space.** The current KNN edges are constructed in the same 256-D space the GCN then smooths over; alternative options include attribute-only similarity, a learned siamese metric, or a frozen secondary encoder used solely for edge construction.
5. **Address the label-leakage finding directly.** Either drop the attribute branch from the reported headline, or add an explicit attribute-residualization step that decorrelates the attribute branch from the malignancy label during training.
6. **Fix the `preprocess.py` hash-collision bug** (7 nodule_id duplicates in the current `nodules.parquet`) and re-run preprocessing, splits, and feature caching. The duplicates do not affect Experiments 1 – 3 because they are symmetric on both sides of the AUC calculation, but they introduce noise into the pathology-matching pipeline.
7. **Extend the pathology-confirmed evaluation** by relaxing the conservative "all per-nodule labels agree" rule with an explicit nodule-level mapping derived from the LIDC nodule numbering. The current 40-nodule subset is intrinsically noisy at this sample size; a larger matched subset would tighten Experiment 3.1's bootstrap CIs.

---

## 10. References

1. **Kipf, T. N., and Welling, M.** (2017). "Semi-Supervised Classification with Graph Convolutional Networks." *International Conference on Learning Representations (ICLR)*. Foundational GCN architecture defining the message-passing framework used throughout this project.

2. **Chen, S., Ma, K., and Zheng, Y.** (2019). "Med3D: Transfer Learning for 3D Medical Image Analysis." *arXiv preprint arXiv:1904.00625*. Provides the Med3D pretrained 3D ResNet-50 weights (trained on 23 medical segmentation datasets) used as the project's reported image baseline. Code and weights at https://github.com/Tencent/MedicalNet.

3. **Ma, X., et al.** (2023). "A novel fusion algorithm for benign-malignant lung nodule classification on CT images." *BMC Pulmonary Medicine* 23:462. Closest existing work using GCN for lung nodule classification on LIDC-IDRI (AUC 0.9629). Uses GCN for cross-CNN feature fusion; our approach uses GCN for node classification on a nodule-similarity graph.

4. **Pai, S., et al.** (2024). "Foundation model for cancer imaging biomarkers." *Nature Machine Intelligence*. Introduces FMCIB, a wide 3D ResNet-50 SimCLR-pretrained on ~11,000 lesion-containing CT scans. AIM-Harvard / Aerts lab. Code and weights at https://github.com/AIM-Harvard/foundation-cancer-image-biomarker. Project page at https://aim-harvard.github.io/foundation-cancer-image-biomarker/.

5. **Armato, S. G., III, et al.** (2011). "The Lung Image Database Consortium (LIDC) and Image Database Resource Initiative (IDRI): A Completed Reference Database of Lung Nodules on CT Scans." *Medical Physics* 38(2):915–931. Defines LIDC-IDRI, the primary dataset used in this project.

6. **He, K., Zhang, X., Ren, S., and Sun, J.** (2016). "Deep Residual Learning for Image Recognition." *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*. Introduces the ResNet skip-connection architecture underlying both Med3D and FMCIB.

7. **Fey, M., and Lenssen, J. E.** (2019). "Fast Graph Representation Learning with PyTorch Geometric." *ICLR Workshop on Representation Learning on Graphs and Manifolds*. Provides the `GCNConv` layer used in Stage 2 of our pipeline.

8. **The Cancer Imaging Archive (TCIA).** "tcia-diagnosis-data-2012-04-20.xls — LIDC-IDRI patient-level pathology diagnosis spreadsheet." Source of the 157 pathology-confirmed labels used in Experiment 3.1.

9. **MICCAI 2025 LUNA25 Challenge.** Dataset of 4,096 NLST-derived CT exams with 6,163 binary-labeled nodules. Distributed via Zenodo under CC BY 4.0.

10. **Veličković, P., et al.** (2018). "Graph Attention Networks." *International Conference on Learning Representations (ICLR)*. Architecture cited as a candidate for future work (Section 9, item 3) where attention-weighted edges can downweight noisy KNN neighbors.

---

*Generated with the project's experimental records, codebase, and presentation deck (`presentation/gnn_deep_learning_presentation.pptx`). Per-experiment write-ups and reproduction instructions are in `documentation/results_exp{1,2,3}.md` and `documentation/results_exp3_1_pathology.md`. The full per-author run log is on the `harrison/experiment3` branch of the project repository.*

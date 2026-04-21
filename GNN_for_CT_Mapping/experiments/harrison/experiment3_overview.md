# Experiment 3 — Overview

Companion to [`figures/architecture_exp3.png`](figures/architecture_exp3.png). Describes the experimental design, why each axis is being probed, the hypotheses, and how to read each possible outcome.

![Experiment 3 architecture](figures/architecture_exp3.png)

---

## 1. What Experiment 3 is

A **3 × 2 ablation** of Stage 1, with everything downstream held fixed:

| | Med3D ResNet-50 *(baseline)* | FMCIB foundation model *(follow-up)* |
|:-:|:-:|:-:|
| **Image only** | cell 1 | cell 2 |
| **Image + attributes** | cell 3 | cell 4 |
| **Image + attributes + spatial** | cell 5 | cell 6 |

- **Feature-modality axis** (rows): which Stage-1 branches are turned on.
- **Encoder axis** (columns): which frozen 3D CNN populates the image branch.

Every cell uses the same 2-layer GCN head from Experiments 1 and 2, the same committed patient-level 5-fold CV splits, the same loss, and the same early-stopping schedule. Only Stage 1's input mix and image encoder change per cell — so any performance difference is attributable to exactly one of those two knobs.

The experiment is explicitly not about searching the full hyperparameter surface; it's a targeted test of two specific questions that Experiments 1 and 2 could not resolve.

---

## 2. Why this experiment, and why now

Experiments 1 and 2 both told the same story: **the MLP (0.968 AUC) beats the GCN (0.949 AUC) with every graph construction we can name**. That leaves two candidate explanations that are still standing, both pinned to Stage 1 rather than Stage 2:

1. **Label leakage through the attribute branch.** The 8 LIDC descriptive attributes (subtlety, sphericity, margin, lobulation, spiculation, texture, internal structure, calcification) are rated 1–5 by the same four radiologists who then rate malignancy 1–5 in the same reading session. Those attribute ratings are highly correlated with the label by construction. The MLP might not be "learning from the CT image" at all — it might be reading off the radiologist's own label via correlated attribute scores. If that's true, the GCN had no residual signal to add in the first place.
2. **Encoder ceiling.** Med3D (2019) is a reasonable 3D feature extractor but not state-of-the-art. If the image branch is signal-poor, it leaves the attribute branch's near-perfect shortcut as the dominant contributor — reinforcing the leakage pattern above. A modern cancer-specific foundation model (FMCIB, 2024) may produce more discriminative image features, which would shift the balance.

**Experiment 3 is the clean test for both.** The image-only cells (1 and 2) show what the image branch alone can do — the direct leakage diagnostic. The encoder axis (Med3D vs FMCIB) pinpoints whether the image branch's ceiling is the encoder or the task.

---

## 3. The three feature configurations

### Image only  —  rows 1
Only the image branch fires. Attribute and spatial branches are dropped (not zero-masked — actually excluded from the concat so the branch widths sum differently). The Stage 1 fusion projects the image-encoder output directly to 256-D.

**What it isolates:** how well the frozen 3D CNN's features alone separate benign from malignant on LIDC. No radiologist-attribute leakage possible.

### Image + attributes  —  rows 2
Image branch + 8 per-attribute embeddings concatenated. Spatial branch dropped.

**What it isolates:** the joint signal of image + attributes, minus anatomical-location priors. The gap versus row 1 is the attribute branch's contribution — *before* accounting for any geographic information.

### Image + attributes + spatial  —  rows 3
Full Stage 1 fusion as used in Experiments 1 and 2.

**What it isolates:** the full-configuration ceiling on this dataset. Reproduces Experiments 1 and 2's features for a direct comparison.

Keeping all three configurations in a grid (rather than only reporting the full one) is what lets us *attribute* the MLP's 0.968 AUC to specific input sources.

---

## 4. The two image encoders

### Med3D ResNet-50 — the baseline
- Tencent's `resnet_50_23dataset.pth`: 3D ResNet pretrained via Chen et al. 2019 on 23 medical segmentation datasets.
- 48³-voxel patches at isotropic 1 mm³; outputs a 2048-D globally-pooled feature after `layer4`.
- Kept as the reported baseline specifically for comparability with **Ma et al. 2023** (the closest prior GCN-on-LIDC work). Swapping it out for Experiment 3 would weaken that headline comparison — hence we *run both* encoders in parallel and keep Med3D in the main results tables even if FMCIB wins.

### FMCIB foundation model — the follow-up
- **FMCIB** (Foundation Model for Cancer Imaging Biomarkers, Pai et al. 2024, *Nature Machine Intelligence*). AIM-Harvard / Aerts lab.
- SimCLR-pretrained 3D ResNet-50 on ~11,000 cancer-relevant CTs, specifically designed to consume nodule-centered 50³ patches and emit a 4096-D vector.
- Why it matters: cancer-specific pretraining and a contrastive objective that already clusters "similar lesions" in the embedding space. If any modern encoder is going to beat Med3D on a lung-nodule task at the 3060's memory budget, it's FMCIB.
- **Patch-size caveat:** FMCIB expects 50³ voxels, not 48³. A fresh 50³ crop is extracted from the same nodule centroid (not a resize of the 48³ patch), so the features are comparable to FMCIB's training distribution.
- The trainable linear projection on the image branch absorbs the 4096-D → 256-D reduction, mirroring Med3D's 2048-D → 256-D.

---

## 5. Hypotheses

**H1-modality:** adding the attribute branch increases macro-AUC by ≥ 0.005 over image-only (paired Wilcoxon, p < 0.05). If true, attributes carry real signal. If *much larger* than 0.005, the "leakage" interpretation strengthens.

**H1-spatial:** adding the spatial branch on top of image + attributes does not degrade macro-AUC, and specifically helps on the pathology-confirmed subset where upper-lobe priors are meaningful.

**H1-encoder:** FMCIB beats Med3D by ≥ 0.01 macro-AUC in the winning feature config (paired Wilcoxon, p < 0.05).

These are individually one-sided expectations; the global decision rule is simpler: **report whichever feature configuration and encoder win on the pathology-confirmed subset, with the caveat that Med3D stays in the main tables for comparability.**

---

## 6. Pathology-confirmed subset  —  the primary eval axis

LIDC's headline labels are **radiologist consensus**, not pathology. That's the root cause of the leakage concern: if the features include inputs from the same radiologists who made the label, the model can't learn *more than* what those radiologists know, and any "leakage test" using their own labels is circular.

The pathology-confirmed subset is different:

- ~157 nodules across LIDC have **biopsy, surgical resection, or clinical-follow-up confirmation** recorded in `tcia-diagnosis-data-2012-04-20.xls`. These are the closest thing LIDC offers to genuine ground truth.
- Metrics restricted to this subset measure whether the model has learned something medically real, not just something that correlates with radiologist opinion.
- Because the subset is small (~157 spread across 5 folds), fold-wise statistics are noisy. We pool predictions across folds for the headline number and report per-fold variance separately.

**The pathology subset is the primary validation axis for Experiment 3.** Full-cohort metrics are reported as a sanity check, but any disagreement between the two is the finding worth writing up.

---

## 7. How to read each outcome

### Scenario A — image-only MLP / GCN at or near the full-configuration number
If row 1 (image only) hits ~0.95 AUC even approximately, the image branch *is* learning from the CT. The attribute branch adds less than it appears to from the full-configuration result, and the "leakage" hypothesis from Experiment 1 is weaker than expected. In this case the GCN's failure is genuinely structural (circular KNN, oversmoothing, high-baseline fragility) and future work should pivot to graph-architecture changes — GAT, different pretext embeddings, attention-weighted message passing.

### Scenario B — image-only falls to ~0.85 AUC, full-config recovers to ~0.95
The attribute branch is doing the heavy lifting, and it's doing so with signal that's correlated with the label by construction. The MLP's 0.968 AUC substantially reflects this leakage. Experiment 1's "GCN fails" result reframes as "there was no residual signal for a graph to add." Future work should either drop the attribute branch, adopt a learned embedding space for the graph (decorrelated from the label), or shift the primary metric to pathology-confirmed ground truth where the leakage path doesn't exist.

### Scenario C — FMCIB meaningfully beats Med3D (≥ 0.01 AUC in the winning row)
The image ceiling was an encoder ceiling. Adopting FMCIB as the reported encoder — while keeping Med3D numbers in the results table for comparability with Ma et al. 2023 — is warranted. Experiment 4's LIDC → LUNA25 transfer becomes more interesting because FMCIB's contrastive pretraining is known to transfer better across distributions than Med3D's supervised pretraining.

### Scenario D — FMCIB does not beat Med3D
Encoder choice is not the bottleneck. Attribute leakage (Scenario B) or graph structure are the live hypotheses. Proceed to Experiment 4 using the winning feature config but the original Med3D encoder.

### Scenario E — something breaks on the pathology subset
Performance on the pathology-confirmed subset differs markedly from the full-cohort number, with the pathology subset being much worse. This would be the strongest positive evidence that the full-cohort AUC is being inflated by label leakage — the model is learning radiologist opinion, not pathology. Expected if the leakage hypothesis is correct, but dramatic enough to be a finding in its own right.

---

## 8. What stays fixed — scope discipline

Everything downstream of Stage 1 is held constant across all six cells:

- CV splits: committed `data/splits/fold_{0..4}.json`, seed 42.
- Graph: k = 10, cosine similarity, inductive val-node insertion.
- GCN: 2-layer GCNConv → ReLU → Dropout(0.3) → Linear head.
- Optimizer: Adam (lr 1e-3, wd 1e-4), weighted cross-entropy.
- Early stopping: patience 15 on val AUC, up to 100 epochs.
- Per-cell seed reset (so only Stage 1 varies; Stage 2's random init is identical cell-to-cell).

Experiment 2's null result on graph construction is also respected: we keep `(k=10, cosine)` as the default rather than adopting the weakly-directional `k=5` winner, because the plan's rule says null results keep the default.

---

## 9. Compute budget

- **FMCIB feature extraction:** one-time, ~30 min on the RTX 3060 for all 1,128 nodules. New cache at `outputs/features/fmcib.parquet`.
- **Training:** 6 cells × 5 folds = 30 GCN runs. At Experiment 1's timing (a few minutes per fold-model on cached features), the full sweep fits comfortably in an hour of wall clock.
- The expensive step is therefore the FMCIB cache, not the sweep.

---

## 10. Deliverables after the run

- `outputs/predictions/exp3_{med3d,fmcib}_{image,image_attrs,full}_fold{0..4}.parquet` — per-nodule probabilities, one file per cell per fold.
- `outputs/predictions/exp3_summary.parquet` — per-cell AUC/AUPRC/Brier on full cohort and pathology subset.
- `experiments/harrison/figures/exp3_auc_heatmap.png` — 3 × 2 heatmap for the full cohort.
- `experiments/harrison/figures/exp3_pathology_heatmap.png` — 3 × 2 heatmap restricted to the pathology-confirmed subset (primary).
- `documentation/results_exp3.md` — the outcome, significance tests per axis, adopted feature config and encoder for Experiment 4.

The **pathology-subset heatmap is the headline plot** — that's the figure to lead the write-up with.

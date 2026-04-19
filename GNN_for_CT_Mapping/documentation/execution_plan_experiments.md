# Execution Plan — Experiments 1, 2 & 3

**Scope:** High-level plan for the first three experiments from `proposal/proposal_gnn_v4.md`:

1. **Experiment 1 — GCN vs. MLP baseline** on LIDC-IDRI.
2. **Experiment 2 — Graph construction ablation** over `k ∈ {5, 10, 15, 20}` and cosine vs. Euclidean similarity.
3. **Experiment 3 — Feature-modality × encoder ablation** — 3 feature configs × 2 encoders (Med3D, FMCIB), validated on the pathology-confirmed subset.

This document is a planning artifact, not a spec — file paths, config keys, and metric targets are load-bearing; exact implementation details are left to the person picking up each phase.

---

## 0. Shared Prerequisites

All three experiments depend on the same data, features, and splits. Build this infrastructure once in `GNN_for_CT_Mapping/src/` so each experiment reuses it.

### 0.1 Data access and preprocessing

- **Download LIDC-IDRI** via the NBIA Data Retriever per TCIA instructions; point `configs/paths.local.yaml` at the DICOM root.
- **Configure pylidc** (`~/.pylidcrc`) to find the DICOM root.
- **Nodule extraction** (`src/data/lidc_nodules.py`):
  - Iterate over all patients' `pl.Scan` objects; cluster annotations into nodules via `pl.annotations_to_nodules` (default IoU=0.5).
  - Keep nodules ≥3mm.
  - Compute per-nodule mean malignancy score over its 1–4 annotations.
  - Apply the binarization from `configs/default.yaml`: mean ≤ 2 → benign (0), mean ≥ 4 → malignant (1), else exclude. Log counts.
  - Persist a single `nodules.parquet` with: `patient_id`, `nodule_id`, list of annotation IDs, centroid `(x, y, z)` in mm, bounding box, per-annotation attributes (subtlety, sphericity, margin, lobulation, spiculation, texture, internal structure, calcification), and the binary label.
  - Flag the ~157 pathology-confirmed diagnoses from the `tcia-diagnosis-data-*.xlsx` file; reserve them as a held-out "hard eval" subset. Experiment 3's pathology-subset eval depends on this being reliable.

### 0.2 Feature extraction (frozen, cache once)

- **Image patches:** 48³-voxel patches centered on each nodule's centroid; HU-clip to `[-1000, 400]`, normalize to `[0, 1]`, resample to isotropic 1 mm³.
- **Image encoder (Med3D, primary):** Med3D ResNet-50 loaded from Tencent/MedicalNet pretrained weights; freeze all parameters; run inference in eval mode and cache the output feature vector per annotation to `outputs/features/med3d_resnet50.parquet`. This is the reported baseline encoder (chosen for comparability with Ma et al. 2023).
- **Image encoder (FMCIB, Experiment 3 only):** FMCIB features are extracted from 50³-voxel patches per the official preprocessing in `foundation-cancer-image-biomarker`; cache to `outputs/features/fmcib.parquet`. Do this extraction once here so Experiment 3 just loads the cache. See Section 3.4 for details.
- **Clinical branch:** each of the 8 attributes is an integer 1–6 (internal structure goes up to 4); index into small `nn.Embedding` tables (trained end-to-end in Stage 2).
- **Spatial branch:** `(x, y, z)` nodule centroid in mm → sinusoidal positional encoding with the same dim as Stage 2 expects.
- **Fusion:** concatenate branches, `LayerNorm`, project to `model.node_feature_dim = 256`.

### 0.3 Cross-validation splits

- **Patient-level 5-fold CV** (never split nodules from the same patient across folds). Write `data/splits/fold_{0..4}.json` containing `{patient_ids: [...]}` for train/val per fold.
- Use `sklearn.model_selection.StratifiedGroupKFold` with `groups=patient_id` and stratification on the patient's majority label.
- Record class-balance per fold in the split log.

### 0.4 Reproducibility and logging

- Seed (`training.seed = 42`) sets `torch`, `numpy`, `random`, and `torch.backends.cudnn.deterministic = True`.
- Log every run to a uniquely named subdir of `runs/` (TensorBoard) and `outputs/checkpoints/`. Include the git commit hash and effective config in a `run_metadata.json`.

### 0.5 Exit criteria for Section 0

- `nodules.parquet` built with logged benign/malignant/excluded counts.
- Frozen Med3D features cached for every annotation; re-running produces identical vectors (bit-for-bit).
- Frozen FMCIB features cached for every annotation; same reproducibility check.
- 5 CV folds saved; balance and patient overlap validated in a notebook.
- End-to-end smoke test: one forward pass through a dummy MLP and a dummy GCN on fold 0 completes without error.

---

## 1. Experiment 1 — GCN vs. MLP baseline

### 1.1 Objective

Test whether graph-based message passing over a nodule-similarity graph improves malignancy classification vs. an MLP that sees identical features but no graph structure.

### 1.2 Hypothesis

H1 (primary): GCN macro-AUC > MLP macro-AUC by ≥ 0.01 with a paired-fold Wilcoxon signed-rank p < 0.05 across the 5 CV folds.

H0: no significant difference — GCN's inductive bias doesn't help on LIDC.

### 1.3 Models

Both models consume the fused 256-dim node features from Section 0.2 (using Med3D image features).

**MLP baseline (`src/models/mlp.py`):**
- `Linear(256, 128) → ReLU → Dropout(0.3) → Linear(128, 64) → ReLU → Dropout(0.3) → Linear(64, 2)`.
- Parameter count target: within ±20 % of the GCN's parameter count so capacity isn't the explanation if GCN wins.

**GCN (`src/models/gcn.py`):**
- `GCNConv(256, 128) → ReLU → Dropout(0.3) → GCNConv(128, 64) → ReLU → Dropout(0.3) → Linear(64, 2)`.
- Use `torch_geometric.nn.GCNConv` with self-loops and symmetric normalization.
- Graph edges: KNN over fused features with **cosine similarity**, `k = 10` (the default from `configs/default.yaml`).

### 1.4 Graph construction at train time (non-obvious detail)

Patient-level CV + a cohort-wide KNN graph is easy to mis-implement. The safe plan:

- Build the KNN graph over **all train-fold nodules only** — excluding val/test nodules — to fit the GCN weights.
- At eval time, insert val/test nodules as new nodes whose edges connect to their k nearest training neighbors (no val/test → val/test edges, no training nodes getting new edges to unseen nodes). This is a standard inductive evaluation for cohort graphs.
- Document this clearly in the run log. An accidental transductive setup is the #1 failure mode and will inflate results.

### 1.5 Training protocol

- **Optimizer:** Adam, `lr = 1e-3`, `weight_decay = 1e-4`.
- **Loss:** weighted cross-entropy; `class_weights = auto` computes inverse-frequency weights per fold.
- **Epochs:** 100, early stop on val AUC with patience 15.
- **Batching:** one graph per step (the GCN) / minibatches of 256 nodes (the MLP). Both use the same shuffle RNG so epoch ordering is comparable.
- **Augmentation:** each epoch, stochastically sample one of the up to 4 radiologist annotations per nodule. This varies the attribute vector (and, in a later iteration, would re-extract the image patch).

### 1.6 Evaluation

Per fold, compute on the val nodules:

- **Discrimination:** AUC, AUPRC.
- **Operating point metrics** at the Youden-J optimal threshold on train: accuracy, sensitivity, specificity, F1.
- **Calibration:** Brier score and a 10-bin reliability diagram (saved to `figures/`).

Across folds: mean ± std; paired Wilcoxon signed-rank for GCN vs MLP on AUC and AUPRC.

**Hard subset:** report the same metrics restricted to the ~157 pathology-confirmed nodules (if any land in val folds).

### 1.7 Deliverables

- `experiments/<author>/notebooks/exp1_gcn_vs_mlp.ipynb` with figures.
- `outputs/predictions/exp1_{gcn,mlp}_fold{0..4}.parquet`.
- A one-page writeup in `documentation/results_exp1.md`: table of metrics, significance test result, and a 2–3 sentence interpretation tied back to H1/H0.

### 1.8 Risk register

- **Label leakage through clinical attributes.** The malignancy label is a radiologist rating; the 8 attributes come from the same raters. Experiment 3's image-only cell is what isolates learned imaging signal. Flag any claim about "learning from images" until Exp 3 confirms.
- **Tiny graph in per-patient splits.** If any experiment later switches to per-patient graphs, most LIDC patients have only 1–3 nodules and KNN with k=10 is ill-defined. Keep the cohort-wide graph; don't silently switch.
- **Class imbalance at the fold level.** Check each fold's val malignant count ≥ 30; if not, re-seed the fold assignment until it is.

---

## 2. Experiment 2 — Graph construction ablation

### 2.1 Objective

Identify the optimal neighborhood size `k` and similarity metric for the cohort-wide nodule-similarity graph. Decide whether cosine or Euclidean better reflects "meaningful" nodule similarity in the fused feature space.

### 2.2 Hypothesis

H1: there exists a `(k, metric)` combination that outperforms the default `(k=10, cosine)` by ≥ 0.005 AUC (paired fold, p < 0.05).

Secondary expectation: cosine > Euclidean because fused features are concatenated across heterogeneous branches with different natural scales, and cosine is scale-invariant.

### 2.3 Grid

8 cells: `k ∈ {5, 10, 15, 20}` × `metric ∈ {cosine, euclidean}`. For each cell:

- Re-use Experiment 1's frozen features and CV splits.
- Re-build the KNN graph under the new `(k, metric)` at each fold (train-only nodes for fitting; inductive insertion for val).
- Train the same 2-layer GCN architecture from Experiment 1.
- Fix all other hyperparameters at their defaults — this is a graph-construction ablation, not a joint sweep.

### 2.4 Evaluation

Same metrics as Experiment 1.

Visualizations:

- Heatmap of mean val AUC over `(k, metric)`.
- For the winning cell, plot degree distribution and edge-length distribution; compare to a random-graph baseline of matched average degree.

### 2.5 Decision rule

- If one cell wins significantly (paired Wilcoxon over folds, Bonferroni-corrected across the 8 cells), adopt it as the default `(k, metric)` for Experiment 3 and beyond.
- If no cell significantly beats `(10, cosine)`, keep the default and report null result.

### 2.6 Deliverables

- `experiments/<author>/notebooks/exp2_graph_ablation.ipynb`.
- `outputs/predictions/exp2_k{5,10,15,20}_{cosine,euclidean}_fold{0..4}.parquet`.
- `documentation/results_exp2.md` with the AUC heatmap and the adopted configuration.

### 2.7 Compute budget

8 cells × 5 folds × 100 epochs × one small GCN = well under 2 hours total on the 3060 once features are cached. The feature extraction (Section 0.2) dominates wall-clock cost and only runs once.

### 2.8 Risk register

- **Over-smoothing at high `k`:** large `k` makes the normalized adjacency closer to a uniform average; in a shallow GCN this matters less than in deep ones, but watch the val AUC curve for collapse.
- **Unit mismatch in Euclidean:** feature branches have different natural scales; `LayerNorm` at the end of fusion (Section 0.2) largely addresses this, but verify post-norm variances are comparable across branches before running the Euclidean cells.
- **Silent graph rebuild bugs:** cache the KNN indices to disk and hash them (`sha256` of sorted edge list) so re-running a cell with the same seed produces the same graph.

---

## 3. Experiment 3 — Feature-modality × encoder ablation

### 3.1 Objective

Two axes probed in a single 3×2 grid:

- **Feature modality:** which of the three node-feature branches (image / image+attributes / image+attributes+spatial) actually drive GCN performance.
- **Image encoder:** whether FMCIB — a cancer-specific 3D foundation model trained on ~11k CTs — outperforms Med3D as the image-branch encoder on this task.

Pathology-confirmed subset performance is the primary validation axis here, because LIDC's all-nodule labels are radiologist consensus rather than ground truth, and Experiment 3 is where "do the features align with real pathology?" gets a clean answer.

### 3.2 Hypotheses

- **H1-modality:** adding the clinical attributes branch improves macro-AUC by ≥ 0.005 over image-only (paired fold, p < 0.05).
- **H1-spatial:** adding the spatial branch on top of image+attributes improves macro-AUC by a non-negative margin; spatial is expected to help mainly on the pathology-confirmed subset (upper-lobe prior).
- **H1-encoder:** FMCIB macro-AUC ≥ Med3D macro-AUC by ≥ 0.01 in the winning feature config (paired fold, p < 0.05).

### 3.3 Grid

6 cells: `feature_config ∈ {image, image+attrs, image+attrs+spatial}` × `encoder ∈ {Med3D, FMCIB}`. For each cell:

- Reuse CV splits from Section 0.3.
- Use the winning `(k, metric)` from Experiment 2 (fall back to `(10, cosine)` if Exp 2 produced a null result).
- Use the same 2-layer GCN architecture from Experiment 1.
- Vary only the two ablation axes — fix all other hyperparameters at defaults.

### 3.4 FMCIB feature extraction (one-time, runs in Section 0.2)

- Use the official inference pipeline from `AIM-Harvard/foundation-cancer-image-biomarker`. Pip install the package or clone and run its `get_features` script.
- **Patch size:** FMCIB expects 50³ voxels at 1 mm³ isotropic spacing. Extract a fresh 50³ crop centered on the same nodule centroid used for Med3D — do NOT resize the Med3D 48³ patch to 50³; re-crop from the original volume to avoid interpolation artifacts. Document this in the feature-extraction log.
- **Output dimensionality:** 4096. The trainable linear projection into `node_feature_dim = 256` absorbs this; no other code change needed downstream.
- Cache to `outputs/features/fmcib.parquet`; re-running must be bit-for-bit reproducible.

### 3.5 Node-feature assembly per feature config

- **image:** image branch (from the chosen encoder) → linear projection → `LayerNorm` → node feature. Clinical and spatial branches are zero'd (or, preferably, simply omitted from the concat).
- **image+attrs:** image branch + clinical attribute embeddings → concat → `LayerNorm` → node feature.
- **image+attrs+spatial:** full fusion as in Section 0.2.

Keep the final `node_feature_dim = 256` for all three configs so the GCN architecture is fixed across cells.

### 3.6 Training protocol

Same as Experiments 1 & 2 (Adam, lr=1e-3, weight_decay=1e-4, 100 epochs, early stop patience 15, weighted CE, radiologist-sampling augmentation, inductive graph construction per Section 1.4).

### 3.7 Evaluation

Per fold, per cell:

- **All-nodule val metrics:** AUC, AUPRC, accuracy/sensitivity/specificity/F1 at Youden-J, Brier.
- **Pathology-confirmed subset metrics:** same metrics restricted to the ~157 pathology-confirmed nodules (if any land in val). This is the primary eval axis for Experiment 3.

Across folds:

- 2D heatmap (3 rows × 2 cols) of mean ± std macro-AUC for all-nodule and pathology subsets (two heatmaps).
- Paired Wilcoxon signed-rank tests:
  - Per modality axis (e.g., image+attrs vs image, within each encoder).
  - Per encoder axis (FMCIB vs Med3D, within each feature config).
- Bonferroni correction across the cell-level comparisons.

### 3.8 Decision rules

- **Modality:** whichever feature config wins on the pathology subset becomes the adopted configuration for Experiment 4.
- **Encoder:** if FMCIB wins, report FMCIB as the recommended encoder but **keep Med3D results in the main tables** so the Ma et al. 2023 comparability claim holds.
- **Attribute-leakage check:** if image-only + FMCIB ≈ image+attrs + Med3D (within 0.005 AUC), that's a strong signal the attribute branch was carrying label leakage rather than unique signal. Call this out in the writeup rather than hiding it.

### 3.9 Deliverables

- `experiments/<author>/notebooks/exp3_modality_encoder.ipynb`.
- `outputs/predictions/exp3_{med3d,fmcib}_{image,image_attrs,full}_fold{0..4}.parquet`.
- `documentation/results_exp3.md` with both heatmaps (all-nodule and pathology-subset), significance tables, and the adopted modality / encoder decisions for Experiment 4.

### 3.10 Compute budget

- 6 cells × 5 folds × 100 epochs × small GCN: ~1–2 hours on the 3060 once features are cached.
- FMCIB feature extraction: one-time, expect ~30 min on a 3060 for the full nodule set. Happens in Section 0.2.

### 3.11 Risk register

- **FMCIB patch-size mismatch.** FMCIB expects 50³, Med3D uses 48³. Take two separate crops from the original volume — don't resize. A resized 48³→50³ patch introduces interpolation artifacts that will make FMCIB look worse than it is.
- **Encoder feature-dim mismatch.** Med3D outputs a much smaller dim than FMCIB's 4096. Both are absorbed by the trainable linear projection to 256, but verify the post-projection activation statistics are comparable (z-score or per-encoder LayerNorm warmup). If FMCIB's projection starts far from Med3D's in activation scale, the early-epoch losses will be misleading.
- **Pathology-subset sample size.** Only ~157 pathology-confirmed nodules total, spread across 5 folds. Any single fold may have very few. If fold-wise pathology-subset AUC has wide confidence intervals, aggregate across folds (pooled prediction set) for the headline number and report the per-fold variance separately.
- **Attribute leakage revealed as a positive.** If the attribute branch shows inflated gains compared to image-only, it's not "the model learned." Treat it as diagnostic and adjust framing in the writeup.

---

## 4. Out of scope for this plan

- **Experiment 4** (LUNA25 cross-dataset generalization) — separate plan. Uses the winning modality and encoder from Experiment 3.
- **Architecture variants beyond 2-layer GCN** (e.g., GAT, GraphSAGE) — separate experiment. Worth considering once the three experiments in this plan have stable baselines.

---

## 5. Suggested ordering

1. Section 0 (shared prerequisites, including Med3D **and** FMCIB feature caching) — ~1 week of wall-clock, mostly data download + one-time feature extraction.
2. Experiment 1 — ~3–5 days including writeup.
3. Experiment 2 — ~2 days once Experiment 1's training loop is reusable.
4. Experiment 3 — ~3–4 days. Same training loop as Experiments 1 & 2; most of the time goes into the pathology-subset analysis and the two-axis writeup.

Commit cadence: after each numbered sub-section (0.1, 0.2, ...) the repo should be in a runnable state.

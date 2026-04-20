# Experiment 1 — Results

**Question:** Does a GCN head that aggregates over a KNN nodule-similarity graph beat a parameter-matched MLP on identical multi-modal features?

**Answer:** No. The MLP baseline wins across all 5 folds by a small but significant margin. **H1 is rejected.**

---

## Setup snapshot

- **Data:** 1,128 labeled nodules from 588 LIDC-IDRI patients (805 benign / 323 malignant).
- **Splits:** patient-level 5-fold StratifiedGroupKFold (`seed=42`), committed at `data/splits/fold_{0..4}.json`.
- **Features:** frozen Med3D ResNet-50 (Tencent/MedicalNet `resnet_50_23dataset.pth`) → 2048-D, projected to 256-D via the Stage 1 fusion module (image + 8 attribute embeddings + sinusoidal spatial encoding).
- **Stage 2 heads (parameter-matched):**
  - **MLP:** `Linear(256,128) → ReLU → Dropout(0.3) → Linear(128,64) → ReLU → Dropout(0.3) → Linear(64,2)`.
  - **GCN:** `GCNConv(256,128) → ReLU → Dropout(0.3) → GCNConv(128,64) → ReLU → Dropout(0.3) → Linear(64,2)` over a cohort-wide KNN graph (k=10, cosine, inductive val-node insertion).
- **Training:** Adam (lr 1e-3, wd 1e-4), weighted cross-entropy, 100 epochs with patience-15 early stop on val AUC.

## Per-fold val metrics

| Fold | MLP AUC | GCN AUC | Δ(GCN − MLP) | MLP AUPRC | GCN AUPRC | MLP Brier | GCN Brier |
|:----:|:-------:|:-------:|:------------:|:---------:|:---------:|:---------:|:---------:|
|  0   | 0.9895  | 0.9892  | −0.0004      | 0.9754    | 0.9751    | 0.0739    | 0.0719    |
|  1   | 0.9652  | 0.9515  | −0.0137      | 0.9561    | 0.9379    | 0.0652    | 0.0844    |
|  2   | 0.9471  | 0.9112  | −0.0359      | 0.8895    | 0.8276    | 0.0869    | 0.1095    |
|  3   | 0.9723  | 0.9562  | −0.0161      | 0.9260    | 0.8870    | 0.0630    | 0.0762    |
|  4   | 0.9657  | 0.9347  | −0.0309      | 0.8939    | 0.8336    | 0.0734    | 0.1020    |

**Folds where GCN beats MLP:** 0 / 5.

## Aggregate (mean ± std across folds)

| Metric | MLP                | GCN                | Δ (GCN − MLP) |
|:-------|:-------------------|:-------------------|:-------------:|
| AUC    | 0.9680 ± 0.0153    | 0.9486 ± 0.0287    | **−0.0194**   |
| AUPRC  | 0.9282 ± 0.0377    | 0.8922 ± 0.0644    | **−0.0359**   |
| Brier  | 0.0725 ± 0.0094    | 0.0888 ± 0.0163    | +0.0163 (GCN worse) |

## Significance

Paired Wilcoxon signed-rank (5 folds, GCN vs MLP):

| Metric | two-sided p | one-sided p (GCN > MLP) | one-sided p (GCN < MLP) |
|:-------|:-----------:|:-----------------------:|:-----------------------:|
| AUC    | 0.0625      | 1.0000                  | **0.0312**              |
| AUPRC  | 0.0625      | 1.0000                  | **0.0312**              |

At the plan's α=0.05 threshold, the **one-sided test that MLP > GCN is significant** on both AUC and AUPRC. With only 5 folds the best achievable two-sided Wilcoxon p is 0.0625 (all 5 signs same direction → minimum two-sided p), so the two-sided test is underpowered — but the direction, magnitude, and consistency are unambiguous.

## Verdict vs the plan's hypotheses

- **H1 (GCN beats MLP by ≥ 0.01 AUC, p < 0.05):** **rejected.**
- The observed effect is the opposite sign (GCN trails by 0.019 AUC, 0.036 AUPRC) and is consistent across every fold.
- Calibration (Brier) is also worse for the GCN.

## Why the graph hurt rather than helped

These are the leading hypotheses. Experiment 3 is the clean test for most of them.

1. **Attribute-driven label leakage is doing most of the work.** The 8 LIDC descriptive attributes are 1–5 ratings from the same four radiologists who assigned the malignancy score. Those attributes are already strongly correlated with the label — note that the MLP reaches AUC 0.97. In that regime there's very little residual signal for the graph to contribute, and the message-passing step mostly averages already-confident predictions with noisier neighbors. The image-only and image+attrs variants in Experiment 3 are the clean test.
2. **The KNN is built in the same feature space the GCN then operates on.** Edges connect nodules whose *features* are similar; the GCN then smooths over those same features. That's largely redundant with the MLP's first linear layer and adds variance without adding signal.
3. **Inductive val-node insertion at high AUC baselines is noisy.** Val nodes join the graph via edges to their k=10 nearest training neighbors. When the MLP is already right 96–99% of the time, mis-specified neighbors for a small number of val nodes degrade predictions more than correct neighbors help.
4. **Over-smoothing is unlikely at 2 layers** but worth ruling out in Experiment 2 at higher `k` values — large `k` pushes the adjacency closer to a uniform mean.

## Sanity checks that passed

- No train/val patient leakage verified at split build time.
- Val malignant counts per fold: 67, 77, 63, 62, 54 (plan requires ≥ 30).
- Feature extraction deterministic (verify_med3d determinism check passed before the run).
- MLP and GCN trained with identical optimizer, schedule, seed, feature matrix, and early-stop criterion; only the head differs.
- Med3D features mean 0.21 / std 0.27 — consistent with the verify smoke test on random input, i.e. the encoder did not collapse.

## Artifacts from this run

- `outputs/predictions/exp1_{mlp,gcn}_fold{0..4}.parquet` — per-nodule probability + label for every val nodule.
- `outputs/predictions/exp1_summary.parquet` — per-fold AUC/AUPRC/Brier table.
- `runs/harrison_exp1/fold{i}_{mlp,gcn}/` — TensorBoard scalars (train loss, val AUC) per epoch.

## Recommended next moves

- **Experiment 2** (graph construction ablation) — cheap, reuses the same cached features. If no (k, metric) cell recovers the GCN, it's strong evidence the failure is structural, not a bad hyperparameter.
- **Experiment 3** (feature modality × encoder ablation) — the critical test. Image-only runs will quantify how much the MLP's 0.97 AUC is due to radiologist-attribute signal vs. learned imaging features. A large drop would confirm hypothesis (1) above.
- **Pathology-confirmed subset eval** — our 0.97 AUC is against radiologist-consensus labels, not pathology. The ~157 pathology-confirmed nodules are the cleaner ground truth; any follow-up should report metrics on that subset alongside the full-cohort number.

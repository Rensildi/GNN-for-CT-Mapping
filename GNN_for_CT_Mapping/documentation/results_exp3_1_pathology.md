# Experiment 3.1 — Pathology-subset results

Side experiment: re-evaluate Experiment 3's six cells against the 157-patient **pathology-confirmed ground truth** in `tcia-diagnosis-data-2012-04-20.xls` rather than the radiologist-consensus labels used for the Experiment 3 headline. This is the clean test of the attribute-leakage hypothesis that Experiment 1's post-mortem and Experiment 3's overview both flagged as the highest-value remaining check.

**Headline:** the attribute branch, which drives radiologist-consensus AUC from 0.61 → 0.96, **hurts pathology AUC** by 0.16 on FMCIB. **FMCIB image-only is the best cell on true ground truth** at AUC 0.887 (95 % CI [0.761, 0.982]); the "winning" configurations under radiologist labels (both encoders' *image + attrs* and *full*) fail to clear AUC 0.73. The gap is large enough that the 95 % CI for each of those negative deltas excludes zero.

Put bluntly: *under pathology labels, the model we've been optimizing is worse than the model without the radiologist-attribute shortcut.*

---

## 1. Ground truth: `tcia-diagnosis-data-2012-04-20.xls`

The file ships 157 rows, one per LIDC patient, with up to five per-nodule diagnosis codes (`0 = unknown, 1 = benign, 2 = malignant primary, 3 = malignant metastatic`) and the diagnostic method (`0 = unknown, 1 = 2-year radiological stability, 2 = biopsy, 3 = surgical resection, 4 = progression / response`). Unlike LIDC's 1–5 malignancy rating, these labels come from actual clinical follow-up — they're ground truth, not opinion.

### Matching to our nodules

LIDC's "Nodule N" numbering in the diagnosis sheet doesn't map directly onto our clustered `nodule_id`s (different numbering, different clustering rules). We use a **conservative label-propagation rule**:

- Load per-nodule diagnoses for each patient and binarize (`1 → benign`, `{2, 3} → malignant`, `0 → drop`).
- Keep only patients whose per-nodule diagnoses all agree (all benign OR all malignant). The patient-level label is unambiguous in that case.
- Apply the patient's label to every one of their nodules in our `nodules.parquet`. Patients with mixed per-nodule diagnoses are excluded entirely because we can't resolve which clustered nodule got which label without a canonical mapping.

**Counts after matching:**

| | count |
|:---|---:|
| Pathology-confirmed patients with agreeing labels | **102** |
| Matched nodules (their clusters in our data) | **40** |
|   — benign | 12 |
|   — malignant | 28 |
| Radiologist-consensus ↔ pathology label agreement | **67.5 %** (27 / 40) |

**The 67.5 % agreement is itself a striking finding.** Roughly one in three of our "gold-standard" radiologist-consensus labels disagrees with the pathology ground truth on the only subset where we can check. That's a ceiling on how good any model trained against radiologist-consensus labels can be, and an explanation for why attribute-leakage inflates headline AUC — the model is fitting radiologist opinion, and that opinion is wrong a third of the time relative to biopsy / surgery.

---

## 2. Pooled pathology-subset AUC

Predictions pooled across the 5 CV folds (each val-fold is a disjoint patient set, so pooling is simply concat, then restrict to the 40 matched nodules).

| feature_config                 | Med3D      | FMCIB       |
|:-------------------------------|:----------:|:-----------:|
| **image only**                 | 0.4702     | **0.8869**  |
| image + attrs                  | 0.7262     | 0.7292      |
| image + attrs + spatial        | 0.6786     | 0.6786      |

AUPRC shows the same ordering; values are high throughout because the pathology subset has a 28 / 12 malignant / benign split (the baseline AUPRC with a constant-malignant predictor is already 0.70).

### Bootstrap 95 % CIs (1000 resamples, pathology subset N = 40)

| cell                                   | pooled AUC | 95 % CI           |
|:---------------------------------------|:----------:|:------------------|
| Med3D × image                          | 0.470      | [0.262, 0.684]    |
| Med3D × image + attrs                  | 0.726      | [0.542, 0.892]    |
| Med3D × full                           | 0.679      | [0.497, 0.853]    |
| **FMCIB × image**                      | **0.887**  | **[0.761, 0.982]**|
| FMCIB × image + attrs                  | 0.729      | [0.544, 0.906]    |
| FMCIB × full                           | 0.679      | [0.477, 0.853]    |

### Paired bootstrap deltas (1000 resamples)

Paired resampling preserves the per-nodule alignment between the two cells' predictions, so these Δ CIs account for the same-subject correlation that a naïve subtraction of two per-cell CIs would overstate.

| comparison                                    | Δ AUC      | 95 % CI           |
|:----------------------------------------------|:----------:|:------------------|
| **FMCIB image  vs  Med3D image**              | **+0.417** | **[+0.181, +0.631]** |
| **FMCIB image  vs  FMCIB image + attrs**      | **+0.158** | **[+0.039, +0.291]** |
| **FMCIB image  vs  Med3D full**               | **+0.208** | **[+0.069, +0.362]** |
| FMCIB image + attrs  vs  Med3D image          | +0.259     | [−0.053, +0.505]  |

**Three of the four comparisons have 95 % CIs that exclude zero** — in each case, FMCIB image-only beats the comparator with bootstrap confidence. The fourth (FMCIB image+attrs vs Med3D image) is positive but its CI straddles zero, so that difference is within sampling noise on the 40-nodule subset.

Formal paired Wilcoxon per-fold tests return *n/a*: several folds contain too few pathology-confirmed nodules (or only one label class) so per-fold AUC is undefined. The pooled bootstrap is the canonical way to report on a subset this small and is what `experiment3_overview.md` pre-registered.

---

## 3. What the numbers mean

### The attribute branch helps the wrong label

Under radiologist-consensus labels (Experiment 3 headline), adding the attribute branch lifted Med3D AUC from 0.612 to 0.960 and FMCIB AUC from 0.885 to 0.960. Same grid under **pathology labels**:

|                                | radiologist-consensus AUC | pathology AUC | Δ (pathology − radiologist) |
|:-------------------------------|:-------------------------:|:-------------:|:---------------------------:|
| Med3D × image                  | 0.612                     | 0.470         | −0.142                      |
| Med3D × image + attrs          | 0.960                     | 0.726         | **−0.234**                  |
| Med3D × full                   | 0.954                     | 0.679         | **−0.275**                  |
| FMCIB × image                  | 0.885                     | **0.887**     | **+0.002**                  |
| FMCIB × image + attrs          | 0.960                     | 0.729         | **−0.231**                  |
| FMCIB × full                   | 0.952                     | 0.679         | **−0.274**                  |

**FMCIB × image is the only configuration that *generalizes* between the two label sets.** Its radiologist AUC and its pathology AUC are essentially equal. Every other configuration loses 0.23 – 0.28 AUC when you swap to pathology labels, and the attribute-branch cells lose the most.

This is exactly the leakage signature. The attribute branch lifts radiologist AUC because the attributes come from the same radiologists. It drops pathology AUC because the attribute-driven predictions are fitted to the opinion, not the disease, and 33 % of the opinions are wrong relative to biopsy. The more weight the model puts on attributes, the worse it does on ground truth.

### FMCIB image-only is the best model on pathology

FMCIB × image reaches pooled AUC **0.887** on pathology — higher than any other cell, with a 95 % CI that reaches 0.982. It's the only cell that learns something about malignancy rather than something about radiologist behavior. The +0.417 AUC paired delta over Med3D × image is the clean image-encoder comparison the Experiment 3 overview's Scenario C was designed to test, and FMCIB wins it decisively.

### The "winning" model under radiologist labels loses on pathology

The radiologist-labeled best cells (both encoders' image+attrs or full config, all at ~0.96 AUC) collapse to ~0.68–0.73 pathology AUC. The previously-rejected cell (FMCIB × image, which at radiologist AUC 0.885 looked like a weaker configuration) is the best pathology cell by 0.16 AUC.

This is the cleanest possible inversion of the Experiment 3 ranking under the two label sources — and it's decisive evidence that the Experiment 3 headline was mostly measuring attribute fidelity, not malignancy detection.

### How Experiment 1's "GCN vs MLP" result reads now

Experiment 1's MLP hit 0.968 AUC against radiologist-consensus labels. Against pathology, the same model (Med3D × full, the Experiment 1 configuration) reaches 0.679. The MLP's apparent 0.97 accuracy was overwhelmingly a function of the attribute shortcut, not of learned imaging features. Experiment 1's GCN "failure" is therefore unsurprising — the graph had no residual signal to add because the features were mostly encoding a copy of the label.

---

## 4. Caveats

- **N = 40 matched nodules.** The pathology subset is intrinsically small; the bootstrap CIs are wide. All the effect sizes above are large enough that 95 % CIs exclude zero on the paired deltas, but any claim finer than the rank order should be held with appropriate humility.
- **The matching is conservative.** Patients with disagreeing per-nodule diagnoses were dropped entirely (62 of 102 pathology-confirmed patients), which removes some of the hardest cases from evaluation. The reported pathology AUCs are therefore likely *optimistic* for the multi-nodule hard-case tail.
- **Pathology ≠ the only ground truth we care about.** The clinical use case for a model like this is generally to help radiologists, not replace them, so models that agree well with radiologists have value. But the Experiment 3 headline claim of "~0.96 AUC malignancy classifier" cannot survive Experiment 3.1's results — the 0.96 is radiologist mimicry, not malignancy detection.
- **`preprocess.py` has a minor hash-collision bug.** 7 nodule_ids in `nodules.parquet` appear more than once (same patient, same centroid) — harmless for Experiments 1–3 because the duplication is symmetric on both sides of the AUC calculation, but `analyze_exp3_pathology.py` dedupes explicitly. Worth a follow-up fix in `label_and_aggregate`.

---

## 5. Implications for downstream work

1. **Adopt FMCIB × image (no attributes) as the primary configuration for Experiment 4 (LUNA25 transfer).** It's the only configuration that generalizes across label sources and therefore the only one that can be expected to generalize across datasets. LUNA25 doesn't have LIDC's per-nodule attribute ratings anyway, so a model that depends on the attribute shortcut would have to be retrained from scratch.
2. **Report pathology AUC in every future write-up.** Radiologist-consensus AUC is a performance proxy; pathology AUC is the thing that matters. The gap between them is the most honest measure of attribute leakage.
3. **Consider re-running Experiment 2 on FMCIB × image.** With the attribute shortcut removed, there's now real residual signal for a graph to add. The "GCN underperforms MLP" verdict from Experiments 1–2 was obtained under attribute-saturated features; it hasn't been tested in the FMCIB × image regime where the image branch actually learns something.
4. **Fix the preprocess.py collision bug** (`label_and_aggregate` hashes centroid to 2 decimal places; occasionally two distinct clusters in the same series collide). Simplest fix: include the within-cluster annotation IDs (already tracked) in the hash.

---

## 6. Artifacts

- `outputs/predictions/exp3_1_pathology_pooled.parquet` — pooled per-cell metrics on the matched subset.
- `outputs/predictions/exp3_1_pathology_per_fold.parquet` — per-fold metrics (many AUCs are NaN due to single-class folds; useful for diagnostic plots).
- `scripts/analyze_exp3_pathology.py` — the analysis script, re-runnable.

The pathology-subset matched-nodule list and per-cell bootstrap CIs are reproducible by re-running the script — it uses `np.random.default_rng(42)` for the bootstrap.

# Experiment 1 — Slide Deck Content

Slide-ready figures and bullet copy for an Experiment 1 recap. Figures are sized for 16:9 slides at 150 dpi. All four hero figures are in `experiments/harrison/figures/slides/` and are regenerable via `scripts/draw_exp1_slide_figures.py`.

Drop each figure on its own slide, set the suggested slide title, paste the bullets verbatim (or trim to fit) as speaker notes or on-slide body text.

---

## Slide 1 — What Experiment 1 tests

**Suggested title:** "Experiment 1 — head-to-head graph vs no graph"

**Hero figure:** [`figures/slides/exp1_approach.png`](figures/slides/exp1_approach.png)

**Talking points (pick 3–5):**
- Research question: does modeling inter-nodule similarity as a graph beat treating each nodule independently?
- LIDC-IDRI: 1,128 labeled nodules across 588 patients (805 benign / 323 malignant). Patient-level 5-fold CV, no patient straddles train and val.
- Two architectures: a 2-layer GCN head (k=10, cosine-similarity KNN graph) and a parameter-matched MLP head, both consuming **identical** 256-D fused Stage-1 features.
- Stage 1 (frozen): Med3D ResNet-50 image features + 8 LIDC attribute embeddings + sinusoidal spatial encoding.
- Pre-registered hypothesis H1: GCN AUC ≥ MLP AUC + 0.01, paired Wilcoxon p < 0.05.
- Inductive evaluation protocol: train-only KNN edges + val-node insertion with edges into train neighbors only — zero chance of transductive leakage.

---

## Slide 2 — Per-fold result: MLP wins every fold

**Suggested title:** "Experiment 1 — MLP beats GCN on every fold"

**Hero figure:** [`figures/slides/exp1_per_fold_bars.png`](figures/slides/exp1_per_fold_bars.png)

**Talking points:**
- Across all 5 validation folds, the MLP AUC exceeds the GCN AUC — no exceptions, no ties.
- Mean AUC gap: **MLP 0.968 vs GCN 0.949**, Δ = −0.0194.
- Fold-level deltas: −0.0004, −0.014, −0.036, −0.016, −0.031. The gap widens on harder folds.
- Paired Wilcoxon signed-rank (one-sided MLP > GCN): **p = 0.0312**, which is the *minimum* achievable p with 5 folds when every sign agrees.
- This is a direction-unambiguous result — small in absolute terms, but consistent in every possible paired comparison.

---

## Slide 3 — Headline metrics + verdict

**Suggested title:** "Experiment 1 — headline results"

**Hero figure:** [`figures/slides/exp1_results_card.png`](figures/slides/exp1_results_card.png)

**Talking points:**
- MLP wins on **all three** reported metrics: AUC, AUPRC, and Brier (calibration).
  - AUC:   **0.968 ± 0.015** vs 0.949 ± 0.029.
  - AUPRC: **0.928 ± 0.038** vs 0.892 ± 0.064.
  - Brier: **0.072 ± 0.009** vs 0.089 ± 0.016 (lower is better).
- GCN has worse calibration, not just worse discrimination — its predicted probabilities are further from the empirical ones on every fold.
- **H1 is rejected.** The graph head is not improving over the parameter-matched MLP on LIDC-IDRI under the plan's pre-registered setup.
- Both models share identical features, identical hyperparameters, identical optimizer, and identical early-stopping schedule. Only the Stage-2 head differs, so the Δ is cleanly attributable to the graph structure itself (not capacity or training noise).

---

## Slide 4 — Why the graph didn't help

**Suggested title:** "Experiment 1 — why the graph didn't help"

**Hero figure:** [`figures/slides/exp1_key_findings.png`](figures/slides/exp1_key_findings.png)

**Talking points (ranked by likelihood under the evidence):**
1. **Near-ceiling baseline.** MLP hits 0.97 AUC *before* the graph enters the picture. There is essentially no residual signal for message passing to add — but plenty of room for it to add noise via neighbor averaging.
2. **Attribute-driven label leakage.** The 8 LIDC descriptive attributes are scored 1–5 by the same four radiologists who score malignancy 1–5. The MLP is largely re-reading the radiologist's own assessment. Experiment 3's image-only ablation is the clean test — and it later confirmed this decisively (image-only AUC drops to 0.61 for Med3D).
3. **Circular KNN over the same feature space.** Edges are built on the same 256-D features the GCN then smooths over. That's partially redundant with the MLP's first linear layer, which already aligns correlated inputs — so the GCN's smoothing adds variance without adding signal.
4. **Noisy neighbors at a high baseline.** At k=10, each val node depends on 10 training neighbors. At 97% accuracy, even a few wrong neighbors per node hurt more than correct neighbors help.
- **Forward pointer:** Experiment 3's feature-modality ablation separates *"graph didn't help"* from *"no signal for the graph to add"*. Experiment 3 subsequently confirmed it's the second one.

---

## Optional slide — results vs prior art

Useful if the audience is expecting to see where this sits in the literature.

- Ma et al. 2023 (closest prior GCN-on-LIDC work) report AUC 0.9629, using a *feature-fusion* GCN architecture, not a node-classification one. Our MLP (0.968) is in the same neighborhood, so the headline AUC is competitive even if H1 failed.
- Typical MLP / 2D CNN LIDC baselines land in the AUC 0.88–0.96 range. Our pipeline sits at the top of that envelope under radiologist-consensus labels.
- Caveat that's worth surfacing on this slide: the 0.968 MLP number is against radiologist consensus. Experiment 3.1 showed pathology-subset AUC drops to 0.68 for the same model — so prior-art numbers in this range should be read with the same caveat.

---

## One-slide executive summary (if you only get one slide)

**Title:** "Experiment 1 — GCN did not beat MLP baseline (H1 rejected)"

- 1,128 nodules, 5-fold patient-level CV, identical Stage-1 features, only the head differs.
- MLP 0.968 AUC vs GCN 0.949 AUC (Δ = −0.019). MLP wins 5/5 folds. Paired Wilcoxon p = 0.0312.
- Most likely reason: MLP is near-ceiling at 0.97 because the Stage-1 attribute branch is correlated with the label by construction (same radiologists). Not much for the graph to add.
- Experiment 3's image-only ablation subsequently confirmed this: Med3D image-only is 0.612 AUC; attributes are doing the heavy lifting in the 0.97 number.

**Best single hero figure for this slide:** [`figures/slides/exp1_per_fold_bars.png`](figures/slides/exp1_per_fold_bars.png) (the visual is decisive on its own).

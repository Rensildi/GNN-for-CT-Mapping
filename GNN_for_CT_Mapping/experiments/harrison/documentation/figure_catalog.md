# Figure Catalog — which figures to use per experiment

A one-stop map from each experiment / sub-section to the slide-ready or paper-ready figure that supports it. Paths are relative to the repo root.

Five figures were added in this round (marked **NEW**) to fill gaps where the existing set didn't communicate the headline finding cleanly.

---

## Approach / architecture

| Use case                                                     | Figure                                                        |
|:-------------------------------------------------------------|:--------------------------------------------------------------|
| Full pipeline diagram (Stage 1 + Stage 2 + heads, single image) | `experiments/harrison/architecture_exp1.png`                  |
| MLP head detail (per-nodule independence emphasized)         | `experiments/harrison/figures/architecture_mlp.png`           |
| GCN head detail (KNN star + message-passing recipe)          | `experiments/harrison/figures/architecture_gcn.png`           |
| Experiment 3 grid (3 × 2 cells with branch indicators)       | `experiments/harrison/figures/architecture_exp3.png`          |

---

## Experiment 1 — GCN vs. MLP baseline

| Slide / section                  | Figure                                                                 |
|:---------------------------------|:-----------------------------------------------------------------------|
| Approach / setup                 | `experiments/harrison/figures/slides/exp1_approach.png`                |
| Hero result (5/5 folds favor MLP)| `experiments/harrison/figures/slides/exp1_per_fold_bars.png`           |
| Headline metrics + verdict       | `experiments/harrison/figures/slides/exp1_results_card.png`            |
| Why the graph didn't help        | `experiments/harrison/figures/slides/exp1_key_findings.png`            |

Coverage is full — no additional figure created in this round.

---

## Experiment 2 — graph-construction ablation

| Slide / section                          | Figure                                                                       |
|:-----------------------------------------|:-----------------------------------------------------------------------------|
| 8-cell heatmap (k × metric)              | `experiments/harrison/figures/exp2_auc_heatmap.png`                          |
| **NEW**  Oversmoothing fingerprint       | **`experiments/harrison/figures/slides/exp2_k_trend.png`**                   |

The new k-trend line plot makes the monotonic AUC decline from k = 5 → k = 20 visible at a glance — the heatmap doesn't render the trend nearly as well. Use the line plot as the slide hero; keep the heatmap for the paper if you want all 8 cells in one place.

---

## Experiment 3 — feature-modality × encoder ablation

| Slide / section                          | Figure                                                                       |
|:-----------------------------------------|:-----------------------------------------------------------------------------|
| 3 × 2 cell grid (architecture)           | `experiments/harrison/figures/architecture_exp3.png`                         |
| 3 × 2 AUC heatmap (radiologist labels)   | `experiments/harrison/figures/exp3_auc_heatmap.png`                          |
| **NEW**  FMCIB-vs-Med3D + ceiling story  | **`experiments/harrison/figures/slides/exp3_modality_lift_bars.png`**        |

The new grouped-bar chart highlights the two sharp messages from Experiment 3 — the +0.27 AUC FMCIB advantage on image-only, and the encoder convergence at the attribute ceiling — that the heatmap blurs. Use the bars on the headline slide; keep the heatmap as a reference table.

---

## Experiment 3.1 — pathology-subset re-evaluation

| Slide / section                          | Figure                                                                                |
|:-----------------------------------------|:--------------------------------------------------------------------------------------|
| **NEW**  Pathology heatmap (analogue of Exp 3's) | **`experiments/harrison/figures/exp3_1_pathology_heatmap.png`**                |
| **NEW**  Cross-label inversion (smoking gun)     | **`experiments/harrison/figures/slides/exp3_1_cross_label_delta.png`**         |
| **NEW**  Bootstrap 95 % CIs (forest plot)        | **`experiments/harrison/figures/slides/exp3_1_bootstrap_cis.png`**             |

Experiment 3.1 had no figures before this round. Recommended slide order:

1. **`exp3_1_pathology_heatmap.png`** — start with the pathology AUC heatmap so the audience sees the matrix on the same axes as Experiment 3.
2. **`exp3_1_cross_label_delta.png`** — then the side-by-side radiologist vs pathology comparison. This is the headline visual: FMCIB × image-only is the only cell that doesn't drop. Use this as the *one* slide if you can only afford one Experiment 3.1 figure.
3. **`exp3_1_bootstrap_cis.png`** — close with the forest plot to anchor the statistical claim that FMCIB × image-only's CI dominates the rest at N = 40.

---

## Experiment 4 — LUNA25 cross-dataset transfer

No project-specific figures yet. Section 6.5 of the report cites slide 34 of the project deck for the LUNA25 risk discussion and slide 32 for the preliminary Graph Transformer numbers (95.42 % accuracy). Final transfer figures (LUNA25 AUC heatmap, LIDC → LUNA25 transfer comparison) are pending and will appear here once Experiment 4 is complete.

---

## Slide deck recommendations

If you're building a single deck recap that walks through Experiments 1 → 3.1, the minimum-viable figure list is:

1. `slides/exp1_approach.png` — sets up the question.
2. `slides/exp1_per_fold_bars.png` — Experiment 1 verdict.
3. `slides/exp2_k_trend.png` — Experiment 2 result + over-smoothing diagnosis.
4. `slides/exp3_modality_lift_bars.png` — Experiment 3 verdict (encoder + ceiling).
5. `slides/exp3_1_cross_label_delta.png` — Experiment 3.1 verdict (the inversion).
6. `slides/exp3_1_bootstrap_cis.png` — Experiment 3.1 statistical credibility.
7. `slides/exp1_key_findings.png` — close with the four leading explanations and pointer to Experiment 4.

That's seven slides covering everything quantitative we've done so far.

---

## Regeneration

All five new figures (and the older ones) are produced by re-runnable matplotlib scripts in `experiments/harrison/scripts/`:

| Script                                   | Figures it produces                                                  |
|:-----------------------------------------|:---------------------------------------------------------------------|
| `draw_architecture.py`                   | `architecture_exp1.png`                                              |
| `draw_head_architectures.py`             | `figures/architecture_mlp.png`, `figures/architecture_gcn.png`       |
| `draw_architecture_exp3.py`              | `figures/architecture_exp3.png`                                      |
| `draw_exp2_heatmap.py`                   | `figures/exp2_auc_heatmap.png`                                       |
| `draw_exp3_heatmap.py`                   | `figures/exp3_auc_heatmap.png`                                       |
| `draw_exp1_slide_figures.py`             | four `slides/exp1_*` figures                                         |
| `draw_additional_result_figures.py`      | the five new figures from this round                                 |

After any change to the Exp summary parquets, re-run the relevant script(s) to refresh the PNGs in place.

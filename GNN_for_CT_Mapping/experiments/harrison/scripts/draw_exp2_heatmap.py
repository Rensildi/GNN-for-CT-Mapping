"""Render the Experiment 2 (k × metric) mean-AUC heatmap.

Produces `experiments/harrison/figures/exp2_auc_heatmap.png`.

Run:
    python -m GNN_for_CT_Mapping.experiments.harrison.scripts.draw_exp2_heatmap
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SUMMARY_PATH = Path("GNN_for_CT_Mapping/outputs/predictions/exp2_summary.parquet")
# Exp 1 references so readers can place the Exp 2 numbers in context.
EXP1_MLP_MEAN_AUC = 0.9680
EXP1_GCN_MEAN_AUC = 0.9486  # (k=10, cosine) from Exp 1's single seeded run


def render(out_path: Path) -> None:
    df = pd.read_parquet(SUMMARY_PATH)
    pivot = (df.pivot_table(index="k", columns="metric", values="auc", aggfunc="mean")
               .reindex(index=[5, 10, 15, 20], columns=["cosine", "euclidean"]))

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    # Color scale deliberately tight around the observed range so the small
    # differences between cells are actually visible.
    vmin = pivot.to_numpy().min() - 0.003
    vmax = pivot.to_numpy().max() + 0.003
    im = ax.imshow(pivot.to_numpy(), cmap="YlGnBu", vmin=vmin, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=11)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=11)
    ax.set_xlabel("similarity metric", fontsize=11)
    ax.set_ylabel("k (neighbors)", fontsize=11)
    ax.set_title("Experiment 2 — mean val AUC across 5 folds",
                 fontsize=13, weight="bold", pad=14)

    # Annotate every cell with its mean AUC.
    for i, k in enumerate(pivot.index):
        for j, metric in enumerate(pivot.columns):
            val = pivot.iloc[i, j]
            # White text when the cell is dark enough; black otherwise.
            text_color = "white" if val > (vmin + vmax) / 2 + 0.001 else "black"
            ax.text(j, i, f"{val:.4f}", ha="center", va="center",
                    fontsize=11, color=text_color, weight="bold")

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("mean AUC", fontsize=10)

    # Reference lines as a caption so readers can anchor the numbers.
    caption = (
        f"Exp 1 MLP baseline: AUC {EXP1_MLP_MEAN_AUC:.4f}    |    "
        f"Exp 1 GCN (k=10, cosine): AUC {EXP1_GCN_MEAN_AUC:.4f}"
    )
    fig.text(0.5, -0.02, caption, ha="center", va="top", fontsize=9, color="dimgray")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    figures_dir = Path(__file__).resolve().parents[1] / "figures"
    render(figures_dir / "exp2_auc_heatmap.png")

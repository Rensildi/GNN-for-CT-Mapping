"""Render the Experiment 3 (feature_config × encoder) mean-AUC heatmap.

Produces `experiments/harrison/figures/exp3_auc_heatmap.png`.

Run:
    python -m GNN_for_CT_Mapping.experiments.harrison.scripts.draw_exp3_heatmap
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


SUMMARY_PATH = Path("GNN_for_CT_Mapping/outputs/predictions/exp3_summary.parquet")
# Exp 1/2 references so readers can place the numbers in context.
EXP1_MLP_MEAN_AUC = 0.9680
EXP1_GCN_MEAN_AUC = 0.9486


def render(out_path: Path) -> None:
    df = pd.read_parquet(SUMMARY_PATH)
    pivot = (df.pivot_table(index="feature_config", columns="encoder",
                            values="auc", aggfunc="mean")
               .reindex(index=["image", "image_attrs", "full"],
                        columns=["med3d", "fmcib"]))

    # Pretty labels so axis ticks are readable without a legend.
    row_labels = ["image only", "image + attrs", "image + attrs + spatial"]
    col_labels = ["Med3D ResNet-50", "FMCIB (wide ResNet-50)"]

    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    # Full [0, 1] AUC range so the MLP ceiling-vs-floor contrast stays visible
    # — clipping to a tight range would obscure the near-chance image-only
    # Med3D cell that is the headline finding.
    im = ax.imshow(pivot.to_numpy(), cmap="YlGnBu", vmin=0.55, vmax=1.0, aspect="auto")

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=12)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=12)
    ax.set_xlabel("image encoder", fontsize=12)
    ax.set_ylabel("Stage-1 feature configuration", fontsize=12)
    ax.set_title("Experiment 3 — mean val AUC across 5 folds",
                 fontsize=14, weight="bold", pad=14)

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iloc[i, j]
            text_color = "white" if val > 0.78 else "black"
            ax.text(j, i, f"{val:.4f}", ha="center", va="center",
                    fontsize=13, color=text_color, weight="bold")

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("mean AUC", fontsize=11)

    caption = (
        f"Exp 1 MLP baseline (full, Med3D): AUC {EXP1_MLP_MEAN_AUC:.4f}   |   "
        f"Exp 1 GCN (full, Med3D): AUC {EXP1_GCN_MEAN_AUC:.4f}"
    )
    fig.text(0.5, -0.02, caption, ha="center", va="top",
             fontsize=10, color="dimgray")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    figures_dir = Path(__file__).resolve().parents[1] / "figures"
    render(figures_dir / "exp3_auc_heatmap.png")

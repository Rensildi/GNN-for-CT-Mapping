"""Render the high-level Experiment 1 architecture diagram.

Produces `experiments/harrison/architecture_exp1.png`. Kept here so the
figure can be regenerated after any architecture tweak; the PNG itself is
a committed artifact users can preview inline without running anything.

Run:
    python -m GNN_for_CT_Mapping.experiments.harrison.scripts.draw_architecture
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


# Color palette — kept consistent with the legend at the bottom of the
# figure so readers can decode the boxes at a glance.
INPUT_COLOR = "#DCE8F5"    # light blue — raw per-nodule inputs
FROZEN_COLOR = "#E8E8E8"   # gray       — frozen pretrained weights
TRAINED_COLOR = "#D4EDDA"  # light green — trainable modules
GRAPH_COLOR = "#FFF3CD"    # light yellow — graph construction
OUTPUT_COLOR = "#F8D7DA"   # light red   — loss / output


def _box(ax, x, y, w, h, text, facecolor, fontsize=9, weight="normal"):
    """Draw a rounded rectangle centered-text block."""
    rect = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02",
        linewidth=1.3,
        facecolor=facecolor,
        edgecolor="black",
    )
    ax.add_patch(rect)
    ax.text(
        x + w / 2, y + h / 2, text,
        ha="center", va="center",
        fontsize=fontsize, weight=weight,
    )


def _arrow(ax, x1, y1, x2, y2):
    """Draw an arrow from (x1, y1) to (x2, y2)."""
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", linewidth=1.3, color="black"),
    )


def render(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(15, 13))
    ax.set_xlim(0, 15)
    ax.set_ylim(-2.2, 12)
    ax.set_aspect("equal")
    ax.axis("off")

    # --- Title ---
    ax.text(7.5, 11.5, "Experiment 1 — GCN vs. MLP Nodule Malignancy Classification",
            ha="center", va="center", fontsize=15, weight="bold")
    ax.text(7.5, 11.05,
            "Stage 1: frozen multi-modal feature fusion    |    "
            "Stage 2: trainable node-classification head (GCN vs MLP)",
            ha="center", va="center", fontsize=10, color="dimgray")

    # --- Row 1: per-nodule inputs ---
    y_in = 9.2
    _box(ax, 0.5, y_in, 4.0, 1.2,
         "48³ CT patch\n(HU-clipped, [0,1] normalized,\nisotropic 1 mm³ resampled)",
         INPUT_COLOR)
    _box(ax, 5.5, y_in, 4.0, 1.2,
         "8 LIDC-IDRI attributes\n(subtlety, sphericity, margin,\nlobulation, spiculation, texture,\ninternal structure, calcification)",
         INPUT_COLOR, fontsize=8)
    _box(ax, 10.5, y_in, 4.0, 1.2,
         "Nodule centroid\n(x, y, z) in mm\n(DICOM patient coords)",
         INPUT_COLOR)

    # --- Row 2: Stage 1 encoders ---
    y_enc = 7.2
    _box(ax, 0.5, y_enc, 4.0, 1.2,
         "Med3D ResNet-50\n(frozen, pretrained on 23\nmedical segmentation datasets)\n→ 2048-D pooled features",
         FROZEN_COLOR, fontsize=8)
    _box(ax, 5.5, y_enc, 4.0, 1.2,
         "8 nn.Embedding tables\n(one per attribute, 8-D each)\n→ 64-D concatenation",
         TRAINED_COLOR, fontsize=8)
    _box(ax, 10.5, y_enc, 4.0, 1.2,
         "Sinusoidal positional encoding\n(16-D per axis × 3 axes)\n→ 48-D",
         INPUT_COLOR, fontsize=8)

    for x in (2.5, 7.5, 12.5):
        _arrow(ax, x, y_in, x, y_enc + 1.2)

    # --- Row 3: image-branch projection only ---
    y_proj = 5.7
    _box(ax, 0.5, y_proj, 4.0, 0.8,
         "Linear (2048 → 256)\n(trainable)",
         TRAINED_COLOR, fontsize=8)
    _arrow(ax, 2.5, y_enc, 2.5, y_proj + 0.8)

    # --- Row 4: fusion ---
    y_fuse = 4.2
    _box(ax, 2.5, y_fuse, 10.0, 1.0,
         "Concatenate → LayerNorm → Linear → 256-D unified node feature",
         TRAINED_COLOR, fontsize=10, weight="bold")

    # Arrows from each encoder's output into the fusion box.
    _arrow(ax, 2.5, y_proj, 5.0, y_fuse + 1.0)       # image
    _arrow(ax, 7.5, y_enc, 7.5, y_fuse + 1.0)        # clinical
    _arrow(ax, 12.5, y_enc, 10.0, y_fuse + 1.0)      # spatial

    # --- Row 5: Stage 2 graph construction split ---
    y_graph = 2.4
    _box(ax, 0.5, y_graph, 6.5, 1.2,
         "Cohort-wide KNN Graph\nk = 10, cosine similarity\nInductive eval: train-only edges fit first,\nval nodes inserted with edges → train neighbors only\n(no val-to-val edges)",
         GRAPH_COLOR, fontsize=8)
    _box(ax, 8.0, y_graph, 6.5, 1.2,
         "No graph\n(node features processed independently)",
         GRAPH_COLOR, fontsize=9)

    _arrow(ax, 4.5, y_fuse, 3.75, y_graph + 1.2)
    _arrow(ax, 10.5, y_fuse, 11.25, y_graph + 1.2)

    # --- Row 6: Stage 2 heads (GCN vs MLP) ---
    y_head = 0.3
    _box(ax, 0.5, y_head, 6.5, 1.6,
         "2-layer GCN\nGCNConv(256, 128) → ReLU → Dropout(0.3)\nGCNConv(128, 64)  → ReLU → Dropout(0.3)\nLinear(64, 2)",
         TRAINED_COLOR, fontsize=9)
    _box(ax, 8.0, y_head, 6.5, 1.6,
         "MLP baseline (parameter-matched)\nLinear(256, 128) → ReLU → Dropout(0.3)\nLinear(128, 64)  → ReLU → Dropout(0.3)\nLinear(64, 2)",
         TRAINED_COLOR, fontsize=9)

    _arrow(ax, 3.75, y_graph, 3.75, y_head + 1.6)
    _arrow(ax, 11.25, y_graph, 11.25, y_head + 1.6)

    # --- Row 7: shared output / loss ---
    y_loss = -1.3
    _box(ax, 5.0, y_loss, 5.0, 0.8,
         "Weighted Cross-Entropy\nbenign (0)  vs.  malignant (1)",
         OUTPUT_COLOR, fontsize=10, weight="bold")

    _arrow(ax, 3.75, y_head, 6.0, y_loss + 0.8)
    _arrow(ax, 11.25, y_head, 9.0, y_loss + 0.8)

    # --- Legend ---
    y_leg = -2.0
    lx = 0.5
    for label, color in [
        ("Input", INPUT_COLOR),
        ("Frozen pretrained", FROZEN_COLOR),
        ("Trainable", TRAINED_COLOR),
        ("Graph ops", GRAPH_COLOR),
        ("Loss / output", OUTPUT_COLOR),
    ]:
        _box(ax, lx, y_leg, 2.6, 0.5, label, color, fontsize=8)
        lx += 2.9

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    # Default path: user folder root. Overridable via a single CLI arg for
    # quick experimentation.
    import sys
    default = Path(__file__).resolve().parents[1] / "architecture_exp1.png"
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else default
    render(target)

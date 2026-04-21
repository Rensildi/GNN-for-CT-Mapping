"""Render the high-level Experiment 1 architecture diagram.

Produces `experiments/harrison/architecture_exp1.png`. Kept here so the
figure can be regenerated after any architecture tweak; the PNG itself is
a committed artifact readers can preview inline without running anything.

Layout principles (rewritten after initial version had cramped text):
    - 20 x 18 canvas so each box gets real estate.
    - Three column centers at x = 3.5, 10.0, 16.5. Each box is 5.8 wide
      which gives ~0.4-inch margins on either side of normal-length
      content strings.
    - fontsize >= 11 everywhere; 13-14 for row-title content.
    - Every multi-line box gets a height derived from its line count, not
      a fixed 1.2 inches, so long rows don't force small text.

Run:
    python -m GNN_for_CT_Mapping.experiments.harrison.scripts.draw_architecture
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


# Palette — consistent with the head diagrams in draw_head_architectures.py.
INPUT_COLOR = "#DCE8F5"    # light blue   — raw per-nodule inputs
FROZEN_COLOR = "#E8E8E8"   # gray         — frozen pretrained weights
TRAINED_COLOR = "#D4EDDA"  # light green  — trainable modules
GRAPH_COLOR = "#FFF3CD"    # light yellow — graph construction
OUTPUT_COLOR = "#F8D7DA"   # light red    — loss / output


# Canvas & column grid.
CANVAS_W = 20.0
CANVAS_H = 18.0
COL_CENTERS = (3.5, 10.0, 16.5)  # x-coord of the 3 vertical columns
COL_BOX_W = 5.8                   # each box is centered on its column


def _box(ax, x, y, w, h, text, facecolor, fontsize=12, weight="normal"):
    """Draw a rounded rectangle with centered text."""
    rect = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.03",
        linewidth=1.4,
        facecolor=facecolor,
        edgecolor="black",
    )
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, weight=weight)


def _col_box(ax, col_idx, y, h, text, facecolor, fontsize=12, weight="normal"):
    """Draw a box centered on one of the three column axes."""
    cx = COL_CENTERS[col_idx]
    x = cx - COL_BOX_W / 2
    _box(ax, x, y, COL_BOX_W, h, text, facecolor, fontsize=fontsize, weight=weight)


def _arrow(ax, x1, y1, x2, y2):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", linewidth=1.6, color="black"),
    )


def render(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(CANVAS_W, CANVAS_H))
    ax.set_xlim(0, CANVAS_W)
    ax.set_ylim(-2.3, CANVAS_H + 0.5)
    ax.set_aspect("equal")
    ax.axis("off")

    # --- Title ---
    ax.text(CANVAS_W / 2, CANVAS_H + 0.0,
            "Experiment 1  —  GCN vs. MLP Nodule Malignancy Classification",
            ha="center", va="center", fontsize=22, weight="bold")
    ax.text(CANVAS_W / 2, CANVAS_H - 0.7,
            "Stage 1: frozen multi-modal feature fusion     |     Stage 2: trainable node-classification head (GCN vs MLP)",
            ha="center", va="center", fontsize=13, color="dimgray")

    # --- Row 1: per-nodule inputs ---
    y_in = 14.5
    h_in = 1.5
    _col_box(ax, 0, y_in, h_in,
             "48³ CT patch\n(HU-clipped, [0, 1] normalized,\nisotropic 1 mm³ resampled)",
             INPUT_COLOR, fontsize=12)
    _col_box(ax, 1, y_in, h_in,
             "8 LIDC-IDRI attributes\n(subtlety, sphericity, margin, lobulation,\nspiculation, texture, internal structure,\ncalcification)",
             INPUT_COLOR, fontsize=11)
    _col_box(ax, 2, y_in, h_in,
             "Nodule centroid\n(x, y, z) in mm\n(DICOM patient coordinates)",
             INPUT_COLOR, fontsize=12)

    # --- Row 2: Stage 1 encoders ---
    y_enc = 12.4
    h_enc = 1.5
    _col_box(ax, 0, y_enc, h_enc,
             "Med3D ResNet-50\n(frozen, pretrained on\n23 medical segmentation datasets)\n→ 2048-D pooled features",
             FROZEN_COLOR, fontsize=11)
    _col_box(ax, 1, y_enc, h_enc,
             "8 nn.Embedding tables\n(one per attribute, 8-D each)\n→ 64-D concatenation",
             TRAINED_COLOR, fontsize=12)
    _col_box(ax, 2, y_enc, h_enc,
             "Sinusoidal positional encoding\n(16-D per axis × 3 axes)\n→ 48-D",
             INPUT_COLOR, fontsize=12)

    for col_idx in (0, 1, 2):
        cx = COL_CENTERS[col_idx]
        _arrow(ax, cx, y_in, cx, y_enc + h_enc)

    # --- Row 3: image-branch trainable projection (column 0 only) ---
    y_proj = 10.6
    h_proj = 1.0
    _col_box(ax, 0, y_proj, h_proj,
             "Linear (2048 → 256)\n(trainable)",
             TRAINED_COLOR, fontsize=12)
    _arrow(ax, COL_CENTERS[0], y_enc, COL_CENTERS[0], y_proj + h_proj)

    # --- Row 4: fusion (spans all three columns) ---
    y_fuse = 8.8
    h_fuse = 1.2
    fuse_left = COL_CENTERS[0] - COL_BOX_W / 2
    fuse_width = COL_CENTERS[2] - COL_CENTERS[0] + COL_BOX_W
    _box(ax, fuse_left, y_fuse, fuse_width, h_fuse,
         "Concatenate   →   LayerNorm   →   Linear   →   256-D unified node feature",
         TRAINED_COLOR, fontsize=15, weight="bold")

    # Arrows from each encoder output into the fusion bar.
    _arrow(ax, COL_CENTERS[0], y_proj, COL_CENTERS[0], y_fuse + h_fuse)
    _arrow(ax, COL_CENTERS[1], y_enc, COL_CENTERS[1], y_fuse + h_fuse)
    _arrow(ax, COL_CENTERS[2], y_enc, COL_CENTERS[2], y_fuse + h_fuse)

    # --- Row 5: Stage 2 split (KNN graph on left, "no graph" on right) ---
    y_graph = 5.9
    h_graph = 1.7
    # Two wide boxes — reuse col_box geometry but stretch them.
    gcn_box_w = 8.5
    mlp_box_w = 8.5
    gap = CANVAS_W - 2 * 0.75 - 2 * gcn_box_w  # remainder -> center gap
    gcn_x = 0.75
    mlp_x = CANVAS_W - 0.75 - mlp_box_w
    _box(ax, gcn_x, y_graph, gcn_box_w, h_graph,
         "Cohort-wide KNN Graph\nk = 10,  cosine similarity\nInductive eval: train-only edges fit first;\nval nodes inserted with edges → train neighbors only\n(no val-to-val edges)",
         GRAPH_COLOR, fontsize=12)
    _box(ax, mlp_x, y_graph, mlp_box_w, h_graph,
         "No graph\n(node features processed independently)",
         GRAPH_COLOR, fontsize=13)

    # Arrows from fusion into each branch.
    _arrow(ax, CANVAS_W / 2 - 2.0, y_fuse, gcn_x + gcn_box_w / 2, y_graph + h_graph)
    _arrow(ax, CANVAS_W / 2 + 2.0, y_fuse, mlp_x + mlp_box_w / 2, y_graph + h_graph)

    # --- Row 6: Stage 2 heads ---
    y_head = 2.6
    h_head = 2.5
    _box(ax, gcn_x, y_head, gcn_box_w, h_head,
         "2-layer GCN\n\n"
         "GCNConv (256 → 128)   +   ReLU   +   Dropout(0.3)\n"
         "GCNConv (128 → 64)    +   ReLU   +   Dropout(0.3)\n"
         "Linear (64 → 2)",
         TRAINED_COLOR, fontsize=13)
    _box(ax, mlp_x, y_head, mlp_box_w, h_head,
         "MLP baseline  (parameter-matched)\n\n"
         "Linear (256 → 128)    +   ReLU   +   Dropout(0.3)\n"
         "Linear (128 → 64)     +   ReLU   +   Dropout(0.3)\n"
         "Linear (64 → 2)",
         TRAINED_COLOR, fontsize=13)

    _arrow(ax, gcn_x + gcn_box_w / 2, y_graph, gcn_x + gcn_box_w / 2, y_head + h_head)
    _arrow(ax, mlp_x + mlp_box_w / 2, y_graph, mlp_x + mlp_box_w / 2, y_head + h_head)

    # --- Row 7: shared loss ---
    y_loss = 0.6
    h_loss = 1.2
    loss_w = 8.0
    _box(ax, (CANVAS_W - loss_w) / 2, y_loss, loss_w, h_loss,
         "Weighted Cross-Entropy\nbenign (0)   vs.   malignant (1)",
         OUTPUT_COLOR, fontsize=15, weight="bold")

    _arrow(ax, gcn_x + gcn_box_w / 2, y_head, (CANVAS_W - loss_w) / 2 + loss_w * 0.3, y_loss + h_loss)
    _arrow(ax, mlp_x + mlp_box_w / 2, y_head, (CANVAS_W - loss_w) / 2 + loss_w * 0.7, y_loss + h_loss)

    # --- Legend (single row of swatches along the bottom) ---
    y_leg = -1.6
    leg_w, leg_h = 3.4, 0.7
    items = [
        ("Input", INPUT_COLOR),
        ("Frozen pretrained", FROZEN_COLOR),
        ("Trainable", TRAINED_COLOR),
        ("Graph ops", GRAPH_COLOR),
        ("Loss / output", OUTPUT_COLOR),
    ]
    total = len(items) * leg_w + (len(items) - 1) * 0.4
    lx = (CANVAS_W - total) / 2
    for label, color in items:
        _box(ax, lx, y_leg, leg_w, leg_h, label, color, fontsize=12)
        lx += leg_w + 0.4

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    import sys
    default = Path(__file__).resolve().parents[1] / "architecture_exp1.png"
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else default
    render(target)

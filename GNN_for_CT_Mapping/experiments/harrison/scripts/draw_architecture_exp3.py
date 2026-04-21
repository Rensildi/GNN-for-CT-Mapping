"""Render the Experiment 3 architecture diagram.

Produces `experiments/harrison/figures/architecture_exp3.png`.

Experiment 3 is a 3 (feature config) x 2 (encoder) ablation, so the
figure has three parts stacked top-to-bottom:

    1. The 3 x 2 cell grid, showing which Stage-1 branches are active
       in each cell and which encoder populates the image branch.
    2. The shared pipeline every cell funnels into: Stage 1 fusion (just
       the active branches per cell), 2-layer GCN, classification head,
       loss.
    3. The evaluation axis: all-nodule metrics plus the pathology-
       confirmed subset, which is the primary validation axis for this
       experiment.

Run:
    python -m GNN_for_CT_Mapping.experiments.harrison.scripts.draw_architecture_exp3
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


# Palette — identical to the other architecture scripts so colors keep
# their meaning across the whole figures set.
INPUT_COLOR = "#DCE8F5"      # light blue   — per-nodule inputs
FROZEN_COLOR = "#E8E8E8"     # gray         — frozen pretrained weights
TRAINED_COLOR = "#D4EDDA"    # light green  — trainable modules
GRAPH_COLOR = "#FFF3CD"      # light yellow — graph construction
OUTPUT_COLOR = "#F8D7DA"     # light red    — loss / output
EVAL_COLOR = "#E6D9F2"       # light purple — evaluation
INACTIVE_COLOR = "#F5F5F5"   # near-white   — branch off in this cell

# Per-encoder accent shades so Med3D and FMCIB cells are visually
# distinguishable at a glance without leaning on text only.
MED3D_ACCENT = "#F0E8D8"
FMCIB_ACCENT = "#E0E9D6"


CANVAS_W = 22.0
CANVAS_H = 22.0


def _box(ax, x, y, w, h, text, facecolor, fontsize=12, weight="normal",
         edge_color="black", edge_lw=1.4):
    rect = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.03",
        linewidth=edge_lw,
        facecolor=facecolor,
        edgecolor=edge_color,
    )
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, weight=weight)


def _plain_box(ax, x, y, w, h, facecolor, edge_color="black", edge_lw=1.4):
    """Same as _box but no text — so caller can place multiple text lines."""
    rect = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.03",
        linewidth=edge_lw,
        facecolor=facecolor,
        edgecolor=edge_color,
    )
    ax.add_patch(rect)


def _arrow(ax, x1, y1, x2, y2, color="black", lw=1.6):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", linewidth=lw, color=color),
    )


def _draw_branch_indicator(ax, x, y, w, h, label, active, active_color):
    """Small indicator block showing whether a Stage-1 branch is active."""
    fill = active_color if active else INACTIVE_COLOR
    edge = "black" if active else "#B0B0B0"
    _box(ax, x, y, w, h,
         ("\u2713  " if active else "\u2715  ") + label,
         fill, fontsize=11,
         weight=("bold" if active else "normal"),
         edge_color=edge, edge_lw=(1.4 if active else 1.0))


def _draw_cell(ax, x, y, w, h, encoder_name, encoder_spec, accent,
               use_image, use_attrs, use_spatial, cell_label):
    """Render one grid cell showing encoder + which branches are on."""
    # Outer box
    _plain_box(ax, x, y, w, h, accent, edge_lw=1.6)
    # Header: encoder name and spec
    ax.text(x + w / 2, y + h - 0.35, encoder_name,
            ha="center", va="center", fontsize=13, weight="bold")
    ax.text(x + w / 2, y + h - 0.85, encoder_spec,
            ha="center", va="center", fontsize=10, color="dimgray")
    # Three stacked branch indicators.
    indicator_w = w - 0.6
    indicator_h = 0.55
    ix = x + 0.3
    # Spacing from header to first indicator.
    top = y + h - 1.4
    for i, (label, active) in enumerate([
        ("image branch", use_image),
        ("attribute branch", use_attrs),
        ("spatial branch", use_spatial),
    ]):
        _draw_branch_indicator(
            ax, ix, top - i * (indicator_h + 0.1),
            indicator_w, indicator_h, label, active, TRAINED_COLOR,
        )
    # Small cell tag bottom-right.
    ax.text(x + w - 0.2, y + 0.2, cell_label,
            ha="right", va="bottom", fontsize=9, color="#888888", style="italic")


def render(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(CANVAS_W, CANVAS_H))
    ax.set_xlim(0, CANVAS_W)
    ax.set_ylim(-2.6, CANVAS_H + 0.8)
    ax.set_aspect("equal")
    ax.axis("off")

    # --- Title ---
    ax.text(CANVAS_W / 2, CANVAS_H + 0.2,
            "Experiment 3  —  Feature-modality \u00d7 Encoder Ablation",
            ha="center", va="center", fontsize=24, weight="bold")
    ax.text(CANVAS_W / 2, CANVAS_H - 0.6,
            "3 feature configurations  \u00d7  2 image encoders  =  6 cells   |   "
            "GCN architecture, splits, and hyperparameters are held fixed",
            ha="center", va="center", fontsize=13, color="dimgray")

    # --- Grid section ---
    # Grid layout: two columns (encoders), three rows (feature configs).
    grid_top = CANVAS_H - 1.4
    grid_bottom = grid_top - 10.2
    cell_w = 6.5
    cell_h = 2.9
    # Horizontal layout:
    #   row-label column | Med3D column | FMCIB column
    row_label_x = 1.0
    row_label_w = 4.0
    col1_x = row_label_x + row_label_w + 0.6
    col2_x = col1_x + cell_w + 0.8

    # Column headers (encoder names) above the grid.
    ax.text(col1_x + cell_w / 2, grid_top + 0.4,
            "Med3D ResNet-50  (baseline)",
            ha="center", va="center", fontsize=15, weight="bold")
    ax.text(col1_x + cell_w / 2, grid_top - 0.05,
            "48\u00b3 patches  \u2192  frozen encoder  \u2192  2048-D",
            ha="center", va="center", fontsize=11, color="dimgray")

    ax.text(col2_x + cell_w / 2, grid_top + 0.4,
            "FMCIB  (committed follow-up)",
            ha="center", va="center", fontsize=15, weight="bold")
    ax.text(col2_x + cell_w / 2, grid_top - 0.05,
            "50\u00b3 patches  \u2192  frozen cancer-CT foundation model  \u2192  4096-D",
            ha="center", va="center", fontsize=11, color="dimgray")

    # Column-header divider line.
    ax.plot([col1_x - 0.3, col2_x + cell_w + 0.3],
            [grid_top - 0.35, grid_top - 0.35],
            color="#888888", linewidth=1.0, zorder=0)

    # Feature config rows.
    feature_configs = [
        (
            "Image only",
            "image branch only; attributes and spatial disabled",
            True, False, False,
        ),
        (
            "Image  +  attributes",
            "Med3D / FMCIB image features fused with 8 radiologist attribute embeddings",
            True, True, False,
        ),
        (
            "Image  +  attributes  +  spatial",
            "full Stage-1 fusion (matches Experiments 1 & 2)",
            True, True, True,
        ),
    ]

    cell_idx = 1
    for r, (row_title, row_desc, use_image, use_attrs, use_spatial) in enumerate(feature_configs):
        row_cy = grid_top - 1.0 - r * (cell_h + 0.45) - cell_h / 2
        # Row label box on the far left.
        _plain_box(ax, row_label_x, row_cy - cell_h / 2, row_label_w, cell_h, INPUT_COLOR)
        ax.text(row_label_x + row_label_w / 2, row_cy + 0.55, row_title,
                ha="center", va="center", fontsize=14, weight="bold")
        ax.text(row_label_x + row_label_w / 2, row_cy - 0.3, row_desc,
                ha="center", va="center", fontsize=10, color="#404040", wrap=True)

        # Med3D cell.
        _draw_cell(ax, col1_x, row_cy - cell_h / 2, cell_w, cell_h,
                   "Med3D ResNet-50", "image features \u2192 2048-D",
                   MED3D_ACCENT, use_image, use_attrs, use_spatial,
                   f"cell {cell_idx}")
        cell_idx += 1
        # FMCIB cell.
        _draw_cell(ax, col2_x, row_cy - cell_h / 2, cell_w, cell_h,
                   "FMCIB foundation model", "image features \u2192 4096-D",
                   FMCIB_ACCENT, use_image, use_attrs, use_spatial,
                   f"cell {cell_idx}")
        cell_idx += 1

    # --- Funnel arrow into the shared pipeline ---
    funnel_y_top = grid_bottom + 0.3
    funnel_y_bot = funnel_y_top - 1.0
    # Three small converging arrows: one from each row's right side down toward
    # the shared pipeline's top edge center.
    pipeline_cx = CANVAS_W / 2
    for r in range(3):
        row_cy = grid_top - 1.0 - r * (cell_h + 0.45) - cell_h / 2
        # Draw an arrow from under each row toward the pipeline.
        mid_x = (col1_x + col2_x + cell_w) / 2
        _arrow(ax, mid_x, row_cy - cell_h / 2, pipeline_cx, funnel_y_bot,
               color="#555555", lw=1.3)

    ax.text(pipeline_cx, funnel_y_top - 0.4,
            "Every cell runs the same downstream pipeline \u2014 only Stage 1's active branches and image encoder change.",
            ha="center", va="center", fontsize=12, style="italic", color="dimgray")

    # --- Shared pipeline ---
    y_fuse = funnel_y_bot - 1.4
    h_fuse = 1.2
    fuse_left = 2.5
    fuse_width = CANVAS_W - 5.0
    _box(ax, fuse_left, y_fuse, fuse_width, h_fuse,
         "Stage 1 fusion  (active branches only)  \u2192   Concat   \u2192   LayerNorm   \u2192   Linear   \u2192   256-D node feature",
         TRAINED_COLOR, fontsize=14, weight="bold")
    _arrow(ax, pipeline_cx, funnel_y_bot, pipeline_cx, y_fuse + h_fuse)

    # GCN head (unchanged from Exp 1 / Exp 2).
    y_gcn = y_fuse - 1.6
    h_gcn = 1.7
    gcn_left = 4.0
    gcn_width = CANVAS_W - 8.0
    _box(ax, gcn_left, y_gcn, gcn_width, h_gcn,
         "2-layer GCN  (inherited from Experiment 1;  k = 10, cosine  by default)\n\n"
         "GCNConv (256 \u2192 128)   +   ReLU   +   Dropout(0.3)\n"
         "GCNConv (128 \u2192 64)    +   ReLU   +   Dropout(0.3)\n"
         "Linear (64 \u2192 2)",
         GRAPH_COLOR, fontsize=12)
    _arrow(ax, pipeline_cx, y_fuse, pipeline_cx, y_gcn + h_gcn)

    # Output head.
    y_out = y_gcn - 1.5
    h_out = 0.95
    out_w = 8.0
    _box(ax, (CANVAS_W - out_w) / 2, y_out, out_w, h_out,
         "softmax   \u2192   P(benign)   /   P(malignant)",
         OUTPUT_COLOR, fontsize=13, weight="bold")
    _arrow(ax, pipeline_cx, y_gcn, pipeline_cx, y_out + h_out)

    y_loss = y_out - 1.3
    h_loss = 0.85
    loss_w = 6.5
    _box(ax, (CANVAS_W - loss_w) / 2, y_loss, loss_w, h_loss,
         "Weighted Cross-Entropy",
         OUTPUT_COLOR, fontsize=13)
    _arrow(ax, pipeline_cx, y_out, pipeline_cx, y_loss + h_loss)

    # --- Evaluation axis ---
    y_eval_top = y_loss - 1.0
    y_eval_bot = y_eval_top - 2.3
    ax.text(pipeline_cx, y_eval_top - 0.1,
            "Evaluation  —  run each of the 6 cells across 5 CV folds",
            ha="center", va="center", fontsize=13, weight="bold")

    eval_box_w = 8.5
    eval_box_h = 1.8
    left_eval = 2.0
    right_eval = CANVAS_W - 2.0 - eval_box_w
    # Left eval panel: all-nodule metrics.
    _box(ax, left_eval, y_eval_bot, eval_box_w, eval_box_h,
         "All-nodule val metrics   (secondary)\n\n"
         "AUC  |  AUPRC  |  sensitivity / specificity\n"
         "at Youden-J  |  Brier  |  reliability diagram",
         EVAL_COLOR, fontsize=12)
    # Right eval panel: pathology-confirmed subset (primary axis).
    _box(ax, right_eval, y_eval_bot, eval_box_w, eval_box_h,
         "Pathology-confirmed subset   (PRIMARY axis)\n\n"
         "\u2248 157 nodules with biopsy / surgery / follow-up ground truth\n"
         "metrics restricted to this subset, pooled across folds",
         EVAL_COLOR, fontsize=12, weight="bold",
         edge_color="#7A4ABD", edge_lw=2.2)

    # Arrows from loss into both eval panels.
    _arrow(ax, pipeline_cx, y_loss, left_eval + eval_box_w / 2, y_eval_bot + eval_box_h,
           color="#555555", lw=1.3)
    _arrow(ax, pipeline_cx, y_loss, right_eval + eval_box_w / 2, y_eval_bot + eval_box_h,
           color="#555555", lw=1.3)

    # --- Legend ---
    y_leg = -1.9
    items = [
        ("Input", INPUT_COLOR),
        ("Frozen pretrained", FROZEN_COLOR),
        ("Trainable", TRAINED_COLOR),
        ("Graph ops", GRAPH_COLOR),
        ("Loss / output", OUTPUT_COLOR),
        ("Evaluation", EVAL_COLOR),
    ]
    leg_w, leg_h = 3.1, 0.7
    total = len(items) * leg_w + (len(items) - 1) * 0.3
    lx = (CANVAS_W - total) / 2
    for label, color in items:
        _box(ax, lx, y_leg, leg_w, leg_h, label, color, fontsize=11)
        lx += leg_w + 0.3

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    figures_dir = Path(__file__).resolve().parents[1] / "figures"
    render(figures_dir / "architecture_exp3.png")

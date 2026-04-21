"""Render side-by-side diagrams of the MLP and GCN Stage 2 heads.

Produces two PNGs in `experiments/harrison/figures/`:
    - architecture_mlp.png — MLP baseline, emphasizing per-nodule independence.
    - architecture_gcn.png — 2-layer GCN, emphasizing KNN graph + message passing.

Both diagrams share the same Stage 1 fusion block so readers see that the
only real difference between the two models is the Stage 2 head.

Layout principles enforced here (fixing the original cramped version):
    - Wide canvas (14 in) so text never needs to shrink below 11 pt.
    - Vertical flow with a single column of boxes — avoids beside-by-side
      overlap between caption text and graph illustration.
    - Conservative text-to-box ratio (box width ≈ 1.3 × expected text
      width) so trailing characters like "Dropout(0.3)" are never clipped.

Run:
    python -m GNN_for_CT_Mapping.experiments.harrison.scripts.draw_head_architectures
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


# Palette — matches draw_architecture.py.
INPUT_COLOR = "#DCE8F5"
FROZEN_COLOR = "#E8E8E8"
TRAINED_COLOR = "#D4EDDA"
GRAPH_COLOR = "#FFF3CD"
OUTPUT_COLOR = "#F8D7DA"
EDGE_COLOR = "#7A99C1"
HILITE_COLOR = "#FF6B6B"


def _box(ax, x, y, w, h, text, facecolor, fontsize=12, weight="normal"):
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


def _arrow(ax, x1, y1, x2, y2, color="black", lw=1.5):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", linewidth=lw, color=color),
    )


CANVAS_W = 16.0  # wide enough that the longest Stage-1 line fits with margin


def _setup_axes(title: str, subtitle: str, height: float = 15):
    """Large single-column canvas. All content is centered around x=CANVAS_W/2."""
    fig, ax = plt.subplots(figsize=(CANVAS_W, height))
    ax.set_xlim(0, CANVAS_W)
    ax.set_ylim(-1.2, height + 0.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.text(CANVAS_W / 2, height + 0.0, title, ha="center", va="center",
            fontsize=22, weight="bold")
    ax.text(CANVAS_W / 2, height - 0.6, subtitle, ha="center", va="center",
            fontsize=13, color="dimgray")
    return fig, ax


def _add_stage1_block(ax, y_top: float) -> float:
    """Shared Stage 1 fusion summary. Returns y-coordinate of the block bottom."""
    height = 2.0
    left, width = 0.5, CANVAS_W - 1.0
    _box(ax, left, y_top - height, width, height,
         "Stage 1  —  frozen multi-modal feature fusion   (identical for both models)\n\n"
         "image  (Med3D ResNet-50)     +     8 attribute embeddings     +     sinusoidal spatial encoding\n\n"
         "→   Concat   →   LayerNorm   →   Linear   →   256-D node feature per nodule",
         TRAINED_COLOR, fontsize=13)
    return y_top - height


def _add_legend(ax, y: float) -> None:
    """Color legend in a single row along the bottom."""
    items = [
        ("Shared / trainable", TRAINED_COLOR),
        ("Graph ops", GRAPH_COLOR),
        ("Loss / output", OUTPUT_COLOR),
        ("Nodule features", INPUT_COLOR),
    ]
    box_w = 3.2
    box_h = 0.6
    total = len(items) * box_w + (len(items) - 1) * 0.4
    x = (CANVAS_W - total) / 2
    for label, color in items:
        _box(ax, x, y, box_w, box_h, label, color, fontsize=11)
        x += box_w + 0.4


# --- MLP diagram ---------------------------------------------------------

def render_mlp(out_path: Path) -> None:
    H = 15
    cx = CANVAS_W / 2  # 8.0 — every column is centered on this.
    fig, ax = _setup_axes(
        "MLP baseline  (Experiment 1)",
        "Each nodule is scored independently  —  no inter-nodule information",
        height=H,
    )

    stage1_bottom = _add_stage1_block(ax, y_top=H - 1.2)

    # Single nodule box on the left, explanatory side note on the right.
    y_nodule = stage1_bottom - 1.8
    nod_w = 3.8
    _box(ax, cx - nod_w - 0.6, y_nodule, nod_w, 1.1,
         "one nodule\n256-D feature vector",
         INPUT_COLOR, fontsize=12)
    note_w = 5.6
    _box(ax, cx + 0.6, y_nodule - 0.15, note_w, 1.4,
         "Processed per-nodule.\nNo edges, no neighbors — nodule i's prediction\ndoes not depend on nodule j's features.",
         INPUT_COLOR, fontsize=11)
    # Arrow from the fused feature into the single nodule box.
    _arrow(ax, cx, stage1_bottom, cx - nod_w / 2 - 0.6, y_nodule + 1.1)

    # MLP stack — wide enough that "Dropout(0.3)" never clips.
    stack_w = 10.0
    stack_x = cx - stack_w / 2
    dy = 1.15
    gap = 0.22
    layers = [
        "Linear (256 → 128)     +     ReLU     +     Dropout(0.3)",
        "Linear (128 → 64)     +     ReLU     +     Dropout(0.3)",
        "Linear (64 → 2)     →     logits per nodule",
    ]
    y_top = y_nodule - 1.3
    for i, text in enumerate(layers):
        y = y_top - i * dy
        _box(ax, stack_x, y, stack_w, dy - gap, text, TRAINED_COLOR, fontsize=13)
    stack_bottom_y = y_top - (len(layers) - 1) * dy

    # Arrow from the single nodule into the top of the MLP stack.
    _arrow(ax, cx - nod_w / 2 - 0.6, y_nodule, cx, y_top + (dy - gap))
    # Chain arrows between the MLP layers.
    for i in range(len(layers) - 1):
        y = y_top - i * dy
        _arrow(ax, cx, y, cx, y - gap)

    # Output head — generously wide so "P(benign) / P(malignant)" fits.
    out_y = stack_bottom_y - 1.5
    out_w = 8.0
    _box(ax, cx - out_w / 2, out_y, out_w, 1.0,
         "softmax     →     P(benign)     /     P(malignant)",
         OUTPUT_COLOR, fontsize=13, weight="bold")
    _arrow(ax, cx, stack_bottom_y, cx, out_y + 1.0)

    # Loss.
    loss_y = out_y - 1.4
    loss_w = 6.5
    _box(ax, cx - loss_w / 2, loss_y, loss_w, 0.9,
         "Weighted Cross-Entropy",
         OUTPUT_COLOR, fontsize=13)
    _arrow(ax, cx, out_y, cx, loss_y + 0.9)

    _add_legend(ax, y=-0.95)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


# --- GCN diagram ---------------------------------------------------------

def render_gcn(out_path: Path) -> None:
    H = 18
    flow_cx = CANVAS_W / 2  # central vertical axis (8.0)
    fig, ax = _setup_axes(
        "2-layer GCN  (Experiment 1)",
        "Each nodule's prediction is informed by its k = 10 nearest training neighbors in the 256-D feature space",
        height=H,
    )

    stage1_bottom = _add_stage1_block(ax, y_top=H - 1.2)

    # --- Two side-by-side explanatory panels sit below Stage 1 ---
    panel_top = stage1_bottom - 0.6
    panel_h = 4.0
    panel_bot = panel_top - panel_h
    panel_w = 7.3
    left_x = 0.5                   # left panel starts here
    right_x = CANVAS_W - 0.5 - panel_w  # right panel starts here

    # Left panel: the star-graph illustration
    _box(ax, left_x, panel_bot, panel_w, panel_h, "", GRAPH_COLOR, fontsize=1)
    ax.text(left_x + panel_w / 2, panel_top - 0.4,
            "KNN graph  (k = 10, cosine)",
            ha="center", va="center", fontsize=14, weight="bold")
    ax.text(left_x + panel_w / 2, panel_top - 0.95,
            "built on training nodules' 256-D features",
            ha="center", va="center", fontsize=11, color="dimgray")

    # Star graph centered inside the panel below the caption.
    gcx = left_x + panel_w / 2 - 0.6
    gcy = panel_bot + 1.6
    r = 1.05
    n_neighbors = 6
    for i in range(n_neighbors):
        theta = -math.pi / 2 + 2 * math.pi * i / n_neighbors
        nx, ny = gcx + r * math.cos(theta), gcy + r * math.sin(theta)
        ax.plot([gcx, nx], [gcy, ny], color=EDGE_COLOR, linewidth=1.6, zorder=1)
        ax.plot(nx, ny, "o", markersize=14, color="#4A6FA5",
                markeredgecolor="black", markeredgewidth=0.7, zorder=2)
    # Highlighted target node in the center.
    ax.plot(gcx, gcy, "o", markersize=20, color=HILITE_COLOR,
            markeredgecolor="black", markeredgewidth=0.9, zorder=3)
    ax.text(gcx + 1.5, gcy, "target nodule",
            ha="left", va="center", fontsize=12, color="#9E3030", weight="bold")

    # Right panel: message-passing recipe.
    _box(ax, right_x, panel_bot, panel_w, panel_h, "", INPUT_COLOR, fontsize=1)
    ax.text(right_x + panel_w / 2, panel_top - 0.4,
            "Message passing  (per GCNConv layer)",
            ha="center", va="center", fontsize=14, weight="bold")
    body_lines = [
        "For each node v :",
        "   1.  Gather its neighbors' current feature vectors.",
        "   2.  Weighted-average using the normalized adjacency.",
        "   3.  Apply a learned linear map + nonlinearity.",
        "",
        "Result:  each node's new representation is a",
        "function of its own features AND its neighbors'.",
    ]
    body_text = "\n".join(body_lines)
    ax.text(right_x + 0.4, panel_bot + panel_h / 2 - 0.35, body_text,
            ha="left", va="center", fontsize=12, color="black")

    # Arrow from Stage 1 center down to the graph panel's top edge.
    _arrow(ax, flow_cx, stage1_bottom, gcx, panel_top)

    # --- 2-layer GCN stack — centered ---
    stack_w = 11.0
    stack_x = flow_cx - stack_w / 2
    dy = 1.15
    gap = 0.22
    layers = [
        "GCNConv (256 → 128)     +     ReLU     +     Dropout(0.3)",
        "GCNConv (128 → 64)     +     ReLU     +     Dropout(0.3)",
        "Linear (64 → 2)     →     logits per nodule",
    ]
    y_top = panel_bot - 1.3
    for i, text in enumerate(layers):
        y = y_top - i * dy
        _box(ax, stack_x, y, stack_w, dy - gap, text, TRAINED_COLOR, fontsize=13)
    stack_bottom_y = y_top - (len(layers) - 1) * dy

    # Arrow from the graph panel into the top of the GCN stack.
    _arrow(ax, gcx, panel_bot, flow_cx, y_top + (dy - gap))
    for i in range(len(layers) - 1):
        y = y_top - i * dy
        _arrow(ax, flow_cx, y, flow_cx, y - gap)

    # Output head.
    out_y = stack_bottom_y - 1.5
    out_w = 8.0
    _box(ax, flow_cx - out_w / 2, out_y, out_w, 1.0,
         "softmax     →     P(benign)     /     P(malignant)",
         OUTPUT_COLOR, fontsize=13, weight="bold")
    _arrow(ax, flow_cx, stack_bottom_y, flow_cx, out_y + 1.0)

    # Loss.
    loss_y = out_y - 1.4
    loss_w = 6.5
    _box(ax, flow_cx - loss_w / 2, loss_y, loss_w, 0.9,
         "Weighted Cross-Entropy",
         OUTPUT_COLOR, fontsize=13)
    _arrow(ax, flow_cx, out_y, flow_cx, loss_y + 0.9)

    _add_legend(ax, y=-0.95)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    figures_dir = Path(__file__).resolve().parents[1] / "figures"
    render_mlp(figures_dir / "architecture_mlp.png")
    render_gcn(figures_dir / "architecture_gcn.png")

"""Render slide-ready figures summarizing Experiment 1.

Four PNGs land in `experiments/harrison/figures/slides/`:

    - exp1_approach.png       : compact three-panel approach card.
    - exp1_per_fold_bars.png  : grouped bar chart of MLP vs GCN AUC per fold.
    - exp1_results_card.png   : headline metric table with the verdict.
    - exp1_key_findings.png   : the four leading explanations for the
                                observed MLP > GCN direction.

Each is sized at ~10 x 5.6 inches (16:9 slide ratio at 150 dpi) with
large fonts and minimal chrome — intended as a single hero figure per
slide, not as a detailed diagram the reader has to study.

Run:
    python -m GNN_for_CT_Mapping.experiments.harrison.scripts.draw_exp1_slide_figures
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Palette matches the rest of the figures suite so decks look coherent.
DEEP_BLUE = "#1F4E79"
ACCENT_BLUE = "#2E75B6"
DARK_GRAY = "#333333"
MID_GRAY = "#6B6B6B"
LIGHT_GREEN = "#D4EDDA"
LIGHT_YELLOW = "#FFF3CD"
LIGHT_RED = "#F8D7DA"
LIGHT_BLUE = "#DCE8F5"
MLP_COLOR = "#2E75B6"   # baseline
GCN_COLOR = "#E07B39"   # treatment


# Headline numbers come straight from outputs/predictions/exp1_summary.parquet.
EXP1_PER_FOLD = {
    0: {"mlp": 0.9895, "gcn": 0.9892},
    1: {"mlp": 0.9652, "gcn": 0.9515},
    2: {"mlp": 0.9471, "gcn": 0.9112},
    3: {"mlp": 0.9723, "gcn": 0.9562},
    4: {"mlp": 0.9657, "gcn": 0.9347},
}
EXP1_MEAN = {"mlp_auc": 0.9680, "mlp_std": 0.0153,
             "gcn_auc": 0.9486, "gcn_std": 0.0287,
             "mlp_auprc": 0.9282, "mlp_auprc_std": 0.0377,
             "gcn_auprc": 0.8922, "gcn_auprc_std": 0.0644,
             "mlp_brier": 0.0725, "mlp_brier_std": 0.0094,
             "gcn_brier": 0.0888, "gcn_brier_std": 0.0163}
EXP1_WILCOXON_P = 0.0312  # one-sided p for MLP > GCN on AUC


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _rounded_box(ax, x, y, w, h, facecolor, edgecolor="black", lw=1.4):
    rect = mpatches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.03",
        facecolor=facecolor, edgecolor=edgecolor, linewidth=lw,
    )
    ax.add_patch(rect)
    return rect


def _save(fig, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out_path}")


# ---------------------------------------------------------------------------
# Figure 1 — Approach card
# ---------------------------------------------------------------------------

def render_approach(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 7.3))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7.3)
    ax.set_aspect("equal")
    ax.axis("off")

    # Title
    ax.text(6.5, 6.85, "Experiment 1  —  GCN vs. MLP head-to-head on LIDC-IDRI",
            ha="center", va="center", fontsize=22, weight="bold", color=DEEP_BLUE)
    ax.text(6.5, 6.3,
            "Does modeling inter-nodule similarity improve malignancy classification?",
            ha="center", va="center", fontsize=14, color=MID_GRAY, style="italic")

    # Accent divider
    ax.plot([5.5, 7.5], [5.95, 5.95], color=ACCENT_BLUE, linewidth=2.5)

    # Three panels
    panel_y = 1.5
    panel_h = 4.0
    panel_w = 3.9
    gap = 0.3
    total_w = 3 * panel_w + 2 * gap
    start_x = (13 - total_w) / 2

    # Panel 1 — Data
    _rounded_box(ax, start_x, panel_y, panel_w, panel_h, LIGHT_BLUE)
    ax.text(start_x + panel_w / 2, panel_y + panel_h - 0.45, "DATA",
            ha="center", va="center", fontsize=16, weight="bold", color=DEEP_BLUE)
    ax.text(start_x + panel_w / 2, panel_y + panel_h - 1.15,
            "LIDC-IDRI", ha="center", va="center", fontsize=14, weight="bold")
    data_lines = [
        "1,128 labeled nodules",
        "588 patients",
        "805 benign  /  323 malignant",
        "",
        "Patient-level 5-fold CV",
        "(StratifiedGroupKFold,  seed 42)",
    ]
    for i, line in enumerate(data_lines):
        ax.text(start_x + panel_w / 2, panel_y + panel_h - 1.85 - i * 0.4, line,
                ha="center", va="center", fontsize=11,
                color=DARK_GRAY if line else MID_GRAY)

    # Panel 2 — Shared Stage 1 + head variation
    x2 = start_x + panel_w + gap
    _rounded_box(ax, x2, panel_y, panel_w, panel_h, LIGHT_GREEN)
    ax.text(x2 + panel_w / 2, panel_y + panel_h - 0.45, "ARCHITECTURE",
            ha="center", va="center", fontsize=16, weight="bold", color=DEEP_BLUE)
    ax.text(x2 + panel_w / 2, panel_y + panel_h - 1.15,
            "shared Stage 1  +  two heads",
            ha="center", va="center", fontsize=13, weight="bold")
    arch_lines = [
        "Stage 1  (frozen):",
        "Med3D ResNet-50  +  8 attributes",
        "+ spatial  →  256-D node feature",
        "",
        "Stage 2  heads  (parameter-matched):",
        "• MLP        (no graph)",
        "• 2-layer GCN  (k = 10, cosine)",
    ]
    for i, line in enumerate(arch_lines):
        style = "italic" if line.endswith(":") else "normal"
        weight = "bold" if line.endswith(":") else "normal"
        ax.text(x2 + panel_w / 2, panel_y + panel_h - 1.85 - i * 0.35, line,
                ha="center", va="center", fontsize=10.5,
                color=DARK_GRAY, style=style, weight=weight)

    # Panel 3 — Hypothesis
    x3 = x2 + panel_w + gap
    _rounded_box(ax, x3, panel_y, panel_w, panel_h, LIGHT_YELLOW)
    ax.text(x3 + panel_w / 2, panel_y + panel_h - 0.45, "HYPOTHESIS",
            ha="center", va="center", fontsize=16, weight="bold", color=DEEP_BLUE)
    ax.text(x3 + panel_w / 2, panel_y + panel_h - 1.3, "H1",
            ha="center", va="center", fontsize=20, weight="bold", color=DEEP_BLUE)
    ax.text(x3 + panel_w / 2, panel_y + panel_h - 1.95,
            "GCN AUC  >  MLP AUC",
            ha="center", va="center", fontsize=13, weight="bold")
    ax.text(x3 + panel_w / 2, panel_y + panel_h - 2.4,
            "by  ≥  0.01",
            ha="center", va="center", fontsize=12, color=DARK_GRAY)
    ax.text(x3 + panel_w / 2, panel_y + panel_h - 2.9,
            "paired Wilcoxon  p  <  0.05",
            ha="center", va="center", fontsize=11, color=MID_GRAY, style="italic")
    ax.text(x3 + panel_w / 2, panel_y + 0.55,
            "If GCN wins: graph structure helps.\nIf MLP wins: no residual signal\nfor the graph to add.",
            ha="center", va="center", fontsize=10, color=MID_GRAY, style="italic")

    # Footer
    ax.text(6.5, 0.5,
            "Inductive graph construction  —  train-only KNN edges,  val nodes inserted with edges into train neighbors only",
            ha="center", va="center", fontsize=10.5, color=MID_GRAY)

    _save(fig, out_path)


# ---------------------------------------------------------------------------
# Figure 2 — Per-fold AUC bars
# ---------------------------------------------------------------------------

def render_per_fold_bars(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6.3))

    folds = sorted(EXP1_PER_FOLD.keys())
    mlp_vals = [EXP1_PER_FOLD[f]["mlp"] for f in folds]
    gcn_vals = [EXP1_PER_FOLD[f]["gcn"] for f in folds]

    bar_w = 0.38
    xs = np.arange(len(folds))

    bars_mlp = ax.bar(xs - bar_w / 2, mlp_vals, bar_w, color=MLP_COLOR,
                      edgecolor="black", linewidth=0.8, label="MLP (baseline)")
    bars_gcn = ax.bar(xs + bar_w / 2, gcn_vals, bar_w, color=GCN_COLOR,
                      edgecolor="black", linewidth=0.8, label="GCN (treatment)")

    # Annotate bars with values.
    for bar, value in list(zip(bars_mlp, mlp_vals)) + list(zip(bars_gcn, gcn_vals)):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.002,
                f"{value:.3f}",
                ha="center", va="bottom", fontsize=10, color=DARK_GRAY)

    # Mean horizontal reference lines.
    mlp_mean = EXP1_MEAN["mlp_auc"]
    gcn_mean = EXP1_MEAN["gcn_auc"]
    ax.axhline(mlp_mean, color=MLP_COLOR, linestyle="--", linewidth=1.2, alpha=0.65)
    ax.axhline(gcn_mean, color=GCN_COLOR, linestyle="--", linewidth=1.2, alpha=0.65)
    ax.text(len(folds) - 0.1, mlp_mean, f"  MLP mean {mlp_mean:.3f}",
            va="center", ha="left", fontsize=10, color=MLP_COLOR, weight="bold")
    ax.text(len(folds) - 0.1, gcn_mean, f"  GCN mean {gcn_mean:.3f}",
            va="center", ha="left", fontsize=10, color=GCN_COLOR, weight="bold")

    ax.set_xticks(xs)
    ax.set_xticklabels([f"Fold {f}" for f in folds], fontsize=12)
    ax.set_ylabel("validation AUC", fontsize=13)
    ax.set_ylim(0.89, 1.00)
    ax.set_title("Experiment 1  —  MLP beats GCN on every fold",
                 fontsize=17, weight="bold", pad=14, color=DEEP_BLUE)
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=11, frameon=True)

    # Keep extra headroom on the right so the mean-line labels don't clip.
    ax.set_xlim(-0.7, len(folds) + 0.05)

    _save(fig, out_path)


# ---------------------------------------------------------------------------
# Figure 3 — Results card
# ---------------------------------------------------------------------------

def render_results_card(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 7.3))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7.3)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.text(6.5, 6.85, "Experiment 1  —  Headline Results",
            ha="center", va="center", fontsize=22, weight="bold", color=DEEP_BLUE)
    ax.plot([5.5, 7.5], [6.35, 6.35], color=ACCENT_BLUE, linewidth=2.5)

    # Three big metric panels.
    panel_w = 3.9
    panel_h = 2.7
    panel_y = 3.2
    gap = 0.3
    total_w = 3 * panel_w + 2 * gap
    start_x = (13 - total_w) / 2

    metrics = [
        ("AUC", EXP1_MEAN["mlp_auc"], EXP1_MEAN["mlp_std"],
                EXP1_MEAN["gcn_auc"], EXP1_MEAN["gcn_std"]),
        ("AUPRC", EXP1_MEAN["mlp_auprc"], EXP1_MEAN["mlp_auprc_std"],
                  EXP1_MEAN["gcn_auprc"], EXP1_MEAN["gcn_auprc_std"]),
        ("Brier  (lower = better)", EXP1_MEAN["mlp_brier"], EXP1_MEAN["mlp_brier_std"],
                                    EXP1_MEAN["gcn_brier"], EXP1_MEAN["gcn_brier_std"]),
    ]
    for i, (label, m, m_std, g, g_std) in enumerate(metrics):
        x = start_x + i * (panel_w + gap)
        _rounded_box(ax, x, panel_y, panel_w, panel_h, LIGHT_BLUE)
        ax.text(x + panel_w / 2, panel_y + panel_h - 0.4, label,
                ha="center", va="center", fontsize=13,
                weight="bold", color=DEEP_BLUE)

        # MLP column
        ax.text(x + panel_w * 0.28, panel_y + panel_h - 1.0, "MLP",
                ha="center", va="center", fontsize=11, color=MLP_COLOR, weight="bold")
        ax.text(x + panel_w * 0.28, panel_y + panel_h - 1.55,
                f"{m:.3f}", ha="center", va="center",
                fontsize=20, weight="bold", color=MLP_COLOR)
        ax.text(x + panel_w * 0.28, panel_y + panel_h - 2.0,
                f"± {m_std:.3f}", ha="center", va="center",
                fontsize=10, color=MID_GRAY)

        # "vs" divider
        ax.text(x + panel_w * 0.5, panel_y + panel_h - 1.45, "vs",
                ha="center", va="center", fontsize=12, color=MID_GRAY, style="italic")

        # GCN column
        ax.text(x + panel_w * 0.72, panel_y + panel_h - 1.0, "GCN",
                ha="center", va="center", fontsize=11, color=GCN_COLOR, weight="bold")
        ax.text(x + panel_w * 0.72, panel_y + panel_h - 1.55,
                f"{g:.3f}", ha="center", va="center",
                fontsize=20, weight="bold", color=GCN_COLOR)
        ax.text(x + panel_w * 0.72, panel_y + panel_h - 2.0,
                f"± {g_std:.3f}", ha="center", va="center",
                fontsize=10, color=MID_GRAY)

        # Delta at the bottom of the panel.
        delta = g - m
        sign = "−" if delta < 0 else "+"
        color = GCN_COLOR if (delta > 0 and "Brier" not in label) or (delta < 0 and "Brier" in label) else MLP_COLOR
        ax.text(x + panel_w / 2, panel_y + 0.3,
                f"Δ (GCN − MLP) = {sign}{abs(delta):.4f}",
                ha="center", va="center", fontsize=11, weight="bold",
                color=color)

    # Verdict bar — wider + two-line subtitle so it never overflows.
    _rounded_box(ax, 0.8, 1.15, 11.4, 1.6, LIGHT_RED, edgecolor=DEEP_BLUE, lw=2.0)
    ax.text(6.5, 2.45,
            "H1 rejected  —  MLP wins on every metric, on every fold",
            ha="center", va="center", fontsize=16, weight="bold", color=DEEP_BLUE)
    ax.text(6.5, 1.85,
            f"Paired Wilcoxon (one-sided MLP > GCN):   p = {EXP1_WILCOXON_P}   on AUC and AUPRC",
            ha="center", va="center", fontsize=11, color=DARK_GRAY, style="italic")
    ax.text(6.5, 1.45,
            "(the minimum achievable with 5 folds when every sign agrees)",
            ha="center", va="center", fontsize=10, color=MID_GRAY, style="italic")

    # Footer
    ax.text(6.5, 0.55,
            "1,128 nodules  ·  588 patients  ·  5-fold patient-level CV  ·  identical features; only the Stage-2 head differs",
            ha="center", va="center", fontsize=10.5, color=MID_GRAY)

    _save(fig, out_path)


# ---------------------------------------------------------------------------
# Figure 4 — Key findings / why the graph didn't help
# ---------------------------------------------------------------------------

def render_key_findings(out_path: Path) -> None:
    # Taller canvas so each panel has room for 3-4 lines of body text
    # without overflow, plus a footer strip that doesn't collide.
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.text(7, 8.55, "Experiment 1  —  Why the Graph Didn't Help",
            ha="center", va="center", fontsize=22, weight="bold", color=DEEP_BLUE)
    ax.plot([6, 8], [8.05, 8.05], color=ACCENT_BLUE, linewidth=2.5)

    ax.text(7, 7.75,
            "Four live hypotheses  —  ranked from most to least likely under the evidence",
            ha="center", va="center", fontsize=12, color=MID_GRAY, style="italic")

    # Each panel's body text is pre-wrapped with explicit line breaks so
    # every line comfortably fits inside the panel width. Panels are also
    # taller now so multi-line bodies don't crowd the border.
    findings = [
        (
            "1.  Near-ceiling baseline",
            "MLP hits 0.97 AUC before the graph enters the picture.",
            "There is essentially no residual signal\n"
            "for message passing to add  —  but\n"
            "plenty of room for it to add noise.",
            LIGHT_RED,
        ),
        (
            "2.  Attribute-driven label leakage",
            "8 LIDC attributes come from the same\nradiologists who score malignancy.",
            "The MLP is re-reading the radiologist's\n"
            "own assessment.  Experiment 3's\n"
            "image-only ablation is the clean test.",
            LIGHT_YELLOW,
        ),
        (
            "3.  Circular KNN over the same\n      feature space",
            "Edges connect nodules whose 256-D\nfeatures are already similar.",
            "Smoothing is largely redundant with the\n"
            "MLP's first linear layer  —  adding\n"
            "variance without new signal.",
            LIGHT_GREEN,
        ),
        (
            "4.  Noisy neighbors at a high baseline",
            "Each val node depends on its k = 10\nnearest training neighbors.",
            "At 97% accuracy, a few wrong neighbors\n"
            "per val node hurt more than correct\n"
            "neighbors help.",
            LIGHT_BLUE,
        ),
    ]

    panel_w = 6.4
    panel_h = 3.1
    panel_gap_x = 0.4
    panel_gap_y = 0.35
    top_y = 4.15
    bot_y = top_y - panel_h - panel_gap_y

    positions = [
        (0.4, top_y),
        (0.4 + panel_w + panel_gap_x, top_y),
        (0.4, bot_y),
        (0.4 + panel_w + panel_gap_x, bot_y),
    ]

    for (title, headline, detail, bg), (x, y) in zip(findings, positions):
        _rounded_box(ax, x, y, panel_w, panel_h, bg)
        # Each panel uses a fixed 3-zone vertical layout so the text blocks
        # can never collide, regardless of how many lines each piece has:
        #
        #   panel top
        #     ├── title     (up to 2 lines, top-anchored at 0.35 from top)
        #   center divider  (not drawn, just a y-landmark)
        #     ├── headline  (up to 2 lines, top-anchored just below center)
        #     └── detail    (up to 3 lines, top-anchored further below)
        #   panel bottom
        title_top = y + panel_h - 0.35
        headline_top = y + panel_h - 1.35
        detail_top = y + panel_h - 2.15

        ax.text(x + 0.3, title_top, title,
                ha="left", va="top", fontsize=13.5, weight="bold", color=DEEP_BLUE)
        ax.text(x + 0.3, headline_top, headline,
                ha="left", va="top", fontsize=11, weight="bold", color=DARK_GRAY,
                linespacing=1.25)
        ax.text(x + 0.3, detail_top, detail,
                ha="left", va="top", fontsize=10.5, color=DARK_GRAY,
                linespacing=1.25)

    # Footer forward-looking note.
    _rounded_box(ax, 1.0, 0.3, 12.0, 0.85, LIGHT_BLUE, edgecolor=DEEP_BLUE, lw=1.8)
    ax.text(7, 0.72,
            "Experiment 3's feature-modality ablation is the critical test  —  "
            "it separates \"graph didn't help\" from \"no signal to add\".",
            ha="center", va="center", fontsize=11.5, style="italic", color=DEEP_BLUE,
            weight="bold")

    _save(fig, out_path)


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parents[1] / "figures" / "slides"
    render_approach(out_dir / "exp1_approach.png")
    render_per_fold_bars(out_dir / "exp1_per_fold_bars.png")
    render_results_card(out_dir / "exp1_results_card.png")
    render_key_findings(out_dir / "exp1_key_findings.png")

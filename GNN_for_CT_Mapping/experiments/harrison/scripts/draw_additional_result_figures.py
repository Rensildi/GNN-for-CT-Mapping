"""Render five additional result figures that fill gaps in the slide /
paper figure set:

    Experiment 2:
        - exp2_k_trend.png            — line plot of AUC vs k, both metrics.
                                        Visualizes the oversmoothing fingerprint.

    Experiment 3:
        - exp3_modality_lift_bars.png — grouped bars per cell, makes the
                                        FMCIB image-only advantage and the
                                        attribute-driven ceiling visible at a glance.

    Experiment 3.1 (pathology):
        - exp3_1_pathology_heatmap.png       — 3 x 2 pathology AUC heatmap
                                               (analogue of Exp 3's heatmap).
        - exp3_1_cross_label_delta.png       — per-cell radiologist vs pathology
                                               AUC, with the FMCIB-image cell flagged
                                               as the only non-dropper. THE smoking gun.
        - exp3_1_bootstrap_cis.png           — forest plot: pooled pathology AUC
                                               + 95% bootstrap CI per cell.

Run:
    python -m GNN_for_CT_Mapping.experiments.harrison.scripts.draw_additional_result_figures
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


# ---------------------------------------------------------------------------
# Palette — kept identical across experiments so slide decks read coherently.
# ---------------------------------------------------------------------------
DEEP_BLUE = "#1F4E79"
ACCENT_BLUE = "#2E75B6"
DARK_GRAY = "#333333"
MID_GRAY = "#6B6B6B"
MED3D_COLOR = "#2E75B6"
FMCIB_COLOR = "#E07B39"
RADIOLOGIST_COLOR = "#7A99C1"
PATHOLOGY_COLOR = "#7A4ABD"
HILITE = "#FF6B6B"


REPO_ROOT = Path(__file__).resolve().parents[4]
SLIDES_DIR = REPO_ROOT / "GNN_for_CT_Mapping/experiments/harrison/figures/slides"
FIGURES_DIR = REPO_ROOT / "GNN_for_CT_Mapping/experiments/harrison/figures"


# ---------------------------------------------------------------------------
# Hard-coded summary numbers from outputs/predictions/*.parquet.
# We re-derive bootstrap CIs on demand from the pathology subset.
# ---------------------------------------------------------------------------

# Exp 2 — mean val AUC across 5 folds.
EXP2_K_VALUES = [5, 10, 15, 20]
EXP2_MEAN_AUC = {
    "cosine":    [0.9615, 0.9549, 0.9513, 0.9497],
    "euclidean": [0.9624, 0.9560, 0.9534, 0.9496],
}
EXP2_STD_AUC = {
    "cosine":    [0.0176, 0.0259, 0.0246, 0.0257],
    "euclidean": [0.0176, 0.0249, 0.0230, 0.0262],
}
EXP1_MLP_MEAN_AUC = 0.9680
EXP1_GCN_MEAN_AUC = 0.9486

# Exp 3 — mean val AUC under radiologist labels (3 x 2 grid).
EXP3_AUC = {
    ("med3d", "image"):           0.6120,
    ("med3d", "image_attrs"):     0.9604,
    ("med3d", "full"):            0.9538,
    ("fmcib", "image"):           0.8852,
    ("fmcib", "image_attrs"):     0.9603,
    ("fmcib", "full"):            0.9523,
}

# Exp 3.1 — pooled pathology AUC and bootstrap 95% CIs from
# `analyze_exp3_pathology.py`.
EXP31_POOLED_AUC = {
    ("med3d", "image"):           0.4702,
    ("med3d", "image_attrs"):     0.7262,
    ("med3d", "full"):            0.6786,
    ("fmcib", "image"):           0.8869,
    ("fmcib", "image_attrs"):     0.7292,
    ("fmcib", "full"):            0.6786,
}
EXP31_CI = {
    ("med3d", "image"):           (0.2618, 0.6841),
    ("med3d", "image_attrs"):     (0.5417, 0.8918),
    ("med3d", "full"):            (0.4965, 0.8528),
    ("fmcib", "image"):           (0.7610, 0.9822),
    ("fmcib", "image_attrs"):     (0.5441, 0.9060),
    ("fmcib", "full"):            (0.4772, 0.8528),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save(fig, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out_path}")


def _pretty_config(name: str) -> str:
    return {"image": "image only",
            "image_attrs": "image + attrs",
            "full": "image + attrs + spatial"}[name]


# ---------------------------------------------------------------------------
# Exp 2 — k-trend line plot (oversmoothing fingerprint)
# ---------------------------------------------------------------------------

def render_exp2_k_trend(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 6.0))

    # Cosine annotations above each marker, euclidean below — values for
    # the two metrics are within ~0.001 of each other so a shared placement
    # would overlap.
    label_offsets = {"cosine": (0.0050, "bottom"),
                     "euclidean": (-0.0055, "top")}
    for metric, color, marker in [("cosine", MED3D_COLOR, "o"),
                                  ("euclidean", FMCIB_COLOR, "s")]:
        means = EXP2_MEAN_AUC[metric]
        stds = EXP2_STD_AUC[metric]
        ax.errorbar(
            EXP2_K_VALUES, means, yerr=stds,
            color=color, marker=marker, markersize=10, linewidth=2.0,
            capsize=5, label=metric, zorder=3,
        )
        dy, va = label_offsets[metric]
        for k, m in zip(EXP2_K_VALUES, means):
            ax.text(k, m + dy, f"{m:.4f}", ha="center", va=va,
                    fontsize=10, color=color, weight="bold")

    # Reference lines for Exp 1's MLP and GCN means.
    ax.axhline(EXP1_MLP_MEAN_AUC, color=DARK_GRAY, linestyle="--", linewidth=1.2)
    ax.text(20.5, EXP1_MLP_MEAN_AUC, f"  Exp 1 MLP {EXP1_MLP_MEAN_AUC:.3f}",
            va="center", ha="left", fontsize=10, color=DARK_GRAY, weight="bold")
    ax.axhline(EXP1_GCN_MEAN_AUC, color=MID_GRAY, linestyle=":", linewidth=1.2)
    ax.text(20.5, EXP1_GCN_MEAN_AUC, f"  Exp 1 GCN {EXP1_GCN_MEAN_AUC:.3f}",
            va="center", ha="left", fontsize=10, color=MID_GRAY)

    ax.set_xticks(EXP2_K_VALUES)
    ax.set_xlabel("k  (number of nearest neighbors)", fontsize=12)
    ax.set_ylabel("validation AUC (mean ± std across 5 folds)", fontsize=12)
    ax.set_title("Experiment 2  —  AUC declines monotonically with larger k\n(over-smoothing fingerprint)",
                 fontsize=14, weight="bold", color=DEEP_BLUE, pad=12)
    ax.set_xlim(3.5, 24)
    ax.set_ylim(0.91, 0.985)
    ax.grid(linestyle=":", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend(title="similarity metric", fontsize=11, title_fontsize=11,
              loc="lower left", frameon=True)

    _save(fig, out_path)


# ---------------------------------------------------------------------------
# Exp 3 — modality-lift grouped bars
# ---------------------------------------------------------------------------

def render_exp3_modality_lift(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6.2))

    configs = ["image", "image_attrs", "full"]
    encoder_offsets = {"med3d": -0.2, "fmcib": +0.2}
    bar_w = 0.36

    xs_main = np.arange(len(configs))
    for encoder, color in [("med3d", MED3D_COLOR), ("fmcib", FMCIB_COLOR)]:
        heights = [EXP3_AUC[(encoder, c)] for c in configs]
        positions = xs_main + encoder_offsets[encoder]
        bars = ax.bar(positions, heights, bar_w, color=color, edgecolor="black",
                      linewidth=0.8, label=encoder.upper())
        for bar, h in zip(bars, heights):
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.012,
                    f"{h:.3f}", ha="center", va="bottom",
                    fontsize=10, color=DARK_GRAY, weight="bold")

    # Reference line for the Exp 1 MLP mean AUC.
    ax.axhline(EXP1_MLP_MEAN_AUC, color=DARK_GRAY, linestyle="--", linewidth=1.2)
    ax.text(2.45, EXP1_MLP_MEAN_AUC, f"  Exp 1 MLP\n  baseline {EXP1_MLP_MEAN_AUC:.3f}",
            va="center", ha="left", fontsize=9.5, color=DARK_GRAY, weight="bold")

    ax.set_xticks(xs_main)
    ax.set_xticklabels([_pretty_config(c) for c in configs], fontsize=12)
    ax.set_ylabel("validation AUC (mean across 5 folds)", fontsize=12)
    ax.set_title("Experiment 3  —  FMCIB beats Med3D on image-only;  "
                 "encoders converge once attributes are added",
                 fontsize=14, weight="bold", color=DEEP_BLUE, pad=12)
    ax.set_ylim(0.55, 1.02)
    ax.set_xlim(-0.6, 2.9)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend(title="image encoder", loc="lower right",
              fontsize=11, title_fontsize=11, frameon=True)

    # Annotation arrow highlighting the FMCIB image-only advantage.
    ax.annotate(
        "",
        xy=(0 + encoder_offsets["fmcib"], EXP3_AUC[("fmcib", "image")] - 0.015),
        xytext=(0 + encoder_offsets["med3d"], EXP3_AUC[("med3d", "image")] - 0.015),
        arrowprops=dict(arrowstyle="->", color=HILITE, linewidth=2),
    )
    delta = EXP3_AUC[("fmcib", "image")] - EXP3_AUC[("med3d", "image")]
    ax.text(0, 0.74, f"Δ = +{delta:.2f}\nFMCIB advantage",
            ha="center", va="center", fontsize=10.5, color=HILITE, weight="bold",
            bbox=dict(facecolor="white", edgecolor=HILITE, boxstyle="round,pad=0.4"))

    _save(fig, out_path)


# ---------------------------------------------------------------------------
# Exp 3.1 — pathology heatmap
# ---------------------------------------------------------------------------

def render_exp3_1_pathology_heatmap(out_path: Path) -> None:
    configs = ["image", "image_attrs", "full"]
    encoders = ["med3d", "fmcib"]
    arr = np.array([[EXP31_POOLED_AUC[(e, c)] for e in encoders] for c in configs])

    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    vmin, vmax = 0.40, 0.92  # leave headroom so values like 0.887 don't blow out the cmap.
    im = ax.imshow(arr, cmap="YlGnBu", vmin=vmin, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(encoders)))
    ax.set_xticklabels(["Med3D ResNet-50", "FMCIB (wide ResNet-50)"], fontsize=12)
    ax.set_yticks(range(len(configs)))
    ax.set_yticklabels([_pretty_config(c) for c in configs], fontsize=12)
    ax.set_xlabel("image encoder", fontsize=12)
    ax.set_ylabel("Stage-1 feature configuration", fontsize=12)
    ax.set_title("Experiment 3.1  —  pooled AUC on pathology-confirmed subset (N = 40)",
                 fontsize=13.5, weight="bold", color=DEEP_BLUE, pad=14)

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            val = arr[i, j]
            text_color = "white" if val > 0.72 else "black"
            ax.text(j, i, f"{val:.4f}", ha="center", va="center",
                    fontsize=14, color=text_color, weight="bold")

    # Highlight the winning cell.
    win_i = configs.index("image")
    win_j = encoders.index("fmcib")
    ax.add_patch(mpatches.Rectangle(
        (win_j - 0.495, win_i - 0.495), 0.99, 0.99,
        linewidth=3.5, edgecolor=HILITE, facecolor="none", zorder=5,
    ))

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("pooled pathology AUC", fontsize=11)

    fig.text(0.5, -0.02,
             "Highlighted cell:  FMCIB × image-only is the only configuration whose "
             "performance does not collapse under pathology labels.",
             ha="center", va="top", fontsize=10, color=DARK_GRAY, style="italic")

    _save(fig, out_path)


# ---------------------------------------------------------------------------
# Exp 3.1 — cross-label comparison chart
# ---------------------------------------------------------------------------

def render_exp3_1_cross_label_delta(out_path: Path) -> None:
    """Per-cell pair of bars: radiologist AUC vs pathology AUC.

    Cells are sorted by Δ (pathology − radiologist) ascending so the
    biggest-dropping cells are at the top of the chart and the unique
    non-dropper (FMCIB image) anchors the bottom.
    """
    rows = []
    for (encoder, cfg), rad_auc in EXP3_AUC.items():
        path_auc = EXP31_POOLED_AUC[(encoder, cfg)]
        rows.append({
            "encoder": encoder, "cfg": cfg,
            "radiologist": rad_auc, "pathology": path_auc,
            "delta": path_auc - rad_auc,
        })
    df = pd.DataFrame(rows).sort_values("delta")

    fig, ax = plt.subplots(figsize=(11.5, 6.5))
    y = np.arange(len(df))
    bar_h = 0.36

    # Two horizontal bars per row.
    rad_bars = ax.barh(y - bar_h / 2, df["radiologist"], bar_h,
                       color=RADIOLOGIST_COLOR, edgecolor="black", linewidth=0.7,
                       label="radiologist-consensus AUC")
    path_bars = ax.barh(y + bar_h / 2, df["pathology"], bar_h,
                        color=PATHOLOGY_COLOR, edgecolor="black", linewidth=0.7,
                        label="pathology AUC")

    for bar, val in zip(rad_bars, df["radiologist"]):
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", ha="left", fontsize=10,
                color=DARK_GRAY)
    for bar, val in zip(path_bars, df["pathology"]):
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", ha="left", fontsize=10,
                color=DARK_GRAY)

    # Highlight the FMCIB × image cell — the only one that doesn't drop.
    for i, (_, row) in enumerate(df.iterrows()):
        if row["encoder"] == "fmcib" and row["cfg"] == "image":
            ax.add_patch(mpatches.Rectangle(
                (-0.005, i - 0.5), 1.05, 1.0,
                linewidth=2.5, edgecolor=HILITE, facecolor=(1, 0.95, 0.85, 0.35),
                zorder=0,
            ))

    ax.set_yticks(y)
    ax.set_yticklabels([f"{r['encoder'].upper()}  ·  {_pretty_config(r['cfg'])}"
                       for _, r in df.iterrows()], fontsize=11)
    ax.set_xlabel("AUC", fontsize=12)
    ax.set_xlim(0, 1.08)
    ax.set_title("Experiment 3.1  —  pathology labels invert the ranking",
                 fontsize=14, weight="bold", color=DEEP_BLUE, pad=12)
    ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=11, frameon=True)

    _save(fig, out_path)


# ---------------------------------------------------------------------------
# Exp 3.1 — bootstrap-CI forest plot
# ---------------------------------------------------------------------------

def render_exp3_1_bootstrap_cis(out_path: Path) -> None:
    """Forest plot: pooled pathology AUC + 95% bootstrap CI per cell, sorted."""
    rows = []
    for (encoder, cfg), auc in EXP31_POOLED_AUC.items():
        lo, hi = EXP31_CI[(encoder, cfg)]
        rows.append({"encoder": encoder, "cfg": cfg,
                     "auc": auc, "lo": lo, "hi": hi})
    df = pd.DataFrame(rows).sort_values("auc")

    fig, ax = plt.subplots(figsize=(11, 6.3))
    y = np.arange(len(df))

    for i, (_, r) in enumerate(df.iterrows()):
        # Encoder colour for the point; CI bar in the same hue.
        color = MED3D_COLOR if r["encoder"] == "med3d" else FMCIB_COLOR
        ax.hlines(i, r["lo"], r["hi"], color=color, linewidth=2.5)
        ax.plot(r["auc"], i, "o", color=color, markersize=11,
                markeredgecolor="black", markeredgewidth=0.7, zorder=3)
        # Annotate point AUC.
        ax.text(r["auc"], i + 0.18, f"{r['auc']:.3f}",
                ha="center", va="bottom", fontsize=10, color=color, weight="bold")
        # CI numbers at endpoints.
        ax.text(r["lo"] - 0.005, i, f"{r['lo']:.2f}", ha="right", va="center",
                fontsize=9, color=MID_GRAY)
        ax.text(r["hi"] + 0.005, i, f"{r['hi']:.2f}", ha="left", va="center",
                fontsize=9, color=MID_GRAY)

    # Reference line at chance.
    ax.axvline(0.5, color=DARK_GRAY, linestyle="--", linewidth=1.0, zorder=1)
    ax.text(0.5, len(df) - 0.4, "chance (AUC = 0.5)",
            ha="center", va="bottom", fontsize=9.5, color=DARK_GRAY)

    ax.set_yticks(y)
    ax.set_yticklabels([f"{r['encoder'].upper()}  ·  {_pretty_config(r['cfg'])}"
                       for _, r in df.iterrows()], fontsize=11)
    ax.set_xlim(0.20, 1.02)
    ax.set_xlabel("pooled pathology-subset AUC  (N = 40)", fontsize=12)
    ax.set_title("Experiment 3.1  —  bootstrap 95% confidence intervals  "
                 "(1,000 resamples)",
                 fontsize=13.5, weight="bold", color=DEEP_BLUE, pad=12)
    ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)

    # Custom legend.
    handles = [mpatches.Patch(color=MED3D_COLOR, label="Med3D"),
               mpatches.Patch(color=FMCIB_COLOR, label="FMCIB")]
    ax.legend(handles=handles, loc="lower right", fontsize=11,
              title="image encoder", title_fontsize=11, frameon=True)

    _save(fig, out_path)


if __name__ == "__main__":
    render_exp2_k_trend(SLIDES_DIR / "exp2_k_trend.png")
    render_exp3_modality_lift(SLIDES_DIR / "exp3_modality_lift_bars.png")
    render_exp3_1_pathology_heatmap(FIGURES_DIR / "exp3_1_pathology_heatmap.png")
    render_exp3_1_cross_label_delta(SLIDES_DIR / "exp3_1_cross_label_delta.png")
    render_exp3_1_bootstrap_cis(SLIDES_DIR / "exp3_1_bootstrap_cis.png")

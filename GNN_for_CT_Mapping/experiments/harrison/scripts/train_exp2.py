"""Experiment 2 — graph construction ablation.

Sweep the 8-cell grid of (k, metric) from §2 of the execution plan and
report per-fold val AUC / AUPRC / Brier. Reuses Experiment 1's cached
Med3D features, patient-level CV splits, and the training helpers from
train_exp1.py. The GCN architecture is fixed (same as Exp 1) — only the
graph construction changes.

Why this is fast:
    Per-fold setup (dataset loading, Stage 1 fusion forward pass, reorder
    masks, class weights) runs ONCE per fold. Only the KNN graph
    construction and a fresh GCN training run execute per (k, metric) cell.
    That's 5 expensive setups + 40 small training runs (5 folds * 8 cells).

Run:
    source AI/bin/activate
    python -m GNN_for_CT_Mapping.experiments.harrison.scripts.train_exp2 \\
        --config GNN_for_CT_Mapping/experiments/harrison/configs/experiment.yaml

Outputs:
    outputs/predictions/exp2_gcn_k{K}_{metric}_fold{i}.parquet
    outputs/predictions/exp2_summary.parquet
    runs/harrison_exp2/fold{i}_gcn_k{K}_{metric}/  (TensorBoard)
"""
from __future__ import annotations

import argparse
import itertools
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from ..models import GCNClassifier, MultiModalFusion
from .dataset import NoduleDataset
from .graph import build_train_edges, insert_val_nodes
from .train_exp1 import (
    FoldSplit,
    _deep_merge,
    build_feature_matrix,
    compute_class_weights,
    fit_model,
    load_folds,
)


# Grid from execution plan §2.3.
K_GRID: tuple[int, ...] = (5, 10, 15, 20)
METRIC_GRID: tuple[str, ...] = ("cosine", "euclidean")


@dataclass
class FoldSetup:
    """Everything we compute once per fold, shared across all (k, metric)
    cells. Keeping this as a dataclass keeps the train_cell call site
    compact and makes it obvious what's reused vs. recomputed.
    """

    x_reordered: torch.Tensor      # (N, node_feature_dim) fused features
    y_reordered: torch.Tensor      # (N,) long
    train_mask_re: torch.Tensor    # (N,) bool — first n_train positions are True
    val_mask_re: torch.Tensor
    train_features_np: np.ndarray  # (n_train, D) — cached for sklearn KNN
    val_features_np: np.ndarray    # (n_val, D)
    meta: pd.DataFrame             # reordered patient_id / nodule_id / label
    class_weights: torch.Tensor    # (2,) — inverse frequency on train
    device: torch.device


def setup_fold(
    fold: FoldSplit,
    nodules_parquet: Path,
    features_parquet: Path,
    cfg: dict,
) -> FoldSetup:
    """Per-fold setup that does not depend on (k, metric).

    Mirrors the prologue of train_exp1.run_fold up to (but not including)
    graph construction — any change here should also be reflected there.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_patients = fold.train_patient_ids + fold.val_patient_ids
    dataset = NoduleDataset(nodules_parquet, features_parquet, patient_ids=all_patients)

    # Stage 1 fusion. The output is detached inside build_feature_matrix,
    # so the random init is the effective fixed Stage 1 projection. Seed
    # already set in main() before we get here.
    fusion = MultiModalFusion(
        image_dim=cfg["model"]["image_feature_dim"],
        node_feature_dim=cfg["model"]["node_feature_dim"],
        clinical_embed_dim=cfg["model"]["clinical_embed_dim"],
    ).to(device)
    x, y, meta = build_feature_matrix(dataset, fusion, device)

    # Boolean masks for which rows are train vs val — matches train_exp1.
    train_set = set(fold.train_patient_ids)
    train_mask_np = meta["patient_id"].isin(train_set).to_numpy()

    # Reorder so train rows sit at indices [0, n_train) — insert_val_nodes
    # places val nodes at n_train + j in the combined edge_index, so the
    # feature matrix must match that ordering.
    train_rows = np.where(train_mask_np)[0]
    val_rows = np.where(~train_mask_np)[0]
    reorder = np.concatenate([train_rows, val_rows])
    x_reordered = x[reorder]
    y_reordered = y[reorder]
    meta_reordered = meta.iloc[reorder].reset_index(drop=True)

    n_train = len(train_rows)
    train_mask_re = torch.zeros(len(reorder), dtype=torch.bool, device=device)
    train_mask_re[:n_train] = True
    val_mask_re = ~train_mask_re

    class_weights = compute_class_weights(y_reordered[train_mask_re]).to(device)

    # Pre-cache numpy arrays — sklearn's KNN consumes numpy, and we'd
    # otherwise re-copy 5 folds * 8 cells = 40 times per run.
    train_features_np = x_reordered[train_mask_re].detach().cpu().numpy()
    val_features_np = x_reordered[val_mask_re].detach().cpu().numpy()

    return FoldSetup(
        x_reordered=x_reordered,
        y_reordered=y_reordered,
        train_mask_re=train_mask_re,
        val_mask_re=val_mask_re,
        train_features_np=train_features_np,
        val_features_np=val_features_np,
        meta=meta_reordered,
        class_weights=class_weights,
        device=device,
    )


def train_cell(
    setup: FoldSetup,
    fold_num: int,
    k: int,
    metric: str,
    cfg: dict,
    predictions_dir: Path,
    runs_dir: Path,
) -> dict[str, float]:
    """Train one GCN on one (k, metric) cell on one fold."""
    # Build inductive KNN graph under this cell's (k, metric).
    train_edges = build_train_edges(
        setup.train_features_np, k=k, metric=metric
    )
    combined_edges, _ = insert_val_nodes(
        setup.train_features_np,
        setup.val_features_np,
        train_edges,
        k=k,
        metric=metric,
    )
    combined_edges = combined_edges.to(setup.device)

    # Fresh GCN per cell — reset the seed so each cell starts from the
    # same random init, i.e. the only cell-level variable is the graph.
    torch.manual_seed(cfg["training"]["seed"])
    np.random.seed(cfg["training"]["seed"])

    model = GCNClassifier(
        in_dim=cfg["model"]["node_feature_dim"],
        hidden_dims=(cfg["model"]["gcn_hidden_dim"], cfg["model"]["gcn_hidden_dim"] // 2),
        dropout=cfg["model"]["dropout"],
    ).to(setup.device)

    run_name = f"fold{fold_num}_gcn_k{k}_{metric}"
    writer = SummaryWriter(log_dir=str(runs_dir / run_name))
    model, metrics = fit_model(
        model,
        setup.x_reordered,
        setup.y_reordered,
        setup.train_mask_re,
        setup.val_mask_re,
        combined_edges,
        class_weights=setup.class_weights,
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
        epochs=cfg["training"]["epochs"],
        patience=cfg["training"].get("patience", 15),
        writer=writer,
    )
    writer.close()

    # Persist per-nodule predictions (val rows only) for downstream
    # significance tests and heatmap plotting.
    with torch.no_grad():
        logits = model(setup.x_reordered, combined_edges)
        probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
    val_mask_np = setup.val_mask_re.cpu().numpy()
    val_meta = setup.meta[val_mask_np].copy()
    val_meta["prob_malignant"] = probs[val_mask_np]
    val_meta["k"] = k
    val_meta["metric"] = metric

    predictions_dir.mkdir(parents=True, exist_ok=True)
    out_path = predictions_dir / f"exp2_gcn_k{k}_{metric}_fold{fold_num}.parquet"
    val_meta.to_parquet(out_path, index=False)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--default-config",
        type=Path,
        default=Path("GNN_for_CT_Mapping/configs/default.yaml"),
    )
    parser.add_argument(
        "--nodules",
        type=Path,
        default=Path("GNN_for_CT_Mapping/data/nodules.parquet"),
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("GNN_for_CT_Mapping/outputs/features/med3d_resnet50.parquet"),
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=Path("GNN_for_CT_Mapping/data/splits"),
    )
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        default=Path("GNN_for_CT_Mapping/outputs/predictions"),
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("GNN_for_CT_Mapping/runs/harrison_exp2"),
    )
    args = parser.parse_args()

    with args.default_config.open() as f:
        cfg = yaml.safe_load(f)
    with args.config.open() as f:
        override = yaml.safe_load(f) or {}
    _deep_merge(cfg, override)

    torch.manual_seed(cfg["training"]["seed"])
    np.random.seed(cfg["training"]["seed"])

    folds = load_folds(args.splits_dir)
    cells = list(itertools.product(K_GRID, METRIC_GRID))

    all_results: list[dict] = []
    total_cells = len(folds) * len(cells)
    completed = 0

    for fold in folds:
        print(f"=== Fold {fold.fold} — per-fold setup ===")
        setup = setup_fold(fold, args.nodules, args.features, cfg)
        for k, metric in cells:
            completed += 1
            print(f"  cell {completed}/{total_cells}  fold={fold.fold}  k={k}  metric={metric}")
            metrics = train_cell(
                setup,
                fold.fold,
                k,
                metric,
                cfg,
                args.predictions_dir,
                args.runs_dir,
            )
            print(f"    -> auc={metrics['auc']:.4f}  auprc={metrics['auprc']:.4f}  brier={metrics['brier']:.4f}")
            all_results.append({
                "fold": fold.fold,
                "k": k,
                "metric": metric,
                **metrics,
            })

    summary_path = args.predictions_dir / "exp2_summary.parquet"
    pd.DataFrame(all_results).to_parquet(summary_path, index=False)
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()

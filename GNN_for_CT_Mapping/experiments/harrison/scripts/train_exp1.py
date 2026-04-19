"""Experiment 1 training loop — GCN vs. MLP baseline.

End-to-end runnable script: for every CV fold, it trains both the MLP
baseline and the GCN head on the same fused features, logs val metrics to
TensorBoard, and persists per-nodule predictions to parquet so the paired
Wilcoxon test can be run offline.

Training design choices (§1.5 of the execution plan):
    - Adam, lr=1e-3, weight_decay=1e-4
    - Weighted cross-entropy with class weights computed per fold
    - 100 epochs with early stopping on val AUC (patience 15)
    - Cohort-wide graph with inductive val-node insertion

Run:
    python -m experiments.harrison.scripts.train_exp1 \\
        --config GNN_for_CT_Mapping/experiments/harrison/configs/experiment.yaml

Outputs:
    - outputs/predictions/exp1_{gcn,mlp}_fold{i}.parquet
    - runs/harrison_exp1/fold{i}_{mlp,gcn}/  (TensorBoard)
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from ..models import GCNClassifier, MLPClassifier, MultiModalFusion
from .dataset import NoduleDataset, collate_stack
from .graph import build_train_edges, insert_val_nodes


@dataclass
class FoldSplit:
    fold: int
    train_patient_ids: list[str]
    val_patient_ids: list[str]


def load_folds(splits_dir: Path) -> list[FoldSplit]:
    """Read all fold_*.json files and return as FoldSplit objects."""
    folds = []
    for p in sorted(splits_dir.glob("fold_*.json")):
        with p.open() as f:
            d = json.load(f)
        folds.append(FoldSplit(
            fold=d["fold"],
            train_patient_ids=d["train_patient_ids"],
            val_patient_ids=d["val_patient_ids"],
        ))
    return folds


def build_feature_matrix(dataset: NoduleDataset, fusion: MultiModalFusion, device: torch.device):
    """Run the fusion module over an entire dataset in one shot.

    Returns:
        features: (N, node_feature_dim) float32 tensor on `device`
        labels:   (N,) long tensor on `device`
        metadata: pandas DataFrame with patient_id, nodule_id, and label
            columns (kept on CPU for parquet persistence).
    """
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=256, shuffle=False, collate_fn=collate_stack
    )
    feat_chunks: list[torch.Tensor] = []
    label_chunks: list[torch.Tensor] = []
    patient_ids: list[str] = []
    nodule_ids: list[str] = []
    for batch in loader:
        image = batch["image_features"].to(device)
        attrs = batch["attributes"].to(device)
        coords = batch["coords"].to(device)
        labels = batch["label"]
        feats = fusion(image_features=image, attributes=attrs, coords=coords)
        feat_chunks.append(feats.detach())
        label_chunks.append(labels)
        patient_ids.extend(batch["patient_id"])
        nodule_ids.extend(batch["nodule_id"])
    features = torch.cat(feat_chunks, dim=0)
    labels = torch.cat(label_chunks, dim=0).to(device)
    meta = pd.DataFrame({"patient_id": patient_ids, "nodule_id": nodule_ids, "label": labels.cpu().numpy()})
    return features, labels, meta


def compute_class_weights(labels: torch.Tensor) -> torch.Tensor:
    """Inverse-frequency class weights so malignant doesn't get swamped."""
    counts = torch.bincount(labels, minlength=2).float()
    inv = 1.0 / torch.clamp(counts, min=1.0)
    # Normalize so the weights sum to num_classes — keeps loss scale stable
    # across folds with different class balance.
    return inv * (counts.numel() / inv.sum())


def fit_model(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor,
    edge_index: torch.Tensor | None,
    class_weights: torch.Tensor,
    *,
    lr: float,
    weight_decay: float,
    epochs: int,
    patience: int,
    writer: SummaryWriter,
) -> tuple[nn.Module, dict[str, float]]:
    """Train one model with early stopping on val AUC, return best checkpoint.

    Implementation note — we do full-batch training on the cohort graph
    (all nodes every step), which is feasible at LIDC's ~1k node scale. For
    larger datasets, swap to neighbor sampling via torch_geometric.loader.
    """
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    best_auc = -1.0
    best_state = None
    epochs_since_improve = 0

    for epoch in range(epochs):
        model.train()
        optim.zero_grad()
        logits = model(x, edge_index) if edge_index is not None else model(x)
        loss = criterion(logits[train_mask], y[train_mask])
        loss.backward()
        optim.step()

        # Val metrics — computed on the full combined graph in the GCN case
        # so val nodes benefit from their inductive edges into train.
        model.eval()
        with torch.no_grad():
            val_logits = model(x, edge_index) if edge_index is not None else model(x)
            val_probs = torch.softmax(val_logits[val_mask], dim=-1)[:, 1].cpu().numpy()
            val_labels = y[val_mask].cpu().numpy()
        if val_labels.min() != val_labels.max():
            auc = float(roc_auc_score(val_labels, val_probs))
        else:
            auc = float("nan")

        writer.add_scalar("loss/train", float(loss.item()), epoch)
        writer.add_scalar("auc/val", auc, epoch)

        if not np.isnan(auc) and auc > best_auc:
            best_auc = auc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final val metrics at the best checkpoint for the results parquet.
    model.eval()
    with torch.no_grad():
        final_logits = model(x, edge_index) if edge_index is not None else model(x)
        final_probs = torch.softmax(final_logits[val_mask], dim=-1)[:, 1].cpu().numpy()
        final_labels = y[val_mask].cpu().numpy()
    metrics = {
        "auc": float(roc_auc_score(final_labels, final_probs)) if final_labels.min() != final_labels.max() else float("nan"),
        "auprc": float(average_precision_score(final_labels, final_probs)),
        "brier": float(brier_score_loss(final_labels, final_probs)),
    }
    return model, metrics


def run_fold(
    fold: FoldSplit,
    nodules_parquet: Path,
    features_parquet: Path,
    cfg: dict,
    predictions_dir: Path,
    runs_dir: Path,
) -> dict[str, dict[str, float]]:
    """Train and evaluate MLP + GCN for one CV fold.

    Returns a nested dict: {model_name: {metric_name: value}}.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build a single dataset that covers train+val patients so the features
    # sit in one tensor; train_mask and val_mask select rows after.
    all_patients = fold.train_patient_ids + fold.val_patient_ids
    dataset = NoduleDataset(nodules_parquet, features_parquet, patient_ids=all_patients)

    fusion = MultiModalFusion(
        image_dim=cfg["model"]["image_feature_dim"],
        node_feature_dim=cfg["model"]["node_feature_dim"],
        clinical_embed_dim=cfg["model"]["clinical_embed_dim"],
    ).to(device)

    x, y, meta = build_feature_matrix(dataset, fusion, device)

    # Boolean masks for which rows are train vs val — derived from the
    # patient split rather than index order so we don't rely on DataLoader
    # ordering assumptions.
    train_set = set(fold.train_patient_ids)
    train_mask_np = meta["patient_id"].isin(train_set).to_numpy()
    train_mask = torch.as_tensor(train_mask_np, device=device)
    val_mask = ~train_mask
    class_weights = compute_class_weights(y[train_mask]).to(device)

    # Build inductive graph: train-only edges, then val nodes inserted.
    train_features_np = x[train_mask].detach().cpu().numpy()
    val_features_np = x[val_mask].detach().cpu().numpy()
    train_edge_index = build_train_edges(
        train_features_np,
        k=cfg["graph"]["k_neighbors"],
        metric=cfg["graph"]["similarity"],
    )
    combined_edge_index, _n_train = insert_val_nodes(
        train_features_np,
        val_features_np,
        train_edge_index,
        k=cfg["graph"]["k_neighbors"],
        metric=cfg["graph"]["similarity"],
    )
    combined_edge_index = combined_edge_index.to(device)

    # The graph orders nodes as [train ... , val ...], so we must reorder
    # (x, y) to match or the edge indices will point at the wrong rows.
    train_rows = np.where(train_mask_np)[0]
    val_rows = np.where(~train_mask_np)[0]
    reorder = np.concatenate([train_rows, val_rows])
    x_reordered = x[reorder]
    y_reordered = y[reorder]
    # Build new masks in the reordered space.
    n_train = len(train_rows)
    train_mask_re = torch.zeros(len(reorder), dtype=torch.bool, device=device)
    train_mask_re[:n_train] = True
    val_mask_re = ~train_mask_re
    meta = meta.iloc[reorder].reset_index(drop=True)

    results: dict[str, dict[str, float]] = {}
    for name, model in [
        ("mlp", MLPClassifier(
            in_dim=cfg["model"]["node_feature_dim"],
            hidden_dims=(cfg["model"]["gcn_hidden_dim"], cfg["model"]["gcn_hidden_dim"] // 2),
            dropout=cfg["model"]["dropout"],
        )),
        ("gcn", GCNClassifier(
            in_dim=cfg["model"]["node_feature_dim"],
            hidden_dims=(cfg["model"]["gcn_hidden_dim"], cfg["model"]["gcn_hidden_dim"] // 2),
            dropout=cfg["model"]["dropout"],
        )),
    ]:
        model = model.to(device)
        writer = SummaryWriter(log_dir=str(runs_dir / f"fold{fold.fold}_{name}"))
        edge_index_arg = combined_edge_index if name == "gcn" else None
        model, metrics = fit_model(
            model, x_reordered, y_reordered, train_mask_re, val_mask_re, edge_index_arg,
            class_weights=class_weights,
            lr=cfg["training"]["lr"],
            weight_decay=cfg["training"]["weight_decay"],
            epochs=cfg["training"]["epochs"],
            patience=cfg["training"].get("patience", 15),
            writer=writer,
        )
        writer.close()
        results[name] = metrics

        # Persist per-nodule predictions on the val split.
        with torch.no_grad():
            logits = model(x_reordered, edge_index_arg) if edge_index_arg is not None else model(x_reordered)
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        val_meta = meta[val_mask_re.cpu().numpy()].copy()
        val_meta["prob_malignant"] = probs[val_mask_re.cpu().numpy()]
        out_path = predictions_dir / f"exp1_{name}_fold{fold.fold}.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        val_meta.to_parquet(out_path, index=False)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True,
                        help="YAML config (merged on top of default.yaml)")
    parser.add_argument("--default-config", type=Path,
                        default=Path("GNN_for_CT_Mapping/configs/default.yaml"))
    parser.add_argument("--nodules", type=Path,
                        default=Path("GNN_for_CT_Mapping/data/nodules.parquet"))
    parser.add_argument("--features", type=Path,
                        default=Path("GNN_for_CT_Mapping/outputs/features/med3d_resnet50.parquet"))
    parser.add_argument("--splits-dir", type=Path,
                        default=Path("GNN_for_CT_Mapping/data/splits"))
    parser.add_argument("--predictions-dir", type=Path,
                        default=Path("GNN_for_CT_Mapping/outputs/predictions"))
    parser.add_argument("--runs-dir", type=Path,
                        default=Path("GNN_for_CT_Mapping/runs/harrison_exp1"))
    args = parser.parse_args()

    # Merge default.yaml with the experiment override.
    with args.default_config.open() as f:
        cfg = yaml.safe_load(f)
    with args.config.open() as f:
        override = yaml.safe_load(f) or {}
    _deep_merge(cfg, override)

    torch.manual_seed(cfg["training"]["seed"])
    np.random.seed(cfg["training"]["seed"])

    folds = load_folds(args.splits_dir)
    all_results: list[dict] = []
    for fold in folds:
        print(f"=== Fold {fold.fold} ===")
        results = run_fold(
            fold,
            args.nodules,
            args.features,
            cfg,
            args.predictions_dir,
            args.runs_dir,
        )
        print(results)
        for model_name, metrics in results.items():
            all_results.append({"fold": fold.fold, "model": model_name, **metrics})

    pd.DataFrame(all_results).to_parquet(
        args.predictions_dir / "exp1_summary.parquet", index=False
    )


def _deep_merge(base: dict, override: dict) -> None:
    """In-place deep merge (override wins on leaves)."""
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


if __name__ == "__main__":
    main()

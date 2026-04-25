"""Experiment 3 — Feature-modality × encoder ablation.

3 feature configs × 2 encoders = 6 cells. For each cell we train the
same 2-layer GCN from Exp 1 over the committed CV splits. The Stage 1
fusion's active branches and the image encoder are the only things that
vary per cell.

Runs:
    python -m GNN_for_CT_Mapping.experiments.harrison.scripts.train_exp3 \\
        --config GNN_for_CT_Mapping/experiments/harrison/configs/experiment.yaml

Outputs:
    outputs/predictions/exp3_{encoder}_{feature_config}_fold{i}.parquet
    outputs/predictions/exp3_summary.parquet
    runs/harrison_exp3/fold{i}_gcn_{encoder}_{feature_config}/  (TensorBoard)
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
from .dataset import NoduleDataset, collate_stack
from .graph import build_train_edges, insert_val_nodes
from .train_exp1 import (
    _deep_merge,
    compute_class_weights,
    fit_model,
    load_folds,
)


# Config dimensions for the 3x2 grid.
ENCODERS = ("med3d", "fmcib")
ENCODER_FEATURE_DIMS = {"med3d": 2048, "fmcib": 4096}
FEATURE_CONFIGS = ("image", "image_attrs", "full")
CONFIG_FLAGS = {
    # (use_image, use_attrs, use_spatial)
    "image": (True, False, False),
    "image_attrs": (True, True, False),
    "full": (True, True, True),
}


def _build_feature_matrix_custom(
    dataset: NoduleDataset,
    fusion: MultiModalFusion,
    device: torch.device,
    use_image: bool,
    use_attrs: bool,
    use_spatial: bool,
) -> tuple[torch.Tensor, torch.Tensor, pd.DataFrame]:
    """Like train_exp1.build_feature_matrix but aware of which Stage-1
    branches are active.

    The fusion module is instantiated with only the requested branches,
    and we only pass in the corresponding inputs to `fusion.forward`.
    """
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=256, shuffle=False, collate_fn=collate_stack
    )
    feat_chunks: list[torch.Tensor] = []
    label_chunks: list[torch.Tensor] = []
    patient_ids: list[str] = []
    nodule_ids: list[str] = []
    for batch in loader:
        kwargs: dict = {}
        if use_image:
            kwargs["image_features"] = batch["image_features"].to(device)
        if use_attrs:
            kwargs["attributes"] = batch["attributes"].to(device)
        if use_spatial:
            kwargs["coords"] = batch["coords"].to(device)
        feats = fusion(**kwargs)
        feat_chunks.append(feats.detach())
        label_chunks.append(batch["label"])
        patient_ids.extend(batch["patient_id"])
        nodule_ids.extend(batch["nodule_id"])
    features = torch.cat(feat_chunks, dim=0)
    labels = torch.cat(label_chunks, dim=0).to(device)
    meta = pd.DataFrame({
        "patient_id": patient_ids,
        "nodule_id": nodule_ids,
        "label": labels.cpu().numpy(),
    })
    return features, labels, meta


@dataclass
class FoldSetup:
    x_reordered: torch.Tensor
    y_reordered: torch.Tensor
    train_mask_re: torch.Tensor
    val_mask_re: torch.Tensor
    train_features_np: np.ndarray
    val_features_np: np.ndarray
    meta: pd.DataFrame
    class_weights: torch.Tensor
    device: torch.device


def setup_fold(
    fold,
    nodules_parquet: Path,
    features_parquet: Path,
    encoder: str,
    feature_config: str,
    cfg: dict,
) -> FoldSetup:
    """Per (fold, encoder, feature_config) setup. Must be re-run per cell
    because the Stage 1 fusion geometry changes with feature_config and
    with image_feature_dim per encoder.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_patients = fold.train_patient_ids + fold.val_patient_ids
    dataset = NoduleDataset(nodules_parquet, features_parquet, patient_ids=all_patients)

    use_image, use_attrs, use_spatial = CONFIG_FLAGS[feature_config]
    fusion = MultiModalFusion(
        image_dim=ENCODER_FEATURE_DIMS[encoder],
        node_feature_dim=cfg["model"]["node_feature_dim"],
        clinical_embed_dim=cfg["model"]["clinical_embed_dim"],
        use_image=use_image,
        use_clinical=use_attrs,
        use_spatial=use_spatial,
    ).to(device)
    x, y, meta = _build_feature_matrix_custom(
        dataset, fusion, device, use_image, use_attrs, use_spatial
    )

    train_set = set(fold.train_patient_ids)
    train_mask_np = meta["patient_id"].isin(train_set).to_numpy()
    train_rows = np.where(train_mask_np)[0]
    val_rows = np.where(~train_mask_np)[0]
    reorder = np.concatenate([train_rows, val_rows])
    x_re = x[reorder]
    y_re = y[reorder]
    meta_re = meta.iloc[reorder].reset_index(drop=True)
    n_train = len(train_rows)
    train_mask_re = torch.zeros(len(reorder), dtype=torch.bool, device=device)
    train_mask_re[:n_train] = True
    val_mask_re = ~train_mask_re
    class_weights = compute_class_weights(y_re[train_mask_re]).to(device)
    train_features_np = x_re[train_mask_re].detach().cpu().numpy()
    val_features_np = x_re[val_mask_re].detach().cpu().numpy()

    return FoldSetup(
        x_reordered=x_re, y_reordered=y_re,
        train_mask_re=train_mask_re, val_mask_re=val_mask_re,
        train_features_np=train_features_np, val_features_np=val_features_np,
        meta=meta_re, class_weights=class_weights, device=device,
    )


def train_cell(
    setup: FoldSetup,
    fold_num: int,
    encoder: str,
    feature_config: str,
    cfg: dict,
    predictions_dir: Path,
    runs_dir: Path,
) -> dict[str, float]:
    k = cfg["graph"]["k_neighbors"]
    metric = cfg["graph"]["similarity"]
    train_edges = build_train_edges(setup.train_features_np, k=k, metric=metric)
    combined_edges, _ = insert_val_nodes(
        setup.train_features_np,
        setup.val_features_np,
        train_edges,
        k=k,
        metric=metric,
    )
    combined_edges = combined_edges.to(setup.device)

    torch.manual_seed(cfg["training"]["seed"])
    np.random.seed(cfg["training"]["seed"])

    model = GCNClassifier(
        in_dim=cfg["model"]["node_feature_dim"],
        hidden_dims=(cfg["model"]["gcn_hidden_dim"], cfg["model"]["gcn_hidden_dim"] // 2),
        dropout=cfg["model"]["dropout"],
    ).to(setup.device)

    run_name = f"fold{fold_num}_gcn_{encoder}_{feature_config}"
    writer = SummaryWriter(log_dir=str(runs_dir / run_name))
    model, metrics = fit_model(
        model,
        setup.x_reordered, setup.y_reordered,
        setup.train_mask_re, setup.val_mask_re,
        combined_edges,
        class_weights=setup.class_weights,
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
        epochs=cfg["training"]["epochs"],
        patience=cfg["training"].get("patience", 15),
        writer=writer,
    )
    writer.close()

    # Per-nodule val predictions, tagged with the cell identifiers.
    with torch.no_grad():
        logits = model(setup.x_reordered, combined_edges)
        probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
    val_mask_np = setup.val_mask_re.cpu().numpy()
    val_meta = setup.meta[val_mask_np].copy()
    val_meta["prob_malignant"] = probs[val_mask_np]
    val_meta["encoder"] = encoder
    val_meta["feature_config"] = feature_config
    predictions_dir.mkdir(parents=True, exist_ok=True)
    out_path = predictions_dir / f"exp3_{encoder}_{feature_config}_fold{fold_num}.parquet"
    val_meta.to_parquet(out_path, index=False)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--default-config", type=Path,
                        default=Path("GNN_for_CT_Mapping/configs/default.yaml"))
    parser.add_argument("--nodules", type=Path,
                        default=Path("GNN_for_CT_Mapping/data/nodules.parquet"))
    parser.add_argument("--med3d-features", type=Path,
                        default=Path("GNN_for_CT_Mapping/outputs/features/med3d_resnet50.parquet"))
    parser.add_argument("--fmcib-features", type=Path,
                        default=Path("GNN_for_CT_Mapping/outputs/features/fmcib.parquet"))
    parser.add_argument("--splits-dir", type=Path,
                        default=Path("GNN_for_CT_Mapping/data/splits"))
    parser.add_argument("--predictions-dir", type=Path,
                        default=Path("GNN_for_CT_Mapping/outputs/predictions"))
    parser.add_argument("--runs-dir", type=Path,
                        default=Path("GNN_for_CT_Mapping/runs/harrison_exp3"))
    args = parser.parse_args()

    with args.default_config.open() as f:
        cfg = yaml.safe_load(f)
    with args.config.open() as f:
        override = yaml.safe_load(f) or {}
    _deep_merge(cfg, override)

    # Global seed — re-seeded per cell inside train_cell().
    torch.manual_seed(cfg["training"]["seed"])
    np.random.seed(cfg["training"]["seed"])

    folds = load_folds(args.splits_dir)
    feature_paths = {"med3d": args.med3d_features, "fmcib": args.fmcib_features}

    cells = list(itertools.product(ENCODERS, FEATURE_CONFIGS))
    total = len(folds) * len(cells)

    all_results: list[dict] = []
    i = 0
    for fold in folds:
        print(f"=== Fold {fold.fold} ===")
        for encoder, feature_config in cells:
            i += 1
            print(f"  cell {i}/{total}   encoder={encoder}   feature_config={feature_config}")
            setup = setup_fold(fold, args.nodules, feature_paths[encoder],
                               encoder, feature_config, cfg)
            metrics = train_cell(
                setup, fold.fold, encoder, feature_config,
                cfg, args.predictions_dir, args.runs_dir,
            )
            print(f"    -> auc={metrics['auc']:.4f}  auprc={metrics['auprc']:.4f}  brier={metrics['brier']:.4f}")
            all_results.append({
                "fold": fold.fold,
                "encoder": encoder,
                "feature_config": feature_config,
                **metrics,
            })

    summary_path = args.predictions_dir / "exp3_summary.parquet"
    pd.DataFrame(all_results).to_parquet(summary_path, index=False)
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()

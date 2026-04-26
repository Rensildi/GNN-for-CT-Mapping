"""Inductive KNN graph construction"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors


def build_train_edges(train_features: np.ndarray, k: int = 10, metric: str = "cosine") -> torch.Tensor:
    """Build a symmetric KNN edge_index over training nodes."""
    n = train_features.shape[0]
    nbrs = NearestNeighbors(n_neighbors=min(k + 1, n), metric=metric, algorithm="brute")
    nbrs.fit(train_features)
    _, indices = nbrs.kneighbors(train_features)
    indices = indices[:, 1:]  # drop self

    src = np.repeat(np.arange(n), indices.shape[1])
    dst = indices.reshape(-1)
    edges_fwd = np.stack([src, dst], axis=0)
    edges_bwd = np.stack([dst, src], axis=0)
    return torch.as_tensor(np.concatenate([edges_fwd, edges_bwd], axis=1), dtype=torch.long)


def insert_val_nodes(
    train_features: np.ndarray,
    val_features: np.ndarray,
    train_edge_index: torch.Tensor,
    k: int = 10,
    metric: str = "cosine",
) -> tuple[torch.Tensor, int]:
    """Append validation nodes and connect them only to training neighbors."""
    n_train = train_features.shape[0]
    n_val = val_features.shape[0]
    nbrs = NearestNeighbors(n_neighbors=min(k, n_train), metric=metric, algorithm="brute")
    nbrs.fit(train_features)
    _, indices = nbrs.kneighbors(val_features)

    val_src = np.repeat(np.arange(n_val) + n_train, indices.shape[1])
    val_dst = indices.reshape(-1)
    val_edges_fwd = np.stack([val_src, val_dst], axis=0)
    val_edges_bwd = np.stack([val_dst, val_src], axis=0)

    base_edges = train_edge_index.detach().cpu().numpy()
    edges = np.concatenate([base_edges, val_edges_fwd, val_edges_bwd], axis=1)
    return torch.as_tensor(edges, dtype=torch.long), n_train

"""Inductive KNN graph construction for the cohort-wide nodule graph.

Implements §1.4 of the execution plan: fit edges only on train-fold nodules,
then insert val/test nodules at eval time as new nodes whose edges only
point at their k nearest training neighbors (no train-to-val feedback and
no val-to-val edges). This is the correct inductive setup for a cohort
graph under patient-level cross-validation.

KNN is computed in the *fused node feature* space, on cosine (default) or
Euclidean distance — §1.3 fixes cosine / k=10 for Experiment 1.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors


def build_train_edges(
    train_features: np.ndarray,
    k: int = 10,
    metric: str = "cosine",
) -> torch.Tensor:
    """Build a symmetric KNN edge_index over training nodules only.

    Args:
        train_features: (N_train, D) float32 array.
        k: neighbor count per node.
        metric: "cosine" or "euclidean" — passed through to sklearn.

    Returns:
        PyG-format edge_index tensor of shape (2, 2 * N_train * k).
    """
    n = train_features.shape[0]
    # sklearn's cosine metric expects brute-force; it's fast enough at
    # LIDC scale (~1k train nodules per fold).
    nbrs = NearestNeighbors(n_neighbors=min(k + 1, n), metric=metric, algorithm="brute")
    nbrs.fit(train_features)
    _, indices = nbrs.kneighbors(train_features)
    # indices[:, 0] is always the node itself — drop it so we don't double
    # the self-loop (GCNConv adds self-loops internally).
    indices = indices[:, 1:]

    # PyG expects a (2, E) long tensor. Build forward edges then symmetrize.
    src = np.repeat(np.arange(n), indices.shape[1])
    dst = indices.reshape(-1)
    edges_fwd = np.stack([src, dst], axis=0)
    edges_bwd = np.stack([dst, src], axis=0)
    edges = np.concatenate([edges_fwd, edges_bwd], axis=1)
    return torch.as_tensor(edges, dtype=torch.long)


def insert_val_nodes(
    train_features: np.ndarray,
    val_features: np.ndarray,
    train_edge_index: torch.Tensor,
    k: int = 10,
    metric: str = "cosine",
) -> tuple[torch.Tensor, int]:
    """Return an edge_index for the combined (train || val) node set.

    Val nodes are appended AFTER train nodes (so val indices are
    N_train .. N_train + N_val - 1). Each val node gets k directed edges
    into its nearest training neighbors — and symmetric edges back so the
    GCN can also propagate information into the val node during forward
    eval. Crucially, val-to-val edges are NOT added.

    Args:
        train_features: (N_train, D) float32 array used to fit the KNN.
        val_features:   (N_val, D)   float32 array of held-out val nodes.
        train_edge_index: edges from build_train_edges.
        k: neighbor count per val node.
        metric: "cosine" or "euclidean".

    Returns:
        (combined_edge_index, n_train) — the caller builds the full node
        feature matrix by concatenating train_features then val_features.
    """
    n_train = train_features.shape[0]
    n_val = val_features.shape[0]

    nbrs = NearestNeighbors(n_neighbors=min(k, n_train), metric=metric, algorithm="brute")
    nbrs.fit(train_features)
    _, indices = nbrs.kneighbors(val_features)

    # Val node j sits at index n_train + j in the combined matrix.
    val_src = np.repeat(np.arange(n_val) + n_train, indices.shape[1])
    val_dst = indices.reshape(-1)
    val_edges_fwd = np.stack([val_src, val_dst], axis=0)
    val_edges_bwd = np.stack([val_dst, val_src], axis=0)

    edges = np.concatenate([
        train_edge_index.numpy(),
        val_edges_fwd,
        val_edges_bwd,
    ], axis=1)
    return torch.as_tensor(edges, dtype=torch.long), n_train

"""2-layer GCN classifier for Experiment 1.

Uses PyTorch Geometric's GCNConv (Kipf & Welling 2017) to message-pass over
the cohort-wide KNN nodule-similarity graph. The 2-layer depth gives a 2-hop
receptive field, which is the standard choice for node classification on
small graphs — deeper stacks suffer from over-smoothing (node representations
collapsing toward the graph mean).

Architecture (from documentation/execution_plan_experiments.md §1.3):
    GCNConv(256, 128) -> ReLU -> Dropout(0.3)
    GCNConv(128, 64)  -> ReLU -> Dropout(0.3)
    Linear(64, 2)
"""
from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import GCNConv


class GCNClassifier(nn.Module):
    """Two-layer GCN that aggregates information across KNN neighbors.

    Self-loops and symmetric normalization are enabled by default on GCNConv
    (the normalized adjacency D^(-1/2) A D^(-1/2)), matching the classical
    Kipf & Welling formulation.
    """

    def __init__(
        self,
        in_dim: int = 256,
        hidden_dims: tuple[int, int] = (128, 64),
        num_classes: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        h1, h2 = hidden_dims

        # First message-passing layer: neighborhood-aggregated features are
        # passed through a learned linear map.
        self.conv1 = GCNConv(in_dim, h1, add_self_loops=True, normalize=True)
        # Second message-passing layer: extends the effective receptive field
        # to 2-hop neighbors.
        self.conv2 = GCNConv(h1, h2, add_self_loops=True, normalize=True)

        self.dropout = nn.Dropout(dropout)
        # Final classification head — kept as a plain Linear so the
        # per-layer parameter counts stay comparable to the MLP baseline.
        self.classifier = nn.Linear(h2, num_classes)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (N, in_dim) fused node features.
            edge_index: (2, E) long tensor of graph edges in PyG format
                (row i = source, row 1 = target). Must include self-loops or
                rely on add_self_loops=True inside GCNConv.

        Returns:
            logits of shape (N, num_classes).
        """
        h = self.conv1(x, edge_index)
        h = torch.relu(h)
        h = self.dropout(h)

        h = self.conv2(h, edge_index)
        h = torch.relu(h)
        h = self.dropout(h)

        return self.classifier(h)

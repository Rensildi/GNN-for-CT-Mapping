"""Two-layer GCN classifier """
from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import GCNConv


class GCNClassifier(nn.Module):
    """GCN head over the KNN nodule-similarity graph."""

    def __init__(
        self,
        in_dim: int = 256,
        hidden_dims: tuple[int, int] = (128, 64),
        num_classes: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        h1, h2 = hidden_dims
        self.conv1 = GCNConv(in_dim, h1, add_self_loops=True, normalize=True)
        self.conv2 = GCNConv(h1, h2, add_self_loops=True, normalize=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(h2, num_classes)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x, edge_index)
        h = torch.relu(h)
        h = self.dropout(h)
        h = self.conv2(h, edge_index)
        h = torch.relu(h)
        h = self.dropout(h)
        return self.classifier(h)

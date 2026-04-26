"""Parameter-matched MLP baseline for Rensildi's ResNet18 branch."""
from __future__ import annotations

import torch
from torch import nn


class MLPClassifier(nn.Module):
    """Independent classifier that consumes the same 256-D fused features as the GCN."""

    def __init__(
        self,
        in_dim: int = 256,
        hidden_dims: tuple[int, int] = (128, 64),
        num_classes: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        h1, h2 = hidden_dims
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(h2, num_classes),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor | None = None) -> torch.Tensor:
        del edge_index
        return self.net(x)

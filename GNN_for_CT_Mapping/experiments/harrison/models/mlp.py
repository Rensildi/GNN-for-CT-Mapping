"""MLP baseline for Experiment 1.

The MLP receives the same fused 256-dim node features as the GCN but has no
access to graph structure — each nodule is classified independently. This is
the control that isolates the effect of message passing in the GCN.

Architecture (from documentation/execution_plan_experiments.md §1.3):
    Linear(256, 128) -> ReLU -> Dropout(0.3)
    Linear(128, 64)  -> ReLU -> Dropout(0.3)
    Linear(64, 2)
"""
from __future__ import annotations

import torch
from torch import nn


class MLPClassifier(nn.Module):
    """Two-hidden-layer MLP that consumes fused node features.

    The parameter count is kept close to the GCN's so capacity isn't the
    explanation if one model wins. Forward pass is identical in interface to
    the GCN so the training loop can swap between them without branching.
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
        # nn.Sequential keeps the layer stack readable and matches the plan's
        # spec line-for-line.
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
        """Forward pass.

        Args:
            x: (N, in_dim) fused node features.
            edge_index: accepted but ignored — kept for signature parity with
                the GCN so callers don't have to branch on model type.

        Returns:
            logits of shape (N, num_classes).
        """
        del edge_index  # explicitly document that graph structure is unused.
        return self.net(x)

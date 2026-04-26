"""Stage 1 multi-modal feature fusion

This is the same fusion idea as Harrison's implementation: image features,
8 LIDC radiologist attributes, and spatial centroid coordinates are combined
into one 256-D node vector that the MLP or GCN consumes. 
"""
from __future__ import annotations

import math

import torch
from torch import nn


LIDC_ATTRIBUTE_VOCAB = {
    "subtlety": 5,
    "internalStructure": 4,
    "calcification": 6,
    "sphericity": 5,
    "margin": 5,
    "lobulation": 5,
    "spiculation": 5,
    "texture": 5,
}
LIDC_ATTRIBUTE_NAMES: tuple[str, ...] = tuple(LIDC_ATTRIBUTE_VOCAB.keys())


def sinusoidal_positional_encoding(coords: torch.Tensor, dim: int) -> torch.Tensor:
    """Transformer-style sinusoidal positional encoding for 3D coordinates.

    Args:
        coords: ``(N, 3)`` centroid coordinates in millimeters.
        dim: even encoding width per axis.

    Returns:
        ``(N, 3 * dim)`` encoded spatial features.
    """
    if dim % 2 != 0:
        raise ValueError(f"dim must be even, got {dim}")
    half = dim // 2
    freqs = torch.exp(
        torch.arange(half, device=coords.device, dtype=coords.dtype)
        * (-math.log(10000.0) / half)
    )
    scaled = coords.unsqueeze(-1) * freqs
    encoded = torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1)
    return encoded.view(coords.shape[0], -1)


class MultiModalFusion(nn.Module):
    """Fuse ResNet18 image features, LIDC attributes, and coordinates."""

    def __init__(
        self,
        image_dim: int = 512,
        node_feature_dim: int = 256,
        clinical_embed_dim: int = 8,
        spatial_dim_per_axis: int = 16,
        use_image: bool = True,
        use_clinical: bool = True,
        use_spatial: bool = True,
    ) -> None:
        super().__init__()
        if not (use_image or use_clinical or use_spatial):
            raise ValueError("At least one feature branch must be enabled.")

        self.use_image = use_image
        self.use_clinical = use_clinical
        self.use_spatial = use_spatial
        self.spatial_dim_per_axis = spatial_dim_per_axis

        concat_dim = 0
        if use_image:
            self.image_proj = nn.Linear(image_dim, node_feature_dim)
            concat_dim += node_feature_dim
        if use_clinical:
            self.attr_embeddings = nn.ModuleDict({
                name: nn.Embedding(vocab_size, clinical_embed_dim)
                for name, vocab_size in LIDC_ATTRIBUTE_VOCAB.items()
            })
            concat_dim += clinical_embed_dim * len(LIDC_ATTRIBUTE_VOCAB)
        if use_spatial:
            concat_dim += 3 * spatial_dim_per_axis

        self.norm = nn.LayerNorm(concat_dim)
        self.out_proj = nn.Linear(concat_dim, node_feature_dim)

    def forward(
        self,
        image_features: torch.Tensor | None = None,
        attributes: torch.Tensor | None = None,
        coords: torch.Tensor | None = None,
    ) -> torch.Tensor:
        parts: list[torch.Tensor] = []

        if self.use_image:
            if image_features is None:
                raise ValueError("image_features required when use_image=True")
            parts.append(self.image_proj(image_features))

        if self.use_clinical:
            if attributes is None:
                raise ValueError("attributes required when use_clinical=True")
            attr_parts = [
                self.attr_embeddings[name](attributes[:, i])
                for i, name in enumerate(LIDC_ATTRIBUTE_NAMES)
            ]
            parts.append(torch.cat(attr_parts, dim=-1))

        if self.use_spatial:
            if coords is None:
                raise ValueError("coords required when use_spatial=True")
            parts.append(sinusoidal_positional_encoding(coords, self.spatial_dim_per_axis))

        fused = torch.cat(parts, dim=-1)
        return self.out_proj(self.norm(fused))

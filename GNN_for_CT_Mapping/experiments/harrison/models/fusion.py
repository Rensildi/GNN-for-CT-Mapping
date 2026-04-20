"""Stage 1 multi-modal feature fusion.

Combines three per-nodule feature branches into a unified node feature
vector that Stage 2 (MLP or GCN) consumes:

    1. Image branch    : frozen Med3D ResNet-50 features projected to a
                         trainable embedding.
    2. Clinical branch : the 8 LIDC-IDRI radiologist attributes
                         (subtlety, internal structure, calcification,
                         sphericity, margin, lobulation, spiculation,
                         texture) each indexed into its own learned
                         nn.Embedding.
    3. Spatial branch  : nodule centroid (x, y, z) encoded via sinusoidal
                         positional encoding.

All branches are concatenated, LayerNorm'd, and projected to the plan's
node_feature_dim (default 256). See
`documentation/execution_plan_experiments.md` §0.2.

The image-branch dimensionality is configurable so the same class can
accept Med3D (~2048) or — once Experiment 3 runs — FMCIB (4096) features.
"""
from __future__ import annotations

import math
import torch
from torch import nn


# LIDC-IDRI attribute vocabulary sizes.
# These are the integer rating scales each radiologist applied per nodule.
# Values are 1-indexed in the LIDC XML; embeddings are indexed at rating-1
# (so the Embedding's num_embeddings equals the top of the rating scale).
#
# Keys use the LIDC XML schema's camelCase tag names so they match
# `scripts/parse_lidc_xml.py::ATTRIBUTE_TAGS` and the `<name>_mean`
# columns emitted by `scripts/preprocess.py` without any translation.
LIDC_ATTRIBUTE_VOCAB = {
    "subtlety": 5,           # 1–5
    "internalStructure": 4,  # 1–4
    "calcification": 6,      # 1–6
    "sphericity": 5,         # 1–5
    "margin": 5,             # 1–5
    "lobulation": 5,         # 1–5
    "spiculation": 5,        # 1–5
    "texture": 5,            # 1–5
}
# Kept as an ordered tuple so concatenation order is stable across runs.
LIDC_ATTRIBUTE_NAMES: tuple[str, ...] = tuple(LIDC_ATTRIBUTE_VOCAB.keys())


def sinusoidal_positional_encoding(coords: torch.Tensor, dim: int) -> torch.Tensor:
    """Transformer-style sinusoidal positional encoding over 3D coordinates.

    For each input coordinate (x, y, z), produces `dim` features built from
    sin/cos at geometrically-spaced frequencies. Applied per-axis and
    concatenated, so the output has dim = 3 * dim features (one block per
    axis). This is a non-learned encoding — the downstream LayerNorm + linear
    projection is what gives the model flexibility to reweight axes.

    Args:
        coords: (N, 3) float tensor of nodule centroids in mm.
        dim:    per-axis encoding dimensionality. Must be even.

    Returns:
        (N, 3 * dim) tensor of sinusoidal features.
    """
    if dim % 2 != 0:
        raise ValueError(f"dim must be even, got {dim}")

    # Half-dim sized frequency bank, shared across all three axes.
    half = dim // 2
    # Geometrically-spaced wavelengths from 2π to ~2π * 10_000, matching the
    # Transformer convention. Using log-space then exp keeps it numerically
    # stable.
    freqs = torch.exp(
        torch.arange(half, device=coords.device, dtype=coords.dtype)
        * (-math.log(10000.0) / half)
    )
    # (N, 3, 1) * (half,) -> (N, 3, half): each axis gets the full freq bank.
    scaled = coords.unsqueeze(-1) * freqs  # broadcast over axes.
    # Interleave sin/cos then flatten the per-axis dimension for the concat.
    encoded = torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1)
    # (N, 3, dim) -> (N, 3 * dim).
    return encoded.view(coords.shape[0], -1)


class MultiModalFusion(nn.Module):
    """Fuse image, clinical, and spatial branches into a single node vector.

    The image branch is typically driven by cached frozen-encoder features
    (see models/med3d.py and scripts/extract_features.py) — this module does
    NOT run the CNN itself. The trainable linear projection on the image
    vector is part of Stage 2 (see §0.2 of the execution plan).

    For Experiment 3's feature-modality ablation, the caller can pass
    use_clinical=False / use_spatial=False to drop those branches. All
    branches still feed a projection to node_feature_dim so the GCN
    architecture remains fixed across cells.
    """

    def __init__(
        self,
        image_dim: int,
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

        # Total feature width after concatenation; computed piecewise so the
        # linear projection below matches whatever branches are active.
        concat_dim = 0

        if use_image:
            # Image features come in at whatever the encoder outputs (2048
            # for Med3D ResNet-50, 4096 for FMCIB). Projecting to
            # node_feature_dim here gives the downstream fusion a consistent
            # width regardless of encoder.
            self.image_proj = nn.Linear(image_dim, node_feature_dim)
            concat_dim += node_feature_dim

        if use_clinical:
            # One Embedding per LIDC attribute. Indexed at rating-1 (LIDC
            # ratings are 1-based; Embedding is 0-based).
            self.attr_embeddings = nn.ModuleDict({
                name: nn.Embedding(vocab_size, clinical_embed_dim)
                for name, vocab_size in LIDC_ATTRIBUTE_VOCAB.items()
            })
            concat_dim += clinical_embed_dim * len(LIDC_ATTRIBUTE_VOCAB)

        if use_spatial:
            # sin+cos -> dim per axis; 3 axes -> 3 * dim total.
            concat_dim += 3 * spatial_dim_per_axis

        # LayerNorm before the final projection — the branches have very
        # different natural scales and this is the cheapest way to align
        # them without per-branch normalization.
        self.norm = nn.LayerNorm(concat_dim)
        self.out_proj = nn.Linear(concat_dim, node_feature_dim)

    def forward(
        self,
        image_features: torch.Tensor | None = None,
        attributes: torch.Tensor | None = None,
        coords: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Fuse and project the enabled branches.

        Args:
            image_features: (N, image_dim) frozen encoder output per nodule.
                Required when use_image=True.
            attributes:     (N, 8) long tensor of attribute ratings in
                LIDC_ATTRIBUTE_NAMES order, using 0-indexed values (i.e.
                already shifted from the 1-based LIDC ratings). Required
                when use_clinical=True.
            coords:         (N, 3) float tensor of nodule centroids (mm).
                Required when use_spatial=True.

        Returns:
            (N, node_feature_dim) fused node features.
        """
        parts: list[torch.Tensor] = []

        if self.use_image:
            assert image_features is not None, "image_features required when use_image=True"
            parts.append(self.image_proj(image_features))

        if self.use_clinical:
            assert attributes is not None, "attributes required when use_clinical=True"
            # Look up each attribute separately then concatenate. Keeps the
            # per-attribute embedding tables decoupled so one rating scale
            # change doesn't affect the others.
            attr_parts = [
                self.attr_embeddings[name](attributes[:, i])
                for i, name in enumerate(LIDC_ATTRIBUTE_NAMES)
            ]
            parts.append(torch.cat(attr_parts, dim=-1))

        if self.use_spatial:
            assert coords is not None, "coords required when use_spatial=True"
            parts.append(sinusoidal_positional_encoding(coords, self.spatial_dim_per_axis))

        # Concatenate all enabled branches along the feature axis.
        fused = torch.cat(parts, dim=-1)
        # Normalize and project to the canonical node_feature_dim.
        fused = self.norm(fused)
        return self.out_proj(fused)

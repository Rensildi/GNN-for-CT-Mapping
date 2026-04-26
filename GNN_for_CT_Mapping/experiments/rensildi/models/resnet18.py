"""Frozen ResNet18 image"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


RESNET18_FEATURE_DIM = 512


class ResNet18CTEncoder(nn.Module):
    """Frozen ResNet18 feature extractor for CT nodule patches.

    Args:
        weights: ``"imagenet"`` loads torchvision ImageNet weights;
            ``"none"`` uses random initialization. ImageNet weights are not
            medical-domain pretraining, but they give a lightweight baseline
            encoder that is easy to run on a 12 GB GPU.
    """

    feature_dim = RESNET18_FEATURE_DIM

    def __init__(self, weights: Literal["imagenet", "none"] = "imagenet") -> None:
        super().__init__()
        tv_weights = ResNet18_Weights.DEFAULT if weights == "imagenet" else None
        backbone = resnet18(weights=tv_weights)
        backbone.fc = nn.Identity()
        self.backbone = backbone

        # ImageNet normalization constants for pretrained torchvision ResNet18.
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self._freeze()

    def _freeze(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

    def train(self, mode: bool = True) -> "ResNet18CTEncoder":
        super().train(mode)
        # Keep BatchNorm statistics fixed during feature extraction.
        self.backbone.eval()
        return self

    @staticmethod
    def _central_orthogonal_slices(patches: torch.Tensor) -> torch.Tensor:
        """Convert ``(N,1,D,H,W)`` CT patches to ``(N,3,H,W)`` slices."""
        if patches.ndim == 4:
            # Already a 2D tensor: (N, C, H, W). Make it 3-channel if needed.
            if patches.shape[1] == 1:
                return patches.repeat(1, 3, 1, 1)
            if patches.shape[1] == 3:
                return patches
            raise ValueError(f"Expected 1 or 3 channels for 2D input, got {patches.shape[1]}")

        if patches.ndim != 5 or patches.shape[1] != 1:
            raise ValueError(
                "Expected patches with shape (N, 1, D, H, W) or (N, C, H, W), "
                f"got {tuple(patches.shape)}"
            )

        _, _, d, h, w = patches.shape
        axial = patches[:, 0, d // 2, :, :]       # (N, H, W)
        coronal = patches[:, 0, :, h // 2, :]     # (N, D, W)
        sagittal = patches[:, 0, :, :, w // 2]    # (N, D, H)

        # Resize the non-axial views to the axial plane size before stacking.
        coronal = F.interpolate(coronal.unsqueeze(1), size=(h, w), mode="bilinear", align_corners=False).squeeze(1)
        sagittal = F.interpolate(sagittal.unsqueeze(1), size=(h, w), mode="bilinear", align_corners=False).squeeze(1)
        return torch.stack([axial, coronal, sagittal], dim=1)

    @torch.no_grad()
    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """Return a frozen 512-D feature vector for each CT patch."""
        x = self._central_orthogonal_slices(patches.float())
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        return self.backbone(x)

    @classmethod
    def from_checkpoint(cls, checkpoint_path: Path | None = None, weights: Literal["imagenet", "none"] = "imagenet") -> "ResNet18CTEncoder":
        """Create the encoder and optionally load a local ResNet18 checkpoint.

        The checkpoint may either be a raw state_dict or a dictionary with one
        of the common keys: ``state_dict``, ``model``, or ``net``.
        """
        encoder = cls(weights=weights)
        if checkpoint_path is None:
            return encoder

        raw = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(raw, dict) and "state_dict" in raw:
            state = raw["state_dict"]
        elif isinstance(raw, dict) and "model" in raw:
            state = raw["model"]
        elif isinstance(raw, dict) and "net" in raw:
            state = raw["net"]
        else:
            state = raw

        cleaned = {}
        for k, v in state.items():
            k = k.removeprefix("module.")
            k = k.removeprefix("backbone.")
            if k.startswith("fc."):
                continue
            cleaned[k] = v

        missing, unexpected = encoder.backbone.load_state_dict(cleaned, strict=False)
        missing_non_fc = [k for k in missing if not k.startswith("fc")]
        if missing_non_fc:
            raise RuntimeError(f"Checkpoint missing ResNet18 keys: {missing_non_fc[:5]}")
        if unexpected:
            raise RuntimeError(f"Checkpoint has unexpected keys: {unexpected[:5]}")
        encoder._freeze()
        return encoder

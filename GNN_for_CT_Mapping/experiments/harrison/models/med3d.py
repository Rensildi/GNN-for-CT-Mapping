"""Med3D ResNet-50 image encoder wrapper.

Med3D (Chen et al. 2019) is a 3D ResNet pretrained on 23 medical segmentation
datasets. We use it as a frozen feature extractor — the per-nodule image
branch of Stage 1 — to leverage 3D medical-image priors that would be hard
to learn from LIDC's ~1k labeled nodules end-to-end.

Pretrained weights:
    Tencent/MedicalNet (https://github.com/Tencent/MedicalNet) provides
    resnet_50.pth for the 3D ResNet-50 variant. Download locally, don't
    commit, and point the loader at the file via configs/paths.local.yaml.

Notes on the architecture:
    MedicalNet's resnet.py is a torchvision-style 3D ResNet with the final
    FC layer replaced by a segmentation head. For classification / feature
    extraction we want the post-average-pool 2048-dim vector — see the
    `extract_features` method.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


class Med3DResNet50Encoder(nn.Module):
    """Frozen Med3D ResNet-50 feature extractor.

    Usage:
        encoder = Med3DResNet50Encoder.from_checkpoint(Path("/path/to/resnet_50.pth"))
        feats = encoder(patches)  # (N, 2048) fully pooled features

    This wrapper deliberately does NOT host the ResNet-50 implementation
    itself — MedicalNet's resnet.py lives outside this repo and should be
    added as a vendored submodule or pip install when the full pipeline is
    wired up. Until then, `from_checkpoint` raises with a clear message so
    the error surface is obvious.
    """

    def __init__(self, backbone: nn.Module, feature_dim: int = 2048) -> None:
        super().__init__()
        self.backbone = backbone
        self.feature_dim = feature_dim
        self._freeze()

    def _freeze(self) -> None:
        """Freeze every parameter and switch to eval mode.

        Stage 1 is frozen — gradients must not flow into the backbone during
        Stage 2 training. This also turns off dropout/batchnorm updates that
        would otherwise quietly adapt.
        """
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

    def train(self, mode: bool = True) -> "Med3DResNet50Encoder":
        # Override so `.train()` on a parent module doesn't flip the frozen
        # backbone into training mode.
        super().train(mode)
        self.backbone.eval()
        return self

    @torch.no_grad()
    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """Run the frozen backbone and return pooled features.

        Args:
            patches: (N, 1, D, H, W) float tensor of preprocessed CT patches
                (HU-clipped, normalized, isotropic 1 mm spacing).

        Returns:
            (N, feature_dim) float tensor of pooled features.
        """
        # MedicalNet's resnet returns a 5D tensor (N, C, D, H, W). We global
        # average pool over the spatial dims to get a classification-ready
        # feature vector.
        features = self.backbone(patches)
        if features.ndim == 5:
            features = features.mean(dim=(2, 3, 4))
        return features

    @classmethod
    def from_checkpoint(cls, checkpoint_path: Path) -> "Med3DResNet50Encoder":
        """Load Tencent/MedicalNet pretrained weights into a 3D ResNet-50.

        This loader is intentionally left as a TODO skeleton — wiring up
        MedicalNet's resnet.py cleanly requires vendoring that module and
        handling the checkpoint's `module.` prefix (from DataParallel
        training). Complete once the weights are downloaded locally.
        """
        raise NotImplementedError(
            "Med3D loader not yet wired up. Next steps: (1) clone "
            "https://github.com/Tencent/MedicalNet, (2) copy its models/"
            "resnet.py into this package, (3) here instantiate "
            "resnet50(sample_input_D=48, sample_input_H=48, sample_input_W=48, "
            "num_seg_classes=2), (4) load state_dict from "
            f"{checkpoint_path} stripping the 'module.' prefix."
        )

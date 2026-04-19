"""Med3D ResNet-50 frozen image encoder for Stage 1.

Med3D (Chen et al. 2019) is a 3D ResNet pretrained by Tencent on 23 medical
segmentation datasets. We use it as a frozen feature extractor — the
per-nodule image branch of Stage 1 — to leverage 3D medical-image priors
that would be hard to learn from LIDC's ~1k labeled nodules end-to-end.

Pretrained weights:
    The vendored `vendor/medicalnet_resnet.py` defines the architecture.
    Tencent hosts the pretrained weights on Google Drive / Tencent Weiyun
    (see https://github.com/Tencent/MedicalNet). Download `resnet_50.pth`
    manually and point this loader at the file — the download is not
    wget-friendly so we don't automate it. Keep the `.pth` out of git (the
    repo's `.gitignore` already excludes `*.pth`).

Feature-extraction design:
    MedicalNet's resnet.py was trained as a segmentation model, so its
    `forward` ends with a `conv_seg` decoder that outputs a per-voxel
    prediction. For classification we stop after `layer4` (the last 2048-
    channel stage) and apply global average pooling to get a single
    feature vector per patch. `conv_seg` weights are discarded at load
    time — they'd only be dead weight in VRAM otherwise.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from .vendor.medicalnet_resnet import resnet50


# The final ResNet stage outputs Bottleneck.expansion * 512 channels.
# Bottleneck.expansion is 4 in MedicalNet's resnet.py, so features are 2048-D.
MED3D_RESNET50_FEATURE_DIM = 2048


class Med3DResNet50Encoder(nn.Module):
    """Frozen Med3D ResNet-50 feature extractor.

    The wrapper owns the underlying ResNet and overrides `forward` to stop
    at the last conv stage (layer4) then global-average-pool over spatial
    dims. Everything stays on `no_grad` — gradients must not flow into the
    backbone during Stage 2 training.

    Usage:
        encoder = Med3DResNet50Encoder.from_checkpoint(
            Path("/path/to/Med3D/resnet_50.pth")
        ).to(device)
        with torch.no_grad():
            feats = encoder(patches_5d)  # (N, 2048)
    """

    feature_dim = MED3D_RESNET50_FEATURE_DIM

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self._freeze()

    def _freeze(self) -> None:
        """Freeze every parameter and force the backbone into eval mode.

        BatchNorm running stats must also stop updating, hence eval(). We
        override `train()` below so `.train()` on a parent module doesn't
        silently reactivate the backbone.
        """
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

    def train(self, mode: bool = True) -> "Med3DResNet50Encoder":
        super().train(mode)
        self.backbone.eval()
        return self

    @torch.no_grad()
    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """Run the frozen backbone and return globally-pooled 2048-D features.

        Args:
            patches: (N, 1, D, H, W) float tensor of preprocessed CT patches
                (HU-clipped, normalized, isotropic 1 mm spacing — see
                scripts/extract_features.py).

        Returns:
            (N, 2048) float tensor of pooled features.
        """
        bb = self.backbone
        x = bb.conv1(patches)
        x = bb.bn1(x)
        x = bb.relu(x)
        x = bb.maxpool(x)
        x = bb.layer1(x)
        x = bb.layer2(x)
        x = bb.layer3(x)
        x = bb.layer4(x)
        # x shape after layer4: (N, 2048, D', H', W'). Global-avg-pool over
        # the spatial dims yields a classification-ready (N, 2048) vector.
        return x.mean(dim=(2, 3, 4))

    @classmethod
    def build_backbone(
        cls,
        sample_input_D: int = 48,
        sample_input_H: int = 48,
        sample_input_W: int = 48,
    ) -> nn.Module:
        """Construct the MedicalNet ResNet-50 backbone with random init.

        `sample_input_*` are required by the upstream constructor but only
        affect `conv_seg` (which we don't use). `num_seg_classes=2` is a
        dummy — any value works since we discard the segmentation head.
        `no_cuda=True` avoids upstream's CUDA-specific casts inside
        `downsample_basic_block`; we stick to shortcut_type='B' (default)
        so that branch isn't exercised anyway.
        """
        return resnet50(
            sample_input_D=sample_input_D,
            sample_input_H=sample_input_H,
            sample_input_W=sample_input_W,
            num_seg_classes=2,
            shortcut_type="B",
            no_cuda=True,
        )

    @classmethod
    def from_checkpoint(cls, checkpoint_path: Path) -> "Med3DResNet50Encoder":
        """Load Tencent/MedicalNet pretrained ResNet-50 weights.

        The checkpoint was saved from a DataParallel-wrapped model, so all
        keys have a `module.` prefix. It also contains `conv_seg.*` weights
        from the segmentation head which we discard. Loading is done with
        `strict=False` to tolerate the drops; we assert afterwards that
        every *backbone* parameter was populated so we fail loudly if the
        checkpoint has a different layout than expected.
        """
        backbone = cls.build_backbone()

        raw = torch.load(checkpoint_path, map_location="cpu")
        # Checkpoints are sometimes wrapped — peel off common envelopes.
        if isinstance(raw, dict) and "state_dict" in raw:
            state = raw["state_dict"]
        elif isinstance(raw, dict) and "net" in raw:
            state = raw["net"]
        else:
            state = raw

        cleaned: dict[str, torch.Tensor] = {}
        for k, v in state.items():
            # Strip the DataParallel prefix.
            if k.startswith("module."):
                k = k[len("module."):]
            # Drop the segmentation head — we stop at layer4.
            if k.startswith("conv_seg."):
                continue
            cleaned[k] = v

        missing, unexpected = backbone.load_state_dict(cleaned, strict=False)
        # Missing should only cover `conv_seg.*` (which we skip deliberately
        # because our forward doesn't go through it). Anything else means
        # the checkpoint is incompatible and we want a loud failure.
        missing_backbone = [k for k in missing if not k.startswith("conv_seg")]
        if missing_backbone:
            raise RuntimeError(
                f"Med3D checkpoint missing backbone keys: {missing_backbone[:5]}..."
            )
        # Unexpected keys indicate the checkpoint has extra weights we
        # didn't account for — surface as a warning via assertion-like
        # behavior so nothing silently loads partially.
        if unexpected:
            raise RuntimeError(
                f"Med3D checkpoint has unexpected keys (first 5 shown): {unexpected[:5]}. "
                "Update from_checkpoint's key-filter if upstream changed."
            )

        return cls(backbone)

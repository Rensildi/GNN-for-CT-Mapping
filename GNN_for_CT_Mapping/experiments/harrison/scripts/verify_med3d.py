"""Smoke-check a downloaded Med3D ResNet-50 checkpoint.

Run this after downloading `resnet_50.pth` from Tencent/MedicalNet. It:

    1. Builds the architecture.
    2. Loads the checkpoint through the same code path that Stage 1 uses.
    3. Runs a forward pass on a random 48^3 patch.
    4. Prints feature statistics (shape, mean, std, min, max) so you can
       eyeball that the weights produce a sensible, non-collapsed output.

If any step fails the script exits non-zero — use this as a gate before
kicking off the long `extract_features.py` run.

Usage:
    python -m GNN_for_CT_Mapping.experiments.harrison.scripts.verify_med3d \\
        --checkpoint /path/to/Med3D/resnet_50.pth
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ..models.med3d import MED3D_RESNET50_FEATURE_DIM, Med3DResNet50Encoder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Path to Med3D resnet_50.pth")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Loading {args.checkpoint} ...")
    encoder = Med3DResNet50Encoder.from_checkpoint(args.checkpoint).to(args.device)

    frozen_params = sum(1 for p in encoder.parameters() if not p.requires_grad)
    all_params = sum(1 for _ in encoder.parameters())
    print(f"  parameters: {frozen_params}/{all_params} frozen")
    assert frozen_params == all_params, "Stage 1 backbone must be fully frozen."

    # Run on a pseudo-patch shaped like the real pipeline's patches. Random
    # input means we can't compare to a reference output, but we can confirm
    # the forward pass runs and the features aren't degenerate (all zeros or
    # wildly large).
    torch.manual_seed(0)
    patch = torch.randn(2, 1, 48, 48, 48, device=args.device)
    with torch.no_grad():
        feats = encoder(patch)

    assert feats.shape == (2, MED3D_RESNET50_FEATURE_DIM), feats.shape
    print(f"  forward OK — features {tuple(feats.shape)}")
    print(f"    mean={feats.mean().item():+.4f}  std={feats.std().item():.4f}")
    print(f"    min={feats.min().item():+.4f}  max={feats.max().item():+.4f}")

    # Sanity: consecutive runs on identical input should be bit-identical
    # because everything is frozen and eval mode is set. Any drift here
    # would signal BatchNorm running stats not being frozen.
    with torch.no_grad():
        feats2 = encoder(patch)
    if not torch.equal(feats, feats2):
        raise SystemExit("Non-deterministic forward pass — frozen backbone is misconfigured.")
    print("  determinism OK — repeated forward pass is bit-identical.")


if __name__ == "__main__":
    main()

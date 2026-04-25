"""Cache FMCIB features for every nodule.

Mirrors `extract_features.py` (Med3D) in structure, but runs Pai et al.
2024's FMCIB foundation model as the image encoder. The underlying
architecture is a wide 3D ResNet-50 (widen_factor=2) pretrained with
SimCLR on ~11k cancer-relevant CTs; the loader expects 50³ patches at
isotropic 1 mm³ spacing, normalized as (HU + 1024) / 3072 per FMCIB's
own preprocessing (`fmcib.preprocessing.get_transforms`).

We reuse our existing DICOM loader and resampling code — it's already
isotropic-1mm and HU-faithful — but we extract a 50³ patch (vs. Med3D's
48³) and apply FMCIB's own normalization rather than the lung HU window
we use for Med3D. This matches FMCIB's pretraining distribution.

Run:
    python -m GNN_for_CT_Mapping.experiments.harrison.scripts.extract_features_fmcib \\
        --nodules    GNN_for_CT_Mapping/data/nodules.parquet \\
        --metadata   /media/talos/SchoolHD/deep_learning/LIDC-IDRI/metadata/metadata.csv \\
        --weights-dir GNN_for_CT_Mapping/outputs/fmcib \\
        --out        GNN_for_CT_Mapping/outputs/features/fmcib.parquet
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .dicom_loader import extract_patch, load_series, world_to_voxel
from .extract_features import build_isotropic_ct_volume
from .preprocess import build_series_to_dicom_map


# FMCIB preprocessing constants (from fmcib.preprocessing.get_transforms):
#   NormalizeIntensityd(subtrahend=-1024, divisor=3072)
# i.e. output = (HU + 1024) / 3072, which maps [-1024, 2048] -> [0, 1].
FMCIB_HU_SUBTRAHEND = -1024.0
FMCIB_HU_DIVISOR = 3072.0

# FMCIB's pretrained patch size. The plan refers to 50³; confirmed by
# `get_transforms(spatial_size=(50, 50, 50))` defaults in the upstream
# source.
FMCIB_PATCH_SIZE = (50, 50, 50)


def _normalize_fmcib(patch: np.ndarray) -> np.ndarray:
    """Apply FMCIB's exact HU normalization — no clipping."""
    return ((patch.astype(np.float32) - FMCIB_HU_SUBTRAHEND) / FMCIB_HU_DIVISOR)


def _load_fmcib_model(weights_dir: Path) -> torch.nn.Module:
    """Load FMCIB from the weights file in `weights_dir`.

    fmcib_model() hardcodes its weight-file lookup to `cwd / model_weights.torch`
    and downloads if missing. We chdir into `weights_dir` so the model picks
    up the pre-downloaded file without re-downloading 738 MB.
    """
    from fmcib.models import fmcib_model
    original_cwd = Path(os.getcwd())
    os.chdir(weights_dir)
    try:
        model = fmcib_model(eval_mode=True)
    finally:
        os.chdir(original_cwd)
    for p in model.parameters():
        p.requires_grad = False
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodules", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True,
                        help="Path to LIDC-IDRI metadata/metadata.csv")
    parser.add_argument("--weights-dir", type=Path, required=True,
                        help="Directory containing FMCIB model_weights.torch")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for encoder forward passes. FMCIB is "
                             "~230 MB + activations; 8 patches comfortably fits "
                             "on the 3060.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_fmcib_model(args.weights_dir).to(device)

    nodules = pd.read_parquet(args.nodules)
    series_to_dir = build_series_to_dicom_map(args.metadata)

    out_rows: list[dict] = []

    for series_uid, series_group in tqdm(nodules.groupby("series_uid"), desc="series"):
        meta = series_to_dir.get(series_uid)
        if meta is None or not meta["dicom_dir"].exists():
            continue
        dicom_dir = meta["dicom_dir"]

        try:
            raw_volume = load_series(dicom_dir)
            iso_volume = build_isotropic_ct_volume(raw_volume)
        except Exception:
            continue

        patches: list[np.ndarray] = []
        nodule_ids: list[str] = []
        for _, row in series_group.iterrows():
            centroid = (
                float(row["centroid_x_mm"]),
                float(row["centroid_y_mm"]),
                float(row["centroid_z_mm"]),
            )
            z_idx, y_idx, x_idx = world_to_voxel(centroid, iso_volume)
            patch = extract_patch(iso_volume, (z_idx, y_idx, x_idx), size=FMCIB_PATCH_SIZE)
            patches.append(_normalize_fmcib(patch))
            nodule_ids.append(str(row["nodule_id"]))

        features: list[np.ndarray] = []
        for start in range(0, len(patches), args.batch_size):
            chunk = np.stack(patches[start:start + args.batch_size], axis=0)
            tensor = torch.as_tensor(chunk, dtype=torch.float32, device=device).unsqueeze(1)
            with torch.no_grad():
                feats = model(tensor).cpu().numpy()
            features.append(feats)
        features_arr = np.concatenate(features, axis=0)

        for nid, f in zip(nodule_ids, features_arr):
            out_rows.append({"nodule_id": nid, "features": f.astype(np.float32).tolist()})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out_rows).to_parquet(args.out, index=False)
    print(f"Wrote {len(out_rows)} FMCIB feature rows to {args.out}")


if __name__ == "__main__":
    main()

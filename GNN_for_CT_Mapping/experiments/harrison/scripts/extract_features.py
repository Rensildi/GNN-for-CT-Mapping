"""Cache frozen Med3D ResNet-50 features for every nodule.

§0.2 of the execution plan: run the image backbone once over all nodules,
save the pooled feature vectors to parquet, and let every subsequent
experiment load from that cache. The frozen backbone means this is the
expensive step — caching keeps downstream training-loop iterations cheap.

Run:
    python -m experiments.harrison.scripts.extract_features \\
        --nodules   GNN_for_CT_Mapping/data/nodules.parquet \\
        --metadata  /media/talos/SchoolHD/deep_learning/LIDC-IDRI/metadata/metadata.csv \\
        --checkpoint /path/to/Med3D/resnet_50.pth \\
        --out       GNN_for_CT_Mapping/outputs/features/med3d_resnet50.parquet

Preprocessing applied to every patch before the encoder sees it:
    - clamp HU to [-1000, 400] (air..soft tissue window)
    - normalize to [0, 1] by (hu + 1000) / 1400
    - resample to isotropic 1 mm spacing via scipy.ndimage.zoom
    - crop/pad a 48 x 48 x 48 patch centered on the nodule centroid
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import zoom
from tqdm import tqdm

from ..models.med3d import Med3DResNet50Encoder
from .dicom_loader import (
    CTVolume,
    extract_patch,
    load_series,
    world_to_voxel,
)
from .preprocess import build_series_to_dicom_map


HU_MIN = -1000.0
HU_MAX = 400.0
PATCH_VOXEL_SIZE = (48, 48, 48)


def hu_normalize(patch: np.ndarray) -> np.ndarray:
    """Clip to the lung HU window and scale to [0, 1] float32."""
    clipped = np.clip(patch.astype(np.float32), HU_MIN, HU_MAX)
    return (clipped - HU_MIN) / (HU_MAX - HU_MIN)


def resample_to_isotropic(volume: CTVolume) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Zoom the volume to isotropic 1 mm spacing.

    Returns the resampled volume and the new spacing (always 1.0, 1.0, 1.0).
    Uses linear interpolation — the CT volume is already at 0.5–2.5 mm Z
    spacing so this is only a small adjustment.
    """
    z_sp, y_sp, x_sp = volume.voxel_spacing_mm
    zoom_factors = (z_sp, y_sp, x_sp)
    resampled = zoom(volume.volume.astype(np.float32), zoom_factors, order=1)
    return resampled, (1.0, 1.0, 1.0)


def extract_one(volume: CTVolume, centroid_mm: tuple[float, float, float]) -> np.ndarray:
    """Produce a preprocessed 48^3 patch ready for the encoder.

    The resampling step above is done *per series*, not per nodule, so
    callers should batch nodules from the same series together. This
    function assumes `volume` is already in the space you want.
    """
    z_idx, y_idx, x_idx = world_to_voxel(centroid_mm, volume)
    patch = extract_patch(volume, (z_idx, y_idx, x_idx), size=PATCH_VOXEL_SIZE)
    return hu_normalize(patch)


def build_isotropic_ct_volume(original: CTVolume) -> CTVolume:
    """Re-wrap a resampled volume as a new CTVolume with spacing 1 mm."""
    resampled, new_spacing = resample_to_isotropic(original)
    # After resampling, world_to_voxel needs the origin in the original mm
    # coordinate system (unchanged) with the new spacing (1,1,1).
    return CTVolume(
        volume=resampled.astype(np.int16),
        voxel_spacing_mm=new_spacing,
        origin_mm=original.origin_mm,
        series_uid=original.series_uid,
        sop_instance_uids=original.sop_instance_uids,
        image_positions_z=original.image_positions_z,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodules", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True,
                        help="Path to LIDC-IDRI metadata/metadata.csv")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Med3D ResNet-50 .pth weights")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Number of patches per encoder forward pass")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = Med3DResNet50Encoder.from_checkpoint(args.checkpoint).to(device)

    nodules = pd.read_parquet(args.nodules)
    series_to_dir = build_series_to_dicom_map(args.metadata)

    out_rows: list[dict] = []

    # Group by series so we only load each CT volume once; many nodules
    # share a series.
    for series_uid, series_group in tqdm(nodules.groupby("series_uid"), desc="series"):
        meta = series_to_dir.get(series_uid)
        # build_series_to_dicom_map returns {'dicom_dir': Path, 'patient_id': str}
        # since we don't re-read patient_id from DICOM here — see preprocess.py.
        if meta is None or not meta["dicom_dir"].exists():
            continue
        dicom_dir = meta["dicom_dir"]

        try:
            raw_volume = load_series(dicom_dir)
            iso_volume = build_isotropic_ct_volume(raw_volume)
        except Exception:
            continue

        # Extract patches for this series, then batch through the encoder.
        patches: list[np.ndarray] = []
        nodule_ids: list[str] = []
        for _, row in series_group.iterrows():
            centroid = (
                float(row["centroid_x_mm"]),
                float(row["centroid_y_mm"]),
                float(row["centroid_z_mm"]),
            )
            patches.append(extract_one(iso_volume, centroid))
            nodule_ids.append(str(row["nodule_id"]))

        # Encoder expects (N, 1, D, H, W). Push in mini-batches so 3060
        # VRAM stays happy even with long series (~50 nodules occasionally).
        features: list[np.ndarray] = []
        for start in range(0, len(patches), args.batch_size):
            chunk = np.stack(patches[start:start + args.batch_size], axis=0)
            tensor = torch.as_tensor(chunk, dtype=torch.float32, device=device).unsqueeze(1)
            with torch.no_grad():
                feats = encoder(tensor).cpu().numpy()
            features.append(feats)
        features_arr = np.concatenate(features, axis=0)

        for nid, f in zip(nodule_ids, features_arr):
            out_rows.append({"nodule_id": nid, "features": f.astype(np.float32).tolist()})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out_rows).to_parquet(args.out, index=False)
    print(f"Wrote {len(out_rows)} feature rows to {args.out}")


if __name__ == "__main__":
    main()

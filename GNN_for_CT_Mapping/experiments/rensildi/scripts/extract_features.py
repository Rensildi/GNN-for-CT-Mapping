"""Cache frozen ResNet18 features for every LIDC nodule."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import torch
from scipy.ndimage import zoom
from tqdm import tqdm

from ..models.resnet18 import ResNet18CTEncoder


HU_MIN = -1000.0
HU_MAX = 400.0
PATCH_VOXEL_SIZE = (48, 48, 48)


@dataclass
class CTVolume:
    volume: np.ndarray                       # shape: (Z, Y, X), HU units
    voxel_spacing_mm: tuple[float, float, float]
    origin_mm: tuple[float, float, float]    # DICOM world coordinate origin (x, y, z)
    series_uid: str


def hu_normalize(patch: np.ndarray) -> np.ndarray:
    """Clip to lung CT range and scale to [0, 1]."""
    clipped = np.clip(patch.astype(np.float32), HU_MIN, HU_MAX)
    return (clipped - HU_MIN) / (HU_MAX - HU_MIN)


def _first_existing_column(columns: list[str], candidates: list[str]) -> str | None:
    normalized = {c.lower().replace(" ", "").replace("_", ""): c for c in columns}
    for candidate in candidates:
        key = candidate.lower().replace(" ", "").replace("_", "")
        if key in normalized:
            return normalized[key]
    return None


def build_series_to_dicom_map(metadata_csv: Path, dicom_root: Path | None = None) -> dict[str, Path]:
    """Map SeriesInstanceUID to a local folder containing DICOM slices.

    TCIA metadata exports are not always named consistently, so this function
    accepts common column spellings. If the directory column is relative,
    --dicom-root is prepended.
    """
    metadata = pd.read_csv(metadata_csv)
    columns = list(metadata.columns)
    series_col = _first_existing_column(columns, [
        "Series UID", "SeriesInstanceUID", "SeriesInstanceUid", "series_uid", "SeriesInstance UID",
    ])
    path_col = _first_existing_column(columns, [
        "File Location", "FileLocation", "File Path", "Path", "Folder", "dicom_dir", "Downloaded File Path",
    ])
    if series_col is None or path_col is None:
        raise ValueError(
            "Could not infer metadata columns. Needed a series UID column and a DICOM folder/path column. "
            f"Columns found: {columns}"
        )

    out: dict[str, Path] = {}
    for _, row in metadata.iterrows():
        series_uid = str(row[series_col])
        p = Path(str(row[path_col]))
        if not p.is_absolute() and dicom_root is not None:
            p = dicom_root / p
        # Some metadata paths point at one file; use the parent folder.
        if p.suffix.lower() in {".dcm", ".dicom"}:
            p = p.parent
        out[series_uid] = p
    return out


def load_series(dicom_dir: Path) -> CTVolume:
    """Load one CT series as a HU volume ordered by physical z position."""
    files = sorted([p for p in dicom_dir.rglob("*") if p.is_file()])
    slices = []
    for p in files:
        try:
            ds = pydicom.dcmread(str(p), force=True)
            if not hasattr(ds, "PixelData"):
                continue
            z = float(ds.ImagePositionPatient[2]) if hasattr(ds, "ImagePositionPatient") else float(ds.InstanceNumber)
            slices.append((z, ds))
        except Exception:
            continue
    if not slices:
        raise FileNotFoundError(f"No readable DICOM slices found in {dicom_dir}")

    slices.sort(key=lambda item: item[0])
    arrays = []
    z_positions = []
    for z, ds in slices:
        arr = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        arrays.append(arr * slope + intercept)
        z_positions.append(z)

    volume = np.stack(arrays, axis=0)
    first = slices[0][1]
    px_spacing = [float(v) for v in first.PixelSpacing]
    if len(z_positions) > 1:
        z_spacing = float(np.median(np.diff(z_positions)))
        z_spacing = abs(z_spacing) if z_spacing != 0 else float(getattr(first, "SliceThickness", 1.0))
    else:
        z_spacing = float(getattr(first, "SliceThickness", 1.0))

    origin = tuple(float(v) for v in first.ImagePositionPatient)
    series_uid = str(getattr(first, "SeriesInstanceUID", dicom_dir.name))
    return CTVolume(volume=volume, voxel_spacing_mm=(z_spacing, px_spacing[0], px_spacing[1]), origin_mm=origin, series_uid=series_uid)


def resample_to_isotropic(volume: CTVolume) -> CTVolume:
    """Resample CT volume to approximately 1 mm isotropic spacing."""
    z_sp, y_sp, x_sp = volume.voxel_spacing_mm
    resampled = zoom(volume.volume.astype(np.float32), (z_sp, y_sp, x_sp), order=1)
    return CTVolume(
        volume=resampled,
        voxel_spacing_mm=(1.0, 1.0, 1.0),
        origin_mm=volume.origin_mm,
        series_uid=volume.series_uid,
    )


def world_to_voxel(centroid_mm: tuple[float, float, float], volume: CTVolume) -> tuple[int, int, int]:
    """Convert DICOM world coordinates (x, y, z) to volume indices (z, y, x)."""
    x, y, z = centroid_mm
    ox, oy, oz = volume.origin_mm
    z_sp, y_sp, x_sp = volume.voxel_spacing_mm
    return (
        int(round((z - oz) / z_sp)),
        int(round((y - oy) / y_sp)),
        int(round((x - ox) / x_sp)),
    )


def extract_patch(volume: CTVolume, center_zyx: tuple[int, int, int], size: tuple[int, int, int] = PATCH_VOXEL_SIZE) -> np.ndarray:
    """Crop a centered patch with zero padding when the nodule is near borders."""
    zc, yc, xc = center_zyx
    dz, dy, dx = size
    starts = [zc - dz // 2, yc - dy // 2, xc - dx // 2]
    ends = [starts[0] + dz, starts[1] + dy, starts[2] + dx]

    patch = np.zeros(size, dtype=np.float32)
    src_slices = []
    dst_slices = []
    for axis, (start, end, max_len) in enumerate(zip(starts, ends, volume.volume.shape)):
        src_start = max(0, start)
        src_end = min(max_len, end)
        dst_start = max(0, -start)
        dst_end = dst_start + max(0, src_end - src_start)
        src_slices.append(slice(src_start, src_end))
        dst_slices.append(slice(dst_start, dst_end))

    patch[tuple(dst_slices)] = volume.volume[tuple(src_slices)]
    return patch


def extract_one(volume: CTVolume, centroid_mm: tuple[float, float, float]) -> np.ndarray:
    center = world_to_voxel(centroid_mm, volume)
    patch = extract_patch(volume, center, PATCH_VOXEL_SIZE)
    return hu_normalize(patch)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodules", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dicom-root", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional local ResNet18 checkpoint")
    parser.add_argument("--weights", choices=["imagenet", "none"], default="imagenet")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = ResNet18CTEncoder.from_checkpoint(args.checkpoint, weights=args.weights).to(device)

    nodules = pd.read_parquet(args.nodules)
    series_to_dir = build_series_to_dicom_map(args.metadata, args.dicom_root)
    out_rows: list[dict] = []

    for series_uid, series_group in tqdm(nodules.groupby("series_uid"), desc="series"):
        dicom_dir = series_to_dir.get(str(series_uid))
        if dicom_dir is None or not dicom_dir.exists():
            continue
        try:
            volume = resample_to_isotropic(load_series(dicom_dir))
        except Exception as exc:
            print(f"Skipping {series_uid}: {exc}")
            continue

        patches: list[np.ndarray] = []
        nodule_ids: list[str] = []
        for _, row in series_group.iterrows():
            centroid = (float(row["centroid_x_mm"]), float(row["centroid_y_mm"]), float(row["centroid_z_mm"]))
            patches.append(extract_one(volume, centroid))
            nodule_ids.append(str(row["nodule_id"]))

        features: list[np.ndarray] = []
        for start in range(0, len(patches), args.batch_size):
            chunk = np.stack(patches[start:start + args.batch_size], axis=0)
            tensor = torch.as_tensor(chunk, dtype=torch.float32, device=device).unsqueeze(1)
            with torch.no_grad():
                feats = encoder(tensor).cpu().numpy()
            features.append(feats)

        if features:
            features_arr = np.concatenate(features, axis=0)
            for nid, feat in zip(nodule_ids, features_arr):
                out_rows.append({"nodule_id": nid, "features": feat.astype(np.float32).tolist()})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out_rows).to_parquet(args.out, index=False)
    print(f"Wrote {len(out_rows)} ResNet18 feature rows to {args.out}")


if __name__ == "__main__":
    main()

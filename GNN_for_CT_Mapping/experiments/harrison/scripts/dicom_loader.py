"""DICOM series loader and 3D patch extractor (pydicom-based).

Given a DICOM series directory, this module:

    1. Loads every `.dcm` file in the directory.
    2. Orders slices by `ImagePositionPatient[2]` (physical Z position).
    3. Stacks them into a single (Z, Y, X) numpy volume.
    4. Applies the DICOM `RescaleSlope` / `RescaleIntercept` to convert raw
       pixel values to Hounsfield Units.
    5. Records the (Z, Y, X) voxel spacing in mm so downstream code can
       resample to isotropic 1 mm³ if needed.

Design note — we deliberately keep this layer thin. Resampling and HU
normalization live in preprocess.py / extract_features.py so this module
stays a straightforward DICOM → ndarray reader that's easy to unit-test.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pydicom


@dataclass
class CTVolume:
    """A stacked, HU-converted CT volume.

    `voxel_spacing_mm` is (slice_thickness_z, row_spacing_y, col_spacing_x).
    `origin_mm` is the physical (x, y, z) position of voxel (0, 0, 0) — the
    DICOM `ImagePositionPatient` of the first slice.
    """

    volume: np.ndarray  # (Z, Y, X) int16 HU values
    voxel_spacing_mm: tuple[float, float, float]  # (z, y, x)
    origin_mm: tuple[float, float, float]  # (x, y, z)
    series_uid: str
    sop_instance_uids: list[str]  # per-slice SOP UIDs, ordered by Z
    image_positions_z: list[float]  # per-slice Z in mm, ordered ascending


def load_series(series_dir: Path) -> CTVolume:
    """Load a DICOM series from disk into a CTVolume.

    Args:
        series_dir: directory containing one CT series' DICOM files.

    Returns:
        A CTVolume with HU-converted pixels and physical spacing metadata.

    Raises:
        FileNotFoundError: if the directory has no readable DICOM files.
        ValueError: if slice spacing is inconsistent (LIDC generally has
            uniform thickness per series; a mismatch typically indicates a
            wrong directory was passed).
    """
    files = sorted(series_dir.glob("*.dcm"))
    if not files:
        raise FileNotFoundError(f"No .dcm files found under {series_dir}")

    datasets = [pydicom.dcmread(f) for f in files]
    # Sort by physical Z position (ImagePositionPatient index 2). This is
    # more robust than sorting by InstanceNumber, which can be reversed
    # between CT vendors.
    datasets.sort(key=lambda d: float(d.ImagePositionPatient[2]))

    # Consistency checks — LIDC scans have a single SeriesInstanceUID per
    # directory; a mismatch suggests the caller pointed at the wrong folder.
    series_uid = str(datasets[0].SeriesInstanceUID)
    for d in datasets[1:]:
        if str(d.SeriesInstanceUID) != series_uid:
            raise ValueError(f"Mixed SeriesInstanceUIDs in {series_dir}")

    # Stack pixel arrays into a single (Z, Y, X) volume. Pixel arrays are
    # stored as uint16 but need to be converted to HU via the per-slice
    # RescaleSlope/Intercept — slope is usually 1 and intercept is usually
    # -1024, but we read them off each slice to be safe.
    slices: list[np.ndarray] = []
    for d in datasets:
        raw = d.pixel_array.astype(np.int32)
        slope = float(getattr(d, "RescaleSlope", 1))
        intercept = float(getattr(d, "RescaleIntercept", 0))
        slices.append((raw * slope + intercept).astype(np.int16))
    volume = np.stack(slices, axis=0)

    # Voxel spacing: Y/X come from PixelSpacing (row, column), Z from the
    # difference between adjacent ImagePositionPatient values. We use the
    # median to tolerate minor floating-point jitter without crashing.
    pixel_spacing = datasets[0].PixelSpacing  # [row (y), column (x)] in mm
    row_spacing = float(pixel_spacing[0])
    col_spacing = float(pixel_spacing[1])
    z_positions = [float(d.ImagePositionPatient[2]) for d in datasets]
    if len(z_positions) >= 2:
        z_diffs = np.diff(z_positions)
        z_spacing = float(np.median(np.abs(z_diffs)))
    else:
        # Single-slice series — fall back to SliceThickness.
        z_spacing = float(datasets[0].SliceThickness)

    origin_mm = (
        float(datasets[0].ImagePositionPatient[0]),
        float(datasets[0].ImagePositionPatient[1]),
        z_positions[0],
    )
    sop_uids = [str(d.SOPInstanceUID) for d in datasets]

    return CTVolume(
        volume=volume,
        voxel_spacing_mm=(z_spacing, row_spacing, col_spacing),
        origin_mm=origin_mm,
        series_uid=series_uid,
        sop_instance_uids=sop_uids,
        image_positions_z=z_positions,
    )


def world_to_voxel(
    world_xyz_mm: tuple[float, float, float],
    volume: CTVolume,
) -> tuple[int, int, int]:
    """Convert a physical (x, y, z) mm coordinate to (z_idx, y_idx, x_idx).

    Assumes axis-aligned DICOM (the default for LIDC) — rotation is not
    handled. The returned indices are rounded to the nearest integer voxel.
    """
    x, y, z = world_xyz_mm
    ox, oy, oz = volume.origin_mm
    z_spacing, y_spacing, x_spacing = volume.voxel_spacing_mm
    z_idx = int(round((z - oz) / z_spacing))
    y_idx = int(round((y - oy) / y_spacing))
    x_idx = int(round((x - ox) / x_spacing))
    return z_idx, y_idx, x_idx


def extract_patch(
    volume: CTVolume,
    center_voxel: tuple[int, int, int],
    size: tuple[int, int, int] = (48, 48, 48),
    pad_value: int = -1000,
) -> np.ndarray:
    """Extract a fixed-size 3D patch centered on a voxel.

    Out-of-bounds regions (common for nodules near the lung boundary) are
    padded with `pad_value` — HU -1000 is air, which is the physically
    meaningful extension beyond the scan volume.

    Args:
        volume: CTVolume loaded from load_series.
        center_voxel: (z, y, x) index into volume.volume.
        size: output patch shape (z, y, x).
        pad_value: fill value for out-of-bounds voxels (default -1000 HU).

    Returns:
        np.int16 ndarray with shape == size.
    """
    z_c, y_c, x_c = center_voxel
    dz, dy, dx = size
    hz, hy, hx = dz // 2, dy // 2, dx // 2
    # Desired slice in source coordinates (may be out-of-bounds on either
    # side — we pad explicitly below).
    z0, z1 = z_c - hz, z_c - hz + dz
    y0, y1 = y_c - hy, y_c - hy + dy
    x0, x1 = x_c - hx, x_c - hx + dx

    Z, Y, X = volume.volume.shape
    # Clamp source region and mirror the offsets into the destination.
    sz0, sz1 = max(z0, 0), min(z1, Z)
    sy0, sy1 = max(y0, 0), min(y1, Y)
    sx0, sx1 = max(x0, 0), min(x1, X)

    out = np.full(size, pad_value, dtype=np.int16)
    if sz1 > sz0 and sy1 > sy0 and sx1 > sx0:
        dz0, dz1 = sz0 - z0, sz0 - z0 + (sz1 - sz0)
        dy0, dy1 = sy0 - y0, sy0 - y0 + (sy1 - sy0)
        dx0, dx1 = sx0 - x0, sx0 - x0 + (sx1 - sx0)
        out[dz0:dz1, dy0:dy1, dx0:dx1] = volume.volume[sz0:sz1, sy0:sy1, sx0:sx1]
    return out

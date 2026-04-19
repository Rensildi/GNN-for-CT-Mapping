"""PyTorch Dataset that serves cached per-nodule feature rows.

The heavy I/O (DICOM reads, Med3D inference) is done once offline by
`preprocess.py` + `extract_features.py`. At training time this Dataset
just loads the resulting parquet files and exposes per-nodule tensors:

    - image_features: (image_dim,) frozen encoder output
    - attributes:     (8,) long tensor of 0-indexed attribute ratings
    - coords:         (3,) float tensor of nodule centroid (mm)
    - label:          scalar long
    - patient_id:     str (for split filtering)

Missing values (attribute or image feature) are filled with sensible
defaults and flagged via masks — Experiment 1 uses a GCN/MLP that
requires a fully populated row, so callers should filter first.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..models.fusion import LIDC_ATTRIBUTE_NAMES


class NoduleDataset(Dataset):
    """Serves cached features for a subset of nodules.

    Args:
        nodules_parquet: path to `nodules.parquet` from preprocess.py.
        features_parquet: path to `med3d_resnet50.parquet` (or FMCIB) from
            extract_features.py. Must have a `nodule_id` column and a
            `features` list/array column of length `image_feature_dim`.
        patient_ids: optional iterable of patient IDs to restrict to — used
            to produce train/val splits via build_splits.py output.
    """

    def __init__(
        self,
        nodules_parquet: Path,
        features_parquet: Path,
        patient_ids: list[str] | None = None,
    ) -> None:
        nodules = pd.read_parquet(nodules_parquet)
        if patient_ids is not None:
            nodules = nodules[nodules["patient_id"].isin(patient_ids)].reset_index(drop=True)

        features = pd.read_parquet(features_parquet)
        # Inner join on nodule_id drops any nodule whose image features
        # weren't cached — the training loop shouldn't see partial rows.
        joined = nodules.merge(features, on="nodule_id", how="inner").reset_index(drop=True)

        if joined.empty:
            raise ValueError(
                "No nodules left after joining nodules.parquet with features. "
                "Did extract_features.py run successfully?"
            )

        self._df = joined

        # Per-attribute column name translation from LIDC's XML tag names
        # (used in fusion.LIDC_ATTRIBUTE_NAMES) to the `_mean` columns
        # written by preprocess.py.
        self._attr_columns = [f"{name}_mean" for name in LIDC_ATTRIBUTE_NAMES]

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        row = self._df.iloc[idx]

        # Image features come out of the parquet as a 1-D numpy array.
        # Casting to float32 here keeps the downstream layers happy on the
        # 3060 — float16 would need autocast setup we're deferring.
        image = torch.as_tensor(np.asarray(row["features"], dtype=np.float32))

        # Attribute means are floats in the 1-based LIDC scale; round to the
        # nearest integer and shift to 0-indexed for the embedding layers.
        # NaN (missing) is treated as the lowest bucket (0) — acceptable
        # since unscored nodules are rare.
        attr_vals = []
        for col in self._attr_columns:
            v = row[col]
            if pd.isna(v):
                attr_vals.append(0)
            else:
                attr_vals.append(max(0, int(round(float(v))) - 1))
        attributes = torch.as_tensor(attr_vals, dtype=torch.long)

        coords = torch.as_tensor(
            [float(row["centroid_x_mm"]), float(row["centroid_y_mm"]), float(row["centroid_z_mm"])],
            dtype=torch.float32,
        )
        label = torch.as_tensor(int(row["label"]), dtype=torch.long)

        return {
            "image_features": image,
            "attributes": attributes,
            "coords": coords,
            "label": label,
            "patient_id": str(row["patient_id"]),
            "nodule_id": str(row["nodule_id"]),
        }


def collate_stack(batch: list[dict]) -> dict[str, torch.Tensor | list[str]]:
    """Simple collate that stacks tensors along a new batch dim.

    We define this explicitly rather than relying on default_collate because
    we keep string fields (patient_id, nodule_id) as lists, not attempted
    tensor conversions.
    """
    out: dict[str, torch.Tensor | list[str]] = {
        "image_features": torch.stack([b["image_features"] for b in batch]),
        "attributes": torch.stack([b["attributes"] for b in batch]),
        "coords": torch.stack([b["coords"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
        "patient_id": [b["patient_id"] for b in batch],
        "nodule_id": [b["nodule_id"] for b in batch],
    }
    return out

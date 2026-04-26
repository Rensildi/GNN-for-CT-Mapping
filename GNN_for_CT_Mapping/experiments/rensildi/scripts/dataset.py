"""PyTorch Dataset for cached per-nodule feature rows."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..models.fusion import LIDC_ATTRIBUTE_NAMES


class NoduleDataset(Dataset):
    """Serves cached features for one subset of nodules."""

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
        joined = nodules.merge(features, on="nodule_id", how="inner").reset_index(drop=True)
        if joined.empty:
            raise ValueError(
                "No rows after joining nodules and features. Check nodule_id keys and feature extraction output."
            )

        self._df = joined
        self._attr_columns = [f"{name}_mean" for name in LIDC_ATTRIBUTE_NAMES]

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        row = self._df.iloc[idx]
        image = torch.as_tensor(np.asarray(row["features"], dtype=np.float32))

        attr_vals: list[int] = []
        for col in self._attr_columns:
            v = row[col]
            if pd.isna(v):
                attr_vals.append(0)
            else:
                # LIDC ratings are 1-based; embeddings are 0-based.
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
    return {
        "image_features": torch.stack([b["image_features"] for b in batch]),
        "attributes": torch.stack([b["attributes"] for b in batch]),
        "coords": torch.stack([b["coords"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
        "patient_id": [b["patient_id"] for b in batch],
        "nodule_id": [b["nodule_id"] for b in batch],
    }

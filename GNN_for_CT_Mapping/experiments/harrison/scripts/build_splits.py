"""Patient-level 5-fold cross-validation splits.

§0.3 of the execution plan: never split nodules from the same patient
across folds. We use `StratifiedGroupKFold` which is the sklearn tool for
exactly this — groups=patient_id, stratify on the patient-level majority
label so class balance stays roughly uniform across folds.

Writes `data/splits/fold_{0..4}.json` with shape:

    {
        "fold": 0,
        "train_patient_ids": [...],
        "val_patient_ids":   [...],
        "class_balance_train": {"0": int, "1": int},
        "class_balance_val":   {"0": int, "1": int}
    }
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def patient_majority_label(nodules: pd.DataFrame) -> pd.Series:
    """Assign each patient a single label for stratification purposes.

    Majority vote across the patient's nodules; ties break malignant so the
    rarer class isn't accidentally drained into the training set.
    """
    def _majority(labels: pd.Series) -> int:
        counts = labels.value_counts()
        if len(counts) == 1:
            return int(counts.index[0])
        # Tie on count -> prefer malignant (class 1) to keep it balanced
        # across folds.
        if counts.iloc[0] == counts.iloc[1]:
            return 1 if 1 in counts.index else int(counts.index[0])
        return int(counts.idxmax())

    return nodules.groupby("patient_id")["label"].apply(_majority)


def build_splits(nodules: pd.DataFrame, n_folds: int, seed: int) -> list[dict]:
    """Generate the per-fold assignment dicts."""
    patient_labels = patient_majority_label(nodules)
    patients = patient_labels.index.to_numpy()
    strata = patient_labels.to_numpy()
    # Groups argument is ignored by StratifiedGroupKFold beyond ensuring no
    # group crosses folds — since our "group" is already the patient ID and
    # we pass the per-patient label, we give groups=patients (one group per
    # entity).
    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    folds: list[dict] = []
    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(patients, strata, groups=patients)):
        train_patients = patients[train_idx].tolist()
        val_patients = patients[val_idx].tolist()
        train_nodules = nodules[nodules["patient_id"].isin(train_patients)]
        val_nodules = nodules[nodules["patient_id"].isin(val_patients)]

        folds.append({
            "fold": fold_idx,
            "train_patient_ids": train_patients,
            "val_patient_ids": val_patients,
            "class_balance_train": {
                "0": int((train_nodules["label"] == 0).sum()),
                "1": int((train_nodules["label"] == 1).sum()),
            },
            "class_balance_val": {
                "0": int((val_nodules["label"] == 0).sum()),
                "1": int((val_nodules["label"] == 1).sum()),
            },
        })
    return folds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodules", type=Path, required=True,
                        help="Path to nodules.parquet from preprocess.py")
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="Directory where fold_{i}.json files are written")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    nodules = pd.read_parquet(args.nodules)
    folds = build_splits(nodules, n_folds=args.n_folds, seed=args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for fold in folds:
        out_path = args.out_dir / f"fold_{fold['fold']}.json"
        with out_path.open("w") as f:
            json.dump(fold, f, indent=2)
        print(f"Wrote {out_path}  train={fold['class_balance_train']}  val={fold['class_balance_val']}")


if __name__ == "__main__":
    main()

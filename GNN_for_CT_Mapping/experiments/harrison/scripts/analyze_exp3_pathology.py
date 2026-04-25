"""Experiment 3.1 — pathology-subset analysis.

Cross-references the per-nodule predictions saved by train_exp3.py
against the pathology-confirmed ground truth in
`tcia-diagnosis-data-2012-04-20.xls` (157 LIDC-IDRI patients with
biopsy / surgery / follow-up diagnoses), then computes AUC / AUPRC /
Brier per cell on this cleaner label set and writes a summary parquet.

Why this matters:
    The headline Experiment 3 result used radiologist-consensus labels
    (the same people whose attribute ratings are fed to the model), so
    the attribute branch's contribution is partially circular. The
    pathology subset comes from biopsies and surgical resections, so it
    gives a ground-truth-against-attributes sanity check.

Matching strategy:
    The diagnosis sheet lists up to 5 per-nodule pathology labels per
    patient, but those "Nodule N" numbers are LIDC's internal IDs and
    don't obviously map to our clustered nodule_ids. We therefore use a
    conservative match:

        - A patient with exactly 1 pathology-confirmed nodule that has
          a resolved label (1=benign, 2/3=malignant) contributes that
          label to all of their nodules in nodules.parquet ONLY IF the
          patient also has exactly 1 clustered nodule in our data.
        - A patient with multiple pathology nodules whose labels all
          agree (all benign OR all malignant) contributes that shared
          label to all of their nodules in our data.
        - All other patients are excluded from the pathology subset.

    This is lossy but unambiguous — we never guess at a nodule-level
    mapping where the diagnosis sheet doesn't make the correspondence
    explicit.

Run:
    python -m GNN_for_CT_Mapping.experiments.harrison.scripts.analyze_exp3_pathology
"""
from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


# --- Paths ----------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[4]
XLS_PATH = REPO_ROOT / "GNN_for_CT_Mapping/data/annotations/tcia-diagnosis-data-2012-04-20.xls"
NODULES_PATH = REPO_ROOT / "GNN_for_CT_Mapping/data/nodules.parquet"
PREDICTIONS_DIR = REPO_ROOT / "GNN_for_CT_Mapping/outputs/predictions"


ENCODERS = ("med3d", "fmcib")
FEATURE_CONFIGS = ("image", "image_attrs", "full")


# Column-name shorthands because the xls headers are multi-line prose.
PATIENT_ID_COL = "TCIA Patient ID"
PATIENT_DX_COL_PREFIX = "Diagnosis at the Patient Level"
NODULE_DX_COL_PREFIX = "Diagnosis at the Nodule Level"


def _dx_to_binary(dx_value) -> int | None:
    """Map TCIA diagnosis code → binary label or None if unknown / missing.

    Codes: 0 = unknown, 1 = benign, 2 = malignant primary, 3 = malignant metastatic.
    """
    if pd.isna(dx_value):
        return None
    try:
        code = int(round(float(dx_value)))
    except (TypeError, ValueError):
        return None
    if code == 1:
        return 0
    if code in (2, 3):
        return 1
    return None  # code 0 (unknown) or out-of-range


def load_pathology_labels() -> pd.DataFrame:
    """Return DataFrame of (patient_id, pathology_label) for patients we can
    confidently label using the matching strategy documented above."""
    raw = pd.read_excel(XLS_PATH, sheet_name="Diagnosis Truth")

    # Identify the 5 per-nodule diagnosis columns. The xls header actually
    # starts with "Nodule N\nDiagnosis at the Nodule Level ..."; the
    # per-nodule "... Diagnosis Method ..." columns are separate and we
    # don't want those. Match "Nodule N" prefix, reject anything
    # containing "Method".
    nodule_dx_cols = [
        c for c in raw.columns
        if c.lstrip().startswith("Nodule ")
        and "Diagnosis at the Nodule Level" in c
        and "Method" not in c
    ]
    nodule_dx_cols = sorted(nodule_dx_cols, key=lambda c: int(c.split()[1]))
    if len(nodule_dx_cols) != 5:
        raise RuntimeError(
            f"Expected 5 per-nodule diagnosis columns, found {len(nodule_dx_cols)}:\n{nodule_dx_cols}"
        )

    rows = []
    for _, r in raw.iterrows():
        patient_id = str(r[PATIENT_ID_COL]).strip()
        # Collect whichever per-nodule labels the patient has.
        per_nodule_labels = []
        for col in nodule_dx_cols:
            lbl = _dx_to_binary(r[col])
            if lbl is not None:
                per_nodule_labels.append(lbl)
        if not per_nodule_labels:
            continue
        # Patients where all per-nodule labels agree get an unambiguous
        # patient-level pathology label. Patients with mixed labels are
        # skipped — we can't tell which nodule in our parquet got which
        # pathology label without an explicit mapping.
        if len(set(per_nodule_labels)) == 1:
            rows.append({
                "patient_id": patient_id,
                "pathology_label": per_nodule_labels[0],
                "n_pathology_nodules": len(per_nodule_labels),
            })
    return pd.DataFrame(rows)


def build_matched_labels(pathology_df: pd.DataFrame, nodules_df: pd.DataFrame) -> pd.DataFrame:
    """Attach pathology labels to our nodules.parquet rows.

    Uses the conservative matching rule: apply the patient-level pathology
    label only to patients where our clustering produced a compatible
    number of nodules.

    preprocess.py has a rare hash-collision bug that causes 7 nodule_ids
    in nodules.parquet to appear more than once (same patient, same
    centroid). We dedupe on nodule_id first so each physical nodule is
    counted once in the match.
    """
    nodules_df = nodules_df.drop_duplicates(subset="nodule_id", keep="first")
    pathology_by_patient = pathology_df.set_index("patient_id")
    nodule_counts = nodules_df.groupby("patient_id").size()

    matched = []
    for _, nod in nodules_df.iterrows():
        pid = nod["patient_id"]
        if pid not in pathology_by_patient.index:
            continue
        # Either patient had only 1 pathology-labeled nodule AND 1 clustered
        # nodule, OR patient had multiple pathology nodules that agreed.
        n_path = pathology_by_patient.loc[pid, "n_pathology_nodules"]
        n_ours = nodule_counts[pid]
        if n_path == 1 and n_ours == 1:
            acceptable = True
        elif n_path > 1:
            acceptable = True  # agreement was already verified in load_pathology_labels
        else:
            acceptable = False
        if not acceptable:
            continue
        matched.append({
            "nodule_id": nod["nodule_id"],
            "patient_id": pid,
            "radiologist_label": int(nod["label"]),
            "pathology_label": int(pathology_by_patient.loc[pid, "pathology_label"]),
        })
    return pd.DataFrame(matched)


def _metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """AUC / AUPRC / Brier with graceful handling of degenerate subsets."""
    if y_true.min() == y_true.max() or len(y_true) < 2:
        return {"auc": float("nan"), "auprc": float("nan"),
                "brier": float(brier_score_loss(y_true, y_prob))
                         if len(y_true) else float("nan"),
                "n": int(len(y_true))}
    return {
        "auc": float(roc_auc_score(y_true, y_prob)),
        "auprc": float(average_precision_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "n": int(len(y_true)),
    }


def pool_cell_predictions(encoder: str, feature_config: str) -> pd.DataFrame:
    """Pool per-nodule predictions across all 5 folds for one cell.

    Each fold's val set is disjoint by construction (patient-level splits
    partition the patient universe), so pooling is simply concat. We
    dedupe on (fold, nodule_id) because nodules.parquet has a handful of
    duplicate rows that propagate through the pipeline — each duplicate
    carries the same probability, so keeping the first is lossless.
    """
    pieces = []
    for fold in range(5):
        path = PREDICTIONS_DIR / f"exp3_{encoder}_{feature_config}_fold{fold}.parquet"
        df = pd.read_parquet(path)
        df["fold"] = fold
        pieces.append(df)
    pooled = pd.concat(pieces, ignore_index=True)
    return pooled.drop_duplicates(subset=["fold", "nodule_id"], keep="first")


def main() -> None:
    # 1. Load pathology labels + match to our nodules.
    pathology_df = load_pathology_labels()
    nodules_df = pd.read_parquet(NODULES_PATH)
    matched = build_matched_labels(pathology_df, nodules_df)

    print(f"Pathology-confirmed patients (after agreement filter): {len(pathology_df)}")
    print(f"Matched nodules (appear in both nodules.parquet and pathology sheet): {len(matched)}")
    print(f"  matched benign:    {int((matched['pathology_label'] == 0).sum())}")
    print(f"  matched malignant: {int((matched['pathology_label'] == 1).sum())}")
    print()

    # 2. Agreement between radiologist-consensus and pathology labels on
    # the subset. Tells us how often the two label sources disagree.
    agree = (matched["radiologist_label"] == matched["pathology_label"]).sum()
    print(f"Radiologist label vs pathology label agreement on matched subset: "
          f"{agree}/{len(matched)} = {100*agree/len(matched):.1f}%")
    print()

    # 3. Per-cell pathology-subset metrics. For every (encoder, feature_config),
    # pool the 5 fold val-prediction parquets and restrict to matched nodules.
    results = []
    for encoder, feature_config in itertools.product(ENCODERS, FEATURE_CONFIGS):
        pooled = pool_cell_predictions(encoder, feature_config)
        joined = pooled.merge(
            matched[["nodule_id", "pathology_label"]],
            on="nodule_id",
            how="inner",
        )
        y_true = joined["pathology_label"].to_numpy()
        y_prob = joined["prob_malignant"].to_numpy()
        metrics = _metrics(y_true, y_prob)
        metrics.update({"encoder": encoder, "feature_config": feature_config})
        results.append(metrics)

    summary = pd.DataFrame(results)
    # Also compute per-fold pathology metrics so we can run paired Wilcoxons.
    per_fold_rows = []
    for encoder, feature_config in itertools.product(ENCODERS, FEATURE_CONFIGS):
        pooled = pool_cell_predictions(encoder, feature_config)
        joined = pooled.merge(
            matched[["nodule_id", "pathology_label"]],
            on="nodule_id",
            how="inner",
        )
        for fold_num, subset in joined.groupby("fold"):
            y_true = subset["pathology_label"].to_numpy()
            y_prob = subset["prob_malignant"].to_numpy()
            m = _metrics(y_true, y_prob)
            m.update({"encoder": encoder, "feature_config": feature_config,
                      "fold": int(fold_num)})
            per_fold_rows.append(m)
    per_fold_df = pd.DataFrame(per_fold_rows)

    # Persist both pooled and per-fold results for downstream analyses.
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_parquet(PREDICTIONS_DIR / "exp3_1_pathology_pooled.parquet", index=False)
    per_fold_df.to_parquet(PREDICTIONS_DIR / "exp3_1_pathology_per_fold.parquet", index=False)

    # 4. Print-friendly heatmap (pooled across folds).
    pivot_auc = summary.pivot(index="feature_config", columns="encoder", values="auc").reindex(
        index=list(FEATURE_CONFIGS), columns=list(ENCODERS))
    pivot_auprc = summary.pivot(index="feature_config", columns="encoder", values="auprc").reindex(
        index=list(FEATURE_CONFIGS), columns=list(ENCODERS))
    pivot_n = summary.pivot(index="feature_config", columns="encoder", values="n").reindex(
        index=list(FEATURE_CONFIGS), columns=list(ENCODERS))

    print("=== Pooled pathology-subset AUC ===")
    print(pivot_auc.round(4).to_string())
    print()
    print("=== Pooled pathology-subset AUPRC ===")
    print(pivot_auprc.round(4).to_string())
    print()
    print("=== Pooled sample count per cell (should be identical across cells) ===")
    print(pivot_n.astype(int).to_string())
    print()

    # 5. Paired Wilcoxon on pathology-subset per-fold AUC where possible.
    # Subsets per fold are tiny (a few nodules), so AUC can be NaN if
    # the fold-subset is single-class. We only report comparisons where
    # all 5 folds have a valid AUC for both arms.
    def _paired_auc(enc_a, cfg_a, enc_b, cfg_b):
        a = per_fold_df[(per_fold_df.encoder == enc_a) &
                        (per_fold_df.feature_config == cfg_a)].sort_values("fold")["auc"].to_numpy()
        b = per_fold_df[(per_fold_df.encoder == enc_b) &
                        (per_fold_df.feature_config == cfg_b)].sort_values("fold")["auc"].to_numpy()
        if np.isnan(a).any() or np.isnan(b).any():
            return None, a, b
        try:
            _, p = wilcoxon(a, b, alternative="greater")
        except ValueError:
            return float("nan"), a, b
        return float(p), a, b

    print("=== Paired Wilcoxon (one-sided 'A > B') — per-fold pathology AUC ===")
    print("   (n/a means one or more folds had single-class pathology subsets, "
          "so AUC was undefined for that fold)")
    for cfg in FEATURE_CONFIGS:
        p, a, b = _paired_auc("fmcib", cfg, "med3d", cfg)
        if p is None:
            print(f"  encoder axis within {cfg:12s}:  n/a")
        else:
            delta = float((a - b).mean())
            print(f"  encoder axis within {cfg:12s}:  FMCIB mean AUC - Med3D = {delta:+.4f}   p = {p:.4f}")
    for enc in ENCODERS:
        p, a, b = _paired_auc(enc, "image_attrs", enc, "image")
        if p is None:
            print(f"  modality axis within {enc:5s}:  image_attrs > image  n/a")
        else:
            delta = float((a - b).mean())
            print(f"  modality axis within {enc:5s}:  image_attrs - image = {delta:+.4f}   p = {p:.4f}")

    # 6. Bootstrap 95% CIs per cell — the N=40 matched subset makes per-fold
    # metrics unreliable (some folds are single-class); the pooled AUC is
    # the canonical headline. Bootstrap over the 40 nodules to get its
    # sampling distribution.
    print()
    print("=== Bootstrap 95% CI for pooled pathology AUC (1000 resamples) ===")
    rng = np.random.default_rng(42)
    n_boot = 1000
    all_pooled = {}
    for encoder, feature_config in itertools.product(ENCODERS, FEATURE_CONFIGS):
        pooled = pool_cell_predictions(encoder, feature_config)
        joined = pooled.merge(
            matched[["nodule_id", "pathology_label"]],
            on="nodule_id",
            how="inner",
        )
        all_pooled[(encoder, feature_config)] = joined
    for (encoder, feature_config), joined in all_pooled.items():
        y = joined["pathology_label"].to_numpy()
        p = joined["prob_malignant"].to_numpy()
        aucs = []
        n = len(y)
        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            yb, pb = y[idx], p[idx]
            if yb.min() == yb.max():
                continue
            aucs.append(roc_auc_score(yb, pb))
        lo, hi = float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))
        print(f"  {encoder:5s} × {feature_config:12s}   pooled AUC {roc_auc_score(y, p):.4f}   "
              f"95% CI [{lo:.4f}, {hi:.4f}]")

    # 7. Paired bootstrap for the most interesting cross-cell comparisons:
    # pathology AUC of (fmcib, image) vs (med3d, full). Paired resampling
    # preserves the per-nodule alignment of the two cells' predictions.
    print()
    print("=== Paired bootstrap — Δ pooled pathology AUC, 95% CI (1000 resamples) ===")
    def _paired_boot(a_cell, b_cell, label):
        a = all_pooled[a_cell][["nodule_id", "pathology_label", "prob_malignant"]].rename(
            columns={"prob_malignant": "p_a"})
        b = all_pooled[b_cell][["nodule_id", "prob_malignant"]].rename(
            columns={"prob_malignant": "p_b"})
        m = a.merge(b, on="nodule_id", how="inner")
        y = m["pathology_label"].to_numpy()
        pa = m["p_a"].to_numpy()
        pb = m["p_b"].to_numpy()
        point = roc_auc_score(y, pa) - roc_auc_score(y, pb)
        deltas = []
        n = len(y)
        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            yb = y[idx]
            if yb.min() == yb.max():
                continue
            deltas.append(roc_auc_score(yb, pa[idx]) - roc_auc_score(yb, pb[idx]))
        lo, hi = float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))
        print(f"  {label}:  Δ AUC = {point:+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]")

    _paired_boot(("fmcib", "image"), ("med3d", "image"),
                 "FMCIB image  vs  Med3D image")
    _paired_boot(("fmcib", "image"), ("fmcib", "image_attrs"),
                 "FMCIB image  vs  FMCIB image+attrs")
    _paired_boot(("fmcib", "image"), ("med3d", "full"),
                 "FMCIB image  vs  Med3D full")
    _paired_boot(("fmcib", "image_attrs"), ("med3d", "image"),
                 "FMCIB image+attrs  vs  Med3D image")


if __name__ == "__main__":
    main()

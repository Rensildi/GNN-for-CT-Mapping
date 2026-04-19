"""Build nodules.parquet from LIDC XML annotations + metadata.csv.

This is the end-to-end entry point for Section 0.1 of the execution plan.
Run once after downloading LIDC-IDRI; the resulting parquet is the canonical
nodule table for every experiment.

Pipeline:

    1. Read metadata.csv to map SeriesInstanceUID -> patient_id and to
       locate each series' DICOM directory on disk.
    2. For each annotation XML under tcia-lidc-xml/**/, parse it and match
       to the right series via SeriesInstanceUID.
    3. For each reader's nodule annotation, compute the centroid in
       physical (x, y, z) mm using the series' DICOM spacing.
    4. Cluster per-reader nodules across readers by centroid distance
       (<= CLUSTER_DIST_MM) into unified "nodule" records with up to 4
       annotations.
    5. Per unified nodule, compute mean malignancy score, binarize via
       <= 2 benign / >= 4 malignant, drop ambiguous (~3), and persist.

Output parquet columns:

    patient_id            str
    nodule_id             str   (deterministic SHA hash of series+centroid)
    series_uid            str
    num_annotations       int   (1..4)
    centroid_x_mm         float
    centroid_y_mm         float
    centroid_z_mm         float
    mean_malignancy       float (pre-binarization, in [1.0, 5.0])
    label                 int   (0 benign, 1 malignant)
    <attr>_mean           float (one column per ATTRIBUTE_TAGS entry)
    annotation_ids        list[str] (reader-local nodule IDs)

Run:
    python -m experiments.harrison.scripts.preprocess \\
        --lidc-root /media/talos/SchoolHD/deep_learning/LIDC-IDRI \\
        --annotations-root GNN_for_CT_Mapping/data/annotations \\
        --out GNN_for_CT_Mapping/data/nodules.parquet

Note: the centroid-distance clustering is a simpler heuristic than pylidc's
IoU-based clustering but is sufficient for LIDC's ~7k nodules where readers
generally mark the same anatomical location within a few mm. If clustering
quality becomes a concern, switch to an IoU-based approach over the 3D
masks reconstructed from RoiSlice.edge_map contours.
"""
from __future__ import annotations

import argparse
import hashlib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .dicom_loader import load_series, CTVolume
from .parse_lidc_xml import (
    ATTRIBUTE_TAGS,
    ReaderNodule,
    parse_xml,
)


# Radii and thresholds straight from configs/default.yaml and the proposal.
# Centralized here so one change propagates to the whole preprocessing run.
CLUSTER_DIST_MM = 8.0  # two per-reader annotations are "the same nodule" if
                       # their centroids are within this distance.
MALIGNANCY_BENIGN_MAX = 2.0    # mean <= this -> benign (label 0)
MALIGNANCY_MALIGNANT_MIN = 4.0 # mean >= this -> malignant (label 1)


@dataclass
class NoduleCluster:
    """An aggregated nodule across up to 4 readers, in physical (mm) space."""

    centroid_mm: np.ndarray  # (3,) averaged (x, y, z) centroid
    annotations: list[ReaderNodule]


def compute_reader_centroid(
    nodule: ReaderNodule,
    volume: CTVolume,
) -> np.ndarray:
    """Compute a per-reader nodule's centroid in physical (x, y, z) mm.

    Averages the mean of each slice's edge-map pixel coordinates weighted
    equally across slices — a crude but stable centroid estimator that
    doesn't require rasterizing the contour into a mask.
    """
    xs_mm: list[float] = []
    ys_mm: list[float] = []
    zs_mm: list[float] = []
    ox, oy, _oz = volume.origin_mm
    _z_spacing, y_spacing, x_spacing = volume.voxel_spacing_mm

    for roi in nodule.rois:
        if not roi.edge_map:
            continue
        pts = np.asarray(roi.edge_map, dtype=np.float32)  # (P, 2) (xi, yi)
        x_pix_mean = float(pts[:, 0].mean())
        y_pix_mean = float(pts[:, 1].mean())
        # Pixel -> physical: world = origin + index * spacing.
        xs_mm.append(ox + x_pix_mean * x_spacing)
        ys_mm.append(oy + y_pix_mean * y_spacing)
        zs_mm.append(roi.image_z_mm)

    # At least one ROI is guaranteed by parse_lidc_xml (it filters empties).
    return np.array([np.mean(xs_mm), np.mean(ys_mm), np.mean(zs_mm)], dtype=np.float32)


def cluster_readers(
    reader_nodules_with_centroids: list[tuple[ReaderNodule, np.ndarray]],
    max_dist_mm: float = CLUSTER_DIST_MM,
) -> list[NoduleCluster]:
    """Greedy cluster per-reader nodules into unified nodules by centroid.

    Algorithm:
        - Iterate through (reader_nodule, centroid) tuples.
        - For each, check existing clusters; if any cluster's centroid is
          within `max_dist_mm`, join it (and update the cluster centroid to
          the mean of its members).
        - Otherwise start a new cluster.

    This is O(N^2) but N (per-scan nodule count * readers) is small — a few
    dozen at most. A KD-tree would be overkill.
    """
    clusters: list[NoduleCluster] = []
    for reader_nod, centroid in reader_nodules_with_centroids:
        best_idx = -1
        best_dist = float("inf")
        for idx, cluster in enumerate(clusters):
            d = float(np.linalg.norm(cluster.centroid_mm - centroid))
            # Prefer the closest matching cluster when multiple are in range,
            # to avoid arbitrary assignment near overlap boundaries.
            if d < max_dist_mm and d < best_dist:
                best_idx = idx
                best_dist = d

        if best_idx == -1:
            clusters.append(NoduleCluster(centroid_mm=centroid.copy(), annotations=[reader_nod]))
        else:
            cl = clusters[best_idx]
            cl.annotations.append(reader_nod)
            # Running mean of the centroid so successive adds don't drift
            # toward the first-seen annotation.
            n = len(cl.annotations)
            cl.centroid_mm = cl.centroid_mm + (centroid - cl.centroid_mm) / n

    return clusters


def label_and_aggregate(cluster: NoduleCluster) -> dict | None:
    """Derive a row for nodules.parquet from a clustered nodule.

    Returns None when the cluster's mean malignancy falls in the ambiguous
    (~3) middle — those nodules are intentionally excluded per the plan.
    """
    # Collect the 8 attribute ratings (for per-attribute means) and the
    # malignancy ratings (for labeling). Malignancy is read from the
    # dedicated field on ReaderNodule, not from `characteristics`, to keep
    # the label strictly separate from the feature branch.
    malignancies: list[float] = []
    attr_values: dict[str, list[float]] = {tag: [] for tag in ATTRIBUTE_TAGS}
    for ann in cluster.annotations:
        # The characteristics dict uses 0-indexed values; add 1 to recover
        # the original 1–5 / 1–6 scale for readable per-attribute means.
        for tag in ATTRIBUTE_TAGS:
            if tag in ann.characteristics:
                attr_values[tag].append(ann.characteristics[tag] + 1)
        if ann.malignancy is not None:
            malignancies.append(float(ann.malignancy))

    if not malignancies:
        # No reader scored malignancy — can't label, skip.
        return None

    mean_mal = float(np.mean(malignancies))
    if mean_mal <= MALIGNANCY_BENIGN_MAX:
        label = 0
    elif mean_mal >= MALIGNANCY_MALIGNANT_MIN:
        label = 1
    else:
        # Ambiguous zone (~3) — excluded from training per the proposal.
        return None

    # Deterministic, short nodule ID: hash of series UID + centroid to 2 dp.
    # Lets us re-derive the same ID across preprocessing runs.
    first_ann = cluster.annotations[0]
    series_part = first_ann.rois[0].image_sop_uid or ""
    key = f"{series_part}|{cluster.centroid_mm[0]:.2f}|{cluster.centroid_mm[1]:.2f}|{cluster.centroid_mm[2]:.2f}"
    nod_id = hashlib.sha1(key.encode()).hexdigest()[:12]

    row = {
        "nodule_id": nod_id,
        "num_annotations": len(cluster.annotations),
        "centroid_x_mm": float(cluster.centroid_mm[0]),
        "centroid_y_mm": float(cluster.centroid_mm[1]),
        "centroid_z_mm": float(cluster.centroid_mm[2]),
        "mean_malignancy": mean_mal,
        "label": label,
        "annotation_ids": [ann.nodule_id for ann in cluster.annotations],
    }
    # Per-attribute mean (as float in original 1-based scale). Using NaN for
    # empty lists keeps downstream groupby/mean operations safe.
    for tag, vals in attr_values.items():
        row[f"{tag}_mean"] = float(np.mean(vals)) if vals else float("nan")
    return row


def build_series_to_dicom_map(metadata_csv: Path) -> dict[str, dict]:
    """Map SeriesInstanceUID -> {'dicom_dir': Path, 'patient_id': str}.

    LIDC's TCIA `metadata.csv` has one row per CT series. The on-disk
    layout produced by the NBIA Data Retriever puts each series at

        <lidc_root>/lidc_idri/<PatientID>/<StudyInstanceUID>/<SeriesInstanceUID>/

    so we construct the DICOM directory directly from three columns rather
    than trusting the CSV's file-path column (which is sometimes a stale
    mirror, e.g. referring to a different machine's username).

    Column-name handling is robust to whitespace and case because TCIA has
    shipped at least three minor variants over the years.
    """
    df = pd.read_csv(metadata_csv)
    # metadata.csv lives at <lidc_root>/metadata/metadata.csv — so two
    # parents up is the dataset root.
    lidc_root = metadata_csv.parent.parent

    # Normalize column names to no-space lowercase so lookup is robust.
    cols = {c.replace(" ", "").lower(): c for c in df.columns}
    series_col = cols.get("seriesinstanceuid") or cols.get("seriesuid")
    study_col = cols.get("studyinstanceuid") or cols.get("studyuid")
    patient_col = cols.get("patientid")
    if not (series_col and study_col and patient_col):
        raise KeyError(
            f"metadata.csv is missing required columns. Found: {list(df.columns)}"
        )

    mapping: dict[str, dict] = {}
    for _, row in df.iterrows():
        series_uid = str(row[series_col]).strip()
        study_uid = str(row[study_col]).strip()
        patient_id = str(row[patient_col]).strip()
        if not series_uid or series_uid == "nan":
            continue
        dicom_dir = lidc_root / "lidc_idri" / patient_id / study_uid / series_uid
        mapping[series_uid] = {"dicom_dir": dicom_dir, "patient_id": patient_id}
    return mapping


def iter_xml_files(annotations_root: Path):
    """Yield every .xml file under tcia-lidc-xml/ (recursive)."""
    xml_root = annotations_root / "tcia-lidc-xml"
    if not xml_root.exists():
        raise FileNotFoundError(f"Expected XML annotations under {xml_root}")
    yield from sorted(xml_root.rglob("*.xml"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lidc-root", type=Path, required=True,
                        help="Dataset root containing lidc_idri/ and metadata/ (e.g. /media/talos/SchoolHD/deep_learning/LIDC-IDRI)")
    parser.add_argument("--annotations-root", type=Path, required=True,
                        help="Path to GNN_for_CT_Mapping/data/annotations")
    parser.add_argument("--out", type=Path, required=True,
                        help="Output parquet path")
    parser.add_argument("--limit", type=int, default=None,
                        help="Dev only: stop after this many XML files. Useful for smoke-testing before a full run.")
    args = parser.parse_args()

    metadata_csv = args.lidc_root / "metadata" / "metadata.csv"
    series_to_dir = build_series_to_dicom_map(metadata_csv)

    rows: list[dict] = []

    xml_files = list(iter_xml_files(args.annotations_root))
    if args.limit is not None:
        xml_files = xml_files[:args.limit]

    for xml_path in tqdm(xml_files, desc="XML"):
        scan = parse_xml(xml_path)
        meta = series_to_dir.get(scan.series_instance_uid)
        if meta is None or not meta["dicom_dir"].exists():
            # Missing DICOM for this XML — skip without failing the whole run.
            continue
        dicom_dir = meta["dicom_dir"]
        patient_id = meta["patient_id"]

        # Only read DICOM headers here if we actually have nodules to place.
        if not scan.nodules:
            continue

        try:
            volume = load_series(dicom_dir)
        except (FileNotFoundError, ValueError):
            continue

        # Compute per-reader centroids in physical space, then cluster.
        pairs = [(n, compute_reader_centroid(n, volume)) for n in scan.nodules]
        clusters = cluster_readers(pairs)

        for cluster in clusters:
            row = label_and_aggregate(cluster)
            if row is None:
                continue
            row["patient_id"] = patient_id
            row["series_uid"] = scan.series_instance_uid
            rows.append(row)

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    # Log the class balance so the user spots data issues before training.
    print(f"Wrote {len(df)} nodules to {args.out}")
    if len(df):
        benign = int((df["label"] == 0).sum())
        malignant = int((df["label"] == 1).sum())
        print(f"  benign={benign}  malignant={malignant}")


if __name__ == "__main__":
    main()

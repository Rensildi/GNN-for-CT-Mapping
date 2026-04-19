"""LIDC-IDRI annotation XML parser (pure stdlib, no pylidc).

Each LIDC-IDRI scan has a single `.xml` file containing readings from up to
four radiologists. The root element is `LidcReadMessage` under the
`http://www.nih.gov` namespace. Inside, one `readingSession` per radiologist
contains:

    - `unblindedReadNodule` elements for nodules >= 3 mm (with an 8-attribute
      characteristics block and one `roi` per slice containing the 2D
      contour).
    - `nonNodule` elements for lesions < 3 mm (we ignore these here — we
      only train on nodules >= 3 mm per the proposal).

Coordinate system:
    - `imageZposition` (mm) names a specific DICOM slice along the Z axis.
    - `edgeMap` entries are the contour's (xCoord, yCoord) pixel indices
      within that slice.

This parser returns raw per-reader nodule dicts; clustering across readers
(matching nodule N from reader 1 to nodule M from reader 2) happens later in
preprocess.py, since it needs centroids in mm which in turn need DICOM
spacing.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path


# Shared NIH namespace — every LIDC XML tag lives under this prefix.
NS = {"lidc": "http://www.nih.gov"}

# Fixed 8-attribute order used throughout the codebase. Kept in sync with
# models/fusion.py::LIDC_ATTRIBUTE_NAMES so preprocess and fusion agree on
# the column meaning.
ATTRIBUTE_TAGS: tuple[str, ...] = (
    "subtlety",
    "internalStructure",
    "calcification",
    "sphericity",
    "margin",
    "lobulation",
    "spiculation",
    "texture",
)


@dataclass
class RoiSlice:
    """One 2D contour on a specific Z slice.

    The contour is a list of pixel indices (xi, yi) along the nodule's
    outline in that slice. Together, the full set of RoiSlices in a nodule
    defines the 3D region — we use them to compute the centroid and bounding
    box but don't reconstruct the mask voxel-by-voxel for classification.
    """

    image_z_mm: float  # slice position along Z axis, in mm (DICOM coord)
    edge_map: list[tuple[int, int]]  # (xi, yi) pixel coords on this slice
    inclusion: bool = True  # True = inside nodule, False = hole
    image_sop_uid: str | None = None  # DICOM SOP Instance UID of this slice


@dataclass
class ReaderNodule:
    """One reader's annotation of a single nodule >= 3 mm.

    Malignancy is tracked separately from `characteristics` because it's the
    training label, not a feature — keeping it in its own field prevents
    accidental leakage into the Stage 1 fusion (which indexes the 8
    descriptive attributes only).
    """

    reader_index: int  # 0..3 — position of the reader in the XML file
    nodule_id: str     # string ID assigned by the reader (reader-local)
    characteristics: dict[str, int] = field(default_factory=dict)
    malignancy: int | None = None  # 1-based LIDC rating 1..5; None if absent
    rois: list[RoiSlice] = field(default_factory=list)


@dataclass
class ScanAnnotations:
    """All annotations for one CT scan (one XML file)."""

    series_instance_uid: str
    study_instance_uid: str
    patient_id: str | None  # not always present in LIDC XML
    nodules: list[ReaderNodule]  # flat list across all readers


def _text(el: ET.Element | None, default: str = "") -> str:
    """Safe text accessor — returns default when the element or text is None."""
    if el is None or el.text is None:
        return default
    return el.text.strip()


def _parse_roi(roi_el: ET.Element) -> RoiSlice:
    """Parse one `<roi>` element into an RoiSlice.

    The XML schema puts each contour point in an `<edgeMap>` sibling with
    `<xCoord>` and `<yCoord>` children.
    """
    z_mm = float(_text(roi_el.find("lidc:imageZposition", NS), "0.0"))

    # inclusion is "TRUE"/"FALSE" (uppercase string) in LIDC XML; anything
    # other than explicit "FALSE" is treated as inclusive.
    inclusion_txt = _text(roi_el.find("lidc:inclusion", NS), "TRUE").upper()
    inclusion = inclusion_txt != "FALSE"

    sop_uid = _text(roi_el.find("lidc:imageSOP_UID", NS)) or None

    # Each <edgeMap> is one point along the contour. There are typically
    # 10–100 points per slice.
    edges: list[tuple[int, int]] = []
    for em in roi_el.findall("lidc:edgeMap", NS):
        x = int(_text(em.find("lidc:xCoord", NS), "0"))
        y = int(_text(em.find("lidc:yCoord", NS), "0"))
        edges.append((x, y))

    return RoiSlice(image_z_mm=z_mm, edge_map=edges, inclusion=inclusion, image_sop_uid=sop_uid)


def _parse_nodule(nodule_el: ET.Element, reader_index: int) -> ReaderNodule | None:
    """Parse one `<unblindedReadNodule>` element.

    Returns None if the nodule has no ROIs (shouldn't happen in a valid XML
    but we defend against it). The 8-attribute characteristics block is
    sometimes absent for reader-ambiguous cases; when present we parse all
    8 integers and shift from LIDC's 1-based rating to 0-indexed so the
    embeddings in models/fusion.py can index directly.
    """
    nodule_id = _text(nodule_el.find("lidc:noduleID", NS))

    characteristics: dict[str, int] = {}
    malignancy: int | None = None
    char_el = nodule_el.find("lidc:characteristics", NS)
    if char_el is not None:
        for tag in ATTRIBUTE_TAGS:
            raw = _text(char_el.find(f"lidc:{tag}", NS))
            if raw:
                # Shift 1-based rating to 0-indexed so downstream embeddings
                # can use it directly without re-shifting per call site.
                characteristics[tag] = int(raw) - 1
        # Malignancy is kept in its original 1-based form because labeling
        # code needs the raw scale to binarize via <=2/>=4 thresholds.
        mal_raw = _text(char_el.find("lidc:malignancy", NS))
        if mal_raw:
            malignancy = int(mal_raw)

    rois = [_parse_roi(r) for r in nodule_el.findall("lidc:roi", NS)]
    if not rois:
        return None

    return ReaderNodule(
        reader_index=reader_index,
        nodule_id=nodule_id,
        characteristics=characteristics,
        malignancy=malignancy,
        rois=rois,
    )


def parse_xml(xml_path: Path) -> ScanAnnotations:
    """Parse one LIDC annotation XML file.

    Args:
        xml_path: path to a single `*.xml` annotation file, e.g.
            `GNN_for_CT_Mapping/data/annotations/tcia-lidc-xml/185/158.xml`.

    Returns:
        ScanAnnotations with all per-reader nodules >= 3 mm. Non-nodules and
        tiny (<3 mm) lesions are dropped. Use preprocess.py to cluster
        across readers into unified nodule records.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # ResponseHeader has the scan-level identifiers. Study/Series UIDs are
    # what we use to match the XML to a DICOM series directory on disk.
    header = root.find("lidc:ResponseHeader", NS)
    series_uid = _text(header.find("lidc:SeriesInstanceUid", NS)) if header is not None else ""
    study_uid = _text(header.find("lidc:StudyInstanceUID", NS)) if header is not None else ""
    # LIDC XMLs rarely include patient ID directly; preprocess.py uses
    # metadata.csv to map series_uid -> patient_id instead.
    patient_id = _text(header.find("lidc:PatientID", NS)) if header is not None else None
    if not patient_id:
        patient_id = None

    nodules: list[ReaderNodule] = []
    for reader_idx, session_el in enumerate(root.findall("lidc:readingSession", NS)):
        for nodule_el in session_el.findall("lidc:unblindedReadNodule", NS):
            nod = _parse_nodule(nodule_el, reader_idx)
            if nod is not None:
                nodules.append(nod)

    return ScanAnnotations(
        series_instance_uid=series_uid,
        study_instance_uid=study_uid,
        patient_id=patient_id,
        nodules=nodules,
    )

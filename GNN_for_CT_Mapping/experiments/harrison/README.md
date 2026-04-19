# Harrison — Experiment 1 implementation

Scaffolding for Experiment 1 (GCN vs. MLP baseline) from
[`documentation/execution_plan_experiments.md`](../../documentation/execution_plan_experiments.md).

## Layout

```
experiments/harrison/
├── configs/
│   └── experiment.yaml        # Overrides on top of configs/default.yaml
├── notebooks/
│   └── gpu_test.ipynb         # torch/CUDA smoke test
├── models/
│   ├── fusion.py              # Stage 1: multi-modal fusion (image + 8 attrs + spatial)
│   ├── med3d.py               # Frozen Med3D ResNet-50 wrapper
│   ├── mlp.py                 # Stage 2: MLP baseline
│   └── gcn.py                 # Stage 2: 2-layer GCN
└── scripts/
    ├── parse_lidc_xml.py      # Parse LidcReadMessage XML (no pylidc)
    ├── dicom_loader.py        # pydicom-based DICOM series loader + patch extractor
    ├── preprocess.py          # Build nodules.parquet
    ├── build_splits.py        # Patient-level 5-fold CV splits
    ├── extract_features.py    # Cache frozen Med3D features
    ├── dataset.py             # PyTorch Dataset over cached features
    ├── graph.py               # Inductive KNN graph construction
    └── train_exp1.py          # End-to-end training loop
```

## Status

**Ready:**
- Stage 1 / Stage 2 models (`fusion.py`, `mlp.py`, `gcn.py`)
- XML parser, DICOM loader, patch extractor
- `preprocess.py` pipeline (XML → nodules.parquet) and `build_splits.py`
- Inductive KNN construction and the training loop
- GPU test notebook

**Needs finishing before end-to-end run:**
- `models/med3d.py::from_checkpoint` — vendoring MedicalNet's `resnet.py`
  from `https://github.com/Tencent/MedicalNet` and wiring the state dict
  loader. A stub raises `NotImplementedError` with the exact next-steps.
- Running `preprocess.py` → `build_splits.py` → `extract_features.py`
  once over the full LIDC dataset to populate
  `GNN_for_CT_Mapping/data/nodules.parquet`, `data/splits/`, and
  `outputs/features/med3d_resnet50.parquet`.

## Usage (once Med3D is wired up)

Run all commands from the repo root with the AI venv active:

```bash
source AI/bin/activate

# 1. Build nodules.parquet from LIDC XML + DICOM
python -m GNN_for_CT_Mapping.experiments.harrison.scripts.preprocess \
    --lidc-root        /media/talos/SchoolHD/deep_learning/LIDC-IDRI \
    --annotations-root GNN_for_CT_Mapping/data/annotations \
    --out              GNN_for_CT_Mapping/data/nodules.parquet

# 2. Compute patient-level 5-fold CV splits
python -m GNN_for_CT_Mapping.experiments.harrison.scripts.build_splits \
    --nodules GNN_for_CT_Mapping/data/nodules.parquet \
    --out-dir GNN_for_CT_Mapping/data/splits

# 3. Cache frozen Med3D features (one-time, ~30 min on 3060)
python -m GNN_for_CT_Mapping.experiments.harrison.scripts.extract_features \
    --nodules     GNN_for_CT_Mapping/data/nodules.parquet \
    --metadata    /media/talos/SchoolHD/deep_learning/LIDC-IDRI/metadata/metadata.csv \
    --checkpoint  /path/to/Med3D/resnet_50.pth \
    --out         GNN_for_CT_Mapping/outputs/features/med3d_resnet50.parquet

# 4. Train GCN + MLP across all folds
python -m GNN_for_CT_Mapping.experiments.harrison.scripts.train_exp1 \
    --config GNN_for_CT_Mapping/experiments/harrison/configs/experiment.yaml

# 5. Watch metrics
tensorboard --logdir GNN_for_CT_Mapping/runs/harrison_exp1
```

## Design notes

- **No pylidc.** XML parsing is done with stdlib `xml.etree.ElementTree`;
  DICOM loading with `pydicom`. Clustering per-reader annotations into
  unified nodules uses centroid-distance matching (8 mm threshold) rather
  than the IoU-based approach pylidc uses. Sufficient for LIDC's ~7k
  nodules; revisit if cross-reader disagreement ever shows up as noise.
- **Inductive graph eval.** Train nodules fit the KNN graph; val nodes are
  inserted at eval time with edges to their k nearest training neighbors
  only. No val-to-val edges. §1.4 of the execution plan has the rationale.
- **Frozen Stage 1 features.** Med3D runs once, cached to parquet; every
  subsequent training run loads from the cache. 3060 can run Med3D
  inference comfortably; training the GCN/MLP on top is trivial.

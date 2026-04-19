# Harrison — Experiment 1 implementation

Scaffolding for Experiment 1 (GCN vs. MLP baseline) from
[`documentation/execution_plan_experiments.md`](../../documentation/execution_plan_experiments.md).

## Layout

```
experiments/harrison/
├── configs/
│   └── experiment.yaml        # Overrides on top of configs/default.yaml
├── notebooks/
│   └── gpu_test.ipynb         # torch / CUDA smoke test
├── models/
│   ├── fusion.py              # Stage 1: multi-modal fusion (image + 8 attrs + spatial)
│   ├── med3d.py               # Frozen Med3D ResNet-50 wrapper (checkpoint loader)
│   ├── mlp.py                 # Stage 2: MLP baseline
│   ├── gcn.py                 # Stage 2: 2-layer GCN
│   └── vendor/
│       ├── medicalnet_resnet.py  # Tencent/MedicalNet ResNet-50 (MIT, vendored)
│       └── LICENSE.MedicalNet    # Upstream license
└── scripts/
    ├── parse_lidc_xml.py      # Parse LidcReadMessage XML (stdlib only, no pylidc)
    ├── dicom_loader.py        # pydicom DICOM series loader + patch extractor
    ├── preprocess.py          # Build nodules.parquet from XML + DICOM
    ├── build_splits.py        # Patient-level 5-fold StratifiedGroupKFold
    ├── verify_med3d.py        # Smoke-check a downloaded Med3D checkpoint
    ├── extract_features.py    # Cache frozen Med3D features
    ├── dataset.py             # PyTorch Dataset over cached features
    ├── graph.py               # Inductive KNN graph construction
    └── train_exp1.py          # End-to-end GCN + MLP training loop
```

## One-time setup

```bash
source AI/bin/activate
pip install -r requirements.txt
nbstripout --install          # git filter so notebooks commit clean
```

## Download Med3D weights (manual)

Tencent hosts the pretrained weights on Google Drive / Tencent Weiyun,
behind a click-through link — there's no direct wget.

1. Visit https://github.com/Tencent/MedicalNet and follow the
   "Pretrain models" link (the top-level `README.md`). Download the
   archive that contains `resnet_50.pth`.
2. Unpack locally. The file is ~90 MB.
3. Save the path; `.pth` files are gitignored so nothing will leak into a
   commit.

Then smoke-test the checkpoint:

```bash
python -m GNN_for_CT_Mapping.experiments.harrison.scripts.verify_med3d \
    --checkpoint /path/to/Med3D/resnet_50.pth
```

Expected output: `forward OK — features (2, 2048)` plus non-zero
mean/std/min/max and `determinism OK`.

## End-to-end Experiment 1 run

Run every command from the repo root with the AI venv active.

```bash
# 1. Build nodules.parquet from LIDC XML + DICOM (~30–60 min on the 3060)
python -m GNN_for_CT_Mapping.experiments.harrison.scripts.preprocess \
    --lidc-root        /media/talos/SchoolHD/deep_learning/LIDC-IDRI \
    --annotations-root GNN_for_CT_Mapping/data/annotations \
    --out              GNN_for_CT_Mapping/data/nodules.parquet
# Add --limit 5 to iterate quickly on any changes before the full run.

# 2. Patient-level 5-fold CV splits
python -m GNN_for_CT_Mapping.experiments.harrison.scripts.build_splits \
    --nodules GNN_for_CT_Mapping/data/nodules.parquet \
    --out-dir GNN_for_CT_Mapping/data/splits

# 3. Cache frozen Med3D features (one-time, ~30 min on the 3060)
python -m GNN_for_CT_Mapping.experiments.harrison.scripts.extract_features \
    --nodules     GNN_for_CT_Mapping/data/nodules.parquet \
    --metadata    /media/talos/SchoolHD/deep_learning/LIDC-IDRI/metadata/metadata.csv \
    --checkpoint  /path/to/Med3D/resnet_50.pth \
    --out         GNN_for_CT_Mapping/outputs/features/med3d_resnet50.parquet

# 4. Train GCN + MLP across all folds (~1–2 hours on the 3060)
python -m GNN_for_CT_Mapping.experiments.harrison.scripts.train_exp1 \
    --config GNN_for_CT_Mapping/experiments/harrison/configs/experiment.yaml

# 5. Inspect metrics
tensorboard --logdir GNN_for_CT_Mapping/runs/harrison_exp1
```

## Current blocker

Step 3 and step 4 need Tencent/MedicalNet's `resnet_50.pth`. The loader,
forward path, and downstream pipeline are fully wired — just point them
at the checkpoint and they run.

## Design notes

- **No pylidc.** XML parsing via stdlib `xml.etree.ElementTree`; DICOM
  loading via `pydicom`. Malignancy is kept in its own field on the
  parsed nodule struct, not in the 8-attribute dict, to prevent accidental
  leakage into the Stage 1 feature fusion.
- **Clustering across readers** uses centroid-distance matching
  (8 mm threshold) rather than pylidc's IoU-based approach. Sufficient at
  LIDC scale; revisit if cross-reader disagreement ever shows up as noise.
- **DICOM path construction.** metadata.csv's file-location column is
  sometimes a stale mirror (different usernames); we reconstruct the path
  from `PatientID/StudyInstanceUID/SeriesInstanceUID` under `lidc_root`
  instead.
- **Frozen Med3D.** The segmentation-head `conv_seg` weights are dropped
  at load time — Stage 1 stops after `layer4` and global-average-pools to
  2048-D features, so conv_seg would just be dead VRAM.
- **Inductive graph eval.** Train nodules fit the KNN graph; val nodes
  are inserted at eval time with edges to their k nearest training
  neighbors only. No val-to-val edges. See §1.4 of the execution plan.

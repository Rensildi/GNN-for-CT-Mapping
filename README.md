# GNN for CT Mapping — Lung Nodule Malignancy Classification

Graph Neural Network-based malignancy classification of lung nodules in CT scans, developed for CSC 7760 Deep Learning.

**Authors:** Rensildi Kalanxhi, Harrison Lavins, Swathi Gopal

---

## Goal

Current deep learning models classify lung nodules independently, ignoring relationships between similar cases. This project builds a GNN that constructs a patient-level graph over nodules and performs malignancy classification by aggregating information across neighbors. A secondary goal is uncertainty quantification using Mahalanobis distance in the learned embedding space to flag out-of-distribution cases.

## Architecture

<img src="proposal/architecture_diagram_v3.png" width="500" alt="System architecture diagram"/>

Two-stage pipeline:

**Stage 1 — Multi-modal feature extraction (frozen at inference)**

Each nodule is represented by three fused components:
- **Image:** 48³-voxel CT patch → frozen Med3D ResNet-50 → linear projection
- **Clinical:** 8 LIDC-IDRI radiologist attributes (subtlety, sphericity, margin, etc.) → learned embeddings
- **Spatial:** (x, y, z) coordinates → sinusoidal positional encoding

**Stage 2 — Graph construction and classification (trained)**

- KNN graph over nodule feature vectors (cosine similarity, default k=10)
- 2-layer custom GCN with dropout
- Binary classification head (benign / malignant)
- Weighted cross-entropy loss for class imbalance
- Mahalanobis distance in embedding space for OOD detection

## Datasets

| Dataset | Scans | Nodules | Labels |
|---------|-------|---------|--------|
| LIDC-IDRI (primary) | 1,018 | 7,371 | Up to 4 radiologist scores per nodule; binarized at ≤2 / ≥4 |
| LUNA25 (cross-dataset eval) | 4,096 | 6,163 | Binary from clinical follow-up |

Full CT volumes are not stored in this repository. See [`GNN_for_CT_Mapping/data/README.md`](GNN_for_CT_Mapping/data/README.md) for download instructions.

## Repository Structure

```
.
├── GNN_for_CT_Mapping/
│   ├── src/                # Importable Python package
│   │   ├── data/           # Dataset classes, pylidc loaders, preprocessing
│   │   ├── models/         # GCN layers, feature extractors, fusion
│   │   ├── training/       # Training loops, loss functions, metrics
│   │   └── utils/          # Shared helpers
│   ├── scripts/            # Entry-point scripts (train, evaluate, preprocess)
│   ├── notebooks/          # Jupyter notebooks for EDA and analysis
│   ├── configs/
│   │   ├── default.yaml    # Hyperparameters
│   │   └── paths.yaml      # Dataset path template (copy → paths.local.yaml)
│   ├── data/
│   │   ├── annotations/    # LIDC-IDRI nodule metadata and attribute CSVs
│   │   └── splits/         # Patient-level CV fold definitions
│   ├── outputs/            # gitignored — checkpoints and predictions
│   ├── runs/               # gitignored — TensorBoard event files
│   └── figures/            # Saved plots and diagrams (committed)
└── proposal/               # Project proposal (Markdown + LaTeX source + PDF + architecture diagram)
```

## Setup

```bash
# Create and activate the virtual environment
python3 -m venv AI
source AI/bin/activate

# Install dependencies (once requirements.txt is added)
pip install -r requirements.txt
```

Configure local dataset paths:

```bash
cp GNN_for_CT_Mapping/configs/paths.yaml GNN_for_CT_Mapping/configs/paths.local.yaml
# Edit paths.local.yaml with your local CT volume directories
```

Launch TensorBoard:

```bash
tensorboard --logdir GNN_for_CT_Mapping/runs
```

Compile the proposal PDF:

```bash
cd proposal/Proposal_LaTeX
pdflatex Proposal.tex
```

For LaTeX environment setup (Windows and Ubuntu), see [`proposal/Proposal_LaTeX/LATEX_SETUP.md`](proposal/Proposal_LaTeX/LATEX_SETUP.md).

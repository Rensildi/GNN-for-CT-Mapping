# GNN for CT Mapping — Lung Nodule Malignancy Classification

Graph Neural Network–based malignancy classification of lung nodules in CT scans. CSC 7760 Deep Learning, Spring 2026.

**Authors:** Harrison Lavins, Rensildi Kalanxhi, Swathi Gopal

---

## Goal

Modern CT-based lung-nodule classifiers achieve AUC 0.88–0.96 on LIDC-IDRI but treat each nodule as an independent classification target. This project asks whether modeling **inter-nodule similarity** as a graph improves benign-versus-malignant classification over per-nodule classification.

The headline question — *does graph-based similarity reasoning beat a parameter-matched MLP on LIDC-IDRI?* — has been answered, with nuance: see the [final report](report/v2/gnn_lung_nodule_classification.pdf).

## Headline findings

- **Experiment 1.** GCN does **not** beat the parameter-matched MLP under radiologist-consensus labels (AUC 0.949 vs 0.968, paired Wilcoxon p = 0.031, MLP wins five of five folds; H1 rejected).
- **Experiment 2.** Graph-construction sweep over $k \in \{5, 10, 15, 20\}$ × {cosine, euclidean} does not recover the gap; smaller $k$ is monotonically better (over-smoothing fingerprint).
- **Experiment 3.** FMCIB outperforms Med3D by **+0.27 AUC** on image-only inputs but the gap closes once attribute features are added — both encoders saturate at the attribute-driven ceiling.
- **Experiment 3.1.** Re-evaluation against 40 pathology-confirmed nodules **inverts** the ranking: FMCIB image-only is the best cell (AUC 0.887, 95% CI [0.761, 0.982]), and attribute-saturated configurations lose 0.23–0.28 AUC. Radiologist-consensus and pathology labels agree on only 67.5% of the matched subset.
- **Experiment 4 (in progress).** Cross-dataset transfer to LUNA25 reaches ~73% balanced accuracy after schema-aware adaptation; preliminary Graph Transformer matches the GCN once the attribute shortcut is removed.

The interpretation: the LIDC-IDRI attribute branch acts as a label shortcut rather than orthogonal clinical signal. FMCIB image-only is the only feature configuration whose performance is invariant to label source.

## Architecture

<img src="proposal/architecture_diagram_v3.png" width="500" alt="System architecture diagram"/>

Two-stage pipeline:

**Stage 1 — Multi-modal feature fusion (frozen at inference).** Each nodule is represented by a 256-dimensional vector concatenating three branches and projected through a LayerNorm + Linear:
- **Image:** $48^3$- or $50^3$-voxel CT patch → frozen 3D ResNet (Med3D ResNet-50 baseline; FMCIB foundation model as the committed follow-up encoder) → trainable linear projection.
- **Clinical:** 8 LIDC-IDRI radiologist attributes (subtlety, internal structure, calcification, sphericity, margin, lobulation, spiculation, texture) → learned embedding tables.
- **Spatial:** $(x, y, z)$ centroid → sinusoidal positional encoding.

**Stage 2 — Graph construction and node classification (trained end-to-end).**
- Cohort-wide KNN graph in the 256-d Stage-1 space (default $k = 10$, cosine similarity), built **inductively**: training-fold edges only; validation nodules attach to their $k$ nearest training neighbors at evaluation, no validation-to-validation edges.
- Two GCNConv layers with symmetric normalization, ReLU, and Dropout(0.3); linear head → benign/malignant logits.
- Parameter-matched MLP control (within $\pm 20\%$ params) shares all hyperparameters except the architecture itself.
- Weighted cross-entropy loss; patient-level five-fold StratifiedGroupKFold; Adam (lr 1e-3, wd 1e-4), patience-15 early stopping on validation AUC.

## Datasets

| Dataset | Scans | Nodules | Labels |
|---------|-------|---------|--------|
| **LIDC-IDRI** (primary) | 1,010 patients / 1,308 studies | 1,128 labeled (805 benign, 323 malignant) across 588 patients after preprocessing | Up to 4 radiologist scores; binarized at mean $\le 2$ benign / $\ge 4$ malignant |
| **LIDC pathology subset** (Experiment 3.1) | 102 patients with all-agree per-nodule diagnoses | 40 matched (12 benign, 28 malignant) | Biopsy / surgical resection / 2-year follow-up (TCIA `tcia-diagnosis-data-2012-04-20.xls`) |
| **LUNA25** (cross-dataset eval, Experiment 4) | 4,096 NLST exams | 6,163 | Binary from clinical follow-up (Zenodo, CC BY 4.0) |

Full CT volumes are not stored in this repository. See [`GNN_for_CT_Mapping/data/README.md`](GNN_for_CT_Mapping/data/README.md) for download instructions.

## Experiments

| # | Question | Owner |
|---|----------|-------|
| 1 | GCN vs. parameter-matched MLP under full multi-modal features | Harrison |
| 2 | Graph-construction ablation over $(k, \text{metric})$ | Harrison |
| 2.1 | ResNet-18 encoder replication of Experiments 1 & 2 | Rensildi |
| 3 | Feature-modality × encoder grid (image / +attrs / +spatial × Med3D / FMCIB) | Harrison |
| 3.1 | Re-evaluation of every Exp. 3 cell against pathology-confirmed labels | Harrison |
| 4 | Cross-dataset generalization to LUNA25; preliminary Graph Transformer | Swathi |

## Repository structure

```
.
├── README.md
├── requirements.txt
├── CLAUDE.md                    # gitignored
├── GNN_for_CT_Mapping/
│   ├── src/                     # Shared, promoted Python package (data/, models/, training/, utils/)
│   ├── scripts/                 # Shared entry-point scripts
│   ├── notebooks/               # Shared analysis notebooks
│   ├── configs/
│   │   ├── default.yaml         # Shared hyperparameters (edit via PR)
│   │   └── paths.yaml           # Dataset path template (copy → paths.local.yaml)
│   ├── data/
│   │   ├── annotations/         # LIDC-IDRI XML, attribute CSVs, nodule parquets
│   │   └── splits/              # Patient-level CV fold definitions
│   ├── documentation/           # Cross-experiment notes
│   ├── experiments/
│   │   ├── harrison/            # Exp. 1, 2, 3, 3.1 — preprocessing, GCN/MLP heads, encoder ablation
│   │   │   ├── scripts/         # train_exp{1,2,3}.py, extract_features{,_fmcib}.py, draw_*.py, …
│   │   │   ├── configs/         # experiment.yaml override
│   │   │   ├── documentation/   # Per-experiment write-ups
│   │   │   ├── figures/         # Slide-ready and report figures
│   │   │   ├── models/          # In-progress model variants
│   │   │   └── notebooks/
│   │   ├── rensildi/            # Exp. 2.1 ResNet-18 replication, project-structure scaffolding
│   │   └── swathi/              # Exp. 4 LUNA25 transfer, Graph Transformer prototype
│   ├── outputs/                 # gitignored — checkpoints, predictions, cached features (Med3D + FMCIB)
│   ├── runs/                    # gitignored — TensorBoard event files (per-experiment subdirs)
│   └── figures/                 # Shared plots and diagrams (committed)
├── proposal/                    # Project proposal: docx + LaTeX source + PDF + architecture diagram
│   └── Proposal_LaTeX/          # Build with pdflatex; see LATEX_SETUP.md
├── report/                      # Final research report
│   ├── build_report_docx.py     # Markdown → docx generator
│   ├── v1/                      # First draft (md + docx)
│   └── v2/                      # Final NeurIPS-style two-column paper
│       ├── gnn_lung_nodule_classification.tex
│       ├── gnn_lung_nodule_classification.pdf
│       ├── gnn_lung_nodule_classification.docx
│       └── figures/
└── presentation/                # Slide deck (build_presentation.py + compiled PDF)
```

## Setup

Run these once after cloning:

```bash
# Activate the virtual environment
source AI/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install the notebook output filter (prevents spurious notebook conflicts)
nbstripout --install

# Set up your local dataset paths
cp GNN_for_CT_Mapping/configs/paths.yaml GNN_for_CT_Mapping/configs/paths.local.yaml
# Edit paths.local.yaml with the paths to your local CT volume directories
```

The codebase is developed and tested on a single NVIDIA RTX 3060 (12 GB VRAM); all architecture choices fit within that budget.

## Workflow

### Day-to-day experimentation

Personal work — notebooks, model variants, training scripts, and result figures — lives under `GNN_for_CT_Mapping/experiments/<your-name>/`. This isolates each person's work and avoids merge conflicts on shared files.

Each person has a `configs/experiment.yaml` that is merged on top of the shared `default.yaml` at load time. Specify only the keys you are changing:

```yaml
# experiments/harrison/configs/experiment.yaml
model:
  gcn_layers: 3
graph:
  k_neighbors: 15
```

Model checkpoints and predictions write to `outputs/` under a named subdirectory so runs don't overwrite each other (e.g. `outputs/checkpoints/harrison_3layer_gcn/`). That directory is gitignored.

### Promoting work to shared code

When a model variant, preprocessing step, or utility is ready for the team to build on, open a pull request to move it from `experiments/<your-name>/` into `src/`. Changes to `src/` and `configs/default.yaml` always go through a PR.

### Avoiding merge conflicts

- **Notebooks:** `nbstripout` is configured as a git filter (`.gitattributes`) and strips cell outputs and execution counts on commit.
- **Configs:** Use your personal `experiment.yaml` for overrides rather than editing `default.yaml` directly.
- **Models:** Write variants in `experiments/<your-name>/models/` first; promote to `src/models/` via PR.
- **Binary files:** `.gitattributes` marks images, PDFs, checkpoints, and serialized data as binary so git never attempts to merge them.

### TensorBoard

```bash
tensorboard --logdir GNN_for_CT_Mapping/runs
```

## Building the artifacts

### Proposal PDF

```bash
cd proposal/Proposal_LaTeX
pdflatex Proposal.tex
```

For LaTeX environment setup (Windows and Ubuntu), see [`proposal/Proposal_LaTeX/LATEX_SETUP.md`](proposal/Proposal_LaTeX/LATEX_SETUP.md).

### Final report (NeurIPS-style two-column paper)

```bash
cd report/v2
pdflatex gnn_lung_nodule_classification.tex
pdflatex gnn_lung_nodule_classification.tex   # second pass for cross-references
```

The compiled PDF is committed at [`report/v2/gnn_lung_nodule_classification.pdf`](report/v2/gnn_lung_nodule_classification.pdf).

### Slide deck

```bash
python presentation/build_presentation.py
```

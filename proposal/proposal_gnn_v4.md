# Deep Learning Project Proposal

Swathi Gopal, Harrison Lavins, Rensildi Kalanxhi

---

## Title of the Project

GNN-Based Lung Nodule Malignancy Classification Using Multi-Modal Feature Embedding on CT Scans

## Goal(s) of the Project

The goal of this project is to determine whether modeling inter-nodule similarity as a graph improves lung nodule malignancy classification compared to classifying each nodule independently. We construct a nodule-similarity graph where each node represents a lung nodule, characterized by three feature types: 3D imaging features from a frozen pretrained medical CNN, learned embeddings of radiologist-scored clinical attributes, and sinusoidal positional encodings of the nodule's spatial location. Edges connect nodules with similar multi-modal feature profiles. A Graph Convolutional Network (GCN) classifies each node as benign or malignant by aggregating information from neighboring nodes. We compare this approach against an MLP baseline using identical features but no graph structure, and evaluate cross-dataset generalization from LIDC-IDRI to LUNA25.

## Background of the Project

This project lies at the intersection of **medical research** (lung cancer diagnosis from CT imaging) and **computer vision** (3D volumetric feature extraction), applying **graph-based deep learning** to a clinical classification task.

Lung cancer is the leading cause of cancer-related mortality worldwide. Deep learning models for CT-based nodule classification have achieved strong performance (AUC 0.88–0.96 on LIDC-IDRI), but most treat each nodule independently, ignoring relationships between similar cases. Graph Neural Networks address this by modeling nodules as nodes in a similarity graph, where message passing allows each nodule's classification to be informed by its neighbors. Patient-similarity GNNs have improved classification in cancer subtyping and prognosis prediction, but have not been applied to lung nodule malignancy classification on CT imaging data. Our project fills this gap by applying a GCN to a k-nearest-neighbor similarity graph built from multi-modal nodule features. The node features fuse three sources: 3D imaging features from a frozen Med3D ResNet-50 pretrained on 23 medical datasets, learned embeddings for the eight LIDC-IDRI radiologist attributes, and sinusoidal positional encodings of nodule coordinates. This multi-modal fusion ensures the graph captures both visual appearance and clinical characteristics when connecting similar nodules.

## Reference Papers

1. Kipf, T. N. & Welling, M. (2017), "Semi-Supervised Classification with Graph Convolutional Networks," *ICLR 2017*. — Foundational GCN architecture for semi-supervised node classification on graphs; defines the message-passing framework used in this project.

2. Chen, S. et al. (2019), "Med3D: Transfer Learning for 3D Medical Image Analysis," *arXiv:1904.00625*. — Provides the Med3D pretrained 3D ResNet-50 weights (trained on 23 medical segmentation datasets) used as the frozen image feature extractor.

3. Ma, X. et al. (2023), "A novel fusion algorithm for benign-malignant lung nodule classification on CT images," *BMC Pulmonary Medicine*, 23:462. — Closest existing work using GCN for lung nodule classification on LIDC-IDRI (AUC 0.9629). Uses GCN for feature fusion across CNNs; our approach differs by using GCN for node classification on a nodule-similarity graph with multi-modal features.

## Deep Learning Model (Base Model)

The pipeline has two stages, illustrated in the architecture diagram below.

![Architecture Diagram](architecture_diagram_v3.png)

**Figure 1.** Architecture: multi-modal feature extraction feeds a GCN on a nodule-similarity graph.

**Stage 1 — Multi-modal feature extraction (frozen, one-time):** For each nodule, a 48³-voxel 3D patch (HU-clipped, normalized) is passed through a frozen Med3D ResNet-50, producing a high-dimensional vector projected to a compact representation via a trainable linear layer. The eight LIDC-IDRI radiologist attributes (subtlety, sphericity, margin, lobulation, spiculation, texture, internal structure, calcification) are encoded via per-feature learned embedding layers. Nodule spatial coordinates (x, y, z) are encoded with sinusoidal positional encoding. All components are concatenated and layer-normalized into a unified node feature vector.

**Stage 2 — Graph construction and GCN classification (trained):** A k-nearest-neighbor graph (cosine similarity) connects nodules with similar multi-modal feature profiles. A 2-layer GCN with dropout performs message passing across this graph, updating each node's representation by aggregating information from its neighbors. A linear classification head outputs benign vs. malignant probabilities. Training uses weighted cross-entropy loss to handle class imbalance. The model is implemented in PyTorch Geometric using the GCNConv layer. Data augmentation is achieved by stochastically sampling one of the up to four radiologist annotations per nodule in each training epoch, varying the patch crop and attribute values while keeping the graph structure fixed.

## Experiments

**Experiment 1 — GCN vs. MLP baseline:** Train the GCN on the LIDC-IDRI nodule-similarity graph and compare against an MLP baseline using identical multi-modal features but no graph structure. Report AUC, accuracy, sensitivity, and specificity using 5-fold cross-validation with patient-level splits. This directly tests whether graph-based neighbor aggregation improves classification.

**Experiment 2 — Graph construction ablation:** Vary the number of neighbors k (5, 10, 15, 20) and compare cosine similarity vs. Euclidean distance for edge construction. This identifies the optimal graph density and similarity metric for this task.

**Experiment 3 — Feature modality ablation:** Evaluate the contribution of each feature source by comparing: (a) image features only, (b) image + radiologist attributes, (c) image + attributes + spatial encoding. Validate against the 157 pathologically confirmed LIDC-IDRI diagnoses to assess whether multi-modal features align with ground-truth pathology.

**Experiment 4 — Cross-dataset generalization on LUNA25:** Apply the LIDC-IDRI-trained model to LUNA25 nodules (natively binary-labeled from NLST clinical follow-up). This tests whether learned representations generalize to nodules from a different patient population with different scanners and protocols.

## Dataset(s)

**LIDC-IDRI** (primary): 1,018 thoracic CT scans with 7,371 annotated nodules (≥3mm) from The Cancer Imaging Archive. Each nodule is rated by up to four radiologists on a 1–5 malignancy scale with eight descriptive attributes. Labels are binarized: mean score ≤2 → benign, ≥4 → malignant, ≈3 excluded, yielding ~800–1,400 labeled nodules. A separate set of 157 pathologically confirmed diagnoses provides ground-truth validation. Preprocessing uses the pylidc library to extract 3D patches with metadata. Augmentation leverages the multiple annotations per nodule: each epoch stochastically samples one annotation per nodule, varying the patch crop and attribute values.

**LUNA25** (cross-dataset evaluation): 4,096 CT exams with 6,163 nodules from the National Lung Screening Trial, released for MICCAI 2025. Nodules have binary malignancy labels confirmed by clinical follow-up, with 3D coordinates, patient age, and sex. Available on Zenodo under CC BY 4.0. Serves as a natural generalization test because it uses different source patients, scanners, and protocols than LIDC-IDRI.

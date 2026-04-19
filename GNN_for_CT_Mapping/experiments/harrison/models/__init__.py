# Model implementations for Experiment 1 — GCN vs. MLP baseline.
#
# The architecture follows `documentation/execution_plan_experiments.md` §1.3:
#   - fusion.py : Stage 1 multi-modal feature fusion (image + attributes + spatial)
#   - med3d.py  : frozen Med3D ResNet-50 image encoder wrapper
#   - mlp.py    : MLP baseline (no graph structure)
#   - gcn.py    : 2-layer GCN with dropout (message passes over KNN graph)
#
# Both Stage 2 heads (mlp.py, gcn.py) consume the same 256-dim fused node
# features, so the head-to-head comparison isolates the effect of graph
# aggregation.

from .fusion import MultiModalFusion
from .mlp import MLPClassifier
from .gcn import GCNClassifier

__all__ = ["MultiModalFusion", "MLPClassifier", "GCNClassifier"]

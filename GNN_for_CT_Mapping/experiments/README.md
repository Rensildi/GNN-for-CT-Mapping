# Experiments

Each team member has a personal working directory here for exploratory notebooks, architecture variants, and config overrides. Work in progress and one-off experiments live here; code that becomes part of the shared pipeline gets promoted to `../src/` via a pull request.

## Structure

```
experiments/
├── harrison/
├── swathi/
└── rensildi/
    ├── notebooks/    # Personal Jupyter notebooks
    ├── models/       # Personal model/architecture variants
    ├── configs/
    │   └── experiment.yaml   # Overrides merged on top of ../../configs/default.yaml
    ├── scripts/      # Personal training / eval scripts
    └── figures/      # Plots generated from personal experiments
```

## Conflict-avoidance rules

- **Don't edit another person's directory.** All personal experiment work stays under your own subdirectory.
- **Don't edit `../src/` directly for experimental changes.** Write your variant in `your-name/models/` first. Open a PR to promote it to `../src/` once the team agrees.
- **Strip notebook outputs before committing.** Install nbstripout once after cloning — it runs automatically as a git filter:
  ```bash
  pip install nbstripout && nbstripout --install
  ```
- **Config overrides only.** `experiment.yaml` should only contain keys that differ from `../../configs/default.yaml` — not a full copy — so changes to shared defaults propagate automatically.

## Config override convention

`experiment.yaml` is merged on top of `../../configs/default.yaml` at load time. Example:

```yaml
# Only override what you're changing
model:
  gcn_layers: 3
graph:
  k_neighbors: 15
```

## Outputs

Checkpoints and predictions write to `../../outputs/` using a subdirectory named after the experiment (e.g. `outputs/checkpoints/harrison_3layer_gcn/`). That directory is gitignored.

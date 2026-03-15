# III_dbgnn_behavior

DBGNN behavior analysis workspace with a strict split:

- main logic in `src/`
- notebooks as thin runners
- focus: DBGNN experiments and GCN-based analysis

## Structure

- `src/workflows/`
  - `train.py`: dataset load + train/load + macro metrics
  - `counterfactual.py`: counterfactual search and plot generation
  - `ho_effect.py`: higher-order impact + ablation summary
- `notebooks/`
  - `01_train_dbgnn.ipynb`
  - `02_counterfactual.ipynb`
  - `03_higher_order_effects.ipynb`

## Notebook Order

1. `01_train_dbgnn.ipynb`
2. `02_counterfactual.ipynb`
3. `03_higher_order_effects.ipynb`

## Notes

- Dataset bundles are stored in `runs/<run_name>/dataset/`.
- Counterfactual plots are saved under `notebooks/plots/`.

# Notebooks

Current notebooks (run in order):

1. `01_train_dbgnn.ipynb`
2. `02_counterfactual.ipynb`
3. `03_higher_order_effects.ipynb`
4. `04_gcn_structure_removal.ipynb`
5. `05_ba_shapes_structure_removal.ipynb`

Compatibility aliases (same logic as 04/05):

- `gnn_structure_removal_experiment.ipynb`
- `ba_shapes_structure_removal_experiment_fixed.ipynb`

These notebooks are thin wrappers.
Core logic has been moved to:

- `src/notebook_scripts/01_train_dbgnn.py`
- `src/notebook_scripts/02_counterfactual.py`
- `src/notebook_scripts/03_higher_order_effects.py`
- `src/notebook_scripts/04_gcn_structure_removal.py`
- `src/notebook_scripts/05_ba_shapes_structure_removal.py`

Execution helper:

- `src/notebook_runner.py`

Dataset/config selection:

- In each notebook, edit the dataset variables and `overrides` dict in the final code cell.
- `01_train_dbgnn.ipynb`: set `dataset_key` (`hospital`, `highschool`, `office`, `temporal_clusters`).
- `02_counterfactual.ipynb` / `03_higher_order_effects.ipynb`: set `dataset_name` + `dataset_kwargs`.
- These values override defaults in the corresponding script under `src/notebook_scripts/`.

# Data Module Overview (refactored)

This folder contains the data pipeline used by the notebooks and experiments.

## One-liner usage (recommended)

```python
from time_to_explain.data import load_dataset

bundle, paths, summary = load_dataset(
    "multihost",
    ensure_real=True,      # auto-download real datasets when possible
    force_download=False,  # set True to re-download
    overwrite=False,       # set True to re-generate / re-process
    do_index=True,         # build resources/datasets/explain_index/<name>.csv
)
```

## Main entry points

- `tgnn_prepare.py` — unified "prepare" function for both **synthetic** and **real** datasets:
  - synthetic recipes: generated and exported to TGAT/TGN `ml_*.csv` format
  - real datasets: downloaded (if supported) and processed to `ml_*.csv`
- `tgnn_setup.py` — download + processing utilities for **real** datasets.
  - includes TemGX OpenReview supplementary download for **multihost**
- `synthetic.py` / `synthetic_recipes/` — synthetic dataset generation recipes
- `io.py` — load and save processed datasets (`ml_*.csv`, `ml_*.npy`, `ml_*_node.npy`)
- `explain_index.py` — explain-index generation (`resources/datasets/explain_index/<name>.csv`)
- `tempme_preprocess.py` — optional TempME-specific preprocessing (kept separate)

## Outputs layout

All datasets use the flat layout:

```
resources/datasets/raw/<name>.csv
resources/datasets/processed/ml_<name>.csv
resources/datasets/processed/ml_<name>.npy
resources/datasets/processed/ml_<name>_node.npy
resources/datasets/explain_index/<name>.csv
```

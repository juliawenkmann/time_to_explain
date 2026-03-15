# CBM standalone workspace

This folder is a self-contained copy of the CBM notebook workflow.

## Main notebook
- `notebooks/cbm.ipynb`

## Included runtime assets
- `src/cbm/` (core CBM logic + preset/config helpers extracted from the notebook)
- `src/data/` (minimal dataset loaders/registry used by the notebook)
- `src/viz/` (heavy visualization/explanation code extracted from notebook cells)
- `src/runtime_utils.py` (seed/device helpers)
- `data/genders.csv` (needed by SMS/highschool label handling)
- `data/temporal_clusters.tedges` (local temporal-clusters dataset)
- `notebooks/runs/`, `figures/`, `tmp/jupyter-notebook/` (output locations)

## Run
```bash
cd IV_cbm_implementation
python -m pip install -r requirements.txt
jupyter lab
```

Open the notebook and run cells top-to-bottom.

# time_to_explain_official

This repository is the working thesis workspace. It is split into four main
parts that can be run independently:

- `I_explainer_benchmark`
  Temporal-graph explainer benchmark and evaluation pipeline.
- `II_instability_sources`
  Instability-source analysis and figure generation.
- `III_dbgnn_behavior`
  DBGNN behavior, counterfactual, and structure-removal experiments.
- `IV_cbm_implementation`
  Standalone CBM notebook workflow.

Run Jupyter from the repository root. Most notebooks bootstrap their paths from
the root layout and are easiest to execute from here.

## Repository Layout

```text
time_to_explain_official/
  I_explainer_benchmark/
    configs/      notebook + model + explainer configs
    notebooks/    main benchmark notebooks
    scripts/      helper scripts
    src/          benchmark code
    submodules/   vendored model/explainer code
    resources/    local data, checkpoints, results (gitignored)
  II_instability_sources/
    notebooks/    instability notebook
    src/          reusable logic for figures
    figures/      generated artifacts (gitignored)
  III_dbgnn_behavior/
    notebooks/    DBGNN notebooks
    src/          extracted notebook logic and workflows
    runs/         generated runs (gitignored)
  IV_cbm_implementation/
    notebooks/    CBM notebook
    src/          extracted runtime code
    data/         local data assets (gitignored)
```

## Environment Setup

The repository is notebook-heavy. A shared Python environment is usually the
best workflow.

### Base environment for Part I and Part II

```bash
python -m pip install -r requirements.txt
```

Recommended:

- Python `3.10` or `3.11`
- a named Jupyter kernel for the environment you actually use
- launch Jupyter from the repository root

### Extra environment for Part III

```bash
python -m pip install -e './III_dbgnn_behavior[dbgnn,viz]'
```

If you do not want the editable install, the fallback is:

```bash
python -m pip install -r III_dbgnn_behavior/notebooks/requirements.txt
```

### Extra environment for Part IV

```bash
python -m pip install -r IV_cbm_implementation/requirements.txt
```

## How To Run Each Part

### I_ explainer benchmark

Open the benchmark notebooks:

```bash
jupyter lab I_explainer_benchmark/notebooks
```

Code and config live in:

- `I_explainer_benchmark/src`
- `I_explainer_benchmark/configs`
- `I_explainer_benchmark/scripts`

Local-only runtime assets live in:

- `I_explainer_benchmark/resources`
- `I_explainer_benchmark/submodules/explainer/TempME/processed`
- model/explainer checkpoints inside `resources/models`

Recommended notebook order:

1. `I_explainer_benchmark/notebooks/00_prepare_datasets.ipynb`
   Prepare or verify processed datasets and explain-index files.
2. `I_explainer_benchmark/notebooks/01_train_models.ipynb`
   Train or reuse backbone models.
3. Explainer notebooks in `I_explainer_benchmark/notebooks/explainer_notebooks/`
   Run the explainers and export benchmark-ready results.
4. `I_explainer_benchmark/notebooks/02_summarize_evaluation.ipynb`
   Aggregate metrics and build summary plots.
5. `I_explainer_benchmark/notebooks/03_qualitative_simulate_v1.ipynb`
   Generate the qualitative case-study figures.

Explainer notebook order:

1. `02_cody.ipynb`
2. `03_greedy.ipynb`
3. `04_temgx.ipynb`
4. `05_tgnnexplainer.ipynb`
5. `06_pg.ipynb`
6. `07_tempme.ipynb`
7. `08_khop.ipynb`
8. `09_random.ipynb`
9. `01_my_cf.ipynb`

Current benchmark defaults are set up around `simulate_v1` and `tgn`.

Useful notes:

- Results, checkpoints, processed data, and summary exports are written below
  `I_explainer_benchmark/resources/`.
- If a notebook needs a vendored explainer/model checkpoint, the expected local
  path is usually somewhere below `I_explainer_benchmark/resources/models/`.
- If you only want synthetic-dataset generation, use the scripts in
  `I_explainer_benchmark/scripts/`.

### II_ instability sources

You can run Part II either as a script or as a notebook.

Script entrypoint:

```bash
python II_instability_sources/generate_instability_figures.py
```

Notebook entrypoint:

```bash
jupyter lab II_instability_sources/notebooks/temporal_instability_viz.ipynb
```

Outputs are written to `II_instability_sources/figures/`.

### III_ DBGNN behavior

Open the notebook folder:

```bash
jupyter lab III_dbgnn_behavior/notebooks
```

Main logic has already been moved out of notebooks into:

- `III_dbgnn_behavior/src/notebook_scripts`
- `III_dbgnn_behavior/src/workflows`
- `III_dbgnn_behavior/src/explainers`
- `III_dbgnn_behavior/src/eval`

Recommended notebook order:

1. `III_dbgnn_behavior/notebooks/01_train_dbgnn.ipynb`
2. `III_dbgnn_behavior/notebooks/02_counterfactual.ipynb`
3. `III_dbgnn_behavior/notebooks/03_higher_order_effects.ipynb`
4. `III_dbgnn_behavior/notebooks/04_gcn_structure_removal.ipynb`
5. `III_dbgnn_behavior/notebooks/05_ba_shapes_structure_removal.ipynb`

Notes:

- `III_dbgnn_behavior/notebooks/00_dbgnn.ipynb` is useful for inspection, but
  the main reproducible path is `01` through `05`.
- Compatibility aliases and older experiments are under
  `III_dbgnn_behavior/notebooks/legacy/`.
- Generated runs and plots are local-only and live under
  `III_dbgnn_behavior/runs/`, `III_dbgnn_behavior/plots/`, and
  `III_dbgnn_behavior/notebooks/plots/`.

### IV_ CBM implementation

Open the CBM notebook:

```bash
jupyter lab IV_cbm_implementation/notebooks/cbm.ipynb
```

The runtime code is split into:

- `IV_cbm_implementation/src/cbm`
- `IV_cbm_implementation/src/data`
- `IV_cbm_implementation/src/viz`

Local assets and generated outputs are written below:

- `IV_cbm_implementation/data`
- `IV_cbm_implementation/figures`
- `IV_cbm_implementation/plots`
- `IV_cbm_implementation/tmp`

## Practical Workflow

If you are working across the full thesis repo, the cleanest order is:

1. Start Jupyter from the repository root.
2. Run Part I notebooks for benchmark data, model checkpoints, and explainer
   outputs.
3. Run Part II for the instability figures.
4. Run Part III for DBGNN-specific analysis.
5. Run Part IV for the CBM workflow.

## Large Files And Git

Large local artifacts are intentionally not part of the clean source snapshot.
The root `.gitignore` excludes:

- `I_explainer_benchmark/resources/`
- large local data folders
- generated run/plot/figure directories
- notebook temp/cache files

That means the source code, configs, notebooks, and lightweight metadata stay
versioned, while heavy experimental outputs remain local.

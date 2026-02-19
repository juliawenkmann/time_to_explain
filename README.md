# Temporal Graph XAI Benchmark

This repository provides a unified framework for benchmarking explainability
methods on temporal graph neural networks (TGNNs).  It wraps the original
implementations of PGExplainer, T-GNNExplainer, CoDy, and GreedyCF, exposes a
consistent API for adding new models/metrics/explainers, and ships with a CLI
for running reproducible experiments from YAML configurations.

## Quick Start

1. **Install dependencies**
   ```bash
   pip install -e .
   ```

2. **Prepare data (download/process/index)**
   ```bash
   python -m time_to_explain.cli data prepare --only reddit,wikipedia --index-size 500
   # optional synthetic builds:
   # python -m time_to_explain.cli data prepare --synthetic erdos_small,hawkes_small --overwrite-synthetic
   ```

3. **Prepare datasets/checkpoints**  
   Place preprocessed datasets under `resources/processed/<dataset>` and model
   checkpoints under `resources/models/<dataset>` (matching the layout used in
   the original projects).

4. **Run an experiment**
   ```bash
   python -m time_to_explain.cli eval run --config configs/experiments/example.yaml
   ```
   The CLI will instantiate the requested dataset, model, explainers, metrics,
   and sampler, execute explanations for the sampled events, and write results to
   `experiment.output_dir` (CSV + Parquet by default).

## Configuration Schema

Experiments are fully described in YAML:

```yaml
dataset:
  builder: processed            # registered dataset builder
  path: resources/processed/wikipedia

model:
  builder: ttgn                 # registered model builder
  checkpoint_path: resources/models/wikipedia/checkpoints/TGN-wikipedia-19.pth

explainers:
  - builder: pgexplainer        # wraps TPGExplainer
    alias: pg
    embedding: static
    fit:                        # optional training hyper-parameters
      epochs: 50
  - builder: tgnnexplainer
    alias: tgnn
    pgexplainer_checkpoint: outputs/pgexplainer/checkpoints/pgexplainer.pt

metrics:
  - builder: fidelity_minus
    k: 10

experiment:
  output_dir: outputs/wikipedia_default
  sampler:
    builder: random
    count: 32
    section: test
    seed: 42
```

Keys map directly to registry entries (see below).  Any mapping can include an
`alias` field, used for human-readable names in the result table.

## Built-in Components

| Group       | Registry Key          | Description                                            |
|-------------|-----------------------|--------------------------------------------------------|
| Dataset     | `processed`           | Loads `*_data.csv` + `*_edge_features.npy` + `*_node_features.npy`. |
| Model       | `tgn`, `tgat`, `ttgn`, `explainer` | Wrappers around the original Twitter TGN/TGAT/TTGN implementations. `explainer` always yields the TTGN-based wrapper exposing the superset of methods required by all explainers. |
| Explainer   | `pgexplainer`         | Adapter for the dynamic PGExplainer used in T-GNNExplainer. |
|             | `tgnnexplainer`       | Adapter for T-GNNExplainer (Monte-Carlo tree search).  |
|             | `cody`                | Adapter for the CoDy counterfactual explainer.         |
|             | `greedycf`            | Adapter for the GreedyCF counterfactual explainer.     |
| Metric      | `fidelity_plus`       | 1 - |z_expl - z_full| using explanation-only edges.     |
|             | `fidelity_minus`      | |z_full - z_removed| after dropping explanation edges. |
|             | `sparsity`            | |E_expl| / |E_candidates|.                             |
|             | `aufsc`               | Area under the fidelity-sparsity curve.               |
| Sampler     | `random`              | Samples event IDs via `dataset.extract_random_event_ids`. |

Additional components can be registered by decorating factories with the
helpers in `time_to_explain.core.registries` (e.g., `@register_explainer`).

## Pipelines (CLI + Notebooks)

- Call the CLI for reproducible runs:
  ```
  python -m time_to_explain.cli eval run --config configs/experiments/example.yaml
  python -m time_to_explain.cli eval sweep --glob 'configs/experiments/*.yaml'
  ```
- Keep notebooks thin: import the helpers in `time_to_explain.pipelines` or the
  stubs under `notebooks/src/` (`model_loader.py`, `explainer_factory.py`) and
  reuse them instead of re-writing setup code in each notebook.

### Training Models & PGExplainer

1. **Backbone model** — Call
   `time_to_explain.models.training.train_model_from_config` with the
   `explainer` model builder to produce checkpoints compatible with every
   explainer:

   ```python
   from time_to_explain.models.training import train_model_from_config

   model_train_config = {
       "dataset": {
           "builder": "processed",
           "path": "resources/processed/wikipedia",
           "directed": False,
           "bipartite": True,
       },
       "model": {
           "builder": "explainer",
           "base_model": "TGAT",
           "directed": False,
           "bipartite": True,
           "device": "auto",
           "cuda": False,
           "candidates_size": 75,
       },
       "training": {
           "epochs": 30,
           "learning_rate": 1e-4,
           "early_stop_patience": 5,
           "output_dir": "outputs/wikipedia_model",
           "checkpoint_dir": "outputs/wikipedia_model/checkpoints",
           "model_dir": "outputs/wikipedia_model/models",
           "results_path": "outputs/wikipedia_model/train_results.pkl",
       },
   }

   train_result = train_model_from_config(model_train_config)
   best_checkpoint = train_result["last_checkpoint"]
   ```

2. **PGExplainer** — Reuse the checkpoint when calling
   `time_to_explain.explainer.train_pgexplainer.train_pgexplainer_from_config`:

   ```python
   from time_to_explain.explainer.train_pgexplainer import train_pgexplainer_from_config

   pg_train_config = {
       "dataset": model_train_config["dataset"],
       "model": {
           "builder": "explainer",
           "base_model": "TGAT",
           "directed": False,
           "bipartite": True,
           "device": "auto",
           "cuda": False,
           "candidates_size": 75,
           "checkpoint_path": str(best_checkpoint),
       },
       "training": {
           "epochs": 30,
           "learning_rate": 1e-4,
           "batch_size": 16,
           "candidates_size": 75,
           "output_dir": "outputs/wikipedia_model/pg_explainer",
       },
   }

   explainer, wrapper, out_dir = train_pgexplainer_from_config(pg_train_config)
   ```

## Outputs

The CLI produces a flat table with one row per (event, explainer) pair,
including:

- dataset/model identifiers
- explanation metadata (type, size, timings)
- metric values (e.g., `metric.fidelity_minus.value`, `metric.sparsity.ratio`)
- optional debug information emitted by the underlying explainer

Results are stored as CSV and Parquet under the configured `output_dir`.

## Documentation

- [Architecture overview](docs/architecture.md) – detailed description of the
  core abstractions and extension points.

## Legacy Scripts

Original notebooks and helper scripts remain under their previous paths
(`time_to_explain/explainer`, `time_to_explain/models/train_tgnn.py`, etc.) and
can coexist with the new framework if required for reproduction.

# Temporal Graph XAI Framework Architecture

This repository now exposes a uniform pipeline for training, explaining, and
benchmarking temporal graph neural networks (TGNNs).  The goal is to make it
straightforward to orchestrate experiments that compare multiple explainers on
identical datasets, under shared metrics and sampling strategies.

## Package Layout

```
time_to_explain/
├── cli.py                # Entry point that launches experiments from YAML configs
├── core/                 # Registries, factories, experiment runner, shared types
├── datasets/             # Dataset builders registered with the core registry
├── explainers/           # Explainer adapters that normalise 3rd-party outputs
├── metrics/              # Metric adapters (e.g., fidelity-)
├── models/               # Wrapper adapters around TGNN backbones
├── samplers.py           # Event sampling strategies
└── ...                   # Legacy scripts kept for backwards-compatibility
```

All pluggable components register themselves through
`time_to_explain.core.registries`.  Importing `time_to_explain` eagerly loads
all registries so the CLI can discover available options without manual wiring.

## Core Abstractions

| Concept            | Location                      | Responsibility |
|--------------------|-------------------------------|----------------|
| `BaseTemporalGNN`  | `models/base.py`              | Normalised interface for TGNN backbones (`predict_event`, `compute_edge_probabilities`, `compute_edge_probabilities_for_subgraph`, …). Implemented by `WrapperTemporalGNN`, which adapts the existing TTGN/TGN/TGAT wrappers. |
| `BaseExplainer`    | `explainers/base.py`          | Provides lifecycle hooks (`setup`, `fit`, `before_event`, `explain`, `after_event`). Concrete adapters convert third-party outputs into the canonical `ExplanationResult` dataclass. |
| `BaseMetric`       | `metrics/base.py`             | Computes `MetricResult` objects from an `ExplanationContext`. Metrics may return a single result or a list (e.g., fidelity@k for several k). |
| Registries         | `core/registries.py`          | Lightweight registries (`DATASETS`, `MODELS`, `EXPLAINERS`, `METRICS`, `SAMPLERS`) with decorator helpers (`@register_model`, …). |
| Experiment runner  | `core/runner.py`              | Orchestrates sampling events, invoking explainers, collecting metrics, and persisting outputs. |

### Explanation Objects

The new adapters emit `ExplanationResult` instances.  Each result contains a
`primary` `Explanation` (edge IDs, optional importance scores, metadata), an
optional list of candidate explanations, timing/statistics dictionaries, and a
reference to the raw backend output for reproducibility.

### Metrics

Metrics receive both the `ExplanationContext` (model, dataset, event id) and the
`ExplanationResult`.  The provided `FidelityMinusMetric` reproduces fidelity- by
recomputing predictions after dropping the top-k explanation edges.  Additional
metrics can be registered by inheriting from `BaseMetric` and returning one or
more `MetricResult` objects.

## Execution Flow

1. `time_to_explain/cli.py` reads a YAML configuration.
2. Factories in `core/factories.py` instantiate the dataset, model, explainers,
   metrics, and optional event sampler.
3. `ExperimentRunner` samples event IDs (from `experiment.event_ids` or via a
   registered sampler), sets the model to evaluation mode, and iterates over
   `(event, explainer)` pairs.
4. Explain adapters translate backend outputs into canonical structures.
5. Metric adapters evaluate explanations and store results in a flat table.
6. Results are written to CSV/Parquet/JSON under `experiment.output_dir`.

## Extending the Framework

1. **Datasets** – Create a builder that returns a `ContinuousTimeDynamicGraphDataset`
   and decorate it with `@register_dataset("my_dataset")`.
2. **Models** – Implement/adapt a `BaseTemporalGNN`, expose a factory decorated
   with `@register_model` that receives `(config, dataset)` and returns an
   instance.
3. **Explainers** – Subclass `BaseExplainer`, implement `setup` and `explain`,
   and register a factory that wires configuration to your class.
4. **Metrics** – Subclass `BaseMetric`, override `compute`, and register a
   builder with `@register_metric`.
5. **Samplers** – Return a callable that yields event IDs and register it via
   `@register_sampler`.

Ensure new components live inside `time_to_explain/<domain>` so they are
available when `time_to_explain` is imported.

### Training Utilities

- `time_to_explain/models/training.py` provides
  `train_model_from_config`, producing explainer-compatible checkpoints when
  combined with the `explainer` model builder.
- `time_to_explain/explainer/train_pgexplainer.py` exposes
  `train_pgexplainer_from_config`, which consumes the same registry-based
  configuration schema for end-to-end PGExplainer training.

## Compatibility

Legacy scripts (e.g., notebook-driven pipelines) remain under their previous
paths.  The new framework wraps the same underlying TGNN wrappers and explainers
so previously trained checkpoints continue to work.

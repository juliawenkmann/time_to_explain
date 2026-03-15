# II_instability_sources

Clean workspace for instability-source analysis and figure generation.

## Structure

- `src/instability_sources/`
  - Core logic for synthetic instability cases and figure generation.
- `notebooks/`
  - Experiment/inspection notebooks.
- `figures/`
  - Generated PDF artifacts plus the GRU hidden-trajectory GIF.
- `generate_instability_figures.py`
  - Thin entrypoint script that runs the full figure build.

## Run

From repo root:

```bash
python II_instability_sources/generate_instability_figures.py
```

This recreates the same figure set under `II_instability_sources/figures/`.
It now also exports `01c_gru_hidden_trajectory.gif`, which starts at the divergence event and shows the post-removal split of the two hidden-state paths.

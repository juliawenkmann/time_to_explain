Sanity checks overview

General sanity checks (all explainers)
- Script: `time_to_explain/test/run_explainer_sanity_checks.py`
- Uses the same pipeline as `notebooks/04_compare_explainers.ipynb` to load data, build the model, and build explainers.
- Checks per explainer:
  - candidates exist (from the extractor payload or explainer extras)
  - importance vector is aligned to candidates (or alignment is possible)
  - candidate id consistency (explainer ids must be a subset of payload ids; warns on missing ids, size mismatch, or order mismatch)
  - importance values are finite (no NaN/Inf)
  - `predict_proba` and `predict_proba_with_mask` return finite outputs for a top-k mask
  - model-dependence check: at least one anchor shows `|pred_full - pred_keep| > tolerance`
  - monotonicity-ish check: `|pred_full - pred_keep|` should generally decrease as top-k increases
- Model/wrapper wiring checks:
  - `backbone.get_prob(...)` matches `TGNNWrapper.predict(...)` on the same event (within tolerance)
  - `TGNNWrapper.compute_edge_probabilities_for_subgraph(..., edges_to_drop=[])` matches the full prediction
- Cross-explainer check:
  - warns if two explainers produce identical top-k selections on every anchor (default top-k = 10)

CoDy-specific sanity checks
- Counterfactual flip: removing CoDy's explanation set flips the logit or label (warns if not).
- Candidate set overlap: compares CoDy sampler subgraph ids against extractor `candidate_eidx` (warns on empty or low Jaccard overlap).

TempME-specific sanity checks
- Detailed checklist: `time_to_explain/test/tempme_sanity_checks.md`
- Helper functions: `time_to_explain/test/tempme_sanity_checks.py`
- Covers motif sampling invariants, event anonymization `h(e)`, motif class codes, IB term sanity, and TGN wiring tests.

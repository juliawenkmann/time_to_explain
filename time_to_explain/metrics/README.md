# Metric Reference — `time_to_explain`

This document describes the fidelity-style metrics and **ACC-AUC** used to evaluate GNN explanations by perturbing graphs via edge masks and comparing the model’s behavior with/without those perturbations.

---

## Common Assumptions & APIs

- **Explanations**
  - `result.importance_edges`: importance scores for candidate edges (aligned with `context.subgraph.payload["candidate_eidx"]`).
  - Higher score ⇒ “more important”.

- **Model APIs**
  - `predict_proba(subgraph, target)` → probabilities **or** logits (see `result_as_logit`).
  - `predict_proba_with_mask(subgraph, target, edge_mask)` where `edge_mask` is a list/array aligned to candidate edges with **1 = keep** and **0 = drop**.

- **Scoring helpers (internal)**
  - If `result_as_logit=True`, vectors are softmaxed and scalars are sigmoided before extracting the **max class probability**; otherwise values are treated as probabilities.

- **Rounding / Discreteness**
  - When converting sparsity fractions to counts, values are rounded (`round`) and clamped to `[0, |E|]`. If a positive sparsity would round to 0 while edges exist, at least 1 edge is selected.

---

## Metrics

### 1) `fidelity_drop` (alias: `fidelity_minus`)
**Question it answers:** *Do the edges marked as “important” actually matter?*  
If we **drop** the top-ranked edges, how much does the model’s confidence change?

**Definition (per k or sparsity s):**
1. Rank edges by importance.
2. Build a mask with **0 at top-k (or top-s fraction)** and **1 elsewhere**.
3. Compute  
   \[
   \text{fidelity\_drop}@k/s \;=\; \big| f(G) \;-\; f(G_{\text{drop-top}}) \big|
   \]
   where \( f(\cdot) \) is the (max) class probability after optional logit→prob conversion.

**Interpretation:** Larger is better (bigger drop ⇒ explainer identified truly critical edges).  
**Direction:** `HIGHER_IS_BETTER`.

**Key config:**
- `sparsity_levels: float | list[float]` in \([0,1]\) (fraction selected for ranking, **dropped** here).
- `topk: int | list[int]` alternative to fractions.
- `result_as_logit: bool` (default `True`), `normalize: "minmax"|"none"`, `by: "value"|"abs"`.

---

### 2) `fidelity_keep`
**Question it answers:** *Are the top-ranked edges sufficient to preserve the model’s confidence?*  
If we **keep only** the top-ranked edges (drop everything else), how much does the confidence change?

**Definition (per k or sparsity s):**
1. Rank edges by importance.
2. Build a mask with **1 at top-k (or top-s)** and **0 elsewhere**.
3. Compute  
   \[
   \text{fidelity\_keep}@k/s \;=\; \big| f(G) \;-\; f(G_{\text{keep-top}}) \big|
   \]

**Interpretation:** Numerically larger = greater deviation from original; in a *sufficiency* reading you might prefer “smaller is better”, but this module standardizes on absolute differences with `HIGHER_IS_BETTER` for consistency across fidelity variants.  
**Direction:** `HIGHER_IS_BETTER`.

**Key config:** same as `fidelity_drop`, but mask mode is **keep-top**.

---

### 3) `fidelity_best`
**Question it answers:** *What is the best (peak) fidelity across a sweep of k/sparsity values?*

**Definition:**
- Runs either `fidelity_drop` or `fidelity_keep` across all configured points.
- Reports all per-point values **plus**:
  - `best` = maximum absolute difference,
  - `best.k` = the evaluation point (k or s) that achieved it.

**Interpretation:** Summarizes peak sensitivity/sufficiency across the grid.  
**Direction:** `HIGHER_IS_BETTER`.

**Key config:** same grid options as the base fidelity metric; `mode: "drop"|"keep"` (default `"drop"`).

---

### 4) `fidelity_tempme`
**Question it answers:** *Does the explanation subset align with the model’s predicted-label dynamics under masking?*

**Setup:**
- For each sparsity \( s \) interpreted as **fraction kept**, keep the **top-s** edges, mask the rest.
- Let \( f(G) \) be the full score for the predicted class; \( f(G_s) \) the score under the mask.
- Let \( Y_f \in \{0,1\} \) be a binary label derived from context or from \( f(G) \) via a threshold.

**Definition (TEMP-ME at level s):**
\[
\text{TEMP-ME}(s) \;=\; \mathbb{1}[Y_f=1]\cdot \big(f(G_s)-f(G)\big)
\;+\;
\mathbb{1}[Y_f=0]\cdot \big(f(G)-f(G_s)\big)
\]

**Interpretation:** Higher values indicate that the kept subset supports the model’s predicted label’s confidence dynamics.  
**Direction:** `HIGHER_IS_BETTER`.

**Key config:**
- `sparsity_levels` (defaults to `[0.05, 0.1, 0.2, 0.3, 0.4, 0.5]`),
- `result_as_logit`, `normalize`, `by`,
- `label_threshold` for deriving binary \( Y_f \) from scores when no label is present.

---

### 5) `acc_auc`
Area under the curve of **prediction match rate** vs. sparsity (0 → 0.3 by default).
At each sparsity level, keep the top-s fraction of edges, mask the rest, and
record whether the model’s predicted label matches the unmasked prediction
(indicator 0/1). The per-level indicators are stored as `acc_auc.acc@s=...`,
and `acc_auc.auc` is the (optionally normalized) trapezoidal AUC over the grid.

**Key config:**
- `s_min`, `s_max`, `num_points`: sparsity grid (fractions in [0, 1], default 0→0.3 with 16 points).
- `result_as_logit`, `normalize`, `by`: importance ranking knobs (same as fidelity metrics).
- `label_threshold`: threshold for binarizing scalar outputs when deriving labels.
- `normalize_auc`: if True (default) divides AUC by span to return [0, 1].

**Outputs:** `acc_auc.acc@s=...` match indicators (floats 0/1) and `acc_auc.auc`
the AUC of the match curve.


### 6) `cohesiveness`

**Question it answers:** *Are the edges selected by the explanation topologically and temporally “close” to each other?*

**Definition (given an explanation edge set \(G_e^{\text{exp}}\)):**
\[
\text{Cohesiveness}
= \frac{1}{|G_e^{\text{exp}}|^2 - |G_e^{\text{exp}}|}
\sum_{e_i \in G_e^{\text{exp}}}
\sum_{\substack{e_j \in G_e^{\text{exp}} \\ j \ne i}}
\cos\!\left(\frac{|t_i - t_j|}{\Delta T}\right)\,\mathbf{1}(e_i \sim e_j),
\]
where \(t_i\) is the timestamp of edge \(e_i\), \(\Delta T\) is a time-scale normalizer, and
\(\mathbf{1}(e_i \sim e_j)\) is 1 if \(e_i\) and \(e_j\) **share at least one endpoint** (undirected adjacency), otherwise 0.

**How we evaluate it:**  
For each sparsity level \(s\) (keep the top-\(s\) fraction by importance), form \(G_e^{\text{exp}}(s)\) and compute the above with:
- Ordered pairs \((i, j)\) with \(i \ne j\) — hence the denominator \(m^2 - m\), where \(m = |G_e^{\text{exp}}(s)|\).
- \(\Delta T\): by default, the range of candidate edge times \(\max t - \min t\); you may fix it via config.

**Interpretation:**  
- Values in \([0, 1]\) (with this \(\cos\) choice and \(|t_i - t_j|/\Delta T \in [0, 1]\)).  
- Higher is better: edges are both **adjacent** in the graph and **close in time**.


**Notes:**
- If fewer than 2 edges are kept (\(m < 2\)), the metric is undefined; we return `NaN`.
- We normalize by **all ordered pairs** \(m^2 - m\); non-adjacent pairs contribute 0 via the indicator.

---

### 7) `prediction_profile`

**Question it answers:** *How do the model’s predictions evolve as we vary the sparsity of the explanation subgraph?*

**Definition:**
1. Rank candidate edges by their importance scores (same normalization knobs as the fidelity metrics).
2. For each configured sparsity level or k-value, build a mask:
   - `mode="keep"` (default): keep only the top edges, drop everything else.
   - `mode="drop"`: drop the top edges, keep the remainder.
3. Evaluate the model under each mask (plus the unmasked subgraph) via `predict_proba_with_mask`.

**Outputs (per level):**
- `prediction_profile.prediction_full`: score on the unmasked subgraph.
- `prediction_profile.prediction_keep.@s=0.2` (or `.prediction_drop`): score under the masked subgraph.
- `prediction_profile.delta_keep.@s=0.2`: signed change `masked - full` (configurable via `include_delta`).
- `prediction_profile.delta_abs_keep.@s=0.2`: absolute change (if `include_abs_delta=True`).
- `prediction_profile.label_full` / `prediction_profile.label_keep.@s=...`: predicted class (argmax for multi-class, thresholded for scalar outputs).
- `prediction_profile.match_keep.@s=...`: indicator (0/1) for whether the masked prediction agrees with the original (helpful for plotting “unchanged prediction” percentages).

**Extras:** Recorded under `metric_details → prediction_profile` in the runner’s JSONL output.
- Each level stores the achieved sparsity, prediction, deltas, and optionally the actual edge mask and edge list of the subgraph that was fed to the model.
- Use `store_edge_details=False` or `edge_detail_limit=N` in the metric config if you want to skip (or cap) the per-level edge payload to keep run artifacts compact.

**Key config:**
- `sparsity_levels` **or** `topk/k`: evaluation grid (fractions in [0,1] or absolute counts).
- `mode`: `"keep"` or `"drop"`.
- `result_as_logit`, `normalize`, `by`: forwarded to the ranking helper.
- `include_delta`, `include_abs_delta`: toggle the delta columns.
- `label_threshold`: scalar threshold used to binarize scalar outputs when deriving prediction labels (default `0.5`).
- `emit_match_flags`: set `False` to skip the `match_*` columns when you only need the raw predictions.
- `store_edge_details`, `store_masks`, `edge_detail_limit`: control how much subgraph metadata gets attached per sparsity level.

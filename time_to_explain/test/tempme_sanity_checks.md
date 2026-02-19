Here are sanity checks I'd run in roughly this order, from "fastest smoke test" -> "proves the wiring is correct" -> "behavior looks like TempME should".

I'll reference TempME's key invariants (motif sampling, event anonymization, IB term, fidelity/sparsity) as the ground truth.

---

## 1) Smoke test: can you generate any explanation without touching TGN?

Goal: verify your motif sampling -> motif encoding -> generator -> selection -> union-of-edges path runs end-to-end.

Run `explainer.explain(...)` using the standalone `AdjacencyListNeighborFinder` over a small slice of data.

Checks

* `len(exp.motifs) > 0` for nodes with history
* `exp.motif_scores.shape == [len(exp.motifs)]`
* `exp.selected_motif_mask.sum() == top_k` when `selection_mode="topk"` (unless fewer motifs exist)
* `len(exp.edges) > 0` unless there is truly no history before `t`

If this fails, it's almost always:

* wrong `n_nodes` passed to the neighbor finder,
* `edge_idx` out of range for `edge_raw_features`,
* timestamps not in the same scale / not strictly increasing in the data stream.

---

## 2) Motif sampling invariants (Algorithm 1)

TempME's sampler is constrained: events must be strictly earlier as you go "back in time" and motifs must remain connected (because each sampled event touches the current node set).

For each sampled motif `I` around anchor `(u0, t0)`, assert:

### Required invariants

* Strict reverse time order:
  `t0 > t1 > t2 > ... > tl` (strict, not >=)
* All events occur before t0: every event has `e.ts < t0`
* Length: `len(motif.events) == motif_len` (unless you enable `allow_shorter=True`)
* Node cap: number of distinct nodes in motif <= `motif_max_nodes`
* Connectivity: the induced subgraph is connected

### Minimal assertion helper you can paste in

```python
def assert_motif_ok(motif, u0, t0, l, n):
    assert len(motif.events) == l
    ts = [e.ts for e in motif.events]
    assert all(t < t0 for t in ts)
    assert all(ts[i] > ts[i+1] for i in range(len(ts)-1)), ts

    nodes = set()
    adj = {}
    for e in motif.events:
        nodes.add(e.src); nodes.add(e.dst)
        adj.setdefault(e.src, set()).add(e.dst)
        adj.setdefault(e.dst, set()).add(e.src)

    assert len(nodes) <= n

    # connectivity check
    start = next(iter(nodes))
    seen = {start}
    stack = [start]
    while stack:
        x = stack.pop()
        for y in adj.get(x, []):
            if y not in seen:
                seen.add(y); stack.append(y)
    assert seen == nodes, (seen, nodes)
```

If you see failures here, you've either:

* built the neighbor finder incorrectly (timestamps not sorted, or `find_before` returns >= cut time),
* or you're explaining at a timestamp earlier than any node history (so sampler can't fill `l` events).

---

## 3) Event anonymization `h(e)` matches Eq. (2)

TempME's structural feature `h(e)` counts "how often this node-pair appears at each sampled position across all motifs for the query".

Sanity check

* Pick one query `(u,v,t)`.
* Get the motifs `M = Mu U Mv`.
* Compute `h_by_pair = event_anonymization_h(M, l)`.
* Brute force the same count and compare.

```python
from collections import defaultdict
import numpy as np

def brute_h(motifs, l):
    d = defaultdict(lambda: np.zeros((l,), dtype=np.float32))
    for mot in motifs:
        for j, e in enumerate(mot.events[:l]):
            pair = tuple(sorted((e.src, e.dst)))
            d[pair][j] += 1.0
    return d

h1 = event_anonymization_h(motifs, num_events=l)
h2 = brute_h(motifs, l)

for k in h2:
    assert np.allclose(h1[k], h2[k]), (k, h1[k], h2[k])
```

If this fails, your motifs are coming out in an unexpected order (e.g., oldest->newest rather than newest->oldest) or you're accidentally mixing different queries' motifs when computing `h`.

---

## 4) Motif "equivalence class code" is stable and non-degenerate

TempME groups motifs by equivalence class using a digit-string representation (Appendix B).

Checks

* For a given query, `class_ids` should not be all identical unless the neighborhood is trivial.
* Across many queries, histogram should show multiple classes.

Quick check:

```python
codes = [motif_code(m, anchor=a) for m,a in zip(motifs, anchors)]
print("unique codes:", len(set(codes)), " / ", len(codes))
```

If you always get 1 unique code, suspect:

* `motif_max_nodes` too small,
* neighbor finder only returns one neighbor repeatedly,
* or your data has many duplicates at identical timestamps and the strict `< cut_time` prevents variety (this is a known limitation discussed in the paper's appendix).

---

## 5) IB term sanity: "should be ~0 in the matching case" (Eq. 6)

The KL-style IB term (Eq. 6) should be near zero when:

* the mean selection probability `s` matches `p_prior`, and
* the class distribution `q_i` matches the null model `m_i`.

Concrete check:

1. Build any list of `class_ids`.
2. Set `p = p_prior * ones`.
3. Set `null_m` equal to the class histogram implied by `class_ids` (or just set it equal to the computed `q`).

Then `ib_kl_term(...)` should return something very close to `0` (numerical eps aside), and never NaN/Inf.

Also check that `ib >= -1e-6` (KL should be non-negative; tiny negatives can happen from float error).

If IB goes NaN: you likely have zero probabilities in `null_m` for a class that appears in `class_ids` (fix by smoothing, which the provided estimator does).

---

## 6) The single most important wiring test with official TGN

### Identity test: filtered neighbor finder with "allow all edges" must equal baseline

This verifies that:

* `patch_neighbor_finder(...)` actually affects the base model,
* `tgn_predict_proba(...)` calls the right API,
* and the neighbor-finder plumbing is correct.

Procedure

1. Pick one real query `(u, v, t, edge_idx)` from your event stream.
2. Compute:

   * `p_full = tgn_predict_proba(model, u, v, t, edge_idx, n_neighbors=...)`
   * `p_all = explainer._base_model_proba_with_allowed_edges(... allowed_edge_idxs=ALL_EDGE_IDXS ...)`
3. Assert `abs(p_full - p_all) < 1e-6` (or a small tolerance).

Example:

```python
all_edge_idxs = set(map(int, data.edge_idxs))  # or set(range(len(edge_raw_features)))

p_full = tgn_predict_proba(tgn_model, u, v, t, edge_idx, n_neighbors=20, device=device)
p_all  = explainer._base_model_proba_with_allowed_edges(
    tgn_model, all_edge_idxs, u, v, t, edge_idx, n_neighbors=20
)

print("diff:", float((p_full - p_all).abs()))
assert float((p_full - p_all).abs()) < 1e-6
```

If this fails:

* you're not patching the actual neighbor finder used inside the embedding module,
* or the model uses another internal pointer that wasn't patched (rare fork behavior),
* or the neighbor finder APIs differ from what `tgn_predict_proba` expects.

### Restoration test: neighbor finder must be restored after patch

```python
orig = tgn_model.neighbor_finder
with patch_neighbor_finder(tgn_model, some_filtered_nf):
    assert tgn_model.neighbor_finder is some_filtered_nf
assert tgn_model.neighbor_finder is orig
```

---

## 7) Filtering test: "allow none" must change the result (usually)

Set `allowed_edge_idxs = set()` and compare `p_none` to `p_full`.

You expect `p_none` != `p_full`. If they're identical across many queries, either:

* patching isn't taking effect, or
* your TGN is dominated by its memory state / raw features and does not rely much on neighbor sampling for those queries.

So use this as a diagnostic, not a hard assertion.

---

## 8) Monotonicity-ish check: "more allowed edges -> closer to full-graph prediction"

This is a really good end-to-end sanity check that doesn't require the explainer to be trained yet.

Procedure

* For one query, build explanations with increasing `top_k` (e.g., 1, 2, 5, 10).
* For each, compute `p_k = f(G_exp_k)[e]` using the filtered neighbor finder.
* Track `|p_k - p_full|`.

Expected trend: `|p_k - p_full|` generally decreases as `top_k` increases (not strictly monotonic if neighbor sampling is stochastic).

To reduce noise, set the base neighbor finder to deterministic mode (if possible) or set seeds.

---

## 9) Fidelity / sparsity checks match the paper's definitions

TempME's evaluation sanity checks are:

* Sparsity: explanation is small relative to computational graph
* Fidelity: explanation preserves the base model's decision direction

Even if you don't reproduce the paper's numbers, you should see:

* sparsity is small when you use small `top_k`
* fidelity improves when `top_k` increases

Practical quick metric:

* compute `p_full`
* compute `p_exp`
* look at `abs(p_exp - p_full)` or BCE against the teacher label

If fidelity doesn't improve at all as you increase top_k, suspect the filtered evaluation is not actually restricting anything.

---

## 10) Training sanity checks (REINFORCE version)

Because training is stochastic/noisy, don't expect a perfectly smooth loss curve. But these should hold:

Must-haves

* `loss` is finite (no NaNs)
* gradients are non-zero on explainer params (at least sometimes)
* the running baseline value changes over time

After a short run (e.g., few thousand events)

* average BCE(student_prob, teacher_label) should trend downward in expectation
* the mean selected motifs count should not collapse to always-1 or always-all (unless `p_prior` forces it)

Log these per N steps:

* `teacher_prob`, `student_prob`
* `ib` value
* `p.mean()`, `p.min()`, `p.max()`
* `selected_motif_mask.sum()`
* `len(allowed_edges)` and `len(exp.edges)`

If training collapses:

* check `p_prior` (too small can force near-zero selection),
* check null model coverage (classes missing -> IB pushes weirdly),
* check that you call `train_step` at the correct point where TGN memory is consistent for `(u,v,t)`.

---

## Common "wired wrong" symptoms and what they usually mean

* `IndexError` on `edge_raw_features[edge_idx]`
  Your `edge_idx` values aren't aligned with the edge feature matrix (need a remap).

* Sampler returns 0 motifs for almost every query
  Wrong timestamps (e.g., passing an index instead of time), or neighbor finder missing history.

* `p_full` != `p_all` in the identity test
  Patch isn't hitting the right neighbor finder reference in the model.

* Everything deterministic even with sampling mode
  You're not actually using the stochastic path (`selection_mode="sample"`), or RNG is fixed globally.

---

If you want, paste the output of these three numbers for a single event:

* `diff_full_vs_all`
* `diff_full_vs_none`
* `|p_k - p_full|` for k in {1,2,5,10}

and I can tell you quickly whether the integration is correct or where it's likely leaking.

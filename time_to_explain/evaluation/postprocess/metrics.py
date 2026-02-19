"""Metric computation for explanation evaluation.

This module computes:

- fidelity_plus (fid+): does the predicted *label* change when removing the
  explanation from the full candidate graph?
- fidelity_minus (fid-): does the predicted *label* stay the same when keeping
  *only* the explanation events?
- AUFSC (area under the (cumulative) fidelity-sparsity curve).

It operates on scalar prediction scores (logits by default). No model inference
happens here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np


@dataclass(frozen=True)
class DecisionRule:
    """How to convert a scalar score into a binary predicted label."""

    score_is_probability: bool = False
    threshold: float = 0.0

    def label(self, score: float) -> int:
        if self.score_is_probability:
            return int(score >= self.threshold)
        # logits: 0 corresponds to p=0.5 for sigmoid
        return int(score >= self.threshold)


def fidelity_plus(original_score: float, counterfactual_score: float, rule: DecisionRule) -> int:
    """1 iff label(original) != label(counterfactual)."""
    return int(rule.label(original_score) != rule.label(counterfactual_score))


def fidelity_minus(original_score: float, explanation_only_score: float, rule: DecisionRule) -> int:
    """1 iff label(original) == label(explanation_only)."""
    return int(rule.label(original_score) == rule.label(explanation_only_score))


def compute_aufsc(
    sparsity: np.ndarray,
    fidelity: np.ndarray,
    max_sparsity: float = 1.0,
    n_grid: int = 101,
) -> float:
    """Compute AUFSC using a cumulative mean fidelity curve.

    We build a curve y(t) for t in [0, max_sparsity] where

        y(t) = mean(fidelity_i for which sparsity_i <= t)

    and then integrate y(t) over t.

    Notes
    -----
    - This follows the common "achieved up to a sparsity limit" interpretation
      used in counterfactual explanation papers.
    - If no examples fall under a threshold t, y(t) is defined as 0.
    """
    if sparsity.size == 0:
        return 0.0

    s = np.asarray(sparsity, dtype=float)
    f = np.asarray(fidelity, dtype=float)

    # Guard: clip to [0, max_sparsity] so mis-computed sparsities don't blow up.
    max_sparsity = float(max_sparsity)
    s = np.clip(s, 0.0, max_sparsity)

    grid = np.linspace(0.0, max_sparsity, int(n_grid))
    y = np.zeros_like(grid)

    # Sort once for efficient prefix means.
    order = np.argsort(s)
    s_sorted = s[order]
    f_sorted = f[order]

    # prefix sums for mean queries
    prefix_sum = np.cumsum(f_sorted)

    idx = 0
    for i, t in enumerate(grid):
        # advance idx to include all s <= t
        while idx < s_sorted.size and s_sorted[idx] <= t:
            idx += 1
        if idx == 0:
            y[i] = 0.0
        else:
            y[i] = float(prefix_sum[idx - 1]) / float(idx)

    # Area under curve in [0, max_sparsity]
    area = float(np.trapz(y, grid))

    # Optional normalization to [0,1] if max_sparsity != 1.
    if max_sparsity > 0:
        area /= max_sparsity

    return area


def summarize_binary(values: Iterable[int | float]) -> dict:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "count": 0}
    return {"mean": float(arr.mean()), "count": int(arr.size)}

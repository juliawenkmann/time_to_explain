from __future__ import annotations
from typing import Any, Mapping, Dict, List, Optional
import numpy as np

from time_to_explain.core.registry import register_metric
from time_to_explain.core.types import ExplanationContext, ExplanationResult, MetricResult
from time_to_explain.core.metrics import BaseMetric, MetricDirection


def _as_array(x) -> Optional[np.ndarray]:
    if x is None:
        return None
    arr = np.asarray(x, dtype=float)
    return arr

def _gini(x: np.ndarray) -> float:
    n = x.size
    if n == 0:
        return float("nan")
    s = np.sort(np.abs(x))
    cs = np.cumsum(s)
    denom = cs[-1]
    if denom <= 0:
        return 0.0
    # 1 - 2 * (sum_{i=1..n} (n+1-i) * s_i) / (n * sum s_i)
    return float(1.0 - 2.0 * np.sum(cs) / (n * denom))

def _entropy(x: np.ndarray, eps: float) -> float:
    if x.size == 0:
        return float("nan")
    x = np.abs(x)
    x[x < eps] = 0.0
    tot = x.sum()
    if tot <= 0:
        return 0.0
    p = x / tot
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


class SparsityMetric(BaseMetric):
    """
    Reports sparsity diagnostics over edges/nodes/time importances:
      - zero_frac, l0, density
      - gini (skew), entropy (spread)
      - mean, std

    Config (all optional):
      eps: float = 1e-8
      components: ["edges","nodes","time"]  (any subset)
    """
    def __init__(self, config: Mapping[str, Any] | None = None):
        cfg = dict(config or {})
        super().__init__(
            name="sparsity",
            direction=MetricDirection.HIGHER_IS_BETTER,  # “more sparse” is better
            config=cfg,
        )
        self.eps: float = float(cfg.get("eps", 1e-8))
        comps = cfg.get("components", ["edges", "nodes", "time"])
        self.components: List[str] = [c for c in comps if c in ("edges", "nodes", "time")]

    def _pack(self, x: np.ndarray) -> Dict[str, float]:
        n = int(x.size)
        if n == 0:
            return {"n": 0, "zero_frac": float("nan"), "l0": 0, "density": float("nan"),
                    "gini": float("nan"), "entropy": float("nan"),
                    "mean": float("nan"), "std": float("nan")}
        zero = int(np.sum(np.abs(x) < self.eps))
        return {
            "n": n,
            "l0": zero,
            "zero_frac": zero / float(n),
            "density": 1.0 - zero / float(n),
            "gini": _gini(x),
            "entropy": _entropy(x, self.eps),
            "mean": float(np.mean(x)),
            "std": float(np.std(x)),
        }

    def compute(self, context: ExplanationContext, result: ExplanationResult) -> MetricResult:
        values: Dict[str, float] = {}
        if "edges" in self.components:
            e = _as_array(result.importance_edges)
            if e is None or e.size == 0:
                values["edges.n"] = 0
                values["edges.l0"] = 0
            else:
                for k, v in self._pack(e).items():
                    values[f"edges.{k}"] = v
        if "nodes" in self.components:
            n = _as_array(result.importance_nodes)
            if n is None or n.size == 0:
                values["nodes.n"] = 0
                values["nodes.l0"] = 0
            else:
                for k, v in self._pack(n).items():
                    values[f"nodes.{k}"] = v
        if "time" in self.components:
            t = _as_array(result.importance_time)
            if t is None or t.size == 0:
                values["time.n"] = 0
                values["time.l0"] = 0
            else:
                for k, v in self._pack(t).items():
                    values[f"time.{k}"] = v

        return MetricResult(
            name=self.name,
            values=values,
            direction=self.direction.value,
            run_id=context.run_id,
            explainer=result.explainer,
            context_fp=context.fingerprint(),
            extras={"eps": self.eps, "components": self.components},
        )


@register_metric("sparsity")
def build_sparsity(config: Mapping[str, Any] | None = None):
    """Registry factory → builds the metric object from config."""
    return SparsityMetric(config)

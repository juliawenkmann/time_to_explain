from __future__ import annotations
from typing import Any, Dict, Optional
import math

from .types import ExplanationContext, ExplanationResult, ModelProtocol

def _softmax(x):
    m = max(x)
    ex = [math.exp(v - m) for v in x]
    s = sum(ex)
    return [v/s for v in ex]

def sparsity(_: ExplanationContext, res: ExplanationResult, **__) -> Dict[str, float]:
    out: Dict[str, float] = {}
    eps = 1e-8
    if res.importance_edges:
        zeros = sum(1 for v in res.importance_edges if abs(v) < eps)
        out["sparsity_edges"] = zeros / float(len(res.importance_edges))
    if res.importance_nodes:
        zeros = sum(1 for v in res.importance_nodes if abs(v) < eps)
        out["sparsity_nodes"] = zeros / float(len(res.importance_nodes))
    return out

def fidelity(context: ExplanationContext, res: ExplanationResult, *, model: ModelProtocol,
             keep_topk: Optional[int] = None, threshold: Optional[float] = None) -> Dict[str, float]:
    assert hasattr(model, "predict_proba_with_mask"), "model.predict_proba_with_mask is required for fidelity"
    if not res.importance_edges:
        return {"fidelity_edges": float("nan")}

    imp = res.importance_edges
    mini, maxi = min(imp), max(imp)
    norm = [(v - mini)/(maxi-mini + 1e-12) for v in imp]

    if keep_topk is not None:
        idx = sorted(range(len(norm)), key=lambda i: norm[i], reverse=True)
        mask = [0.0]*len(norm)
        for i in idx[:keep_topk]: mask[i] = 1.0
    elif threshold is not None:
        mask = [1.0 if v >= threshold else 0.0 for v in norm]
    else:
        k = max(1, int(0.2*len(norm)))
        idx = sorted(range(len(norm)), key=lambda i: norm[i], reverse=True)
        mask = [0.0]*len(norm)
        for i in idx[:k]: mask[i] = 1.0

    subg = context.subgraph  # type: ignore
    p_full = model.predict_proba(subg, context.target)
    p_mask = model.predict_proba_with_mask(subg, context.target, edge_mask=mask)

    def to_scalar(p):
        if isinstance(p, (list, tuple)):
            return max(_softmax(list(p)))
        try:
            import numpy as np
            arr = np.asarray(p).reshape(-1)
            return float(arr.max())
        except Exception:
            return float(p)
    return {"fidelity_edges": to_scalar(p_full) - to_scalar(p_mask)}

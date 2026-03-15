from __future__ import annotations

from collections import defaultdict
from typing import Any, Optional

import pandas as pd
import torch

from pathpy_utils import idx_to_node_list


def ho_edge_df(exp, assets, *, edge_mask: Optional[torch.Tensor] = None) -> pd.DataFrame:
    """Return higher-order edges with human-readable De Bruijn node IDs.

    For order-2 De Bruijn graphs in `temporal_clusters`, higher-order node IDs
    are tuples (u, v).
    """

    idx_to_ho = idx_to_node_list(assets.g2)
    ei = exp.edge_index
    scores = exp.edge_score

    E = int(ei.size(1))
    if edge_mask is None:
        idx_iter = range(E)
    else:
        if edge_mask.dim() != 1 or edge_mask.numel() != E:
            raise ValueError(f"edge_mask must be 1D of shape [E]={E}, got {tuple(edge_mask.shape)}")
        m = edge_mask
        if m.dtype != torch.bool:
            m = m != 0
        idx_iter = m.nonzero(as_tuple=False).view(-1).tolist()

    rows = []
    for e in idx_iter:
        src = idx_to_ho[int(ei[0, e])]
        dst = idx_to_ho[int(ei[1, e])]
        rows.append({"src_ho": src, "dst_ho": dst, "score": float(scores[e].item())})
    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)


def ho_path_df(exp, assets, *, edge_mask: Optional[torch.Tensor] = None) -> pd.DataFrame:
    """Interpret each higher-order edge as a first-order length-2 walk.

    For order-2 De Bruijn:
      - higher-order node = (u, v)  (a first-order edge)
      - higher-order edge ( (u,v) -> (v,w) ) = walk u->v->w
    """

    idx_to_ho = idx_to_node_list(assets.g2)
    ei = exp.edge_index
    scores = exp.edge_score

    E = int(ei.size(1))
    if edge_mask is None:
        idx_iter = range(E)
    else:
        if edge_mask.dim() != 1 or edge_mask.numel() != E:
            raise ValueError(f"edge_mask must be 1D of shape [E]={E}, got {tuple(edge_mask.shape)}")
        m = edge_mask
        if m.dtype != torch.bool:
            m = m != 0
        idx_iter = m.nonzero(as_tuple=False).view(-1).tolist()

    rows = []
    for e in idx_iter:
        a = idx_to_ho[int(ei[0, e])]
        b = idx_to_ho[int(ei[1, e])]

        # Most commonly: a=(u,v), b=(v,w)
        u, v = a
        v2, w = b
        rows.append(
            {
                "u": u,
                "v": v,
                "w": w,
                "src_ho": a,
                "dst_ho": b,
                "consistent": bool(v == v2),
                "score": float(scores[e].item()),
            }
        )

    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)


def project_ho_edges_to_first_order_edges(
    exp,
    assets,
    *,
    agg: str = "sum",
    edge_mask: Optional[torch.Tensor] = None,
) -> pd.DataFrame:
    """Project higher-order edge importance down to first-order edge importance.

    Each higher-order edge corresponds to a first-order walk u->v->w.
    We distribute its score to the two first-order edges (u,v) and (v,w).

    Args:
        exp: EdgeExplanation in higher-order explain-space.
        assets: TemporalClustersAssets (must include g2).
        agg: how to aggregate multiple contributions to the same first-order edge.
             One of {"sum", "mean", "max"}.

    Returns:
        DataFrame with columns: src, dst, score
    """

    if agg not in {"sum", "mean", "max"}:
        raise ValueError("agg must be one of {'sum','mean','max'}")

    idx_to_ho = idx_to_node_list(assets.g2)
    ei = exp.edge_index
    scores = exp.edge_score

    # Collect multiple contributions per first-order edge
    contribs: dict[tuple[Any, Any], list[float]] = defaultdict(list)

    E = int(ei.size(1))
    if edge_mask is None:
        idx_iter = range(E)
    else:
        if edge_mask.dim() != 1 or edge_mask.numel() != E:
            raise ValueError(f"edge_mask must be 1D of shape [E]={E}, got {tuple(edge_mask.shape)}")
        m = edge_mask
        if m.dtype != torch.bool:
            m = m != 0
        idx_iter = m.nonzero(as_tuple=False).view(-1).tolist()

    for e in idx_iter:
        a = idx_to_ho[int(ei[0, e])]
        b = idx_to_ho[int(ei[1, e])]
        u, v = a
        v2, w = b
        s = float(scores[e].item())

        contribs[(u, v)].append(s)
        contribs[(v2, w)].append(s)

    rows = []
    for (src, dst), vals in contribs.items():
        if agg == "sum":
            score = float(sum(vals))
        elif agg == "mean":
            score = float(sum(vals) / max(1, len(vals)))
        else:  # max
            score = float(max(vals))
        rows.append({"src": src, "dst": dst, "score": score, "n_contrib": len(vals)})

    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)

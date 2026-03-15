from __future__ import annotations

"""Utilities for higher-order (order-2) De Bruijn graphs.

Several components (candidate-set selection, benchmarks, plotting helpers)
benefit from having causal *triples* aligned with the higher-order edge list.

For an order-2 De Bruijn graph:

  - A higher-order node is typically a pair ``(u, v)`` representing a first-
    order edge.
  - A higher-order edge ``(u, v) -> (v, w)`` represents the first-order walk
    ``u -> v -> w``.

We encode this as a tensor ``ho_triples[e] = (u_idx, v_idx, w_idx)`` for each
higher-order edge ``e``.
"""

from typing import Optional

import torch

from data.pathpy_utils import idx_to_node_list


def _edge_index_tensor(ei) -> torch.Tensor:
    """Return edge_index as a plain torch.Tensor with shape [2, E]."""
    if hasattr(ei, "as_tensor"):
        return ei.as_tensor()
    if isinstance(ei, torch.Tensor):
        return ei
    return torch.as_tensor(ei)


def attach_ho_triples_to_data(
    data,
    *,
    g,
    g2,
    edge_index_attr: str = "edge_index_higher_order",
    triples_attr: str = "ho_triples",
    device: Optional[torch.device] = None,
) -> Optional[torch.Tensor]:
    """Attach a tensor of order-2 causal triples to a PyG Data object.

    The function is intentionally "best effort": if required attributes are
    missing or node IDs cannot be mapped to integer indices, it returns None
    and leaves the Data object unchanged.

    Args:
        data: torch_geometric-style Data.
        g: first-order PathpyG graph (must expose ``mapping.to_idx``).
        g2: order-2 PathpyG graph.
        edge_index_attr: name of the higher-order edge_index attribute.
        triples_attr: attribute name to attach.
        device: optional device for the created tensor.

    Returns:
        The created tensor of shape [E, 3] (dtype long) or None.
    """

    if hasattr(data, triples_attr):
        t = getattr(data, triples_attr)
        if isinstance(t, torch.Tensor) and t.ndim == 2 and t.size(-1) == 3:
            return t

    if g is None or g2 is None:
        return None
    if not hasattr(data, edge_index_attr):
        return None

    ei2 = _edge_index_tensor(getattr(data, edge_index_attr))
    if ei2.ndim != 2 or ei2.size(0) != 2:
        return None
    E2 = int(ei2.size(1))
    if E2 == 0:
        return None

    idx_to_ho = idx_to_node_list(g2)

    triples = []
    for e in range(E2):
        src_ho = idx_to_ho[int(ei2[0, e])]
        dst_ho = idx_to_ho[int(ei2[1, e])]

        # Most commonly for order-2 De Bruijn:
        #   src_ho = (u, v)
        #   dst_ho = (v, w)
        # Robust fallback: treat node IDs as sequences and take the last two.
        if isinstance(src_ho, (list, tuple)) and len(src_ho) >= 2:
            u_id = src_ho[-2]
            v_id = src_ho[-1]
        else:
            # Cannot interpret
            return None

        if isinstance(dst_ho, (list, tuple)) and len(dst_ho) >= 1:
            w_id = dst_ho[-1]
        else:
            return None

        # Map node IDs to *indices* used by the PyG Data object.
        try:
            u = int(g.mapping.to_idx(u_id))
            v = int(g.mapping.to_idx(v_id))
            w = int(g.mapping.to_idx(w_id))
        except Exception:
            # Fallback: assume ids are already indices.
            try:
                u, v, w = int(u_id), int(v_id), int(w_id)
            except Exception:
                return None

        triples.append((u, v, w))

    out = torch.tensor(triples, dtype=torch.long, device=device or ei2.device)
    setattr(data, triples_attr, out)
    return out

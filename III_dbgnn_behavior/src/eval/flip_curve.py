from __future__ import annotations

from typing import Optional

import pandas as pd
import torch

from eval.metrics import ranked_edge_indices, target_prob


@torch.no_grad()
def compute_flip_curve(
    *,
    adapter,
    data,
    node_idx: int,
    scores: torch.Tensor,
    candidate_mask: Optional[torch.Tensor] = None,
    orig_class: Optional[int] = None,
    max_k: Optional[int] = 200,
    step: int = 1,
    descending: bool = True,
    drop_mode: str = "remove",
) -> pd.DataFrame:
    """Compute a prediction/probability curve as edges are dropped in score order.

    This is intended as a lightweight debugging tool:
      - take an explainer's edge scores
      - rank candidate edges by score
      - drop the top-k edges (within the candidate set)
      - record the predicted class and the probability of the original class

    Notes:
      - The prediction as a function of k is **not** guaranteed to be monotone.
        The curve can flip and later flip back.

    Args:
        adapter: model adapter (must provide explain_space, clone_with_perturbed_edges, predict_logits).
        data: full graph data.
        node_idx: node to evaluate.
        scores: edge scores of shape [E] (higher = dropped earlier).
        candidate_mask: optional boolean mask of shape [E] restricting which edges may be dropped.
        orig_class: class whose probability to track; defaults to the model's baseline prediction.
        max_k: cap on the number of candidate edges dropped. If None, evaluates all candidates.
        step: evaluate every `step` deletions (>=1). The returned curve always includes k=0 and k=cap.
        descending: if True, drop highest-score edges first.
        drop_mode: "remove" to drop edges from edge_index, "zero" to zero their edge weights.

    Returns:
        DataFrame with one row per evaluated k and columns:
          - k: number of dropped edges
          - pred_class: predicted class after dropping k edges
          - p_orig: probability assigned to orig_class
          - flipped: 1 if pred_class != orig_class else 0
          - node, orig_class, candidate_count (metadata)
    """

    node_idx = int(node_idx)
    if step < 1:
        raise ValueError(f"step must be >= 1, got {step}")

    drop_mode = str(drop_mode).lower()
    if drop_mode not in {"remove", "zero"}:
        raise ValueError("drop_mode must be 'remove' or 'zero'")

    # Resolve explain-space edges from the adapter.
    space = adapter.explain_space()
    edge_index_full = getattr(data, space.edge_index_attr)
    edge_weight_full = None
    if space.edge_weight_attr is not None and hasattr(data, space.edge_weight_attr):
        edge_weight_full = getattr(data, space.edge_weight_attr)
    if drop_mode == "zero" and edge_weight_full is None:
        raise ValueError("drop_mode='zero' requires edge weights in the explain space")

    E = int(edge_index_full.size(1))
    if scores.dim() != 1 or int(scores.numel()) != E:
        raise ValueError(f"scores must be 1D of shape [E]={E}, got shape {tuple(scores.shape)}")

    ranked_idx = ranked_edge_indices(scores, candidate_mask=candidate_mask, descending=descending)
    C = int(ranked_idx.numel())
    cap = C if max_k is None else min(int(max_k), C)
    cap = max(0, cap)

    # Baseline prediction on the unperturbed graph.
    logits0 = adapter.predict_logits(data)
    if orig_class is None:
        orig_class = int(logits0[node_idx].argmax().item())
    orig_class = int(orig_class)

    records = [
        {
            "k": 0,
            "pred_class": int(logits0[node_idx].argmax().item()),
            "p_orig": float(target_prob(logits0, node_idx, orig_class)),
        }
    ]

    if cap == 0:
        df = pd.DataFrame.from_records(records)
        df["flipped"] = (df["pred_class"] != orig_class).astype(int)
        df["node"] = node_idx
        df["orig_class"] = orig_class
        df["candidate_count"] = C
        return df

    E_device = edge_index_full.device

    def _eval_k(k: int) -> tuple[int, float]:
        if drop_mode == "remove":
            keep_mask = torch.ones(E, dtype=torch.bool, device=E_device)
            if k > 0:
                keep_mask[ranked_idx[:k]] = False

            ei = edge_index_full[:, keep_mask]
            ew = edge_weight_full[keep_mask] if edge_weight_full is not None else None
            data2 = adapter.clone_with_perturbed_edges(data, ei, new_edge_weight=ew)
        else:
            ew = edge_weight_full.detach().clone()
            if k > 0:
                ew[ranked_idx[:k]] = 0.0
            data2 = adapter.clone_with_perturbed_edges(data, edge_index_full, new_edge_weight=ew)
        logits = adapter.predict_logits(data2)
        pred = int(logits[node_idx].argmax().item())
        p = float(target_prob(logits, node_idx, orig_class))
        return pred, p

    # Evaluate k in steps, always including the cap point.
    for k in range(step, cap + 1, step):
        pred, p = _eval_k(k)
        records.append({"k": int(k), "pred_class": int(pred), "p_orig": float(p)})

    if int(records[-1]["k"]) != cap:
        pred, p = _eval_k(cap)
        records.append({"k": int(cap), "pred_class": int(pred), "p_orig": float(p)})

    df = pd.DataFrame.from_records(records)
    df["flipped"] = (df["pred_class"] != orig_class).astype(int)
    df["node"] = node_idx
    df["orig_class"] = orig_class
    df["candidate_count"] = C
    return df

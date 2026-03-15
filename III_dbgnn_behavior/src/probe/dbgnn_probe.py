from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class DBGNNProbes:
    """Intermediate representations produced by a DBGNN forward pass.

    Conventions:
    - x_fo_states[0] is the *input* first-order feature matrix `data.x`
    - x_ho_states[0] is the *input* higher-order feature matrix `data.x_h`
    - Each subsequent entry is the post-activation output of one conv layer.
    """

    x_fo_states: List[torch.Tensor]
    x_ho_states: List[torch.Tensor]
    x_after_bipartite: torch.Tensor
    logits: torch.Tensor


def _maybe_edge_weight(data, name: str) -> Optional[torch.Tensor]:
    return getattr(data, name, None) if hasattr(data, name) else None


def forward_with_probes(
    model: torch.nn.Module,
    data,
    *,
    eval_mode: bool = True,
    detach: bool = True,
) -> DBGNNProbes:
    """Run a DBGNN forward pass while capturing intermediate states.

    This mirrors the reference implementation in pathpyG:
    https://www.pathpy.net/0.2.0-dev/reference/pathpyG/nn/ (DBGNN forward)

    Args:
        model: PathpyG DBGNN instance (or anything with compatible attributes).
        data: torch_geometric.data.Data with DBGNN fields.
        eval_mode: If True, runs with model.eval() (dropout disabled).
        detach: If True, stores detached tensors (safer for notebooks).

    Returns:
        DBGNNProbes with layerwise states.
    """

    was_training = bool(model.training)
    if eval_mode:
        model.eval()
    else:
        model.train()

    # ---- First-order stream ----
    x = data.x
    x_fo_states: List[torch.Tensor] = [x.detach() if detach else x]

    edge_weight = _maybe_edge_weight(data, "edge_weights")
    for layer in getattr(model, "first_order_layers"):
        x = F.dropout(x, p=float(getattr(model, "p_dropout", 0.0)), training=model.training)
        x = F.elu(layer(x, data.edge_index, edge_weight))
        x_fo_states.append(x.detach() if detach else x)

    x = F.dropout(x, p=float(getattr(model, "p_dropout", 0.0)), training=model.training)

    # ---- Higher-order stream ----
    x_h = data.x_h
    x_ho_states: List[torch.Tensor] = [x_h.detach() if detach else x_h]

    edge_weight_ho = _maybe_edge_weight(data, "edge_weights_higher_order")
    for layer in getattr(model, "higher_order_layers"):
        x_h = F.dropout(x_h, p=float(getattr(model, "p_dropout", 0.0)), training=model.training)
        x_h = F.elu(layer(x_h, data.edge_index_higher_order, edge_weight_ho))
        x_ho_states.append(x_h.detach() if detach else x_h)

    x_h = F.dropout(x_h, p=float(getattr(model, "p_dropout", 0.0)), training=model.training)

    # ---- Bipartite message passing (higher-order -> first-order) ----
    x_bi = torch.nn.functional.elu(
        model.bipartite_layer((x_h, x), data.bipartite_edge_index, n_ho=data.num_ho_nodes, n_fo=data.num_nodes)
    )
    x_bi = F.dropout(x_bi, p=float(getattr(model, "p_dropout", 0.0)), training=model.training)

    # ---- Linear classifier ----
    logits = model.lin(x_bi)

    # Restore training state
    if was_training:
        model.train()
    else:
        model.eval()

    return DBGNNProbes(
        x_fo_states=x_fo_states,
        x_ho_states=x_ho_states,
        x_after_bipartite=x_bi.detach() if detach else x_bi,
        logits=logits.detach() if detach else logits,
    )


def to_numpy(x: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    return x.detach().cpu().numpy()


def tsne_2d(
    x: Union[torch.Tensor, np.ndarray],
    *,
    seed: int = 0,
    perplexity: float = 30.0,
    learning_rate: str = "auto",
    init: str = "random",
) -> np.ndarray:
    """Deterministic-ish 2D t-SNE helper used in Phase D."""
    from sklearn.manifold import TSNE

    X = to_numpy(x).astype(np.float32, copy=False)
    if X.ndim != 2:
        raise ValueError("x must have shape [n, d]")
    if X.shape[0] < 2:
        return np.zeros((X.shape[0], 2), dtype=float)

    return TSNE(
        n_components=2,
        learning_rate=learning_rate,
        init=init,
        perplexity=float(perplexity),
        random_state=int(seed),
    ).fit_transform(X)


def silhouette_table(
    states: Sequence[Union[torch.Tensor, np.ndarray]],
    labels: np.ndarray,
    *,
    metric: str = "euclidean",
) -> pd.DataFrame:
    """Compute a silhouette score per representation (layer).

    Useful as a quick, checkable diagnostic for "are clusters separating?".
    """
    from sklearn.metrics import silhouette_score

    y = np.asarray(labels)
    rows = []
    for i, s in enumerate(states):
        X = to_numpy(s)
        score = np.nan
        if X.shape[0] >= 3 and len(np.unique(y)) >= 2:
            try:
                score = float(silhouette_score(X, y, metric=metric))
            except Exception:
                score = np.nan
        rows.append({"layer": int(i), "n": int(X.shape[0]), "dim": int(X.shape[1]), "silhouette": score})
    return pd.DataFrame(rows)

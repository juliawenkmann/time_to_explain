from __future__ import annotations

import torch


def cluster_id_temporal_clusters(node: int) -> int:
    """Ground-truth cluster id for the temporal_clusters toy dataset."""
    return int(node) // 10


def compute_stay_label_from_triples(triples: torch.Tensor) -> torch.Tensor:
    """Return a float tensor [E] where 1.0 iff cluster(u)==cluster(v)==cluster(w).

    Args:
        triples: LongTensor [E,3] with columns (u,v,w)

    Returns:
        FloatTensor [E] in {0.0, 1.0}
    """
    if not isinstance(triples, torch.Tensor):
        # Accept array-like inputs but return a tensor on CPU.
        triples = torch.as_tensor(triples)

    if triples.ndim != 2 or triples.size(1) != 3:
        raise ValueError(f"triples must have shape [E,3], got {tuple(triples.shape)}")

    u = triples[:, 0].to(torch.long)
    v = triples[:, 1].to(torch.long)
    w = triples[:, 2].to(torch.long)

    cu = u // 10
    cv = v // 10
    cw = w // 10

    stay = (cu == cv) & (cw == cv)
    return stay.to(torch.float32)


def compute_label_stay_from_triples(
    triples: torch.Tensor,
    *,
    y: torch.Tensor,
) -> torch.Tensor:
    """Return 1.0 iff y[u]==y[v]==y[w] for each triple (u,v,w).

    This is a *label-consistent* generalization of :func:`compute_stay_label_from_triples`.
    It is useful for synthetic datasets where clusters are defined by labels
    (e.g., the connected temporal cluster dataset) and keeps the same output
    contract (FloatTensor in {0,1}).

    Args:
        triples: LongTensor [E,3] with columns (u,v,w)
        y: LongTensor [N] node labels aligned with base graph node indices.

    Returns:
        FloatTensor [E] in {0.0, 1.0}
    """
    if not isinstance(triples, torch.Tensor):
        triples = torch.as_tensor(triples)
    if not isinstance(y, torch.Tensor):
        y = torch.as_tensor(y)

    if triples.ndim != 2 or triples.size(1) != 3:
        raise ValueError(f"triples must have shape [E,3], got {tuple(triples.shape)}")
    if y.ndim != 1:
        raise ValueError(f"y must be a 1D tensor [N], got {tuple(y.shape)}")

    u = triples[:, 0].to(torch.long)
    v = triples[:, 1].to(torch.long)
    w = triples[:, 2].to(torch.long)

    yu = y[u]
    yv = y[v]
    yw = y[w]

    stay = (yu == yv) & (yw == yv)
    return stay.to(torch.float32)


def attach_stay_gt_to_data(
    data,
    *,
    triples_attr: str = "ho_triples",
    score_attr: str = "gt_stay_score_higher_order",
    label_attr: str = "gt_stay_label_higher_order",
) -> None:
    """Attach cluster-stay ground truth to a PyG Data object in-place.

    Requirements:
        data.<triples_attr> exists and is a LongTensor [E,3] aligned with the
        higher-order edge list used for explanations.

    Adds:
        data.<score_attr>: FloatTensor [E] (currently equal to the label)
        data.<label_attr>: FloatTensor [E] in {0,1}

    Note:
        This GT is *deterministic* for the temporal_clusters dataset and does
        not depend on any model or target class used by explainers.
    """
    if not hasattr(data, triples_attr):
        raise AttributeError(f"Data has no attribute {triples_attr!r}. Attach ho_triples first.")

    triples = getattr(data, triples_attr)
    if not isinstance(triples, torch.Tensor):
        raise TypeError(f"data.{triples_attr} must be a torch.Tensor")

    y = compute_stay_label_from_triples(triples).to(device=triples.device)

    # Use a simple, interpretable importance score:
    #   score = 1[stay] * (higher-order transition count)
    # This gives a stable ranking among GT-positive triples.
    score = y
    if hasattr(data, "edge_weights_higher_order"):
        w = getattr(data, "edge_weights_higher_order")
        if isinstance(w, torch.Tensor) and w.numel() == y.numel():
            score = y * w.to(torch.float32).abs()

    setattr(data, label_attr, y)
    setattr(data, score_attr, score)


def attach_label_stay_gt_to_data(
    data,
    *,
    triples_attr: str = "ho_triples",
    y_attr: str = "y",
    score_attr: str = "gt_stay_score_higher_order",
    label_attr: str = "gt_stay_label_higher_order",
) -> None:
    """Attach "stay-in-same-label" ground truth for higher-order edges.

    This variant defines GT-positive triples as those where all three nodes in
    the triple share the *node label* from ``data.<y_attr>``.

    It is a good default for synthetic datasets whose classes correspond to
    clusters, and it keeps the same attributes used by :class:`StayGTOracleExplainer`.

    Requirements:
        - data.<triples_attr> : LongTensor [E,3]
        - data.<y_attr> : LongTensor [N]
    """
    if not hasattr(data, triples_attr):
        raise AttributeError(f"Data has no attribute {triples_attr!r}. Attach ho_triples first.")
    if not hasattr(data, y_attr):
        raise AttributeError(f"Data has no attribute {y_attr!r}.")

    triples = getattr(data, triples_attr)
    y = getattr(data, y_attr)
    if not isinstance(triples, torch.Tensor):
        raise TypeError(f"data.{triples_attr} must be a torch.Tensor")
    if not isinstance(y, torch.Tensor):
        raise TypeError(f"data.{y_attr} must be a torch.Tensor")

    lbl = compute_label_stay_from_triples(triples, y=y).to(device=triples.device)

    score = lbl
    if hasattr(data, "edge_weights_higher_order"):
        w = getattr(data, "edge_weights_higher_order")
        if isinstance(w, torch.Tensor) and w.numel() == lbl.numel():
            score = lbl * w.to(torch.float32).abs()

    setattr(data, label_attr, lbl)
    setattr(data, score_attr, score)

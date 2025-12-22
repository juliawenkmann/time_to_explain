from __future__ import annotations
import warnings
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd

from .base import DatasetRecipe
from time_to_explain.core.registry import register_dataset
from time_to_explain.core.types import DatasetBundle


# ----------------------- Hawkes simulator (no external deps) -----------------------

def simulate_hawkes_exp_kernels(
    baseline: np.ndarray,
    adjacency: np.ndarray,
    decays: np.ndarray,
    horizon: float,
    seed: Optional[int] = None,
    max_events: int = 1_000_000,
) -> List[np.ndarray]:
    """
    Simulate a multi-dimensional Hawkes process with exponential kernels using
    Ogata's thinning (multivariate extension).

    Intensity for target k at time t:
        λ_k(t) = μ_k + sum_l α_{l,k} * sum_{t_i^l < t} exp(-β_{l,k}(t - t_i^l))

    Parameters
    ----------
    baseline : (D,) array-like of μ_k >= 0
    adjacency : (D, D) array-like of α_{l,k} >= 0  (source l -> target k)
    decays : scalar or (D, D) array-like of β_{l,k} > 0
    horizon : float, simulate on [0, horizon]
    seed : int, optional
    max_events : hard safety cap to avoid runaway simulations

    Returns
    -------
    list[np.ndarray]
        A list of length D; timestamps[d] is a 1D float array of event times for component d.
    """
    baseline = np.asarray(baseline, dtype=float).reshape(-1)
    dim = baseline.shape[0]
    alpha = np.asarray(adjacency, dtype=float).reshape(dim, dim)

    # Broadcast decays to (dim, dim)
    if np.isscalar(decays):
        beta = np.full((dim, dim), float(decays), dtype=float)
    else:
        beta = np.asarray(decays, dtype=float).reshape(dim, -1)
        if beta.shape[1] == 1:
            beta = np.repeat(beta, dim, axis=1)
        if beta.shape != (dim, dim):
            raise ValueError(f"`decays` must broadcast to ({dim},{dim}), got {beta.shape}")

    if np.any(baseline < 0) or np.any(alpha < 0) or np.any(beta <= 0):
        raise ValueError("Require baseline>=0, adjacency>=0, decays>0.")

    # Optional stability hint (not mandatory for finite-horizon sim)
    # Stationary condition typically requires spectral radius of G = alpha / beta < 1
    with np.errstate(divide="ignore", invalid="ignore"):
        G = np.divide(alpha, beta, out=np.zeros_like(alpha), where=(beta > 0))
    try:
        spr = max(abs(np.linalg.eigvals(G)))
        if spr >= 1.0:
            warnings.warn(
                f"[HawkesExp] Branching ratio spectral radius ≈ {spr:.3f} ≥ 1; "
                "the process may be highly clustered or explosive on long horizons.",
                RuntimeWarning,
            )
    except Exception:
        # If eigen computation fails, proceed silently (finite horizon still OK)
        pass

    rng = np.random.default_rng(seed)

    # State
    t = 0.0
    timestamps: List[List[float]] = [[] for _ in range(dim)]
    # R[l, k] = sum_{events in source l} exp(-β_{l,k} * (t - t_i^l))  at *current t*
    R = np.zeros((dim, dim), dtype=float)
    # Current intensities at t: λ_k(t) = μ_k + sum_l α_{l,k} * R[l,k]
    lam = baseline.copy()

    # Simulation loop
    n_events = 0
    while t < horizon and n_events < max_events:
        lam_bar = lam.sum()
        if lam_bar <= 0.0:
            # No intensity: process is silent forever (μ == 0 and no memory)
            break

        # Propose next time using upper bound lam_bar (sum intensity decreases between events)
        w = rng.exponential(scale=1.0 / lam_bar)
        s = t + w
        if s > horizon:
            break

        # Decay memory to proposed time
        dt = s - t
        # elementwise exp(-β * dt)
        R *= np.exp(-beta * dt)

        # True intensities at s
        lam_s = baseline + (alpha * R).sum(axis=0)
        lam_sum_s = lam_s.sum()

        # Thinning accept/reject
        if rng.random() * lam_bar <= lam_sum_s:
            # Accept an event at time s; choose its dimension
            probs = lam_s / lam_sum_s
            k = rng.choice(dim, p=probs)
            timestamps[k].append(float(s))

            # Update memory instantly: a new source event in row k adds 1 to all targets
            R[k, :] += 1.0

            # Intensities jump by the k-th row of alpha
            lam = lam_s + alpha[k, :]
            t = s
            n_events += 1
        else:
            # No event; move time forward with decayed intensities
            lam = lam_s
            t = s

    return [np.array(ts, dtype=float) for ts in timestamps]


# ----------------------- Dataset recipe using the simulator -----------------------

@register_dataset("hawkes_exp")
class HawkesExp(DatasetRecipe):
    """Multi-dimensional Hawkes with exponential kernels (pure NumPy).
    Maps each dimension k to a fixed (u,i); choose `target_type` to mark positives.
    """
    @classmethod
    def default_config(cls):
        return dict(
            dim=4,
            horizon=10.0,
            baseline=[0.1, 0.1, 0.1, 0.1],
            adjacency=[
                [0.0, 0.2, 0.0, 0.0],
                [0.0, 0.0, 0.2, 0.0],
                [0.0, 0.0, 0.0, 0.5],
                [0.1, 0.0, 0.0, 0.0],
            ],
            decays=1.5,
            type_edge_mapping=[[0, 1], [2, 3], [1, 2], [3, 0]],
            target_type=3,
            target_label=1,
            other_label=-1,
            negative_label=0,
            negative_ratio=1.0,
            include_etype=True,
            negative_etype=-1,
            num_nodes=20,
            node_feat_dim=8,
            bipartite=False,
            seed=42,
            baseline_scale=1.0,
        )

    def generate(self) -> DatasetBundle:
        cfg = {**self.default_config(), **self.config}
        rng = np.random.default_rng(cfg.get("seed", None))

        dim = int(cfg["dim"])
        horizon = float(cfg["horizon"])
        num_nodes = int(cfg.get("num_nodes", dim))

        def _expand_vector(values, name: str) -> np.ndarray:
            arr = np.asarray(values, dtype=float)
            if arr.ndim == 0:
                return np.full(dim, float(arr))
            arr = arr.reshape(-1)
            if arr.size == 0:
                raise ValueError(f"{name} must contain at least one value.")
            if arr.size == dim:
                return arr
            reps = (dim + arr.size - 1) // arr.size
            return np.tile(arr, reps)[:dim]

        def _expand_matrix(values, name: str) -> np.ndarray:
            arr = np.asarray(values, dtype=float)
            if arr.ndim == 0:
                return np.full((dim, dim), float(arr))
            if arr.ndim != 2:
                raise ValueError(f"{name} must be a square 2D array or broadcastable value.")
            if arr.shape[0] != arr.shape[1]:
                raise ValueError(f"{name} must be square, got shape {arr.shape}.")
            base_dim = arr.shape[0]
            if base_dim == 0:
                raise ValueError(f"{name} must contain at least one row.")
            if base_dim == dim:
                return arr
            reps = (dim + base_dim - 1) // base_dim
            tiled = np.tile(arr, (reps, reps))
            return tiled[:dim, :dim]

        def _expand_mapping(mapping_cfg) -> np.ndarray:
            if isinstance(mapping_cfg, str):
                mode = mapping_cfg.lower()
                if mode in {"cycle", "sequential"}:
                    if num_nodes < 2:
                        raise ValueError("Need at least two nodes for cycle mapping.")
                    u = np.arange(dim, dtype=int) % num_nodes
                    i = (u + 1) % num_nodes
                    return np.stack([u, i], axis=1)
                if mode == "random":
                    if num_nodes < 2:
                        raise ValueError("Need at least two nodes for random mapping.")
                    u = rng.integers(0, num_nodes, size=dim, dtype=int)
                    i = rng.integers(0, num_nodes, size=dim, dtype=int)
                    if num_nodes > 1:
                        mask = u == i
                        i[mask] = (i[mask] + 1) % num_nodes
                    return np.stack([u, i], axis=1)
                raise ValueError(f"Unknown mapping mode '{mapping_cfg}'.")

            arr = np.asarray(mapping_cfg, dtype=int)
            if arr.ndim != 2 or arr.shape[1] != 2:
                raise ValueError("type_edge_mapping must be an array of shape (dim, 2).")
            if (arr < 0).any():
                raise ValueError("type_edge_mapping entries must be non-negative.")
            base_dim = arr.shape[0]
            if base_dim == 0:
                raise ValueError("type_edge_mapping must contain at least one pair.")
            if num_nodes <= 0:
                raise ValueError("num_nodes must be positive.")
            if base_dim >= dim:
                mapping = arr[:dim]
            else:
                unique_nodes = np.unique(arr)
                stride = int(cfg.get("mapping_stride", unique_nodes.size if unique_nodes.size else base_dim))
                if stride <= 0:
                    stride = max(1, base_dim)
                blocks = (dim + base_dim - 1) // base_dim
                mapping_blocks = []
                for block in range(blocks):
                    offset = block * stride
                    block_rows = (arr + offset) % num_nodes
                    mapping_blocks.append(block_rows)
                mapping = np.vstack(mapping_blocks)[:dim]
            if (mapping < 0).any() or (mapping >= num_nodes).any():
                raise ValueError("Expanded type_edge_mapping references nodes outside the range [0, num_nodes).")
            return mapping

        mu = _expand_vector(cfg["baseline"], "baseline")
        baseline_scale = float(cfg.get("baseline_scale", 1.0))
        if baseline_scale <= 0:
            raise ValueError("baseline_scale must be positive.")
        mu *= baseline_scale
        alpha = _expand_matrix(cfg["adjacency"], "adjacency")

        decays = cfg["decays"]
        if np.isscalar(decays):
            beta = float(decays) * np.ones((dim, dim), dtype=float)
        else:
            beta = np.array(decays, dtype=float).reshape(dim, -1)
            if beta.shape[1] == 1:
                beta = np.repeat(beta, dim, axis=1)
            if beta.shape != (dim, dim):
                raise ValueError(f"`decays` must broadcast to ({dim},{dim}), got {beta.shape}")

        # --- Simulate Hawkes (replaces tick.SimuHawkesExpKernels) ---
        timestamps = simulate_hawkes_exp_kernels(
            baseline=mu,
            adjacency=alpha,
            decays=beta,
            horizon=horizon,
            seed=cfg.get("seed", None),
        )

        type_edge_mapping = _expand_mapping(cfg["type_edge_mapping"])
        target_type = int(cfg["target_type"])
        target_label = int(cfg.get("target_label", 1))
        other_label = int(cfg.get("other_label", -1))
        include_etype = bool(cfg.get("include_etype", True))
        negative_label = int(cfg.get("negative_label", 0))
        negative_ratio = float(cfg.get("negative_ratio", 1.0))
        negative_etype = int(cfg.get("negative_etype", -1))

        # Build interaction rows
        rows = []
        for k in range(dim):
            u, i = map(int, type_edge_mapping[k])
            label = target_label if k == target_type else other_label
            ts_k = np.asarray(timestamps[k], dtype=float)
            for t in ts_k:
                row = [u, i, float(t), int(label)]
                if include_etype:
                    row.append(int(k))
                rows.append(tuple(row))

        # Negatives sampled uniformly in the target-type support (or [0, horizon] as fallback)
        if negative_ratio > 0:
            ts_target = np.asarray(timestamps[target_type], dtype=float)
            num_neg = int(np.ceil(len(ts_target) * negative_ratio))
            if num_neg > 0:
                if len(ts_target) >= 1:
                    t_low, t_high = float(ts_target.min()), float(ts_target.max())
                    if t_high <= t_low:
                        t_low, t_high = 0.0, horizon
                else:
                    t_low, t_high = 0.0, horizon
                neg_ts = rng.uniform(low=t_low, high=t_high, size=num_neg)
                u_neg, i_neg = map(int, type_edge_mapping[target_type])
                for t in neg_ts:
                    row = [u_neg, i_neg, float(t), negative_label]
                    if include_etype:
                        row.append(negative_etype)
                    rows.append(tuple(row))

        columns = ["u", "i", "ts", "label"]
        if include_etype:
            columns.append("etype")
        df = pd.DataFrame(rows, columns=columns).sort_values("ts").reset_index(drop=True)

        # Node and (optional) edge features
        n = num_nodes
        d = int(cfg["node_feat_dim"])
        node_features = rng.normal(size=(n, d))
        if include_etype and len(df):
            edge_features = df["etype"].to_numpy().reshape(-1, 1)
        else:
            edge_features = None

        meta: Dict[str, object] = {
            "recipe": "hawkes_exp",
            "config": cfg,
            "node_feat_dim": d,
            "edge_feat_dim": 1 if include_etype else 0,
            "bipartite": bool(cfg.get("bipartite", False)),
        }
        return {
            "interactions": df,
            "node_features": node_features,
            "edge_features": edge_features,
            "metadata": meta,
        }

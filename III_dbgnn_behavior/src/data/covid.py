from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional

import numpy as np
import torch

import pathpyG as pp

from data.ho_triples import attach_ho_triples_to_data


@dataclass(frozen=True)
class CovidAssets:
    """Extra objects derived from the COVID dataset that DBGNN visualizations need."""

    t: pp.TemporalGraph
    m: pp.MultiOrderModel
    g: pp.Graph
    g2: pp.Graph | None
    meta: dict


def _read_tedges(path: str) -> list[tuple[int, int, int]]:
    edges: list[tuple[int, int, int]] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p for p in line.replace(",", " ").split() if p]
            if len(parts) < 3:
                continue
            u, v, t = map(int, parts[:3])
            edges.append((u, v, t))
    return edges


def _generate_contacts_random(
    *,
    num_nodes: int,
    num_steps: int,
    contacts_per_step: int = 80,
    make_undirected: bool = True,
    seed: int = 0,
) -> tuple[list[tuple[int, int, int]], list[list[tuple[int, int]]]]:
    rng = np.random.default_rng(seed)
    undirected_by_time: list[list[tuple[int, int]]] = [[] for _ in range(num_steps)]
    directed_edges: list[tuple[int, int, int]] = []

    for t in range(num_steps):
        for _ in range(int(contacts_per_step)):
            u = int(rng.integers(0, num_nodes))
            v = int(rng.integers(0, num_nodes))
            while v == u:
                v = int(rng.integers(0, num_nodes))
            a, b = (u, v) if u < v else (v, u)
            undirected_by_time[t].append((a, b))

            directed_edges.append((u, v, t))
            if make_undirected:
                directed_edges.append((v, u, t))

    return directed_edges, undirected_by_time


def _directed_to_undirected_by_time(directed_edges: list[tuple[int, int, int]], num_steps: int) -> list[list[tuple[int, int]]]:
    undirected_by_time: list[list[tuple[int, int]]] = [[] for _ in range(num_steps)]
    for u, v, t in directed_edges:
        a, b = (u, v) if u < v else (v, u)
        if 0 <= t < num_steps:
            undirected_by_time[t].append((a, b))
    return undirected_by_time


def _simulate_covid_states(
    *,
    undirected_by_time: list[list[tuple[int, int]]],
    num_nodes: int,
    infectious_period: int = 10,
    initial_infected: int = 3,
    infection_lag: int = 1,
    seed: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    remaining_now = np.zeros(num_nodes, dtype=np.int32)
    init = rng.choice(num_nodes, size=min(int(initial_infected), num_nodes), replace=False)
    remaining_now[init] = int(infectious_period)

    T = len(undirected_by_time)
    y = np.zeros((T, num_nodes), dtype=np.int64)
    remaining = np.zeros((T, num_nodes), dtype=np.int32)

    for t in range(T):
        y[t] = (remaining_now > 0).astype(np.int64)
        remaining[t] = remaining_now

        if t == T - 1:
            break

        sick = remaining_now > 0
        remaining_next = np.maximum(remaining_now - 1, 0)

        new_inf = np.zeros(num_nodes, dtype=bool)
        for u, v in undirected_by_time[t]:
            if sick[u] and not sick[v]:
                new_inf[v] = True
            elif sick[v] and not sick[u]:
                new_inf[u] = True

        if int(infection_lag) != 1:
            raise ValueError("Only infection_lag=1 is supported (contacts at t -> sick at t+1).")

        remaining_next[new_inf] = int(infectious_period)
        remaining_now = remaining_next

    return y, remaining


def _build_dbgnn_samples_next_step(
    *,
    directed_edges: list[tuple[int, int, int]],
    y: np.ndarray,
    remaining: np.ndarray,
    num_nodes: int,
    infectious_period: int = 10,
    window: int = 2,
    max_order: int = 2,
    mapping: str = "last",
) -> tuple[list[object], dict]:
    T = int(y.shape[0])

    edges_by_time: list[list[tuple[int, int, int]]] = [[] for _ in range(T)]
    for (u, v, t) in directed_edges:
        if 0 <= t < T:
            edges_by_time[t].append((u, v, int(t)))

    samples = []
    for t in range(int(window) - 1, T - 1):
        slice_edges: list[tuple[int, int, int]] = []
        for s in range(t - int(window) + 1, t + 1):
            slice_edges.extend(edges_by_time[s])

        if len(slice_edges) == 0:
            continue

        tg = pp.TemporalGraph.from_edge_list(slice_edges, num_nodes=int(num_nodes))
        m = pp.MultiOrderModel.from_temporal_graph(tg, max_order=int(max_order))
        data = m.to_dbgnn_data(max_order=int(max_order), mapping=str(mapping))

        x_is_sick = torch.tensor(y[t], dtype=torch.float32).unsqueeze(1)
        x_rem = torch.tensor(remaining[t], dtype=torch.float32).unsqueeze(1) / float(infectious_period)
        x = torch.cat([x_is_sick, x_rem], dim=1)
        data.x = x

        if int(max_order) >= 2:
            g1 = m.layers[1]
            gk = m.layers[int(max_order)]
            ho_feats = []
            for ho_uid in gk.nodes:
                if isinstance(ho_uid, (tuple, list)):
                    first_uid, last_uid = ho_uid[0], ho_uid[-1]
                else:
                    first_uid = last_uid = ho_uid

                try:
                    first_idx = g1.mapping.to_idx(first_uid)
                    last_idx = g1.mapping.to_idx(last_uid)
                except Exception:
                    first_idx = int(first_uid)
                    last_idx = int(last_uid)

                ho_feats.append(torch.cat([x[first_idx], x[last_idx]], dim=0))

            if len(ho_feats) == 0:
                data.x_h = torch.zeros((int(gk.n), x.shape[1] * 2), dtype=torch.float32)
            else:
                data.x_h = torch.stack(ho_feats, dim=0).float()

        data.y = torch.tensor(y[t + 1], dtype=torch.long)
        data.time_index = int(t)
        samples.append(data)

    meta = dict(
        edges_by_time=edges_by_time,
        num_nodes=int(num_nodes),
        T=int(T),
        window=int(window),
        max_order=int(max_order),
        mapping=str(mapping),
        infectious_period=int(infectious_period),
    )
    return samples, meta


def _build_viz_assets_for_time(meta: dict, t_idx: int) -> SimpleNamespace:
    T = int(meta["T"])
    window = int(meta["window"])
    max_order = int(meta["max_order"])
    edges_by_time = meta["edges_by_time"]
    num_nodes = int(meta["num_nodes"])

    slice_edges: list[tuple[int, int, int]] = []
    for s in range(t_idx - window + 1, t_idx + 1):
        if 0 <= s < T:
            slice_edges.extend(edges_by_time[s])

    tg = pp.TemporalGraph.from_edge_list(slice_edges, num_nodes=num_nodes)
    m = pp.MultiOrderModel.from_temporal_graph(tg, max_order=max_order)
    g = m.layers[1]
    g2 = m.layers[2] if max_order >= 2 else None
    return SimpleNamespace(t=tg, m=m, g=g, g2=g2)


def load_covid(
    *,
    device: torch.device,
    num_test: float = 0.3,  # kept for API compatibility; unused
    seed: Optional[int] = None,
    **kwargs,
) -> tuple[object, CovidAssets]:
    """Build a single DBGNN sample for the COVID task and its viz assets.

    This mirrors the logic in 01_train_unified.ipynb but returns only one time
    sample (default: first test sample).
    """
    _ = num_test  # unused
    kw = dict(kwargs)

    use_synthetic = bool(kw.get("use_synthetic", True))
    contact_edges_path = kw.get("contact_edges_path", None)
    if not use_synthetic and not contact_edges_path:
        raise ValueError("Provide contact_edges_path when use_synthetic=False.")

    rng_seed = int(seed or 0)

    if use_synthetic:
        num_nodes = int(kw.get("num_nodes", 30))
        num_steps = int(kw.get("num_steps", 60))
        contacts_per_step = int(kw.get("contacts_per_step", 80))
        make_undirected = bool(kw.get("make_undirected", True))
        directed_edges, undirected_by_time = _generate_contacts_random(
            num_nodes=num_nodes,
            num_steps=num_steps,
            contacts_per_step=contacts_per_step,
            make_undirected=make_undirected,
            seed=rng_seed,
        )
    else:
        edges = _read_tedges(str(contact_edges_path))
        num_nodes = int(kw.get("num_nodes", 1 + max(max(u, v) for (u, v, _) in edges)))
        num_steps = int(kw.get("num_steps", 1 + max(t for (_, _, t) in edges)))
        make_undirected = bool(kw.get("make_undirected", True))
        if make_undirected:
            directed_edges = [(u, v, t) for (u, v, t) in edges] + [(v, u, t) for (u, v, t) in edges]
        else:
            directed_edges = edges
        undirected_by_time = _directed_to_undirected_by_time(directed_edges, num_steps)

    y, remaining = _simulate_covid_states(
        undirected_by_time=undirected_by_time,
        num_nodes=num_nodes,
        infectious_period=int(kw.get("infectious_period", 10)),
        initial_infected=int(kw.get("initial_infected", 3)),
        infection_lag=int(kw.get("infection_lag", 1)),
        seed=rng_seed + 1,
    )

    samples, meta = _build_dbgnn_samples_next_step(
        directed_edges=directed_edges,
        y=y,
        remaining=remaining,
        num_nodes=num_nodes,
        infectious_period=int(kw.get("infectious_period", 10)),
        window=int(kw.get("window", 2)),
        max_order=int(kw.get("max_order", 2)),
        mapping=str(kw.get("mapping", "last")),
    )
    if len(samples) == 0:
        raise RuntimeError("No COVID samples were generated (empty contact slices).")

    time_split = float(kw.get("time_split", 0.7))
    split = int(time_split * len(samples))
    test_samples = samples[split:] if split < len(samples) else []

    sample_index = kw.get("sample_index", None)
    if sample_index is not None:
        idx = int(sample_index)
        if idx < 0:
            idx = len(samples) + idx
        if idx < 0 or idx >= len(samples):
            raise IndexError(f"sample_index out of range: {sample_index}")
        viz_data = samples[idx]
    else:
        viz_data = test_samples[0] if len(test_samples) else samples[-1]

    data = viz_data.to(device)
    assets_viz = _build_viz_assets_for_time(meta, t_idx=int(viz_data.time_index))
    assets = CovidAssets(t=assets_viz.t, m=assets_viz.m, g=assets_viz.g, g2=assets_viz.g2, meta=meta)

    # Attach causal triples aligned with higher-order edges for explainers.
    # This is a best-effort utility; if something is missing, it no-ops.
    attach_ho_triples_to_data(data, g=assets.g, g2=assets.g2)
    return data, assets


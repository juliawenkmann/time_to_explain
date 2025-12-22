from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from .base import DatasetRecipe
from time_to_explain.core.registry import register_dataset
from time_to_explain.core.types import DatasetBundle


@dataclass
class GenConfig:
    num_nodes: int
    num_motifs: int
    triad_pool_size: int
    motif_repeats: int
    noise_per_motif: int
    allow_target_reuse: bool
    disjoint_triads: bool
    node_feat_dim: int
    edge_feat_dim: int
    dt: float
    seed: int
    target_label: int
    support_label: int
    noise_label: int
    edge_feature_mode: str
    node_feature_mode: str
    name: str = "triadic_closure"
    out_dir: str = "data"


def generate_tri_closure_stream(
    cfg: GenConfig,
) -> Tuple[List[Tuple[int, int, float, float]], Dict[int, List[int]], List[int]]:
    """
    Returns:
      events: list of (u, i, ts, label) in time order
      gt_raw: dict[target_event_row_idx] = [support_row_idx_1, support_row_idx_2]
      roles: list of role ids per event (0=noise, 1=support, 2=target)
    """
    rng = np.random.default_rng(cfg.seed)

    events: List[Tuple[int, int, float, float]] = []
    gt_raw: Dict[int, List[int]] = {}
    roles: List[int] = []

    used_target_pairs = set()

    t = 0.0

    def sample_edge(exclude_nodes: set[int] | None = None) -> Tuple[int, int]:
        while True:
            u = int(rng.integers(0, cfg.num_nodes))
            v = int(rng.integers(0, cfg.num_nodes))
            if u == v:
                continue
            if exclude_nodes and (u in exclude_nodes or v in exclude_nodes):
                continue
            return u, v

    def sample_triad() -> Tuple[int, int, int]:
        for _ in range(10_000):
            u, v, w = map(int, rng.choice(cfg.num_nodes, size=3, replace=False))
            if not cfg.allow_target_reuse and (u, v) in used_target_pairs:
                continue
            return u, v, w
        raise RuntimeError("Unable to sample a new triad under current reuse constraints.")

    triad_pool: List[Tuple[int, int, int]] = []
    triad_pool_size = int(cfg.triad_pool_size)
    if cfg.disjoint_triads:
        if triad_pool_size <= 0:
            triad_pool_size = cfg.num_nodes // 3
        if triad_pool_size <= 0:
            raise ValueError("disjoint_triads requires num_nodes >= 3")
        if triad_pool_size * 3 > cfg.num_nodes:
            raise ValueError("triad_pool_size too large for disjoint_triads.")
        nodes = rng.permutation(cfg.num_nodes)
        for k in range(triad_pool_size):
            u, v, w = map(int, nodes[3 * k : 3 * k + 3])
            triad_pool.append((u, v, w))
        if not cfg.allow_target_reuse:
            used_target_pairs.update((u, v) for u, v, _w in triad_pool)
    elif triad_pool_size > 0:
        max_attempts = max(1000, cfg.triad_pool_size * 50)
        attempts = 0
        while len(triad_pool) < triad_pool_size and attempts < max_attempts:
            attempts += 1
            u, v, w = sample_triad()
            triad_pool.append((u, v, w))
            if not cfg.allow_target_reuse:
                used_target_pairs.add((u, v))
        if len(triad_pool) < triad_pool_size:
            raise ValueError(
                "triad_pool_size too large for the reuse constraints; "
                "increase num_nodes or allow target reuse."
            )

    for _ in range(cfg.num_motifs):
        # ---- noise block (kept small so the pattern is very learnable) ----
        # We also avoid using the motif nodes in the noise block to keep explanations clean.
        # We'll pick motif nodes first, then add noise that avoids them.
        if triad_pool:
            u, v, w = triad_pool[int(rng.integers(0, len(triad_pool)))]
        else:
            u, v, w = sample_triad()
            if not cfg.allow_target_reuse:
                used_target_pairs.add((u, v))

        motif_nodes = {u, v, w}

        for _rep in range(max(1, int(cfg.motif_repeats))):
            for _k in range(cfg.noise_per_motif):
                nu, nv = sample_edge(exclude_nodes=motif_nodes)
                events.append((nu, nv, t, float(cfg.noise_label)))
                roles.append(0)
                t += cfg.dt

            # ---- motif: (u,w) then (w,v) then target (u,v) ----
            s1_idx = len(events)
            events.append((u, w, t, float(cfg.support_label)))
            roles.append(1)
            t += cfg.dt

            s2_idx = len(events)
            events.append((w, v, t, float(cfg.support_label)))
            roles.append(1)
            t += cfg.dt

            target_idx = len(events)
            events.append((u, v, t, float(cfg.target_label)))
            roles.append(2)
            t += cfg.dt

            gt_raw[target_idx] = [s1_idx, s2_idx]

    return events, gt_raw, roles


def write_raw_csv(path: Path, events: List[Tuple[int, int, float, float]], edge_feat_dim: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    # preprocess_data.py reads e[4:] as edge features; we give it zeros.
    if edge_feat_dim < 0:
        raise ValueError("edge_feat_dim must be >= 0")

    header_cols = ["u", "i", "ts", "label"] + [f"f{k}" for k in range(edge_feat_dim)]
    header = ",".join(header_cols) + "\n"

    if edge_feat_dim == 0:
        feat_suffix = ""
    else:
        feat_suffix = "," + ",".join(["0"] * edge_feat_dim)

    with path.open("w", encoding="utf-8") as f:
        f.write(header)
        for (u, i, ts, label) in events:
            f.write(f"{u},{i},{ts},{label}{feat_suffix}\n")


@register_dataset("triadic_closure")
class TriadicClosure(DatasetRecipe):
    @classmethod
    def default_config(cls) -> Dict[str, Any]:
        return dict(
            num_nodes=80,
            num_motifs=1000,
            triad_pool_size=0,
            motif_repeats=1,
            noise_per_motif=1,
            allow_target_reuse=False,
            disjoint_triads=False,
            node_feat_dim=8,
            edge_feat_dim=1,
            dt=1.0,
            seed=0,
            target_label=1,
            support_label=0,
            noise_label=0,
            edge_feature_mode="zeros",
            node_feature_mode="random",
            bipartite=False,
        )

    def generate(self) -> DatasetBundle:
        cfg = {**self.default_config(), **(self.config or {})}
        if float(cfg.get("dt", 1.0)) <= 0:
            raise ValueError("dt must be > 0 to keep event order well-defined")

        gen_cfg = GenConfig(
            num_nodes=int(cfg["num_nodes"]),
            num_motifs=int(cfg["num_motifs"]),
            triad_pool_size=int(cfg.get("triad_pool_size", 0)),
            motif_repeats=int(cfg.get("motif_repeats", 1)),
            noise_per_motif=int(cfg["noise_per_motif"]),
            allow_target_reuse=bool(cfg.get("allow_target_reuse", False)),
            disjoint_triads=bool(cfg.get("disjoint_triads", False)),
            node_feat_dim=int(cfg["node_feat_dim"]),
            edge_feat_dim=int(cfg["edge_feat_dim"]),
            dt=float(cfg["dt"]),
            seed=int(cfg["seed"]),
            target_label=int(cfg["target_label"]),
            support_label=int(cfg["support_label"]),
            noise_label=int(cfg["noise_label"]),
            edge_feature_mode=str(cfg.get("edge_feature_mode", "zeros")),
            node_feature_mode=str(cfg.get("node_feature_mode", "random")),
        )

        events, gt_raw, roles = generate_tri_closure_stream(gen_cfg)
        df = pd.DataFrame(events, columns=["u", "i", "ts", "label"])
        df["event_id"] = np.arange(len(df), dtype=int)
        df = df.sort_values("ts").reset_index(drop=True)
        order = df["event_id"].to_numpy()
        roles_sorted = np.asarray(roles, dtype=int)[order]

        old_to_new = {int(old): int(new) for new, old in enumerate(df["event_id"].tolist())}
        targets = [old_to_new[int(k)] for k in gt_raw.keys()]
        rationales = {
            old_to_new[int(k)]: [old_to_new[int(v0)] for v0 in vs]
            for k, vs in gt_raw.items()
        }

        df = df.drop(columns=["event_id"])

        rng = np.random.default_rng(int(cfg["seed"]) + 1)
        node_feat_mode = str(cfg.get("node_feature_mode", "random")).lower()
        node_feat_dim = int(cfg["node_feat_dim"])
        if node_feat_mode in {"zeros", "zero"}:
            node_features = np.zeros((int(cfg["num_nodes"]), node_feat_dim), dtype=float)
        elif node_feat_mode in {"random", "rand"}:
            node_features = rng.normal(size=(int(cfg["num_nodes"]), node_feat_dim))
        else:
            raise ValueError(f"Unknown node_feature_mode '{node_feat_mode}'.")

        edge_feat_dim = int(cfg["edge_feat_dim"])
        if edge_feat_dim > 0 and len(df) > 0:
            edge_mode = str(cfg.get("edge_feature_mode", "zeros")).lower()
            if edge_mode in {"role", "roles", "label", "labels"} and edge_feat_dim < 1:
                raise ValueError("edge_feature_mode requires edge_feat_dim >= 1")
            edge_features = np.zeros((len(df), edge_feat_dim), dtype=float)
            if edge_mode in {"role", "roles"}:
                edge_features[:, 0] = roles_sorted
            elif edge_mode in {"label", "labels"}:
                edge_features[:, 0] = df["label"].to_numpy(dtype=float)
        else:
            edge_features = None

        meta: Dict[str, Any] = {
            "recipe": "triadic_closure",
            "config": cfg,
            "bipartite": bool(cfg.get("bipartite", False)),
            "node_feat_dim": int(cfg["node_feat_dim"]),
            "edge_feat_dim": edge_feat_dim,
            "edge_feature_mode": str(cfg.get("edge_feature_mode", "zeros")),
            "node_feature_mode": str(cfg.get("node_feature_mode", "random")),
            "disjoint_triads": bool(cfg.get("disjoint_triads", False)),
            "ground_truth": {
                "targets": targets,
                "rationales": rationales,
                "label_semantics": "label==1 marks triadic-closure target edges.",
                "explanation_semantics": "Each target edge is explained by the two immediately preceding support edges.",
            },
        }

        return {
            "interactions": df,
            "node_features": node_features,
            "edge_features": edge_features,
            "metadata": meta,
        }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--name", type=str, default="triadic_closure", help="Dataset name (writes data/{name}.csv)")
    p.add_argument("--out_dir", type=str, default="data", help="Output dir (default: ./data)")
    p.add_argument("--num_nodes", type=int, default=2000)
    p.add_argument("--num_motifs", type=int, default=20000, help="Each motif contributes 3 edges + noise")
    p.add_argument("--triad_pool_size", type=int, default=0, help="Reuse a pool of triads (0 = no reuse)")
    p.add_argument("--motif_repeats", type=int, default=1, help="Repeat the (support, support, target) motif")
    p.add_argument("--noise_per_motif", type=int, default=1, help="Keep small for easy learning (0-5)")
    p.add_argument("--allow_target_reuse", action="store_true", help="Allow repeating the same (u,v) target pair")
    p.add_argument("--disjoint_triads", action="store_true", help="Use disjoint triads (no node overlap)")
    p.add_argument("--node_feat_dim", type=int, default=8)
    p.add_argument("--edge_feat_dim", type=int, default=172, help="Edge feature dimension (zeros).")
    p.add_argument("--edge_feature_mode", type=str, default="zeros", help="zeros | role | label")
    p.add_argument("--node_feature_mode", type=str, default="random", help="random | zeros")
    p.add_argument("--dt", type=float, default=1.0, help="Timestamp increment per event")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--target_label", type=int, default=1)
    p.add_argument("--support_label", type=int, default=0)
    p.add_argument("--noise_label", type=int, default=0)
    args = p.parse_args()

    cfg = GenConfig(
        name=args.name,
        out_dir=args.out_dir,
        num_nodes=args.num_nodes,
        num_motifs=args.num_motifs,
        triad_pool_size=args.triad_pool_size,
        motif_repeats=args.motif_repeats,
        noise_per_motif=args.noise_per_motif,
        allow_target_reuse=bool(args.allow_target_reuse),
        disjoint_triads=bool(args.disjoint_triads),
        node_feat_dim=args.node_feat_dim,
        edge_feat_dim=args.edge_feat_dim,
        dt=args.dt,
        seed=args.seed,
        target_label=args.target_label,
        support_label=args.support_label,
        noise_label=args.noise_label,
        edge_feature_mode=str(args.edge_feature_mode),
        node_feature_mode=str(args.node_feature_mode),
    )

    events, gt_raw, _roles = generate_tri_closure_stream(cfg)

    out_dir = Path(cfg.out_dir)
    raw_csv = out_dir / f"{cfg.name}.csv"
    write_raw_csv(raw_csv, events, cfg.edge_feat_dim)

    gt_processed = {str(k + 1): [v[0] + 1, v[1] + 1] for k, v in gt_raw.items()}
    gt_raw_json = {str(k): v for k, v in gt_raw.items()}

    (out_dir / f"{cfg.name}_gt_raw.json").write_text(json.dumps(gt_raw_json, indent=2), encoding="utf-8")
    (out_dir / f"{cfg.name}_gt.json").write_text(json.dumps(gt_processed, indent=2), encoding="utf-8")

    meta = {
        "config": asdict(cfg),
        "num_events": len(events),
        "num_targets": len(gt_raw),
        "target_fraction": float(len(gt_raw) / max(1, len(events))),
        "note": "Targets are the (u,v) edges; each is explained by the two immediately preceding support edges.",
    }
    (out_dir / f"{cfg.name}_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("Wrote:")
    print(" ", raw_csv)
    print(" ", out_dir / f"{cfg.name}_gt_raw.json")
    print(" ", out_dir / f"{cfg.name}_gt.json")
    print(" ", out_dir / f"{cfg.name}_meta.json")
    print(f"Events: {len(events):,}   Targets: {len(gt_raw):,}")


if __name__ == "__main__":
    main()

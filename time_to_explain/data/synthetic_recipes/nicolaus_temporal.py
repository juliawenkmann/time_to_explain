# time_to_explain/data/synthetic_recipes/nicolaus_temporal.py
from __future__ import annotations
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd

from .base import DatasetRecipe
from time_to_explain.core.registry import register_dataset
from time_to_explain.core.types import DatasetBundle


@register_dataset("nicolaus")
class NicolausTemporal(DatasetRecipe):
    """
    Temporal dataset with a ground-truth rationale based on the
    "Haus des Nicolaus" motif.

    Key properties:
    - Generates one or more *house instances* over time. Each instance emits
      exactly one event per motif edge (8 total), ordered so that the *last*
      event is the unique missing edge that completes the house.
    - For each instance, we record:
        * target_event_idx: the row index of the finishing edge (after sort)
        * rationale_event_indices: the 7 rows of previous motif edges
      => The ground-truth explanation is "the house until that point".
    - Optional bipartite lift (Twitter-style TGN compatibility):
        i := i + num_nodes, so (u in [0..N-1], i in [N..2N-1]).
      This keeps the stream shape most TGN repos expect.

    Interactions DataFrame columns:
        ["u", "i", "ts", "label"]
      where label=1 for motif edges, label=0 for noise.
    """

    # ------------------------------ config ---------------------------------- #
    @classmethod
    def default_config(cls) -> Dict[str, Any]:
        return dict(
            # Graph size
            num_nodes=36,              # >= 5; extra nodes act as noise/background
            # Time
            horizon=160.0,             # global timeline [0, horizon]
            num_houses=80,             # how many motif instances to generate
            house_dt=1.0,              # duration window for a single house
            house_gap=1.0,             # minimum gap between house windows
            # Noise
            noise_pairs_per_house=12,  # how many non-motif undirected pairs to sprinkle per house
            noise_rate=0.8,            # Poisson rate on each chosen noise pair across [t0, t0+house_dt]
            # Features
            node_feat_dim=8,
            make_edge_features=True,   # 1-D feature = per-pair interaction count (over full DF)
            # Direction & export compatibility
            directed=False,            # emit undirected as canonical (min, max); bipartite still yields directed stream
            bipartite=False,           # if True: map i -> i + num_nodes (Twitter TGN style)
            # Reproducibility
            seed=42,
        )

    # ---------------------------- motif spec -------------------------------- #
    @staticmethod
    def _motif() -> Tuple[List[int], List[Tuple[int, int]], Dict[int, Tuple[float, float]]]:
        """
        Nodes: A=0, B=1, C=2, D=3, E=4
        Edges (undirected):
          - square: (A,B), (B,C), (C,D), (D,A)
          - roof:   (B,E), (C,E)
          - diagonals: (A,C), (B,D)
        """
        A, B, C, D, E = 0, 1, 2, 3, 4
        nodes = [A, B, C, D, E]
        edges = [
            (A, B), (B, C), (C, D), (D, A),  # square
            (D, E), (C, E),                  # roof (to top corners)
            (A, C), (B, D),                  # diagonals ("X")
        ]
        pos = {
            A: (0.0, 0.0),
            B: (1.0, 0.0),
            C: (1.0, 1.0),
            D: (0.0, 1.0),
            E: (0.5, 1.5),
        }
        return nodes, edges, pos

    # ------------------------------ helpers --------------------------------- #
    @staticmethod
    def _canon(u: int, v: int) -> Tuple[int, int]:
        return (u, v) if u <= v else (v, u)

    # ------------------------------ main gen -------------------------------- #
    def generate(self) -> DatasetBundle:
        cfg = dict(self.config or {})

        # Config
        N = int(cfg.get("num_nodes", 36))
        horizon = float(cfg.get("horizon", 160.0))
        num_houses = int(cfg.get("num_houses", 80))
        house_dt = float(cfg.get("house_dt", 1.0))
        house_gap = float(cfg.get("house_gap", 1.0))

        noise_pairs_per_house = int(cfg.get("noise_pairs_per_house", 12))
        noise_rate = float(cfg.get("noise_rate", 0.8))

        node_feat_dim = int(cfg.get("node_feat_dim", 8))
        make_edge_features = bool(cfg.get("make_edge_features", True))

        directed = bool(cfg.get("directed", False))
        bipartite = bool(cfg.get("bipartite", False))

        seed = int(cfg.get("seed", 42))
        rng = np.random.default_rng(seed)

        assert N >= 5, "num_nodes must be at least 5 to host the motif"
        assert num_houses >= 1 and house_dt > 0.0
        # Ensure we can place windows; if horizon is small, we still pack them as evenly as possible
        total_span = num_houses * house_dt + (num_houses - 1) * house_gap
        if total_span > horizon:
            # pack evenly without gaps if necessary
            house_gap = max(0.0, (horizon - num_houses * house_dt) / max(1, (num_houses - 1)))

        motif_nodes, motif_edges_und, motif_pos = self._motif()
        motif_edges_und = [self._canon(u, v) for (u, v) in motif_edges_und]
        motif_edge_set = set(motif_edges_und)

        # Build a pool of possible non-motif undirected pairs
        all_pairs = [self._canon(u, v) for u in range(N) for v in range(u + 1, N)]
        noise_pool = [p for p in all_pairs if p not in motif_edge_set]

        # Events buffer with extra fields so we can compute ground truth after sorting
        # We'll build in homogeneous space first (u,i ∈ [0..N-1]), then optionally bipartite-lift.
        events: List[Dict[str, Any]] = []
        house_meta: List[Dict[str, Any]] = []

        # Place house windows along the timeline
        if num_houses == 1:
            t0s = [0.5 * (horizon - house_dt)]
        else:
            t0s = [i * (house_dt + house_gap) for i in range(num_houses)]
            # clamp to horizon
            t0s = [min(t, max(0.0, horizon - house_dt)) for t in t0s]

        for h_id, t0 in enumerate(t0s):
            t1 = t0 + house_dt

            # Choose a random order for the 8 motif edges; schedule first 7 in [t0, t1-ε), last one near t1
            order = motif_edges_und.copy()
            rng.shuffle(order)
            last_edge = order[-1]
            early = order[:-1]

            # First 7 edges uniformly in [t0, t1 - margin]
            margin = min(0.05 * house_dt, 0.05)  # keep a small slot for the last event
            early_times = np.sort(t0 + rng.random(len(early)) * max(house_dt - margin, 1e-6))
            # Last edge time near the end
            last_time = float(max(t0 + house_dt - margin * 0.5, t0)) + float(rng.random() * margin * 0.5)

            # Emit the 7 early edges
            for (u0, v0), tt in zip(early, early_times):
                u1, v1 = (u0, v0) if directed else self._canon(u0, v0)
                events.append(dict(u=u1, i=v1, ts=float(tt), label=1,
                                   is_motif=True, house_id=h_id, is_target=False))

            # Emit the finishing (8th) edge
            u0, v0 = last_edge
            u1, v1 = (u0, v0) if directed else self._canon(u0, v0)
            events.append(dict(u=u1, i=v1, ts=float(last_time), label=1,
                               is_motif=True, house_id=h_id, is_target=True))

            # Add a few noise pairs within [t0, t1]
            if noise_pool and noise_pairs_per_house > 0 and noise_rate > 0.0:
                chosen = rng.choice(len(noise_pool),
                                    size=min(noise_pairs_per_house, len(noise_pool)),
                                    replace=False)
                for idx in np.atleast_1d(chosen):
                    u0, v0 = noise_pool[int(idx)]
                    m = int(rng.poisson(noise_rate * house_dt))
                    if m <= 0:
                        continue
                    ts = np.sort(t0 + rng.random(m) * house_dt)
                    for tt in ts:
                        u1, v1 = (u0, v0) if directed else self._canon(u0, v0)
                        events.append(dict(u=u1, i=v1, ts=float(tt), label=0,
                                           is_motif=False, house_id=h_id, is_target=False))

            house_meta.append(dict(
                house_id=h_id,
                window=[float(t0), float(t1)],
                last_edge=last_edge,
                order=order,       # original undirected edges order
            ))

        # Build DataFrame and sort by time
        df = pd.DataFrame(events, columns=["u", "i", "ts", "label", "is_motif", "house_id", "is_target"])
        df = df.sort_values(["ts", "is_target"]).reset_index(drop=True)  # tie-break so finishing edge tends to be last in its window

        # --- BIPARTITE LIFT (Twitter TGN friendly) ---
        item_offset = N
        if bipartite:
            df["i"] = df["i"].astype(np.int64) + item_offset
            # Stream is now (u in [0..N-1], i in [N..2N-1]), directed as required by many TGN repos.

        # Node features
        node_features = np.random.default_rng(seed + 1).normal(size=(N, int(node_feat_dim)))

        # Edge features: per-pair interaction count (over full DF, after optional bipartite lift)
        if make_edge_features and len(df) > 0:
            counts = df.value_counts(["u", "i"]).reset_index(name="count")
            cnt_map = {(int(u), int(i)): int(c) for u, i, c in counts[["u", "i", "count"]].itertuples(index=False, name=None)}
            edge_features = np.array(
                [cnt_map[(int(u), int(i))] for u, i in df[["u", "i"]].itertuples(index=False, name=None)],
                dtype=float,
            ).reshape(-1, 1)
        else:
            edge_features = None

        # Ground-truth mapping: for each house, identify target row and its rationale rows (7 earlier motif edges of that house)
        targets: List[int] = []
        rationales: Dict[int, List[int]] = {}

        for hm in house_meta:
            h = hm["house_id"]
            mask = (df["house_id"] == h) & (df["is_motif"])
            rows_h = df.index[mask].tolist()
            # Among motif rows for this house, pick the one with is_target==True
            tgt_rows = [r for r in rows_h if bool(df.at[r, "is_target"])]
            if len(tgt_rows) != 1:
                # In rare tie-breaking cases, fall back to the latest-in-window motif edge
                tgt_idx = max(rows_h, key=lambda r: float(df.at[r, "ts"])) if rows_h else None
            else:
                tgt_idx = tgt_rows[0]

            if tgt_idx is None:
                continue

            # Rationale = the other motif edges of the same house occurring strictly before tgt time
            t_tgt = float(df.at[tgt_idx, "ts"])
            rationale_rows = [r for r in rows_h if r != tgt_idx and float(df.at[r, "ts"]) <= t_tgt]
            # Keep exactly 7 if over-inclusive
            rationale_rows = sorted(rationale_rows, key=lambda r: float(df.at[r, "ts"]))[:7]

            targets.append(int(tgt_idx))
            rationales[int(tgt_idx)] = [int(r) for r in rationale_rows]

        # Metadata for explainers & visualization
        meta: Dict[str, Any] = {
            "recipe": "nicolaus",
            "config": cfg,
            "bipartite": bipartite,
            "item_offset": item_offset if bipartite else 0,
            "node_feat_dim": int(node_feat_dim),
            "edge_feat_dim": 1 if edge_features is not None else 0,
            "ground_truth": {
                # The static motif definition (homogeneous IDs)
                "motif_nodes": [0, 1, 2, 3, 4],
                "motif_edges_undirected": [tuple(sorted(e)) for e in motif_edges_und],
                "positions": {int(k): tuple(map(float, v)) for k, v in self._motif()[2].items()},
                # Instance-level GT for explainers:
                "targets": targets,                    # row indices into df
                "rationales": rationales,              # mapping: target_row_idx -> [rationale_row_indices...]
                "label_semantics": "label==1 motif (positive) edges; label==0 noise.",
                "explanation_semantics": "For the finishing-edge target, explanation is all prior motif edges (the house so far).",
            },
            "houses": house_meta,  # per-instance timing and order (in homogeneous IDs)
            "notes": "Designed for TGN LP training and explanation evaluation on finishing-edge targets.",
        }

        return {
            "interactions": df[["u", "i", "ts", "label"]].reset_index(drop=True),
            "node_features": node_features,
            "edge_features": edge_features,
            "metadata": meta,
        }

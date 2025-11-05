from __future__ import annotations
import numpy as np
import pandas as pd

from .base import DatasetRecipe
from time_to_explain.core.registry import register_dataset
from time_to_explain.core.types import DatasetBundle

@register_dataset("erdos_temporal")
class ErdosTemporal(DatasetRecipe):
    @classmethod
    def default_config(cls):
        return dict(
            num_nodes=100,
            p=0.05,
            horizon=10.0,
            rate=0.5,
            positive_block_size=10,
            node_feat_dim=8,
            seed=42,
        )

    def generate(self) -> DatasetBundle:
        cfg = {**self.default_config(), **self.config}
        rng = np.random.default_rng(cfg.get("seed", None))

        n = int(cfg["num_nodes"])
        p = float(cfg["p"])
        horizon = float(cfg["horizon"])
        rate = float(cfg["rate"])
        K = int(cfg["positive_block_size"])
        d = int(cfg["node_feat_dim"])

        A = rng.random((n, n)) < p
        np.fill_diagonal(A, False)

        interactions = []
        for u in range(n):
            for i in range(n):
                if not A[u, i]:
                    continue
                m = rng.poisson(lam=rate * horizon)
                if m == 0:
                    continue
                ts = np.sort(rng.random(m) * horizon)
                label = 1 if (u < K and i < K) else 0
                for t in ts:
                    interactions.append((u, i, float(t), int(label)))

        df = pd.DataFrame(interactions, columns=["u", "i", "ts", "label"]).sort_values("ts").reset_index(drop=True)

        node_features = rng.normal(size=(n, d))

        if len(df) > 0:
            counts = df.value_counts(["u","i"]).reset_index(name="count")
            cnt_map = {(int(u), int(i)): int(c) for u,i,c in counts[["u","i","count"]].itertuples(index=False, name=None)}
            edge_features = np.array([cnt_map[(int(u), int(i))] for u, i in df[["u","i"]].itertuples(index=False, name=None)]).reshape(-1,1)
        else:
            edge_features = None

        meta = {
            "recipe": "erdos_temporal",
            "config": cfg,
            "node_feat_dim": d,
            "edge_feat_dim": 1 if edge_features is not None else 0,
        }
        return {
            "interactions": df,
            "node_features": node_features,
            "edge_features": edge_features,
            "metadata": meta
        }

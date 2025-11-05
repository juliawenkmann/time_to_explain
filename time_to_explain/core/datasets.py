# time_to_explain/core/datasets.py
from dataclasses import dataclass
from pathlib import Path
import hashlib, json, numpy as np, pandas as pd
from .types import Subgraph  # your dataclass
from time_to_explain.explainer.tgnnexplainer.tg_dataset import load_tg_dataset, verify_dataframe_unify

def _file_sha1(p: Path) -> str:
    import hashlib
    h = hashlib.sha1()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

@dataclass
class TGEventsDataset:
    name: str
    root: Path

    def load(self):
        df, edge_feats, node_feats = load_tg_dataset(self.name)  # with fallback
        self.df, self.edge_feats, self.node_feats = df, edge_feats, node_feats
        return self

    @property
    def meta(self):
        base = self.root / "resources" / "datasets" / "processed" / self.name
        csv = base / f"ml_{self.name}.csv"
        e   = base / f"ml_{self.name}.npy"
        n   = base / f"ml_{self.name}_node.npy"
        # if canonical not found, skip hash (or compute from where loaded)
        items = []
        for p in [csv, e, n]:
            if p.exists(): items.append((p.name, _file_sha1(p)))
        return {"name": self.name, "artifacts": dict(items)}

    def subgraph_for_event(self, event_idx: int, k_hop=1):
        from time_to_explain.explainer.tgnnexplainer.utils_dataset import k_hop_temporal_subgraph
        df_sub = k_hop_temporal_subgraph(self.df, num_hops=k_hop, event_idx=event_idx)  # :contentReference[oaicite:14]{index=14}
        # Build Subgraph payload; edge_index list of (u, i)
        nodes = sorted(set(df_sub.u.tolist()) | set(df_sub.i.tolist()))
        id_map = {nid: j for j, nid in enumerate(nodes)}
        edge_index = [(id_map[u], id_map[v]) for u, v in zip(df_sub.u.tolist(), df_sub.i.tolist())]
        ts = df_sub.ts.tolist()
        return Subgraph(node_ids=nodes, edge_index=edge_index, timestamps=ts,
                        node_features=None, edge_features=None, payload={"df": df_sub})

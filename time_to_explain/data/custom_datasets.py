import pandas as pd
import numpy as np

def register_custom_csv(self, name: str, csv_path: str, bipartite: bool):
    name = name.lower()
    out = self.data_root / name
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)
    for col in ["src","dst","ts","label"]:
        if col not in df.columns:
            raise ValueError(f"custom CSV missing column '{col}'")
    if "edge_id" not in df.columns:
        df["edge_id"] = np.arange(len(df))
    df = df.sort_values("ts", kind="mergesort")
    df.to_csv(out/"events.csv", index=False)
    (out/"bipartite.txt").write_text("1\n" if bipartite else "0\n")
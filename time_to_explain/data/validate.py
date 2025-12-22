import pandas as pd

REQUIRED_COLS = ["u", "i", "ts", "label"]

# -----------------------------------------------------------------------------
# Dataset validation (adapted from your tg_dataset.py)
# -----------------------------------------------------------------------------

def check_wiki_reddit_dataformat(df: pd.DataFrame) -> None:
    # raw format coming from SNAP: (0-based user, 0-based item, ts, label, comma-separated features)
    assert df.iloc[:, 0].min() == 0
    assert df.iloc[:, 0].max() + 1 == df.iloc[:, 0].nunique()  # 0, 1, 2, ...
    assert df.iloc[:, 1].min() == 0
    assert df.iloc[:, 1].max() + 1 == df.iloc[:, 1].nunique()
    for col in ["u", "i", "ts", "label"]:
        assert col in df.columns.to_list()


def verify_dataframe_unify(df: pd.DataFrame, *, bipartite: bool = True) -> None:
    for col in ["u", "i", "ts", "label", "e_idx", "idx"]:
        assert col in df.columns.to_list()
    assert df["e_idx"].min() == 1
    assert df["e_idx"].max() == len(df)
    assert df["idx"].min() == 1
    assert df["idx"].max() == len(df)

    if bipartite:
        # After reindexing we expect users start at 1 and items continue after users
        assert df.iloc[:, 0].min() == 1
        assert df.iloc[:, 0].max() == df.iloc[:, 0].nunique()
        assert df.iloc[:, 1].min() == df.iloc[:, 0].max() + 1
        assert df.iloc[:, 1].max() == df.iloc[:, 0].max() + df.iloc[:, 1].nunique()
    else:
        # Non-bipartite: users/items share the same ID space.
        assert df.iloc[:, 0].min() >= 1
        assert df.iloc[:, 1].min() >= 1
        nodes = pd.unique(df[["u", "i"]].to_numpy().ravel())
        assert int(nodes.min()) == 1
        assert len(nodes) == max(int(df["u"].max()), int(df["i"].max()))


def verify_interactions(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"interactions missing columns: {missing}. Required: {REQUIRED_COLS}")
    df = df.copy()
    df["u"] = df["u"].astype(int)
    df["i"] = df["i"].astype(int)
    df["label"] = df["label"].astype(int)
    df["ts"] = df["ts"].astype(float)
    df = df.sort_values("ts").reset_index(drop=True)
    return df

def basic_stats(df: pd.DataFrame) -> dict:
    return {
        "num_interactions": int(len(df)),
        "min_ts": float(df["ts"].min()) if len(df) else None,
        "max_ts": float(df["ts"].max()) if len(df) else None,
        "num_users": int(df["u"].nunique()),
        "num_items": int(df["i"].nunique()),
        "label_balance": df["label"].value_counts(dropna=False).to_dict() if "label" in df else {},
    }

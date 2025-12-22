#!/usr/bin/env python3
"""
Unified TGNN data setup (Python-only, no bash).
- Downloads real & simulated datasets
- Processes real datasets (wikipedia, reddit) into TGAT-processed format
- Generates "explain indices" for each dataset
- Exposes a single function `setup_tgnn_data(...)` and a CLI

This merges/simplifies the pipeline previously split across:
  * process.py
  * tg_dataset.py
  * (utils_dataset.py was not needed for the pipeline; we inline only one tiny utility)

Assumptions
-----------
- Default layout uses:
    <ROOT>/resources/datasets/raw/
    <ROOT>/resources/datasets/processed/
- You can override the directories via `data_dir`, `proc_dir`, and `idx_dir`.
- For simulated datasets (v1, v2) we just download the pre-generated processed files.
- For real datasets, we convert CSV -> ml_*.csv + .npy feature files following the
  logic found in your provided process.py.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, Optional, Sequence
import urllib.request
import numpy as np
import pandas as pd
import sys
import random
from time_to_explain.data.validate import check_wiki_reddit_dataformat, verify_dataframe_unify
# -----------------------------------------------------------------------------
# Path helpers
# -----------------------------------------------------------------------------

def resolve_root(root: Optional[os.PathLike] = None) -> Path:
    """
    Resolve repository root. Priority:
    1) explicit `root` arg
    2) env var ROOT / REPO_ROOT / PROJECT_ROOT
    3) current working directory
    """
    if root:
        return Path(root).expanduser().resolve()
    for key in ("ROOT", "REPO_ROOT", "PROJECT_ROOT"):
        if key in os.environ and os.environ[key].strip():
            return Path(os.environ[key]).expanduser().resolve()
    return Path.cwd().resolve()


def ensure_dir(p: os.PathLike) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

# -----------------------------------------------------------------------------
# Processing helpers (adapted from your process.py)
# -----------------------------------------------------------------------------

def simulate_dataset_train_flag(df: pd.DataFrame) -> np.ndarray:
    labels = df["label"].to_numpy()
    return (labels == 1) | (labels == 0)


def rename_columns_wiki_reddit(file_path: Path) -> None:
    """
    SNAP wiki/reddit files come with 4 leading cols then a single 'comma_separated_list_of_features' col.
    We expand/rename them to: u, i, ts, label, f0...f{K-1}
    """
    df = pd.read_csv(file_path, skiprows=1, header=None)
    feat_nums = df.shape[1] - 4
    new_columns = ["u", "i", "ts", "label"] + [f"f{i}" for i in range(feat_nums)]
    rename_dict = {i: new_columns[i] for i in range(len(new_columns))}
    df.rename(columns=rename_dict, inplace=True)
    df.to_csv(file_path, index=False)
    print(f"Renamed columns in {file_path}")


def reindex(df: pd.DataFrame) -> pd.DataFrame:
    """
    Shift user ids to start at 1, and items to continue after the max user id, also 1-based.
    """
    df = df.copy()
    df["i"] += df["u"].max() + 1
    df["u"] += 1
    df["i"] += 1
    df["e_idx"] = df.index.values + 1
    df["idx"] = df["e_idx"]
    return df


def process_real_dataset(
    data_name: str,
    data_csv: Path,
    out_dir: Path,
) -> None:
    """
    Convert raw wikipedia/reddit CSV into TGAT-style processed files:
      - ml_<name>.csv with columns [u,i,ts,label,idx,e_idx]
      - ml_<name>.npy (edge features; 0-th row reserved as padding)
      - ml_<name>_node.npy (node features; zeros, dimension = edge feat dim)
    """
    out_dir = ensure_dir(out_dir)
    OUT_DF = out_dir / f"ml_{data_name}.csv"
    OUT_EDGE_FEAT = out_dir / f"ml_{data_name}.npy"
    OUT_NODE_FEAT = out_dir / f"ml_{data_name}_node.npy"

    df = pd.read_csv(data_csv)

    # If original 5-col format (with a single 'comma_separated_list_of_features'), expand/rename it.
    if "comma_separated_list_of_features" in df.columns.tolist():
        rename_columns_wiki_reddit(data_csv)
        df = pd.read_csv(data_csv)

    check_wiki_reddit_dataformat(df)
    df = reindex(df)
    verify_dataframe_unify(df)

    # For wiki/reddit, edge features come from columns named f*
    select_columns = [c for c in df.columns if c.startswith("f")]
    edge_feat = np.zeros((len(df) + 1, len(select_columns)), dtype=float)  # row 0 is padding
    if select_columns:
        edge_feat[1:, :] = df[select_columns].to_numpy()

    edge_feat_dim = edge_feat.shape[1]
    num_nodes = int(df["i"].max())
    node_feat = np.zeros((num_nodes + 1, edge_feat_dim), dtype=float)      # all zeros

    # Sanity checks like in your script
    assert len(node_feat) == df["i"].max() + 1
    assert len(edge_feat) == len(df) + 1

    # Save
    df[["u", "i", "ts", "label", "idx", "e_idx"]].to_csv(OUT_DF, index=False)
    np.save(OUT_EDGE_FEAT, edge_feat)
    np.save(OUT_NODE_FEAT, node_feat)
    print(f"[{data_name}] saved {OUT_DF.name}, {OUT_EDGE_FEAT.name}, {OUT_NODE_FEAT.name} -> {out_dir}")


# -----------------------------------------------------------------------------
# Explain-index generation (adapted from your tg_dataset.py)
# -----------------------------------------------------------------------------

def generate_explain_index(
    ml_csv: Path,
    out_dir: Path,
    dataset_name: str,
    size: int = 500,
    seed: int = 42,
    explain_idx_name: Optional[str] = None,
    verbose: bool = True,
    bipartite: Optional[bool] = None,
) -> Path:
    """
    Create a CSV listing event indices to be explained.
    For simulate_v* we sample positives by label==1.
    For wikipedia/reddit we sample uniformly from the 70%-99% tail of event indices.
    """
    rng = np.random.default_rng(seed)
    df = pd.read_csv(ml_csv)
    if bipartite is None:
        try:
            u_max = int(df["u"].max())
            i_min = int(df["i"].min())
            bipartite = i_min >= (u_max + 1)
        except Exception:
            bipartite = True

    verify_dataframe_unify(df, bipartite=bool(bipartite))

    candidates: np.ndarray
    if dataset_name in {"simulate_v1", "simulate_v2"}:
        positive_mask = df["label"] == 1
        candidates = df.loc[positive_mask, "e_idx"].to_numpy()
        if len(candidates) == 0:
            candidates = df["e_idx"].to_numpy()
    elif dataset_name in {"wikipedia", "reddit"}:
        e_num = len(df)
        low = int(e_num * 0.70)
        high = max(int(e_num * 0.99), low + 1)
        candidates = df["e_idx"].to_numpy()[low:high]
        if len(candidates) == 0:
            candidates = df["e_idx"].to_numpy()
    else:
        if "label" in df.columns:
            positives = df.loc[df["label"] > 0, "e_idx"].to_numpy()
            candidates = positives if len(positives) else df["e_idx"].to_numpy()
        else:
            candidates = df["e_idx"].to_numpy()

    if len(candidates) == 0:
        raise ValueError(f"No candidate events found for dataset '{dataset_name}'.")
    if size > len(candidates):
        if verbose:
            print(
                f"[{dataset_name}] Requested {size} explain indices, "
                f"but only {len(candidates)} candidates available. Using all candidates."
            )
        size = len(candidates)
    explain_idxs = rng.choice(candidates, size=size, replace=False)

    explain_idxs = sorted(int(x) for x in explain_idxs)
    out_dir = ensure_dir(out_dir)
    out_path = out_dir / (f"{explain_idx_name}.csv" if explain_idx_name else f"{dataset_name}.csv")
    pd.DataFrame({"event_idx": explain_idxs}).to_csv(out_path, index=False)
    if verbose:
        print(f"[{dataset_name}] explain index -> {out_path}")
    return out_path


# -----------------------------------------------------------------------------
# Downloaders
# -----------------------------------------------------------------------------

def download(url: str, dest: Path, force: bool = False) -> None:
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0 and not force:
        print(f"✔ Skipping (exists): {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"↓ Downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)  # no progress bar to keep it stdlib-only


def download_all(
    root: Path,
    only: Optional[Sequence[str]] = None,
    force: bool = False,
    *,
    data_dir: Optional[Path] = None,
    proc_dir: Optional[Path] = None,
) -> None:
    only = set(x.strip() for x in only) if only else None

    def want(name: str) -> bool:
        return (only is None) or (name in only)

    data_dir = Path(data_dir) if data_dir is not None else (root / "resources" / "datasets" / "raw")
    proc_dir = Path(proc_dir) if proc_dir is not None else (root / "resources" / "datasets" / "processed")

    # Real datasets
    if want("reddit"):
        download("http://snap.stanford.edu/jodie/reddit.csv", data_dir / "reddit.csv", force=force)
    if want("wikipedia"):
        download("http://snap.stanford.edu/jodie/wikipedia.csv", data_dir / "wikipedia.csv", force=force)

    # Simulated v1
    if want("simulate_v1"):
        download("https://m-krastev.github.io/hawkes-sim-datasets/simulate_v1.csv", data_dir / "simulate_v1.csv", force=force)
        download("https://m-krastev.github.io/hawkes-sim-datasets/ml_simulate_v1.csv", proc_dir / "ml_simulate_v1.csv", force=force)
        download("https://m-krastev.github.io/hawkes-sim-datasets/ml_simulate_v1.npy", proc_dir / "ml_simulate_v1.npy", force=force)
        download("https://m-krastev.github.io/hawkes-sim-datasets/ml_simulate_v1_node.npy", proc_dir / "ml_simulate_v1_node.npy", force=force)

    # Simulated v2
    if want("simulate_v2"):
        download("https://m-krastev.github.io/hawkes-sim-datasets/simulate_v2.csv", data_dir / "simulate_v2.csv", force=force)
        download("https://m-krastev.github.io/hawkes-sim-datasets/ml_simulate_v2.csv", proc_dir / "ml_simulate_v2.csv", force=force)
        download("https://m-krastev.github.io/hawkes-sim-datasets/ml_simulate_v2.npy", proc_dir / "ml_simulate_v2.npy", force=force)
        download("https://m-krastev.github.io/hawkes-sim-datasets/ml_simulate_v2_node.npy", proc_dir / "ml_simulate_v2_node.npy", force=force)


# -----------------------------------------------------------------------------
# Orchestrator
# -----------------------------------------------------------------------------

def setup_tgnn_data(
    root: Optional[os.PathLike] = None,
    only: Optional[Sequence[str]] = None,
    force: bool = False,
    do_process: bool = True,
    do_index: bool = True,
    seed: int = 42,
    index_size: int = 500,
    data_dir: Optional[os.PathLike] = None,
    proc_dir: Optional[os.PathLike] = None,
    idx_dir: Optional[os.PathLike] = None,
) -> None:
    """
    One-call pipeline:
      1) download datasets
      2) process real datasets into TGAT format
      3) generate explain indices

    Args:
      root: repo root (defaults to CWD or env vars ROOT/REPO_ROOT/PROJECT_ROOT)
      only: optional subset, e.g. ["wikipedia", "simulate_v2"]
      force: re-download files even if present
      do_process: run the processing step for real datasets
      do_index: generate explain indices
      seed: RNG seed for index sampling
      index_size: how many indices to sample for each dataset
    """
    root = resolve_root(root)
    print(f"ROOT: {root}")

    data_dir = Path(data_dir) if data_dir is not None else (root / "resources" / "datasets" / "raw")
    proc_dir = Path(proc_dir) if proc_dir is not None else (root / "resources" / "datasets" / "processed")
    idx_dir = Path(idx_dir) if idx_dir is not None else (root / "resources" / "datasets" / "explain_index")
    ensure_dir(data_dir)
    ensure_dir(proc_dir)
    ensure_dir(idx_dir)

    only_set = set(x.strip() for x in only) if only else None
    def want(name: str) -> bool:
        return (only_set is None) or (name in only_set)

    # 1) Downloads
    download_all(root, only=only, force=force, data_dir=data_dir, proc_dir=proc_dir)

    # 2) Process real datasets
    if do_process:
        if want("wikipedia"):
            wikipedia_csv = data_dir / "wikipedia.csv"
            if not wikipedia_csv.exists():
                print("! wikipedia.csv not found; skipping processing.")
            else:
                process_real_dataset("wikipedia", wikipedia_csv, proc_dir)
        if want("reddit"):
            reddit_csv = data_dir / "reddit.csv"
            if not reddit_csv.exists():
                print("! reddit.csv not found; skipping processing.")
            else:
                process_real_dataset("reddit", reddit_csv, proc_dir)
    else:
        print("Skipping processing step (--no-process requested).")

    # 3) Generate indices
    if do_index:
        rng = random.Random(seed)
        if want("wikipedia"):
            ml_csv = proc_dir / "ml_wikipedia.csv"
            if ml_csv.exists():
                generate_explain_index(ml_csv, idx_dir, "wikipedia", size=index_size, seed=seed)
            else:
                print("! Processed ml_wikipedia.csv missing; skipping index gen.")
        if want("reddit"):
            ml_csv = proc_dir / "ml_reddit.csv"
            if ml_csv.exists():
                generate_explain_index(ml_csv, idx_dir, "reddit", size=index_size, seed=seed)
            else:
                print("! Processed ml_reddit.csv missing; skipping index gen.")

        if want("simulate_v1"):
            ml_csv = proc_dir / "ml_simulate_v1.csv"
            if ml_csv.exists():
                generate_explain_index(ml_csv, idx_dir, "simulate_v1", size=index_size, seed=seed)
            else:
                print("! ml_simulate_v1.csv missing; did simulated downloads complete?")
        if want("simulate_v2"):
            ml_csv = proc_dir / "ml_simulate_v2.csv"
            if ml_csv.exists():
                generate_explain_index(ml_csv, idx_dir, "simulate_v2", size=index_size, seed=seed)
            else:
                print("! ml_simulate_v2.csv missing; did simulated downloads complete?")
    else:
        print("Skipping index generation (--no-index requested).")

    print("Done.")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unified Python-only TGNN data setup")
    p.add_argument("--root", type=str, default=None, help="Repo root (defaults to CWD or env var ROOT/REPO_ROOT/PROJECT_ROOT)")
    p.add_argument("--only", type=str, default=None, help="Comma list: reddit,wikipedia,simulate_v1,simulate_v2")
    p.add_argument("--force", action="store_true", help="Re-download even if file exists")
    p.add_argument("--no-process", dest="do_process", action="store_false", help="Skip processing the real datasets")
    p.add_argument("--no-index", dest="do_index", action="store_false", help="Skip generating explain indices")
    p.add_argument("--seed", type=int, default=42, help="Random seed for explain index sampling")
    p.add_argument("--index-size", type=int, default=500, help="How many indices per dataset")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    only = [s.strip() for s in args.only.split(",")] if args.only else None
    setup_tgnn_data(
        root=args.root,
        only=only,
        force=args.force,
        do_process=args.do_process,
        do_index=args.do_index,
        seed=args.seed,
        index_size=args.index_size,
    )


if __name__ == "__main__":
    main()

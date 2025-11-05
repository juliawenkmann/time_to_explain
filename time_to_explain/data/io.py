from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd

from .validate import verify_dataframe_unify, verify_interactions
from time_to_explain.core.types import DatasetBundle


def resolve_repo_root() -> Path:
    for key in ("PROJECT_ROOT", "REPO_ROOT", "TIME_TO_EXPLAIN_ROOT"):
        if key in os.environ:
            return Path(os.environ[key]).expanduser().resolve()

    here = Path(__file__).resolve()
    package_root = here.parents[1]

    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        if out:
            return Path(out)
    except Exception:
        pass

    for candidate in (here.parent, *here.parents):
        if any((candidate / marker).exists() for marker in (".git", "pyproject.toml", "setup.cfg", "setup.py")):
            return candidate

    return package_root


ROOT_DIR = resolve_repo_root()


def _processed_dir(dataset_name: str, root_dir: Optional[Path] = None) -> Path:
    base = Path(root_dir) if root_dir is not None else ROOT_DIR
    return base / "resources" / "datasets" / "processed" / dataset_name


def _raw_dir(dataset_name: str, root_dir: Optional[Path] = None) -> Path:
    base = Path(root_dir) if root_dir is not None else ROOT_DIR
    return base / "resources" / "datasets" / "raw"


def save_data(
    df: pd.DataFrame,
    node_feats: np.ndarray,
    edge_feats: np.ndarray,
    dataset_name: str,
    root_dir: Optional[Path] = None,
) -> Dict[str, str]:
    verify_dataframe_unify(df)

    assert len(node_feats) == df.i.max() + 1
    assert len(edge_feats) == len(df) + 1

    base_root = Path(root_dir) if root_dir is not None else ROOT_DIR
    processed = _processed_dir(dataset_name, root_dir=base_root)
    processed.mkdir(parents=True, exist_ok=True)
    raw_dir = _raw_dir(dataset_name, root_dir=base_root)
    raw_dir.mkdir(parents=True, exist_ok=True)

    raw_csv = raw_dir / f"{dataset_name}.csv"
    raw_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(raw_csv, index=False)

    csv_path = processed / f"ml_{dataset_name}.csv"
    edge_path = processed / f"ml_{dataset_name}.npy"
    node_path = processed / f"ml_{dataset_name}_node.npy"

    df.to_csv(csv_path, index=False)
    np.save(edge_path, edge_feats)
    np.save(node_path, node_feats)

    return {
        "raw_csv": str(raw_csv),
        "csv": str(csv_path),
        "edge_npy": str(edge_path),
        "node_npy": str(node_path),
    }


def load_processed_dataset(
    dataset: Union[str, Path],
    *,
    root_dir: Optional[Path] = None,
) -> DatasetBundle:
    dataset_path = Path(dataset)

    if dataset_path.exists():
        if dataset_path.is_dir():
            base = dataset_path
        else:
            base = dataset_path.parent
        dataset_name = base.name
    else:
        dataset_name = str(dataset)
        base = _processed_dir(dataset_name, root_dir)

    csv_path = base / f"ml_{dataset_name}.csv"
    edge_path = base / f"ml_{dataset_name}.npy"
    node_path = base / f"ml_{dataset_name}_node.npy"
    meta_path = base / f"ml_{dataset_name}.json"

    if not csv_path.exists():
        raise FileNotFoundError(f"Expected interactions csv at {csv_path}")

    interactions = pd.read_csv(csv_path)
    edge_features = np.load(edge_path) if edge_path.exists() else None
    node_features = np.load(node_path) if node_path.exists() else None

    metadata: Dict[str, Any] = {"dataset_name": dataset_name, "source": "processed"}
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)

    return {
        "interactions": interactions,
        "edge_features": edge_features,
        "node_features": node_features,
        "metadata": metadata,
    }


def save_metadata(dataset_name: str, metadata: Dict[str, Any], *, root_dir: Optional[Path] = None) -> Path:
    processed = _processed_dir(dataset_name, root_dir)
    processed.mkdir(parents=True, exist_ok=True)
    meta_path = processed / f"ml_{dataset_name}.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
    return meta_path


def export_tgn_csv(
    bundle: DatasetBundle,
    dataset_name: str,
    *,
    root_dir: Optional[Path] = None,
    overwrite: bool = False,
    seed: Optional[int] = 0,
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    if not isinstance(bundle, dict):
        raise TypeError("bundle must be a DatasetBundle dictionary")

    root = Path(root_dir) if root_dir is not None else ROOT_DIR
    processed_dir = _processed_dir(dataset_name, root)
    if processed_dir.exists() and not overwrite:
        raise FileExistsError(
            f"Processed dataset directory {processed_dir} already exists. Pass overwrite=True to replace it."
        )

    df = verify_interactions(bundle["interactions"])

    user_ids = sorted(df["u"].unique())
    item_ids = sorted(df["i"].unique())

    user_map = {old: idx + 1 for idx, old in enumerate(user_ids)}
    item_offset = len(user_ids)
    item_map = {old: item_offset + idx + 1 for idx, old in enumerate(item_ids)}

    df_tgn = df.copy()
    df_tgn["u"] = df_tgn["u"].map(user_map)
    df_tgn["i"] = df_tgn["i"].map(item_map)
    df_tgn["idx"] = np.arange(1, len(df_tgn) + 1)
    df_tgn["e_idx"] = df_tgn["idx"]

    verify_dataframe_unify(df_tgn)

    node_features = bundle.get("node_features")
    if node_features is None:
        meta = metadata or bundle.get("metadata") or {}
        cfg = meta.get("config", {})
        feat_dim = int(
            meta.get("node_feat_dim")
            or cfg.get("node_feat_dim")
            or len(user_ids) + len(item_ids)
        )
        rng = np.random.default_rng(seed)
        raw_node = rng.normal(size=(len(user_ids) + len(item_ids), max(1, feat_dim)))
    else:
        raw_node = np.asarray(node_features)
        if raw_node.ndim != 2:
            raise ValueError("node_features must be a 2D array")
        max_needed = max(user_ids + item_ids) if (user_ids or item_ids) else -1
        if raw_node.shape[0] <= max_needed:
            raise ValueError("node_features does not cover all node ids")

    feat_dim = raw_node.shape[1]
    node_feats = np.zeros((len(user_ids) + len(item_ids) + 1, feat_dim), dtype=raw_node.dtype)
    for old, new in user_map.items():
        node_feats[new] = raw_node[int(old)]
    for old, new in item_map.items():
        node_feats[new] = raw_node[int(old)]

    edge_features = bundle.get("edge_features")
    if edge_features is None:
        edge_dim = feat_dim
        edge_feats = np.zeros((len(df_tgn) + 1, edge_dim), dtype=node_feats.dtype)
        edge_feats[1:] = node_feats[df_tgn["u"].to_numpy()] + node_feats[df_tgn["i"].to_numpy()]
    else:
        edge_arr = np.asarray(edge_features)
        if edge_arr.shape[0] != len(df_tgn):
            raise ValueError("edge_features length must match number of interactions")
        if edge_arr.ndim == 1:
            edge_arr = edge_arr.reshape(-1, 1)
        edge_feats = np.zeros((len(df_tgn) + 1, edge_arr.shape[1]), dtype=edge_arr.dtype)
        edge_feats[1:] = edge_arr

    file_paths = save_data(df_tgn, node_feats, edge_feats, dataset_name, root_dir=root)

    meta_out: Dict[str, Any] = dict(metadata or bundle.get("metadata") or {})
    meta_out.setdefault("dataset_name", dataset_name)
    meta_out.setdefault("source", "processed")
    save_metadata(dataset_name, meta_out, root_dir=root)

    return Path(file_paths["csv"]).parent

def load_tg_dataset(dataset_name: str, root_dir: Optional[Path] = None):
    bundle = load_processed_dataset(dataset_name, root_dir=root_dir)
    df = verify_interactions(bundle["interactions"])

    edge_feats = bundle["edge_features"]
    node_feats = bundle["node_features"]

    if node_feats is None or edge_feats is None:
        raise FileNotFoundError(
            f"Missing node/edge feature arrays for dataset {dataset_name}. Expected *_node.npy and .npy files."
        )

    df["e_idx"] = df.index.values + 1
    assert df.i.max() + 1 == len(node_feats)
    assert df.e_idx.max() + 1 == len(edge_feats)

    n_users = df.iloc[:, 0].max()
    n_items = df.iloc[:, 1].max() - df.iloc[:, 0].max()
    print(
        f"#Dataset: {dataset_name}, #Users: {n_users}, #Items: {n_items}, "
        f"#Interactions: {len(df)}, #Timestamps: {df.ts.nunique()}"
    )
    print(f"#node feats shape: {node_feats.shape}, #edge feats shape: {edge_feats.shape}")

    return df, edge_feats, node_feats


def load_explain_idx(explain_idx_filepath, start=0, end=None):
    df = pd.read_csv(explain_idx_filepath)
    event_idxs = df["event_idx"].to_list()
    if end is not None:
        event_idxs = event_idxs[start:end]
    else:
        event_idxs = event_idxs[start:]

    print(f"{len(event_idxs)} events to explain")

    return event_idxs


def load_events_data(path):
    df = pd.read_csv(path)
    return df

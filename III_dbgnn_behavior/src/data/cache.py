from __future__ import annotations

"""Small helpers for persisting loaded datasets for later inspection.

Why this exists
---------------
When working with netzschleuder datasets (and temporal/higher-order transforms),
it is often useful to persist the *exact* `torch_geometric.data.Data` object
used for training/explanations, as well as minimal raw-graph information.

The goal is debugging / reproducibility, not a perfect round-trip for every
pathpyG object.
"""

from pathlib import Path
from typing import Any, Mapping, Optional

import json

import numpy as np
import torch


def _jsonable(x: Any) -> Any:
    """Best-effort conversion to something JSON serializable."""
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    # torch / numpy scalars
    if hasattr(x, "item"):
        try:
            return x.item()
        except Exception:
            pass
    return str(x)


def save_dataset_bundle(
    *,
    out_dir: str | Path,
    data: Any,
    assets: Any | None = None,
    extra_meta: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Save a dataset bundle to disk.

    Saved files (best-effort):
        - data.pt: torch-saved PyG Data (moved to CPU if possible)
        - meta.json: lightweight metadata
        - t_edge_index.pt / t_time.pt / t_node_ids.npy: if `assets.t` exists

    Returns:
        The created directory path.
    """

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # Save PyG Data
    # ----------------------------
    try:
        d_cpu = data.cpu() if hasattr(data, "cpu") else data
    except Exception:
        d_cpu = data
    try:
        torch.save(d_cpu, out / "data.pt")
    except Exception as e:
        # If saving fails, still write metadata so users can see why.
        (out / "data_save_error.txt").write_text(str(e), encoding="utf-8")

    # ----------------------------
    # Metadata
    # ----------------------------
    meta: dict[str, Any] = {}
    if extra_meta:
        meta.update({str(k): _jsonable(v) for k, v in extra_meta.items()})

    # Basic shapes
    try:
        meta["num_nodes"] = int(getattr(data, "num_nodes"))
    except Exception:
        pass
    try:
        ei = getattr(data, "edge_index")
        meta["num_edges"] = int(ei.size(1))
    except Exception:
        pass
    try:
        y = getattr(data, "y")
        if torch.is_tensor(y):
            meta["num_classes"] = int(torch.unique(y.cpu()).numel())
    except Exception:
        pass

    try:
        import pathpyG as pp  # noqa: F401

        meta["pathpyG_version"] = getattr(pp, "__version__", "unknown")
    except Exception:
        pass

    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # ----------------------------
    # Best-effort: save raw pathpyG graph info
    # ----------------------------
    if assets is not None:
        t = getattr(assets, "t", None)
        if t is not None and hasattr(t, "data"):
            # Edge index
            try:
                torch.save(t.data.edge_index.detach().cpu(), out / "t_edge_index.pt")
            except Exception:
                pass

            # Time (temporal graphs)
            if hasattr(t.data, "time") and getattr(t.data, "time") is not None:
                try:
                    torch.save(t.data.time.detach().cpu(), out / "t_time.pt")
                except Exception:
                    pass

            # Mapping node ids
            mapping = getattr(t, "mapping", None)
            if mapping is not None:
                node_ids = None
                for attr in ("node_ids", "ids"):
                    if hasattr(mapping, attr):
                        node_ids = getattr(mapping, attr)
                        break
                if node_ids is not None:
                    try:
                        np.save(out / "t_node_ids.npy", np.asarray(node_ids))
                    except Exception:
                        pass

            # Save node attribute tensors if present (prefixed with "node_")
            try:
                keys = list(t.data.keys()) if hasattr(t.data, "keys") else []
                for k in keys:
                    if not str(k).startswith("node_"):
                        continue
                    v = getattr(t.data, k)
                    if torch.is_tensor(v):
                        torch.save(v.detach().cpu(), out / f"{k}.pt")
            except Exception:
                pass


        # Also persist first- and second-order De Bruijn graphs when present.
        # These are invaluable for debugging plotting/indexing issues.
        for _name in ("g", "g2"):
            gg = getattr(assets, _name, None)
            if gg is None or not hasattr(gg, "data"):
                continue

            # Edge index
            try:
                torch.save(gg.data.edge_index.detach().cpu(), out / f"{_name}_edge_index.pt")
            except Exception:
                try:
                    # Some pathpyG versions use an EdgeIndex wrapper
                    ei = gg.data.edge_index.as_tensor()
                    torch.save(ei.detach().cpu(), out / f"{_name}_edge_index.pt")
                except Exception:
                    pass

            # Edge weights (if available)
            if hasattr(gg.data, "edge_weight") and getattr(gg.data, "edge_weight") is not None:
                try:
                    torch.save(gg.data.edge_weight.detach().cpu(), out / f"{_name}_edge_weight.pt")
                except Exception:
                    pass

            # Mapping node ids
            mapping = getattr(gg, "mapping", None)
            if mapping is not None:
                node_ids = None
                for attr in ("node_ids", "ids"):
                    if hasattr(mapping, attr):
                        node_ids = getattr(mapping, attr)
                        break
                if node_ids is not None:
                    try:
                        np.save(out / f"{_name}_node_ids.npy", np.asarray(node_ids))
                    except Exception:
                        pass

    return out

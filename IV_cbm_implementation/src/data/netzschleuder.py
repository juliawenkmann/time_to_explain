from __future__ import annotations

import csv
import inspect
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple, Union
import warnings

import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from torch_geometric.transforms import RandomNodeSplit

import pathpyG as pp

from data.ho_triples import attach_ho_triples_to_data


def _patch_pathpy_string_node_attributes() -> None:
    """Patch pathpyG's netzschleuder node-attribute parsing to tolerate strings.

    Why this exists
    ---------------
    Some netzschleuder records (e.g. SocioPatterns school datasets) contain
    node attributes that are *strings* (like class labels such as "2A" or
    categories like "M"/"F").

    Certain versions of pathpyG attempt to convert all node-attribute columns
    into `torch.Tensor`s directly. When a column contains strings, this can
    raise errors like:

        ValueError: too many dimensions 'str'

    This helper applies a very small monkey-patch to pathpyG's internal
    `_parse_df_column` function used by `read_netzschleuder_graph`, converting
    string/object columns into *categorical codes* (via `pandas.factorize`) on
    the fly.

    The patch is:
    - **idempotent** (safe to call multiple times)
    - **best-effort** (if pathpyG internals changed, we simply don't patch)
    - **only changes behaviour on failure** (tries the original code first)
    """

    try:
        import pathpyG.io.pandas as pp_pandas  # type: ignore
    except Exception:
        return

    if getattr(pp_pandas, "_GNNBENCH_PATCH_STR_NODE_ATTRS", False):
        return

    orig = getattr(pp_pandas, "_parse_df_column", None)
    if orig is None or not callable(orig):
        return

    def _patched_parse_df_column(df, data, attr, idx, prefix):  # type: ignore[no-untyped-def]
        try:
            return orig(df=df, data=data, attr=attr, idx=idx, prefix=prefix)
        except Exception as e:
            msg = str(e)

            # Common failure when torch.tensor(...) is called on string arrays.
            if "too many dimensions" in msg and "str" in msg:
                try:
                    import pandas as pd

                    codes, _uniques = pd.factorize(df[attr].astype(str), sort=True)
                    data[prefix + attr] = torch.tensor(
                        codes[idx].astype(np.int64),
                        device=data.edge_index.device,
                    )
                    return None
                except Exception:
                    # Fall through to "skip" behaviour below.
                    pass

            # If the attribute is unsupported / malformed, skip it instead of
            # aborting the entire load. We only *need* the target attribute,
            # which will be handled by the factorize branch above if needed.
            if "Unsupported data type" in msg:
                return None

            raise

    pp_pandas._parse_df_column = _patched_parse_df_column  # type: ignore[attr-defined]
    pp_pandas._GNNBENCH_PATCH_STR_NODE_ATTRS = True


def _maybe_fix_one_based_node_indexing(g: Union[pp.Graph, pp.TemporalGraph]) -> None:
    """Best-effort fixes for common netzschleuder indexing pitfalls.

    In practice, two *different* issues show up for some netzschleuder records:

    1) **One-based indices**: edge endpoints are stored as integers 1..N but are
       interpreted as already-indexed values. This yields out-of-range indices
       and crashes when plotting. If the pattern is unambiguous (min==1 and
       max==len(mapping)), we shift ``edge_index`` by -1.

    2) **Isolated nodes**: netzschleuder's record metadata may report a larger
       number of vertices than the set of vertices that actually appear in the
       edge list. ``pathpyG`` builds the ``IndexMap`` from *edge endpoints*,
       which excludes isolated nodes, but it still sets ``num_nodes`` from the
       metadata. This makes ``g.n`` larger than ``len(g.mapping)``, and calls
       like ``pp.plot(g)`` can crash with:

           IndexError: index K is out of bounds for axis 0 with size K

       For DBGNN training we only need the nodes that appear in the edge list.
       Therefore, if ``num_nodes`` and mapping length disagree, we *compress*
       node attributes to the mapping order (using ``mapping.node_ids`` as an
       index into the full attribute arrays) and then set ``num_nodes`` to the
       mapping length.

    This function is intentionally conservative: if we cannot safely detect and
    correct the mismatch, we leave the graph unchanged.
    """

    try:
        mapping = getattr(g, "mapping", None)
        if mapping is None:
            return

        # ------------------------------------------------------------------
        # Mapping length + (optional) node id list
        # ------------------------------------------------------------------
        node_ids = None
        for attr in ("node_ids", "ids"):
            if hasattr(mapping, attr):
                node_ids = getattr(mapping, attr)
                break

        node_ids_arr = None
        if node_ids is not None:
            try:
                node_ids_arr = np.asarray(node_ids)
            except Exception:
                node_ids_arr = None

        try:
            n_map = int(len(node_ids_arr) if node_ids_arr is not None else len(mapping))
        except Exception:
            return

        if not hasattr(g, "data"):
            return
        data = g.data

        # ------------------------------------------------------------------
        # (1) One-based indices in edge_index (unambiguous pattern only)
        # ------------------------------------------------------------------
        ei_raw = getattr(data, "edge_index", None)
        if ei_raw is not None:
            ei = ei_raw.as_tensor() if hasattr(ei_raw, "as_tensor") else ei_raw
            if torch.is_tensor(ei) and ei.numel() > 0:
                ei_min = int(ei.min().item())
                ei_max = int(ei.max().item())
                if ei_min == 1 and ei_max == int(n_map):
                    data.edge_index = ei - 1

        # ------------------------------------------------------------------
        # (2) num_nodes vs mapping length mismatch (isolated nodes)
        # ------------------------------------------------------------------
        n_data = getattr(data, "num_nodes", None)
        try:
            n_data_int = int(n_data) if n_data is not None else int(n_map)
        except Exception:
            n_data_int = int(n_map)

        if n_data_int != int(n_map):
            # Reindex node_* attributes if they look like "full" arrays indexed by
            # original node ids (0..n_data-1). This is the common case for
            # netzschleuder nodes.csv which uses the `index` column.
            idx_tensor = None
            if node_ids_arr is not None and node_ids_arr.size == n_map:
                # Only attempt fancy indexing if ids are integer-like and within bounds.
                try:
                    if node_ids_arr.dtype.kind in "iu":
                        if int(node_ids_arr.min()) >= 0 and int(node_ids_arr.max()) < int(n_data_int):
                            idx_tensor = torch.tensor(
                                node_ids_arr.astype(np.int64),
                                dtype=torch.long,
                                device=data.edge_index.device,
                            )
                except Exception:
                    idx_tensor = None

            # Compress node attributes
            try:
                keys = list(data.keys()) if hasattr(data, "keys") else []
            except Exception:
                keys = []

            for key in keys:
                if not str(key).startswith("node_"):
                    continue
                try:
                    v = data[key]
                except Exception:
                    continue
                if not torch.is_tensor(v):
                    continue
                if v.dim() == 0:
                    continue
                if int(v.size(0)) != int(n_data_int):
                    continue

                try:
                    if idx_tensor is not None:
                        data[key] = v[idx_tensor]
                    else:
                        data[key] = v[: int(n_map)]
                except Exception:
                    # If reindexing fails for some reason, skip the attribute.
                    continue

            # Finally, align num_nodes / n on the mapping length.
            try:
                data.num_nodes = int(n_map)
            except Exception:
                pass
            if hasattr(g, "_n"):
                try:
                    setattr(g, "_n", int(n_map))
                except Exception:
                    pass
    except Exception:
        # Best-effort only.
        return

@dataclass(frozen=True)
class NetzschleuderAssets:
    """Extra objects derived from a netzschleuder dataset that the DBGNN builder needs.

    Notes:
        * For temporal netzschleuder datasets, `t` is a `pp.TemporalGraph`.
        * For static netzschleuder datasets (no `time_attr`), `t` is a `pp.Graph`.
          In this case, we build a higher-order model from *path data* (2-step walks)
          so the DBGNN pipeline still works.
    """

    record: str
    network: Optional[str]
    time_attr: Optional[str]
    target_attr: str
    label_encoder: Optional[LabelEncoder]

    # "Raw" graph loaded from netzschleuder.
    t: Union[pp.Graph, pp.TemporalGraph]

    # Higher-order model and its first/second-order De Bruijn graphs.
    m: pp.MultiOrderModel
    g: pp.Graph
    g2: pp.Graph


# Defaults for a few commonly used netzschleuder records.
#
# Tip: For *any* other record, you can set:
#   dataset_name = "<record_name>"
#   dataset_kwargs = {"network": "...", "time_attr": "...", "target_attr": "..."}
# Convenience aliases for common record names (user-friendly -> netzschleuder record).
RECORD_ALIASES: Dict[str, str] = {
    "highschool": "sp_high_school",
    "high_school": "sp_high_school",
    "workplace": "sp_office",
    "office": "sp_office",
    "hospital": "sp_hospital",
}

NETZSCHLEUDER_DEFAULTS: Dict[str, Dict[str, Any]] = {


    # Copenhagen Networks Study (multilayer)
    "copenhagen": {
        "network": "sms",  # reasonable size and has timestamps
        "time_attr": "timestamp",
        "time_bin_size": 300,  # sampled every ~5 minutes (300s)
        "target_attr": "female",
    },
    # SocioPatterns / high school 2013 (multiple networks)
    "sp_high_school": {
        "network": "proximity",
        "time_attr": "time",
        "time_bin_size": 20,
        # In the SocioPatterns high school dataset, "class" is usually the
        # most informative node label for supervised learning.
        "target_attr": "class",
    },
    # SocioPatterns / high school 2011-2012 (multiple networks)
    "sp_high_school_new": {
        "network": "2012",
        "time_attr": "time",
        "time_bin_size": 20,
        "target_attr": "gender",
    },
    # SocioPatterns / hospital ward
    "sp_hospital": {
        "network": None,
        "time_attr": "time",
        "time_bin_size": 20,
        "target_attr": "status",  # NUR/PAT/MED/ADM
    },
    # SocioPatterns / primary school
    "sp_primary_school": {
        "network": None,
        "time_attr": "time",
        "time_bin_size": 20,
        "target_attr": "class",
    },
    # SocioPatterns / office contacts
    "sp_office": {
        "network": None,
        "time_attr": "time",
        "time_bin_size": 20,
        "target_attr": "department",
    },
    # Co-location proxy data (multiple networks, *very* large for some variants)
    "sp_colocation": {
        "network": "InVS13",  # smaller than Thiers13, still large
        "time_attr": "time",
        "time_bin_size": 20,
        "target_attr": "group",
    },
    # Small static demo graphs (from the pathpyG tutorial)
    "karate": {
        "network": "78",
        "time_attr": None,  # static
        "target_attr": "node_groups",
        # For static graphs we build paths from 2-step walks.
        "static_paths_mode": "all_length2",
        "static_max_paths": 200_000,
    },
    "polbooks": {
        "network": None,
        "time_attr": None,  # static
        "target_attr": "node_value",
        "static_paths_mode": "all_length2",
        "static_max_paths": 200_000,
    },
}


def _read_record_metadata(name: str, *, base_url: str) -> Dict[str, Any]:
    """Best-effort record metadata retrieval (used only for nicer defaults/errors)."""
    return pp.io.read_netzschleuder_record(name=name, base_url=base_url)


def _available_networks(meta: Dict[str, Any]) -> Tuple[str, ...]:
    """Try to extract the network list for a record.

    The netzschleuder API has evolved; we support multiple possible shapes.
    """
    if not isinstance(meta, dict):
        return tuple()

    # Common shape in some API versions: {"networks": {...}} or {"networks": [...]}
    nets = meta.get("networks", None)
    if isinstance(nets, dict):
        return tuple(str(k) for k in nets.keys())
    if isinstance(nets, list):
        out = []
        for item in nets:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, dict):
                # best guess: "name" field
                n = item.get("name", None)
                if n is not None:
                    out.append(str(n))
        return tuple(out)

    # Older heuristic used in some code: meta["analyses"] contains a dict per network.
    analyses = meta.get("analyses", None)
    if isinstance(analyses, dict) and "is_directed" not in analyses:
        return tuple(str(k) for k in analyses.keys())

    return tuple()


def _resolve_dataset_params(
    *,
    record: str,
    network: Optional[str],
    time_attr: Optional[str],
    target_attr: Optional[str],
    time_bin_size: Optional[int],
    base_url: str,
    static_paths_mode: Optional[str] = None,
    static_max_paths: Optional[int] = None,
) -> Tuple[Optional[str], Optional[str], str, Optional[int], str, int]:
    """Resolve defaults and validate obvious issues.

    Returns:
        network_param: network string or None (pass-through to pathpyG)
        time_attr_res: time attribute name or None (None => static graph)
        target_attr_res: node attribute used as label
        time_bin: optional bin size for temporal graphs
        static_mode: how to create path data for static graphs
        static_max_paths: cap for number of generated 2-step walks
    """
    defaults = NETZSCHLEUDER_DEFAULTS.get(record, {})

    if network is None:
        network = defaults.get("network", None)

    # If the record contains multiple networks and network is missing, pick one deterministically.
    # We keep this best-effort: if metadata fails, pathpyG will raise a helpful error.
    if not network:
        try:
            meta = _read_record_metadata(record, base_url=base_url)
            nets = _available_networks(meta)
            if nets:
                network = sorted(nets)[0]
        except Exception:
            pass

    network_param = str(network) if network else None

    # Temporal vs static
    if time_attr is None:
        time_attr = defaults.get("time_attr", None)

    time_attr_res: Optional[str] = str(time_attr) if time_attr else None

    # Target attribute (required)
    if target_attr is None:
        target_attr = defaults.get("target_attr", None)
    if not target_attr:
        raise ValueError(
            "Please provide dataset_kwargs={'target_attr': <node_attribute_name>} "
            "to define the node-classification task for a netzschleuder dataset."
        )
    target_attr_res = str(target_attr)

    # Time binning only applies to temporal graphs
    time_bin = None
    if time_attr_res is not None:
        if time_bin_size is None:
            time_bin_size = defaults.get("time_bin_size", None)
        if time_bin_size is not None:
            time_bin = int(time_bin_size)
            if time_bin <= 0:
                raise ValueError("time_bin_size must be a positive integer.")

    # Static path generation params
    if static_paths_mode is None:
        static_paths_mode = defaults.get("static_paths_mode", "auto")
    static_mode = str(static_paths_mode)

    if static_max_paths is None:
        static_max_paths = defaults.get("static_max_paths", 200_000)
    static_max_paths_int = int(static_max_paths)
    if static_max_paths_int <= 0:
        raise ValueError("static_max_paths must be a positive integer.")

    return network_param, time_attr_res, target_attr_res, time_bin, static_mode, static_max_paths_int


def _safe_to_device(obj, device: torch.device):
    """Best-effort device move for pathpyG / PyG objects across versions.

    Newer pathpyG objects typically expose `.to(device)`, while older versions
    may not. In the latter case, we try to move tensor fields in `obj.data`
    when possible and otherwise return the object unchanged.
    """
    to_fn = getattr(obj, "to", None)
    if callable(to_fn):
        try:
            moved = to_fn(device)
            return obj if moved is None else moved
        except Exception:
            # Fall back to tensor-level move below.
            pass

    data = getattr(obj, "data", None)
    if data is None or not hasattr(data, "keys"):
        return obj

    try:
        keys = list(data.keys())
    except Exception:
        keys = []

    for key in keys:
        try:
            val = data[key]
        except Exception:
            continue
        if torch.is_tensor(val):
            try:
                data[key] = val.to(device)
            except Exception:
                continue
    return obj


def _read_netzschleuder_graph_compat(
    *,
    name: str,
    network: Optional[str],
    time_attr: Optional[str],
    base_url: str,
):
    """Call `read_netzschleuder_graph` robustly across pathpyG versions.

    Some versions accept `network=...`; others do not expose this argument.
    """
    fn = pp.io.read_netzschleuder_graph
    kwargs: Dict[str, Any] = {
        "name": name,
        "time_attr": time_attr,
        "base_url": base_url,
    }

    accepted: Optional[set[str]] = None
    try:
        accepted = set(inspect.signature(fn).parameters.keys())
    except Exception:
        accepted = None

    if network is not None:
        if accepted is None or "network" in accepted:
            kwargs["network"] = network
        elif "networks" in accepted:
            kwargs["networks"] = network
        else:
            warnings.warn(
                "pathpyG.read_netzschleuder_graph does not expose a network selector; "
                f"ignoring network={network!r}.",
                RuntimeWarning,
            )

    try:
        return fn(**kwargs)
    except TypeError as e:
        msg = str(e)
        if "unexpected keyword argument 'network'" in msg and "network" in kwargs:
            kwargs.pop("network", None)
            warnings.warn(
                "pathpyG does not accept keyword 'network'; retrying without network selection.",
                RuntimeWarning,
            )
            return fn(**kwargs)
        if "unexpected keyword argument 'networks'" in msg and "networks" in kwargs:
            kwargs.pop("networks", None)
            warnings.warn(
                "pathpyG does not accept keyword 'networks'; retrying without network selection.",
                RuntimeWarning,
            )
            return fn(**kwargs)
        raise


def _resolve_pyg_key(data, requested: str) -> str:
    """Find a key in a PyG Data object even if naming differs slightly."""
    keys = list(data.keys()) if hasattr(data, "keys") else []
    if requested in keys:
        return requested

    # Common prefixes
    candidates = [
        f"node_{requested}",
        f"node__{requested}",
        f"node{requested}",
    ]
    for c in candidates:
        if c in keys:
            return c

    # Case-insensitive exact match
    low = requested.lower()
    for k in keys:
        if k.lower() == low:
            return k

    # Fuzzy: endswith
    for k in keys:
        if k.lower().endswith(low):
            return k

    raise KeyError(
        f"Could not find node attribute {requested!r} in PyG data keys. "
        f"Available keys include: {keys[:50]}{'...' if len(keys) > 50 else ''}"
    )


def _copenhagen_gender_csv_candidates(custom_path: Optional[str]) -> Tuple[Path, ...]:
    """Return candidate paths for Copenhagen SMS gender labels."""
    out: list[Path] = []
    if custom_path:
        out.append(Path(custom_path).expanduser())

    # Default to common notebook/workspace locations.
    cwd = Path.cwd()
    workspace_root = Path(__file__).resolve().parents[2]
    out.extend(
        [
            cwd / "genders.csv",
            cwd / "data" / "genders.csv",
            cwd / "julia_code" / "genders.csv",
            workspace_root / "genders.csv",
            workspace_root / "data" / "genders.csv",
        ]
    )

    # De-duplicate while preserving order.
    uniq: list[Path] = []
    seen = set()
    for p in out:
        rp = p.resolve() if p.exists() else p
        key = str(rp)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(p)
    return tuple(uniq)


def _load_gender_csv_map(csv_path: Path) -> Dict[int, int]:
    """Load {user_id -> female_binary} from a CSV file."""

    def _norm(s: str) -> str:
        return str(s).strip().lower().replace(" ", "").replace("#", "")

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header row: {csv_path}")

        field_map = {_norm(k): k for k in reader.fieldnames}
        user_col = None
        for c in ("user", "userid", "id", "nodeid"):
            if c in field_map:
                user_col = field_map[c]
                break
        if user_col is None:
            raise ValueError(
                f"Could not find user id column in {csv_path}. "
                f"Headers: {reader.fieldnames}"
            )

        val_col = None
        for c in ("female", "gender", "isfemale"):
            if c in field_map:
                val_col = field_map[c]
                break
        if val_col is None:
            raise ValueError(
                f"Could not find gender value column in {csv_path}. "
                f"Headers: {reader.fieldnames}"
            )

        out: Dict[int, int] = {}
        for row in reader:
            uid_raw = row.get(user_col, "")
            val_raw = row.get(val_col, "")

            if uid_raw is None or str(uid_raw).strip() == "":
                continue
            uid = int(str(uid_raw).strip())

            v = str(val_raw).strip().lower()
            if v in {"0", "1"}:
                out[uid] = int(v)
            elif v in {"female", "f", "true"}:
                out[uid] = 1
            elif v in {"male", "m", "false"}:
                out[uid] = 0
            else:
                raise ValueError(
                    f"Unsupported gender value {val_raw!r} for user {uid} in {csv_path}"
                )

    if not out:
        raise ValueError(f"No rows parsed from {csv_path}")
    return out


def _maybe_override_copenhagen_sms_labels_from_csv(
    *,
    record: str,
    network_param: Optional[str],
    target_attr_res: str,
    t_data: Any,
    csv_path: Optional[str] = None,
) -> Optional[torch.Tensor]:
    """Optionally replace Copenhagen SMS labels from a local genders.csv file.

    Why:
        Some pathpyG/netzschleuder versions expose `node_female` as all zeros
        for `copenhagen/sms`. When a local `genders.csv` is available, we can
        recover a valid binary target by mapping `node_id -> female`.
    """
    if str(record).strip().lower() != "copenhagen":
        return None
    if str(network_param or "").strip().lower() != "sms":
        return None
    if str(target_attr_res).strip().lower() not in {"female", "gender"}:
        return None

    candidates = _copenhagen_gender_csv_candidates(csv_path)
    chosen = next((p for p in candidates if p.exists()), None)
    if chosen is None:
        warnings.warn(
            "No genders.csv file found for copenhagen/sms label override. "
            f"Checked: {[str(p) for p in candidates]}",
            RuntimeWarning,
        )
        return None

    try:
        csv_map = _load_gender_csv_map(chosen)
    except Exception as e:
        warnings.warn(f"Failed to parse genders.csv at {chosen}: {e!r}", RuntimeWarning)
        return None

    try:
        id_key = _resolve_pyg_key(t_data, "id")
        node_ids = getattr(t_data, id_key)
    except Exception as e:
        warnings.warn(
            "Could not resolve node id attribute for genders.csv mapping "
            f"(path={chosen}): {e!r}",
            RuntimeWarning,
        )
        return None

    if torch.is_tensor(node_ids):
        ids = node_ids.detach().cpu().to(torch.long).tolist()
    else:
        ids = np.asarray(node_ids).reshape(-1).astype(np.int64).tolist()

    missing = [nid for nid in ids if nid not in csv_map]
    if missing:
        warnings.warn(
            f"genders.csv missing {len(missing)} node ids for copenhagen/sms "
            f"(example: {missing[:10]}). Marking missing labels as -1 (unlabeled).",
            RuntimeWarning,
        )

    # Use CSV labels when available; mark missing node ids as -1 (unlabeled).
    y = torch.tensor([int(csv_map.get(nid, -1)) for nid in ids], dtype=torch.long)
    uniq = torch.unique(y[y >= 0]) if bool((y >= 0).any()) else torch.tensor([], dtype=torch.long)
    if int(uniq.numel()) < 2:
        warnings.warn(
            f"genders.csv override produced a single class ({uniq.tolist()}). "
            "Falling back to pathpyG labels.",
            RuntimeWarning,
        )
        return None

    warnings.warn(
        f"Using genders.csv override for copenhagen/sms labels: {chosen}",
        UserWarning,
    )
    return y


def _encode_labels(values: Any) -> Tuple[torch.Tensor, Optional[LabelEncoder]]:
    """Encode arbitrary label arrays to contiguous int64 tensor."""

    def _as_int(x: Any) -> int:
        """Convert a scalar (python / numpy / torch 0-d) to int safely."""
        try:
            # torch scalar / numpy scalar
            if hasattr(x, "item"):
                return int(x.item())
        except Exception:
            pass
        return int(x)

    # torch.Tensor -> map unique values to 0..C-1
    if torch.is_tensor(values):
        v = values.detach().cpu()
        if v.numel() == 0:
            raise ValueError("Empty label tensor")
        # If float but represents categories, cast to int.
        if v.dtype.is_floating_point:
            if not torch.all(torch.isfinite(v)):
                raise ValueError("Label tensor contains non-finite values")
            if not torch.allclose(v, v.round()):
                # continuous labels -> not a classification target
                raise ValueError("Label tensor looks continuous (non-integer float).")
            v = v.round().to(torch.long)
        else:
            v = v.to(torch.long)

        uniq = torch.unique(v)
        # Handle common 1/2 encoding like karate in tutorial
        if torch.equal(uniq, torch.tensor([1, 2])):
            v = v - 1
            uniq = torch.unique(v)

        # Remap to contiguous ids
        mapping = {_as_int(u): i for i, u in enumerate(sorted(uniq.tolist()))}
        y = torch.tensor([mapping[_as_int(x)] for x in v.tolist()], dtype=torch.long)
        return y, None

    # numpy / list / scalar
    arr = np.asarray(values)
    if arr.ndim != 1:
        arr = arr.reshape(-1)

    # numeric
    if arr.dtype.kind in "iub":
        y = torch.tensor(arr.astype(np.int64), dtype=torch.long)
        uniq = torch.unique(y)
        mapping = {_as_int(u): i for i, u in enumerate(sorted(uniq.tolist()))}
        y = torch.tensor([mapping[_as_int(x)] for x in y.tolist()], dtype=torch.long)
        return y, None

    # floats: try discrete
    if arr.dtype.kind == "f":
        if not np.isfinite(arr).all():
            raise ValueError("Labels contain non-finite values")
        if not np.allclose(arr, np.round(arr)):
            raise ValueError("Label array looks continuous (non-integer float).")
        y = torch.tensor(arr.round().astype(np.int64), dtype=torch.long)
        uniq = torch.unique(y)
        mapping = {_as_int(u): i for i, u in enumerate(sorted(uniq.tolist()))}
        y = torch.tensor([mapping[_as_int(x)] for x in y.tolist()], dtype=torch.long)
        return y, None

    # strings / objects -> LabelEncoder
    le = LabelEncoder()
    # robust: convert to str but keep NaNs as "nan"
    y_np = le.fit_transform(arr.astype(str))
    y = torch.tensor(y_np.astype(np.int64), dtype=torch.long)
    return y, le


def _mapping_idx_to_id(mapping: Any, idx: int) -> Any:
    """Best-effort conversion from an index (0..n-1) to the original node ID."""
    # pathpyG IndexMap has changed over time; support a few common conventions.
    if hasattr(mapping, "to_id"):
        return mapping.to_id(idx)
    if hasattr(mapping, "to_ids"):
        # to_ids expects iterable of indices
        out = mapping.to_ids([idx])
        try:
            return out[0]
        except Exception:
            return out
    if hasattr(mapping, "ids"):
        return mapping.ids[idx]
    if hasattr(mapping, "node_ids"):
        return mapping.node_ids[idx]
    # last resort: return the index itself
    return idx


def _build_static_path_data_from_graph(
    g0: pp.Graph,
    *,
    device: torch.device,
    max_order: int,
    mode: str = "auto",
    max_paths: int = 200_000,
    seed: Optional[int] = None,
) -> pp.PathData:
    """Create PathData from a *static* graph so we can build a MultiOrderModel.

    We generate observations of 2-step walks (u -> v -> w). These are the minimal
    paths needed to construct a second-order De Bruijn graph (k=2).

    Args:
        mode:
            - "all_length2": enumerate all 2-step walks, capped by max_paths
            - "random_walks": sample 2-step walks, up to max_paths (with repetition)
            - "auto": enumerate if the estimated number of 2-step walks is <= max_paths,
                      otherwise sample.
    """
    if int(max_order) < 2:
        raise ValueError("Static netzschleuder graphs require max_order>=2 for the DBGNN pipeline.")

    # Work on CPU for neighbor lists even if the graph lives on GPU.
    ei = g0.data.edge_index
    src = ei[0].detach().cpu().numpy()
    dst = ei[1].detach().cpu().numpy()
    num_nodes = int(getattr(g0.data, "num_nodes", getattr(g0, "n", 0)))
    num_edges = int(src.shape[0])

    out_nbrs: list[list[int]] = [[] for _ in range(num_nodes)]
    for u, v in zip(src, dst):
        out_nbrs[int(u)].append(int(v))

    # Estimate #2-step walks (directed): sum_v indeg(v)*outdeg(v)
    indeg = np.bincount(dst, minlength=num_nodes)
    outdeg = np.bincount(src, minlength=num_nodes)
    est = int(np.sum(indeg * outdeg))

    mode = str(mode).lower()
    if mode == "auto":
        mode = "all_length2" if est <= max_paths else "random_walks"
    if mode not in {"all_length2", "random_walks"}:
        raise ValueError(f"Unknown static_paths_mode={mode!r}. Use 'auto', 'all_length2', or 'random_walks'.")

    rng = np.random.default_rng(int(seed) if seed is not None else None)

    # Precompute IDs so we can keep the original netzschleuder node IDs/mapping.
    ids = [_mapping_idx_to_id(g0.mapping, i) for i in range(num_nodes)]

    paths_counter: Counter = Counter()

    if mode == "all_length2":
        # Enumerate, but stop at max_paths to prevent blow-ups.
        n_added = 0
        for u in range(num_nodes):
            for v in out_nbrs[u]:
                for w in out_nbrs[v]:
                    paths_counter[(ids[u], ids[v], ids[w])] += 1
                    n_added += 1
                    if n_added >= max_paths:
                        break
                if n_added >= max_paths:
                    break
            if n_added >= max_paths:
                break

    else:
        # Sample 2-step walks by picking an edge (u,v) uniformly, then w uniformly from out(v).
        if num_edges == 0:
            raise ValueError("Static graph has no edges; cannot generate path data.")
        # Prebuild edge list for fast sampling
        edges = list(zip(src.tolist(), dst.tolist()))

        n_target = max_paths
        attempts = 0
        max_attempts = int(max_paths) * 20  # allow rejection (nodes with no outgoing)
        while sum(paths_counter.values()) < n_target and attempts < max_attempts:
            attempts += 1
            u, v = edges[int(rng.integers(0, num_edges))]
            v = int(v)
            if not out_nbrs[v]:
                continue
            w = int(out_nbrs[v][int(rng.integers(0, len(out_nbrs[v])))])

            paths_counter[(ids[int(u)], ids[v], ids[w])] += 1

        if sum(paths_counter.values()) == 0:
            raise ValueError("Could not generate any 2-step walks (graph may be too sparse).")

    node_seqs = list(paths_counter.keys())
    weights = [float(w) for w in paths_counter.values()]

    pd = pp.PathData(mapping=g0.mapping, device=device)
    pd.append_walks(node_seqs, weights)
    return pd


def load_netzschleuder(
    *,
    device: torch.device,
    record: str,
    network: Optional[str] = None,
    time_attr: Optional[str] = None,
    target_attr: Optional[str] = None,
    time_bin_size: Optional[int] = None,
    base_url: str = "https://networks.skewed.de",
    max_order: int = 2,
    num_test: float = 0.3,
    seed: Optional[int] = None,
    # Static-graph support (time_attr=None)
    static_paths_mode: str = "auto",
    static_max_paths: int = 200_000,
    # Optional local CSV override for copenhagen/sms female labels.
    gender_csv_path: Optional[str] = None,
) -> Tuple[object, NetzschleuderAssets]:
    """Load a netzschleuder dataset and build the DBGNN PyG Data.

    Temporal netzschleuder datasets
    -------------------------------
    If `time_attr` is provided (or is available via defaults), we load a `pp.TemporalGraph`
    and build a `pp.MultiOrderModel` from time-respecting paths:

        t = pp.io.read_netzschleuder_graph(..., time_attr=...)
        m = pp.MultiOrderModel.from_temporal_graph(t, max_order=2)
        data = m.to_dbgnn_data(max_order=2, mapping='last')

    Static netzschleuder datasets
    -----------------------------
    If `time_attr` is None, `pp.io.read_netzschleuder_graph` returns a `pp.Graph`.
    In that case, we generate *path data* from 2-step walks and build a higher-order
    model from `pp.PathData`:

        g0 = pp.io.read_netzschleuder_graph(..., time_attr=None)
        paths = PathData(...)  # 2-step walks
        m = pp.MultiOrderModel.from_path_data(paths, max_order=2)
        data = m.to_dbgnn_data(...)

    In both cases, node labels are taken from a node attribute (`target_attr`),
    encoded to contiguous class IDs, and split via `RandomNodeSplit`.

    Args:
        record: netzschleuder record name (e.g., "sp_hospital", "karate", ...).
        network: network name inside the record (needed for records with multiple networks).
        time_attr: edge attribute containing timestamps. If None -> treat as static graph.
        target_attr: node attribute to predict.
        time_bin_size: optional integer bin size for timestamps (temporal only).
        static_paths_mode: "auto" | "all_length2" | "random_walks" (static only).
        static_max_paths: cap for the number of generated 2-step walks (static only).
        gender_csv_path: optional path to a CSV with columns like
            `user,female` used to override `copenhagen/sms` labels.
    """
    # Apply friendly record aliases (e.g., "workplace" -> "sp_office").
    if record is not None:
        _rec_key = str(record).strip().lower()
        if _rec_key in RECORD_ALIASES:
            record = RECORD_ALIASES[_rec_key]

    network_param, time_attr_res, target_attr_res, time_bin, static_mode, static_max_paths_int = _resolve_dataset_params(
        record=record,
        network=network,
        time_attr=time_attr,
        target_attr=target_attr,
        time_bin_size=time_bin_size,
        base_url=base_url,
        static_paths_mode=static_paths_mode,
        static_max_paths=static_max_paths,
    )

    # Patch pathpyG's CSV attribute parsing to be resilient to string columns.
    # This prevents crashes like "ValueError: too many dimensions 'str'" on
    # datasets that store categorical node attributes as strings.
    _patch_pathpy_string_node_attributes()

    # ------------------------------------------------------------------
    # Load raw graph
    # ------------------------------------------------------------------
    if time_attr_res is not None:
        # Temporal graph
        t_raw = _read_netzschleuder_graph_compat(
            name=record,
            network=network_param,
            time_attr=time_attr_res,
            base_url=base_url,
        )
        if not isinstance(t_raw, pp.TemporalGraph):
            raise TypeError(
                f"Expected a TemporalGraph for record={record!r} network={network_param!r} time_attr={time_attr_res!r}, "
                f"but got {type(t_raw).__name__}."
            )

        # Some netzschleuder records store node ids as 1..N. Depending on the
        # pathpyG version, those ids can be interpreted as indices, producing an
        # off-by-one mismatch (n=N+1 while mapping has N ids). Fix this early so
        # plotting and downstream processing are stable.
        _maybe_fix_one_based_node_indexing(t_raw)

        # Normalize / bin timestamps into discrete time steps.
        # This is important because higher-order models for time-respecting paths use time differences.
        if hasattr(t_raw, "data") and hasattr(t_raw.data, "time") and t_raw.data.time is not None:
            # shift to 0
            try:
                t_raw.data.time = t_raw.data.time - t_raw.data.time.min()
            except Exception:
                pass
            if time_bin is not None:
                # floor-divide into bins (e.g., 20s intervals -> steps)
                t_raw.data.time = (t_raw.data.time // int(time_bin)).to(torch.long)

        t = _safe_to_device(t_raw, device)

        # Build higher-order model (De Bruijn graphs) from temporal paths
        m = pp.MultiOrderModel.from_temporal_graph(t, max_order=int(max_order))

        # Some netzschleuder datasets still propagate 1-based indexing into the
        # layer graphs. Fix those layers before exporting to PyG.
        for k in range(1, int(max_order) + 1):
            try:
                _maybe_fix_one_based_node_indexing(m.layers[k])
            except Exception:
                pass

        g = m.layers[1]
        g2 = m.layers[2]
        data = m.to_dbgnn_data(max_order=int(max_order), mapping="last")

        # Labels from node attribute on the temporal graph
        key = _resolve_pyg_key(t.data, target_attr_res)
        y_raw = getattr(t.data, key)

    else:
        # Static graph
        g0 = _read_netzschleuder_graph_compat(
            name=record,
            network=network_param,
            time_attr=None,
            base_url=base_url,
        )
        if isinstance(g0, pp.TemporalGraph):
            # This can happen if the record is temporal but user forgot time_attr.
            raise ValueError(
                f"Record {record!r} appears to be temporal, but no time_attr was provided. "
                "Please pass dataset_kwargs={'time_attr': '<edge time attribute>'}."
            )

        _maybe_fix_one_based_node_indexing(g0)

        t = _safe_to_device(g0, device)

        # Build path data from 2-step walks and fit a higher-order model
        paths = _build_static_path_data_from_graph(
            t,
            device=device,
            max_order=int(max_order),
            mode=static_mode,
            max_paths=static_max_paths_int,
            seed=seed,
        )
        m = pp.MultiOrderModel.from_path_data(paths, max_order=int(max_order))
        m = _safe_to_device(m, device)

        # Some netzschleuder datasets still propagate 1-based indexing into the
        # layer graphs. Fix those layers before exporting to PyG.
        for k in range(1, int(max_order) + 1):
            try:
                _maybe_fix_one_based_node_indexing(m.layers[k])
            except Exception:
                pass

        g = m.layers[1]
        g2 = m.layers[2]
        data = m.to_dbgnn_data(max_order=int(max_order), mapping="last")

        # Labels from node attribute on the static graph
        key = _resolve_pyg_key(t.data, target_attr_res)
        y_raw = getattr(t.data, key)

    # Optional CSV override for copenhagen/sms `female` labels.
    y_csv_override = _maybe_override_copenhagen_sms_labels_from_csv(
        record=str(record),
        network_param=network_param,
        target_attr_res=target_attr_res,
        t_data=t.data,
        csv_path=gender_csv_path,
    )
    if y_csv_override is not None:
        y_raw = y_csv_override

    # ------------------------------------------------------------------
    # Labels + split
    # ------------------------------------------------------------------
    # Special case: allow -1 as "unlabeled" (used by CSV overrides when node
    # metadata is missing). We encode only labeled nodes and keep unlabeled as -1.
    if torch.is_tensor(y_raw):
        y_raw_t = y_raw.detach().cpu()
        unlabeled_mask = y_raw_t < 0
        if bool(unlabeled_mask.any()):
            labeled_mask = ~unlabeled_mask
            if int(labeled_mask.sum().item()) == 0:
                raise ValueError("All nodes are unlabeled after CSV override; cannot build a classification task.")
            y = torch.full((int(y_raw_t.numel()),), -1, dtype=torch.long)
            y_labeled, le = _encode_labels(y_raw_t[labeled_mask])
            y[labeled_mask] = y_labeled
        else:
            y, le = _encode_labels(y_raw_t)
    else:
        y, le = _encode_labels(y_raw)

    if y.numel() != int(data.num_nodes):
        raise ValueError(
            f"Label length mismatch: got {y.numel()} labels from attribute {key!r}, but data.num_nodes={int(data.num_nodes)}. "
            "This likely means the node attribute is not aligned with the node mapping used by the graph."
        )
    data.y = y.to(device)

    # Split
    if seed is not None:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed))
        try:
            data = RandomNodeSplit(num_val=0, num_test=num_test)(data, generator=gen)
        except TypeError:
            data = RandomNodeSplit(num_val=0, num_test=num_test)(data)
    else:
        data = RandomNodeSplit(num_val=0, num_test=num_test)(data)

    # Exclude unlabeled nodes from all supervision masks.
    if bool((data.y < 0).any()):
        unlabeled = data.y < 0
        for mask_name in ("train_mask", "val_mask", "test_mask"):
            if hasattr(data, mask_name):
                m = getattr(data, mask_name)
                if torch.is_tensor(m) and int(m.size(0)) == int(data.y.size(0)):
                    setattr(data, mask_name, m & (~unlabeled))

    if hasattr(data, "to"):
        data = data.to(device)

    # Attach (u,v,w) triples aligned with higher-order edges.
    attach_ho_triples_to_data(data, g=g, g2=g2)

    return (
        data,
        NetzschleuderAssets(
            record=record,
            network=network_param,
            time_attr=time_attr_res,
            target_attr=target_attr_res,
            label_encoder=le,
            t=t,
            m=m,
            g=g,
            g2=g2,
        ),
    )


# ---------------------------------------------------------------------
# Convenience wrappers registered in the dataset registry
# ---------------------------------------------------------------------


def load_copenhagen(*, device: torch.device, num_test: float = 0.3, seed: Optional[int] = None, **kwargs):
    return load_netzschleuder(device=device, record="copenhagen", num_test=num_test, seed=seed, **kwargs)


def load_sp_colocation(*, device: torch.device, num_test: float = 0.3, seed: Optional[int] = None, **kwargs):
    return load_netzschleuder(device=device, record="sp_colocation", num_test=num_test, seed=seed, **kwargs)


def load_sp_high_school(*, device: torch.device, num_test: float = 0.3, seed: Optional[int] = None, **kwargs):
    return load_netzschleuder(device=device, record="sp_high_school", num_test=num_test, seed=seed, **kwargs)


def load_sp_high_school_new(*, device: torch.device, num_test: float = 0.3, seed: Optional[int] = None, **kwargs):
    return load_netzschleuder(device=device, record="sp_high_school_new", num_test=num_test, seed=seed, **kwargs)


def load_sp_hospital(*, device: torch.device, num_test: float = 0.3, seed: Optional[int] = None, **kwargs):
    return load_netzschleuder(device=device, record="sp_hospital", num_test=num_test, seed=seed, **kwargs)


def load_sp_primary_school(*, device: torch.device, num_test: float = 0.3, seed: Optional[int] = None, **kwargs):
    return load_netzschleuder(device=device, record="sp_primary_school", num_test=num_test, seed=seed, **kwargs)


def load_sp_office(*, device: torch.device, num_test: float = 0.3, seed: Optional[int] = None, **kwargs):
    return load_netzschleuder(device=device, record="sp_office", num_test=num_test, seed=seed, **kwargs)

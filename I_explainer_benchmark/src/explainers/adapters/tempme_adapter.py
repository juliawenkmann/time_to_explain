from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from collections import deque
from contextlib import contextmanager
import inspect
import importlib
import importlib.util
import os
import shutil
import random
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch

from ...core.types import BaseExplainer, ExplanationContext, ExplanationResult

try:  # pragma: no cover - platform dependent
    import fcntl
except Exception:  # pragma: no cover - platform dependent
    fcntl = None


# --------- small utilities ---------

def _event_idx_from_target(target: Dict[str, Any]) -> int:
    eidx = target.get("event_idx") or target.get("index") or target.get("idx")
    if eidx is None:
        raise ValueError("context.target must contain one of: 'event_idx', 'index', 'idx'.")
    return int(eidx)


def _infer_device(cfg_device: Optional[Union[str, torch.device]], model: Any) -> torch.device:
    if cfg_device is not None:
        return torch.device(cfg_device)
    try:
        p = next(model.parameters())
        return p.device
    except Exception:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _resolve_tempme_runtime_device(device: torch.device) -> torch.device:
    """Return a device that can run TempME's torch_scatter ops.

    TempME relies on ``torch_scatter.scatter`` during edge importance projection.
    In many environments, torch_scatter is available for CPU (and sometimes CUDA)
    but not for MPS. If an unsupported device is selected, we fall back to CPU
    to avoid runtime failures in explain().
    """
    dev = torch.device(device)
    if dev.type == "cpu":
        return dev

    # MPS has no torch_scatter kernel support in typical builds.
    if dev.type == "mps":
        print("[TempMEAdapter] MPS selected but torch_scatter is unsupported on MPS; falling back to CPU.")
        return torch.device("cpu")

    # Probe scatter support on the requested accelerator (e.g., CUDA).
    try:
        from torch_scatter import scatter_max  # type: ignore

        src = torch.tensor([1.0], device=dev)
        index = torch.tensor([0], dtype=torch.long, device=dev)
        _ = scatter_max(src, index, dim=0, dim_size=1)[0]
        return dev
    except Exception as exc:  # pragma: no cover - device/environment dependent
        print(
            f"[TempMEAdapter] torch_scatter failed on device={dev}; "
            f"falling back to CPU. ({exc})"
        )
        return torch.device("cpu")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _asset_root() -> Path:
    repo = _repo_root()
    bench = repo / "I_explainer_benchmark"
    return bench if bench.is_dir() else repo


def _submodules_root() -> Path:
    return _asset_root() / "submodules"


def _resolve_tempme_root(explicit: Optional[str]) -> Path:
    """Return the directory that contains TempME's `models/` and `utils/`."""

    def _is_root(p: Path) -> bool:
        return (p / "models" / "explainer.py").exists() and (p / "utils" / "graph.py").exists()

    if explicit:
        p = Path(explicit).expanduser().resolve()
        if _is_root(p):
            return p
        # common: explicit points to a container dir that has `TempME/` inside
        if _is_root(p / "TempME"):
            return (p / "TempME").resolve()
        raise FileNotFoundError(
            f"TempME vendor_dir={p} does not look like the TempME source root (expected models/explainer.py + utils/graph.py)."
        )

    base = _submodules_root() / "explainer"
    candidates = [
        base / "tempme",  # requested location
        base / "TempME",
        base / "tempme" / "TempME",
        base / "TempME" / "TempME",
    ]
    for c in candidates:
        if _is_root(c):
            return c.resolve()
    raise FileNotFoundError(
        "Could not locate TempME source. Tried:\n" + "\n".join(f" - {c}" for c in candidates)
    )


def _resolve_checkpoint_root() -> Path:
    """Default checkpoint root under resources/explainer/tempme."""
    return _asset_root() / "resources" / "explainer" / "tempme"


def _ensure_tempme_params_root(vendor_root: Path) -> None:
    """Ensure TempME's ``params`` path is a usable directory.

    Some setups use a symlink (e.g. ``submodules/explainer/tempme/params``) that
    can become broken. TempME training scripts write checkpoints under
    ``params/tgnn`` and ``params/explainer``; if the symlink target directory does
    not exist, scripts fail with FileNotFoundError. We create the target path
    proactively.
    """
    params_path = vendor_root / "params"
    if params_path.is_symlink():
        link = os.readlink(str(params_path))
        target = Path(link)
        if not target.is_absolute():
            target = (params_path.parent / target)
        target.mkdir(parents=True, exist_ok=True)
        return

    if params_path.exists():
        if params_path.is_file():
            raise NotADirectoryError(f"TempME params path is a file, expected directory: {params_path}")
        return

    params_path.mkdir(parents=True, exist_ok=True)


def _resolved_link_target(path: Path) -> Path:
    link = os.readlink(str(path))
    target = Path(link)
    if not target.is_absolute():
        target = path.parent / target
    return target


def _processed_required_files(*, data: str) -> Tuple[str, ...]:
    return (
        f"{data}_train_cat.h5",
        f"{data}_test_cat.h5",
        f"{data}_train_edge.npy",
        f"{data}_test_edge.npy",
        f"ml_{data}.csv",
        f"ml_{data}.npy",
        f"ml_{data}_node.npy",
    )


def _has_processed_dataset(root: Path, *, data: str) -> bool:
    req = _processed_required_files(data=data)
    return root.exists() and root.is_dir() and all((root / name).exists() for name in req)


def _ensure_tempme_processed_root(vendor_root: Path, *, data: str) -> None:
    """Ensure TempME scripts can read processed *.h5 files for the selected dataset."""
    processed_path = vendor_root / "processed"
    target_dir = _resolved_link_target(processed_path) if processed_path.is_symlink() else processed_path

    if _has_processed_dataset(target_dir, data=data):
        return

    candidates: List[Path] = [
        _asset_root() / "resources" / "data" / str(data) / "processed",
        _asset_root() / "resources" / "datasets" / "processed",
    ]
    candidates.extend(sorted(vendor_root.glob("processed_legacy_backup_*"), reverse=True))
    candidates.extend(sorted(vendor_root.parent.glob("processed_legacy_backup_*"), reverse=True))

    source_dir = next((p for p in candidates if _has_processed_dataset(p, data=data)), None)
    if source_dir is None:
        looked = "\n".join(f" - {p}" for p in candidates)
        req = ", ".join(_processed_required_files(data=data))
        raise FileNotFoundError(
            f"TempME processed files for data='{data}' are missing. "
            f"Expected [{req}] under {target_dir}.\nSearched:\n{looked}"
        )

    target_dir.mkdir(parents=True, exist_ok=True)
    for name in _processed_required_files(data=data):
        dst = target_dir / name
        if dst.exists():
            continue
        src = source_dir / name
        try:
            os.link(str(src), str(dst))
        except Exception:
            shutil.copy2(str(src), str(dst))


def _as_existing_file(path: Optional[Union[str, Path]]) -> Optional[Path]:
    if path is None:
        return None
    p = Path(path).expanduser()
    return p if p.exists() and p.is_file() else None


def _copy_checkpoint_if_needed(src: Path, dst: Path) -> None:
    """Copy an existing checkpoint into ``dst`` unless it is already there."""
    if src.resolve() == dst.resolve():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(str(src), str(dst))
    except Exception:
        shutil.copy2(str(src), str(dst))


@contextmanager
def _exclusive_file_lock(path: Path):
    """Best-effort process lock to avoid duplicate TempME training runs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, "a+")
    try:
        if fcntl is not None:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        if fcntl is not None:
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass
        fh.close()


class _SysPathGuard:
    """Temporarily prepend a path to sys.path."""

    def __init__(self, path: Path) -> None:
        self._path = str(path)
        self._added = False

    def __enter__(self) -> None:
        if self._path not in sys.path:
            sys.path.insert(0, self._path)
            self._added = True

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._added:
            try:
                sys.path.remove(self._path)
            except ValueError:
                pass


def _load_module_from_path(module_name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# TempME anonymous pattern -> category id (0..11)
_ANONY_PATTERNS: Tuple[Tuple[int, int, int], ...] = (
    (1, 2, 0), (1, 2, 1), (1, 2, 3), (1, 2, 2),
    (1, 3, 0), (1, 3, 1), (1, 3, 3), (1, 3, 2),
    (1, 1, 0), (1, 1, 1), (1, 1, 2), (1, 1, 3),
)
_PATTERN_TO_CAT: Dict[Tuple[int, int, int], int] = {p: i for i, p in enumerate(_ANONY_PATTERNS)}


def _anony_to_cat_feat(out_anony: np.ndarray) -> np.ndarray:
    """Map out_anony [B, N, 3] -> cat_feat [B, N, 1] with ids in [0,11]."""
    b, n, _ = out_anony.shape
    cat = np.zeros((b, n, 1), dtype=np.int64)
    flat = out_anony.reshape(-1, 3)
    for j, row in enumerate(flat):
        cat.reshape(-1)[j] = _PATTERN_TO_CAT.get(tuple(int(x) for x in row), 0)
    return cat


def _edge_identity(edge_idx: np.ndarray) -> np.ndarray:
    """Edge-identity tensor used by TempME: [B, N, 3, 3] for 3 edges per walk."""
    b, n, l = edge_idx.shape
    if l != 3:
        raise ValueError(f"TempME expects 3 edges per walk, got edge_idx.shape={edge_idx.shape}")
    out = np.zeros((b, n, l, l), dtype=np.float32)
    for i in range(l):
        for j in range(l):
            eq = (edge_idx[:, :, i] == edge_idx[:, :, j]).astype(np.float32)
            eq *= (edge_idx[:, :, i] != 0).astype(np.float32)
            out[:, :, i, j] = eq
    return out


def _assemble_walks(walks_raw: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert TempME NeighborFinder.find_k_walks output into explainer input tuple."""
    node_idx, edge_idx, time_idx, out_anony = walks_raw
    cat_feat = _anony_to_cat_feat(out_anony)
    marginal = np.zeros((cat_feat.shape[0], cat_feat.shape[1], 1), dtype=np.float32)
    return node_idx.astype(np.int64), edge_idx.astype(np.int64), time_idx.astype(np.float32), cat_feat.astype(np.int64), marginal


def _event_id_array(events: Any) -> np.ndarray:
    cols = getattr(events, "columns", None)
    if cols is not None and "e_idx" in cols:
        return events["e_idx"].to_numpy(dtype=np.int64)
    if cols is not None and "idx" in cols:
        return events["idx"].to_numpy(dtype=np.int64)
    return np.arange(1, len(events) + 1, dtype=np.int64)


def _event_src_dst_ts_arrays(events: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    cols = getattr(events, "columns", None)
    if cols is not None and "u" in cols:
        return (
            events["u"].to_numpy(dtype=np.int64),
            events["i"].to_numpy(dtype=np.int64),
            events["ts"].to_numpy(dtype=np.float64),
        )
    if hasattr(events, "iloc"):
        return (
            np.asarray(events.iloc[:, 0], dtype=np.int64),
            np.asarray(events.iloc[:, 1], dtype=np.int64),
            np.asarray(events.iloc[:, 2], dtype=np.float64),
        )
    src = np.asarray([item["u"] for item in events], dtype=np.int64)
    dst = np.asarray([item["i"] for item in events], dtype=np.int64)
    ts = np.asarray([item["ts"] for item in events], dtype=np.float64)
    return src, dst, ts


def _event_uvte_from_row(events: Any, row: int) -> Tuple[int, int, float, int]:
    cols = getattr(events, "columns", None)
    if hasattr(events, "iloc"):
        if cols is not None and "u" in cols:
            item = events.iloc[row]
            u = int(item["u"])
            v = int(item["i"])
            ts = float(item["ts"])
        else:
            u = int(events.iloc[row, 0])
            v = int(events.iloc[row, 1])
            ts = float(events.iloc[row, 2])
    else:
        item = events[row]
        u = int(item["u"])
        v = int(item["i"])
        ts = float(item["ts"])

    eids = _event_id_array(events)
    eid = int(eids[row]) if 0 <= row < len(eids) else int(row + 1)
    return u, v, ts, eid


def _flatten_edge_imp(
    *,
    subgraph: Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]],
    edge_imp_0: torch.Tensor,
    edge_imp_1: Optional[torch.Tensor],
    include_hop1: bool,
) -> Dict[int, float]:
    """Turn TempME per-position edge importance into a per-edge-id map."""
    _, eidx_records, _ = subgraph

    e0 = np.asarray(eidx_records[0], dtype=np.int64).reshape(-1)
    imp0 = edge_imp_0.detach().cpu().numpy().reshape(-1)

    out: Dict[int, float] = {}

    for eid, val in zip(e0.tolist(), imp0.tolist()):
        if eid <= 0:
            continue
        prev = out.get(int(eid))
        if prev is None or float(val) > prev:
            out[int(eid)] = float(val)

    if include_hop1 and edge_imp_1 is not None and len(eidx_records) > 1:
        e1 = np.asarray(eidx_records[1], dtype=np.int64).reshape(-1)
        imp1 = edge_imp_1.detach().cpu().numpy().reshape(-1)
        for eid, val in zip(e1.tolist(), imp1.tolist()):
            if eid <= 0:
                continue
            prev = out.get(int(eid))
            if prev is None or float(val) > prev:
                out[int(eid)] = float(val)

    return out


@dataclass
class TempMEAdapterConfig:
    """TempME adapter configuration.

    base_type and data should match TempME's original flags:
      base_type: "tgn" | "graphmixer" | "tgat" (tgat support depends on your TempME fork)
      data: dataset name ("wikipedia", "reddit", ...)

    Training args:
      - train_args: shared CLI args passed to both TempME scripts.
      - train_args_base: args only for ``learn_base.py`` (override shared values).
      - train_args_explainer: args only for ``temp_exp_main.py`` (override shared values).
    """

    base_type: str
    data: str

    # Checkpoint handling
    checkpoint_path: Optional[str] = None
    train_if_missing: bool = True
    # Keep backward-compatible behavior: if no explainer/{base_type}/{data}.pt exists,
    # allow fallback to explainer/tgat/{data}.pt.
    allow_tgat_fallback_for_explainer: bool = True
    train_args: Dict[str, Any] = field(default_factory=dict)
    train_args_base: Dict[str, Any] = field(default_factory=dict)
    train_args_explainer: Dict[str, Any] = field(default_factory=dict)

    # TempME sampling hyperparams
    n_degree: int = 20
    walk_num_neighbors: int = 1
    # When external candidate extraction is disjoint from TempME edge ids,
    # use TempME's native scored edges to avoid all-zero projections.
    fallback_to_native_on_no_overlap: bool = True

    # Plumbing
    vendor_dir: Optional[str] = None
    cache: bool = True
    alias: Optional[str] = None
    device: Optional[Union[str, torch.device]] = None
    seed: Optional[int] = None


class TempMEAdapter(BaseExplainer):
    """Adapter that runs the *original* TempME explainer (torch checkpoint) inside your unified API.

    Notes:
      - We import TempME from `submodules/explainer/tempme` (or `vendor_dir`).
      - The saved checkpoint must be created via TempME's `torch.save(explainer, ...)`.
      - If `context.subgraph.payload['candidate_eidx']` exists, we align the returned
        importance vector to that edge-id order.

    Training:
      - If `train_if_missing=True` and the explainer checkpoint isn't found, we run
        TempME's original scripts (`learn_base.py` then `temp_exp_main.py`) via subprocess.
        This is the closest way to "exactly recreate" TempME training without forking its logic.
    """

    def __init__(self, cfg: TempMEAdapterConfig) -> None:
        super().__init__(name="tempme", alias=cfg.alias or "tempme")
        self.cfg = cfg

        self._vendor_root: Optional[Path] = None
        self._NeighborFinder = None  # filled in prepare

        self._events = None
        self._row_by_eid: Dict[int, int] = {}
        self._default_eids: Optional[np.ndarray] = None

        self.device: Optional[torch.device] = None
        self._ngh_finder = None
        self._explainer: Optional[torch.nn.Module] = None
        self._explainer_forward_style: Optional[str] = None
        self._prepared = False
        self._warned_cross_base_ckpt = False

        self._cache: Dict[int, Dict[str, Any]] = {}

    # ----- lifecycle -----

    def prepare(self, *, model: Any, dataset: Any) -> None:
        super().prepare(model=model, dataset=dataset)

        self.device = _resolve_tempme_runtime_device(_infer_device(self.cfg.device, model))

        # Events table
        events = dataset["events"] if isinstance(dataset, dict) and "events" in dataset else dataset
        if isinstance(dataset, dict) and "dataset_name" in dataset:
            # keep config in sync with the pipeline's dataset naming
            self.cfg.data = str(dataset["dataset_name"])
        self._events = events

        # Resolve TempME source + import it
        self._vendor_root = _resolve_tempme_root(self.cfg.vendor_dir)
        _ensure_tempme_params_root(self._vendor_root)
        with _SysPathGuard(self._vendor_root):
            # Ensure pickled class refs resolve during torch.load
            if "utils" in sys.modules:
                del sys.modules["utils"]
            importlib.import_module("models.explainer")
            graph_path = self._vendor_root / "utils" / "graph.py"
            graph_mod = _load_module_from_path("_tempme_utils_graph", graph_path)
            self._NeighborFinder = getattr(graph_mod, "NeighborFinder")

        # Build eid -> row index mapping
        self._build_row_index(events)

        # Build TempME NeighborFinder for sampling
        self._ngh_finder = self._build_neighbor_finder(events)

        # Load (or train+load) explainer
        ckpt_path = self._resolve_explainer_checkpoint_path()
        if ckpt_path is None or not Path(ckpt_path).exists():
            if not self.cfg.train_if_missing:
                raise FileNotFoundError(self._missing_ckpt_message(ckpt_path))
            self._run_tempme_training()
            ckpt_path = self._resolve_explainer_checkpoint_path()
            if ckpt_path is None or not Path(ckpt_path).exists():
                raise FileNotFoundError(self._missing_ckpt_message(ckpt_path))

        self._explainer = self._load_explainer(Path(ckpt_path))
        self._explainer.eval()
        self._explainer_forward_style = self._detect_explainer_forward_style()

        self._prepared = True

    def explain(self, context: ExplanationContext) -> ExplanationResult:
        if not self._prepared:
            raise RuntimeError("TempMEAdapter.prepare(model=..., dataset=...) must be called first.")
        assert self._events is not None and self._ngh_finder is not None and self._explainer is not None
        assert self.device is not None

        event_idx = _event_idx_from_target(context.target)

        if self.cfg.cache and event_idx in self._cache:
            cached = self._cache[event_idx]
            return ExplanationResult(
                run_id=context.run_id,
                explainer=self.alias,
                context_fp=context.fingerprint(),
                importance_edges=list(cached.get("importance_edges") or []),
                importance_nodes=None,
                importance_time=None,
                elapsed_sec=float(cached.get("elapsed_sec", 0.0)),
                extras=dict(cached.get("extras") or {}),
            )

        # Optional determinism
        if self.cfg.seed is not None:
            seed = int(self.cfg.seed)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        t0 = time.time()

        # Resolve anchor row + (u,v,ts,eid)
        row = self._resolve_row(event_idx)
        u, v, ts, eid = self._event_uvte(row)

        # Candidates (edge ids) to align output
        payload = getattr(context.subgraph, "payload", {}) or {}
        candidates = payload.get("candidate_eidx")
        if candidates is not None:
            candidate_eids = [int(x) for x in candidates]
            candidate_mode = "extractor"
        else:
            candidate_eids = None
            candidate_mode = "native_no_extractor"

        importance_map = self._explain_uvt(u=u, v=v, ts=ts, eid=eid)

        overlap_count: Optional[int] = None
        if candidate_eids is not None and importance_map:
            overlap_count = int(len(set(candidate_eids).intersection(importance_map.keys())))
            if overlap_count == 0 and bool(self.cfg.fallback_to_native_on_no_overlap):
                candidate_eids = sorted(importance_map.keys())
                candidate_mode = "native_fallback_no_overlap"

        if candidate_eids is None:
            # fall back to a stable ordering for downstream metrics
            candidate_eids = sorted(importance_map.keys())
            candidate_mode = "native_no_candidates"

        edge_importance = [float(importance_map.get(int(e), 0.0)) for e in candidate_eids]

        extras = {
            "event_idx": event_idx,
            "u": u,
            "i": v,
            "ts": ts,
            "eid": eid,
            "candidate_eidx": candidate_eids,
            "candidate_mode": candidate_mode,
            "candidate_overlap_count": overlap_count,
            "runtime_ms": (time.time() - t0) * 1000.0,
        }

        # Prefer edge_index already attached by your candidate extractor; otherwise provide (u,v) pairs if possible.
        edge_index = getattr(context.subgraph, "edge_index", None)
        if not edge_index and payload.get("candidate_edge_index"):
            edge_index = payload["candidate_edge_index"]
        if edge_index is not None:
            edge_index_list = edge_index
            if not isinstance(edge_index_list, list):
                try:
                    edge_index_list = edge_index_list.tolist()
                except Exception:
                    edge_index_list = list(edge_index_list)
            extras["candidate_edge_index"] = edge_index_list

        result = ExplanationResult(
            run_id=context.run_id,
            explainer=self.alias,
            context_fp=context.fingerprint(),
            importance_edges=edge_importance,
            importance_nodes=None,
            importance_time=None,
            extras=extras,
        )

        if self.cfg.cache:
            self._cache[event_idx] = {
                "importance_edges": list(result.importance_edges or []),
                "elapsed_sec": float(result.elapsed_sec),
                "extras": dict(result.extras or {}),
            }

        return result

    # ----- internals -----

    def _build_row_index(self, events: Any) -> None:
        eids = _event_id_array(events)
        self._default_eids = eids
        self._row_by_eid = {int(e): int(r) for r, e in enumerate(eids.tolist())}

    def _resolve_row(self, event_idx: int) -> int:
        # First try: treat event_idx as a model edge-id
        if event_idx in self._row_by_eid:
            return int(self._row_by_eid[event_idx])

        n = len(self._events)
        # fallback: treat it as 0-based row
        if 0 <= event_idx < n:
            return int(event_idx)
        # fallback: treat it as 1-based row
        if 1 <= event_idx <= n:
            return int(event_idx - 1)

        raise ValueError(f"event_idx={event_idx} not found (N={n}).")

    def _event_uvte(self, row: int) -> Tuple[int, int, float, int]:
        return _event_uvte_from_row(self._events, row)

    def _build_neighbor_finder(self, events: Any):
        assert self._NeighborFinder is not None

        src, dst, ts = _event_src_dst_ts_arrays(events)
        eids = _event_id_array(events)

        max_node = int(max(int(src.max()), int(dst.max())))
        adj_list: List[List[Tuple[int, int, float]]] = [[] for _ in range(max_node + 1)]

        # Undirected adjacency, as in TempME scripts
        for s, d, eid, t in zip(src.tolist(), dst.tolist(), eids.tolist(), ts.tolist()):
            adj_list[int(s)].append((int(d), int(eid), float(t)))
            adj_list[int(d)].append((int(s), int(eid), float(t)))

        ngh_finder = self._NeighborFinder(adj_list)
        if not (hasattr(ngh_finder, "find_k_hop") and hasattr(ngh_finder, "find_k_walks")):
            # Fall back to the full TempME-style NeighborFinder when a slim class is loaded.
            from ...models.neighborfinder import NeighborFinder as FallbackNeighborFinder
            ngh_finder = FallbackNeighborFinder(adj_list)
        return ngh_finder

    def _resolve_checkpoint_path(self, *, kind: str) -> Optional[str]:
        if self._vendor_root is None:
            return None
        base_type = str(self.cfg.base_type)
        data = str(self.cfg.data)
        explicit = _as_existing_file(self.cfg.checkpoint_path) if kind == "explainer" else None
        if explicit is not None:
            self._maybe_warn_cross_base_checkpoint(ckpt_path=explicit, base_type=base_type, data=data)
            return str(explicit)

        cands = self._checkpoint_candidates(kind=kind, base_type=base_type, data=data)
        for path in cands:
            if path.exists():
                if kind == "explainer":
                    self._maybe_warn_cross_base_checkpoint(ckpt_path=path, base_type=base_type, data=data)
                return str(path)
        if kind == "explainer" and self.cfg.checkpoint_path:
            return str(Path(self.cfg.checkpoint_path).expanduser())
        return str(cands[0]) if cands else None

    def _resolve_explainer_checkpoint_path(self) -> Optional[str]:
        return self._resolve_checkpoint_path(kind="explainer")

    def _resolve_base_checkpoint_path(self) -> Optional[str]:
        return self._resolve_checkpoint_path(kind="base")

    def _checkpoint_candidates(self, *, kind: str, base_type: str, data: str) -> List[Path]:
        assert self._vendor_root is not None
        if kind == "base":
            relatives = [Path("tgnn") / f"{base_type}_{data}.pt"]
        elif kind == "explainer":
            relatives = [Path("explainer") / base_type / f"{data}.pt"]
            if bool(getattr(self.cfg, "allow_tgat_fallback_for_explainer", True)):
                relatives.append(Path("explainer") / "tgat" / f"{data}.pt")
        else:
            raise ValueError(f"Unknown checkpoint kind: {kind}")

        backup_roots = sorted(self._vendor_root.glob("params_legacy_backup_*"), reverse=True)
        backup_roots += sorted(self._vendor_root.parent.glob("params_legacy_backup_*"), reverse=True)
        roots = [
            _resolve_checkpoint_root() / "params",
            self._vendor_root / "params",
            self._vendor_root.parent / "params",
            *backup_roots,
        ]

        seen: set[str] = set()
        out: List[Path] = []
        for relative in relatives:
            for root in roots:
                path = root / relative
                key = str(path)
                if key in seen:
                    continue
                out.append(path)
                seen.add(key)
        return out

    def _maybe_warn_cross_base_checkpoint(self, *, ckpt_path: Path, base_type: str, data: str) -> None:
        if self._warned_cross_base_ckpt:
            return
        base = str(base_type).lower()
        if base == "tgat":
            return
        norm = ckpt_path.as_posix().lower()
        if "/explainer/tgat/" not in norm:
            return
        print(
            "[TempMEAdapter] WARNING: using cross-base explainer checkpoint "
            f"'{ckpt_path}' for base_type='{base_type}', data='{data}'. "
            "This is a fallback to TGAT-formatted TempME weights."
        )
        self._warned_cross_base_ckpt = True

    def _vendor_checkpoint_path(self, *, kind: str) -> Path:
        assert self._vendor_root is not None
        if kind == "base":
            return self._vendor_root / "params" / "tgnn" / f"{self.cfg.base_type}_{self.cfg.data}.pt"
        if kind == "explainer":
            return self._vendor_root / "params" / "explainer" / str(self.cfg.base_type) / f"{self.cfg.data}.pt"
        raise ValueError(f"Unknown checkpoint kind: {kind}")

    def _sync_checkpoint_to_vendor_if_needed(self, *, kind: str) -> Optional[Path]:
        if self._vendor_root is None:
            return None
        expected = self._vendor_checkpoint_path(kind=kind)
        found = _as_existing_file(self._resolve_checkpoint_path(kind=kind))
        if found is None:
            return None
        _copy_checkpoint_if_needed(found, expected)
        return expected if expected.exists() else found

    def _missing_ckpt_message(self, ckpt_path: Optional[str]) -> str:
        vendor = str(self._vendor_root) if self._vendor_root is not None else "<unresolved>"
        return (
            "TempME explainer checkpoint not found.\n"
            f"Looked for: {ckpt_path}\n\n"
            "To train it with the original TempME scripts, run (from repo root):\n"
            f"  python {vendor}/learn_base.py --base_type {self.cfg.base_type} --data {self.cfg.data}\n"
            f"  python {vendor}/temp_exp_main.py --base_type {self.cfg.base_type} --data {self.cfg.data}\n"
            "\nOr pass cfg.checkpoint_path explicitly."
        )

    def _load_explainer(self, ckpt_path: Path) -> torch.nn.Module:
        assert self._vendor_root is not None
        with _SysPathGuard(self._vendor_root):
            # ensure modules are importable for torch.load
            importlib.import_module("models.explainer")
            explainer = torch.load(str(ckpt_path), map_location="cpu")
        if hasattr(explainer, "device"):
            explainer.device = self.device
        for mod in explainer.modules():
            if hasattr(mod, "device"):
                mod.device = self.device
        explainer = explainer.to(self.device)
        return explainer

    def _run_tempme_training(self) -> None:
        """Run TempME's original training scripts via subprocess (base model -> explainer)."""
        assert self._vendor_root is not None

        lock_name = f".tempme_train_{self.cfg.base_type}_{self.cfg.data}.lock"
        lock_path = self._vendor_root / "params" / lock_name
        with _exclusive_file_lock(lock_path):
            # Another process may have produced checkpoints while we waited.
            expl_ckpt = self._sync_checkpoint_to_vendor_if_needed(kind="explainer")
            if expl_ckpt is not None and expl_ckpt.exists():
                return

            _ensure_tempme_processed_root(self._vendor_root, data=str(self.cfg.data))

            base_ckpt = self._sync_checkpoint_to_vendor_if_needed(kind="base")
            if base_ckpt is None or not base_ckpt.exists():
                self._run_tempme_script(
                    "learn_base.py",
                    ["--base_type", str(self.cfg.base_type), "--data", str(self.cfg.data)],
                    extra_args=self.cfg.train_args_base,
                )
                # Best-effort post-sync for future runs.
                _ = self._sync_checkpoint_to_vendor_if_needed(kind="base")

            # Train explainer only if still missing.
            expl_ckpt = self._sync_checkpoint_to_vendor_if_needed(kind="explainer")
            if expl_ckpt is None or not expl_ckpt.exists():
                self._run_tempme_script(
                    "temp_exp_main.py",
                    ["--base_type", str(self.cfg.base_type), "--data", str(self.cfg.data)],
                    extra_args=self.cfg.train_args_explainer,
                )
                _ = self._sync_checkpoint_to_vendor_if_needed(kind="explainer")

    def _run_tempme_script(self, script: str, argv: List[str], *, extra_args: Optional[Dict[str, Any]] = None) -> None:
        assert self._vendor_root is not None

        script_path = self._vendor_root / script
        if not script_path.exists():
            raise FileNotFoundError(f"TempME script not found: {script_path}")

        cmd = [sys.executable, str(script_path), *argv]
        # Prefer CUDA when available/selected; otherwise default to CPU.
        default_gpu = -1
        if self.device is not None and self.device.type == "cuda":
            default_gpu = int(self.device.index) if self.device.index is not None else 0
        merged_train_args = {"gpu": default_gpu}
        merged_train_args.update(self.cfg.train_args or {})
        merged_train_args.update(extra_args or {})

        # Allow passing through extra flags via cfg.train_args
        for k, v in merged_train_args.items():
            flag = str(k)
            if not flag.startswith("--"):
                flag = "--" + flag
            if v is None or v is False:
                continue
            if v is True:
                cmd.append(flag)
            else:
                cmd.extend([flag, str(v)])

        env = os.environ.copy()
        mpl_dir = Path(tempfile.gettempdir()) / "tempme_mplconfig"
        mpl_dir.mkdir(parents=True, exist_ok=True)
        env.setdefault("MPLCONFIGDIR", str(mpl_dir))

        tail = deque(maxlen=200)
        proc = subprocess.Popen(
            cmd,
            cwd=str(self._vendor_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            tail.append(line.rstrip("\n"))
        ret = proc.wait()
        if ret != 0:
            last_output = "\n".join(tail) if tail else "<no output captured>"
            raise RuntimeError(
                f"TempME training script failed: {script} (exit {ret})\n"
                f"Command: {' '.join(cmd)}\n"
                f"Last output:\n{last_output}"
            )

    def _detect_explainer_forward_style(self) -> str:
        """Infer which positional signature the loaded TempME checkpoint expects."""
        assert self._explainer is not None
        try:
            params = [
                p
                for p in inspect.signature(self._explainer.forward).parameters.values()
                if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            ]
            names = {p.name.lower() for p in params}
            if {"src_idx_l", "tgt_idx_l"}.issubset(names):
                return "walks_src_cut_tgt"
            if "edge_identify" in names or "edge_ident" in names:
                return "walks_cut_edge_ident"
            if len(params) >= 4:
                return "walks_src_cut_tgt"
        except Exception:
            pass
        return "walks_cut_edge_ident"

    def _call_explainer_forward(
        self,
        *,
        walks: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        cut_ts: np.ndarray,
        edge_ident: np.ndarray,
        src_idx: np.ndarray,
        tgt_idx: np.ndarray,
    ) -> torch.Tensor:
        assert self._explainer is not None
        style = self._explainer_forward_style or self._detect_explainer_forward_style()

        if style == "walks_src_cut_tgt":
            try:
                return self._explainer(walks, src_idx, cut_ts, tgt_idx)
            except TypeError as exc:
                # Fallback for forks where signature probing is unreliable.
                if "edge_ident" in str(exc).lower():
                    self._explainer_forward_style = "walks_cut_edge_ident"
                    return self._explainer(walks, cut_ts, edge_ident)
                raise

        try:
            return self._explainer(walks, cut_ts, edge_ident)
        except TypeError as exc:
            if "tgt_idx_l" in str(exc):
                self._explainer_forward_style = "walks_src_cut_tgt"
                return self._explainer(walks, src_idx, cut_ts, tgt_idx)
            raise

    def _explain_uvt(self, *, u: int, v: int, ts: float, eid: int) -> Dict[int, float]:
        """Return a {edge_id -> importance} map for the event (u,v,ts,eid)."""
        assert self._ngh_finder is not None and self._explainer is not None

        src = np.asarray([u], dtype=np.int64)
        dst = np.asarray([v], dtype=np.int64)
        cut_ts = np.asarray([ts], dtype=np.float32)
        e_l = np.asarray([eid], dtype=np.int64)

        # 2-hop subgraphs as in TempME
        sub_src = self._ngh_finder.find_k_hop(2, src, cut_ts, num_neighbors=int(self.cfg.n_degree), e_idx_l=e_l)
        sub_tgt = self._ngh_finder.find_k_hop(2, dst, cut_ts, num_neighbors=int(self.cfg.n_degree), e_idx_l=e_l)

        # k-walks (graphlets)
        walks_src_raw = self._ngh_finder.find_k_walks(int(self.cfg.n_degree), src, num_neighbors=int(self.cfg.walk_num_neighbors), subgraph_src=sub_src)
        walks_tgt_raw = self._ngh_finder.find_k_walks(int(self.cfg.n_degree), dst, num_neighbors=int(self.cfg.walk_num_neighbors), subgraph_src=sub_tgt)

        walks_src = _assemble_walks(walks_src_raw)
        walks_tgt = _assemble_walks(walks_tgt_raw)

        edge_ident_src = _edge_identity(walks_src[1])
        edge_ident_tgt = _edge_identity(walks_tgt[1])

        with torch.no_grad():
            imp_src = self._call_explainer_forward(
                walks=walks_src, cut_ts=cut_ts, edge_ident=edge_ident_src, src_idx=src, tgt_idx=dst
            )
            imp_tgt = self._call_explainer_forward(
                walks=walks_tgt, cut_ts=cut_ts, edge_ident=edge_ident_tgt, src_idx=dst, tgt_idx=src
            )

            base_type = str(getattr(self._explainer, "base_type", self.cfg.base_type))

            # TempME edge aggregation — prefer the exact helper used by the paper implementation.
            if hasattr(self._explainer, "retrieve_edge_imp_node"):
                include_hop1 = (base_type == "tgn")
                edge_imp_src_0, edge_imp_src_1 = self._explainer.retrieve_edge_imp_node(
                    sub_src, imp_src, walks_src, training=False
                )
                edge_imp_tgt_0, edge_imp_tgt_1 = self._explainer.retrieve_edge_imp_node(
                    sub_tgt, imp_tgt, walks_tgt, training=False
                )
            elif hasattr(self._explainer, "retrieve_edge_imp"):
                # Some forks (e.g., TGAT variants) expose `retrieve_edge_imp` instead.
                include_hop1 = True
                edge_src = self._explainer.retrieve_edge_imp(sub_src, imp_src, walks_src, training=False)
                edge_tgt = self._explainer.retrieve_edge_imp(sub_tgt, imp_tgt, walks_tgt, training=False)
                edge_imp_src_0 = edge_src[0]
                edge_imp_tgt_0 = edge_tgt[0]
                edge_imp_src_1 = edge_src[1] if len(edge_src) > 1 else None
                edge_imp_tgt_1 = edge_tgt[1] if len(edge_tgt) > 1 else None
            else:
                raise NotImplementedError(
                    "Loaded TempME explainer does not expose `retrieve_edge_imp_node` or `retrieve_edge_imp`; "
                    "cannot project graphlet importance to edges."
                )

        map_src = _flatten_edge_imp(subgraph=sub_src, edge_imp_0=edge_imp_src_0, edge_imp_1=edge_imp_src_1, include_hop1=include_hop1)
        map_tgt = _flatten_edge_imp(subgraph=sub_tgt, edge_imp_0=edge_imp_tgt_0, edge_imp_1=edge_imp_tgt_1, include_hop1=include_hop1)

        # Merge by max, same edge-id can appear in both endpoint subgraphs
        out = dict(map_src)
        for eid_i, val in map_tgt.items():
            prev = out.get(eid_i)
            if prev is None or val > prev:
                out[eid_i] = val
        return out

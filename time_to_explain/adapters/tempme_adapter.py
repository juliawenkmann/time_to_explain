# time_to_explain/adapters/tempme_adapter.py
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import importlib
import importlib.util
import random
import subprocess
import sys
import time

import numpy as np
import torch

from time_to_explain.core.types import BaseExplainer, ExplanationContext, ExplanationResult


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

    repo_root = Path(__file__).resolve().parents[2]
    base = repo_root / "submodules" / "explainer"
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
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "resources" / "explainer" / "tempme"


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
    """

    base_type: str
    data: str

    # Checkpoint handling
    checkpoint_path: Optional[str] = None
    train_if_missing: bool = True
    train_args: Dict[str, Any] = field(default_factory=dict)

    # TempME sampling hyperparams
    n_degree: int = 20
    walk_num_neighbors: int = 1

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
        self._prepared = False

        self._cache: Dict[int, Dict[str, Any]] = {}

    # ----- lifecycle -----

    def prepare(self, *, model: Any, dataset: Any) -> None:
        super().prepare(model=model, dataset=dataset)

        self.device = _infer_device(self.cfg.device, model)
        if self.cfg.device is None:
            self.device = torch.device("cpu")

        # Events table
        events = dataset["events"] if isinstance(dataset, dict) and "events" in dataset else dataset
        if isinstance(dataset, dict) and "dataset_name" in dataset:
            # keep config in sync with the pipeline's dataset naming
            self.cfg.data = str(dataset["dataset_name"])
        self._events = events

        # Resolve TempME source + import it
        self._vendor_root = _resolve_tempme_root(self.cfg.vendor_dir)
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
        else:
            candidate_eids = None

        importance_map = self._explain_uvt(u=u, v=v, ts=ts, eid=eid)

        if candidate_eids is None:
            # fall back to a stable ordering for downstream metrics
            candidate_eids = sorted(importance_map.keys())

        edge_importance = [float(importance_map.get(int(e), 0.0)) for e in candidate_eids]

        extras = {
            "event_idx": event_idx,
            "u": u,
            "i": v,
            "ts": ts,
            "eid": eid,
            "candidate_eidx": candidate_eids,
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
        if hasattr(events, "columns") and "e_idx" in events.columns:
            eids = events["e_idx"].to_numpy(dtype=np.int64)
        elif hasattr(events, "columns") and "idx" in events.columns:
            eids = events["idx"].to_numpy(dtype=np.int64)
        else:
            eids = np.arange(1, len(events) + 1, dtype=np.int64)

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
        ev = self._events
        if hasattr(ev, "iloc"):
            # DataFrame-like
            if hasattr(ev, "columns") and "u" in ev.columns:
                u = int(ev.iloc[row]["u"])
                v = int(ev.iloc[row]["i"])
                ts = float(ev.iloc[row]["ts"])
            else:
                u = int(ev.iloc[row, 0])
                v = int(ev.iloc[row, 1])
                ts = float(ev.iloc[row, 2])
        else:
            # list/tuple of dicts
            item = ev[row]
            u = int(item["u"])
            v = int(item["i"])
            ts = float(item["ts"])

        # model edge-id
        if hasattr(ev, "columns") and "e_idx" in ev.columns:
            eid = int(ev.iloc[row]["e_idx"])
        elif hasattr(ev, "columns") and "idx" in ev.columns:
            eid = int(ev.iloc[row]["idx"])
        else:
            eid = int(row + 1)
        return u, v, ts, eid

    def _build_neighbor_finder(self, events: Any):
        assert self._NeighborFinder is not None

        if hasattr(events, "columns") and "u" in events.columns:
            src = events["u"].to_numpy(dtype=np.int64)
            dst = events["i"].to_numpy(dtype=np.int64)
            ts = events["ts"].to_numpy(dtype=np.float64)
        else:
            src = np.asarray(events.iloc[:, 0], dtype=np.int64)
            dst = np.asarray(events.iloc[:, 1], dtype=np.int64)
            ts = np.asarray(events.iloc[:, 2], dtype=np.float64)

        if hasattr(events, "columns") and "e_idx" in events.columns:
            eids = events["e_idx"].to_numpy(dtype=np.int64)
        elif hasattr(events, "columns") and "idx" in events.columns:
            eids = events["idx"].to_numpy(dtype=np.int64)
        else:
            eids = np.arange(1, len(events) + 1, dtype=np.int64)

        max_node = int(max(int(src.max()), int(dst.max())))
        adj_list: List[List[Tuple[int, int, float]]] = [[] for _ in range(max_node + 1)]

        # Undirected adjacency, as in TempME scripts
        for s, d, eid, t in zip(src.tolist(), dst.tolist(), eids.tolist(), ts.tolist()):
            adj_list[int(s)].append((int(d), int(eid), float(t)))
            adj_list[int(d)].append((int(s), int(eid), float(t)))

        ngh_finder = self._NeighborFinder(adj_list)
        if not (hasattr(ngh_finder, "find_k_hop") and hasattr(ngh_finder, "find_k_walks")):
            # Fall back to the full TempME-style NeighborFinder when a slim class is loaded.
            from time_to_explain.core.neigbourfinder import NeighborFinder as FallbackNeighborFinder
            ngh_finder = FallbackNeighborFinder(adj_list)
        return ngh_finder

    def _resolve_explainer_checkpoint_path(self) -> Optional[str]:
        if self.cfg.checkpoint_path:
            return str(Path(self.cfg.checkpoint_path).expanduser())

        if self._vendor_root is None:
            return None

        base_type = str(self.cfg.base_type)
        data = str(self.cfg.data)

        ckpt_root = _resolve_checkpoint_root()
        candidates = [
            ckpt_root / "params" / "explainer" / base_type / f"{data}.pt",
            self._vendor_root / "params" / "explainer" / base_type / f"{data}.pt",
            self._vendor_root / "params" / f"explainer/{base_type}" / f"{data}.pt",
            # some forks save TGAT explainers one directory up
            ckpt_root / "params" / "explainer" / "tgat" / f"{data}.pt",
            self._vendor_root.parent / "params" / "explainer" / "tgat" / f"{data}.pt",
            self._vendor_root / "params" / "explainer" / "tgat" / f"{data}.pt",
        ]
        for p in candidates:
            if p.exists():
                return str(p)
        # Return a reasonable default for error messages
        return str(candidates[0])

    def _resolve_base_checkpoint_path(self) -> Optional[str]:
        if self._vendor_root is None:
            return None
        base_type = str(self.cfg.base_type)
        data = str(self.cfg.data)
        ckpt_root = _resolve_checkpoint_root()
        candidates = [
            ckpt_root / "params" / "tgnn" / f"{base_type}_{data}.pt",
            self._vendor_root / "params" / "tgnn" / f"{base_type}_{data}.pt",
            self._vendor_root.parent / "params" / "tgnn" / f"{base_type}_{data}.pt",
        ]
        for p in candidates:
            if p.exists():
                return str(p)
        return str(candidates[0])

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

        base_ckpt = self._resolve_base_checkpoint_path()
        if base_ckpt is None or not Path(base_ckpt).exists():
            self._run_tempme_script(
                "learn_base.py",
                ["--base_type", str(self.cfg.base_type), "--data", str(self.cfg.data)],
            )

        # Train explainer
        self._run_tempme_script(
            "temp_exp_main.py",
            ["--base_type", str(self.cfg.base_type), "--data", str(self.cfg.data)],
        )

    def _run_tempme_script(self, script: str, argv: List[str]) -> None:
        assert self._vendor_root is not None

        script_path = self._vendor_root / script
        if not script_path.exists():
            raise FileNotFoundError(f"TempME script not found: {script_path}")

        cmd = [sys.executable, str(script_path), *argv]
        # Allow passing through extra flags via cfg.train_args
        for k, v in (self.cfg.train_args or {}).items():
            flag = str(k)
            if not flag.startswith("--"):
                flag = "--" + flag
            if v is None or v is False:
                continue
            if v is True:
                cmd.append(flag)
            else:
                cmd.extend([flag, str(v)])

        subprocess.run(cmd, cwd=str(self._vendor_root), check=True)

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
            imp_src = self._explainer(walks_src, cut_ts, edge_ident_src)
            imp_tgt = self._explainer(walks_tgt, cut_ts, edge_ident_tgt)

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

# time_to_explain/adapters/tempme_tg_adapter.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple, Union
from collections import defaultdict
import math
import random
import time
import torch
from time_to_explain.core.types import BaseExplainer, ExplanationContext, ExplanationResult


# ---------------------------------------------------------------------------
# TempME-style adapter configuration
# ---------------------------------------------------------------------------

ScoreFn = Callable[[Any, Any, int, Optional[Sequence[int]]], float]
"""
score_fn(model, dataset, event_idx, active_event_indices) -> scalar score

- model:     your trained TGNN.
- dataset:   the dataset object passed to .prepare().
- event_idx: index of the *target* event to explain.
- active_event_indices:
    * None  -> run the base model on the full computational graph for event_idx.
    * list[int] -> run the base model on a graph where ONLY these historical events
                   (plus the target itself, if your implementation needs that)
                   are considered available.

Return value:
    A scalar score for the target event (e.g. predicted probability or logit of
    the positive class).
"""


LabelFn = Callable[[Any, Any, int], int]
"""
label_fn(model, dataset, event_idx) -> 0 or 1

Optional callback to get the "reference" label for the base model on a given
event. If omitted, we threshold score_fn(..., None) at 0.5.
"""


@dataclass
class TempMEAdapterConfig:
    # Required
    model_name: str                     # "tgn" | "tgat" | "graphmixer" | ...
    dataset_name: str

    # Basic options
    explanation_level: str = "event"
    results_dir: Optional[str] = None
    debug_mode: bool = False
    alias: Optional[str] = None
    device: Optional[Union[str, torch.device]] = None

    # Motif sampling hyperparameters (cf. Alg. 1 in the paper)
    num_motifs_per_node: int = 32       # C in the paper (per endpoint u and v)
    motif_length: int = 3               # l  (max #events per motif)
    motif_max_nodes: int = 3            # n  (max #nodes per motif)
    motif_delta: float = float("inf")   # δ duration window (in same units as timestamps)

    # Perturbation/explanation options
    score_fn: Optional[ScoreFn] = None
    label_fn: Optional[LabelFn] = None

    # How to aggregate motif scores into event scores
    clip_negative_motif_scores: bool = True

    # Caching of explanations per event_idx
    cache: bool = True


# ---------------------------------------------------------------------------
# TempME-style motif explainer adapter
# ---------------------------------------------------------------------------

class TempMEAdapter(BaseExplainer):
    """
    TempME-style adapter (motif-based explainer for TGNNs).

    This implementation follows the *structure* of TempME:

    1. For a target interaction e = (u, v, t), sample retrospective temporal
       motifs around u and v (Alg. 1 in the paper).
    2. Treat each sampled motif (sequence of event indices) as an "interpretable
       component".
    3. For each motif, measure its importance by running the base model on a
       subgraph containing only events from that motif (plus the target event),
       and checking how well this subgraph preserves the original prediction.
    4. Aggregate motif scores back onto events (events that appear in many
       high-importance motifs get higher scores).
    5. Return an ExplanationResult with an importance vector over candidate
       event indices.

    Notes
    -----
    * This is **not** an exact reimplementation of the paper's
      Information-Bottleneck training objective. Instead, motif importance is
      estimated via perturbation, similar in spirit to PBOne but at the motif
      level.
    * It is designed to be reasonably plug-and-play with minimal assumptions
      about your dataset, but you must provide a `score_fn` in the config.
    """

    def __init__(self, cfg: TempMEAdapterConfig) -> None:
        super().__init__(name="tempme", alias=cfg.alias or "tempme")
        self.cfg = cfg

        self.device: Optional[torch.device] = (
            torch.device(cfg.device) if cfg.device is not None else None
        )
        self._prepared: bool = False

        # Raw events (from dataset)
        self._events: Optional[Sequence[Any]] = None
        self._events_df: Any = None
        self._event_to_row: Dict[int, int] = {}
        self._one_based_event_idx: bool = False
        self._num_events: int = 0
        # Parsed per-event attributes
        self._event_src: List[int] = []
        self._event_dst: List[int] = []
        self._event_time: List[float] = []
        self._events_by_node: DefaultDict[int, List[int]] = defaultdict(list)

        # Cache: event_idx -> packed explanation dict
        self._cache: Dict[int, Dict[str, Any]] = {}

    # ------------------------------------------------------------------ #
    # Lifecycle                                                         #
    # ------------------------------------------------------------------ #

    def prepare(self, *, model: Any, dataset: Any) -> None:
        """
        Initialize adapter with model + dataset and index events for motif sampling.
        """
        super().prepare(model=model, dataset=dataset)
        # Mirror base attributes with public names for legacy callers
        self.model = model
        self.dataset = dataset

        # Device detection
        if self.device is None:
            try:
                p = next(model.parameters())
                self.device = p.device
            except Exception:
                self.device = torch.device(
                    "cuda:0" if torch.cuda.is_available() else "cpu"
                )

        # Score function is required
        if self.cfg.score_fn is None:
            raise ValueError(
                "TempMEAdapterConfig.score_fn must be provided. "
                "Expected signature: score_fn(model, dataset, event_idx, active_event_indices) -> float."
            )

        # Pull out events from dataset
        if isinstance(dataset, dict) and "events" in dataset:
            self._events = dataset["events"]
            if "dataset_name" in dataset:
                self.cfg.dataset_name = dataset["dataset_name"]
        else:
            self._events = dataset

        if self._events is None:
            raise ValueError("TempMEAdapter expects `dataset` or `dataset['events']` to be iterable.")

        # Detect dataframe inputs (for 1-based event idx columns)
        self._events_df = None
        self._event_to_row = {}
        self._one_based_event_idx = False
        try:
            import pandas as pd  # type: ignore

            if isinstance(self._events, pd.DataFrame):
                self._events_df = self._events
                self._one_based_event_idx = self._detect_one_based_idx(self._events_df)
                self._num_events = len(self._events_df)
        except Exception:
            self._events_df = None

        if self._events_df is None:
            try:
                self._num_events = len(self._events)
            except Exception:
                self._num_events = 0

        # Build internal indices for events
        self._index_events()

        self._prepared = True

    # ------------------------------------------------------------------ #
    # Core explanation                                                  #
    # ------------------------------------------------------------------ #

    def explain(self, context: ExplanationContext) -> ExplanationResult:
        assert self._prepared, "Call .prepare(model=..., dataset=...) first."

        # 1) Resolve event index from context
        eidx = (
            context.target.get("event_idx")
            or context.target.get("index")
            or context.target.get("idx")
        )
        if eidx is None and context.subgraph and context.subgraph.payload:
            eidx = context.subgraph.payload.get("event_idx")
        if eidx is None:
            raise ValueError(
                "TempMEAdapter expects an event index in "
                "context.target['event_idx'|'index'|'idx'] or subgraph.payload['event_idx']."
            )

        eidx = int(eidx)
        if self._events is None:
            raise RuntimeError("Events are not indexed. Did you call prepare()?")

        # Map external event id -> dataframe row/sequence index
        if self._event_to_row:
            if eidx not in self._event_to_row:
                max_idx = max(self._event_to_row.keys())
                raise IndexError(f"event_idx {eidx} is out of range for {max_idx} events.")
            row_idx = self._event_to_row[eidx]
        else:
            if self._one_based_event_idx:
                if eidx <= 0 or (self._num_events and eidx > self._num_events):
                    raise IndexError(f"event_idx {eidx} is out of range for {self._num_events} events.")
                row_idx = eidx - 1
            else:
                if self._num_events and (eidx < 0 or eidx >= self._num_events):
                    raise IndexError(f"event_idx {eidx} is out of range for {self._num_events} events.")
                row_idx = eidx

        # Cache lookup
        if self.cfg.cache and eidx in self._cache:
            pack = self._cache[eidx]
            # Convert cached pack into ExplanationResult
            return self._pack_to_result(context, eidx, pack)

        t0 = time.perf_counter()

        # 2) Parse target event
        target_event = self._get_event_by_row(row_idx)
        u0, v0, t_event = self._parse_event(target_event)

        # 3) Base score & label (original prediction)
        score_full = float(self.cfg.score_fn(self.model, self.dataset, eidx, None))

        if self.cfg.label_fn is not None:
            label_full = int(self.cfg.label_fn(self.model, self.dataset, eidx))
        else:
            # Default: binary label from probability / logit threshold 0.5
            label_full = int(score_full >= 0.5)

        # 4) Sample temporal motifs around u0 and v0
        motifs_u = self._sample_motifs_for_node(seed_node=u0, t0=t_event)
        motifs_v = self._sample_motifs_for_node(seed_node=v0, t0=t_event)
        motifs: List[List[int]] = motifs_u + motifs_v

        # Deduplicate motifs (as sorted tuples of event indices)
        seen = set()
        unique_motifs: List[List[int]] = []
        for m in motifs:
            key = tuple(sorted(m))
            if not key:
                continue
            if key not in seen:
                seen.add(key)
                unique_motifs.append(list(key))
        motifs = unique_motifs

        # If we somehow got no motifs, fall back to a degenerate explanation
        if not motifs:
            payload_candidate = None
            if context.subgraph and context.subgraph.payload:
                payload_candidate = context.subgraph.payload.get("candidate_eidx")
            if payload_candidate:
                candidate = [int(e) for e in payload_candidate]
                imp_edges = [1.0 if int(e) == eidx else 0.0 for e in candidate]
            else:
                candidate = [eidx]
                imp_edges = [1.0]
            elapsed = time.perf_counter() - t0
            pack = {
                "candidate_eidx": candidate,
                "importance_edges": imp_edges,
                "motifs": [],
                "motif_scores": [],
                "score_full": score_full,
                "label_full": label_full,
                "elapsed_sec": elapsed,
            }
            if self.cfg.cache:
                self._cache[eidx] = pack
            return self._pack_to_result(context, eidx, pack)

        # 5) Score each motif via perturbation
        motif_scores: List[float] = []
        event_importance: DefaultDict[int, float] = defaultdict(float)

        for motif in motifs:
            # Explanation subgraph: union of motif events + the target event
            # (You can modify your score_fn to treat these as "historical events"
            #  and automatically include the target eidx as the query.)
            active_events = list(set(motif + [eidx]))

            score_motif = float(
                self.cfg.score_fn(self.model, self.dataset, eidx, active_events)
            )

            if label_full == 1:
                raw_imp = score_motif - score_full
            else:
                raw_imp = score_full - score_motif

            if self.cfg.clip_negative_motif_scores:
                imp = max(raw_imp, 0.0)
            else:
                imp = raw_imp

            motif_scores.append(imp)

            # Aggregate motif importance onto events
            for ev in motif:
                event_importance[ev] += imp

        # 6) Build candidate event list & dense importance vector
        def _normalize(values: List[float]) -> List[float]:
            if not values:
                return []
            vals = torch.tensor(values, dtype=torch.float32)
            if vals.abs().max() > 0:
                vals = vals / (vals.abs().max() + 1e-9)
            return vals.tolist()

        candidate_eidx = sorted(event_importance.keys())
        candidate_eidx_raw: Optional[List[int]] = None
        importance_edges_raw: Optional[List[float]] = None
        payload_candidate = None
        if context.subgraph and context.subgraph.payload:
            payload_candidate = context.subgraph.payload.get("candidate_eidx")

        if payload_candidate:
            candidate_eidx_raw = list(candidate_eidx)
            candidate_eidx = [int(e) for e in payload_candidate]
            importance_edges_raw = [float(event_importance.get(int(e), 0.0)) for e in candidate_eidx]
            imp_edges = _normalize(importance_edges_raw)
        elif not candidate_eidx:
            # Extremely edge case: all motif scores collapsed to 0
            candidate_eidx = [eidx]
            imp_edges = [0.0]
        else:
            imp_edges = _normalize([float(event_importance[e]) for e in candidate_eidx])

        elapsed = time.perf_counter() - t0

        pack = {
            "candidate_eidx": candidate_eidx,
            "importance_edges": imp_edges,
            "motifs": motifs,
            "motif_scores": motif_scores,
            "score_full": score_full,
            "label_full": label_full,
            "elapsed_sec": elapsed,
        }
        if candidate_eidx_raw is not None:
            pack["candidate_eidx_raw"] = candidate_eidx_raw
        if importance_edges_raw is not None:
            pack["importance_edges_raw"] = importance_edges_raw

        if self.cfg.cache:
            self._cache[eidx] = pack

        return self._pack_to_result(context, eidx, pack)

    # ------------------------------------------------------------------ #
    # Optional: bulk explain (re-using motif sampling & scoring)        #
    # ------------------------------------------------------------------ #

    def explain_many(self, event_idxs: Sequence[int]) -> Dict[int, Dict[str, Any]]:
        """
        Convenience wrapper returning raw TempME-style motif scores for multiple events.

        Returns
        -------
        Dict[event_idx, packed_result_dict]
        where packed_result_dict has keys:
            - candidate_eidx: List[int]
            - importance_edges: List[float]
            - motifs: List[List[int]]
            - motif_scores: List[float]
            - score_full: float
            - label_full: int
            - elapsed_sec: float
        """
        results: Dict[int, Dict[str, Any]] = {}
        for eidx in event_idxs:
            dummy_ctx = ExplanationContext(
                run_id=None,
                target={"event_idx": int(eidx)},
                subgraph=None,
            )
            res = self.explain(dummy_ctx)
            # Reconstitute exactly what we store in cache
            pack = {
                "candidate_eidx": res.extras["candidate_eidx"],
                "importance_edges": res.importance_edges,
                "motifs": res.extras.get("motifs", []),
                "motif_scores": res.extras.get("motif_scores", []),
                "score_full": res.extras.get("score_full"),
                "label_full": res.extras.get("label_full"),
                "elapsed_sec": res.elapsed_sec,
            }
            results[int(eidx)] = pack
        return results

    # ------------------------------------------------------------------ #
    # Internal helpers                                                  #
    # ------------------------------------------------------------------ #

    def _pack_to_result(
        self,
        context: ExplanationContext,
        eidx: int,
        pack: Dict[str, Any],
    ) -> ExplanationResult:
        """
        Convert an internal packed dict into an ExplanationResult.
        """
        extras = {
            "event_idx": eidx,
            "candidate_eidx": list(pack["candidate_eidx"]),
            "motifs": pack.get("motifs", []),
            "motif_scores": pack.get("motif_scores", []),
            "score_full": pack.get("score_full"),
            "label_full": pack.get("label_full"),
        }
        if "candidate_eidx_raw" in pack:
            extras["candidate_eidx_raw"] = list(pack.get("candidate_eidx_raw") or [])
        if "importance_edges_raw" in pack:
            extras["importance_edges_raw"] = list(pack.get("importance_edges_raw") or [])

        return ExplanationResult(
            run_id=context.run_id,
            explainer=self.alias,
            context_fp=context.fingerprint(),
            importance_edges=list(pack["importance_edges"]),
            importance_nodes=None,
            importance_time=None,
            elapsed_sec=float(pack.get("elapsed_sec", 0.0)),
            extras=extras,
        )

    def _detect_one_based_idx(self, df: Any) -> bool:
        """Heuristically determine whether event indices are 1-based."""
        for col in ("e_idx", "idx"):
            if hasattr(df, "columns") and col in df.columns:
                try:
                    min_val = df[col].min()
                    return min_val == 1
                except Exception:
                    continue
        return False

    def _get_event_by_row(self, row_idx: int) -> Any:
        """
        Return the raw event object given a 0-based dataframe/sequence row index.
        """
        if self._events_df is not None:
            return self._events_df.iloc[int(row_idx)]
        if isinstance(self._events, Sequence):
            return self._events[int(row_idx)]
        return self._events[int(row_idx)]

    def _event_id_from_object(self, row_idx: int, ev: Any) -> int:
        """
        Resolve the canonical event id (matching the dataset convention).
        """
        for key in ("e_idx", "idx"):
            try:
                if isinstance(ev, dict) and key in ev:
                    return int(ev[key])
                if hasattr(ev, "__contains__") and key in ev:
                    return int(ev[key])
                if hasattr(ev, key):
                    return int(getattr(ev, key))
            except Exception:
                continue
        return int(row_idx + 1) if self._one_based_event_idx else int(row_idx)

    def _store_event(self, event_id: int, u: int, v: int, t: float) -> None:
        """Ensure internal lists are large enough and store event attributes."""
        needed = event_id + 1
        if len(self._event_src) < needed:
            pad = needed - len(self._event_src)
            self._event_src.extend([0] * pad)
            self._event_dst.extend([0] * pad)
            self._event_time.extend([0.0] * pad)
        self._event_src[event_id] = int(u)
        self._event_dst[event_id] = int(v)
        self._event_time[event_id] = float(t)

    def _index_events(self) -> None:
        """
        Build internal indices for fast motif sampling:
            - _event_src, _event_dst, _event_time (lists)
            - _events_by_node: node_id -> [event_idx, ...] sorted by time (descending)
        """
        assert self._events is not None

        self._event_src.clear()
        self._event_dst.clear()
        self._event_time.clear()
        self._events_by_node.clear()
        self._event_to_row.clear()

        if self._events_df is not None:
            iterator = self._events_df.iterrows()
        else:
            iterator = enumerate(self._events)

        for row_idx, ev in iterator:
            try:
                u, v, t = self._parse_event(ev)
            except Exception:
                # If we can't parse this event, skip it for motif sampling
                continue

            event_id = self._event_id_from_object(int(row_idx), ev)
            self._store_event(event_id, u, v, float(t))
            self._events_by_node[u].append(event_id)
            self._events_by_node[v].append(event_id)
            self._event_to_row[event_id] = int(row_idx)

        # Sort events per node by *descending* time so we can quickly filter by t < t_prev
        for node, idxs in self._events_by_node.items():
            idxs.sort(key=lambda i: self._event_time[i], reverse=True)

        if self._event_to_row:
            self._num_events = max(self._event_to_row.keys())

    def _parse_event(self, ev: Any) -> Tuple[int, int, float]:
        """
        Heuristic parser for an event object.

        Supported shapes:
        1. Mapping with keys:
            - ("u", "v", "t") or ("u", "i", "ts") or ("u", "i", "t") or ("src", "dst", "time") or ("source", "target", "timestamp")
        2. Sequence / tuple:
            - (u, v, t, ...)  -> take first 3.
        3. Object with attributes:
            - .u, .v, .t
            - .u, .i, .ts
            - .u, .i, .t
            - .src, .dst, .time
            - .source, .target, .timestamp

        If your dataset uses a different layout, it's easy to modify this
        function to parse it correctly.
        """
        try:
            import pandas as pd  # type: ignore

            if isinstance(ev, pd.Series):
                ev = ev.to_dict()
        except Exception:
            pass

        # 1) Mapping / dict
        if isinstance(ev, dict):
            # Common key variants
            for keys in [
                ("u", "v", "t"),
                ("u", "i", "ts"),
                ("u", "i", "t"),
                ("src", "dst", "time"),
                ("source", "target", "timestamp"),
            ]:
                if all(k in ev for k in keys):
                    u, v, t = ev[keys[0]], ev[keys[1]], ev[keys[2]]
                    return int(u), int(v), float(t)

            raise KeyError(
                "Unsupported event dict keys for TempMEAdapter; expected one of "
                "('u','v','t'), ('u','i','ts'), ('u','i','t'), "
                "('src','dst','time'), ('source','target','timestamp')."
            )

        # 2) Sequence / tuple
        if isinstance(ev, (list, tuple)) and len(ev) >= 3:
            u, v, t = ev[0], ev[1], ev[2]
            return int(u), int(v), float(t)

        # 3) Generic object with attributes
        for (u_name, v_name, t_name) in [
            ("u", "v", "t"),
            ("u", "i", "ts"),
            ("u", "i", "t"),
            ("src", "dst", "time"),
            ("source", "target", "timestamp"),
        ]:
            if hasattr(ev, u_name) and hasattr(ev, v_name) and hasattr(ev, t_name):
                u = getattr(ev, u_name)
                v = getattr(ev, v_name)
                t = getattr(ev, t_name)
                return int(u), int(v), float(t)

        raise TypeError(
            "Unsupported event object type for TempMEAdapter. "
            "Please modify _parse_event to handle your dataset's format."
        )

    # ---- motif sampling ------------------------------------------------- #

    def _sample_motifs_for_node(self, seed_node: int, t0: float) -> List[List[int]]:
        """
        Sample C retrospective temporal motifs starting from a seed node,
        following the spirit of Algorithm 1 in the TempME paper.

        Each motif is a sequence of event indices (length <= motif_length).
        Events are strictly before t0 and within a δ-window if motif_delta is finite.
        """
        motifs: List[List[int]] = []
        C = self.cfg.num_motifs_per_node
        l_max = self.cfg.motif_length

        if C <= 0 or l_max <= 0:
            return motifs

        for _ in range(C):
            motif = self._sample_single_motif(seed_node=seed_node, t0=t0)
            if motif:
                motifs.append(motif)

        return motifs

    def _sample_single_motif(self, seed_node: int, t0: float) -> List[int]:
        """
        Sample one motif instance I = (e1, e2, ..., el) as a sequence of event indices.
        """
        S = {seed_node}
        I: List[int] = []

        t_prev = t0
        l_max = self.cfg.motif_length
        n_max = self.cfg.motif_max_nodes
        delta = self.cfg.motif_delta

        for _ in range(l_max):
            candidates = self._collect_candidate_events(
                nodes=S,
                t_prev=t_prev,
                t0=t0,
                delta=delta,
            )

            # Avoid reusing events within the same motif instance
            candidates = [i for i in candidates if i not in I]

            if not candidates:
                break

            e_idx = random.choice(candidates)
            I.append(e_idx)

            if len(S) < n_max:
                S.add(self._event_src[e_idx])
                S.add(self._event_dst[e_idx])

            t_prev = self._event_time[e_idx]

        return I

    def _collect_candidate_events(
        self,
        nodes: Iterable[int],
        t_prev: float,
        t0: float,
        delta: float,
    ) -> List[int]:
        """
        E(S, t_prev) from the paper:
            - Historical events touching any node in `nodes`
            - Occurring strictly before t_prev
            - And (if delta < inf) with t0 - t <= delta
        """
        candidates: List[int] = []
        nodes_set = set(nodes)

        for u in nodes_set:
            for e_idx in self._events_by_node.get(u, []):
                t_e = self._event_time[e_idx]
                if t_e >= t_prev:
                    # events_by_node lists are sorted by descending time,
                    # so we can early-stop once t_e >= t_prev
                    continue
                if delta != float("inf") and (t0 - t_e) > delta:
                    # If t_e is too far in the past relative to t0, skip it.
                    continue
                candidates.append(e_idx)

        if not candidates:
            return []

        # Deduplicate
        return list(set(candidates))

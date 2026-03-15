
"""
CoDy: Counterfactual Explainers for Dynamic Graphs (Qu et al.)

This module provides a clean, self-contained implementation of the CoDy
Monte-Carlo Tree Search explainer (and an optional Greedy baseline) that
can be used with *any* temporal link predictor via a user-provided
`score_fn(omit_edge_idxs)` callback.

It also includes a lightweight wrapper for the official
`twitter-research/tgn` implementation that supports *edge-omission*
counterfactual scoring by filtering the neighbor finder at query time.

Notes
-----
1) Faithfulness vs. practicality:
   - The strict counterfactual setting would require re-running the TGNN's
     state updates on a graph where the omitted events never happened.
     That can be prohibitively expensive.
   - The `TGNLinkPredictor` wrapper below uses the common practical
     approximation: keep the model parameters + memory snapshot fixed
     and filter omitted edges from the temporal neighbor sampling.

2) Determinism:
   For stable explanations, use deterministic neighbor sampling in TGN
   (i.e., most recent neighbors rather than uniform random sampling).

References
----------
- CoDy paper (provided by user): "CoDy: Counterfactual Explainers for Dynamic Graphs"
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union

import copy
import math
import random

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore


# -----------------------------
# Utilities
# -----------------------------

def _logit(p: float, eps: float = 1e-7) -> float:
    """Convert a probability to a logit in a numerically stable way."""
    p = float(p)
    p = min(max(p, eps), 1.0 - eps)
    return math.log(p / (1.0 - p))


def _delta_towards_flip(p_orig: float, p_new: float) -> float:
    """
    CoDy's delta (Eq. 6): shift towards the opposite sign of p_orig.
    If p_orig >= 0, we want to reduce the score => delta = p_orig - p_new.
    If p_orig < 0, we want to increase the score => delta = p_new - p_orig.
    """
    if p_orig >= 0:
        return p_orig - p_new
    return p_new - p_orig


def _is_counterfactual(p_orig: float, p_new: float, threshold: float = 0.0) -> bool:
    """
    Counterfactual if the sign (relative to threshold) flips.
    For logits, threshold is 0.
    """
    return (p_orig >= threshold and p_new < threshold) or (p_orig < threshold and p_new >= threshold)


def _safe_norm_shift(p_orig: float, delta: float, eps: float = 1e-12) -> float:
    """Normalized shift used for node score (Eq. 8): max(0, delta/|p_orig|)."""
    denom = max(abs(p_orig), eps)
    return max(0.0, delta / denom)


# -----------------------------
# Temporal graph indexing (candidate extraction)
# -----------------------------

@dataclass(frozen=True)
class EventRecord:
    """A single temporal interaction/event."""
    src: int
    dst: int
    ts: float
    edge_idx: int


class TemporalGraphIndex:
    """
    Minimal temporal graph index to support:
      - extracting a k-hop, time-filtered candidate event set
      - computing node hop distances (for spatio-temporal ranking)

    The index stores, for each node, the incident events as time-sorted arrays
    of (timestamp, neighbor, edge_idx).
    """

    def __init__(
        self,
        sources: Sequence[int],
        destinations: Sequence[int],
        timestamps: Sequence[Union[int, float]],
        edge_idxs: Sequence[int],
        *,
        assume_undirected: bool = True,
    ) -> None:
        if not (len(sources) == len(destinations) == len(timestamps) == len(edge_idxs)):
            raise ValueError("sources, destinations, timestamps, edge_idxs must have equal length")

        self.sources = np.asarray(sources, dtype=np.int64)
        self.destinations = np.asarray(destinations, dtype=np.int64)
        self.timestamps = np.asarray(timestamps, dtype=np.float64)
        self.edge_idxs = np.asarray(edge_idxs, dtype=np.int64)
        self.assume_undirected = bool(assume_undirected)

        self._node_adj: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._edge_ts: Dict[int, float] = {}  # edge_idx -> timestamp
        self._edge_endpoints: Dict[int, Tuple[int, int]] = {}  # edge_idx -> (src, dst)

        self._build()

    def _build(self) -> None:
        from collections import defaultdict

        tmp: Dict[int, List[Tuple[float, int, int]]] = defaultdict(list)

        for s, d, t, e in zip(self.sources, self.destinations, self.timestamps, self.edge_idxs):
            s_i = int(s)
            d_i = int(d)
            t_f = float(t)
            e_i = int(e)

            self._edge_ts[e_i] = t_f
            self._edge_endpoints[e_i] = (s_i, d_i)

            tmp[s_i].append((t_f, d_i, e_i))
            if self.assume_undirected:
                tmp[d_i].append((t_f, s_i, e_i))

        for node, lst in tmp.items():
            lst.sort(key=lambda x: x[0])  # ascending time
            ts_arr = np.asarray([x[0] for x in lst], dtype=np.float64)
            neigh_arr = np.asarray([x[1] for x in lst], dtype=np.int64)
            eidx_arr = np.asarray([x[2] for x in lst], dtype=np.int64)
            self._node_adj[int(node)] = (ts_arr, neigh_arr, eidx_arr)

    def edge_timestamp(self, edge_idx: int) -> float:
        return self._edge_ts[int(edge_idx)]

    def edge_endpoints(self, edge_idx: int) -> Tuple[int, int]:
        return self._edge_endpoints[int(edge_idx)]

    def _node_events_before(
        self,
        node: int,
        cutoff_time: float,
        *,
        max_events: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the tail of events for `node` with ts < cutoff_time."""
        if node not in self._node_adj:
            return (np.empty((0,), dtype=np.float64),
                    np.empty((0,), dtype=np.int64),
                    np.empty((0,), dtype=np.int64))

        ts_arr, neigh_arr, eidx_arr = self._node_adj[node]
        # index of first event with ts >= cutoff
        idx = int(np.searchsorted(ts_arr, cutoff_time, side="left"))
        if idx <= 0:
            return (np.empty((0,), dtype=np.float64),
                    np.empty((0,), dtype=np.int64),
                    np.empty((0,), dtype=np.int64))

        start = 0
        if max_events is not None:
            start = max(0, idx - int(max_events))
        return ts_arr[start:idx], neigh_arr[start:idx], eidx_arr[start:idx]

    def k_hop_candidates(
        self,
        src: int,
        dst: int,
        cutoff_time: float,
        *,
        k_hops: int,
        m_max: int,
        per_node_cap: Optional[int] = None,
    ) -> Tuple[List[int], Dict[int, int]]:
        """
        Compute candidate event ids C(G, εi, k, m_max) for target link (src, dst, cutoff_time).

        Strategy:
          - BFS from {src, dst} for k_hops expansions, using only events with ts < cutoff_time.
          - Collect all encountered edge_idx and node hop-distances.
          - Keep only the m_max most recent events (global) among collected.

        Returns:
          candidates: list of edge_idx (sorted by recency descending)
          node_distance: dict node_id -> hop distance from {src, dst}
        """
        if k_hops < 0:
            raise ValueError("k_hops must be >= 0")
        if m_max <= 0:
            return [], {int(src): 0, int(dst): 0}

        src = int(src)
        dst = int(dst)
        cutoff_time = float(cutoff_time)

        if per_node_cap is None:
            per_node_cap = m_max  # pragmatic cap to avoid exploding degrees

        # BFS over nodes; we keep distances for spatio-temporal ranking
        node_dist: Dict[int, int] = {src: 0, dst: 0}
        frontier: Set[int] = {src, dst}

        # edge_idx -> timestamp (for sorting)
        collected: Dict[int, float] = {}

        for hop in range(k_hops + 1):
            # Collect edges incident to current frontier nodes.
            # We still collect at hop==k_hops; we just don't expand beyond it.
            next_frontier: Set[int] = set()

            for node in frontier:
                ts_arr, neigh_arr, eidx_arr = self._node_events_before(
                    node, cutoff_time, max_events=per_node_cap
                )
                for t, n, e in zip(ts_arr, neigh_arr, eidx_arr):
                    e_i = int(e)
                    collected[e_i] = float(t)
                    n_i = int(n)
                    # Expand to neighbors only if we can still go further
                    if hop < k_hops and n_i not in node_dist:
                        node_dist[n_i] = hop + 1
                        next_frontier.add(n_i)

            frontier = next_frontier
            if not frontier:
                break

        # Keep m_max most recent edges
        if not collected:
            return [], node_dist

        sorted_edges = sorted(collected.items(), key=lambda kv: kv[1], reverse=True)
        top_edges = [e for e, _t in sorted_edges[:m_max]]
        return top_edges, node_dist


# -----------------------------
# CoDy search tree data structures
# -----------------------------

@dataclass
class SearchNode:
    """
    Node tuple (Eq. 10 in the paper):
      (s_j, p_j, parent_j, children_j, selections_j, score_j, selectable_j)

    Here:
      - events is s_j (frozenset of omitted edge_idxs)
      - prediction is p_j (float logit)
      - parent is parent_j
      - children is children_j
      - selections is selections_j
      - score is score_j (float, normalized shift)
      - selectable is selectable_j
      - expanded indicates whether we already expanded this node (children created)
    """
    events: frozenset
    prediction: Optional[float] = None
    parent: Optional["SearchNode"] = None
    children: List["SearchNode"] = None  # type: ignore
    selections: int = 0
    score: Optional[float] = None
    selectable: bool = True
    expanded: bool = False

    def __post_init__(self) -> None:
        if self.children is None:
            self.children = []

    @property
    def depth(self) -> int:
        return len(self.events)

    def is_leaf(self) -> bool:
        return self.expanded and len(self.children) == 0


# -----------------------------
# Selection policies
# -----------------------------

class SelectionPolicy:
    """
    Implements the 4 ranking policies from the CoDy paper:
      - random
      - temporal
      - spatio-temporal
      - event-impact
    """

    def __init__(
        self,
        name: str,
        *,
        target_time: float,
        candidate_edge_ts: Mapping[int, float],
        node_dist: Optional[Mapping[int, int]] = None,
        edge_endpoints: Optional[Mapping[int, Tuple[int, int]]] = None,
        rng: Optional[random.Random] = None,
        event_impact: Optional[Mapping[int, float]] = None,
    ) -> None:
        name = name.strip().lower().replace("_", "-")
        if name not in {"random", "temporal", "spatio-temporal", "spatio_temporal", "event-impact", "event_impact"}:
            raise ValueError(f"Unknown selection policy: {name}")

        if name == "spatio_temporal":
            name = "spatio-temporal"
        if name == "event_impact":
            name = "event-impact"

        self.name = name
        self.target_time = float(target_time)
        self.candidate_edge_ts = dict((int(k), float(v)) for k, v in candidate_edge_ts.items())
        self.node_dist = dict((int(k), int(v)) for k, v in (node_dist or {}).items())
        self.edge_endpoints = dict((int(k), (int(v[0]), int(v[1]))) for k, v in (edge_endpoints or {}).items())
        self.rng = rng or random.Random(0)
        self.event_impact = dict((int(k), float(v)) for k, v in (event_impact or {}).items())

        # Random policy: pre-sample a stable random order to make ranking deterministic per seed.
        if self.name == "random":
            keys = list(self.candidate_edge_ts.keys())
            self.rng.shuffle(keys)
            self._random_rank = {e: i for i, e in enumerate(keys)}
        else:
            self._random_rank = {}

    def rank_event(self, edge_idx: int) -> Tuple:
        """
        Lower tuples are considered "higher priority" (i.e., selected first).
        """
        e = int(edge_idx)
        if self.name == "random":
            return (self._random_rank.get(e, 10**9),)

        t = self.candidate_edge_ts.get(e, None)
        if t is None:
            # Unknown event -> push to back
            return (10**9, 10**9)

        dt = abs(self.target_time - t)

        if self.name == "temporal":
            return (dt,)

        if self.name == "spatio-temporal":
            # Distance between event endpoints and target endpoints approximated by node_dist
            a, b = self.edge_endpoints.get(e, (None, None))
            da = self.node_dist.get(a, 10**9) if a is not None else 10**9
            db = self.node_dist.get(b, 10**9) if b is not None else 10**9
            d_spatial = min(da, db)
            return (d_spatial, dt)

        if self.name == "event-impact":
            # Larger impact should rank earlier => use negative for ascending sort
            imp = self.event_impact.get(e, -10**9)
            return (-imp,)

        raise RuntimeError("Unhandled policy")

    def rank_child(self, child: SearchNode) -> Tuple:
        """
        Rank a child node by the single event added relative to its parent.
        """
        if child.parent is None:
            # Root has no added event; push it to the end.
            return (10**9,)

        diff = child.events.difference(child.parent.events)
        if len(diff) != 1:
            # Shouldn't happen in our search tree construction.
            return (10**9,)

        (added_event,) = tuple(diff)
        return self.rank_event(int(added_event))


# -----------------------------
# CoDy explainer
# -----------------------------

@dataclass
class ExplanationResult:
    omitted_edge_idxs: List[int]
    original_score: float
    perturbed_score: float
    is_counterfactual: bool
    iterations_run: int
    best_score: float  # normalized shift score
    policy: str
    candidate_size: int


class CoDyExplainer:
    """
    CoDy Monte-Carlo Tree Search explainer.

    Parameters
    ----------
    score_fn:
        Callable that returns a *logit-like* scalar for the explained event
        when a set of historical events is omitted.

        Signature: score_fn(omit_edge_idxs: Set[int]) -> float

        Important: score_fn must treat its argument as the set of *historical*
        edge indices to omit from the model's input (for scoring the explained event).
    graph_index:
        TemporalGraphIndex for candidate event extraction.
    k_hops:
        Spatial constraint (k-hop neighborhood).
        The paper recommends k = number of TGNN layers.
    m_max:
        Temporal constraint (only keep m_max most recent candidates).
    it_max:
        Maximum number of MCTS iterations (each triggers at most 1 model call).
    alpha:
        Exploration/exploitation trade-off in Eq. (7).
        alpha=1 => pure exploitation, alpha=0 => pure exploration.
    policy:
        One of {"random", "temporal", "spatio-temporal", "event-impact"}.
    seed:
        Random seed used for the random policy (and any tie-breaking).
    """

    def __init__(
        self,
        *,
        score_fn: Callable[[Set[int]], float],
        graph_index: TemporalGraphIndex,
        k_hops: int,
        m_max: int,
        it_max: int = 100,
        alpha: float = 0.5,
        policy: str = "temporal",
        seed: int = 0,
    ) -> None:
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha must be in [0, 1]")

        self.score_fn = score_fn
        self.graph_index = graph_index
        self.k_hops = int(k_hops)
        self.m_max = int(m_max)
        self.it_max = int(it_max)
        self.alpha = float(alpha)
        self.policy_name = policy.strip().lower().replace("_", "-")
        self.seed = int(seed)

        self._rng = random.Random(self.seed)

    def explain(
        self,
        src: int,
        dst: int,
        ts: float,
        *,
        edge_idx: Optional[int] = None,  # kept for API symmetry; not used by CoDy itself
        per_node_cap: Optional[int] = None,
    ) -> ExplanationResult:
        """
        Explain the model's prediction for the future link (src, dst) at time ts.
        """
        # 1) Candidate event set C(G, εi, k, m_max)
        candidates, node_dist = self.graph_index.k_hop_candidates(
            src, dst, ts, k_hops=self.k_hops, m_max=self.m_max, per_node_cap=per_node_cap
        )
        candidate_set = set(int(e) for e in candidates)
        candidate_ts = {e: self.graph_index.edge_timestamp(e) for e in candidate_set}
        edge_endpoints = {e: self.graph_index.edge_endpoints(e) for e in candidate_set}

        # 2) Original prediction
        p_orig = float(self.score_fn(set()))
        best_node: Optional[SearchNode] = None
        best_norm_score: float = -float("inf")

        # 3) Optional: precompute event-impact ranking (policy = event-impact)
        event_impact: Dict[int, float] = {}
        precomputed_single_scores: Dict[int, float] = {}
        if self.policy_name in {"event-impact", "event_impact"}:
            for e in candidates:
                p_e = float(self.score_fn({int(e)}))
                precomputed_single_scores[int(e)] = p_e
                event_impact[int(e)] = _delta_towards_flip(p_orig, p_e)

        # 4) Policy object
        policy = SelectionPolicy(
            self.policy_name,
            target_time=float(ts),
            candidate_edge_ts=candidate_ts,
            node_dist=node_dist,
            edge_endpoints=edge_endpoints,
            rng=self._rng,
            event_impact=event_impact,
        )

        # 5) Initialize root and expand it (equivalent to first iteration in Algorithm 1)
        root = SearchNode(events=frozenset(), prediction=p_orig, parent=None, selections=1, score=0.0, selectable=True, expanded=True)

        # Create root children = all single-event omissions
        root.children = []
        for e in candidates:
            e_i = int(e)
            child = SearchNode(
                events=frozenset({e_i}),
                prediction=precomputed_single_scores.get(e_i),
                parent=root,
                selections=0,
                score=None,
                selectable=True,
                expanded=False,
            )
            root.children.append(child)

        # Track the best *observed* node (counterfactual or not)
        # Root's normalized score is 0.
        best_node = root
        best_norm_score = 0.0

        # Keep counterfactual examples to return the minimal one if found
        counterfactual_nodes: List[SearchNode] = []

        # 6) MCTS iterations
        it = 0
        while it < self.it_max and root.selectable:
            selected = self._select(root, policy)
            if selected is None:
                break

            # simulate
            if selected.prediction is None:
                selected.prediction = float(self.score_fn(set(selected.events)))

            # expand (sets score, selections, children, selectable)
            self._expand(selected, p_orig, candidate_set)

            # update best candidates
            if selected.score is not None and selected.score > best_norm_score:
                best_norm_score = float(selected.score)
                best_node = selected

            if selected.prediction is not None and _is_counterfactual(p_orig, selected.prediction):
                counterfactual_nodes.append(selected)

            # backpropagate from parent
            self._backpropagate(selected.parent, p_orig)

            it += 1

        # 7) Select best explanation to return
        if counterfactual_nodes:
            # Prefer minimal |s| (sparsest) and then strongest shift
            counterfactual_nodes.sort(key=lambda n: (len(n.events), -(n.score or 0.0)))
            chosen = counterfactual_nodes[0]
        else:
            chosen = best_node if best_node is not None else root

        omitted = sorted([int(e) for e in chosen.events], key=lambda e: policy.rank_event(e))

        p_new = float(chosen.prediction if chosen.prediction is not None else p_orig)
        return ExplanationResult(
            omitted_edge_idxs=omitted,
            original_score=p_orig,
            perturbed_score=p_new,
            is_counterfactual=_is_counterfactual(p_orig, p_new),
            iterations_run=it,
            best_score=float(chosen.score if chosen.score is not None else 0.0),
            policy=policy.name,
            candidate_size=len(candidate_set),
        )

    def _sel_score(self, child: SearchNode) -> float:
        """Eq. (7): selscore = alpha * exploit + (1-alpha) * explore (UCB1-style)."""
        exploit = float(child.score) if child.score is not None else 0.0

        if child.parent is None:
            return exploit

        # UCB1 exploration term
        if child.selections <= 0:
            explore = float("inf")
        else:
            parent_n = max(child.parent.selections, 1)
            explore = math.sqrt(2.0 * math.log(parent_n) / child.selections)

        return self.alpha * exploit + (1.0 - self.alpha) * explore

    def _select(self, node: SearchNode, policy: SelectionPolicy) -> Optional[SearchNode]:
        """Algorithm 3 (recursive selection)."""
        # If not expanded, return the node itself.
        if not node.expanded:
            return node

        selectable_children = [c for c in node.children if c.selectable]
        if not selectable_children:
            node.selectable = False
            return None

        # Select by max selection score
        scores = [self._sel_score(c) for c in selectable_children]
        max_score = max(scores)

        # Ties: break using selection policy δ
        # We treat infinities carefully; all unvisited children get inf and tie-break dominates.
        tied = [c for c, s in zip(selectable_children, scores) if s == max_score]

        if len(tied) > 1:
            tied.sort(key=lambda c: policy.rank_child(c))
            chosen = tied[0]
        else:
            chosen = tied[0]

        # Recurse
        return self._select(chosen, policy)

    def _expand(self, node: SearchNode, p_orig: float, candidate_set: Set[int]) -> None:
        """Algorithm 5 (expansion)."""
        # Mark node as visited/expanded
        node.selections = max(node.selections, 1)  # first time we expand => 1
        if node.prediction is None:
            raise RuntimeError("expand() called before simulation / prediction is set")

        delta = _delta_towards_flip(p_orig, node.prediction)
        node.score = _safe_norm_shift(p_orig, delta)
        node.expanded = True

        # Counterfactual nodes are not selectable (children only increase complexity)
        if _is_counterfactual(p_orig, node.prediction):
            node.selectable = False
            node.children = []
            return

        # Add children by adding exactly one remaining event
        remaining = [e for e in candidate_set if e not in node.events]
        if not remaining:
            node.selectable = False
            node.children = []
            return

        node.children = [
            SearchNode(
                events=frozenset(set(node.events).union({int(e)})),
                prediction=None,
                parent=node,
                selections=0,
                score=None,
                selectable=True,
                expanded=False,
            )
            for e in remaining
        ]

    def _backpropagate(self, node: Optional[SearchNode], p_orig: float) -> None:
        """Algorithm 6 (backpropagation) as an iterative loop."""
        while node is not None:
            # Update selection count
            node.selections += 1

            if node.prediction is None:
                # Root should always have a prediction; but keep it safe.
                base_score = 0.0
            else:
                delta = _delta_towards_flip(p_orig, node.prediction)
                base_score = _safe_norm_shift(p_orig, delta)

            # Aggregate children scores weighted by their selections
            agg = 0.0
            for c in node.children:
                if c.score is None:
                    continue
                agg += float(c.score) * float(max(c.selections, 1))

            node.score = (base_score + agg) / float(max(node.selections, 1))

            # If no selectable children remain, the node becomes unselectable
            if node.children and not any(c.selectable for c in node.children):
                node.selectable = False

            node = node.parent


# -----------------------------
# Optional greedy baseline (GreeDy)
# -----------------------------

class GreeDyExplainer:
    """
    Simple greedy baseline described in the CoDy paper.

    Each iteration:
      - sample l child nodes based on policy ranking
      - pick the child that maximally shifts prediction towards flip
      - continue until prediction flips or no improvement
    """

    def __init__(
        self,
        *,
        score_fn: Callable[[Set[int]], float],
        graph_index: TemporalGraphIndex,
        k_hops: int,
        m_max: int,
        l: int = 3,
        it_max: int = 50,
        policy: str = "temporal",
        seed: int = 0,
    ) -> None:
        self.score_fn = score_fn
        self.graph_index = graph_index
        self.k_hops = int(k_hops)
        self.m_max = int(m_max)
        self.l = int(l)
        self.it_max = int(it_max)
        self.policy_name = policy.strip().lower().replace("_", "-")
        self.seed = int(seed)
        self._rng = random.Random(self.seed)

    def explain(self, src: int, dst: int, ts: float, *, per_node_cap: Optional[int] = None) -> ExplanationResult:
        candidates, node_dist = self.graph_index.k_hop_candidates(
            src, dst, ts, k_hops=self.k_hops, m_max=self.m_max, per_node_cap=per_node_cap
        )
        candidate_set = set(int(e) for e in candidates)
        candidate_ts = {e: self.graph_index.edge_timestamp(e) for e in candidate_set}
        edge_endpoints = {e: self.graph_index.edge_endpoints(e) for e in candidate_set}

        p_orig = float(self.score_fn(set()))

        policy = SelectionPolicy(
            self.policy_name,
            target_time=float(ts),
            candidate_edge_ts=candidate_ts,
            node_dist=node_dist,
            edge_endpoints=edge_endpoints,
            rng=self._rng,
            event_impact=None,
        )

        # Maintain current omitted set
        current: Set[int] = set()
        best_set: Set[int] = set()
        best_score = 0.0
        best_p = p_orig

        it = 0
        while it < self.it_max:
            # Children are formed by adding one new event
            remaining = [e for e in candidate_set if e not in current]
            if not remaining:
                break

            # Sample l events according to policy ranking
            remaining.sort(key=lambda e: policy.rank_event(e))
            sampled = remaining[: max(1, min(self.l, len(remaining)))]

            # Evaluate sampled candidates
            best_child = None
            best_child_delta = -float("inf")
            best_child_p = None
            for e in sampled:
                p_new = float(self.score_fn(current.union({int(e)})))
                delta = _delta_towards_flip(p_orig, p_new)
                if delta > best_child_delta:
                    best_child_delta = delta
                    best_child = int(e)
                    best_child_p = p_new

            if best_child is None or best_child_p is None:
                break

            current.add(best_child)

            norm_score = _safe_norm_shift(p_orig, best_child_delta)
            if norm_score > best_score:
                best_score = norm_score
                best_set = set(current)
                best_p = float(best_child_p)

            if _is_counterfactual(p_orig, best_child_p):
                break

            it += 1

        omitted = sorted(list(best_set), key=lambda e: policy.rank_event(e))
        return ExplanationResult(
            omitted_edge_idxs=omitted,
            original_score=p_orig,
            perturbed_score=best_p,
            is_counterfactual=_is_counterfactual(p_orig, best_p),
            iterations_run=it,
            best_score=float(best_score),
            policy=policy.name,
            candidate_size=len(candidate_set),
        )


# -----------------------------
# twitter-research/tgn adapter
# -----------------------------

class FilteredNeighborFinder:
    """
    Wrap a TGN NeighborFinder to *filter out* edges by edge_idx.

    This is a non-invasive way to "omit events" for counterfactual scoring,
    without rebuilding adjacency lists inside TGN.

    The wrapper assumes the base neighbor finder provides a method:
        get_temporal_neighbor(source_nodes, timestamps, n_neighbors=...)
    returning (neighbors, edge_idxs, edge_times), typically as numpy arrays.

    We call the base with a slightly larger n_neighbors to have some slack
    after filtering, then we pad/truncate to the requested n_neighbors.
    """

    def __init__(self, base_neighbor_finder, omit_edge_idxs: Set[int], *, extra_neighbors: int = 50):
        self.base = base_neighbor_finder
        self.omit_edge_idxs = set(int(e) for e in omit_edge_idxs)
        self.extra_neighbors = int(extra_neighbors)

        # Keep common attributes to behave like the base object
        for attr in ["uniform", "seed", "adj_list"]:
            if hasattr(base_neighbor_finder, attr):
                setattr(self, attr, getattr(base_neighbor_finder, attr))

    def get_temporal_neighbor(
        self,
        source_nodes,
        timestamps,
        n_neighbors: int = 20,
        num_neighbors: Optional[int] = None,
        edge_idx_preserve_list: Optional[Sequence[int]] = None,
    ):
        if num_neighbors is not None:
            n_neighbors = int(num_neighbors)
        # Ask base for more neighbors, then filter
        req = int(n_neighbors)
        extra = max(0, self.extra_neighbors)
        base_k = req + extra

        # Call base with positional args to avoid kw mismatches across NeighborFinder variants.
        if edge_idx_preserve_list is not None:
            try:
                neigh, eidx, et = self.base.get_temporal_neighbor(source_nodes, timestamps, base_k, edge_idx_preserve_list)
            except TypeError:
                neigh, eidx, et = self.base.get_temporal_neighbor(source_nodes, timestamps, base_k)
        else:
            neigh, eidx, et = self.base.get_temporal_neighbor(source_nodes, timestamps, base_k)

        neigh_np = np.asarray(neigh)
        eidx_np = np.asarray(eidx)
        et_np = np.asarray(et)

        out_neigh = np.zeros((neigh_np.shape[0], req), dtype=neigh_np.dtype)
        out_eidx = np.zeros((eidx_np.shape[0], req), dtype=eidx_np.dtype)
        out_et = np.zeros((et_np.shape[0], req), dtype=et_np.dtype)

        omit = self.omit_edge_idxs
        if omit:
            omit_arr = np.fromiter(omit, dtype=eidx_np.dtype, count=len(omit))
        else:
            omit_arr = None

        if edge_idx_preserve_list is not None:
            preserve_arr = np.asarray(edge_idx_preserve_list, dtype=eidx_np.dtype).reshape(-1)
        else:
            preserve_arr = None

        for i in range(neigh_np.shape[0]):
            row_e = eidx_np[i]
            if preserve_arr is not None:
                mask = np.isin(row_e, preserve_arr)
            else:
                mask = np.ones_like(row_e, dtype=bool)

            if omit_arr is not None:
                mask = mask & ~np.isin(row_e, omit_arr)

            keep_idx = np.nonzero(mask)[0][:req]

            k = len(keep_idx)
            if k > 0:
                out_neigh[i, :k] = neigh_np[i, keep_idx]
                out_eidx[i, :k] = eidx_np[i, keep_idx]
                out_et[i, :k] = et_np[i, keep_idx]

        return out_neigh, out_eidx, out_et


class TGNLinkPredictor:
    """
    Small helper that turns the official twitter-research/tgn model into a
    `score_fn(omit_edge_idxs) -> logit` callable expected by CoDyExplainer.

    It works by:
      - restoring a fixed memory snapshot (optional but recommended)
      - swapping the model's neighbor finder with a FilteredNeighborFinder
      - calling model.compute_edge_probabilities(...) for the explained edge
      - converting probability -> logit if needed

    This wrapper tries to be robust to small API differences across forks,
    but you may still need tiny adjustments depending on your local TGN code.
    """

    def __init__(
        self,
        *,
        model,
        explained_src: int,
        explained_dst: int,
        explained_ts: float,
        explained_edge_idx: int,
        n_neighbors: int,
        num_nodes: int,
        device=None,
        memory_snapshot: Optional[dict] = None,
        assume_model_returns_prob: Optional[bool] = None,
        extra_neighbors: int = 50,
    ) -> None:
        if torch is None:
            raise ImportError("torch is required for TGNLinkPredictor")

        self.model = model
        self.src = int(explained_src)
        self.dst = int(explained_dst)
        self.ts = float(explained_ts)
        self.edge_idx = int(explained_edge_idx)
        self.n_neighbors = int(n_neighbors)
        self.num_nodes = int(num_nodes)
        self.device = device if device is not None else next(model.parameters()).device
        self.extra_neighbors = int(extra_neighbors)

        self._orig_neighbor_finder = self._get_neighbor_finder(model)
        if self._orig_neighbor_finder is None:
            raise ValueError("Could not locate model neighbor_finder. Provide a model from twitter-research/tgn.")

        self._memory_snapshot = memory_snapshot  # may be None
        self._assume_model_returns_prob = assume_model_returns_prob

        self.model.eval()

    # ---- neighbor finder plumbing ----

    @staticmethod
    def _get_neighbor_finder(model):
        if hasattr(model, "neighbor_finder"):
            return getattr(model, "neighbor_finder")
        # Common in some forks: embedding_module keeps it
        for attr in ["embedding_module", "temporal_embedding_module"]:
            if hasattr(model, attr):
                mod = getattr(model, attr)
                if hasattr(mod, "neighbor_finder"):
                    return getattr(mod, "neighbor_finder")
        return None

    @staticmethod
    def _set_neighbor_finder(model, neighbor_finder) -> None:
        if hasattr(model, "set_neighbor_finder") and callable(getattr(model, "set_neighbor_finder")):
            model.set_neighbor_finder(neighbor_finder)
            return
        if hasattr(model, "neighbor_finder"):
            setattr(model, "neighbor_finder", neighbor_finder)
        for attr in ["embedding_module", "temporal_embedding_module"]:
            if hasattr(model, attr):
                mod = getattr(model, attr)
                if hasattr(mod, "neighbor_finder"):
                    setattr(mod, "neighbor_finder", neighbor_finder)

    # ---- memory snapshot plumbing ----

    @staticmethod
    def snapshot_memory(model) -> dict:
        """
        Best-effort snapshot of TGN memory state.
        Works for the official twitter-research/tgn memory module.
        """
        snap: dict = {}
        if not hasattr(model, "memory"):
            return snap
        mem = model.memory

        # Standard fields in twitter-research/tgn:
        #   mem.memory (Tensor [n_nodes, mem_dim])
        #   mem.last_update (Tensor [n_nodes])
        if hasattr(mem, "memory"):
            snap["memory"] = mem.memory.detach().clone()
        if hasattr(mem, "last_update"):
            snap["last_update"] = mem.last_update.detach().clone()

        # Some versions store raw messages in a dict
        for key in ["messages", "message_store", "raw_messages", "stored_messages"]:
            if hasattr(mem, key):
                snap[key] = copy.deepcopy(getattr(mem, key))

        return snap

    @staticmethod
    def restore_memory(model, snapshot: dict) -> None:
        if not snapshot or not hasattr(model, "memory"):
            return
        mem = model.memory

        if "memory" in snapshot and hasattr(mem, "memory"):
            mem.memory.data.copy_(snapshot["memory"].data)
        if "last_update" in snapshot and hasattr(mem, "last_update"):
            mem.last_update.data.copy_(snapshot["last_update"].data)

        for key in ["messages", "message_store", "raw_messages", "stored_messages"]:
            if key in snapshot and hasattr(mem, key):
                setattr(mem, key, copy.deepcopy(snapshot[key]))

        # Clear any additional buffers if present
        if hasattr(mem, "clear_messages") and callable(getattr(mem, "clear_messages")):
            mem.clear_messages()

    # ---- scoring ----

    def score(self, omit_edge_idxs: Set[int]) -> float:
        """
        Score the explained edge with given omitted historical edge_idxs.
        Returns a logit-like scalar (threshold at 0).
        """
        # Restore memory snapshot for repeatable scoring
        if self._memory_snapshot is not None:
            self.restore_memory(self.model, self._memory_snapshot)

        # Swap neighbor finder
        filtered = FilteredNeighborFinder(
            self._orig_neighbor_finder, omit_edge_idxs, extra_neighbors=self.extra_neighbors
        )
        self._set_neighbor_finder(self.model, filtered)

        try:
            # Run model for the single edge
            with torch.no_grad():
                src_t = np.asarray([self.src], dtype=np.int64)
                dst_t = np.asarray([self.dst], dtype=np.int64)
                ts_t = np.asarray([self.ts], dtype=np.float32)
                eidx_t = np.asarray([self.edge_idx], dtype=np.int64)

                # Dummy negative node (required by official API)
                neg = (self.dst + 1) % self.num_nodes
                if neg == self.dst:
                    neg = (neg + 1) % self.num_nodes
                neg_t = np.asarray([neg], dtype=np.int64)

                if not hasattr(self.model, "compute_edge_probabilities"):
                    raise AttributeError("TGN model does not expose compute_edge_probabilities")

                out = self.model.compute_edge_probabilities(
                    src_t, dst_t, neg_t, ts_t, eidx_t, self.n_neighbors
                )

                # Handle different return formats
                if isinstance(out, (tuple, list)) and len(out) >= 1:
                    pos_out = out[0]
                else:
                    pos_out = out

                pos_val = float(pos_out.detach().cpu().reshape(-1)[0].item())
        finally:
            # Restore original neighbor finder
            self._set_neighbor_finder(self.model, self._orig_neighbor_finder)

        # Decide whether pos_val is prob or logit
        if self._assume_model_returns_prob is None:
            # Heuristic: if in [0,1], likely probability
            assume_prob = (0.0 <= pos_val <= 1.0)
        else:
            assume_prob = bool(self._assume_model_returns_prob)

        return _logit(pos_val) if assume_prob else pos_val

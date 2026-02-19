
"""
MotifMix-TGN: synthetic temporal edge-addition dataset with *ground-truth causal edges*.

Usage (CLI):
  python motifmix_tgn_generator.py --out_dir motifmix_tgn --n_nodes 1000 --n_events 50000 --seed 42

Outputs:
  - events.csv (temporal edge stream; each event has cause_event_ids as JSON list)
  - nodes.csv  (node metadata)
  - README.txt (format + params)

Rules:
  0: noise/random edge (no causes)
  1: triadic closure: (u-w) and (v-w) -> add (u-v)
  2: ordered 2-hop: (u-w) then (w-v) -> add (u-v)
  3: two-witness with opposite order: need 2 witnesses w1,w2 with opposite ordering
  4: K-witness threshold: add (u-v) if u and v share >=K recent witnesses
"""
from __future__ import annotations
import argparse, random, json, os, textwrap
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import pandas as pd

@dataclass
class Event:
    event_id: int
    src: int
    dst: int
    t: int
    rule_id: int
    causes: List[int]

class MotifMixTGNGenerator:
    RULE_NAMES = {
        0: "noise",
        1: "triadic_closure",
        2: "two_hop_ordered",
        3: "two_witness_opposite_order",
        4: "k_witness_threshold"
    }

    def __init__(
        self,
        n_nodes: int,
        n_events: int,
        seed_edges: int,
        seed: int,
        weights: Tuple[float, float, float, float, float],
        W1: int, W2: int, W3: int, W4: int,
        K: int,
        n_communities: int,
        p_intra: float,
        max_tries_per_event: int = 120,
    ):
        assert n_nodes > 5
        assert n_events > seed_edges
        assert abs(sum(weights) - 1.0) < 1e-6
        assert K >= 2

        self.n_nodes = n_nodes
        self.n_events = n_events
        self.seed_edges = seed_edges
        self.seed = seed
        self.weights = weights
        self.W1, self.W2, self.W3, self.W4 = W1, W2, W3, W4
        self.K = K
        self.n_communities = n_communities
        self.p_intra = p_intra
        self.max_tries_per_event = max_tries_per_event

        self.rng = random.Random(seed)

        # node metadata
        self.community = [self.rng.randrange(n_communities) for _ in range(n_nodes)]

        # graph state
        self.adj = [set() for _ in range(n_nodes)]  # neighbors
        self.edge_event: Dict[Tuple[int, int], int] = {}  # undirected (min,max)->event_id
        self.node_events: List[List[int]] = [[] for _ in range(n_nodes)]  # incident event ids by time

        # event storage
        self.events: List[Event] = []

    def _key(self, u: int, v: int) -> Tuple[int, int]:
        return (u, v) if u < v else (v, u)

    def _edge_exists(self, u: int, v: int) -> bool:
        return self._key(u, v) in self.edge_event

    def _add_edge(self, u: int, v: int, t: int, rule_id: int, causes: List[int]) -> int:
        assert u != v
        k = self._key(u, v)
        if k in self.edge_event:
            raise RuntimeError("duplicate edge attempted")
        eid = len(self.events)
        self.edge_event[k] = eid
        self.adj[u].add(v)
        self.adj[v].add(u)
        self.node_events[u].append(eid)
        self.node_events[v].append(eid)
        self.events.append(Event(event_id=eid, src=u, dst=v, t=t, rule_id=rule_id, causes=causes))
        return eid

    def _other(self, eid: int, node: int) -> int:
        e = self.events[eid]
        return e.dst if e.src == node else e.src

    def _time(self, eid: int) -> int:
        return self.events[eid].t

    def _recent_incident_eids(self, node: int, t_now: int, W: int) -> List[int]:
        out = []
        for eid in reversed(self.node_events[node]):
            if t_now - self._time(eid) > W:
                break
            out.append(eid)
        return out

    def _recent_neighbor_dict(self, node: int, t_now: int, W: int) -> Dict[int, Tuple[int, int]]:
        d = {}
        for eid in self._recent_incident_eids(node, t_now, W):
            nb = self._other(eid, node)
            d[nb] = (eid, self._time(eid))
        return d

    def _sample_node_pair_by_community(self) -> Tuple[int, int]:
        if self.rng.random() < self.p_intra:
            c = self.rng.randrange(self.n_communities)
            candidates = [i for i in range(self.n_nodes) if self.community[i] == c]
            if len(candidates) >= 2:
                u, v = self.rng.sample(candidates, 2)
            else:
                u, v = self.rng.sample(range(self.n_nodes), 2)
        else:
            u, v = self.rng.sample(range(self.n_nodes), 2)
        return u, v

    def _make_noise_edge(self, t_now: int) -> Optional[Tuple[int, int, List[int]]]:
        for _ in range(self.max_tries_per_event):
            u, v = self._sample_node_pair_by_community()
            if u != v and not self._edge_exists(u, v):
                return u, v, []
        return None

    def _make_triadic(self, t_now: int) -> Optional[Tuple[int, int, List[int]]]:
        for _ in range(self.max_tries_per_event):
            w = self.rng.randrange(self.n_nodes)
            rec = self._recent_incident_eids(w, t_now, self.W1)
            if len(rec) < 2:
                continue
            e1, e2 = self.rng.sample(rec, 2)
            u = self._other(e1, w)
            v = self._other(e2, w)
            if u == v or self._edge_exists(u, v):
                continue
            return u, v, [e1, e2]
        return None

    def _make_two_hop_ordered(self, t_now: int) -> Optional[Tuple[int, int, List[int]]]:
        for _ in range(self.max_tries_per_event):
            w = self.rng.randrange(self.n_nodes)
            rec = self._recent_incident_eids(w, t_now, self.W2)
            if len(rec) < 2:
                continue
            rec_chrono = sorted(rec, key=lambda eid: self._time(eid))
            i, j = sorted(self.rng.sample(range(len(rec_chrono)), 2))
            e1, e2 = rec_chrono[i], rec_chrono[j]
            u = self._other(e1, w)
            v = self._other(e2, w)
            if u == v or self._edge_exists(u, v):
                continue
            return u, v, [e1, e2]
        return None

    def _make_two_witness_opposite_order(self, t_now: int) -> Optional[Tuple[int, int, List[int]]]:
        for _ in range(self.max_tries_per_event):
            w1 = self.rng.randrange(self.n_nodes)
            rec_w1 = self._recent_incident_eids(w1, t_now, self.W3)
            if len(rec_w1) < 2:
                continue
            e_a, e_b = self.rng.sample(rec_w1, 2)
            u = self._other(e_a, w1)
            v = self._other(e_b, w1)
            if u == v or self._edge_exists(u, v):
                continue

            # Pair causes consistently as (u,w1) and (v,w1)
            if self._other(e_a, w1) == u and self._other(e_b, w1) == v:
                e_u_w1, e_v_w1 = e_a, e_b
                t_u_w1, t_v_w1 = self._time(e_a), self._time(e_b)
            elif self._other(e_a, w1) == v and self._other(e_b, w1) == u:
                e_u_w1, e_v_w1 = e_b, e_a
                t_u_w1, t_v_w1 = self._time(e_b), self._time(e_a)
            else:
                continue

            s1 = (t_u_w1 < t_v_w1)

            du = self._recent_neighbor_dict(u, t_now, self.W3)
            dv = self._recent_neighbor_dict(v, t_now, self.W3)
            common = set(du.keys()) & set(dv.keys())
            common.discard(w1)
            if not common:
                continue

            candidates = []
            for w2 in common:
                e_u_w2, t_u_w2 = du[w2]
                e_v_w2, t_v_w2 = dv[w2]
                s2 = (t_u_w2 < t_v_w2)
                if s2 != s1:
                    candidates.append((w2, e_u_w2, e_v_w2))
            if not candidates:
                continue

            _, e_u_w2, e_v_w2 = self.rng.choice(candidates)
            return u, v, [e_u_w1, e_v_w1, e_u_w2, e_v_w2]
        return None

    def _make_k_witness_threshold(self, t_now: int) -> Optional[Tuple[int, int, List[int]]]:
        for _ in range(self.max_tries_per_event):
            w = self.rng.randrange(self.n_nodes)
            rec_w = self._recent_incident_eids(w, t_now, self.W4)
            if len(rec_w) < 2:
                continue
            e1, e2 = self.rng.sample(rec_w, 2)
            u = self._other(e1, w)
            v = self._other(e2, w)
            if u == v or self._edge_exists(u, v):
                continue

            du = self._recent_neighbor_dict(u, t_now, self.W4)
            dv = self._recent_neighbor_dict(v, t_now, self.W4)
            common = list(set(du.keys()) & set(dv.keys()))
            if len(common) < self.K:
                continue

            scored = []
            for wi in common:
                _, tu = du[wi]
                _, tv = dv[wi]
                scored.append((max(tu, tv), wi))
            scored.sort(reverse=True)
            witnesses = [wi for _, wi in scored[: self.K]]

            causes = []
            for wi in witnesses:
                causes.append(du[wi][0])
                causes.append(dv[wi][0])
            return u, v, causes
        return None

    def generate(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        t = 0
        for _ in range(self.seed_edges):
            cand = self._make_noise_edge(t)
            if cand is None:
                break
            u, v, causes = cand
            self._add_edge(u, v, t, rule_id=0, causes=causes)
            t += 1

        rule_ids = [0, 1, 2, 3, 4]
        for _ in range(len(self.events), self.n_events):
            t_now = t
            rule_id = self.rng.choices(rule_ids, weights=self.weights, k=1)[0]

            if rule_id == 0:
                cand = self._make_noise_edge(t_now)
            elif rule_id == 1:
                cand = self._make_triadic(t_now)
            elif rule_id == 2:
                cand = self._make_two_hop_ordered(t_now)
            elif rule_id == 3:
                cand = self._make_two_witness_opposite_order(t_now)
            else:
                cand = self._make_k_witness_threshold(t_now)

            if cand is None:
                rule_id = 0
                cand = self._make_noise_edge(t_now)
                if cand is None:
                    break

            u, v, causes = cand
            # random orientation in output file
            if self.rng.random() < 0.5:
                src, dst = u, v
            else:
                src, dst = v, u

            self._add_edge(src, dst, t_now, rule_id=rule_id, causes=causes)
            t += 1

        events_df = pd.DataFrame([
            {
                "event_id": e.event_id,
                "src": e.src,
                "dst": e.dst,
                "timestamp": e.t,
                "label": 1,
                "rule_id": e.rule_id,
                "rule_name": self.RULE_NAMES[e.rule_id],
                "cause_event_ids": json.dumps(e.causes),
                "n_causes": len(e.causes),
            }
            for e in self.events
        ])

        # time split
        n = len(events_df)
        train_end = int(0.70 * n)
        val_end   = int(0.85 * n)
        events_df["split"] = "test"
        events_df.loc[:train_end-1, "split"] = "train"
        events_df.loc[train_end:val_end-1, "split"] = "val"

        nodes_df = pd.DataFrame({
            "node_id": list(range(self.n_nodes)),
            "community": self.community,
        })
        return events_df, nodes_df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--n_nodes", type=int, default=1000)
    ap.add_argument("--n_events", type=int, default=50000)
    ap.add_argument("--seed_edges", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--W1", type=int, default=200)
    ap.add_argument("--W2", type=int, default=400)
    ap.add_argument("--W3", type=int, default=800)
    ap.add_argument("--W4", type=int, default=1200)
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--n_communities", type=int, default=20)
    ap.add_argument("--p_intra", type=float, default=0.70)
    args = ap.parse_args()

    weights = (0.15, 0.35, 0.25, 0.15, 0.10)

    gen = MotifMixTGNGenerator(
        n_nodes=args.n_nodes,
        n_events=args.n_events,
        seed_edges=args.seed_edges,
        seed=args.seed,
        weights=weights,
        W1=args.W1, W2=args.W2, W3=args.W3, W4=args.W4,
        K=args.K,
        n_communities=args.n_communities,
        p_intra=args.p_intra,
    )
    events_df, nodes_df = gen.generate()

    os.makedirs(args.out_dir, exist_ok=True)
    events_path = os.path.join(args.out_dir, "events.csv")
    nodes_path  = os.path.join(args.out_dir, "nodes.csv")
    readme_path = os.path.join(args.out_dir, "README.txt")

    events_df.to_csv(events_path, index=False)
    nodes_df.to_csv(nodes_path, index=False)

    readme = f"""\
MotifMix-TGN synthetic temporal dataset with ground-truth causal edges

Files:
- events.csv : event stream sorted by timestamp
- nodes.csv  : node metadata (community)

events.csv columns:
- event_id, src, dst, timestamp
- label: always 1 (positives)
- rule_id, rule_name: generation mechanism (metadata)
- cause_event_ids: JSON list of causal event_ids
- n_causes
- split: train/val/test (time split)

Params:
- n_nodes={args.n_nodes}
- n_events={len(events_df)}
- seed_edges={args.seed_edges}
- seed={args.seed}
- W1={args.W1}, W2={args.W2}, W3={args.W3}, W4={args.W4}
- K={args.K}
- n_communities={args.n_communities}, p_intra={args.p_intra}
"""
    with open(readme_path, "w") as f:
        f.write(textwrap.dedent(readme))

    print("Wrote:")
    print(" ", events_path)
    print(" ", nodes_path)
    print(" ", readme_path)
    print("n_events:", len(events_df))

if __name__ == "__main__":
    main()

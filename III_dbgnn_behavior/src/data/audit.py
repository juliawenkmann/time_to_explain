
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


# Default URL used by pathpyG tutorial (mirrors data.temporal_clusters loader).
DEFAULT_REMOTE_TEDGES_URL = (
    "https://raw.githubusercontent.com/pathpy/pathpyG/refs/heads/main/docs/data/temporal_clusters.tedges"
)


def cluster_id(node: int, *, cluster_size: int = 10) -> int:
    """Ground-truth cluster id for temporal_clusters-like datasets (node_id // cluster_size)."""
    return int(node) // int(cluster_size)


def load_tedges_df(
    local_path: str | Path,
    *,
    remote_url: Optional[str] = None,
    names: Tuple[str, str, str] = ("u", "v", "t"),
) -> pd.DataFrame:
    """Load a .tedges file into a dataframe with columns (u, v, t).

    The expected file format is one edge-event per line:
        u,v,t

    Args:
        local_path: Local file path. If it exists, it will be used.
        remote_url: If local_path doesn't exist and this URL is provided, the file will be downloaded.
        names: Column names for (u, v, t).

    Returns:
        DataFrame sorted by time with integer dtypes.
    """
    path = Path(local_path)
    if path.exists():
        df = pd.read_csv(path, header=None, names=list(names))
    else:
        if remote_url is None:
            raise FileNotFoundError(f"Dataset file not found: {path}")
        # Download on demand (kept minimal; callers can still prefer local files).
        import tempfile
        from urllib import request

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir) / path.name
            request.urlretrieve(str(remote_url), tmp)  # nosec - controlled URL
            df = pd.read_csv(tmp, header=None, names=list(names))

    # Enforce integer dtypes (fail fast if parsing goes wrong).
    for c in names:
        df[c] = df[c].astype(int)

    # Sort by time for temporal analysis.
    df = df.sort_values(names[2]).reset_index(drop=True)
    return df


def shuffle_timestamps(df: pd.DataFrame, *, seed: int = 0) -> pd.DataFrame:
    """Shuffle timestamps (null model) while preserving the static edge multiset."""
    if not {"u", "v", "t"} <= set(df.columns):
        raise ValueError("Expected columns {'u','v','t'}")

    rng = np.random.default_rng(int(seed))
    out = df.copy()
    out["t"] = rng.permutation(out["t"].to_numpy())
    out = out.sort_values("t").reset_index(drop=True)
    return out


def basic_sanity(df: pd.DataFrame) -> Dict[str, object]:
    """Basic dataset sanity checks and summary stats (no plotting)."""
    if not {"u", "v", "t"} <= set(df.columns):
        raise ValueError("Expected columns {'u','v','t'}")

    u = df["u"].to_numpy()
    v = df["v"].to_numpy()
    t = df["t"].to_numpy()

    nodes = np.unique(np.concatenate([u, v]))
    times = np.unique(t)

    t_min = int(t.min()) if len(t) else None
    t_max = int(t.max()) if len(t) else None

    times_unique = (len(times) == len(t))
    times_contiguous = False
    if len(times) and times_unique:
        # contiguous means: max-min+1 == n AND matches the full integer range
        times_contiguous = (t_max - t_min + 1 == len(times)) and np.all(times == np.arange(t_min, t_max + 1))

    n_self_loops = int(np.sum(u == v))

    return {
        "n_events": int(len(df)),
        "n_nodes": int(len(nodes)),
        "nodes_min": int(nodes.min()) if len(nodes) else None,
        "nodes_max": int(nodes.max()) if len(nodes) else None,
        "t_min": t_min,
        "t_max": t_max,
        "times_unique": bool(times_unique),
        "times_contiguous": bool(times_contiguous),
        "n_self_loops": n_self_loops,
        "frac_self_loops": float(n_self_loops / len(df)) if len(df) else 0.0,
    }


def static_cluster_matrix(
    df: pd.DataFrame, *, n_clusters: int = 3, cluster_size: int = 10
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compute the 3x3 cluster mixing matrix on the static edge multiset.

    Returns:
        counts: [n_clusters, n_clusters] counts
        fracs:  counts normalized by total edges
        frac_within: trace(fracs)
    """
    u = df["u"].to_numpy()
    v = df["v"].to_numpy()
    cu = (u // int(cluster_size)).astype(int)
    cv = (v // int(cluster_size)).astype(int)

    counts = np.zeros((int(n_clusters), int(n_clusters)), dtype=int)
    # Only count pairs that fall into the specified cluster range.
    mask = (cu >= 0) & (cu < n_clusters) & (cv >= 0) & (cv < n_clusters)
    np.add.at(counts, (cu[mask], cv[mask]), 1)

    total = counts.sum()
    fracs = counts / total if total else counts.astype(float)
    frac_within = float(np.trace(fracs)) if total else 0.0
    return counts, fracs, frac_within


def extract_causal_triples(
    df: pd.DataFrame, *, delta: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract causal length-2 paths (triples) for a directed temporal edge sequence.

    We follow the tutorial's definition for a length-2 causal path:

        (u, v; t)  ->  (v, w; t')

    iff
      - u->v and v->w share the intermediate node v (directed handoff), AND
      - 0 < t' - t <= delta.

    This function assumes timestamps are integers and uses exact time differences.
    For temporal_clusters.tedges this matches delta=1 and t=0..59999 exactly.

    Returns:
        triples: int array of shape [K, 3] with rows (u, v, w)
        dts:     int array of shape [K] with the corresponding time differences (t' - t)
    """
    if delta < 1:
        raise ValueError("delta must be >= 1")

    u = df["u"].to_numpy().astype(int)
    v = df["v"].to_numpy().astype(int)
    t = df["t"].to_numpy().astype(int)

    # Map time -> row index (requires unique timestamps).
    if len(np.unique(t)) != len(t):
        raise ValueError("extract_causal_triples assumes unique timestamps (one event per time)")

    time_to_idx = {int(tt): i for i, tt in enumerate(t)}

    triples = []
    dts = []
    for i, tt in enumerate(t):
        tt = int(tt)
        for dt in range(1, int(delta) + 1):
            j = time_to_idx.get(tt + dt)
            if j is None:
                continue
            # Directed handoff: first edge ends at v[i], second edge starts at u[j].
            if u[j] == v[i]:
                triples.append((u[i], v[i], v[j]))
                dts.append(dt)

    if not triples:
        return np.zeros((0, 3), dtype=int), np.zeros((0,), dtype=int)

    return np.asarray(triples, dtype=int), np.asarray(dts, dtype=int)


@dataclass(frozen=True)
class CausalTripleStats:
    """Summary stats for (u,v,w) causal triples."""

    n_triples: int
    # delta=1 diagnostic: fraction of t such that (u_t, v_t) hands off into the next event.
    handoff_rate_dt1: Optional[float]

    # Key tutorial diagnostics:
    p_stay_given_prev_within: float
    p_stay_given_prev_cross: float
    p_stay_overall: float

    # Counts (helpful for debugging)
    n_prev_within: int
    n_prev_cross: int


def summarize_causal_triples(
    triples: np.ndarray,
    dts: np.ndarray,
    *,
    cluster_size: int = 10,
) -> CausalTripleStats:
    """Compute the key conditional probabilities used in the tutorial."""
    if triples.ndim != 2 or triples.shape[1] != 3:
        raise ValueError("triples must have shape [K, 3]")
    if len(triples) != len(dts):
        raise ValueError("triples and dts must have the same length")

    if len(triples) == 0:
        return CausalTripleStats(
            n_triples=0,
            handoff_rate_dt1=None,
            p_stay_given_prev_within=float("nan"),
            p_stay_given_prev_cross=float("nan"),
            p_stay_overall=float("nan"),
            n_prev_within=0,
            n_prev_cross=0,
        )

    u = triples[:, 0]
    v = triples[:, 1]
    w = triples[:, 2]

    cu = (u // int(cluster_size)).astype(int)
    cv = (v // int(cluster_size)).astype(int)
    cw = (w // int(cluster_size)).astype(int)

    prev_within = cu == cv
    stay = cw == cv

    n_prev_within = int(prev_within.sum())
    n_prev_cross = int((~prev_within).sum())

    def _safe_mean(x: np.ndarray) -> float:
        return float(x.mean()) if len(x) else float("nan")

    p_stay_within = _safe_mean(stay[prev_within])
    p_stay_cross = _safe_mean(stay[~prev_within])
    p_stay_overall = _safe_mean(stay)

    # delta=1 diagnostic
    dt1 = (dts == 1)
    handoff_rate_dt1 = _safe_mean(dt1)  # among extracted triples (not among all possible times)
    # NOTE: in temporal_clusters with unique contiguous times, all extracted triples have dt=1,
    # so this will be 1.0. For a "handoff rate per time step", compute it separately in the notebook.

    return CausalTripleStats(
        n_triples=int(len(triples)),
        handoff_rate_dt1=handoff_rate_dt1,
        p_stay_given_prev_within=p_stay_within,
        p_stay_given_prev_cross=p_stay_cross,
        p_stay_overall=p_stay_overall,
        n_prev_within=n_prev_within,
        n_prev_cross=n_prev_cross,
    )


def handoff_rate_per_timestep(df: pd.DataFrame) -> float:
    """For delta=1 contiguous times: fraction of consecutive events that form a directed handoff.

    Specifically:
        rate = mean( v_t == u_{t+1} )

    This is the cleanest "how many causal paths exist?" statistic for temporal_clusters.
    """
    df = df.sort_values("t").reset_index(drop=True)
    u = df["u"].to_numpy().astype(int)
    v = df["v"].to_numpy().astype(int)
    if len(df) < 2:
        return float("nan")
    return float(np.mean(v[:-1] == u[1:]))


def phase_a_audit(
    df: pd.DataFrame,
    *,
    n_clusters: int = 3,
    cluster_size: int = 10,
    delta: int = 1,
    shuffle_seed: int = 0,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Run the Phase A audit for a dataframe + a timestamp-shuffled null model.

    Returns:
        stats_df: rows = ["real", "shuffled"], with key scalar metrics
        static_fracs: static cluster mixing matrix (fractions), same for both
    """
    df = df.sort_values("t").reset_index(drop=True)

    # Static topology stats (same for real/shuffled)
    _, static_fracs, frac_within = static_cluster_matrix(df, n_clusters=n_clusters, cluster_size=cluster_size)

    # Causal stats
    triples, dts = extract_causal_triples(df, delta=delta)
    causal = summarize_causal_triples(triples, dts, cluster_size=cluster_size)

    df_shuf = shuffle_timestamps(df, seed=shuffle_seed)
    triples_s, dts_s = extract_causal_triples(df_shuf, delta=delta)
    causal_s = summarize_causal_triples(triples_s, dts_s, cluster_size=cluster_size)

    # Handoff rate per time step (delta=1 diagnostic)
    hr = handoff_rate_per_timestep(df)
    hr_s = handoff_rate_per_timestep(df_shuf)

    rows = []
    for name, c, h in [
        ("real", causal, hr),
        ("shuffled", causal_s, hr_s),
    ]:
        rows.append(
            {
                "name": name,
                "n_events": int(len(df)),
                "n_triples": int(c.n_triples),
                "handoff_rate_per_timestep": float(h),
                "p_stay_given_prev_within": float(c.p_stay_given_prev_within),
                "p_stay_given_prev_cross": float(c.p_stay_given_prev_cross),
                "p_stay_overall": float(c.p_stay_overall),
                "static_frac_within_cluster": float(frac_within),
            }
        )

    stats_df = pd.DataFrame(rows).set_index("name")
    return stats_df, static_fracs

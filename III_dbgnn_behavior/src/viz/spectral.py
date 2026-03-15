"""
Spectral / Laplacian-based visual helpers.

These were originally embedded in notebooks; we keep them here so notebooks stay thin and
to avoid common pitfalls with netzschleuder node-id mappings.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

# SciPy is an indirect dependency of pathpyG; import lazily for robustness.
try:
    import scipy.sparse as sp_sparse
    import scipy.sparse.linalg as sp_linalg
except Exception:  # pragma: no cover
    sp_sparse = None
    sp_linalg = None


def _to_py(x: Any) -> Any:
    """Convert common scalar wrappers (torch / numpy) to python scalars."""
    try:
        # torch scalar
        if hasattr(x, "item") and callable(x.item):
            return x.item()
    except Exception:
        pass
    # numpy scalar
    try:
        if isinstance(x, np.generic):
            return x.item()
    except Exception:
        pass
    return x


def _normalize_node_id(x: Any) -> Union[int, str, Tuple[Any, ...]]:
    """
    Normalize a node id for stable dict lookup.

    - numeric ids -> int
    - tuples/lists -> tuple of normalized components
    - otherwise -> str
    """
    x = _to_py(x)
    if isinstance(x, (tuple, list)):
        return tuple(_normalize_node_id(v) for v in x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    # numeric string?
    try:
        return int(x)
    except Exception:
        return str(x)


def make_id_to_label(*, g1: Any, y_base: np.ndarray) -> Dict[Union[int, str], int]:
    """
    Build a mapping: first-order node-id -> label, robust to netzschleuder graphs where
    node ids are not contiguous 0..N-1 (e.g., missing ids for isolates).

    Parameters
    ----------
    g1:
        First-order graph (m.layers[1]) or any graph that has `mapping.node_ids`.
    y_base:
        Labels aligned to *node indices* (0..N-1), where N == len(g1.mapping.node_ids).
    """
    y_base = np.asarray(y_base).astype(int).ravel()
    node_ids = None
    mapping = getattr(g1, "mapping", None)
    if mapping is not None:
        node_ids = getattr(mapping, "node_ids", None)

    if node_ids is None:
        # Fall back: assume ids are indices.
        return {int(i): int(y_base[i]) for i in range(int(y_base.shape[0]))}

    node_ids_arr = np.asarray(node_ids)
    if node_ids_arr.ndim != 1:
        # For safety; first-order graphs should have 1D ids.
        node_ids_arr = node_ids_arr.reshape(-1)

    if int(node_ids_arr.shape[0]) != int(y_base.shape[0]):
        raise ValueError(
            f"Length mismatch when building id->label mapping: "
            f"len(node_ids)={int(node_ids_arr.shape[0])} vs len(y_base)={int(y_base.shape[0])}."
        )

    out: Dict[Union[int, str], int] = {}
    for i, nid in enumerate(node_ids_arr.tolist()):
        out[_normalize_node_id(nid)] = int(y_base[i])
    return out


def ho_node_class_ids(
    g2: Any,
    *,
    y_base: np.ndarray,
    g1: Optional[Any] = None,
    mixed_class: Optional[int] = None,
) -> List[int]:
    """
    Color 2nd-order (u,v) nodes by:
      - class of v if y[u]==y[v]
      - otherwise `mixed_class`

    Works for netzschleuder where u/v are *node ids* (not indices into y_base).

    Parameters
    ----------
    g2:
        Second-order graph whose nodes are pairs (u,v) (IDs).
    y_base:
        Labels aligned to first-order node *indices*.
    g1:
        First-order graph (needed for ID->index mapping). If None, we assume IDs == indices.
    mixed_class:
        Optional explicit class id for mixed. Default = num_base_classes.
    """
    y_base = np.asarray(y_base).astype(int).ravel()
    if mixed_class is None:
        mixed_class = int(len(sorted(set(y_base.tolist()))))

    if g1 is None:
        id_to_label = {int(i): int(y_base[i]) for i in range(int(y_base.shape[0]))}
    else:
        id_to_label = make_id_to_label(g1=g1, y_base=y_base)

    out: List[int] = []

    nodes = getattr(g2, "nodes", None)
    if nodes is None:
        raise ValueError("g2 has no `.nodes` attribute.")

    for ho in nodes:
        ho = _normalize_node_id(ho)

        # Most common: tuple(u,v)
        if isinstance(ho, tuple) and len(ho) >= 2:
            u_id, v_id = ho[0], ho[1]
        else:
            # Try to parse string "(u,v)" or "u,v"
            s = str(ho).strip().strip("()")
            parts = [p.strip() for p in s.split(",") if p.strip() != ""]
            if len(parts) < 2:
                out.append(int(mixed_class))
                continue
            try:
                u_id = _normalize_node_id(parts[0])
                v_id = _normalize_node_id(parts[1])
            except Exception:
                out.append(int(mixed_class))
                continue

        yu = id_to_label.get(u_id, None)
        yv = id_to_label.get(v_id, None)
        if yu is None or yv is None:
            out.append(int(mixed_class))
        elif int(yu) == int(yv):
            out.append(int(yv))
        else:
            out.append(int(mixed_class))

    return out


def compute_fiedler_vector(
    g: Any,
    *,
    normalization: str = "rw",
    edge_attr: str = "edge_weight",
    symmetrize: bool = True,
    tol: float = 1e-3,
    maxiter: int = 5000,
    seed: int = 0,
) -> np.ndarray:
    """
    Compute a (fast) approximation of the Fiedler vector (2nd smallest eigenvector)
    of a graph Laplacian.

    Uses sparse solvers; avoids `todense()` which is cubic-time and infeasible for n~5k+.

    Notes
    -----
    For directed graphs, the random-walk Laplacian is not symmetric. By default we
    symmetrize the Laplacian (L+L^T)/2 to obtain a real-valued embedding and to use
    the faster/stabler `eigsh` solver.

    Returns
    -------
    fiedler : np.ndarray shape (n,)
    """
    if sp_sparse is None or sp_linalg is None:
        raise ImportError("scipy is required for compute_fiedler_vector but is not available.")

    # ------------------------------------------------------------------
    # IMPORTANT: pathpyG's `g.laplacian(...)` may return a Laplacian whose
    # shape is smaller than `g.n` when the graph contains nodes with no edges
    # (isolates). In that case, downstream notebook plots that use `range(g.n)`
    # will fail with "x and y must be the same size".
    #
    # To make this robust across datasets and pathpyG versions, we build a
    # Laplacian ourselves from `g.data.edge_index` and (optional) edge weights
    # with an explicit shape `(g.n, g.n)`.
    # ------------------------------------------------------------------

    def _as_numpy(a: Any) -> np.ndarray:
        if a is None:
            raise ValueError("Expected array-like, got None")
        if hasattr(a, "as_tensor"):
            a = a.as_tensor()
        # torch tensor?
        try:
            import torch  # local import

            if torch.is_tensor(a):
                return a.detach().cpu().numpy()
        except Exception:
            pass
        return np.asarray(a)

    # Determine the intended number of nodes.
    n = int(getattr(g, "n", 0) or 0)

    # Extract edges.
    data = getattr(g, "data", None)
    if data is None or not hasattr(data, "edge_index"):
        raise ValueError("Graph has no `data.edge_index`; cannot build Laplacian.")

    edge_index = _as_numpy(getattr(data, "edge_index"))
    if edge_index.ndim != 2:
        edge_index = edge_index.reshape(2, -1)
    if edge_index.shape[0] != 2 and edge_index.shape[1] == 2:
        edge_index = edge_index.T
    if edge_index.shape[0] != 2:
        raise ValueError(f"edge_index must have shape (2, E), got {edge_index.shape}.")

    if edge_index.size == 0:
        # No edges: Fiedler vector is all zeros.
        return np.zeros((max(n, 0),), dtype=float)

    src = edge_index[0].astype(np.int64, copy=False)
    dst = edge_index[1].astype(np.int64, copy=False)

    # If g.n is missing/0, infer from edges.
    if n <= 0:
        n = int(max(src.max(), dst.max()) + 1)

    # Optional edge weights.
    w = None
    if hasattr(data, edge_attr):
        try:
            w = _as_numpy(getattr(data, edge_attr)).astype(float, copy=False)
        except Exception:
            w = None
    if w is None:
        w = np.ones((src.shape[0],), dtype=float)

    # Build adjacency.
    A = sp_sparse.coo_matrix((w, (src, dst)), shape=(n, n)).tocsr()

    # Build Laplacian with requested normalization.
    if normalization in ("rw", "random_walk"):
        deg = np.asarray(A.sum(axis=1)).ravel()
        with np.errstate(divide="ignore"):
            inv_deg = np.where(deg > 0, 1.0 / deg, 0.0)
        Dinv = sp_sparse.diags(inv_deg)
        P = Dinv @ A
        L = sp_sparse.eye(n, format="csr") - P
    elif normalization in ("sym", "symmetric"):
        deg = np.asarray(A.sum(axis=1)).ravel()
        with np.errstate(divide="ignore"):
            inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
        Dinv_sqrt = sp_sparse.diags(inv_sqrt)
        L = sp_sparse.eye(n, format="csr") - (Dinv_sqrt @ A @ Dinv_sqrt)
    elif normalization in ("none", None, ""):
        deg = np.asarray(A.sum(axis=1)).ravel()
        D = sp_sparse.diags(deg)
        L = D - A
    else:
        raise ValueError(f"Unknown normalization={normalization!r}")

    # Optional symmetrization to make the operator symmetric.
    if symmetrize:
        L = (L + L.T) * 0.5
        L = L.tocsr()
    if n < 3:
        return np.zeros((n,), dtype=float)

    # Smallest magnitude eigenvalues for Laplacian (0, lambda2, ...)
    # Note: for disconnected graphs, lambda2 may be 0; the returned eigenvector
    # is still a valid embedding and has the correct length n.
    try:
        vals, vecs = sp_linalg.eigsh(
            L,
            k=2,
            which="SM",
            tol=float(tol),
            maxiter=int(maxiter),
        )
        order = np.argsort(vals)
        f = vecs[:, order[1]]
        return np.asarray(np.real(f)).ravel()
    except Exception as e:
        # As a last resort, dense for small graphs only
        if n <= 1500:
            import scipy.linalg as sp_dense_linalg

            vals, vecs = sp_dense_linalg.eig(np.asarray(L.todense()))
            order = np.argsort(np.real(vals))
            f = vecs[:, order[1]]
            return np.asarray(np.real(f)).ravel()
        raise RuntimeError(f"Failed to compute fiedler vector for n={n}: {e}") from e

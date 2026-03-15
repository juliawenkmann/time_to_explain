"""Notebook helpers for dataset resolution and split reporting."""

from __future__ import annotations

import re
from typing import Dict, Mapping, MutableMapping, Tuple, Any

import numpy as np


def resolve_dataset_name(candidates, dataset_registry, get_dataset_loader):
    """Resolve a dataset candidate list using registry-first semantics."""
    for name in candidates:
        if name in dataset_registry:
            return name

    last_err = None
    for name in candidates:
        try:
            _ = get_dataset_loader(name)
            return name
        except Exception as exc:
            last_err = exc

    if last_err is not None:
        raise last_err
    raise RuntimeError("Could not resolve dataset name from candidate list.")


def label_count_map(ids: np.ndarray, y: np.ndarray) -> Dict[int, int]:
    """Return class-count histogram for a subset of internal node IDs."""
    ids = np.asarray(ids, dtype=int)
    if ids.size == 0:
        return {}
    yy = np.asarray(y, dtype=int)[ids]
    u, c = np.unique(yy, return_counts=True)
    return {int(k): int(v) for k, v in zip(u.tolist(), c.tolist())}


def normalize_loader_kwargs(dataset_kwargs: Mapping[str, Any]) -> Dict[str, Any]:
    """Normalize common kwargs aliases before calling dataset loaders.

    Currently supported:
      - `networks` -> `network` (uses first entry if a list/tuple is provided)
    """
    out = dict(dataset_kwargs)
    if "network" not in out and "networks" in out:
        v = out.pop("networks")
        if isinstance(v, (list, tuple)):
            out["network"] = v[0] if len(v) > 0 else None
        elif isinstance(v, dict):
            out["network"] = next(iter(v.keys()), None)
        else:
            out["network"] = v
    return out


def load_dataset_with_retry(
    loader,
    *,
    device,
    num_test: float,
    seed: int,
    dataset_kwargs: Mapping[str, Any],
) -> Tuple[object, object]:
    """Load dataset with robust unsupported-kwarg handling.

    This prevents infinite retry loops by ensuring that each retry actually
    removes or normalizes at least one kwarg.
    """
    kwargs: MutableMapping[str, Any] = normalize_loader_kwargs(dataset_kwargs)
    dropped: set[str] = set()

    while True:
        try:
            return loader(device=device, num_test=num_test, seed=seed, **kwargs)
        except TypeError as exc:
            m = re.search(r"got an unexpected keyword argument '([^']+)'", str(exc))
            if not m:
                raise

            bad = str(m.group(1))
            normalized = False

            # If loader complains about `network` while we only have `networks`,
            # normalize once and retry.
            if bad == "network" and "network" not in kwargs and "networks" in kwargs:
                norm = normalize_loader_kwargs(kwargs)
                kwargs.clear()
                kwargs.update(norm)
                normalized = True
                print("[loader retry] normalized alias: 'networks' -> 'network'")

            if normalized:
                continue

            # If the offending kwarg is not in the forwarded kwargs, the
            # failure happened in a deeper call stack; surface the original error.
            if bad not in kwargs:
                raise

            # Drop exactly once; repeated rejection indicates a deeper issue.
            if bad in dropped:
                raise RuntimeError(
                    f"Loader keeps rejecting kwarg {bad!r}; current kwargs={dict(kwargs)}"
                ) from exc

            dropped.add(bad)
            kwargs.pop(bad, None)
            print(f"[loader retry] dropping unsupported kwarg: {bad!r}")

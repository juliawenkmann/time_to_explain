"""Parsing utilities for evaluation result files.

The evaluation CSV files in this repository often contain columns that look like
lists/arrays, but are stored as strings, e.g.

- "[32026. 32031. 32016.]"  (NumPy pretty-print)
- "[32016 32026 32031 32038]" (space-separated)
- "[1, 2, 3]" (Python/JSON-ish)

This module provides robust helpers to convert those representations into NumPy
arrays.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

import numpy as np

_INT_RE = re.compile(r"-?\d+")
_FLOAT_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _is_nan_like(value: Any) -> bool:
    """Return True for values that represent missingness."""
    if value is None:
        return True
    # numpy.nan is a float and does not equal itself
    try:
        return bool(isinstance(value, float) and value != value)
    except Exception:
        return False


def parse_int_array(value: Any) -> np.ndarray:
    """Parse a value that represents a list/array of ints into an int ndarray."""
    if _is_nan_like(value):
        return np.asarray([], dtype=int)

    if isinstance(value, np.ndarray):
        return value.astype(int, copy=False)

    if isinstance(value, (list, tuple)):
        return np.asarray(value, dtype=int)

    # pandas can store arrays as strings in CSVs
    if isinstance(value, str):
        s = value.strip()
        if s in {"", "[]", "nan", "NaN", "None"}:
            return np.asarray([], dtype=int)
        nums = _INT_RE.findall(s)
        return np.asarray([int(n) for n in nums], dtype=int)

    # fallback for scalar ints
    try:
        return np.asarray([int(value)], dtype=int)
    except Exception as exc:  # pragma: no cover
        raise TypeError(f"Cannot parse int array from value of type {type(value)}") from exc


def parse_float_array(value: Any) -> np.ndarray:
    """Parse a value that represents a list/array of floats into a float ndarray."""
    if _is_nan_like(value):
        return np.asarray([], dtype=float)

    if isinstance(value, np.ndarray):
        return value.astype(float, copy=False)

    if isinstance(value, (list, tuple)):
        return np.asarray(value, dtype=float)

    if isinstance(value, str):
        s = value.strip()
        if s in {"", "[]", "nan", "NaN", "None"}:
            return np.asarray([], dtype=float)
        nums = _FLOAT_RE.findall(s)
        return np.asarray([float(n) for n in nums], dtype=float)

    try:
        return np.asarray([float(value)], dtype=float)
    except Exception as exc:  # pragma: no cover
        raise TypeError(f"Cannot parse float array from value of type {type(value)}") from exc


def parse_int_array_series(values: Iterable[Any]) -> list[np.ndarray]:
    """Vectorized-ish helper used when normalizing a dataframe column."""
    return [parse_int_array(v) for v in values]


def parse_float_array_series(values: Iterable[Any]) -> list[np.ndarray]:
    """Vectorized-ish helper used when normalizing a dataframe column."""
    return [parse_float_array(v) for v in values]

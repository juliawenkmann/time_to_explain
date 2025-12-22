"""
Notebook helpers for instantiating explainers in one place.
Update the functions below to match the explainers you actually use.
"""
from __future__ import annotations
from typing import Any


def build_pgexplainer(**kwargs) -> Any:
    """
    Example placeholder. Swap in your PGExplainer adapter/construction logic.
    """
    raise NotImplementedError("Implement notebooks.src.explainer_factory.build_pgexplainer.")


def build_custom(**kwargs) -> Any:
    """
    Generic hook for any other explainer you need in notebooks.
    """
    raise NotImplementedError("Implement notebooks.src.explainer_factory.build_custom.")

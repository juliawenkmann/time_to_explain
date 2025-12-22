"""
Notebook helpers for building models in a single place.

Override `load_model` with your project-specific logic and keep the notebook
cells small (call into this function instead of re-implementing it everywhere).
"""
from __future__ import annotations
from typing import Any, Dict


def load_model(**kwargs) -> Any:
    """
    Return a ModelProtocol-compatible object.

    Replace this stub with your own implementation, e.g.:
      - load a backbone checkpoint from resources/models/<dataset>/...
      - wrap it with time_to_explain.adapters.TemporalGNNModelAdapter
    """
    raise NotImplementedError("Implement notebooks.src.model_loader.load_model to return your model.")


def load_kwargs_from_notebook(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Optional convenience for notebooks: mutate/validate kwargs here before
    handing them to `load_model`.
    """
    return dict(params)

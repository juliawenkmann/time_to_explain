from __future__ import annotations

from pathlib import Path
import runpy
from typing import Any, Mapping

_SCRIPT_MAP = {
    "01_train_dbgnn": "01_train_dbgnn.py",
    "02_counterfactual": "02_counterfactual.py",
    "03_higher_order_effects": "03_higher_order_effects.py",
    "04_gcn_structure_removal": "04_gcn_structure_removal.py",
    "05_ba_shapes_structure_removal": "05_ba_shapes_structure_removal.py",
}


def available_notebook_scripts() -> tuple[str, ...]:
    return tuple(sorted(_SCRIPT_MAP.keys()))


def run_notebook_script(name: str, overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Execute a notebook script from `src/notebook_scripts` and return its globals."""
    if name not in _SCRIPT_MAP:
        raise ValueError(f"Unknown notebook script {name!r}. Available: {', '.join(available_notebook_scripts())}")

    script_path = Path(__file__).resolve().parent / "notebook_scripts" / _SCRIPT_MAP[name]
    if not script_path.exists():
        raise FileNotFoundError(f"Notebook script not found: {script_path}")

    init_globals: dict[str, Any] = {"__name__": "__main__"}
    if overrides:
        init_globals.update(dict(overrides))

    return runpy.run_path(str(script_path), init_globals=init_globals)

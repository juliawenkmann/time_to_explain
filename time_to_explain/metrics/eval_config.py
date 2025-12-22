# time_to_explain/metrics/eval_config.py
from __future__ import annotations

from pathlib import Path
import json
from typing import Any, Dict

_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG = _ROOT / "configs" / "metrics" / "default.json"


def load_metrics_config(path: str | Path | None = None) -> Dict[str, Any]:
    """Load metric configuration from configs/metrics."""
    config_path = Path(path) if path is not None else _DEFAULT_CONFIG
    if not config_path.exists():
        raise FileNotFoundError(f"Metrics config not found: {config_path}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def default_metrics_config() -> Dict[str, Any]:
    """Default evaluation metric configuration for the notebooks."""
    return load_metrics_config()


METRICS_CFG = default_metrics_config()

__all__ = ["METRICS_CFG", "default_metrics_config", "load_metrics_config"]

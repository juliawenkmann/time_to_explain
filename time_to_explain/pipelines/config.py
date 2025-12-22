from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml(path: str | Path) -> Dict[str, Any]:
    """
    Load a YAML file into a plain dict.
    """
    with Path(path).expanduser().open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_json(path: str | Path) -> Any:
    with Path(path).expanduser().open("r", encoding="utf-8") as f:
        return json.load(f)

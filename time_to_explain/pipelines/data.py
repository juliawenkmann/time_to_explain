from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import pandas as pd

from time_to_explain.data.generate_synthetic_dataset import prepare_dataset
from time_to_explain.data.io import resolve_repo_root
from time_to_explain.data.tgnn_setup import setup_tgnn_data

# Default synthetic recipes backed by the bundled configs.
_CONFIGS = Path(__file__).resolve().parents[2] / "configs" / "datasets"
DEFAULT_SYNTHETIC_RECIPES: Dict[str, Dict[str, object]] = {
    "erdos_small": {
        "recipe": "erdos_temporal",
        "config_path": _CONFIGS / "erdos_small.json",
        "split": (0.8, 0.1, 0.1),
    },
    "hawkes_small": {
        "recipe": "hawkes_exp",
        "config_path": _CONFIGS / "hawkes_small.json",
        "split": (0.8, 0.1, 0.1),
    },
    "nicolaus": {
        "recipe": "nicolaus",
        "config_path": _CONFIGS / "nicolaus.json",
        "split": (0.8, 0.1, 0.1),
    },
}


def prepare_real_data(
    *,
    root: Optional[Path | str] = None,
    only: Optional[Sequence[str]] = None,
    force: bool = False,
    do_process: bool = True,
    do_index: bool = True,
    seed: int = 42,
    index_size: int = 500,
) -> None:
    """
    Download + process real datasets (wikipedia/reddit) and build explain indices.
    """
    base = Path(root).expanduser().resolve() if root else resolve_repo_root()
    setup_tgnn_data(
        root=base,
        only=only,
        force=force,
        do_process=do_process,
        do_index=do_index,
        seed=seed,
        index_size=index_size,
    )


def prepare_synthetic_data(
    names: Iterable[str],
    *,
    root: Optional[Path | str] = None,
    overwrite: bool = False,
    visualize: bool = False,
) -> Dict[str, dict]:
    """
    Generate synthetic datasets using the bundled recipe configs.
    """
    base = Path(root).expanduser().resolve() if root else resolve_repo_root()
    results: Dict[str, dict] = {}
    for name in names:
        spec = DEFAULT_SYNTHETIC_RECIPES.get(name)
        if spec is None:
            raise ValueError(f"Unknown synthetic dataset '{name}'. Known: {sorted(DEFAULT_SYNTHETIC_RECIPES)}")
        results[name] = prepare_dataset(
            project_root=base,
            dataset_name=name,
            recipe=str(spec["recipe"]),
            config_path=spec.get("config_path"),
            split=spec.get("split"),
            overwrite=overwrite,
            visualize=visualize,
        )
    return results


def format_uci_messages(input_path: Path | str, output_path: Path | str) -> Path:
    """
    Reformat the UCI-Messages dataset into the expected CSV shape.
    """
    inp = Path(input_path).expanduser()
    outp = Path(output_path).expanduser()
    outp.parent.mkdir(parents=True, exist_ok=True)

    raw_data = pd.read_csv(inp, sep=" ", header=None)
    raw_data.columns = ["timestamp", "item_id", "user_id", "state_label"]
    raw_data["timestamp"] = pd.to_datetime(raw_data["timestamp"])
    raw_data["timestamp"] = (raw_data["timestamp"] - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")
    reordered = raw_data[["user_id", "item_id", "timestamp", "state_label"]]
    reordered.to_csv(outp, index=False)
    return outp

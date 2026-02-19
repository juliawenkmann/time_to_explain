from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from time_to_explain.core.registry import available_datasets
from time_to_explain.data.synthetic import prepare_dataset


def _parse_override(entry: str) -> Dict[str, Any]:
    if "=" not in entry:
        raise argparse.ArgumentTypeError(f"Override must be KEY=VALUE, got: {entry}")
    key, raw_value = entry.split("=", 1)
    raw_value = raw_value.strip()

    try:
        value = json.loads(raw_value)
    except json.JSONDecodeError:
        lowered = raw_value.lower()
        if lowered in {"true", "false"}:
            value = lowered == "true"
        else:
            try:
                value = int(raw_value)
            except ValueError:
                try:
                    value = float(raw_value)
                except ValueError:
                    value = raw_value
    return {key: value}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate synthetic datasets via registry recipes")
    parser.add_argument(
        "recipe",
        type=str,
        help="Registered synthetic recipe name",
        choices=available_datasets(),
    )
    parser.add_argument(
        "--dataset-name",
        "-n",
        type=str,
        help="Name for the output dataset (defaults to recipe name)",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        help="Optional JSON file with recipe parameters",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        metavar="KEY=VALUE",
        type=_parse_override,
        action="append",
        help="Inline overrides for recipe parameters (repeatable)",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Repository root (defaults to auto-detect)",
    )
    parser.add_argument(
        "--no-export",
        dest="export_tgn",
        action="store_false",
        help="Skip writing processed TGNN/TGAT files",
    )
    parser.add_argument(
        "--visualize",
        type=Path,
        default=None,
        help="If provided, write diagnostic plots to this directory",
    )
    parser.add_argument(
        "--explain-idx",
        type=str,
        default=None,
        help="Comma separated event indices to highlight in explain-instance visuals",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing processed dataset",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print stats, do not write any files",
    )
    return parser


def main(args: Optional[list[str]] = None) -> None:
    parser = build_parser()
    parser.set_defaults(export_tgn=True)
    ns = parser.parse_args(args=args)

    dataset_name = ns.dataset_name or ns.recipe
    overrides: Dict[str, Any] = {}
    if ns.overrides:
        for override in ns.overrides:
            overrides.update(override)

    explain_indices = None
    if ns.explain_idx:
        try:
            explain_indices = [int(part.strip()) for part in str(ns.explain_idx).split(",") if part.strip()]
        except ValueError:
            parser.error("--explain-idx expects a comma separated list of integers")

    prepare_dataset(
        project_root=ns.root,
        dataset_name=dataset_name,
        recipe=ns.recipe,
        config_path=ns.config,
        config=overrides,
        split=None,
        visualize=bool(ns.visualize),
        visualization_dir=ns.visualize,
        explain_indices=explain_indices,
        overwrite=ns.overwrite,
        export_tgn=ns.export_tgn,
        dry_run=ns.dry_run,
        verbose=True,
    )


if __name__ == "__main__":
    main()

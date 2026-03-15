from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from I_explainer_benchmark.src.core.cli import find_repo_root as _find_repo_root
from .simulate_case_study_bars import (
    _as_int_list,
    _load_mcts_recorder,
    _normalize_importance,
    align_case_to_reference,
    case_plot_paths,
    choose_target_events,
    clear_dataset_plot_outputs,
    discover_explainer_keys_from_notebooks,
    empty_run_table,
    find_mcts_recorder_file,
    flatten_official_records,
    list_run_files,
    latest_run_file,
    native_case_for_target,
    parse_jsonl,
)


DEFAULT_MODEL_BY_DATASET = {
    "simulate_v1": "tgn",
    "simulate_v2": "tgn",
}

DEFAULT_PREFERRED_TARGET_EVENTS = {
    "simulate_v1": [],
    "simulate_v2": [2013],
}

DEFAULT_CANONICAL_CASE_EVENTS = {
    "simulate_v1": 7380,
    "simulate_v2": 2013,
}

EXPLAINER_LABEL_BY_KEY = {
    "tgnnexplainer": "TGNNExplainer",
    "cody": "CoDy",
    "greedy": "Greedy",
    "temgx": "TemGX",
    "tempme": "TempME",
    "pg": "PGExplainer",
    "khop": "k-hop",
    "random": "Random",
    "my_cf": "My CF",
}

ALL_EXPLAINERS_PLOT_ORDER = (
    "tgnnexplainer",
    "pg",
    "tempme",
    "cody",
    "greedy",
    "temgx",
    "my_cf",
    "khop",
    "random",
)

SMALL_TEXT_FONT = 13
AXIS_LABEL_FONT = 15
TICK_LABEL_FONT = 13
X_TICK_LABEL_FONT = 16
PANEL_TITLE_FONT = 20
SUPTITLE_FONT = 22
LEGEND_FONT = 13
EMPTY_PANEL_FONT = 16

TYPE_EDGE_MAPPING = {
    0: (1, 3),
    1: (1, 4),
    2: (2, 3),
    3: (2, 4),
}
EDGE_TO_TYPE = {tuple(v): int(k) for k, v in TYPE_EDGE_MAPPING.items()}

A_ALPHA = {
    "simulate_v1": np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [1.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0],
        ],
        dtype=float,
    ),
    "simulate_v2": np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [1.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, -2.0],
        ],
        dtype=float,
    ),
}

PAPER_RELEVANCY = {
    ("simulate_v2", 2013): [-2, 1, 1, 2, -2, -1, 1, 2, -2, 1, 2, -2, 2, -2, -2, 1, -1, 1, 1, 2],
}

PAPER_GT_BAR_HEIGHTS = {
    ("simulate_v1", 7380): [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0],
    ("simulate_v2", 2013): [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1],
}

LABEL_TO_NAME = {
    1: "positive",
    0: "neutral",
    -1: "negative",
}

TYPE_TO_NAME = {
    type_id: f"type {type_id} ({pair[0]}->{pair[1]})"
    for type_id, pair in TYPE_EDGE_MAPPING.items()
}

EVENT_TYPE_TO_COLOR = {
    0: "#81C784",
    1: "#D95F5F",
    2: "#1B5E20",
    3: "#BDBDBD",
}

EVENT_TYPE_TO_LABEL = {
    0: "E0 (u1,i3)",
    1: "E1 (u1,i4)",
    2: "E2 (u2,i3)",
    3: "E3 (u2,i4)",
}

CATEGORY_TO_GROUP = {
    2: "positive",
    1: "positive",
    0: "neutral",
    -1: "negative",
    -2: "negative",
}

SELECTED_COL = "selected_by_explainer"
def influence_category(alpha: float) -> int:
    if alpha >= 2.0:
        return 2
    if alpha > 0.0:
        return 1
    if alpha <= -2.0:
        return -2
    if alpha < 0.0:
        return -1
    return 0


def _normalize_save_formats(formats: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    for raw in formats:
        fmt = str(raw).strip().lower().lstrip(".")
        if fmt and fmt not in normalized:
            normalized.append(fmt)
    return tuple(normalized) or ("png",)


def _panel_message(
    ax: Any,
    text: str,
    *,
    fontsize: int = SMALL_TEXT_FONT,
    x: float = 0.02,
    y: float = 0.5,
    ha: str = "left",
    va: str = "center",
) -> None:
    ax.axis("off")
    ax.text(
        x,
        y,
        str(text),
        transform=ax.transAxes,
        fontsize=fontsize,
        ha=ha,
        va=va,
    )


def _case_legend_handles(
    *,
    short_event_labels: bool = False,
    importance_alpha: float = 0.55,
) -> list[Patch]:
    event_labels = (
        {type_id: f"E{type_id}" for type_id in EVENT_TYPE_TO_COLOR}
        if short_event_labels
        else dict(EVENT_TYPE_TO_LABEL)
    )
    handles = [
        Patch(facecolor="white", edgecolor="black", hatch="//", label="striped prior mask"),
        Patch(facecolor="black", alpha=float(importance_alpha), label="explainer importance"),
    ]
    handles.extend(
        Patch(facecolor=EVENT_TYPE_TO_COLOR[type_id], edgecolor="none", label=event_labels[type_id])
        for type_id in sorted(EVENT_TYPE_TO_COLOR)
    )
    return handles


def _draw_case_bars(
    ax: Any,
    detail: pd.DataFrame,
    *,
    edge_linewidth: float,
    importance_alpha: float,
    set_xlim: bool = False,
) -> np.ndarray:
    x = detail["e_idx"].astype(int).to_numpy()
    colors = detail["influence_color"].tolist()
    ax.bar(
        x,
        detail["gt_bar_height"].to_numpy(dtype=float),
        color="white",
        edgecolor=colors,
        linewidth=float(edge_linewidth),
        hatch="//",
        zorder=1,
    )
    ax.bar(
        x,
        detail["importance_norm"].to_numpy(dtype=float),
        color=colors,
        alpha=float(importance_alpha),
        zorder=2,
    )
    if bool(set_xlim) and x.size:
        ax.set_xlim(x.min() - 0.5, x.max() + 0.5)
    return x


def _style_case_axis(
    ax: Any,
    x: np.ndarray,
    *,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    title_fontsize: int = PANEL_TITLE_FONT,
    title_pad: int = 10,
    tick_divisor: int = 10,
    x_rotation: int = 45,
    y_grid_alpha: float = 0.18,
) -> None:
    ax.set_ylim(0.0, 1.08)
    ax.grid(axis="y", alpha=float(y_grid_alpha), linewidth=0.7)
    if x.size:
        tick_step = max(1, len(x) // max(1, int(tick_divisor)))
        ax.set_xticks(x[::tick_step])
    ax.tick_params(axis="x", rotation=float(x_rotation), labelsize=X_TICK_LABEL_FONT)
    ax.tick_params(axis="y", labelsize=TICK_LABEL_FONT)
    if xlabel is not None:
        ax.set_xlabel(str(xlabel), fontsize=AXIS_LABEL_FONT)
    if ylabel is not None:
        ax.set_ylabel(str(ylabel), fontsize=AXIS_LABEL_FONT)
    if title is not None:
        ax.set_title(str(title), fontsize=title_fontsize, pad=int(title_pad))


def _merge_run_tables_from_files(run_files: list[Path]) -> pd.DataFrame:
    tables: list[pd.DataFrame] = []
    for run_order, run_file in enumerate(run_files):
        table = flatten_official_records(parse_jsonl(run_file))
        if table.empty:
            continue
        table = table.copy()
        table["_source_run_order"] = int(run_order)
        table["source_run_file"] = str(run_file)
        tables.append(table)

    if not tables:
        return empty_run_table()

    merged = pd.concat(tables, ignore_index=True, sort=False)
    event_idx = pd.to_numeric(merged.get("event_idx"), errors="coerce")
    merged = merged.loc[event_idx.notna()].copy()
    merged["event_idx"] = event_idx.loc[event_idx.notna()].astype(int)
    merged = (
        merged.sort_values(["_source_run_order", "event_idx"])
        .drop_duplicates(subset=["event_idx"], keep="last")
        .sort_values("event_idx")
        .reset_index(drop=True)
    )
    merged["num_candidates"] = merged["candidate_eidx"].apply(len)
    merged["num_selected"] = merged["selected_eidx"].apply(len)
    return merged


@dataclass(slots=True)
class CaseStudyConfig:
    dataset_name: str = "simulate_v2"
    explainer_key: str = "tgnnexplainer"
    num_targets_per_dataset: int = 10
    candidate_display_count: int | None = 25
    use_shared_candidate_axis: bool = True
    reset_dataset_output_before_save: bool = True
    save_single_view_plots: bool = True
    save_split_view_plots: bool = True
    save_formats: tuple[str, ...] = ("png",)
    influence_max_hops: int = 2
    mcts_threshold_num: int = 20
    show_mcts_panel: bool = True
    same_target_policy: str = "best_primary_positive_lift"
    failure_policy: str = "worst_positive_lift"
    final_figure_explainers: tuple[str, ...] = ("tgnnexplainer", "pg", "cody", "random")
    final_figure_view_mode: str = "aligned"
    same_target_event: int | None = None
    final_target_event: int | None = None
    failure_target_event: int | None = None
    model_by_dataset: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_MODEL_BY_DATASET))
    preferred_target_events: dict[str, list[int]] = field(default_factory=lambda: dict(DEFAULT_PREFERRED_TARGET_EVENTS))
    canonical_case_events: dict[str, int] = field(default_factory=lambda: dict(DEFAULT_CANONICAL_CASE_EVENTS))
    run_file_overrides: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.dataset_name not in self.model_by_dataset:
            raise ValueError(
                f"dataset_name={self.dataset_name!r} not in {sorted(self.model_by_dataset)}"
            )
        self.save_formats = _normalize_save_formats(self.save_formats)
        self.run_file_overrides = {
            str(key).strip().lower(): str(value)
            for key, value in dict(self.run_file_overrides).items()
            if str(key).strip() and str(value).strip()
        }


class SimulateCaseStudySession:
    def __init__(self, config: CaseStudyConfig, *, repo_root: Path | None = None) -> None:
        self.config = config
        self.repo_root = repo_root or _find_repo_root(start=Path.cwd().resolve(), marker="I_explainer_benchmark")
        self.dataset_name = str(config.dataset_name)
        self.model_name = str(config.model_by_dataset[self.dataset_name])
        self.explainer_key = str(config.explainer_key).strip().lower()

        self.src_dir = self.repo_root / "I_explainer_benchmark" / "src"
        self.explainer_notebooks_dir = self.repo_root / "I_explainer_benchmark" / "notebooks" / "explainer_notebooks"
        self.dataset_dir = self.repo_root / "I_explainer_benchmark" / "resources" / "datasets" / "processed"
        self.out_dir = self.repo_root / "I_explainer_benchmark" / "notebooks" / "outputs" / f"{self.explainer_key}_case_study_bars"
        self.out_dir.mkdir(parents=True, exist_ok=True)

        implemented = discover_explainer_keys_from_notebooks(self.explainer_notebooks_dir)
        if self.explainer_key not in implemented:
            implemented.insert(0, self.explainer_key)
        self.comparison_explainers = [self.explainer_key] + [k for k in implemented if k != self.explainer_key]
        self.explainer_labels = dict(EXPLAINER_LABEL_BY_KEY)
        for key in self.comparison_explainers:
            self.explainer_labels.setdefault(key, key.replace("_", " ").title())

        self.results_dir_by_explainer = {
            key: self.repo_root / "I_explainer_benchmark" / "resources" / "results" / f"official_{key}"
            for key in self.comparison_explainers
        }
        self.candidate_scores_dir_by_explainer = {
            key: self.results_dir_by_explainer[key] / "candidate_scores"
            for key in self.comparison_explainers
        }

        self.primary_table = empty_run_table()
        self.run_tables_by_explainer: dict[str, pd.DataFrame] = {}
        self.run_meta_by_explainer: dict[str, dict[str, Any]] = {}
        self.available_explainers: list[str] = []
        self.unavailable_explainers: list[str] = []
        self.evaluation_explainers: list[str] = []
        self.chosen_events: list[int] = []
        self.target_choice_mode: str | None = None
        self.dataset_events = pd.DataFrame()
        self.cases: dict[int, dict[str, Any]] = {}
        self.cases_by_explainer: dict[str, dict[int, dict[str, Any]]] = {}
        self.cases_native_by_explainer: dict[str, dict[int, dict[str, Any]]] = {}
        self.summary_df = pd.DataFrame()
        self._detail_cache: dict[tuple[int, str, str], pd.DataFrame] = {}
        self._effective_alpha_cache: dict[int, np.ndarray] = {}

        self._load()

    def _load(self) -> None:
        self._load_run_tables()
        self._load_dataset_events()
        self._build_cases()
        self._build_summary()
        if self.config.reset_dataset_output_before_save:
            clear_dataset_plot_outputs(self.out_dir, self.dataset_name, self.model_name)

    def _save_figure(self, fig: Figure, base_path: Path, *, dpi: int) -> None:
        for fmt in self.config.save_formats:
            out_path = Path(base_path).with_suffix(f".{fmt}")
            save_kwargs: dict[str, Any] = {"bbox_inches": "tight"}
            if fmt != "pdf":
                save_kwargs["dpi"] = int(dpi)
            fig.savefig(out_path, **save_kwargs)

    def _save_case_figure(
        self,
        fig: Figure,
        *,
        target_event: int,
        path_key: str,
        dpi: int,
        explainer_key: str | None = None,
        filename_stem: str | None = None,
    ) -> None:
        base_path = case_plot_paths(
            self.out_dir,
            dataset_name=self.dataset_name,
            model_name=self.model_name,
            target_event=int(target_event),
            explainer_key=str(explainer_key or self.explainer_key),
        )[str(path_key)]
        if filename_stem:
            base_path = base_path.parent / str(filename_stem)
        self._save_figure(fig, base_path, dpi=int(dpi))

    def _load_run_tables(self) -> None:
        primary_results_dir = self.results_dir_by_explainer[self.explainer_key]
        primary_override = self.config.run_file_overrides.get(self.explainer_key)
        if primary_override:
            primary_run_file = Path(primary_override)
            if not primary_run_file.exists():
                raise FileNotFoundError(f"Override run file not found: {primary_run_file}")
        else:
            primary_run_file = latest_run_file(
                self.dataset_name,
                self.model_name,
                primary_results_dir,
                self.explainer_key,
                allow_missing=False,
            )
        primary_records = parse_jsonl(primary_run_file)
        self.primary_table = flatten_official_records(primary_records)
        preferred_events = self.config.preferred_target_events.get(self.dataset_name, [])
        self.chosen_events, self.target_choice_mode = choose_target_events(
            self.primary_table,
            preferred_events=preferred_events,
            max_targets=self.config.num_targets_per_dataset,
        )

        self.run_tables_by_explainer[self.explainer_key] = self.primary_table
        self.run_meta_by_explainer[self.explainer_key] = {
            "run_file": str(primary_run_file),
            "run_available": True,
            "num_targets": int(len(self.primary_table)),
        }

        for compare_key in self.comparison_explainers:
            if compare_key == self.explainer_key:
                continue
            compare_results_dir = self.results_dir_by_explainer[compare_key]
            compare_override = self.config.run_file_overrides.get(compare_key)
            run_files: list[Path] = []
            if compare_override:
                override_file = Path(compare_override)
                if not override_file.exists():
                    raise FileNotFoundError(f"Override run file not found: {override_file}")
                run_files = [override_file]
            elif compare_results_dir.exists():
                run_files = list_run_files(
                    self.dataset_name,
                    self.model_name,
                    compare_results_dir,
                    compare_key,
                )

            if not run_files:
                table = empty_run_table()
                run_available = False
            else:
                table = _merge_run_tables_from_files(run_files)
                run_available = not table.empty

            self.run_tables_by_explainer[compare_key] = table
            self.run_meta_by_explainer[compare_key] = {
                "run_file": str(run_files[-1]) if run_files else None,
                "run_files": [str(path) for path in run_files],
                "run_available": bool(run_available),
                "num_targets": int(len(table)),
            }

        self.available_explainers = [
            key for key in self.comparison_explainers if self.run_meta_by_explainer[key]["run_available"]
        ]
        if self.explainer_key not in self.available_explainers:
            raise ValueError(f"Primary explainer {self.explainer_key!r} has no available run.")
        self.unavailable_explainers = [
            key for key in self.comparison_explainers if key not in self.available_explainers
        ]
        self.evaluation_explainers = [
            self.explainer_key,
            *[key for key in self.available_explainers if key != self.explainer_key],
        ]

    def _load_dataset_events(self) -> None:
        path = self.dataset_dir / f"ml_{self.dataset_name}.csv"
        df = pd.read_csv(path)
        required_cols = {"u", "i", "label", "idx", "e_idx", "ts"}
        missing = sorted(required_cols.difference(df.columns))
        if missing:
            raise ValueError(f"{self.dataset_name}: missing required columns: {missing}")
        self.dataset_events = df.drop_duplicates(subset=["e_idx"]).copy()

    def _build_cases(self) -> None:
        self.cases_by_explainer = {key: {} for key in self.evaluation_explainers}
        self.cases_native_by_explainer = {key: {} for key in self.evaluation_explainers}

        for case_rank, target_event in enumerate(self.chosen_events, start=1):
            match = self.primary_table.loc[
                pd.to_numeric(self.primary_table.get("event_idx"), errors="coerce") == int(target_event)
            ]
            if match.empty:
                continue

            row = match.iloc[0]
            candidate_eidx = _as_int_list(row.get("candidate_eidx"))
            if not candidate_eidx:
                candidate_eidx = list(range(max(1, int(target_event) - 20), int(target_event)))

            selected_eidx = _as_int_list(row.get("selected_eidx"))
            selected_set = set(selected_eidx)
            importance_edges = [float(v) for v in (row.get("importance_edges") or [])]
            if len(importance_edges) != len(candidate_eidx):
                importance_edges = [1.0 if int(eid) in selected_set else 0.0 for eid in candidate_eidx]

            mcts_recorder_path = find_mcts_recorder_file(
                dataset_name=self.dataset_name,
                model_name=self.model_name,
                target_event=int(target_event),
                candidate_scores_dir=self.candidate_scores_dir_by_explainer[self.explainer_key],
                threshold_num=self.config.mcts_threshold_num,
            )

            self.cases[int(target_event)] = {
                "case_rank": int(case_rank),
                "target_event": int(target_event),
                "candidate_eidx": list(candidate_eidx),
                "selected_eidx": list(selected_eidx),
                "importance_edges": list(importance_edges),
                "num_candidates": int(len(candidate_eidx)),
                "num_selected": int(len(selected_eidx)),
                "mcts_recorder_path": str(mcts_recorder_path) if mcts_recorder_path is not None else None,
            }

        for target_event, primary_case in self.cases.items():
            reference_candidate_eidx = [int(e) for e in primary_case["candidate_eidx"]]

            for compare_key in self.evaluation_explainers:
                compare_table = self.run_tables_by_explainer[compare_key]
                match = compare_table.loc[
                    pd.to_numeric(compare_table.get("event_idx"), errors="coerce") == int(target_event)
                ]
                compare_row = match.iloc[0] if not match.empty else None

                aligned_selected_eidx, aligned_importance_edges, source_available = align_case_to_reference(
                    compare_row,
                    reference_candidate_eidx,
                )

                mcts_recorder_path = None
                if compare_key == "tgnnexplainer":
                    found = find_mcts_recorder_file(
                        dataset_name=self.dataset_name,
                        model_name=self.model_name,
                        target_event=int(target_event),
                        candidate_scores_dir=self.candidate_scores_dir_by_explainer[compare_key],
                        threshold_num=self.config.mcts_threshold_num,
                    )
                    mcts_recorder_path = str(found) if found is not None else None

                self.cases_by_explainer[compare_key][int(target_event)] = {
                    "target_event": int(target_event),
                    "candidate_eidx": list(reference_candidate_eidx),
                    "selected_eidx": list(aligned_selected_eidx),
                    "importance_edges": list(aligned_importance_edges),
                    "num_candidates": int(len(reference_candidate_eidx)),
                    "num_selected": int(len(aligned_selected_eidx)),
                    "source_target_available": bool(source_available),
                    "mcts_recorder_path": mcts_recorder_path,
                }

                native_case = native_case_for_target(compare_table, target_event=int(target_event))
                native_case.update(
                    {
                        "target_event": int(target_event),
                        "mcts_recorder_path": mcts_recorder_path if compare_key == "tgnnexplainer" else None,
                    }
                )
                self.cases_native_by_explainer[compare_key][int(target_event)] = native_case

    def _build_summary(self) -> None:
        rows: list[dict[str, Any]] = []
        for target_event in self.cases:
            for explainer_key in self.evaluation_explainers:
                detail = self.build_case_table(
                    target_event=int(target_event),
                    explainer_key=explainer_key,
                    view_mode="aligned",
                )
                selected_mask = detail[SELECTED_COL] == 1
                selected = int(selected_mask.sum())
                candidates = int(len(detail))
                pos_cand = int((detail["influence_category"] > 0).sum())
                nonneg_cand = int((detail["influence_category"] >= 0).sum())
                selected_pos = int(((detail["influence_category"] > 0) & selected_mask).sum())
                selected_nonneg = int(((detail["influence_category"] >= 0) & selected_mask).sum())

                positive_alignment = (selected_pos / selected) if selected > 0 else np.nan
                non_negative_alignment = (selected_nonneg / selected) if selected > 0 else np.nan
                positive_candidate_rate = (pos_cand / candidates) if candidates > 0 else np.nan
                non_negative_candidate_rate = (nonneg_cand / candidates) if candidates > 0 else np.nan
                mean_selected_importance = (
                    float(detail.loc[selected_mask, "importance_norm"].mean()) if selected > 0 else np.nan
                )

                rows.append(
                    {
                        "dataset": self.dataset_name,
                        "model": self.model_name,
                        "explainer": explainer_key,
                        "target_event": int(target_event),
                        "candidates": candidates,
                        "selected": selected,
                        "mean_selected_importance": mean_selected_importance,
                        "positive_alignment_score": positive_alignment,
                        "non_negative_alignment_score": non_negative_alignment,
                        "positive_candidate_rate": positive_candidate_rate,
                        "non_negative_candidate_rate": non_negative_candidate_rate,
                        "positive_alignment_lift_vs_baseline": positive_alignment - positive_candidate_rate,
                        "non_negative_alignment_lift_vs_baseline": non_negative_alignment - non_negative_candidate_rate,
                    }
                )
        self.summary_df = pd.DataFrame(rows).sort_values(
            ["explainer", "selected", "target_event"],
            ascending=[True, False, True],
        ).reset_index(drop=True)

    def effective_alpha_matrix(self, max_hops: int) -> np.ndarray:
        hops = max(1, int(max_hops))
        if hops in self._effective_alpha_cache:
            return self._effective_alpha_cache[hops]

        base = np.asarray(A_ALPHA[self.dataset_name], dtype=float)
        total = np.zeros_like(base, dtype=float)
        power = base.copy()
        for _ in range(1, hops + 1):
            total = total + power
            power = power @ base
        self._effective_alpha_cache[hops] = total
        return total

    def target_event_metadata(self, target_event: int) -> dict[str, Any]:
        event_df = self.dataset_events.set_index("e_idx")
        target_idx = int(target_event)
        if target_idx not in event_df.index:
            return {
                "target_event": target_idx,
                "target_u": np.nan,
                "target_i": np.nan,
                "target_type": np.nan,
                "target_type_label": "unknown",
                "target_label": np.nan,
                "target_label_name": "unknown",
                "target_prediction_label": "unknown",
                "target_prediction_source": "dataset_label",
            }

        row = event_df.loc[target_idx]
        target_u = int(row["u"])
        target_i = int(row["i"])
        target_type = EDGE_TO_TYPE.get((target_u, target_i))

        raw_label = pd.to_numeric(pd.Series([row.get("label", np.nan)]), errors="coerce").iloc[0]
        target_label = int(raw_label) if pd.notna(raw_label) else np.nan
        target_label_name = (
            LABEL_TO_NAME.get(int(target_label), "unknown") if pd.notna(target_label) else "unknown"
        )

        if target_type is None:
            target_type_value = np.nan
            target_type_label = f"unknown ({target_u}->{target_i})"
        else:
            target_type_value = int(target_type)
            target_type_label = TYPE_TO_NAME.get(int(target_type), f"unknown ({target_u}->{target_i})")

        return {
            "target_event": target_idx,
            "target_u": target_u,
            "target_i": target_i,
            "target_type": target_type_value,
            "target_type_label": target_type_label,
            "target_label": target_label,
            "target_label_name": target_label_name,
            "target_prediction_label": target_label_name,
            "target_prediction_source": "dataset_label",
        }

    def categorize_candidates(self, target_event: int, candidate_rows: pd.DataFrame) -> tuple[list[int], list[float], list[float], list[float]]:
        key = (self.dataset_name, int(target_event))
        if key in PAPER_RELEVANCY and len(PAPER_RELEVANCY[key]) == len(candidate_rows):
            categories = list(PAPER_RELEVANCY[key])
            nan_values = [float("nan")] * len(categories)
            return categories, nan_values, nan_values, nan_values

        target_meta = self.target_event_metadata(target_event)
        target_pair = (int(target_meta["target_u"]), int(target_meta["target_i"]))
        target_type = EDGE_TO_TYPE[target_pair]
        direct_alpha = np.asarray(A_ALPHA[self.dataset_name], dtype=float)
        effective_alpha = self.effective_alpha_matrix(self.config.influence_max_hops)

        categories: list[int] = []
        alpha_effective_values: list[float] = []
        alpha_direct_values: list[float] = []
        alpha_indirect_values: list[float] = []
        for _, row in candidate_rows.iterrows():
            source_type = EDGE_TO_TYPE[(int(row["u"]), int(row["i"]))]
            alpha_direct = float(direct_alpha[target_type, source_type])
            alpha_effective = float(effective_alpha[target_type, source_type])
            alpha_indirect = float(alpha_effective - alpha_direct)

            alpha_direct_values.append(alpha_direct)
            alpha_indirect_values.append(alpha_indirect)
            alpha_effective_values.append(alpha_effective)
            categories.append(influence_category(alpha_effective))
        return categories, alpha_effective_values, alpha_direct_values, alpha_indirect_values

    def _paper_style_gt_heights(self, target_event: int, categories: list[int]) -> tuple[list[float], str]:
        key = (self.dataset_name, int(target_event))
        if key in PAPER_GT_BAR_HEIGHTS and len(PAPER_GT_BAR_HEIGHTS[key]) == len(categories):
            return [float(v) for v in PAPER_GT_BAR_HEIGHTS[key]], "paper_case_study"
        return [1.0 if int(c) > 0 else 0.0 for c in categories], "positive_prior"

    def _case_store(self, view_mode: str) -> dict[str, dict[int, dict[str, Any]]]:
        if view_mode == "aligned":
            return self.cases_by_explainer
        if view_mode == "native":
            return self.cases_native_by_explainer
        raise ValueError(f"Unsupported view_mode={view_mode!r}")

    def build_case_table(
        self,
        target_event: int,
        explainer_key: str | None = None,
        view_mode: str = "aligned",
    ) -> pd.DataFrame:
        resolved_explainer_key = str(explainer_key or self.explainer_key)
        key = (int(target_event), resolved_explainer_key, view_mode)
        if key in self._detail_cache:
            return self._detail_cache[key].copy()

        case = self._case_store(view_mode).get(resolved_explainer_key, {}).get(int(target_event))

        columns = [
            "e_idx",
            "u",
            "i",
            "label",
            "idx",
            "ts",
            "alpha_effective",
            "alpha_direct",
            "alpha_indirect",
            "influence_category",
            "influence_group",
            "event_type",
            "event_type_label",
            "influence_color",
            "gt_bar_height",
            "gt_mode",
            SELECTED_COL,
            "importance_raw",
            "importance_positive",
            "importance_norm",
            "target_prediction_label",
            "target_prediction_source",
            "target_type_label",
        ]
        if case is None:
            empty = pd.DataFrame(columns=columns)
            self._detail_cache[key] = empty.copy()
            return empty

        source_candidate_eidx = [int(e) for e in _as_int_list(case.get("candidate_eidx"))]
        selected_set = {int(e) for e in _as_int_list(case.get("selected_eidx"))}

        candidate_eidx_full = list(source_candidate_eidx)
        if view_mode == "aligned" and self.config.use_shared_candidate_axis:
            shared_case = self.cases_by_explainer.get(self.explainer_key, {}).get(int(target_event))
            if shared_case is not None:
                shared_candidates = [int(e) for e in _as_int_list(shared_case.get("candidate_eidx"))]
                if shared_candidates:
                    candidate_eidx_full = shared_candidates

        event_df = self.dataset_events.set_index("e_idx")
        explainer_key = resolved_explainer_key
        if not candidate_eidx_full:
            empty = pd.DataFrame(columns=columns)
            self._detail_cache[key] = empty.copy()
            return empty

        candidate_eidx_full = [eid for eid in candidate_eidx_full if int(eid) in event_df.index]
        source_candidate_eidx = [eid for eid in source_candidate_eidx if int(eid) in event_df.index]
        if not candidate_eidx_full:
            empty = pd.DataFrame(columns=columns)
            self._detail_cache[key] = empty.copy()
            return empty

        candidate_eidx = list(candidate_eidx_full)
        if self.config.candidate_display_count is not None and len(candidate_eidx) > int(self.config.candidate_display_count):
            candidate_eidx = candidate_eidx[-int(self.config.candidate_display_count):]

        candidate_rows = event_df.loc[candidate_eidx].reset_index().copy()
        categories, alpha_effective_values, alpha_direct_values, alpha_indirect_values = self.categorize_candidates(
            int(target_event),
            candidate_rows,
        )
        gt_bar_height, gt_mode = self._paper_style_gt_heights(int(target_event), categories)

        importance_raw_full = np.asarray(case.get("importance_edges", []), dtype=float)
        if importance_raw_full.shape[0] == len(source_candidate_eidx):
            importance_by_eid = {
                int(eid): float(score)
                for eid, score in zip(source_candidate_eidx, importance_raw_full)
            }
        elif importance_raw_full.shape[0] == len(candidate_eidx_full):
            importance_by_eid = {
                int(eid): float(score)
                for eid, score in zip(candidate_eidx_full, importance_raw_full)
            }
        else:
            importance_by_eid = {int(eid): 1.0 for eid in selected_set}

        if not selected_set and explainer_key in {"pg"}:
            selected_set = {
                int(eid)
                for eid, score in importance_by_eid.items()
                if abs(float(score)) > 0.0
            }

        importance_raw = np.asarray(
            [float(importance_by_eid.get(int(eid), 0.0)) for eid in candidate_rows["e_idx"].tolist()],
            dtype=float,
        )
        display_importance = importance_raw
        if explainer_key in {"pg"}:
            display_importance = np.abs(display_importance)
        target_meta = self.target_event_metadata(int(target_event))

        candidate_rows["alpha_effective"] = alpha_effective_values
        candidate_rows["alpha_direct"] = alpha_direct_values
        candidate_rows["alpha_indirect"] = alpha_indirect_values
        candidate_rows["influence_category"] = categories
        candidate_rows["influence_group"] = [CATEGORY_TO_GROUP[c] for c in categories]
        candidate_rows["event_type"] = [
            int(EDGE_TO_TYPE.get((int(u), int(i)), -1))
            for u, i in zip(candidate_rows["u"].tolist(), candidate_rows["i"].tolist())
        ]
        candidate_rows["event_type_label"] = [
            EVENT_TYPE_TO_LABEL.get(int(t), "unknown")
            for t in candidate_rows["event_type"].tolist()
        ]
        candidate_rows["influence_color"] = [
            EVENT_TYPE_TO_COLOR.get(int(t), "#9E9E9E")
            for t in candidate_rows["event_type"].tolist()
        ]
        candidate_rows["gt_bar_height"] = gt_bar_height
        candidate_rows["gt_mode"] = gt_mode
        candidate_rows[SELECTED_COL] = candidate_rows["e_idx"].isin(selected_set).astype(int)
        candidate_rows["importance_raw"] = importance_raw
        candidate_rows["importance_positive"] = np.clip(display_importance, 0.0, None)
        candidate_rows["importance_norm"] = _normalize_importance(
            candidate_rows["importance_positive"].to_numpy(dtype=float)
        )
        candidate_rows["target_prediction_label"] = target_meta["target_prediction_label"]
        candidate_rows["target_prediction_source"] = target_meta["target_prediction_source"]
        candidate_rows["target_type_label"] = target_meta["target_type_label"]

        out = candidate_rows[columns].copy()
        self._detail_cache[key] = out.copy()
        return out

    def available_explainers_for_target(
        self,
        target_event: int,
        *,
        view_mode: str = "native",
    ) -> list[str]:
        store = self._case_store(view_mode)
        ordered = [key for key in ALL_EXPLAINERS_PLOT_ORDER if key in self.evaluation_explainers]
        ordered.extend(key for key in self.evaluation_explainers if key not in ordered)
        return [
            key
            for key in ordered
            if (
                int(target_event) in store.get(key, {})
                and bool(store.get(key, {}).get(int(target_event), {}).get("source_target_available", True))
                and int(
                    store.get(key, {}).get(int(target_event), {}).get(
                        "num_candidates",
                        len(_as_int_list(store.get(key, {}).get(int(target_event), {}).get("candidate_eidx"))),
                    )
                )
                > 0
            )
        ]

    def best_target(self) -> int:
        if self.config.final_target_event is not None:
            return int(self.config.final_target_event)

        primary_scores = self.summary_df[
            self.summary_df["explainer"] == self.explainer_key
        ].copy()
        if primary_scores.empty:
            return int(self._default_same_target())

        primary_scores = primary_scores.sort_values(
            [
                "positive_alignment_lift_vs_baseline",
                "non_negative_alignment_lift_vs_baseline",
                "positive_alignment_score",
                "selected",
                "mean_selected_importance",
            ],
            ascending=[False, False, False, False, False],
        )
        return int(primary_scores.iloc[0]["target_event"])

    def _selection_counts_for_final_explainers(self, target_event: int) -> dict[str, int]:
        counts: dict[str, int] = {}
        for explainer_key in self.config.final_figure_explainers:
            case = self.cases_by_explainer.get(explainer_key, {}).get(int(target_event))
            counts[explainer_key] = int(case.get("num_selected", 0)) if case is not None else 0
        return counts

    def failure_target(self) -> int:
        if self.config.failure_target_event is not None:
            return int(self.config.failure_target_event)

        primary_scores = self.summary_df[
            self.summary_df["explainer"] == self.explainer_key
        ].copy()
        if primary_scores.empty:
            return int(self._default_same_target())

        eligible_targets = []
        for target_event in primary_scores["target_event"].astype(int).tolist():
            counts = self._selection_counts_for_final_explainers(int(target_event))
            if all(int(v) > 0 for v in counts.values()):
                eligible_targets.append(int(target_event))

        rows = primary_scores[primary_scores["target_event"].isin(eligible_targets)].copy()
        if rows.empty:
            rows = primary_scores

        if self.config.failure_policy == "worst_positive_lift":
            ranked = rows.sort_values(
                [
                    "positive_alignment_lift_vs_baseline",
                    "non_negative_alignment_lift_vs_baseline",
                    "positive_alignment_score",
                    "selected",
                    "mean_selected_importance",
                ],
                ascending=[True, True, True, True, True],
            )
            return int(ranked.iloc[0]["target_event"])
        return int(rows.sort_values("positive_alignment_lift_vs_baseline", ascending=True).iloc[0]["target_event"])

    def _default_same_target(self) -> int:
        if self.config.same_target_event is not None:
            return int(self.config.same_target_event)
        if self.config.same_target_policy == "best_primary_positive_lift":
            return int(self.best_target())
        canonical = self.config.canonical_case_events.get(self.dataset_name)
        if canonical in set(self.cases):
            return int(canonical)
        return int(next(iter(sorted(self.cases))))

    def _plot_detail_panel(
        self,
        ax: Any,
        detail: pd.DataFrame,
        *,
        target_event: int,
        explainer_label: str,
        mode_label: str,
        source_available: bool,
    ) -> None:
        if detail.empty:
            target_meta = self.target_event_metadata(int(target_event))
            _panel_message(
                ax,
                (
                    f"{mode_label}: no target in run | pred={target_meta['target_prediction_label']} | "
                    f"type={target_meta['target_type_label']} | source_target={'yes' if source_available else 'no'}"
                ),
                fontsize=SMALL_TEXT_FONT,
            )
            return

        x = _draw_case_bars(ax, detail, edge_linewidth=2.0, importance_alpha=0.55)
        _style_case_axis(
            ax,
            x,
            title=explainer_label,
            title_fontsize=PANEL_TITLE_FONT,
            title_pad=10,
            tick_divisor=10,
            x_rotation=45,
            y_grid_alpha=0.18,
        )
        return

    def plot_case_overlay(
        self,
        *,
        target_event: int,
        explainer_key: str | None = None,
        view_mode: str = "aligned",
        save: bool = True,
    ) -> Figure:
        explainer_key = str(explainer_key or self.explainer_key)
        case = self._case_store(view_mode).get(explainer_key, {}).get(
            int(target_event),
            {"source_target_available": False, "mcts_recorder_path": None},
        )
        detail = self.build_case_table(
            target_event=int(target_event),
            explainer_key=explainer_key,
            view_mode=view_mode,
        )
        target_meta = self.target_event_metadata(int(target_event))
        show_mcts = bool(
            self.config.show_mcts_panel and explainer_key == "tgnnexplainer" and view_mode == "aligned"
        )
        recorder_df = _load_mcts_recorder(case.get("mcts_recorder_path")) if show_mcts else None
        has_mcts_recorder = recorder_df is not None and not recorder_df.empty

        if show_mcts:
            fig, (ax, ax_mcts) = plt.subplots(
                2,
                1,
                figsize=(12.5, 6.6),
                gridspec_kw={"height_ratios": [3.0, 1.5]},
                constrained_layout=True,
            )
        else:
            fig, ax = plt.subplots(figsize=(12.5, 4.2), constrained_layout=True)
            ax_mcts = None

        if detail.empty:
            _panel_message(
                ax,
                (
                    f"No {view_mode} candidates available for {self.explainer_labels[explainer_key]} "
                    f"on target={int(target_event)}."
                ),
                fontsize=AXIS_LABEL_FONT,
                y=0.60,
            )
            _panel_message(
                ax,
                (
                    f"target_prediction={target_meta['target_prediction_label']} "
                    f"({target_meta['target_prediction_source']}) | "
                    f"target_type={target_meta['target_type_label']} | "
                    f"source_target={'yes' if case.get('source_target_available', True) else 'no'}"
                ),
                fontsize=SMALL_TEXT_FONT,
                y=0.40,
            )
            if ax_mcts is not None:
                ax_mcts.axis("off")
        else:
            x = _draw_case_bars(
                ax,
                detail,
                edge_linewidth=2.4,
                importance_alpha=0.55,
                set_xlim=True,
            )
            _style_case_axis(
                ax,
                x,
                xlabel="candidate event id (e_idx)",
                ylabel="normalized score",
                tick_divisor=10,
                x_rotation=45,
                y_grid_alpha=0.18,
            )
            ax.set_title(self.explainer_labels[explainer_key], fontsize=PANEL_TITLE_FONT, pad=12)
            ax.legend(
                handles=_case_legend_handles(importance_alpha=0.55),
                loc="upper left",
                bbox_to_anchor=(1.01, 1.0),
                frameon=True,
                fontsize=LEGEND_FONT,
            )

            if show_mcts and ax_mcts is not None:
                if has_mcts_recorder:
                    ax_mcts.plot(
                        recorder_df["rollout"],
                        recorder_df["best_reward"],
                        color="black",
                        linewidth=1.4,
                        label="best_reward",
                    )
                    ax_mcts.set_xlabel("MCTS rollout", fontsize=AXIS_LABEL_FONT)
                    ax_mcts.set_ylabel("best_reward", fontsize=AXIS_LABEL_FONT)
                    ax_mcts.grid(alpha=0.2, linewidth=0.6)
                    ax_mcts.tick_params(axis="both", labelsize=TICK_LABEL_FONT)
                    ax_states = ax_mcts.twinx()
                    ax_states.plot(
                        recorder_df["rollout"],
                        recorder_df["num_states"],
                        color="tab:blue",
                        linewidth=1.0,
                        alpha=0.45,
                        label="num_states",
                    )
                    ax_states.set_ylabel("num_states", fontsize=AXIS_LABEL_FONT)
                    ax_states.tick_params(axis="y", labelsize=TICK_LABEL_FONT)
                    h1, l1 = ax_mcts.get_legend_handles_labels()
                    h2, l2 = ax_states.get_legend_handles_labels()
                    ax_mcts.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=LEGEND_FONT)
                else:
                    ax_mcts.axis("off")

        if save:
            self._save_case_figure(
                fig,
                target_event=int(target_event),
                path_key=view_mode,
                dpi=180,
                explainer_key=explainer_key,
            )
        return fig

    def plot_case_split(
        self,
        *,
        target_event: int,
        explainer_key: str | None = None,
        save: bool = True,
    ) -> Figure:
        explainer_key = str(explainer_key or self.explainer_key)
        fig, axes = plt.subplots(1, 2, figsize=(16.0, 4.8), constrained_layout=True, squeeze=False)
        axes = axes[0]

        detail_aligned = self.build_case_table(
            target_event=int(target_event),
            explainer_key=explainer_key,
            view_mode="aligned",
        )
        detail_native = self.build_case_table(
            target_event=int(target_event),
            explainer_key=explainer_key,
            view_mode="native",
        )

        def _draw(ax: Any, detail: pd.DataFrame, mode_label: str) -> None:
            if detail.empty:
                _panel_message(ax, f"{mode_label}: no candidates", fontsize=SMALL_TEXT_FONT)
                return
            x = _draw_case_bars(ax, detail, edge_linewidth=2.0, importance_alpha=0.55)
            _style_case_axis(
                ax,
                x,
                xlabel="candidate event id",
                ylabel="normalized importance",
                tick_divisor=10,
                x_rotation=45,
                y_grid_alpha=0.18,
            )

        _draw(axes[0], detail_aligned, "aligned")
        _draw(axes[1], detail_native, "native")
        axes[1].legend(
            handles=_case_legend_handles(short_event_labels=True, importance_alpha=0.55),
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            frameon=True,
            fontsize=LEGEND_FONT,
        )
        fig.suptitle(self.explainer_labels[explainer_key], y=1.02, fontsize=SUPTITLE_FONT)
        if save:
            self._save_case_figure(
                fig,
                target_event=int(target_event),
                path_key="split",
                dpi=180,
                explainer_key=explainer_key,
            )
        return fig

    def plot_all_explainers_same_target_aligned_vs_native(
        self,
        *,
        target_event: int,
        filename_stem: str = "all_views",
        save: bool = True,
    ) -> Figure:
        ordered_explainers = self.available_explainers_for_target(int(target_event), view_mode="native")
        n = len(ordered_explainers)
        if n == 0:
            fig, ax = plt.subplots(figsize=(10.5, 3.6), constrained_layout=True)
            _panel_message(
                ax,
                f"No explainer run contains target {int(target_event)}.",
                fontsize=EMPTY_PANEL_FONT,
                x=0.5,
                y=0.5,
                ha="center",
                va="center",
            )
            if save:
                self._save_case_figure(
                    fig,
                    target_event=int(target_event),
                    path_key="all_explainers_split",
                    dpi=180,
                    filename_stem=filename_stem,
                )
            return fig
        fig, axes = plt.subplots(
            n,
            1,
            figsize=(10.5, 3.15 * n + 1.4),
            constrained_layout=False,
            squeeze=False,
        )

        for row_idx, explainer_key in enumerate(ordered_explainers):
            detail_native = self.build_case_table(
                target_event=int(target_event),
                explainer_key=explainer_key,
                view_mode="native",
            )
            native_case = self.cases_native_by_explainer.get(explainer_key, {}).get(
                int(target_event),
                {"source_target_available": False},
            )

            self._plot_detail_panel(
                axes[row_idx, 0],
                detail_native,
                target_event=int(target_event),
                explainer_label=self.explainer_labels[explainer_key],
                mode_label="native",
                source_available=bool(native_case.get("source_target_available", True)),
            )
            if row_idx == n - 1:
                axes[row_idx, 0].set_xlabel("candidate event ids")

        fig.suptitle(
            f"Target {int(target_event)}",
            fontsize=SUPTITLE_FONT,
            y=0.985,
        )
        fig.legend(
            handles=_case_legend_handles(importance_alpha=0.55),
            loc="upper center",
            ncol=3,
            frameon=True,
            bbox_to_anchor=(0.5, 0.95),
            fontsize=LEGEND_FONT,
        )
        fig.subplots_adjust(top=0.87, bottom=0.05, left=0.08, right=0.98, hspace=0.72)
        if save:
            self._save_case_figure(
                fig,
                target_event=int(target_event),
                path_key="all_explainers_split",
                dpi=180,
                filename_stem=filename_stem,
            )
        return fig

    def _draw_final_panel(
        self,
        *,
        ax: Any,
        detail: pd.DataFrame,
        explainer_key: str,
    ) -> None:
        explainer_label = self.explainer_labels.get(explainer_key, explainer_key)
        if detail.empty:
            _panel_message(ax, f"{explainer_label}: no data", fontsize=EMPTY_PANEL_FONT)
            return

        x = _draw_case_bars(ax, detail, edge_linewidth=2.0, importance_alpha=0.60)
        _style_case_axis(
            ax,
            x,
            ylabel="importance",
            tick_divisor=8,
            x_rotation=30,
            y_grid_alpha=0.15,
        )
        selected = int((detail[SELECTED_COL] == 1).sum())
        candidates = int(len(detail))
        selected_pos = int(((detail["influence_category"] > 0) & (detail[SELECTED_COL] == 1)).sum())
        pos_align = (selected_pos / selected) if selected > 0 else np.nan
        pos_txt = "nan" if pd.isna(pos_align) else f"{float(pos_align):.2f}"
        ax.set_title(
            f"{explainer_label} | selected={selected}/{candidates} | pos_align={pos_txt}",
            fontsize=PANEL_TITLE_FONT,
        )

    def plot_final_2x2_figure(
        self,
        *,
        target_event: int,
        title_prefix: str,
        filename_stem: str,
        save: bool = True,
    ) -> Figure:
        fig, axes = plt.subplots(2, 2, figsize=(19, 11), constrained_layout=False)

        for panel_idx, explainer_key in enumerate(self.config.final_figure_explainers):
            row = panel_idx // 2
            col = panel_idx % 2
            detail = self.build_case_table(
                target_event=int(target_event),
                explainer_key=explainer_key,
                view_mode=self.config.final_figure_view_mode,
            )
            self._draw_final_panel(ax=axes[row, col], detail=detail, explainer_key=explainer_key)

        for ax in axes[1, :]:
            ax.set_xlabel("candidate event id", fontsize=AXIS_LABEL_FONT)

        fig.suptitle(
            f"Target {int(target_event)}",
            fontsize=SUPTITLE_FONT,
            y=0.97,
        )
        fig.legend(
            handles=_case_legend_handles(importance_alpha=0.60),
            loc="lower center",
            ncol=3,
            frameon=True,
            bbox_to_anchor=(0.5, 0.01),
            fontsize=LEGEND_FONT,
        )
        fig.subplots_adjust(top=0.90, bottom=0.15, left=0.06, right=0.98, hspace=0.32, wspace=0.18)
        if save:
            self._save_case_figure(
                fig,
                target_event=int(target_event),
                path_key="split",
                dpi=220,
                filename_stem=filename_stem,
            )
        return fig

    def build_default_figures(self, *, save: bool = True) -> list[tuple[str, Figure]]:
        best_target = self.best_target()
        failure_target = self.failure_target()
        figures = [
            (
                f"{self.explainer_key}_best_aligned",
                self.plot_case_overlay(
                    target_event=best_target,
                    explainer_key=self.explainer_key,
                    view_mode="aligned",
                    save=save and self.config.save_single_view_plots,
                ),
            ),
            (
                f"{self.explainer_key}_best_native",
                self.plot_case_overlay(
                    target_event=best_target,
                    explainer_key=self.explainer_key,
                    view_mode="native",
                    save=save and self.config.save_single_view_plots,
                ),
            ),
            (
                f"{self.explainer_key}_best_split",
                self.plot_case_split(
                    target_event=best_target,
                    explainer_key=self.explainer_key,
                    save=save and self.config.save_split_view_plots,
                ),
            ),
            (
                "all_explainers_best_target",
                self.plot_all_explainers_same_target_aligned_vs_native(
                    target_event=best_target,
                    filename_stem="all_views_best",
                    save=save,
                ),
            ),
            (
                "all_explainers_failure_target",
                self.plot_all_explainers_same_target_aligned_vs_native(
                    target_event=failure_target,
                    filename_stem="all_views_fail",
                    save=save,
                ),
            ),
            (
                "final_best_2x2",
                self.plot_final_2x2_figure(
                    target_event=best_target,
                    title_prefix="Best-case 2x2",
                    filename_stem="final_best_2x2",
                    save=save,
                ),
            ),
            (
                "final_failure_2x2",
                self.plot_final_2x2_figure(
                    target_event=failure_target,
                    title_prefix="Failure-case 2x2",
                    filename_stem="final_fail_2x2",
                    save=save,
                ),
            ),
        ]
        return figures

    def summary_targets(self) -> dict[str, int]:
        return {
            "best_target": int(self.best_target()),
            "same_target": int(self._default_same_target()),
            "failure_target": int(self.failure_target()),
            "num_chosen_targets": int(len(self.cases)),
            "num_available_explainers": int(len(self.evaluation_explainers)),
        }


__all__ = [
    "CaseStudyConfig",
    "SimulateCaseStudySession",
]

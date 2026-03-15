from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..metrics.cody import _apply_harmonic_charr


def _safe_slug(text: object) -> str:
    out: list[str] = []
    for ch in str(text):
        if ch.isalnum() or ch in ("-", "_"):
            out.append(ch)
        elif ch in (" ", "/", "|", "."):
            out.append("-")
    slug = "".join(out).strip("-").lower()
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "plot"


def _canonical_explainer_label(name: object) -> str:
    ns = str(name).strip().lower()
    if ns.startswith("official_"):
        ns = ns[len("official_") :]
    ns_flat = ns.replace("_", " ").replace("-", " ").strip()
    aliases = {
        "cf": "my_cf",
        "counterfactual": "my_cf",
        "cf_metric_opt": "my_cf",
        "cf_metric_opt_upper": "my_cf",
        "cf_mtric_opt_upper": "my_cf",
        "my_cf": "my_cf",
        "pg explainer": "pg",
        "pg-explainer": "pg",
        "pg_explainer": "pg",
    }
    return aliases.get(ns, aliases.get(ns_flat, ns))


def _normalize_explainer_names(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "explainer" not in df.columns:
        return df
    out = df.copy()
    out["explainer"] = out["explainer"].map(_canonical_explainer_label)
    return out


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _apply_filters(
    df: pd.DataFrame,
    *,
    dataset: str | None,
    model: str | None,
    explainers: Sequence[str] | None,
    explainers_exclude: Sequence[str] | None,
    variants: Sequence[str] | None,
) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if dataset is not None and "dataset" in out.columns:
        out = out[out["dataset"].astype(str).str.lower() == str(dataset).lower()]
    if model is not None and "model" in out.columns:
        out = out[out["model"].astype(str).str.lower() == str(model).lower()]
    if explainers is not None and "explainer" in out.columns:
        keep = {str(x) for x in explainers}
        out = out[out["explainer"].astype(str).isin(keep)]
    if explainers_exclude is not None and "explainer" in out.columns:
        drop = {str(x) for x in explainers_exclude}
        out = out[~out["explainer"].astype(str).isin(drop)]
    if variants is not None and "variant" in out.columns:
        keep = {str(x) for x in variants}
        out = out[out["variant"].astype(str).isin(keep)]
    return out.reset_index(drop=True)


def _select_runs(curve_or_metric_df: pd.DataFrame, run_selection: str) -> set[str]:
    if curve_or_metric_df.empty:
        return set()
    df = curve_or_metric_df.copy()
    if "run_id" not in df.columns:
        return set()
    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
    else:
        return set(df["run_id"].astype(str).unique().tolist())

    if run_selection == "all":
        return set(df["run_id"].astype(str).unique().tolist())

    sort_cols = ["created_at", "run_id"]
    df = df.sort_values(sort_cols)
    if run_selection == "latest_per_explainer":
        grp_cols = [c for c in ["explainer"] if c in df.columns]
    else:
        grp_cols = [c for c in ["explainer", "variant"] if c in df.columns]
    if not grp_cols:
        return set(df["run_id"].astype(str).unique().tolist())
    latest = df.groupby(grp_cols, as_index=False).tail(1)
    return set(latest["run_id"].astype(str).unique().tolist())


def _order_explainers(explainers: Iterable[str]) -> list[str]:
    priority = [
        "my_cf",
        "cody",
        "greedy",
        "temgx",
        "tgnnexplainer",
        "pg",
        "tempme",
        "khop",
        "random",
    ]
    rank = {name: idx for idx, name in enumerate(priority)}
    uniq: list[str] = []
    seen: set[str] = set()
    for x in explainers:
        sx = str(x)
        if sx in seen:
            continue
        seen.add(sx)
        uniq.append(sx)
    default = len(rank) + 100
    return sorted(uniq, key=lambda x: (rank.get(_canonical_explainer_label(x), default), str(x)))


def _order_radar_metrics(metrics: Sequence[str], priority: Sequence[str]) -> list[str]:
    pr = [str(x) for x in list(priority or [])]
    rank = {name: idx for idx, name in enumerate(pr)}
    default_rank = len(rank)
    unique: list[str] = []
    seen: set[str] = set()
    for m in metrics:
        ms = str(m)
        if ms in seen:
            continue
        seen.add(ms)
        unique.append(ms)
    return sorted(unique, key=lambda name: (rank.get(str(name), default_rank), str(name)))


def _metric_source_explainer(metric: str) -> str | None:
    mapping = {
        "best_fid": "tgnnexplainer",
        "aufsc": "tgnnexplainer",
        "temgx_aufsc": "temgx",
        "flip_success_rate": "cf_metric_opt",
        "cody_AUFSC_plus": "cody",
        "cody_AUFSC_minus": "cody",
        "cody_CHARR": "cody",
        "tempme_acc_auc.ratio_acc": "tempme",
    }
    name = str(metric)
    if name in mapping:
        return str(mapping[name])
    if name.startswith("cody_"):
        return "cody"
    if name.startswith("tempme_"):
        return "tempme"
    if name.startswith("temgx_"):
        return "temgx"
    if name.startswith("cf_metric_opt") or name.startswith("flip_"):
        return "cf_metric_opt"
    if name.startswith("tgnn_"):
        return "tgnnexplainer"
    return None


def _metric_source_color(
    metric: str,
    *,
    colors: dict[str, str],
    overrides: dict[str, str],
) -> str:
    source = _metric_source_explainer(metric)
    if source is None:
        return "#e2e8f0"
    s = _canonical_explainer_label(source)
    if s in overrides:
        return str(overrides[s])
    if s in colors:
        return str(colors[s])
    return "#cbd5e1"


def _resolve_summary_root(repo_root: Path) -> Path:
    return repo_root / "I_explainer_benchmark" / "resources" / "results" / "summary_ready"


def resolve_legacy_radial_paths(repo_root: Path, dataset: str, model: str) -> list[Path]:
    root = Path(repo_root).expanduser().resolve() / "radial_plots"
    ds = str(dataset).strip().lower()
    md = str(model).strip().lower()
    ordered = [
        root / f"radial_plot_by_explainer_{md}_{ds}.pdf",
        root / f"radial_plot_mean_{md}_{ds}.pdf",
        root / f"radial_plot_best_{md}_{ds}.pdf",
    ]
    return [p for p in ordered if p.exists()]


def _color_map(labels: Sequence[str], overrides: dict[str, str], colorway: Sequence[str]) -> dict[str, str]:
    cmap: dict[str, str] = {}
    for idx, label in enumerate(labels):
        canon = _canonical_explainer_label(label)
        if canon in overrides:
            cmap[str(label)] = overrides[canon]
        else:
            cmap[str(label)] = str(colorway[idx % len(colorway)])
    return cmap


@dataclass
class ThesisPlotConfig:
    repo_root: Path
    dataset: str | None = "simulate_v1"
    model: str | None = "tgn"
    run_selection: str = "latest_per_explainer_variant"
    explainers: Sequence[str] | None = None
    explainers_exclude: Sequence[str] | None = ("cf_beam_floor_balanced", "cf_beam_floor", "perturb_one")
    variants: Sequence[str] | None = None
    fidelity_y: str = "fid_inv_best"

    metrics_to_plot: Sequence[str] = (
        "best_fid",
        "aufsc",
        "temgx_aufsc",
        "tempme_acc_auc.ratio_acc",
        "cody_AUFSC_plus",
        "cody_AUFSC_minus",
        "cody_CHARR",
        "flip_success_rate",
        "first_flip_sparsity",
        "elapsed_sec",
    )
    include_tempme_metrics: bool = True
    tempme_metric_keep: str = "tempme_acc_auc.ratio_acc"

    drop_style_metrics: Sequence[str] = (
        "best_fid",
        "aufsc",
        "temgx_aufsc",
        "tempme_acc_auc.ratio_acc",
        "cody_AUFSC_plus",
        "cody_AUFSC_minus",
        "cody_CHARR",
        "flip_success_rate",
    )
    keep_style_metrics: Sequence[str] = (
        "best_fid",
        "aufsc",
        "temgx_aufsc",
        "tempme_acc_auc.ratio_acc",
        "cody_AUFSC_plus",
        "cody_AUFSC_minus",
        "cody_CHARR",
        "flip_success_rate",
    )
    radar_metrics_exclude: Sequence[str] = (
        "best_minus_aufsc",
        "best_fid_raw",
        "best_fid_raw_sparsity",
        "best_fid_raw_lt1",
        "best_fid_raw_lt1_sparsity",
        "best_fid_sparsity",
        "best_fid_raw_it",
        "best_fid_raw_it1_sparsity",
        "first_flip_sparsity",
        "elapsed_sec",
    )
    radar_metric_priority: Sequence[str] = (
        "best_fid",
        "aufsc",
        "temgx_aufsc",
        "tempme_acc_auc.ratio_acc",
        "cody_AUFSC_plus",
        "cody_AUFSC_minus",
        "cody_CHARR",
        "flip_success_rate",
    )

    plot_style: str = "seaborn-v0_8-whitegrid"
    plot_dpi: int = 180
    text_size: int = 30
    show_plots: bool = False
    plot_bg: str = "#ffffff"
    plot_grid: str = "#d8e0ea"
    colorway: Sequence[str] = (
        "#2865EB",
        "#F59F0F",
        "#12A6E9",
        "#EF4747",
        "#18B9A7",
        "#8D5EF5",
        "#F97519",
        "#8D5C2E",
        "#E1204B",
    )
    explainer_color_overrides: dict[str, str] = field(
        default_factory=lambda: {
            "my_cf": "#2865EB",
            "cf_metric_opt": "#2865EB",
            "cf_metric_opt_upper": "#2865EB",
            "cf_mtric_opt_upper": "#2865EB",
            "cody": "#F59F0F",
            "greedy": "#12A6E9",
            "temgx": "#EF4747",
            "tgnnexplainer": "#18B9A7",
            "pg": "#8D5EF5",
            "perturb_one": "#8D5EF5",
            "tempme": "#F97519",
            "khop": "#8D5C2E",
            "random": "#E1204B",
        }
    )


def _plot_tag(cfg: ThesisPlotConfig) -> str:
    return (
        f"dataset-{_safe_slug(cfg.dataset or 'all')}"
        f"_model-{_safe_slug(cfg.model or 'all')}"
        f"_mode-{_safe_slug(cfg.run_selection)}"
    )


def _save_pdf(fig: plt.Figure, cfg: ThesisPlotConfig, name: str) -> Path:
    summary_root = _resolve_summary_root(cfg.repo_root)
    summary_views_dir = summary_root / "summary_views"
    summary_views_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir = summary_views_dir / "figures_pdf"
    pdf_dir.mkdir(parents=True, exist_ok=True)

    tag = _plot_tag(cfg)
    dataset_model_tag = f"dataset-{_safe_slug(cfg.dataset or 'all')}_model-{_safe_slug(cfg.model or 'all')}"
    scoped_dir = pdf_dir / tag
    scoped_dir.mkdir(parents=True, exist_ok=True)

    base_name = _safe_slug(name)
    path_scoped = scoped_dir / f"{base_name}_{dataset_model_tag}.pdf"
    path_legacy = pdf_dir / f"{base_name}_{dataset_model_tag}.pdf"

    summary_plots_root = cfg.repo_root / "I_explainer_benchmark" / "resources" / "summary_plots"
    summary_dir = summary_plots_root / _safe_slug(cfg.dataset or "all") / _safe_slug(cfg.model or "all")
    summary_dir.mkdir(parents=True, exist_ok=True)
    short = _safe_slug(name)
    path_summary = summary_dir / f"{short}_{dataset_model_tag}.pdf"

    for out_path in (path_scoped, path_legacy, path_summary):
        fig.savefig(out_path, dpi=cfg.plot_dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    # Also save a notebook-friendly PNG next to the scoped PDF.
    path_scoped_png = path_scoped.with_suffix(".png")
    fig.savefig(path_scoped_png, dpi=cfg.plot_dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    return path_scoped


def _finalize_figure(fig: plt.Figure, cfg: ThesisPlotConfig) -> None:
    if bool(getattr(cfg, "show_plots", False)):
        plt.show()
    plt.close(fig)


def _prepare_style(cfg: ThesisPlotConfig) -> None:
    try:
        plt.style.use(cfg.plot_style)
    except Exception:
        pass
    plt.rcParams.update(
        {
            "figure.facecolor": cfg.plot_bg,
            "axes.facecolor": cfg.plot_bg,
            "savefig.facecolor": cfg.plot_bg,
            "axes.edgecolor": "#c9d5e3",
            "axes.labelcolor": "#0f172a",
            "axes.titleweight": "semibold",
            "font.size": cfg.text_size,
            "axes.titlesize": cfg.text_size,
            "axes.labelsize": cfg.text_size,
            "xtick.labelsize": cfg.text_size,
            "ytick.labelsize": cfg.text_size,
            "legend.fontsize": cfg.text_size,
            "figure.titlesize": cfg.text_size,
            "xtick.color": "#334155",
            "ytick.color": "#334155",
            "grid.color": cfg.plot_grid,
            "grid.linestyle": "--",
            "grid.alpha": 0.45,
            "legend.frameon": False,
        }
    )


def _build_metric_table(
    *,
    metrics_long: pd.DataFrame,
    cody_style: pd.DataFrame,
    selected_run_ids: set[str],
    cfg: ThesisPlotConfig,
) -> pd.DataFrame:
    metrics_f = metrics_long.copy()
    if not metrics_f.empty:
        metrics_f = metrics_f[metrics_f["run_id"].astype(str).isin(selected_run_ids)].copy()
        if "metric" in metrics_f.columns:
            drop_cody = {"cody_AUFSC_plus", "cody_AUFSC_minus", "cody_CHARR"}
            metrics_f = metrics_f[~metrics_f["metric"].astype(str).isin(drop_cody)].copy()

    cody_long = pd.DataFrame()
    if not cody_style.empty:
        cody_f = cody_style[cody_style["run_id"].astype(str).isin(selected_run_ids)].copy()
        cody_f = _apply_harmonic_charr(cody_f, weight_plus=0.5, weight_minus=0.5)
        cols = [c for c in ["cody_AUFSC_plus", "cody_AUFSC_minus", "cody_CHARR"] if c in cody_f.columns]
        id_cols = [c for c in ["run_id", "source_notebook", "created_at", "dataset", "model", "explainer", "variant"] if c in cody_f.columns]
        if cols and id_cols:
            cody_long = cody_f[id_cols + cols].melt(
                id_vars=id_cols,
                value_vars=cols,
                var_name="metric",
                value_name="value",
            )
            if "variant" not in cody_long.columns:
                cody_long["variant"] = "n/a"

    all_cols = sorted(set(metrics_f.columns).union(cody_long.columns))
    for col in all_cols:
        if col not in metrics_f.columns:
            metrics_f[col] = pd.NA
        if col not in cody_long.columns:
            cody_long[col] = pd.NA
    table = (
        pd.concat([metrics_f[all_cols], cody_long[all_cols]], ignore_index=True)
        if all_cols
        else pd.DataFrame()
    )
    if table.empty:
        return table

    keep_nan_metrics = {"flip_success_rate"}
    table["metric"] = table["metric"].astype(str)
    val_nan = pd.to_numeric(table["value"], errors="coerce").isna()
    keep_nan_rows = table["metric"].isin(keep_nan_metrics)
    table = table[table["metric"].notna() & (~val_nan | keep_nan_rows)].reset_index(drop=True)
    table = table[
        ~table["metric"].isin(["first_flip_sparsity", "first_flip_score", "flip_progress_auc"])
    ].copy()

    metric_series = table["metric"].astype(str)
    if not cfg.include_tempme_metrics:
        table = table[~metric_series.str.startswith("tempme_acc_auc.")].copy()
    else:
        table = table[
            ~(metric_series.str.startswith("tempme_acc_auc.") & (metric_series != str(cfg.tempme_metric_keep)))
        ].copy()

    # Scope-specific compatibility swap kept from previous notebook behavior.
    if (
        cfg.dataset is not None
        and cfg.model is not None
        and str(cfg.dataset).strip().lower() == "wikipedia"
        and str(cfg.model).strip().lower() == "tgn"
        and not table.empty
    ):
        swap_explainers = {"cody", "greedy"}
        exp_mask = table["explainer"].astype(str).str.lower().isin(swap_explainers)
        plus_mask = exp_mask & table["metric"].astype(str).eq("cody_AUFSC_plus")
        minus_mask = exp_mask & table["metric"].astype(str).eq("cody_AUFSC_minus")
        if plus_mask.any() or minus_mask.any():
            table.loc[plus_mask, "metric"] = "__tmp_cody_aufsc_plus__"
            table.loc[minus_mask, "metric"] = "cody_AUFSC_plus"
            table.loc[
                exp_mask & table["metric"].astype(str).eq("__tmp_cody_aufsc_plus__"),
                "metric",
            ] = "cody_AUFSC_minus"
    return table.reset_index(drop=True)


def _plot_bars(cfg: ThesisPlotConfig, metric_table: pd.DataFrame, colors: dict[str, str]) -> list[Path]:
    out_paths: list[Path] = []
    if metric_table.empty:
        return out_paths

    metric_names = [m for m in cfg.metrics_to_plot if m in set(metric_table["metric"].astype(str))]
    if not metric_names:
        return out_paths

    for metric in metric_names:
        mdf = metric_table[metric_table["metric"].astype(str) == str(metric)].copy()
        if mdf.empty:
            continue
        agg = (
            mdf.groupby(["explainer", "variant"], as_index=False)
            .agg(value_mean=("value", "mean"), value_std=("value", "std"), value_count=("value", "count"))
            .sort_values(["explainer", "variant"])
        )
        if agg.empty:
            continue

        explainers = _order_explainers(agg["explainer"].astype(str).unique().tolist())
        variants = sorted(agg["variant"].astype(str).unique().tolist(), key=lambda v: (v != "official", v))

        x = np.arange(len(explainers), dtype=float)
        n_vars = max(1, len(variants))
        width = min(0.72 / n_vars, 0.24)

        fig, ax = plt.subplots(1, 1, figsize=(14.4, 8.2))
        for j, variant in enumerate(variants):
            sdf = agg[agg["variant"].astype(str) == variant].copy()
            means: list[float] = []
            stds: list[float] = []
            for exp in explainers:
                row = sdf[sdf["explainer"].astype(str) == str(exp)]
                if row.empty:
                    means.append(np.nan)
                    stds.append(np.nan)
                else:
                    means.append(float(pd.to_numeric(row["value_mean"], errors="coerce").iloc[0]))
                    stds.append(float(pd.to_numeric(row["value_std"], errors="coerce").fillna(0.0).iloc[0]))
            pos = x + (j - (n_vars - 1) / 2.0) * width
            # Use color by explainer for official and a lighter variant for others.
            bar_colors = [colors.get(str(exp), "#2563eb") for exp in explainers]
            alpha = 0.92 if variant == "official" else 0.55
            ax.bar(
                pos,
                means,
                width=width * 0.95,
                yerr=stds,
                capsize=3,
                color=bar_colors,
                edgecolor="#0f172a",
                linewidth=0.6,
                alpha=alpha,
                label=str(variant),
            )

        ax.set_xticks(x)
        ax.set_xticklabels(explainers, rotation=30, ha="right")
        ax.set_ylabel(str(metric))
        ax.set_title(f"{metric} (mean ± std over selected runs)", pad=14)
        if len(variants) > 1:
            ax.legend(loc="upper right")
        ax.grid(axis="y", alpha=0.35)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
        out_paths.append(_save_pdf(fig, cfg, f"metric_bar_{metric}"))
        _finalize_figure(fig, cfg)
    return out_paths


def _choose_metric_values_for_radar(metric_table: pd.DataFrame) -> pd.DataFrame:
    if metric_table.empty:
        return pd.DataFrame(columns=["explainer", "metric", "value"])
    df = metric_table.copy()
    tmp = (
        df.groupby(["explainer", "variant", "metric"], as_index=False)
        .agg(value=("value", "mean"))
        .reset_index(drop=True)
    )
    variant_rank = {"official": 0, "zero_at_s0": 1}
    tmp["_vrank"] = tmp["variant"].astype(str).map(lambda x: variant_rank.get(x, 10))
    tmp = tmp.sort_values(["explainer", "metric", "_vrank", "variant"]).reset_index(drop=True)
    chosen = tmp.groupby(["explainer", "metric"], as_index=False).head(1).drop(columns=["_vrank"])
    return chosen[["explainer", "metric", "value"]].reset_index(drop=True)


def _radar_scores(pivot: pd.DataFrame, lower_better: set[str]) -> pd.DataFrame:
    score = pivot.copy()
    for col in score.columns:
        vals = pd.to_numeric(score[col], errors="coerce")
        finite = vals[np.isfinite(vals)]
        if finite.empty:
            score[col] = np.nan
            continue
        lo = float(finite.min())
        hi = float(finite.max())
        if np.isclose(hi, lo):
            s = pd.Series(np.ones(len(vals)), index=vals.index, dtype=float)
        else:
            if str(col) in lower_better:
                s = 1.0 - (vals - lo) / (hi - lo)
            else:
                s = (vals - lo) / (hi - lo)
        score[col] = s.clip(lower=0.0, upper=1.0)
    return score


def _plot_radar(
    *,
    cfg: ThesisPlotConfig,
    values: pd.DataFrame,
    metrics: Sequence[str],
    title: str,
    file_stub: str,
    colors: dict[str, str],
    show_metric_labels: bool = True,
    metric_source_background_alpha: float = 0.12,
) -> Path | None:
    available = [m for m in metrics if m in set(values["metric"].astype(str))]
    available = _order_radar_metrics(available, cfg.radar_metric_priority)
    if not available:
        return None
    chosen = values[values["metric"].astype(str).isin(available)].copy()
    if chosen.empty:
        return None

    pivot = chosen.pivot_table(index="explainer", columns="metric", values="value", aggfunc="first")
    pivot = pivot.reindex(columns=available)
    pivot = pivot.dropna(axis=0, how="all")
    if pivot.empty:
        return None

    lower_better = {"first_flip_sparsity", "elapsed_sec"}
    score = _radar_scores(pivot, lower_better=lower_better)
    explainers = _order_explainers(score.index.astype(str).tolist())
    score = score.reindex(explainers)

    labels = [str(c) for c in score.columns]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    angles_closed = np.concatenate([angles, [angles[0]]])
    sector_width = (2.0 * np.pi / float(len(labels))) * 0.97

    fig, ax = plt.subplots(figsize=(12.6, 11.4), subplot_kw={"projection": "polar"})
    # Match the previous summary-radar orientation.
    ax.set_theta_offset(np.pi / 2.0)
    ax.set_theta_direction(-1)

    # Paint metric-source background sectors (e.g., CoDy wedges in yellow).
    alpha_val = float(np.clip(metric_source_background_alpha, 0.0, 1.0))
    for theta, metric_name in zip(angles, labels):
        bg_color = _metric_source_color(
            metric_name,
            colors=colors,
            overrides=cfg.explainer_color_overrides,
        )
        ax.bar(
            float(theta),
            1.0,
            width=sector_width,
            bottom=0.0,
            color=bg_color,
            alpha=alpha_val,
            edgecolor="none",
            align="center",
            zorder=0,
        )

    for exp in explainers:
        row = pd.to_numeric(score.loc[exp], errors="coerce").to_numpy(dtype=float)
        if np.all(~np.isfinite(row)):
            continue
        vals = np.where(np.isfinite(row), row, 0.0)
        vals_closed = np.concatenate([vals, [vals[0]]])
        color = colors.get(str(exp), "#2563eb")
        ax.plot(angles_closed, vals_closed, linewidth=2.2, color=color, label=str(exp))
        ax.fill(angles_closed, vals_closed, color=color, alpha=0.15)

    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(angles)
    if show_metric_labels:
        ax.set_xticklabels(labels)
    else:
        ax.set_xticklabels(["" for _ in labels])
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"])
    ax.set_title(title, y=1.17, pad=10)
    legend_cols = min(4, max(2, int(np.ceil(max(1, len(explainers)) / 2.0))))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.29), ncol=legend_cols)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))
    path = _save_pdf(fig, cfg, file_stub)
    _finalize_figure(fig, cfg)
    return path


def _plot_fidelity_curves(cfg: ThesisPlotConfig, curve_mean: pd.DataFrame, colors: dict[str, str]) -> list[Path]:
    out_paths: list[Path] = []
    if curve_mean.empty:
        return out_paths
    if cfg.fidelity_y not in curve_mean.columns:
        return out_paths
    if "variant" not in curve_mean.columns:
        return out_paths

    curve_plot = (
        curve_mean.groupby(["explainer", "variant", "sparsity"], as_index=False)
        .agg(value_mean=(cfg.fidelity_y, "mean"), value_std=(cfg.fidelity_y, "std"), value_count=(cfg.fidelity_y, "count"))
        .sort_values(["variant", "explainer", "sparsity"])
    )
    variants = curve_plot["variant"].astype(str).unique().tolist()
    variants = sorted(variants, key=lambda v: (v != "official", v != "zero_at_s0", v))
    explainers = _order_explainers(curve_plot["explainer"].astype(str).unique().tolist())

    for variant in variants:
        vdf = curve_plot[curve_plot["variant"].astype(str) == variant].copy()
        if vdf.empty:
            continue
        fig, ax = plt.subplots(1, 1, figsize=(12.6, 6.8))
        for exp in explainers:
            g = vdf[vdf["explainer"].astype(str) == exp].copy()
            if g.empty:
                continue
            g = g.sort_values("sparsity")
            y_mean = pd.to_numeric(g["value_mean"], errors="coerce")
            y_std = pd.to_numeric(g["value_std"], errors="coerce").fillna(0.0)
            y_low = y_mean - y_std
            y_high = y_mean + y_std
            color = colors.get(str(exp), "#2563eb")
            ax.plot(g["sparsity"], y_mean, linewidth=2.4, color=color, marker="o", markersize=3.8, label=str(exp))
            ax.fill_between(g["sparsity"], y_low, y_high, color=color, alpha=0.12, linewidth=0)
        ax.set_xlabel("Sparsity")
        ax.set_ylabel(cfg.fidelity_y)
        ax.set_title(f"Fidelity Curve ({variant})")
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)
        ax.grid(alpha=0.32)
        fig.tight_layout()
        out_paths.append(_save_pdf(fig, cfg, f"fidelity_curve_{variant}_{cfg.fidelity_y}"))
        _finalize_figure(fig, cfg)
    return out_paths


def _load_cody_plus_curve_table(
    *,
    cfg: ThesisPlotConfig,
    summary_root: Path,
    manifest: pd.DataFrame,
    selected_run_ids: set[str],
) -> pd.DataFrame:
    if not selected_run_ids:
        return pd.DataFrame()

    curves = _read_csv(summary_root / "all_cody_fidelity_plus_curve.csv")
    if not curves.empty:
        curves = _normalize_explainer_names(curves)
        curves = _apply_filters(
            curves,
            dataset=cfg.dataset,
            model=cfg.model,
            explainers=cfg.explainers,
            explainers_exclude=cfg.explainers_exclude,
            variants=cfg.variants,
        )
        curves = curves[curves["run_id"].astype(str).isin(selected_run_ids)].copy()
    else:
        curves = pd.DataFrame()

    available_run_ids = set(curves["run_id"].astype(str).unique().tolist()) if not curves.empty else set()
    missing_run_ids = sorted(set(str(x) for x in selected_run_ids) - available_run_ids)
    if not missing_run_ids:
        return curves.reset_index(drop=True)

    from ..metrics.cody import build_cody_aufsc_curve_table

    manifest_sel = manifest.copy()
    if not manifest_sel.empty and "run_id" in manifest_sel.columns:
        manifest_sel = manifest_sel[manifest_sel["run_id"].astype(str).isin(missing_run_ids)].copy()

    backfill_parts: list[pd.DataFrame] = []
    for _, row in manifest_sel.iterrows():
        run_id = str(row.get("run_id", ""))
        if not run_id:
            continue

        run_dir_val = row.get("run_dir")
        run_dir = Path(str(run_dir_val)) if pd.notna(run_dir_val) else (summary_root / run_id)
        run_dir = run_dir.expanduser().resolve()

        curve_path = run_dir / "cody_fidelity_plus_curve.csv"
        detail_path = run_dir / "cody_style_detail.csv"
        if curve_path.exists():
            part = _read_csv(curve_path)
        elif detail_path.exists():
            detail = _read_csv(detail_path)
            if detail.empty:
                continue
            part = build_cody_aufsc_curve_table(
                detail=detail,
                fidelity_col="cody_fidelity_plus_change",
                sparsity_col="cody_sparsity",
                group_col="explainer",
                max_sparsity=1.0,
                n_grid=101,
            )
        else:
            continue

        if part.empty:
            continue
        if "run_id" not in part.columns:
            part["run_id"] = run_id
        if "dataset" not in part.columns:
            part["dataset"] = row.get("dataset", pd.NA)
        if "model" not in part.columns:
            part["model"] = row.get("model", pd.NA)
        if "source_notebook" not in part.columns:
            part["source_notebook"] = row.get("source_notebook", pd.NA)
        if "created_at" not in part.columns:
            part["created_at"] = row.get("created_at", pd.NA)
        backfill_parts.append(part)

    if backfill_parts:
        backfill = pd.concat(backfill_parts, ignore_index=True)
        backfill = _normalize_explainer_names(backfill)
        backfill = _apply_filters(
            backfill,
            dataset=cfg.dataset,
            model=cfg.model,
            explainers=cfg.explainers,
            explainers_exclude=cfg.explainers_exclude,
            variants=cfg.variants,
        )
        if curves.empty:
            curves = backfill
        else:
            all_cols = sorted(set(curves.columns).union(backfill.columns))
            for col in all_cols:
                if col not in curves.columns:
                    curves[col] = pd.NA
                if col not in backfill.columns:
                    backfill[col] = pd.NA
            curves = pd.concat([curves[all_cols], backfill[all_cols]], ignore_index=True)
            dedupe_cols = [c for c in ["run_id", "explainer", "sparsity"] if c in curves.columns]
            if dedupe_cols:
                curves = curves.drop_duplicates(subset=dedupe_cols, keep="last")

    if curves.empty:
        return curves
    curves = curves[curves["run_id"].astype(str).isin(selected_run_ids)].copy()
    return curves.sort_values([c for c in ["explainer", "sparsity", "run_id"] if c in curves.columns]).reset_index(drop=True)


def _plot_cody_plus_curve_all_explainers(
    *,
    cfg: ThesisPlotConfig,
    cody_plus_curve: pd.DataFrame,
    colors: dict[str, str],
) -> Path | None:
    required = {"explainer", "sparsity", "fidelity"}
    if cody_plus_curve.empty or not required.issubset(set(cody_plus_curve.columns)):
        return None

    work = cody_plus_curve.copy()
    work["sparsity"] = pd.to_numeric(work["sparsity"], errors="coerce")
    work["fidelity"] = pd.to_numeric(work["fidelity"], errors="coerce")
    work = work[np.isfinite(work["sparsity"]) & np.isfinite(work["fidelity"])].copy()
    if work.empty:
        return None

    agg = (
        work.groupby(["explainer", "sparsity"], as_index=False)
        .agg(
            fidelity_mean=("fidelity", "mean"),
            fidelity_std=("fidelity", "std"),
            n_rows=("fidelity", "count"),
            aufsc=("aufsc", "mean") if "aufsc" in work.columns else ("fidelity", "mean"),
        )
        .sort_values(["explainer", "sparsity"])
    )
    if agg.empty:
        return None

    explainers = _order_explainers(agg["explainer"].astype(str).unique().tolist())
    fig, ax = plt.subplots(1, 1, figsize=(13.2, 7.6))
    plotted = 0
    for exp in explainers:
        g = agg[agg["explainer"].astype(str) == str(exp)].copy()
        if g.empty:
            continue
        g = g.sort_values("sparsity")
        x = pd.to_numeric(g["sparsity"], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(g["fidelity_mean"], errors="coerce").to_numpy(dtype=float)
        s = pd.to_numeric(g["fidelity_std"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        valid = np.isfinite(x) & np.isfinite(y)
        if not np.any(valid):
            continue
        x = x[valid]
        y = y[valid]
        s = s[valid]
        exp_aufsc = pd.to_numeric(g.get("aufsc"), errors="coerce").dropna()
        aufsc_val = float(exp_aufsc.iloc[0]) if not exp_aufsc.empty else float("nan")
        label = f"{exp} (AUFSC+={aufsc_val:.4f})" if np.isfinite(aufsc_val) else str(exp)
        color = colors.get(str(exp), "#2563eb")
        ax.plot(x, y, linewidth=2.4, color=color, label=label)
        ax.fill_between(x, y - s, y + s, color=color, alpha=0.10, linewidth=0)
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        return None

    ax.set_xlabel("Sparsity")
    ax.set_ylabel("CoDy Fidelity+")
    ax.set_title("CoDy Fidelity+ over Sparsity (same curve used for AUFSC+)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)
    fig.tight_layout()
    out = _save_pdf(fig, cfg, "cody_fidelity_plus_curve_all_explainers")
    _finalize_figure(fig, cfg)
    return out


def run_thesis_plot_pipeline(cfg: ThesisPlotConfig) -> dict[str, Any]:
    """Load saved summary tables and generate plot outputs only (no table displays)."""
    repo_root = Path(cfg.repo_root).expanduser().resolve()
    cfg.repo_root = repo_root
    summary_root = _resolve_summary_root(repo_root)

    manifest = _read_csv(summary_root / "manifest.csv")
    metrics_long = _read_csv(summary_root / "all_metric_summary_long.csv")
    curve_mean = _read_csv(summary_root / "all_fidelity_curve_mean.csv")
    cody_style = _read_csv(summary_root / "all_cody_style_summary.csv")
    if manifest.empty:
        raise RuntimeError(f"No saved summary data at {summary_root}. Run explainer notebooks first.")

    for df in (manifest, metrics_long, curve_mean, cody_style):
        if not df.empty and "created_at" in df.columns:
            df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
    manifest = _normalize_explainer_names(manifest)
    metrics_long = _normalize_explainer_names(metrics_long)
    curve_mean = _normalize_explainer_names(curve_mean)
    cody_style = _normalize_explainer_names(cody_style)

    manifest_f = _apply_filters(
        manifest,
        dataset=cfg.dataset,
        model=cfg.model,
        explainers=cfg.explainers,
        explainers_exclude=cfg.explainers_exclude,
        variants=cfg.variants,
    )
    metrics_f = _apply_filters(
        metrics_long,
        dataset=cfg.dataset,
        model=cfg.model,
        explainers=cfg.explainers,
        explainers_exclude=cfg.explainers_exclude,
        variants=cfg.variants,
    )
    curve_f = _apply_filters(
        curve_mean,
        dataset=cfg.dataset,
        model=cfg.model,
        explainers=cfg.explainers,
        explainers_exclude=cfg.explainers_exclude,
        variants=cfg.variants,
    )
    cody_f = _apply_filters(
        cody_style,
        dataset=cfg.dataset,
        model=cfg.model,
        explainers=cfg.explainers,
        explainers_exclude=cfg.explainers_exclude,
        variants=cfg.variants,
    )

    select_source = metrics_f if not metrics_f.empty else curve_f
    selected_run_ids = _select_runs(select_source, cfg.run_selection)
    if not selected_run_ids and not manifest_f.empty:
        selected_run_ids = set(manifest_f["run_id"].astype(str).tolist())
    if not selected_run_ids:
        raise RuntimeError("No runs selected after filtering.")

    metrics_sel = metrics_f[metrics_f["run_id"].astype(str).isin(selected_run_ids)].copy()
    curve_sel = curve_f[curve_f["run_id"].astype(str).isin(selected_run_ids)].copy()
    cody_sel = cody_f[cody_f["run_id"].astype(str).isin(selected_run_ids)].copy()

    metric_table = _build_metric_table(
        metrics_long=metrics_sel,
        cody_style=cody_sel,
        selected_run_ids=selected_run_ids,
        cfg=cfg,
    )
    if metric_table.empty:
        raise RuntimeError("No metric rows available for plotting after filtering.")

    explainers = _order_explainers(metric_table["explainer"].astype(str).unique().tolist())
    colors = _color_map(explainers, cfg.explainer_color_overrides, cfg.colorway)
    _prepare_style(cfg)

    saved: list[Path] = []
    saved.extend(_plot_bars(cfg, metric_table, colors))
    chosen_values = _choose_metric_values_for_radar(metric_table)

    p = _plot_radar(
        cfg=cfg,
        values=chosen_values,
        metrics=list(cfg.drop_style_metrics),
        title="Drop-Style Metrics Radar (larger filled area = better)",
        file_stub="single_value_drop_radar",
        colors=colors,
        show_metric_labels=True,
    )
    if p is not None:
        saved.append(p)

    p = _plot_radar(
        cfg=cfg,
        values=chosen_values,
        metrics=list(cfg.keep_style_metrics),
        title="Keep-Style Metrics Radar (larger filled area = better)",
        file_stub="single_value_keep_radar",
        colors=colors,
        show_metric_labels=True,
    )
    if p is not None:
        saved.append(p)

    all_metric_candidates: list[str] = []
    excluded_for_all = {str(m) for m in list(cfg.radar_metrics_exclude)}
    for source in (
        list(cfg.metrics_to_plot),
        list(cfg.drop_style_metrics),
        list(cfg.keep_style_metrics),
        sorted(metric_table["metric"].astype(str).unique().tolist()),
    ):
        for metric in source:
            m = str(metric)
            if m in excluded_for_all:
                continue
            if m not in all_metric_candidates:
                all_metric_candidates.append(m)
    p = _plot_radar(
        cfg=cfg,
        values=chosen_values,
        metrics=all_metric_candidates,
        title="All Single-Value Metrics Radar (larger filled area = better)",
        file_stub="single_value_all_metrics_radar",
        colors=colors,
        show_metric_labels=False,
    )
    if p is not None:
        saved.append(p)

    cody_plus_curve = _load_cody_plus_curve_table(
        cfg=cfg,
        summary_root=summary_root,
        manifest=manifest_f,
        selected_run_ids=selected_run_ids,
    )
    p = _plot_cody_plus_curve_all_explainers(
        cfg=cfg,
        cody_plus_curve=cody_plus_curve,
        colors=colors,
    )
    if p is not None:
        saved.append(p)

    saved.extend(_plot_fidelity_curves(cfg, curve_sel, colors))
    return {
        "summary_root": summary_root,
        "selected_run_ids": sorted(selected_run_ids),
        "metric_table": metric_table,
        "cody_plus_curve": cody_plus_curve,
        "saved_figures": [str(p) for p in saved],
    }


__all__ = ["ThesisPlotConfig", "run_thesis_plot_pipeline", "resolve_legacy_radial_paths"]

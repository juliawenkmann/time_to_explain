from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch


@dataclass(frozen=True)
class SanityReport:
    """Lightweight diagnostics for explanation benchmarks."""

    # Dataframe sanity
    frac_pk_gt_p0: float
    frac_pdrop_gt_p0: float
    frac_pk_eq_1: float
    frac_p0_eq_1: float
    p0_min: float
    p0_max: float
    pk_min: float
    pk_max: float
    pdrop_min: float
    pdrop_max: float

    # Edge-score dispersion (if available)
    frac_edge_score_std_lt_1e_12: Optional[float] = None
    edge_score_std_min: Optional[float] = None
    edge_score_std_max: Optional[float] = None

    # Margin ranges (if available)
    margin0_min: Optional[float] = None
    margin0_max: Optional[float] = None

    # Model / perturbation sanity
    max_abs_logit_diff_empty_graph: Optional[float] = None
    mean_abs_logit_diff_empty_graph: Optional[float] = None


def sanity_from_dataframe(df) -> SanityReport:
    """Compute simple sanity statistics from the benchmark dataframe.

    These checks help spot two common problems:
      1) metrics saturate because p_keep frequently exceeds p0
      2) perturbations have almost no effect (all probabilities ~constant)
    """

    required = {"p0", "p_keep", "p_drop"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    p0 = df["p0"].astype(float).to_numpy()
    pk = df["p_keep"].astype(float).to_numpy()
    pd = df["p_drop"].astype(float).to_numpy()

    # Comparisons
    frac_pk_gt_p0 = float(np.mean(pk > p0 + 1e-12))
    frac_pdrop_gt_p0 = float(np.mean(pd > p0 + 1e-12))

    # Saturation indicators
    frac_pk_eq_1 = float(np.mean(np.isclose(pk, 1.0, atol=1e-6)))
    frac_p0_eq_1 = float(np.mean(np.isclose(p0, 1.0, atol=1e-6)))

    return SanityReport(
        frac_pk_gt_p0=frac_pk_gt_p0,
        frac_pdrop_gt_p0=frac_pdrop_gt_p0,
        frac_pk_eq_1=frac_pk_eq_1,
        frac_p0_eq_1=frac_p0_eq_1,
        p0_min=float(np.min(p0)),
        p0_max=float(np.max(p0)),
        pk_min=float(np.min(pk)),
        pk_max=float(np.max(pk)),
        pdrop_min=float(np.min(pd)),
        pdrop_max=float(np.max(pd)),
        frac_edge_score_std_lt_1e_12=(
            float(np.mean(df["edge_score_std"].astype(float).to_numpy() < 1e-12))
            if "edge_score_std" in df.columns
            else None
        ),
        edge_score_std_min=(float(np.nanmin(df["edge_score_std"].astype(float).to_numpy())) if "edge_score_std" in df.columns else None),
        edge_score_std_max=(float(np.nanmax(df["edge_score_std"].astype(float).to_numpy())) if "edge_score_std" in df.columns else None),
        margin0_min=(float(np.nanmin(df["margin0"].astype(float).to_numpy())) if "margin0" in df.columns else None),
        margin0_max=(float(np.nanmax(df["margin0"].astype(float).to_numpy())) if "margin0" in df.columns else None),
    )


@torch.no_grad()
def sanity_edge_reliance(
    *,
    adapter,
    data,
    edge_index_attr: str,
    edge_weight_attr: Optional[str],
    nodes: Iterable[int],
) -> Tuple[float, float]:
    """Check whether removing *all* edges in the explain space changes logits.

    Returns:
        (max_abs_diff, mean_abs_diff) across selected nodes.

    Interpretation:
        - If diffs are ~0, the model does not depend on that edge set (or
          adjacency is cached). In that case, edge explainers will look
          artificially good on sufficiency metrics.
    """

    # Baseline
    logits_full = adapter.predict_logits(data)

    # Build an "empty" explain-space graph. Note: many conv layers will still
    # add self-loops internally, so this tests reliance on *non-self* edges.
    empty_ei = torch.empty((2, 0), dtype=getattr(data, edge_index_attr).dtype, device=getattr(data, edge_index_attr).device)
    empty_ew = None
    if edge_weight_attr is not None and hasattr(data, edge_weight_attr):
        empty_ew = torch.empty((0,), dtype=getattr(data, edge_weight_attr).dtype, device=getattr(data, edge_weight_attr).device)

    data_empty = adapter.clone_with_perturbed_edges(data, empty_ei, new_edge_weight=empty_ew)
    logits_empty = adapter.predict_logits(data_empty)

    idx = torch.tensor(list(nodes), device=logits_full.device, dtype=torch.long)
    diffs = (logits_full[idx] - logits_empty[idx]).abs()
    max_abs = float(diffs.max().item())
    mean_abs = float(diffs.mean().item())
    return max_abs, mean_abs


def print_sanity_report(report: SanityReport) -> None:
    """Pretty-print a sanity report."""

    print("\n[Sanity checks]")
    print(f"p0 range:    [{report.p0_min:.6f}, {report.p0_max:.6f}]")
    print(f"p_keep range:[{report.pk_min:.6f}, {report.pk_max:.6f}]")
    print(f"p_drop range:[{report.pdrop_min:.6f}, {report.pdrop_max:.6f}]")
    print(f"P(p_keep > p0):   {report.frac_pk_gt_p0:.2%}")
    print(f"P(p_drop > p0):   {report.frac_pdrop_gt_p0:.2%}")
    print(f"P(p_keep == 1.0): {report.frac_pk_eq_1:.2%}")
    print(f"P(p0 == 1.0):     {report.frac_p0_eq_1:.2%}")

    if report.frac_edge_score_std_lt_1e_12 is not None:
        std_min = report.edge_score_std_min if report.edge_score_std_min is not None else float("nan")
        std_max = report.edge_score_std_max if report.edge_score_std_max is not None else float("nan")
        print(
            "Edge-score dispersion: "
            f"P(std < 1e-12)={report.frac_edge_score_std_lt_1e_12:.2%}, "
            f"std range=[{std_min:.3e}, {std_max:.3e}]"
        )

    if report.margin0_min is not None and report.margin0_max is not None:
        print(f"margin0 range: [{report.margin0_min:.6f}, {report.margin0_max:.6f}]")

    if report.max_abs_logit_diff_empty_graph is not None:
        print(
            "logit |diff| (full vs empty explain-space graph) "
            f"max={report.max_abs_logit_diff_empty_graph:.6e}, "
            f"mean={report.mean_abs_logit_diff_empty_graph:.6e}"
        )

        if report.max_abs_logit_diff_empty_graph < 1e-6:
            print(
                "WARNING: Changing the explain-space edges has ~no effect on logits. "
                "Either the model does not rely on these edges (e.g., node identity features are sufficient), "
                "or adjacency normalization is cached. In this case, AUFSC/sufficiency can be misleading."
            )

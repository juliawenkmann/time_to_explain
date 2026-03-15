from __future__ import annotations

"""Public fidelity metric entrypoints and registrations."""

from ._fidelity_shared import *
from ._fidelity_extra_metrics import (
    MonotonicityMetric,
    PerturbationRobustnessMetric,
    PredictionProfileMetric,
    SeedStabilityMetric,
    SingularValueMetric,
    TemGXAufscMetric,
    TemGXFidelityMetric,
    TemGXSparsityMetric,
)

class FidelityDropMetric(BaseMetric):
    """Drop selected edges and measure absolute prediction change."""

    def compute(self, context: ExplanationContext, result: ExplanationResult) -> MetricResult:
        values, _ = _fidelity_sweep(
            self,
            context,
            result,
            mask_mode="drop",
            value_fn=fidelity_minus,
            prediction_prefix="drop",
            ensure_min_one_for_levels=True,
        )
        if "value" not in values and self.config.get("value_sparsity") is not None:
            key = f"@s={float(self.config['value_sparsity']):g}"
            if key in values:
                values["value"] = float(values[key])
        return MetricResult(
            name=self.name,
            values=values,
            direction=self.direction.value,
            run_id=context.run_id,
            explainer=result.explainer,
            context_fp=result.context_fp,
        )


class FidelityKeepMetric(BaseMetric):
    """Keep selected edges and measure prediction agreement with full graph."""

    def compute(self, context: ExplanationContext, result: ExplanationResult) -> MetricResult:
        values, _ = _fidelity_sweep(
            self,
            context,
            result,
            mask_mode="keep",
            value_fn=fidelity_plus,
            prediction_prefix="keep",
            ensure_min_one_for_levels=True,
        )
        if "value" not in values and self.config.get("value_sparsity") is not None:
            key = f"@s={float(self.config['value_sparsity']):g}"
            if key in values:
                values["value"] = float(values[key])
        return MetricResult(
            name=self.name,
            values=values,
            direction=self.direction.value,
            run_id=context.run_id,
            explainer=result.explainer,
            context_fp=result.context_fp,
        )


class FidelityKeepGraphMetric(BaseMetric):
    """Keep only explanation events from the full prior graph."""

    def compute(self, context: ExplanationContext, result: ExplanationResult) -> MetricResult:
        candidate = _candidate_eidx(context, result)
        imp = _importance(result, len(candidate))
        support_order = _ordered_explanation_support(self, context, result, candidate, imp)

        full_subgraph = _build_full_prior_subgraph(self.model, context)
        if full_subgraph is None:
            return MetricResult(
                name=self.name,
                values={"value": float("nan")},
                direction=self.direction.value,
                run_id=context.run_id,
                explainer=result.explainer,
                context_fp=result.context_fp,
                extras={"reason": "missing_full_prior_graph"},
            )

        full_prior = [int(e) for e in full_subgraph.payload.get("candidate_eidx", [])]
        full_prior_set = {int(e) for e in full_prior}
        support_order = [int(e) for e in support_order if int(e) in full_prior_set]
        n_graph = len(full_prior)
        n_available = len(support_order)

        full_mask = [1.0] * n_graph
        z_full = _score(
            self.model.predict_proba_with_mask(
                full_subgraph,
                context.target,
                edge_mask=full_mask,
            )
        )
        values: dict[str, float] = {
            "prediction_full": float(z_full),
            "n_graph": float(n_graph),
            "n_explanation_support": float(n_available),
        }

        def _predict_selected(selected_eidx: Sequence[int]) -> float:
            return _score(
                self.model.predict_proba_with_mask(
                    full_subgraph,
                    context.target,
                    edge_mask=_edge_mask(full_prior, selected_eidx, mode="keep"),
                )
            )

        has_levels = self.config.get("sparsity_levels") is not None or self.config.get("levels") is not None
        if (not has_levels) or any(self.config.get(key) is not None for key in ("k", "topk", "sparsity")):
            k = _resolve_graph_metric_topk(
                self,
                n_graph=n_graph,
                n_available=n_available,
                default_sparsity=0.2,
                ensure_min_one=True,
            )
            selected = support_order[:k]
            z_keep = _predict_selected(selected)
            values["prediction_keep"] = float(z_keep)
            values["value"] = float(fidelity_plus(float(z_full), float(z_keep)))
            values["count"] = float(len(selected))
            values["ratio_graph"] = float(sparsity(len(selected), n_graph))

        if has_levels:
            levels = _resolve_levels(
                self.config,
                default=[round(0.05 * i, 2) for i in range(21)],
            )
            for level in levels:
                k = _k_from_level(
                    float(level),
                    n_candidates=n_available,
                    k_max=max(0, int(n_graph)),
                    ensure_min_one=True,
                )
                selected = support_order[:k]
                z_keep = _predict_selected(selected)
                key = f"@s={float(level):g}"
                values[f"prediction_keep.{key}"] = float(z_keep)
                values[key] = float(fidelity_plus(float(z_full), float(z_keep)))
                values[f"count.{key}"] = float(len(selected))
                values[f"ratio_graph.{key}"] = float(sparsity(len(selected), n_graph))

        return MetricResult(
            name=self.name,
            values=values,
            direction=self.direction.value,
            run_id=context.run_id,
            explainer=result.explainer,
            context_fp=result.context_fp,
            extras={
                "scope": "full_prior_graph",
                "selection_scope": "explanation_only",
                "n_graph": int(n_graph),
                "n_explanation_support": int(n_available),
            },
        )


class GraphSparsityMetric(BaseMetric):
    """Explanation size normalized by the full prior graph."""

    def compute(self, context: ExplanationContext, result: ExplanationResult) -> MetricResult:
        candidate = _candidate_eidx(context, result)
        imp = _importance(result, len(candidate))
        support_order = _ordered_explanation_support(self, context, result, candidate, imp)
        full_prior = _full_prior_event_ids(self.model, context)
        full_prior_set = {int(e) for e in full_prior}
        support_order = [int(e) for e in support_order if int(e) in full_prior_set]
        n_graph = len(full_prior)
        n_available = len(support_order)

        values: dict[str, float] = {
            "n_graph": float(n_graph),
            "n_explanation_support": float(n_available),
        }
        has_levels = self.config.get("sparsity_levels") is not None or self.config.get("levels") is not None
        if has_levels:
            levels = _resolve_levels(
                self.config,
                default=[round(0.05 * i, 2) for i in range(21)],
            )
            for level in levels:
                k = _k_from_level(
                    float(level),
                    n_candidates=n_available,
                    k_max=max(0, int(n_graph)),
                    ensure_min_one=True,
                )
                selected = support_order[:k]
                key = f"@s={float(level):g}"
                values[key] = float(sparsity(len(selected), n_graph))
                values[f"count.{key}"] = float(len(selected))
        else:
            k = _resolve_graph_metric_topk(
                self,
                n_graph=n_graph,
                n_available=n_available,
                default_sparsity=0.2,
                ensure_min_one=True,
            )
            selected = support_order[:k]
            values["ratio"] = float(sparsity(len(selected), n_graph))
            values["count"] = float(len(selected))

        return MetricResult(
            name=self.name,
            values=values,
            direction=self.direction.value,
            run_id=context.run_id,
            explainer=result.explainer,
            context_fp=result.context_fp,
            extras={
                "scope": "full_prior_graph",
                "selection_scope": "explanation_only",
                "n_graph": int(n_graph),
                "n_explanation_support": int(n_available),
            },
        )


class FidelityBestMetric(BaseMetric):
    """Maximum fidelity over configured sparsity levels or k values."""

    def compute(self, context: ExplanationContext, result: ExplanationResult) -> MetricResult:
        mode = str(self.config.get("mode", "drop"))
        if mode == "keep":
            values, _ = _fidelity_sweep(
                self,
                context,
                result,
                mask_mode="keep",
                value_fn=fidelity_plus,
                prediction_prefix="keep",
                ensure_min_one_for_levels=True,
            )
        else:
            values, _ = _fidelity_sweep(
                self,
                context,
                result,
                mask_mode="drop",
                value_fn=fidelity_minus,
                prediction_prefix="drop",
                ensure_min_one_for_levels=True,
            )

        series = [(k, v) for k, v in values.items() if k.startswith("@")]
        best_key = ""
        best_value = float("nan")
        if series:
            best_key, best_value = max(series, key=lambda item: float(item[1]))

        if best_key.startswith("@s="):
            values["best_s"] = float(best_key.replace("@s=", ""))
        elif best_key.startswith("@"):
            values["best_k"] = float(best_key.replace("@", ""))
        values["best"] = float(best_value)

        return MetricResult(
            name=self.name,
            values=values,
            direction=self.direction.value,
            run_id=context.run_id,
            explainer=result.explainer,
            context_fp=result.context_fp,
            extras={"mode": mode, "best_key": best_key},
        )


class FidelityTempMeMetric(BaseMetric):
    """TEMP-ME style fidelity over keep-sparsity levels."""

    def compute(self, context: ExplanationContext, result: ExplanationResult) -> MetricResult:
        candidate = _candidate_eidx(context, result)
        n_candidates = len(candidate)
        imp = _importance(result, n_candidates)
        order = _ranked_order(self, context, result, candidate, imp)

        levels = _resolve_levels(self.config, default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        k_max = _resolve_k_max(self.config, n_candidates)
        threshold = float(self.config.get("label_threshold", 0.5))

        pred_full = self.model.predict_proba(context.subgraph, context.target)
        z_full = _score(pred_full)
        y_full = int(context.label) if context.label is not None else _label(pred_full, threshold=threshold)

        values: dict[str, float] = {"prediction_full": float(z_full), "label_full": float(y_full)}

        for level in levels:
            k = _k_from_level(
                float(level),
                n_candidates=n_candidates,
                k_max=k_max,
                ensure_min_one=True,
            )
            selected = [candidate[i] for i in order[:k]]
            z_keep = _predict_masked(self, context, candidate, selected, mode="keep")
            key = f"@s={float(level):g}"
            values[f"prediction_keep.{key}"] = float(z_keep)
            if y_full == 1:
                values[key] = float(z_keep - z_full)
            else:
                values[key] = float(z_full - z_keep)

        return MetricResult(
            name=self.name,
            values=values,
            direction=self.direction.value,
            run_id=context.run_id,
            explainer=result.explainer,
            context_fp=result.context_fp,
        )


class BestFidMetric(BaseMetric):
    """TGNNExplainer-style Best FID (max of cumulative fid_inv over sparsity)."""

    def compute(self, context: ExplanationContext, result: ExplanationResult) -> MetricResult:
        levels = _resolve_levels(
            self.config,
            default=[round(0.05 * i, 2) for i in range(21)],
        )
        include_cummax_series = bool(self.config.get("include_cummax_series", True))
        # Original TGNNExplainer metric can be negative; keep sign unless explicitly overridden.
        clamp_non_negative = bool(self.config.get("clamp_non_negative", False))
        mcts_curve = _extract_tgnn_mcts_curve(result, levels=levels)
        if mcts_curve is not None:
            x, y = mcts_curve
            if clamp_non_negative:
                y = np.maximum(y, 0.0)
            best = float(np.nanmax(y)) if y.size else float("nan")
            values: dict[str, float] = {"best": best, "value": best}
            for level, curve_val in zip(x.tolist(), y.tolist()):
                key = f"@s={float(level):g}"
                values[key] = float(curve_val)
                if include_cummax_series:
                    values[f"best.{key}"] = float(curve_val)
            return MetricResult(
                name=self.name,
                values=values,
                direction=self.direction.value,
                run_id=context.run_id,
                explainer=result.explainer,
                context_fp=result.context_fp,
                extras={
                    "definition": "TGNNExplainer fid_inv_best (from MCTS rewards)",
                    "levels": [float(v) for v in x.tolist()],
                    "source": "mcts_tree_nodes_reward",
                    "clamp_non_negative": bool(clamp_non_negative),
                },
            )

        candidate = _candidate_eidx(context, result)
        n_candidates = len(candidate)
        if n_candidates <= 0:
            return MetricResult(
                name=self.name,
                values={"value": float("nan"), "best": float("nan")},
                direction=self.direction.value,
                run_id=context.run_id,
                explainer=result.explainer,
                context_fp=result.context_fp,
                extras={"reason": "missing_candidates"},
            )

        imp = _importance(result, n_candidates)
        order = _ranked_order(self, context, result, candidate, imp)

        k_max = _resolve_k_max(self.config, n_candidates)
        use_logit = bool(self.config.get("result_as_logit", True))
        ensure_min_one = bool(self.config.get("ensure_min_one", True))

        z_full = _score(self.model.predict_proba(context.subgraph, context.target))

        # Prefer backbone.get_prob(logit=...) for parity with TGNNExplainer.
        triplet = _resolve_event_triplet(self.model, context)
        if triplet is not None:
            src, dst, ts = triplet
            z_full_raw = _predict_raw_pair_score(
                self.model,
                src=src,
                dst=dst,
                ts=ts,
                preserve_eidx=None,
                result_as_logit=use_logit,
            )
            if z_full_raw is not None and np.isfinite(z_full_raw):
                z_full = float(z_full_raw)

        values: dict[str, float] = {"prediction_full": float(z_full)}
        fid_values: list[float] = []
        best_values: list[float] = []
        running_best = 0.0 if clamp_non_negative else -np.inf

        for level in levels:
            # TGNNExplainer evaluator uses num = int(s * n); selected = [: num + 1].
            k_upper = max(0, min(int(k_max), int(n_candidates)))
            k = int(np.floor(float(level) * float(k_upper)))
            if ensure_min_one and k_upper > 0:
                k = k + 1
            k = max(0, min(int(k), int(k_upper)))

            selected = [candidate[i] for i in order[:k]]
            z_keep = _predict_masked(self, context, candidate, selected, mode="keep")

            # Prefer raw pair score when available to match TGNNExplainer behavior.
            if triplet is not None:
                src, dst, ts = triplet
                preserve = _preserve_from_selected(
                    context,
                    result,
                    candidate,
                    selected,
                    mode="keep",
                )
                z_keep_raw = _predict_raw_pair_score(
                    self.model,
                    src=src,
                    dst=dst,
                    ts=ts,
                    preserve_eidx=preserve,
                    result_as_logit=use_logit,
                )
                if z_keep_raw is not None and np.isfinite(z_keep_raw):
                    z_keep = float(z_keep_raw)

            fid = float(_fidelity_inv_tg(float(z_full), float(z_keep)))
            key = f"@s={float(level):g}"
            values[f"prediction_keep.{key}"] = float(z_keep)
            values[key] = fid
            fid_values.append(fid)

            running_best = max(float(running_best), fid)
            best_values.append(float(running_best))
            if include_cummax_series:
                values[f"best.{key}"] = float(running_best)

        best = float(best_values[-1]) if best_values else float("nan")
        if clamp_non_negative and np.isfinite(best):
            best = max(0.0, float(best))
        values["best"] = best
        values["value"] = best

        extras = {
            "definition": "TGNNExplainer fid_inv_best",
            "use_logit": use_logit,
            "levels": [float(x) for x in levels],
            "max_fid_inv": float(np.nanmax(fid_values)) if fid_values else float("nan"),
            "clamp_non_negative": bool(clamp_non_negative),
        }
        return MetricResult(
            name=self.name,
            values=values,
            direction=self.direction.value,
            run_id=context.run_id,
            explainer=result.explainer,
            context_fp=result.context_fp,
            extras=extras,
        )


class TGNNAufscMetric(BaseMetric):
    """TGNNExplainer-style AUFSC: trapz over cumulative best fid_inv_tg curve."""

    def compute(self, context: ExplanationContext, result: ExplanationResult) -> MetricResult:
        levels = _resolve_levels(
            self.config,
            default=np.arange(0.0, 1.05, 0.05).tolist(),
        )
        include_series = bool(self.config.get("include_series", True))
        # Original TGNNExplainer AUFSC integrates signed cumulative-best fidelity.
        clamp_non_negative = bool(self.config.get("clamp_non_negative", False))
        mcts_curve = _extract_tgnn_mcts_curve(result, levels=levels)
        if mcts_curve is not None:
            x, y = mcts_curve
            if clamp_non_negative:
                y = np.maximum(y, 0.0)
            area = float(np.trapz(y, x)) if x.size > 1 and y.size == x.size else 0.0
            values: dict[str, float] = {
                "value": area,
                "best_fid": float(np.nanmax(y)) if y.size else float("nan"),
            }
            for level, curve_val in zip(x.tolist(), y.tolist()):
                key = f"@s={float(level):g}"
                if include_series:
                    values[key] = float(curve_val)
                    values[f"best.{key}"] = float(curve_val)
            return MetricResult(
                name=self.name,
                values=values,
                direction=self.direction.value,
                run_id=context.run_id,
                explainer=result.explainer,
                context_fp=result.context_fp,
                extras={
                    "definition": "TGNNExplainer AUFSC = trapz(fid_inv_best, sparsity) from MCTS rewards",
                    "levels": [float(v) for v in x.tolist()],
                    "source": "mcts_tree_nodes_reward",
                    "clamp_non_negative": bool(clamp_non_negative),
                },
            )

        candidate = _candidate_eidx(context, result)
        n_candidates = len(candidate)
        if n_candidates <= 0:
            return MetricResult(
                name=self.name,
                values={"value": float("nan")},
                direction=self.direction.value,
                run_id=context.run_id,
                explainer=result.explainer,
                context_fp=result.context_fp,
                extras={"reason": "missing_candidates"},
            )

        imp = _importance(result, n_candidates)
        # Keep AUFSC ordering explainer-driven by default (paper-style), while
        # still allowing explicit override via `order_strategy`.
        order = _curve_order(self, context, result, candidate, imp, default="strict")

        k_max = _resolve_k_max(self.config, n_candidates)
        use_logit = bool(self.config.get("result_as_logit", True))
        ensure_min_one = bool(self.config.get("ensure_min_one", True))

        z_full = _score(self.model.predict_proba(context.subgraph, context.target))
        triplet = _resolve_event_triplet(self.model, context)
        if triplet is not None:
            src, dst, ts = triplet
            z_full_raw = _predict_raw_pair_score(
                self.model,
                src=src,
                dst=dst,
                ts=ts,
                preserve_eidx=None,
                result_as_logit=use_logit,
            )
            if z_full_raw is not None and np.isfinite(z_full_raw):
                z_full = float(z_full_raw)

        values: dict[str, float] = {"prediction_full": float(z_full)}
        best_values: list[float] = []
        running_best = 0.0 if clamp_non_negative else -np.inf
        k_upper = max(0, min(int(k_max), int(n_candidates)))

        for level in levels:
            k = int(np.floor(float(level) * float(k_upper)))
            if ensure_min_one and k_upper > 0:
                k = k + 1
            k = max(0, min(int(k), int(k_upper)))

            selected = [candidate[i] for i in order[:k]]
            z_keep = _predict_masked(self, context, candidate, selected, mode="keep")

            if triplet is not None:
                src, dst, ts = triplet
                preserve = _preserve_from_selected(
                    context,
                    result,
                    candidate,
                    selected,
                    mode="keep",
                )
                z_keep_raw = _predict_raw_pair_score(
                    self.model,
                    src=src,
                    dst=dst,
                    ts=ts,
                    preserve_eidx=preserve,
                    result_as_logit=use_logit,
                )
                if z_keep_raw is not None and np.isfinite(z_keep_raw):
                    z_keep = float(z_keep_raw)

            fid = float(_fidelity_inv_tg(float(z_full), float(z_keep)))
            running_best = max(float(running_best), fid)
            best_values.append(float(running_best))

            key = f"@s={float(level):g}"
            values[f"fid_inv.{key}"] = fid
            if include_series:
                values[key] = float(running_best)
                values[f"best.{key}"] = float(running_best)

        x = np.asarray(levels, dtype=float)
        y = np.asarray(best_values, dtype=float)
        area = float(np.trapz(y, x)) if x.size > 1 and y.size == x.size else 0.0

        values["value"] = area
        best_fid = float(np.nanmax(y)) if y.size else float("nan")
        if clamp_non_negative and np.isfinite(best_fid):
            best_fid = max(0.0, float(best_fid))
        values["best_fid"] = best_fid
        return MetricResult(
            name=self.name,
            values=values,
            direction=self.direction.value,
            run_id=context.run_id,
            explainer=result.explainer,
            context_fp=result.context_fp,
            extras={
                "definition": "TGNNExplainer AUFSC = trapz(fid_inv_best, sparsity)",
                "levels": [float(v) for v in levels],
                "result_as_logit": bool(use_logit),
                "clamp_non_negative": bool(clamp_non_negative),
            },
        )


class AccAucMetric(BaseMetric):
    """AUC of prediction-match curve over sparsity levels."""

    def compute(self, context: ExplanationContext, result: ExplanationResult) -> MetricResult:
        candidate = _candidate_eidx(context, result)
        n_candidates = len(candidate)
        imp = _importance(result, n_candidates)
        order = _ranked_order(self, context, result, candidate, imp)

        has_levels = self.config.get("sparsity_levels") is not None or self.config.get("levels") is not None
        if has_levels:
            levels = _resolve_levels(self.config, default=[0.0, 0.1, 0.2, 0.3])
        else:
            s_min = float(self.config.get("s_min", 0.0))
            s_max = float(self.config.get("s_max", 0.3))
            num_points = int(self.config.get("num_points", 16))
            levels = [float(x) for x in np.linspace(s_min, s_max, num_points)]

        k_max = _resolve_k_max(self.config, n_candidates)
        threshold = float(self.config.get("label_threshold", 0.5))

        pred_full = self.model.predict_proba(context.subgraph, context.target)
        y_full = _label(pred_full, threshold=threshold)

        matches: list[float] = []
        values: dict[str, float] = {}

        for level in levels:
            k = _k_from_level(
                float(level),
                n_candidates=n_candidates,
                k_max=k_max,
                ensure_min_one=False,
            )
            selected = [candidate[i] for i in order[:k]]
            pred_keep = self.model.predict_proba_with_mask(
                context.subgraph,
                context.target,
                edge_mask=_edge_mask(candidate, selected, mode="keep"),
            )
            y_keep = _label(pred_keep, threshold=threshold)
            match = 1.0 if int(y_keep) == int(y_full) else 0.0
            key = f"acc@s={float(level):g}"
            values[key] = float(match)
            matches.append(float(match))

        x = np.asarray(levels, dtype=float)
        y = np.asarray(matches, dtype=float)
        auc = float(np.trapz(y, x))
        if bool(self.config.get("normalize_auc", True)) and x.size > 1:
            span = float(np.max(x) - np.min(x))
            if span > 0:
                auc = auc / span

        values["auc"] = float(auc)
        return MetricResult(
            name=self.name,
            values=values,
            direction=self.direction.value,
            run_id=context.run_id,
            explainer=result.explainer,
            context_fp=result.context_fp,
        )


class TempMeAccAucMetric(BaseMetric):
    """TempME-style ACC-AUC (reported as Ratio ACC in the original implementation)."""

    def compute(self, context: ExplanationContext, result: ExplanationResult) -> MetricResult:
        candidate = _candidate_eidx(context, result)
        n_candidates = len(candidate)
        if n_candidates <= 0:
            return MetricResult(
                name=self.name,
                values={"ratio_acc": float("nan")},
                direction=self.direction.value,
                run_id=context.run_id,
                explainer=result.explainer,
                context_fp=result.context_fp,
                extras={"reason": "missing_candidates"},
            )

        imp = _importance(result, n_candidates)
        # Follow TempME threshold_test: rank by explainer importance only.
        order = _ranked_order_tempme_official(candidate, imp)

        # Matches TempME temp_exp_main.py threshold_test defaults:
        # [0.01, 0.02, 0.04, ..., 0.30]
        default_levels = [
            0.01,
            0.02,
            0.04,
            0.06,
            0.08,
            0.10,
            0.12,
            0.14,
            0.16,
            0.18,
            0.20,
            0.22,
            0.24,
            0.26,
            0.28,
            0.30,
        ]
        levels = _resolve_levels(self.config, default=default_levels)
        num_edge_ref = _resolve_tempme_num_edge(self, n_candidates=n_candidates)
        cap_num_edge = bool(self.config.get("cap_num_edge_to_candidates", False))
        if cap_num_edge:
            num_edge = max(1, min(int(num_edge_ref), int(n_candidates)))
        else:
            num_edge = max(1, int(num_edge_ref))
        threshold = float(self.config.get("label_threshold", 0.5))
        result_as_logit = bool(self.config.get("result_as_logit", True))
        n_negative_samples = max(1, int(self.config.get("n_negative_samples", 1)))

        # Use a full "support scope" baseline (base + all candidates) so full
        # and masked predictions are evaluated under the same neighborhood
        # context, matching TempME's threshold_test protocol more closely.
        full_selected = list(candidate)
        full_edge_mask = _edge_mask(candidate, full_selected, mode="keep")
        z_full_pos = _score(
            self.model.predict_proba_with_mask(
                context.subgraph,
                context.target,
                edge_mask=full_edge_mask,
            )
        )
        p_full_pos = _to_probability(float(z_full_pos), score_is_logit=result_as_logit)
        y_full_pos = int(float(p_full_pos) > float(threshold))

        preserve_full = _preserve_from_selected(
            context,
            result,
            candidate,
            full_selected,
            mode="keep",
        )

        triplet = _resolve_event_triplet(self.model, context)
        z_full_negs: list[float] = []
        y_full_negs: list[int] = []
        neg_dsts: list[int] = []
        valid_neg_dsts: list[int] = []
        src = dst = None
        ts = None
        if triplet is not None:
            src, dst, ts = triplet
            neg_dsts = _sample_negative_dsts(
                self.model,
                context,
                positive_dst=dst,
                n_samples=n_negative_samples,
            )
            for neg_dst in neg_dsts:
                z_full_neg = _predict_raw_pair_score(
                    self.model,
                    src=src,
                    dst=neg_dst,
                    ts=ts,
                    preserve_eidx=preserve_full,
                    result_as_logit=result_as_logit,
                )
                if z_full_neg is None or not np.isfinite(z_full_neg):
                    continue
                valid_neg_dsts.append(int(neg_dst))
                z_full_negs.append(float(z_full_neg))
                p_full_neg = _to_probability(float(z_full_neg), score_is_logit=result_as_logit)
                y_full_negs.append(int(float(p_full_neg) > float(threshold)))

        matches: list[float] = []
        ratio_aps: list[float] = []
        ratio_auc: list[float] = []
        ratio_prob: list[float] = []
        ratio_logit: list[float] = []
        values: dict[str, float] = {}

        for level in levels:
            ratio = float(level)
            # TempME rule: topk = min(max(ceil(ratio * num_edge), 1), num_edge)
            k = int(np.ceil(ratio * float(num_edge)))
            if ratio > 0.0:
                k = max(k, 1)
            k = max(0, min(k, int(num_edge), int(n_candidates)))

            selected = [candidate[i] for i in order[:k]]
            pred_keep_pos = self.model.predict_proba_with_mask(
                context.subgraph,
                context.target,
                edge_mask=_edge_mask(candidate, selected, mode="keep"),
            )
            z_keep_pos = _score(pred_keep_pos)
            p_keep_pos = _to_probability(float(z_keep_pos), score_is_logit=result_as_logit)

            y_true = [int(y_full_pos)]
            y_pred_labels = [int(float(p_keep_pos) > float(threshold))]
            y_prob = [float(p_keep_pos)]
            neg_prob_deltas: list[float] = []
            neg_logit_deltas: list[float] = []
            if z_full_negs and src is not None and ts is not None and valid_neg_dsts:
                preserve = _preserve_from_selected(
                    context,
                    result,
                    candidate,
                    selected,
                    mode="keep",
                )
                for neg_dst, z_full_neg, y_full_neg in zip(valid_neg_dsts, z_full_negs, y_full_negs):
                    z_keep_neg = _predict_raw_pair_score(
                        self.model,
                        src=src,
                        dst=neg_dst,
                        ts=ts,
                        preserve_eidx=preserve,
                        result_as_logit=result_as_logit,
                    )
                    if z_keep_neg is None or not np.isfinite(z_keep_neg):
                        continue
                    p_full_neg = _to_probability(float(z_full_neg), score_is_logit=result_as_logit)
                    p_keep_neg = _to_probability(float(z_keep_neg), score_is_logit=result_as_logit)
                    y_true.append(int(y_full_neg))
                    y_pred_labels.append(int(float(p_keep_neg) > float(threshold)))
                    y_prob.append(float(p_keep_neg))
                    neg_prob_deltas.append(float(p_full_neg - p_keep_neg))
                    neg_logit_deltas.append(float(z_full_neg - float(z_keep_neg)))

            match = float(np.mean(np.asarray(y_pred_labels, dtype=int) == np.asarray(y_true, dtype=int)))

            values[f"acc@s={ratio:g}"] = float(match)
            matches.append(float(match))

            # Keep per-threshold probability/logit deltas for diagnostics only.
            pos_prob_delta = float(p_keep_pos - p_full_pos)
            pos_logit_delta = float(float(z_keep_pos) - float(z_full_pos))
            if neg_prob_deltas:
                fid_prob = float(np.mean([pos_prob_delta, *neg_prob_deltas]))
            else:
                fid_prob = pos_prob_delta
            if neg_logit_deltas:
                fid_logit = float(np.mean([pos_logit_delta, *neg_logit_deltas]))
            else:
                fid_logit = pos_logit_delta
            values[f"fid_prob@s={ratio:g}"] = float(fid_prob)
            values[f"fid_logit@s={ratio:g}"] = float(fid_logit)
            ratio_prob.append(float(fid_prob))
            ratio_logit.append(float(fid_logit))

            aps = float("nan")
            auc = float("nan")
            try:
                from sklearn.metrics import average_precision_score, roc_auc_score  # type: ignore

                y_true_arr = np.asarray(y_true, dtype=int)
                y_prob_arr = np.asarray(y_prob, dtype=float)
                if y_true_arr.size > 0:
                    n_pos = int(np.sum(y_true_arr == 1))
                    n_neg = int(y_true_arr.size - n_pos)
                    if n_pos == 0:
                        # Avoid sklearn warning: "No positive class found in y_true".
                        aps = 0.0
                        auc = float("nan")
                    elif n_neg == 0:
                        # Degenerate all-positive case.
                        aps = 1.0
                        auc = float("nan")
                    else:
                        aps = float(average_precision_score(y_true_arr, y_prob_arr))
                        auc = float(roc_auc_score(y_true_arr, y_prob_arr))
            except Exception:
                aps = float("nan")
                auc = float("nan")
            values[f"aps@s={ratio:g}"] = float(aps)
            values[f"auc@s={ratio:g}"] = float(auc)
            ratio_aps.append(float(aps))
            ratio_auc.append(float(auc))

        # Keep raw [0, 1] Ratio ACC by default to mirror TempME threshold_test.
        scale_to_percent = bool(self.config.get("scale_to_percent", False))
        acc_scale = 100.0 if scale_to_percent else 1.0

        if acc_scale != 1.0:
            for level in levels:
                key = f"acc@s={float(level):g}"
                if key in values and np.isfinite(values[key]):
                    values[key] = float(values[key]) * acc_scale

        # TempME "Ratio ACC": arithmetic mean over thresholds (not trapezoidal AUC).
        values["ratio_acc"] = (float(np.mean(matches)) * acc_scale) if matches else float("nan")
        values["ratio_prob"] = float(np.mean(ratio_prob)) if ratio_prob else float("nan")
        values["ratio_logit"] = float(np.mean(ratio_logit)) if ratio_logit else float("nan")
        ratio_aps_valid = [x for x in ratio_aps if np.isfinite(x)]
        ratio_auc_valid = [x for x in ratio_auc if np.isfinite(x)]
        values["ratio_aps"] = float(np.mean(ratio_aps_valid)) if ratio_aps_valid else float("nan")
        values["ratio_auc"] = float(np.mean(ratio_auc_valid)) if ratio_auc_valid else float("nan")

        return MetricResult(
            name=self.name,
            values=values,
            direction=self.direction.value,
            run_id=context.run_id,
            explainer=result.explainer,
            context_fp=result.context_fp,
            extras={
                "levels": [float(x) for x in levels],
                "formula": "mean(acc over levels against full-pred labels y_ori)",
                "topk_rule": "ceil(ratio * num_edge), min 1 for ratio>0",
                "ranking_source": "tempme_threshold_test_explainer_scores",
                "full_scope": "support_edges",
                "scale_to_percent": bool(scale_to_percent),
                "result_as_logit": result_as_logit,
                "n_negative_samples": int(n_negative_samples),
                "num_edge": int(num_edge),
                "num_edge_reference": int(num_edge_ref),
                "cap_num_edge_to_candidates": bool(cap_num_edge),
                "full_pos_score": float(z_full_pos),
                "full_pos_label": int(y_full_pos),
                "full_neg_score_mean": float(np.mean(z_full_negs)) if z_full_negs else float("nan"),
                "full_neg_label_mean": float(np.mean(y_full_negs)) if y_full_negs else float("nan"),
                "uses_negative_pair": bool(z_full_negs),
            },
        )

def build_fidelity_drop(config: Mapping[str, Any] | None = None):
    return FidelityDropMetric(
        name="fidelity_drop",
        direction=MetricDirection.HIGHER_IS_BETTER,
        config=dict(config or {}),
    )


def build_fidelity_minus(config: Mapping[str, Any] | None = None):
    return FidelityDropMetric(
        name="fidelity_minus",
        direction=MetricDirection.HIGHER_IS_BETTER,
        config=dict(config or {}),
    )


def build_fidelity_keep(config: Mapping[str, Any] | None = None):
    return FidelityKeepMetric(
        name="fidelity_keep",
        direction=MetricDirection.HIGHER_IS_BETTER,
        config=dict(config or {}),
    )


def build_fidelity_keep_graph(config: Mapping[str, Any] | None = None):
    return FidelityKeepGraphMetric(
        name="fidelity_keep_graph",
        direction=MetricDirection.HIGHER_IS_BETTER,
        config=dict(config or {}),
    )


def build_graph_sparsity(config: Mapping[str, Any] | None = None):
    return GraphSparsityMetric(
        name="graph_sparsity",
        direction=MetricDirection.LOWER_IS_BETTER,
        config=dict(config or {}),
    )


def build_fidelity_best(config: Mapping[str, Any] | None = None):
    return FidelityBestMetric(
        name="fidelity_best",
        direction=MetricDirection.HIGHER_IS_BETTER,
        config=dict(config or {}),
    )


def build_fidelity_tempme(config: Mapping[str, Any] | None = None):
    return FidelityTempMeMetric(
        name="fidelity_tempme",
        direction=MetricDirection.HIGHER_IS_BETTER,
        config=dict(config or {}),
    )


def build_best_fid(config: Mapping[str, Any] | None = None):
    return BestFidMetric(
        name="best_fid",
        direction=MetricDirection.HIGHER_IS_BETTER,
        config=dict(config or {}),
    )


def build_acc_auc(config: Mapping[str, Any] | None = None):
    return AccAucMetric(
        name="acc_auc",
        direction=MetricDirection.HIGHER_IS_BETTER,
        config=dict(config or {}),
    )


def build_tempme_acc_auc(config: Mapping[str, Any] | None = None):
    return TempMeAccAucMetric(
        name="tempme_acc_auc",
        direction=MetricDirection.HIGHER_IS_BETTER,
        config=dict(config or {}),
    )


def build_prediction_profile(config: Mapping[str, Any] | None = None):
    return PredictionProfileMetric(
        name="prediction_profile",
        direction=MetricDirection.HIGHER_IS_BETTER,
        config=dict(config or {}),
    )


def build_monotonicity(config: Mapping[str, Any] | None = None):
    return MonotonicityMetric(
        name="monotonicity",
        direction=MetricDirection.HIGHER_IS_BETTER,
        config=dict(config or {}),
    )


def build_seed_stability(config: Mapping[str, Any] | None = None):
    return SeedStabilityMetric(
        name="seed_stability",
        direction=MetricDirection.HIGHER_IS_BETTER,
        config=dict(config or {}),
    )


def build_perturbation_robustness(config: Mapping[str, Any] | None = None):
    return PerturbationRobustnessMetric(
        name="perturbation_robustness",
        direction=MetricDirection.HIGHER_IS_BETTER,
        config=dict(config or {}),
    )


def build_temgx_fidelity_minus(config: Mapping[str, Any] | None = None):
    return TemGXFidelityMetric(name="temgx_fidelity_minus", mode="minus", config=config)


def build_temgx_fidelity_plus(config: Mapping[str, Any] | None = None):
    return TemGXFidelityMetric(name="temgx_fidelity_plus", mode="plus", config=config)


def build_temgx_fidelity_minus_logit(config: Mapping[str, Any] | None = None):
    cfg = dict(config or {})
    cfg.setdefault("result_as_logit", True)
    return TemGXFidelityMetric(name="temgx_fidelity_minus_logit", mode="minus", config=cfg)


def build_temgx_fidelity_plus_logit(config: Mapping[str, Any] | None = None):
    cfg = dict(config or {})
    cfg.setdefault("result_as_logit", True)
    return TemGXFidelityMetric(name="temgx_fidelity_plus_logit", mode="plus", config=cfg)


def build_temgx_sparsity(config: Mapping[str, Any] | None = None):
    return TemGXSparsityMetric(config)


def build_tgnn_aufsc(config: Mapping[str, Any] | None = None):
    return TGNNAufscMetric(
        name="tgnn_aufsc",
        direction=MetricDirection.HIGHER_IS_BETTER,
        config=dict(config or {}),
    )


def build_temgx_aufsc(config: Mapping[str, Any] | None = None):
    return TemGXAufscMetric(
        name="temgx_aufsc",
        direction=MetricDirection.HIGHER_IS_BETTER,
        config=dict(config or {}),
    )


def build_singular_value(config: Mapping[str, Any] | None = None):
    return SingularValueMetric(
        name="singular_value",
        direction=MetricDirection.HIGHER_IS_BETTER,
        config=dict(config or {}),
    )


def _register_if_missing(name: str, factory: Callable[[Mapping[str, Any] | None], BaseMetric]) -> None:
    if name not in METRICS.keys():
        register_metric(name)(factory)


_register_if_missing("fidelity_drop", build_fidelity_drop)
_register_if_missing("fidelity_minus", build_fidelity_minus)
_register_if_missing("fidelity_keep", build_fidelity_keep)
_register_if_missing("fidelity_keep_graph", build_fidelity_keep_graph)
_register_if_missing("keep_graph_fidelity", build_fidelity_keep_graph)
_register_if_missing("fidelity_best", build_fidelity_best)
_register_if_missing("fidelity_tempme", build_fidelity_tempme)
_register_if_missing("graph_sparsity", build_graph_sparsity)
_register_if_missing("sparsity_graph", build_graph_sparsity)
_register_if_missing("best_fid", build_best_fid)
_register_if_missing("tgnn_best_fid", build_best_fid)
_register_if_missing("acc_auc", build_acc_auc)
_register_if_missing("tempme_acc_auc", build_tempme_acc_auc)
_register_if_missing("prediction_profile", build_prediction_profile)
_register_if_missing("monotonicity", build_monotonicity)
_register_if_missing("seed_stability", build_seed_stability)
_register_if_missing("stability_seed", build_seed_stability)
_register_if_missing("perturbation_robustness", build_perturbation_robustness)
_register_if_missing("robustness_perturbation", build_perturbation_robustness)
_register_if_missing("temgx_fidelity_minus", build_temgx_fidelity_minus)
_register_if_missing("temgx_fidelity_plus", build_temgx_fidelity_plus)
_register_if_missing("temgx_fidelity_minus_logit", build_temgx_fidelity_minus_logit)
_register_if_missing("temgx_fidelity_plus_logit", build_temgx_fidelity_plus_logit)
_register_if_missing("temgx_sparsity", build_temgx_sparsity)
_register_if_missing("tgnn_aufsc", build_tgnn_aufsc)
_register_if_missing("aufsc_tgnn", build_tgnn_aufsc)
_register_if_missing("temgx_aufsc", build_temgx_aufsc)
_register_if_missing("aufsc_temgx", build_temgx_aufsc)
_register_if_missing("singular_value", build_singular_value)


__all__ = [
    "FidelityDropMetric",
    "FidelityKeepMetric",
    "FidelityKeepGraphMetric",
    "FidelityBestMetric",
    "FidelityTempMeMetric",
    "BestFidMetric",
    "TGNNAufscMetric",
    "AccAucMetric",
    "TempMeAccAucMetric",
    "PredictionProfileMetric",
    "MonotonicityMetric",
    "SeedStabilityMetric",
    "PerturbationRobustnessMetric",
    "GraphSparsityMetric",
    "TemGXFidelityMetric",
    "TemGXSparsityMetric",
    "TemGXAufscMetric",
    "SingularValueMetric",
    "build_fidelity_drop",
    "build_fidelity_minus",
    "build_fidelity_keep",
    "build_fidelity_keep_graph",
    "build_fidelity_best",
    "build_fidelity_tempme",
    "build_graph_sparsity",
    "build_best_fid",
    "build_acc_auc",
    "build_tempme_acc_auc",
    "build_prediction_profile",
    "build_monotonicity",
    "build_seed_stability",
    "build_perturbation_robustness",
    "build_temgx_fidelity_minus",
    "build_temgx_fidelity_plus",
    "build_temgx_fidelity_minus_logit",
    "build_temgx_fidelity_plus_logit",
    "build_temgx_sparsity",
    "build_tgnn_aufsc",
    "build_temgx_aufsc",
    "build_singular_value",
]

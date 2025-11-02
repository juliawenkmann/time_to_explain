from __future__ import annotations
import argparse, json
from typing import Any, Dict, List

from .core.registry import EXPLAINERS, METRICS
from .core.runner import EvaluationRunner, EvalConfig
from .core.subgraph import KHopTemporalExtractor
from .core.types import ModelProtocol, Subgraph

# A tiny model so "smoke" can run out-of-the-box.
class _NullModel(ModelProtocol):
    def predict_proba(self, subgraph: Subgraph, target: Dict[str, Any]) -> Any:
        return [0.4, 0.6]
    def predict_proba_with_mask(self, subgraph: Subgraph, target: Dict[str, Any],
                                edge_mask=None, node_mask=None) -> Any:
        return [0.45, 0.55]

def main():
    p = argparse.ArgumentParser(prog="time_to_explain")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="List registered explainers/metrics")

    runp = sub.add_parser("run", help="Run evaluation from a JSON config")
    runp.add_argument("config", type=str, help="Path to config JSON")

    smokep = sub.add_parser("smoke", help="Quick smoke test with dummy model/explainer")
    smokep.add_argument("--anchors", type=str, default='[{"target_kind":"edge","u":1,"i":2,"ts":1.0}]')

    args = p.parse_args()
    if args.cmd == "list":
        print("Explainers:", EXPLAINERS.keys())
        print("Metrics:", METRICS.keys())
        return

    if args.cmd == "run":
        cfg = json.loads(open(args.config).read())
        model = _NullModel()  # Replace with your ModelProtocol adapter
        dataset: Any = {}     # Replace with your dataset handle
        extractor = KHopTemporalExtractor()
        explainers = []
        for name in cfg.get("explainers", []):
            impl = EXPLAINERS.get(name)
            if callable(impl):
                explainers.append(impl())  # factory or class
            else:
                explainers.append(impl)
        runner = EvaluationRunner(
            model=model, dataset=dataset, extractor=extractor, explainers=explainers,
            config=EvalConfig(out_dir=cfg.get("out_dir","runs"),
                              seed=int(cfg.get("seed",42)),
                              metrics=cfg.get("metrics", ["sparsity"]))
        )
        anchors: List[Dict[str, Any]] = cfg["anchors"]
        out = runner.run(anchors,
                         k_hop=int(cfg.get("k_hop",2)),
                         num_neighbors=int(cfg.get("num_neighbors",50)),
                         window=tuple(cfg.get("window")) if cfg.get("window") else None,
                         run_id=cfg.get("run_id"))
        print(json.dumps(out, indent=2))
        return

    if args.cmd == "smoke":
        anchors = json.loads(args.anchors)
        model = _NullModel()
        dataset: Any = {}
        extractor = KHopTemporalExtractor()
        # Minimal explainer to prove plumbing
        from .core.types import BaseExplainer, ExplanationResult, ExplanationContext
        class _ToyExplainer(BaseExplainer):
            def __init__(self): super().__init__(name="toy")
            def explain(self, context: ExplanationContext) -> ExplanationResult:
                ecount = len(context.subgraph.edge_index) if context.subgraph else 1
                return ExplanationResult(run_id=context.run_id, explainer=self.name,
                                         context_fp=context.fingerprint(),
                                         importance_edges=[1.0]*ecount)
        runner = EvaluationRunner(model=model, dataset=dataset, extractor=extractor,
                                  explainers=[_ToyExplainer()], config=EvalConfig())
        out = runner.run(anchors)
        print(json.dumps(out, indent=2))
        return

if __name__ == "__main__":
    main()

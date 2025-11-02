from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple
import json, time, random, os, csv

from .types import ExplanationContext, BaseExplainer, ModelProtocol, SubgraphExtractorProtocol
from .registry import METRICS

def set_global_seed(seed: int) -> None:
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)          # type: ignore
        torch.cuda.manual_seed_all(seed) # type: ignore
        torch.backends.cudnn.deterministic = True  # type: ignore
        torch.backends.cudnn.benchmark = False     # type: ignore
    except Exception:
        pass
    random.seed(seed)

@dataclass
class EvalConfig:
    out_dir: str = "runs"
    seed: int = 42
    metrics: List[str] = field(default_factory=lambda: ["sparsity"])
    save_jsonl: bool = True
    save_csv: bool = True

class EvaluationRunner:
    def __init__(self, *, model: ModelProtocol, dataset: Any, extractor: SubgraphExtractorProtocol,
                 explainers: Sequence[BaseExplainer], config: Optional[EvalConfig] = None) -> None:
        self.model = model
        self.dataset = dataset
        self.extractor = extractor
        self.explainers = list(explainers)
        self.config = config or EvalConfig()
        set_global_seed(self.config.seed)

    def _ensure_out(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)

    def run(self, anchors: Sequence[Dict[str, Any]], *, k_hop: int = 2,
            num_neighbors: int = 50, window: Optional[Tuple[float,float]] = None,
            run_id: Optional[str] = None) -> Dict[str, Any]:
        run_id = run_id or str(int(time.time()))
        out_dir = os.path.join(self.config.out_dir, run_id)
        self._ensure_out(out_dir)

        for e in self.explainers:
            e.prepare(model=self.model, dataset=self.dataset)

        jsonl_path = os.path.join(out_dir, "results.jsonl")
        csv_path = os.path.join(out_dir, "metrics.csv")
        jsonl_f = open(jsonl_path, "w") if self.config.save_jsonl else None
        csv_f = open(csv_path, "w", newline="") if self.config.save_csv else None
        csv_writer = None

        try:
            for idx, anchor in enumerate(anchors):
                ctx = ExplanationContext(
                    run_id=run_id, target_kind=anchor.get("target_kind","edge"),
                    target=anchor, window=window, k_hop=k_hop, num_neighbors=num_neighbors
                )
                subg = self.extractor.extract(self.dataset, anchor, k_hop=k_hop,
                                              num_neighbors=num_neighbors, window=window)
                ctx.subgraph = subg

                for explainer in self.explainers:
                    t0 = time.perf_counter()
                    res = explainer.explain(ctx)
                    res.elapsed_sec = time.perf_counter() - t0
                    res.context_fp = ctx.fingerprint()

                    row = {"run_id": run_id, "anchor_idx": idx, "explainer": res.explainer,
                           "elapsed_sec": res.elapsed_sec}
                    for m in self.config.metrics:
                        metric_fn = METRICS.get(m)
                        mvals = metric_fn(ctx, res, model=self.model)
                        row.update({f"{m}.{k}": v for k,v in mvals.items()})

                    if jsonl_f:
                        jsonl_f.write(json.dumps({
                            "context": {"target": ctx.target, "target_kind": ctx.target_kind,
                                        "window": ctx.window, "k_hop": k_hop, "num_neighbors": num_neighbors},
                            "result": {
                                "explainer": res.explainer, "elapsed_sec": res.elapsed_sec,
                                "importance_edges": res.importance_edges,
                                "importance_nodes": res.importance_nodes,
                                "importance_time": res.importance_time,
                                "extras": res.extras
                            },
                            "metrics": {k:v for k,v in row.items() if k not in ("run_id","anchor_idx","explainer","elapsed_sec")}
                        }) + "\n")
                    if csv_f:
                        if csv_writer is None:
                            csv_writer = csv.DictWriter(csv_f, fieldnames=list(row.keys()))
                            csv_writer.writeheader()
                        csv_writer.writerow(row)

        finally:
            if jsonl_f: jsonl_f.close()
            if csv_f: csv_f.close()

        return {"out_dir": out_dir, "jsonl": jsonl_path if self.config.save_jsonl else None,
                "csv": csv_path if self.config.save_csv else None}

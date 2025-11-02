from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from pathlib import Path

import torch
from dgl.data.utils import load_graphs

from submodules.explainer.htgexplainer.utils.data import (
    load_COVID_data,
    load_MAG_data,
    load_ML_data,
)

from .config import ARTIFACTS_DIR, HTGNN_BASE, HTGNNExperiment


LOG = logging.getLogger(__name__)


@contextmanager
def _working_directory(path: Path):
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def run_htgnn_experiment(exp: HTGNNExperiment) -> None:
    device = _resolve_device(exp.device)
    output_dir = ARTIFACTS_DIR / "htgnn" / exp.name
    output_dir.mkdir(parents=True, exist_ok=True)

    with _working_directory(HTGNN_BASE):
        if exp.dataset == "mag":
            from submodules.explainer.htgexplainer.model.HTGNN import HTGNN, LinkPredictor
            from submodules.explainer.htgexplainer.model.HTGExplainer_mag import HTGExplainer as MAGExplainer

            glist, _ = load_graphs("data/ogbn_graphs.bin")
            train_feats, train_labels, val_feats, val_labels, test_feats, test_labels = load_MAG_data(
                glist, exp.time_window, device
            )
            base_graph = train_feats[0]
            htgnn = HTGNN(
                graph=base_graph,
                n_inp=128,
                n_hid=32,
                n_layers=2,
                n_heads=1,
                time_window=exp.time_window,
                norm=True,
                device=device,
            )
            predictor = LinkPredictor(n_inp=32, n_classes=1)
            model = torch.nn.Sequential(htgnn, predictor).to(device)
            model.load_state_dict(torch.load(exp.checkpoint, map_location=device))

            explainer = MAGExplainer(
                model_to_explain=model,
                G_train=train_feats,
                G_train_label=train_labels,
                G_val=val_feats,
                G_val_label=val_labels,
                G_test=test_feats,
                G_test_label=test_labels,
                time_win=exp.time_window,
                node_emb=32,
                device=device,
                epochs=int(exp.explainer.get("epochs", 50)),
                lr=float(exp.explainer.get("learning_rate", 1e-3)),
                warmup_epoch=int(exp.explainer.get("warmup_epoch", 5)),
                es_epoch=int(exp.explainer.get("es_epoch", 10)),
                batch_size=int(exp.explainer.get("batch_size", 4)),
                khop=1,
                te="cos",
                he="learnable",
                test_only=True,
            )
            LOG.info("Running HTGExplainer (MAG)")
            explainer.explain()
            torch.save(model.state_dict(), output_dir / "model_state.pt")

        elif exp.dataset == "ml":
            from submodules.explainer.htgexplainer.model.HTGNN_ml import HTGNN, LinkPredictor_ml
            from submodules.explainer.htgexplainer.model.HTGExplainer_ml import HTGExplainer as MLExplainer

            train_feats, train_labels, val_feats, val_labels, test_feats, test_labels = load_ML_data(device)
            base_graph = train_feats[0]
            htgnn = HTGNN(
                graph=base_graph,
                n_inp=64,
                n_hid=32,
                n_layers=2,
                n_heads=1,
                time_window=exp.time_window,
                norm=True,
                device=device,
            )
            predictor = LinkPredictor_ml(n_inp=32, n_classes=1)
            model = torch.nn.Sequential(htgnn, predictor).to(device)
            model.load_state_dict(torch.load(exp.checkpoint, map_location=device))

            explainer = MLExplainer(
                model_to_explain=model,
                G_train=train_feats,
                G_train_label=train_labels,
                G_val=val_feats,
                G_val_label=val_labels,
                G_test=test_feats,
                G_test_label=test_labels,
                time_win=exp.time_window,
                node_emb=32,
                device=device,
                epochs=int(exp.explainer.get("epochs", 50)),
                lr=float(exp.explainer.get("learning_rate", 1e-3)),
                warmup_epoch=int(exp.explainer.get("warmup_epoch", 5)),
                es_epoch=int(exp.explainer.get("es_epoch", 10)),
                batch_size=int(exp.explainer.get("batch_size", 8)),
                khop=1,
                te="cos",
                he="learnable",
                test_only=True,
            )
            LOG.info("Running HTGExplainer (Movielens)")
            explainer.explain()
            torch.save(model.state_dict(), output_dir / "model_state.pt")

        elif exp.dataset == "covid":
            from submodules.explainer.htgexplainer.model.HTGNN import HTGNN, NodePredictor
            from submodules.explainer.htgexplainer.model.HTGExplainer_covid import HTGExplainer as COVIDExplainer

            glist, _ = load_graphs("data/COVID19/COVID_dynamic_graphs.bin")
            train_feats, train_labels, val_feats, val_labels, test_feats, test_labels = load_COVID_data(glist, exp.time_window)
            base_graph = train_feats[0]
            htgnn = HTGNN(
                graph=base_graph,
                n_inp=1,
                n_hid=8,
                n_layers=2,
                n_heads=1,
                time_window=exp.time_window,
                norm=False,
                device=device,
            )
            predictor = NodePredictor(n_inp=8, n_classes=1)
            model = torch.nn.Sequential(htgnn, predictor).to(device)
            model.load_state_dict(torch.load(exp.checkpoint, map_location=device))

            explainer = COVIDExplainer(
                model_to_explain=model,
                G_train=train_feats,
                G_train_label=train_labels,
                G_val=val_feats,
                G_val_label=val_labels,
                G_test=test_feats,
                G_test_label=test_labels,
                time_win=exp.time_window,
                node_emb=8,
                device=device,
                epochs=int(exp.explainer.get("epochs", 50)),
                lr=float(exp.explainer.get("learning_rate", 5e-4)),
                warmup_epoch=int(exp.explainer.get("warmup_epoch", 5)),
                es_epoch=int(exp.explainer.get("es_epoch", 10)),
                batch_size=int(exp.explainer.get("batch_size", 8)),
                khop=1,
                te="cos",
                he="learnable",
                test_only=True,
            )
            LOG.info("Running HTGExplainer (COVID-19)")
            explainer.explain()
            torch.save(model.state_dict(), output_dir / "model_state.pt")

        else:
            raise ValueError(f"Unsupported HTGNN dataset: {exp.dataset}")

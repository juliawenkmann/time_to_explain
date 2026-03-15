from __future__ import annotations

import pandas as pd

from ...data.validate import verify_dataframe_unify as _verify_dataframe_unify


_NON_BIPARTITE_DATASETS = {
    "ucim",
    "ucim_motif",
    "uci",
    "uci_messages",
    "uci-messages",
    "ucimessages",
    "uci_motif",
    "uci-motif",
    "nicolaus",
    "erdos_small",
    "hawkes_small",
    "triadic_closure",
    "stick_figure",
    "sticky_hips",
}


def _infer_bipartite_events(events: pd.DataFrame, dataset_name: str) -> bool:
    dataset_key = str(dataset_name or "").strip().lower()
    if dataset_key in _NON_BIPARTITE_DATASETS:
        return False

    if not {"u", "i"}.issubset(events.columns):
        return True

    u_min, u_max = int(events["u"].min()), int(events["u"].max())
    i_min, i_max = int(events["i"].min()), int(events["i"].max())
    return bool(i_min > u_max or u_min > i_max)


def install_tgnn_dataframe_compat(*, events: pd.DataFrame, dataset_name: str) -> bool:
    """Patch vendored TGNNExplainer dataframe validation for the active dataset.

    The original vendored code assumes unified bipartite ids for every dataset.
    Our benchmark also includes non-bipartite datasets such as UCIM. Those
    datasets keep a shared node id space, so the vendor assertion is too strict.
    """

    bipartite = _infer_bipartite_events(events, dataset_name)

    import tgnnexplainer.xgraph.dataset.tg_dataset as tg_dataset_mod
    import tgnnexplainer.xgraph.dataset.utils_dataset as utils_dataset_mod

    def _verify(df: pd.DataFrame) -> None:
        _verify_dataframe_unify(df, bipartite=bipartite)

    tg_dataset_mod.verify_dataframe_unify = _verify
    utils_dataset_mod.verify_dataframe_unify = _verify
    return bipartite

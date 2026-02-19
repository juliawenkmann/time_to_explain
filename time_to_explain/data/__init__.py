from time_to_explain.utils.constants import ensure_repo_importable

ensure_repo_importable()

from .workflows import (
    dataset_config_dir,
    ensure_ml_format,
    is_synthetic_dataset,
    list_processed_datasets,
    load_processed_dataset_safe,
    prepare_dataset_bundle,
    read_dataset_config,
)
from .explain_index import generate_explain_index, load_explain_idx
from .tgnn_prepare import prepare_tgnn_dataset, tgnn_dataset_paths
#from .tempme_preprocess import TempMEPreprocessConfig, prepare_tempme_dataset

__all__ = [
    "dataset_config_dir",
    "read_dataset_config",
    "list_processed_datasets",
    "is_synthetic_dataset",
    "ensure_ml_format",
    "load_processed_dataset_safe",
    "prepare_dataset_bundle",
    "load_explain_idx",
    "generate_explain_index",
    "prepare_tgnn_dataset",
    "tgnn_dataset_paths",
    #"TempMEPreprocessConfig",
    #"prepare_tempme_dataset",
    "load_dataset",
]



from .io import load_processed_dataset


def load_dataset(dataset_name: str, **prepare_kwargs):
    """Convenience helper: prepare (download/generate) then load the processed DatasetBundle.

    Returns:
        (bundle, paths, summary)
    """
    paths, summary = prepare_tgnn_dataset(dataset_name, **prepare_kwargs)
    bundle = load_processed_dataset(paths.ml_csv)
    return bundle, paths, summary

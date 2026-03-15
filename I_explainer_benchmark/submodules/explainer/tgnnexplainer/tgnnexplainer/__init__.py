from pathlib import Path
import os

ROOT_DIR = Path(__file__).resolve().parent


def _resolve_repo_root() -> Path:
    for key in ("TIME_TO_EXPLAIN_ROOT", "PROJECT_ROOT", "REPO_ROOT"):
        val = os.environ.get(key)
        if val:
            return Path(val).expanduser().resolve()
    for parent in ROOT_DIR.parents:
        if (parent / "resources").is_dir():
            return parent
    return ROOT_DIR


REPO_ROOT = _resolve_repo_root()
RESOURCES_DIR = REPO_ROOT / "resources"

if RESOURCES_DIR.is_dir():
    DATASETS_DIR = RESOURCES_DIR / "datasets"
    RAW_DATA_DIR = DATASETS_DIR / "raw"
    PROCESSED_DATA_DIR = DATASETS_DIR / "processed"
    EXPLAIN_INDEX_DIR = DATASETS_DIR / "explain_index"
else:
    DATASETS_DIR = ROOT_DIR / "xgraph" / "dataset"
    RAW_DATA_DIR = DATASETS_DIR / "data"
    PROCESSED_DATA_DIR = ROOT_DIR / "xgraph" / "models" / "ext" / "tgat" / "processed"
    EXPLAIN_INDEX_DIR = DATASETS_DIR / "explain_index"

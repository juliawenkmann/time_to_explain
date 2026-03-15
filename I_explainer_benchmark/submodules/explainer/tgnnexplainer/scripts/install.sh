#!/bin/bash

# if ls $pwd does not contain tgnnexplainer, and benchmarks, then exit
if [ ! -d "tgnnexplainer" ] || [ ! -d "benchmarks" ]; then
    echo "Please run this script from the root of the repository"
    exit 1
fi

export ROOT="${ROOT:-$PWD}"
export PYTHONPATH="$ROOT:$PYTHONPATH:."

find_resources_root() {
    local dir="$1"
    while [ "$dir" != "/" ]; do
        if [ -d "$dir/resources" ]; then
            echo "$dir"
            return 0
        fi
        dir="$(dirname "$dir")"
    done
    return 1
}

RES_ROOT="${TIME_TO_EXPLAIN_ROOT:-$(find_resources_root "$ROOT")}"
if [ -n "$RES_ROOT" ] && [ -d "$RES_ROOT/resources" ]; then
    RESOURCES_DIR="$RES_ROOT/resources"
    mkdir -p "$RESOURCES_DIR/datasets/raw"
    mkdir -p "$RESOURCES_DIR/datasets/processed"
    mkdir -p "$RESOURCES_DIR/datasets/explain_index"
    mkdir -p "$RESOURCES_DIR/explainer/tgnnexplainer"
    mkdir -p "$RESOURCES_DIR/results/tgnnexplainer/mcts_saved_dir"
    mkdir -p "$RESOURCES_DIR/models/checkpoints"
else
    RESOURCES_DIR=""
fi

echo "Creating additional directories and installing the package"
# create empty directories
if [ -n "$RESOURCES_DIR" ]; then
    mkdir -p "$RESOURCES_DIR/datasets/raw"
    mkdir -p "$RESOURCES_DIR/datasets/processed"
    mkdir -p "$RESOURCES_DIR/datasets/explain_index"
    mkdir -p "$RESOURCES_DIR/explainer/tgnnexplainer"
    mkdir -p "$RESOURCES_DIR/results/tgnnexplainer/mcts_saved_dir"
    mkdir -p "$RESOURCES_DIR/models/checkpoints"
else
    mkdir -p "$ROOT/tgnnexplainer/xgraph/dataset/data"
    mkdir -p "$ROOT/tgnnexplainer/xgraph/models/ext/tgat/processed"
    mkdir -p "$ROOT/tgnnexplainer/xgraph/dataset/explain_index"
    mkdir -p "$ROOT/tgnnexplainer/xgraph/explainer_ckpts"
    mkdir -p "$ROOT/tgnnexplainer/xgraph/saved_mcts_results"
    mkdir -p "$ROOT/tgnnexplainer/xgraph/models/checkpoints"
fi
echo
echo "Directories created"
echo

cd "$ROOT"
# create environment, if it doesn't exist
if [ ! -d "$ROOT/.venv" ]; then
    python3 -m venv "$ROOT/.venv"
    echo "New virtual environment created"
fi
echo
echo "Activating virtual environment and installing the package"
echo
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install "$ROOT"

echo
echo "Installation complete"

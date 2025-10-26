#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# SCRIPT_DIR = .../time_to_explain/time_to_explain/models
REPO_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
# REPO_ROOT = .../time_to_explain  (parent of the package folder)

source "$SCRIPT_DIR/common.bash"

train_tgn() {
  MODEL_PATH="$PARENT_DIR/resources/models/$2"
  if [ ! -d "$MODEL_PATH" ]; then
    mkdir -p "$MODEL_PATH"
    echo "Created new directory for the final model and model checkpoints for the $2 dataset at $MODEL_PATH"
  fi

  # ensure Python sees the package root
  cd "$REPO_ROOT"
  export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

  if [ "${3:-}" = "--bipartite" ]; then
    echo "Training $1 model for the $2 dataset (bipartite)..."
    python -m time_to_explain.models.train_tgnn \
      -d "$PROCESSED_DATA_DIR/$2" \
      --bipartite \
      --cuda \
      --model_path "$MODEL_PATH/" \
      -e 30 \
      --type "$1"
  else
    echo "Training $1 model for the $2 dataset..."
    python -m time_to_explain.models.train_tgnn \
      -d "$PROCESSED_DATA_DIR/$2" \
      --cuda \
      --model_path "$MODEL_PATH/" \
      -e 30 \
      --type "$1"
  fi
}

show_help() {
  echo -e "
Train TGN Model script

Usage: bash $SCRIPT_DIR/train_tgnn_model.bash ${RED}MODEL-TYPE${NC} ${RED}DATASET-NAME${NC} ${RED}--bipartite${NC}
...
"
  exit 1
}

if [ $# -eq 0 ]; then
  show_help
else
  test_exists "$1" "${MODEL_TYPES[@]}"
  test_exists "$2" "${DATASET_NAMES[@]}"
  train_tgn "$1" "$2" "${3:-}"
fi

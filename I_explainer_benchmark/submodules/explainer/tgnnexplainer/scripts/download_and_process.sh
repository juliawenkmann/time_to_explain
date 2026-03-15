#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"

# Ensure local package imports resolve when running from nested paths.
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

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
    DATA_RAW="$RES_ROOT/resources/datasets/raw"
    DATA_PROCESSED="$RES_ROOT/resources/datasets/processed"
    DATA_IDX="$RES_ROOT/resources/datasets/explain_index"
else
    DATA_RAW="$ROOT/tgnnexplainer/xgraph/dataset/data"
    DATA_PROCESSED="$ROOT/tgnnexplainer/xgraph/models/ext/tgat/processed"
    DATA_IDX="$ROOT/tgnnexplainer/xgraph/dataset/explain_index"
fi

mkdir -p "$DATA_RAW" "$DATA_PROCESSED" "$DATA_IDX"

echo $PWD
# Data downloading and preprocessing
curl http://snap.stanford.edu/jodie/reddit.csv > "$DATA_RAW/reddit.csv"
curl http://snap.stanford.edu/jodie/wikipedia.csv > "$DATA_RAW/wikipedia.csv"

# download simulated datasets
# NOTE: the simulated dataset is already pre-generated and pre-processed
#      otherwise, the tick library would be needed which is only stable with Python 3.8
#      see tgnnexplainer/xgraph/dataset/generate_simulate_dataset.py
curl https://m-krastev.github.io/hawkes-sim-datasets/simulate_v1.csv > "$DATA_RAW/simulate_v1.csv"
curl https://m-krastev.github.io/hawkes-sim-datasets/ml_simulate_v1.csv > "$DATA_PROCESSED/ml_simulate_v1.csv"
curl https://m-krastev.github.io/hawkes-sim-datasets/ml_simulate_v1.npy > "$DATA_PROCESSED/ml_simulate_v1.npy"
curl https://m-krastev.github.io/hawkes-sim-datasets/ml_simulate_v1_node.npy > "$DATA_PROCESSED/ml_simulate_v1_node.npy"

curl https://m-krastev.github.io/hawkes-sim-datasets/simulate_v2.csv > "$DATA_RAW/simulate_v2.csv"
curl https://m-krastev.github.io/hawkes-sim-datasets/ml_simulate_v2.csv > "$DATA_PROCESSED/ml_simulate_v2.csv"
curl https://m-krastev.github.io/hawkes-sim-datasets/ml_simulate_v2.npy > "$DATA_PROCESSED/ml_simulate_v2.npy"
curl https://m-krastev.github.io/hawkes-sim-datasets/ml_simulate_v2_node.npy > "$DATA_PROCESSED/ml_simulate_v2_node.npy"

# process the real datasets
cd  "$ROOT/tgnnexplainer/xgraph/models/ext/tgat"
python process.py -d wikipedia
python process.py -d reddit

# generate indices to-be-explained. Seed defaults to 42.
cd "$ROOT/tgnnexplainer/xgraph/dataset"
python tg_dataset.py -d wikipedia -c index
python tg_dataset.py -d reddit -c index
if [ -f "$DATA_RAW/simulate_v1.csv" ]; then
    python tg_dataset.py -d simulate_v1 -c index
fi
if [ -f "$DATA_RAW/simulate_v2.csv" ]; then
    python tg_dataset.py -d simulate_v2 -c index
fi

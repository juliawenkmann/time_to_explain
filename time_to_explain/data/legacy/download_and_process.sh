#!/bin/bash

# Resolve repo root if not provided
ROOT=${ROOT:-$(git rev-parse --show-toplevel)}
TGNN_ROOT="$ROOT/submodules/explainer/tgnnexplainer/tgnnexplainer"

echo "Using ROOT=$ROOT"

# Data downloading and preprocessing
curl http://snap.stanford.edu/jodie/reddit.csv > "$TGNN_ROOT/xgraph/dataset/data/reddit.csv"
curl http://snap.stanford.edu/jodie/wikipedia.csv > "$TGNN_ROOT/xgraph/dataset/data/wikipedia.csv"

# download simulated datasets
# NOTE: the simulated dataset is already pre-generated and pre-processed
#      otherwise, the tick library would be needed which is only stable with Python 3.8
#      see tgnnexplainer/xgraph/dataset/generate_simulate_dataset.py
curl https://m-krastev.github.io/hawkes-sim-datasets/simulate_v1.csv > "$TGNN_ROOT/xgraph/dataset/data/simulate_v1.csv"
curl https://m-krastev.github.io/hawkes-sim-datasets/ml_simulate_v1.csv > "$TGNN_ROOT/xgraph/models/ext/tgat/processed/ml_simulate_v1.csv"
curl https://m-krastev.github.io/hawkes-sim-datasets/ml_simulate_v1.npy > "$TGNN_ROOT/xgraph/models/ext/tgat/processed/ml_simulate_v1.npy"
curl https://m-krastev.github.io/hawkes-sim-datasets/ml_simulate_v1_node.npy > "$TGNN_ROOT/xgraph/models/ext/tgat/processed/ml_simulate_v1_node.npy"

curl https://m-krastev.github.io/hawkes-sim-datasets/simulate_v2.csv > "$TGNN_ROOT/xgraph/dataset/data/simulate_v2.csv"
curl https://m-krastev.github.io/hawkes-sim-datasets/ml_simulate_v2.csv > "$TGNN_ROOT/xgraph/models/ext/tgat/processed/ml_simulate_v2.csv"
curl https://m-krastev.github.io/hawkes-sim-datasets/ml_simulate_v2.npy > "$TGNN_ROOT/xgraph/models/ext/tgat/processed/ml_simulate_v2.npy"
curl https://m-krastev.github.io/hawkes-sim-datasets/ml_simulate_v2_node.npy > "$TGNN_ROOT/xgraph/models/ext/tgat/processed/ml_simulate_v2_node.npy"

# process the real datasets
cd  "$TGNN_ROOT/xgraph/models/ext/tgat"
python process.py -d wikipedia
python process.py -d reddit

# generate indices to-be-explained. Seed defaults to 42.
cd "$TGNN_ROOT/xgraph/dataset"
python tg_dataset.py -d wikipedia -c index
python tg_dataset.py -d reddit -c index
if [ -f "$TGNN_ROOT/xgraph/dataset/data/simulate_v1.csv" ]; then
    python tg_dataset.py -d simulate_v1 -c index
fi
if [ -f "$TGNN_ROOT/xgraph/dataset/data/simulate_v2.csv" ]; then
    python tg_dataset.py -d simulate_v2 -c index
fi

#!/usr/bin/env bash

ROOT="${ROOT:-$PWD}"

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
    MODEL_WEIGHTS="$RES_ROOT/resources/models/checkpoints"
else
    MODEL_WEIGHTS="$ROOT/tgnnexplainer/xgraph/models/checkpoints"
fi

cd $ROOT/tgnnexplainer/xgraph/models/ext/tgn
model=tgn

runs=(0 1 2)

real_epochs=10
sim_epochs=100

sim_datasets=(simulate_v1 simulate_v2)
real_datasets=(wikipedia reddit)
for run in ${runs[@]}
do
    echo "Iteration no. ${run}"

    for dataset in ${sim_datasets[@]}
    do
        echo "dataset: ${dataset}"
        python train_simulate.py -d ${dataset} --prefix tgn-attn --n_runs 1 --n_epoch ${sim_epochs} --n_layer 2 --n_degree 10 --use_memory --memory_update_at_end --gpu 0 \
        --memory_dim 4 # memory_dim should equal to node/edge feature dim

        mkdir -p "$MODEL_WEIGHTS"
        source_path=./saved_checkpoints/tgn-attn-${dataset}-${$(($sim_epochs-1))}.pth
        target_path=$MODEL_WEIGHTS/${model}_${dataset}_best.pth
        cp ${source_path} ${target_path}
        echo ${source_path} ${target_path} 'copied'

    done

    real_datasets=(reddit)
    for dataset in ${real_datasets[@]}
    do
        echo "dataset: ${dataset}\n"
        python train_self_supervised.py -d ${dataset} --prefix tgn-attn --n_runs 1 --n_epoch ${real_epochs} --n_layer 2 --n_degree 10 --use_memory --gpu 0

        # Save last epoch
        mkdir -p "$MODEL_WEIGHTS"
        source_path=./saved_checkpoints/tgn-attn-${dataset}-${$(($real_epochs-1))}.pth
        target_path=$MODEL_WEIGHTS/${model}_${dataset}_best.pth
        cp ${source_path} ${target_path}
        echo ${source_path} ${target_path} 'copied'
    done

done

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

model=tgat

cd  $ROOT/tgnnexplainer/xgraph/models/ext/tgat

# create the savepath directory for the checkpoints
mkdir -p "$MODEL_WEIGHTS"

sim_datasets=(simulate_v1 simulate_v2)
real_datasets=(wikipedia wikipedia)
runs=(0 1 2)

sim_epochs=100
real_epochs=10
for run in ${runs[@]}
do
    echo "Iteration no. ${run}"

    # ========== Train on simulated datasets ==========
    for dataset in ${sim_datasets[@]}
    do
        echo "dataset: ${dataset}"
        python learn_simulate.py -d ${dataset} --bs 256 --n_degree 10 --n_epoch ${sim_epochs} --agg_method attn --attn_mode prod --gpu 1 --n_head 2 --prefix ${dataset}

        source_path=./saved_checkpoints/${dataset}-attn-prod-${$(($sim_epochs-1))}.pth
        target_path=$MODEL_WEIGHTS/${model}_${dataset}_best.pth
        cp ${source_path} ${target_path}

        echo ${source_path} ${target_path} 'copied'
    done

    # ========== Train on real datasets ==========
    for dataset in ${real_datasets[@]}
    do
        echo "dataset: ${dataset}"
        python learn_edge.py -d ${dataset} --bs 512 --n_degree 10 --n_epoch ${real_epochs} --agg_method attn --attn_mode prod --gpu 1 --n_head 2 --prefix ${dataset}

        source_path=./saved_checkpoints/${dataset}-attn-prod-${(($real_epochs-1))}.pth
        target_path=$MODEL_WEIGHTS/${model}_${dataset}_best.pth
        cp ${source_path} ${target_path}
        echo ${source_path} ${target_path} 'copied'

    done
done

#!/bin/bash
TASK_NAME=${1:-LeIsaac-SO101-FoldCloth-BiArm-v0}
DATASET=${2:-./datasets/dataset_cloth.hdf5}

python scripts/environments/state_machine/fold_cloth.py \
    --dataset_file=${DATASET} \
    --task=${TASK_NAME} \
    --num_envs=1 \
    --device=cuda \
    --enable_cameras \
    --record \
    --num_demos=1

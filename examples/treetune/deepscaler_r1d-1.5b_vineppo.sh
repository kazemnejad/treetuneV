# !/usr/bin/env bash

set -e

export VERL_LOGGING_LEVEL=INFO
# export RAY_DEBUG_POST_MORTEM=1

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

export TREETUNEV__EXP_NAME=deepscaler_r1d-1.5b_vineppo
export TREETUNEV__NNODE=1
export TREETUNEV__NUM_GPUS_PER_NODE=$NUM_GPUS

python -m verl.trainer.main_policy_iteration \
    --config-name="polIter_r1d-1.5b_deepscaler_vineppo" \
    $@
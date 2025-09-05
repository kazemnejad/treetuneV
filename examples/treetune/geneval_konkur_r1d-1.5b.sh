# !/usr/bin/env bash

set -e

export VERL_LOGGING_LEVEL=INFO
# export RAY_DEBUG_POST_MORTEM=1

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

export TREETUNEV__EXP_NAME=geneval_math_r1d-1.5b_16k
export TREETUNEV__NNODE=1
export TREETUNEV__NUM_GPUS_PER_NODE=$NUM_GPUS

export WANDB_GROUP=$TREETUNEV__EXP_NAME

python -m verl.trainer.main_geneval \
    --config-name="geneval_math" \
    val_tasks=konkur_only \
    data.val_batch_size=16 \
    ignore_dataset_rollout_n=true \
    actor_rollout_ref.rollout.val_kwargs.n=32 \
    data.max_response_length=16384 \
    data.max_prompt_length=4096 \
    push_to_hub_enabled=true \
    $@
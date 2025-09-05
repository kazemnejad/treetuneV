# !/usr/bin/env bash

set -e

export VERL_LOGGING_LEVEL=INFO
# export RAY_DEBUG_POST_MORTEM=1

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

export TREETUNEV__EXP_NAME=deepscaler_r1d-1.5b_grpo_stable
export TREETUNEV__NNODE=1
export TREETUNEV__NUM_GPUS_PER_NODE=$NUM_GPUS

python -m verl.trainer.main_policy_iteration \
    --config-name="polIter_r1d-1.5b_deepscaler_grpo" \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.actor.clip_ratio_high=0.26 \
    actor_rollout_ref.rollout.calculate_log_probs=true \
    actor_rollout_ref.actor.tis_imp_ratio_cap=2 \
    actor_rollout_ref.actor.policy_loss.loss_mode=vanilla_with_trace_lengths \
    $@
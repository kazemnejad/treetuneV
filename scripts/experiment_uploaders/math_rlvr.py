import os
import re
from pathlib import Path
from typing import Any

import fire

EXPERIMENT_UPLOADER = Path(__file__).parent.parent.absolute() / "upload_experiment.py"

model_to_path = {
    "r1d-1.5b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "qwen3-1.7b": "Qwen/Qwen3-1.7B",
}


def get_experiment_group_name(options: dict[str, Any]) -> str:
    model = options.get("model") or "r1d-1.5b"
    train_tasks = options.get("train_tasks") or "deepscaler"

    base_name_parts = [model, train_tasks]

    if options.get("grpo", True):
        base_name_parts.append("grpo")
    elif options.get("grpo_24k", False):
        base_name_parts.append("grpo_24k")
    elif options.get("vine", False):
        base_name_parts.append("vppo")
    elif options.get("prolong", False):
        base_name_parts.append("prolong")
    elif options.get("exact", False):
        base_name_parts.append("exact")
    elif options.get("gspo", False):
        base_name_parts.append("gspo")

    if options.get("dynamic_sampling", False):
        base_name_parts.append("dyn-smplng")

    if options.get("fsdp_one", False):
        base_name_parts.append("fsdp-one")

    if options.get("tis", False):
        base_name_parts.append("tis")

    if options.get("train_batch_size", None) is not None:
        base_name_parts.append(f"bs{options.get('train_batch_size')}")

    if options.get("clip_ratio_high", None) is not None:
        base_name_parts.append(f"clip{options.get('clip_ratio_high')}")

    if options.get("no_kl_loss", False):
        base_name_parts.append("noKL")

    if options.get("no_weight_decay", False):
        base_name_parts.append("noWDec")

    if options.get("loss_agg_mode", None) is not None:
        base_name_parts.append(f"lssAgg_{options.get('loss_agg_mode')[-10:]}")

    if options.get("fixed_num_optim_steps", None) is not None:
        base_name_parts.append(f"fxdOptStps_{options.get('fixed_num_optim_steps')}")

    if options.get("temperature", None) is not None:
        base_name_parts.append(f"temp_{options.get('temperature')}")

    if options.get("use_flat_batch_ratio", False):
        base_name_parts.append("fltBchCorr")

    if options.get("postfix", None) is not None:
        base_name_parts.append(str(options.get("postfix")))

    return "_".join(base_name_parts)


def main(
    train_tasks: str = None,
    model: str = None,
    num_seeds: int = 1,
    dynamic_sampling: bool = False,
    vine: bool = False,
    grpo: bool = False,
    grpo_24k: bool = False,
    gspo: bool = False,
    prolong: bool = False,  
    dl_hf_before_submit: bool = True,
    fsdp_one: bool = False,
    postfix: str = None,
    disable_val_before_train: bool = False,
    exact: bool = False,
    tis: bool = False,
    train_batch_size: int = None,
    clip_ratio_high: float = None,
    no_kl_loss: bool = False,
    no_weight_decay: bool = False,
    hf_upload: bool = False,
    loss_agg_mode: str = None,
    fixed_num_optim_steps: int = None,
    use_flat_batch_ratio: bool = False,
    temperature: float = None,
):
    experiment_group_name = get_experiment_group_name(locals())
    print("Group name: ", experiment_group_name)

    if grpo:
        config_name = "polIter_r1d-1.5b_deepscaler_grpo"
    elif grpo_24k:
        config_name = "polIter_r1d-1.5b_deepscaler_grpo_24k"
    elif vine:
        config_name = "polIter_r1d-1.5b_deepscaler_vineppo"
    elif prolong:
        config_name = "polIter_r1d-1.5b_deepscaler_prolong"
    elif gspo:
        config_name = "polIter_r1d-1.5b_deepscaler_gspo"
    elif exact:
        config_name = "polIter_r1d-1.5b_deepscaler_grpo_exact"
    else:
        raise ValueError("Either grpo or vine must be True")

    experiment_tags = ["rlvr"]
    extra_args = []

    if train_tasks is not None:
        extra_args.append(f"train_tasks={train_tasks}")

    if model is not None:
        extra_args.append(f"actor_rollout_ref.model.path={model_to_path[model]}")
        if model == "qwen3-1.7b":
            extra_args.append("actor_rollout_ref.rollout.gpu_memory_utilization=0.7")

    if dynamic_sampling:
        extra_args.append("algorithm.filter_groups.enable=True")
        extra_args.append("data.train_batch_size=300")
        extra_args.append("+actor_rollout_ref.rollout.engine_kwargs.sglang.cuda_graph_max_bs=256")

    if fsdp_one:
        extra_args.append("actor_rollout_ref.actor.strategy=fsdp")

    if disable_val_before_train:
        extra_args.append("trainer.val_before_train=False")

    if tis:
        extra_args.append("actor_rollout_ref.rollout.calculate_log_probs=true")
        extra_args.append("actor_rollout_ref.actor.tis_imp_ratio_cap=2")
        extra_args.append("actor_rollout_ref.actor.policy_loss.loss_mode=vanilla_with_trace_lengths")

    if train_batch_size is not None:
        train_batch_size = int(train_batch_size)
        extra_args.append(f"data.train_batch_size={train_batch_size}")
        extra_args.append(f"actor_rollout_ref.actor.ppo_mini_batch_size={train_batch_size // 2}")

    if clip_ratio_high is not None:
        extra_args.append(f"actor_rollout_ref.actor.clip_ratio_high={clip_ratio_high}")

    if no_kl_loss:
        extra_args.append("actor_rollout_ref.actor.use_kl_loss=false")

    if no_weight_decay:
        extra_args.append("actor_rollout_ref.actor.optim.weight_decay=0")

    if loss_agg_mode is not None:
        extra_args.append(f"actor_rollout_ref.actor.loss_agg_mode={loss_agg_mode}")

    if hf_upload:
        extra_args.append("actor_rollout_ref.actor.checkpoint.push_to_hub_enabled=true")

    if temperature is not None:
        extra_args.append(f"actor_rollout_ref.rollout.temperature={temperature}")

    cmd = [
        "python",
        str(EXPERIMENT_UPLOADER),
        f"--config {config_name}",
        "--commands verl.trainer.main_policy_iteration",
        f"--group {experiment_group_name}",
        f"--num_seeds {num_seeds}",
        f"--tags {','.join(experiment_tags)}",
    ]

    if dl_hf_before_submit:
        path_to_dl_script = EXPERIMENT_UPLOADER.parent / "download_hf_models_and_datasets.sh"
        cmd.append(f"--pre-submit {path_to_dl_script}")

    cmd.extend(["--"] + extra_args)
    cmd = " ".join(cmd)

    output = os.popen(cmd).read()

    # Get the experiment id from the output using a regex
    try:
        exp_id = re.search(r"Uploaded Experiment ID: (.*)\n", output).group(1)
        exp_id = exp_id.strip()
    except Exception as e:
        print(f"Failed to get exp_id from output: {output}")
        raise e

    print(exp_id)


if __name__ == "__main__":
    fire.Fire(main)

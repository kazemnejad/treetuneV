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
    tasks = options.get("tasks") or "geneval_math"

    base_name_parts = [model, tasks]

    if options.get("batch_size") is not None:
        base_name_parts.append(f"bs{options.get('batch_size')}")

    if options.get("max_response_length") is not None:
        base_name_parts.append(f"maxResp{options.get('max_response_length')}")

    if options.get("rollout_n") is not None:
        base_name_parts.append(f"rolloutN{options.get('rollout_n')}")

    return "_".join(base_name_parts)


def main(
    tasks: str,
    model: str = None,
    num_seeds: int = 1,
    batch_size: int = None,
    max_response_length: int = None,
    rollout_n: int = None,
):
    experiment_group_name = get_experiment_group_name(locals())
    print("Group name: ", experiment_group_name)

    config_name = "geneval_math"

    experiment_tags = ["geneval"]
    extra_args = []

    if tasks is not None:
        extra_args.append(f"val_tasks={tasks}")

    if model is not None:
        extra_args.append(f"actor_rollout_ref.model.path={model_to_path[model]}")
        if model == "qwen3-1.7b":
            extra_args.append("actor_rollout_ref.rollout.gpu_memory_utilization=0.7")

    if batch_size is not None:
        extra_args.append(f"data.val_batch_size={batch_size}")

    if max_response_length is not None:
        extra_args.append(f"data.max_response_length={max_response_length}")

    if rollout_n is not None:
        extra_args.append(f"actor_rollout_ref.rollout.val_kwargs.n={rollout_n}")
        extra_args.append("ignore_dataset_rollout_n=true")

    cmd = [
        "python",
        str(EXPERIMENT_UPLOADER),
        f"--config {config_name}",
        "--commands verl.trainer.main_policy_iteration",
        f"--group {experiment_group_name}",
        f"--num_seeds {num_seeds}",
        f"--tags {','.join(experiment_tags)}",
    ]

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

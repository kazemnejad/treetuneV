# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2025 Amirhossein Kazemnejad and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Generate and evaluate responses given datasets of prompts
"""

import copy
import glob
import json
import math
import os
import uuid
from collections import defaultdict
from pprint import pprint

import hydra
import numpy as np
import ray
import torch
from omegaconf import OmegaConf
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.tasks.task import Split, get_dataset_paths
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.trainer.main_policy_iteration import create_rl_dataset
from verl.trainer.ppo.metric_utils import process_validation_metrics
from verl.trainer.ppo.ray_trainer import compute_response_mask
from verl.utils import hf_tokenizer
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.fs import copy_to_local, local_mkdir_safe
from verl.utils.tokenizer import hf_processor
from verl.workers.fsdp_workers_with_reward import ActorRolloutRefRewardWorker, AsyncActorRolloutRefRewardWorker


@hydra.main(config_path="config", config_name="gen_eval", version_base=None)
def main(config):
    if not ray.is_initialized():
        # Initialize Ray with a local cluster configuration
        # Set environment variables in the runtime environment to control tokenizer parallelism,
        # NCCL debug level, VLLM logging level, and allow runtime LoRA updating
        # `num_cpus` specifies the number of CPU cores Ray can use, obtained from the configuration
        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    # Make sure all the variables are resolved in the config
    OmegaConf.resolve(config)

    ray.get(main_task.remote(config))


@ray.remote(num_cpus=1)
class CollectorWorker:
    def __init__(self, config, default_local_dir):
        self.config = config
        model_local_path = copy_to_local(config.actor_rollout_ref.model.path)
        trust_remote_code = config.data.get("trust_remote_code", False)
        self.tokenizer = hf_tokenizer(model_local_path, trust_remote_code=trust_remote_code)
        self.processor = hf_processor(model_local_path, trust_remote_code=trust_remote_code, use_fast=True)

        from verl.workers.fsdp_workers_with_reward import MultiWorkerRewardManager

        self._reward_manager = MultiWorkerRewardManager(self.config, self.tokenizer)
        self._local_dir = os.path.join(default_local_dir, "generations")
        local_mkdir_safe(self._local_dir)

    def collect_batch(self, batch: DataProto, global_step: int):
        # Compute Rewards
        split_size = math.ceil(len(batch) / self._reward_manager.num_workers)
        reward_outputs = ray.get(
            [self._reward_manager.compute_reward(mini_batch) for mini_batch in batch.split(split_size)]
        )

        reward_tensor = torch.cat([out[0] for out in reward_outputs], dim=0)
        scores = reward_tensor.sum(-1).cpu().numpy()
        reward_extra_infos_dict = defaultdict(list)
        for out in reward_outputs:
            for k, v in out[1].items():
                reward_extra_infos_dict[k].extend(v)

        batch.non_tensor_batch["prompt"] = np.array(
            self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True), dtype=object
        )
        batch.non_tensor_batch["response"] = np.array(
            self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True), dtype=object
        )
        batch.non_tensor_batch["scores"] = scores
        if "data_source" not in batch.non_tensor_batch:
            batch.non_tensor_batch["data_source"] = np.array(["unknown"] * len(batch), dtype=object)

        if "response_lengths" in batch.batch.keys():
            response_lengths = batch.batch["response_lengths"]
        elif "response_mask" in batch.batch.keys():
            response_lengths = batch.batch["response_mask"].sum(dim=-1)
        else:
            response_lengths = compute_response_mask(batch).sum(dim=-1)
        batch.non_tensor_batch["response_length"] = response_lengths

        reward_extra_infos_dict["response_length"].extend(batch.non_tensor_batch["response_length"].tolist())
        reward_extra_infos_dict["reward"].extend(scores.tolist())
        for k in reward_extra_infos_dict:
            batch.non_tensor_batch[k] = np.array(reward_extra_infos_dict[k], dtype=object)

        # Remove all tensors since we don't want to save them
        meta_info = {"reward_extra_info_keys": sorted(list(reward_extra_infos_dict.keys()))}
        DataProto(non_tensor_batch=batch.non_tensor_batch, meta_info=meta_info).save_to_disk(
            os.path.join(self._local_dir, f"{global_step:06}.pkl")
        )

    def finalize(self):
        metric_dict = {}
        if self.config.get("compute_metrics", True):
            data_sources = []
            sample_inputs = []
            reward_extra_infos_dict = defaultdict(list)

            generations_paths = sorted(glob.glob(os.path.join(self._local_dir, "*.pkl")))
            for generation_path in generations_paths:
                batch = DataProto.load_from_disk(generation_path)
                data_sources.extend(batch.non_tensor_batch["data_source"].tolist())
                sample_inputs.extend(batch.non_tensor_batch["prompt"].tolist())
                for k in batch.meta_info["reward_extra_info_keys"]:
                    reward_extra_infos_dict[k].extend(batch.non_tensor_batch[k].tolist())

            for key_info, lst in reward_extra_infos_dict.items():
                assert len(lst) == 0 or len(lst) == len(sample_inputs), (
                    f"{key_info}: {len(lst)=}, {len(sample_inputs)=}"
                )

            data_src2var2metric2val = process_validation_metrics(data_sources, sample_inputs, reward_extra_infos_dict)
            for data_source, var2metric2val in data_src2var2metric2val.items():
                core_var = "acc" if "acc" in var2metric2val else "reward"
                for var_name, metric2val in var2metric2val.items():
                    n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                    for metric_name, metric_val in metric2val.items():
                        if (
                            (var_name == core_var)
                            and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                            and (f"@{n_max}" in metric_name)
                        ):
                            metric_sec = "val-core"
                        else:
                            metric_sec = "val-aux"
                        pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                        metric_dict[pfx] = metric_val

            with open(os.path.join(self._local_dir, "metrics.json"), "w") as f:
                json.dump(metric_dict, f, indent=4)

        # Push to hub if needed
        should_push_to_hub = (
            self.config.push_to_hub_enabled and self.config.hf_repo_id is not None and os.path.exists(self._local_dir)
        )
        if not should_push_to_hub:
            return metric_dict

        # Push to hub
        try:
            from huggingface_hub import HfApi

            api = HfApi()
            repo_info = api.create_repo(self.config.hf_repo_id, repo_type="dataset", exist_ok=True)
            repo_id = repo_info.repo_id

            print(f"Pushing {self._local_dir} to {repo_id}")
            api.upload_folder(
                repo_id=repo_id,
                folder_path=self._local_dir,
                path_in_repo="",
                repo_type="dataset",
                commit_message="GenEval results",
            )
            print(f"Successfully pushed {self._local_dir} to {repo_info.url}")
        except Exception as e:  # pragma: no cover
            if self.config.push_to_hub_ignore_errors:
                print(f"Push to hub failed but ignored: {e}")
                import traceback

                traceback.print_exc()
            else:
                raise

        return metric_dict


@ray.remote(num_cpus=1)
def main_task(config):
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    local_path = copy_to_local(config.actor_rollout_ref.model.path)
    trust_remote_code = config.data.get("trust_remote_code", False)
    tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
    processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

    dataset = build_datasets(config, tokenizer, processor)
    val_batch_size = config.data.get("val_batch_size", len(dataset))

    print(f"{len(dataset)=}")
    print(f"{val_batch_size=}")

    checkpoint_folder = config.trainer.default_local_dir
    if not os.path.isabs(checkpoint_folder):
        working_dir = os.getcwd()
        checkpoint_folder = os.path.join(working_dir, checkpoint_folder)

    collector = CollectorWorker.remote(config, checkpoint_folder)

    # Create dataloader
    from verl.utils.dataset.rl_dataset import collate_fn

    dataloader = StatefulDataLoader(
        dataset=dataset,
        batch_size=val_batch_size,
        num_workers=4,
        shuffle=config.data.get("validation_shuffle", True),
        drop_last=False,
        collate_fn=collate_fn,
    )

    actor_rollout_cls = (
        AsyncActorRolloutRefRewardWorker
        if config.actor_rollout_ref.rollout.mode == "async"
        else ActorRolloutRefRewardWorker
    )

    # Launch rollout engine
    actor_config = copy.deepcopy(config)
    OmegaConf.update(actor_config, "reward_model.num_workers", 0, force_add=True)
    ray_cls_with_init = RayClassWithInitArgs(
        cls=ray.remote(actor_rollout_cls), config=config.actor_rollout_ref, role="rollout", global_config=actor_config
    )
    resource_pool = RayResourcePool(process_on_nodes=[config.trainer.n_gpus_per_node] * config.trainer.nnodes)
    rollout_wg = RayWorkerGroup(
        resource_pool=resource_pool,
        ray_cls_with_init=ray_cls_with_init,
        device_name=config.trainer.device,
    )
    rollout_wg.init_model()
    del actor_config

    # create async rollout manager and request scheduler
    async_rollout_mode = False
    if config.actor_rollout_ref.rollout.mode == "async":
        from verl.experimental.agent_loop import AgentLoopManagerWithCustomWorker

        async_rollout_mode = True
        async_rollout_manager = AgentLoopManagerWithCustomWorker(
            config=config,
            worker_group=rollout_wg,
        )

    from verl.utils.tracking import Tracking

    logger = Tracking(
        project_name=config.trainer.project_name,
        experiment_name=config.trainer.experiment_name,
        default_backend=config.trainer.logger,
        config=OmegaConf.to_container(config, resolve=True),
    )

    # Load checkpoint if available
    print(f"checkpoint dir: {checkpoint_folder}")
    global_steps = _load_checkpoint(checkpoint_folder, config, dataloader)

    # add tqdm
    total_steps = len(dataloader)
    progress_bar = tqdm(total=len(dataloader), initial=global_steps, desc="GenEval Progress")

    global_steps += 1
    collect_ref = None

    ignore_dataset_rollout_n = config.get("ignore_dataset_rollout_n", False)

    for data in dataloader:
        batch = DataProto.from_single_dict(data)

        is_last_step = global_steps >= total_steps

        # add uid to batch
        batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)

        # repeat test batch
        repeat_times = None
        if not ignore_dataset_rollout_n:
            if "val_sampling_params.n" in batch.non_tensor_batch:
                repeat_times = batch.non_tensor_batch["val_sampling_params.n"]
            elif "val_sampling_params.n" in batch.batch:
                repeat_times = batch.batch["val_sampling_params.n"]

        if repeat_times is not None:
            batch = batch.sample_level_repeat(repeat_times=repeat_times)
        else:
            batch = batch.repeat(repeat_times=config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True)

        # add orig_prompt_len to batch
        if "raw_prompt_ids" in batch.non_tensor_batch:
            orig_prompt_len = np.array(list(map(len, batch.non_tensor_batch["raw_prompt_ids"])))
        else:
            orig_prompt_len = batch.batch["attention_mask"].sum(-1).cpu().numpy()

        # we only do validation on rule-based rm
        if config.reward_model.enable and batch[0].non_tensor_batch["reward_model"]["style"] == "model":
            return {}

        # prepare batch for generation
        reward_model_keys = set({"data_source", "reward_model", "extra_info", "uid"}) & batch.non_tensor_batch.keys()
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_model_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        gen_batch.meta_info = {
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "recompute_log_prob": False,
            "do_sample": config.actor_rollout_ref.rollout.val_kwargs.do_sample,
            "validate": True,
            "do_wake_up": global_steps == 1,  # wake up at the first step
            "do_sleep": is_last_step,
        }

        # pad to be divisible by dp_size
        size_divisor = (
            rollout_wg.world_size if not async_rollout_mode else config.actor_rollout_ref.rollout.agent.num_workers
        )
        gen_batch_padded, pad_size = pad_dataproto_to_divisor(gen_batch, size_divisor)
        if not async_rollout_mode:
            output_gen_batch_padded = rollout_wg.generate_sequences_with_manual_control(gen_batch_padded)
        else:
            output_gen_batch_padded = async_rollout_manager.generate_sequences(gen_batch_padded)

        # unpad
        output_gen_batch = unpad_dataproto(output_gen_batch_padded, pad_size=pad_size)
        batch = batch.union(output_gen_batch)
        batch.non_tensor_batch["orig_prompt_len"] = orig_prompt_len

        collect_ref = collector.collect_batch.remote(batch, global_steps)

        # Save checkpoint if needed
        should_save_checkpoint = config.trainer.save_freq > 0 and (
            is_last_step or global_steps % config.trainer.save_freq == 0
        )
        if should_save_checkpoint:
            # save dataloader
            local_global_step_folder = os.path.join(checkpoint_folder, f"global_step_{global_steps}")
            local_mkdir_safe(local_global_step_folder)
            dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
            dataloader_state_dict = dataloader.state_dict()
            torch.save(dataloader_state_dict, dataloader_local_path)

            # latest checkpointed iteration tracker (for atomic usage)
            local_latest_checkpointed_iteration = os.path.join(checkpoint_folder, "latest_checkpointed_iteration.txt")
            with open(local_latest_checkpointed_iteration, "w") as f:
                f.write(str(global_steps))

        progress_bar.update(1)
        global_steps += 1

    progress_bar.close()
    if collect_ref is not None:
        ray.get(collect_ref)

    metrics = ray.get(collector.finalize.remote())
    if len(metrics) == 0:
        logger.log(data=metrics, step=global_steps)


def build_datasets(config, tokenizer, processor):
    """Create validation datasets and the training sampler.

    Returns a tuple of (val_dataset, val_sampler).
    """

    if config.val_tasks is not None:
        val_files = get_dataset_paths(
            config.val_tasks,
            Split.VALIDATION,
            config.get("tasks_cache_dir", os.environ.get("VERL_TASKS_CACHE_DIR", None)),
        )
    else:
        val_files = config.data.val_files

    val_dataset = create_rl_dataset(val_files, config.data, tokenizer, processor, is_train=False)

    return val_dataset


def _load_checkpoint(checkpoint_folder, config, dataloader):
    resume_mode = config.trainer.resume_mode
    if resume_mode == "disable":
        return 0

    global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

    if resume_mode == "auto":
        if global_step_folder is None:
            print("Running from scratch")
            return 0
    elif resume_mode == "resume_path":
        assert isinstance(config.trainer.resume_from_path, str), "resume ckpt must be str type"
        assert "global_step_" in config.trainer.resume_from_path, "resume ckpt must specify the global_steps"
        global_step_folder = config.trainer.resume_from_path
        if not os.path.isabs(global_step_folder):
            working_dir = os.getcwd()
            global_step_folder = os.path.join(working_dir, global_step_folder)

    print(f"Load from checkpoint folder: {global_step_folder}")

    # set global step
    global_steps = int(global_step_folder.split("global_step_")[-1])

    print(f"Setting global step to {global_steps}")
    print(f"Resuming from {global_step_folder}")

    # load dataloader
    dataloader_local_path = os.path.join(global_step_folder, "data.pt")
    if os.path.exists(dataloader_local_path):
        dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
        dataloader.load_state_dict(dataloader_state_dict)
    else:
        print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    return global_steps


def create_generator_from_results(results_dir: str):
    """Create a generator from the results directory."""
    generations_paths = sorted(glob.glob(os.path.join(results_dir, "*.pkl")))
    for generation_path in generations_paths:
        batch = DataProto.load_from_disk(generation_path)
        yield from batch


def create_hf_dataset_from_results(results_dir: str):
    """Create a HuggingFace dataset from the results directory."""
    generations_paths = sorted(glob.glob(os.path.join(results_dir, "*.pkl")))

    uid_to_items = defaultdict(list)
    uid_to_index = {}
    index = 0
    for generation_path in tqdm(generations_paths, desc="Loading generations"):
        batch = DataProto.load_from_disk(generation_path)
        for item in batch:
            uid_to_items[item.non_tensor_batch["uid"]].append(item)
            uid_to_index[item.non_tensor_batch["uid"]] = index
            index += 1

    rows = []
    uids = sorted(uid_to_index.keys(), key=lambda x: uid_to_index[x])
    for uid in uids:
        items = uid_to_items[uid]
        respones = [item.non_tensor_batch["response"] for item in items]
        response_lengths = [item.non_tensor_batch["response_length"] for item in items]
        scores = [item.non_tensor_batch["scores"] for item in items]
        reward_extra_infos_keys = items[0].meta_info["reward_extra_info_keys"]
        reward_extra_infos = {key: [item.non_tensor_batch[key] for item in items] for key in reward_extra_infos_keys}

        keys_to_ignore = {"response", "response_length", "scores", *reward_extra_infos_keys}
        row = {
            "response": respones,
            "response_length": response_lengths,
            "scores": scores,
            **reward_extra_infos,
            **{k: v for k, v in items[0].non_tensor_batch.items() if k not in keys_to_ignore},
        }
        rows.append(row)

    from datasets import Dataset

    return Dataset.from_list(rows)


if __name__ == "__main__":
    main()

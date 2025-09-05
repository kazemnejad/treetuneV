# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import json
import logging
import os
import shutil
import warnings
from dataclasses import asdict, dataclass
from typing import Optional

import ray
import torch
import torch.distributed
from accelerate import init_empty_weights
from omegaconf import DictConfig
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardedOptimStateDictConfig, ShardedStateDictConfig, StateDictType
from transformers import GenerationConfig, PreTrainedTokenizer, ProcessorMixin

from verl.utils.device import is_cuda_available
from verl.utils.fs import copy_to_local, is_non_local, local_mkdir_safe
from verl.utils.fsdp_utils import fsdp_version, get_fsdp_full_state_dict, get_fsdp_state_ctx
from verl.utils.logger import log_with_rank

from .checkpoint_manager import BaseCheckpointManager

# Setup logging
logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


@ray.remote(num_cpus=0)
def _push_to_hub_ray_task(
    repo_id: str,
    hf_local_path: str,
    global_step: int,
    ignore_errors: bool,
):
    """Background upload to Hugging Face Hub using a Ray task.

    The task swallows all exceptions when ignore_errors=True.
    """
    try:
        from huggingface_hub import HfApi

        branch_name = f"ckpt-{global_step:06d}"

        api = HfApi()
        repo_id = api.create_repo(repo_id, repo_type="model", exist_ok=True).repo_id

        logger.info(f"Pushing {hf_local_path} to {repo_id} on branch {branch_name}")
        api.create_branch(repo_id, branch=branch_name, exist_ok=True)
        api.upload_folder(
            repo_id=repo_id,
            folder_path=hf_local_path,
            path_in_repo="",
            repo_type="model",
            commit_message=f"Checkpoint push for step {global_step:06d}",
            revision=branch_name,
        )
        logger.info(f"Successfully pushed {hf_local_path} to {repo_id}@{branch_name}")
    except Exception as e:  # pragma: no cover
        if ignore_errors:
            logger.info(f"Push to hub failed but ignored: {e}")
            import traceback

            traceback.print_exc()
        else:
            raise


@dataclass
class FSDPConfig:
    """Configuration for FSDP checkpointing.

    Args:
        FSDP_version (int): Version of FSDP being used.
        world_size (int): Number of processes in the distributed training setup.
    """

    FSDP_version: int
    world_size: int


class FSDPCheckpointManager(BaseCheckpointManager):
    """
    Manage FSDP checkpointing in SPMD training.

    - Saves/loads per-rank sharded model & optimizer states
    - Persists full lr_scheduler and RNG state
    - Stores HF tokenizer/processor and model/config for unified restore

    Args:
        model (FSDP): Wrapped model instance.
        optimizer (Optimizer): Training optimizer.
        lr_scheduler (LRScheduler): Learning-rate scheduler.
        processing_class (PreTrainedTokenizer or ProcessorMixin, optional):
            Pre-/post-processing artifact handler.
        checkpoint_contents DictConfig: Configuration for checkpoint contents.
            - 'load': Components to load; must contain 'model'. Defaults to ['model', 'optimizer', 'extra'].
            - 'save': Components to save; must contain 'model'. Defaults to ['model', 'optimizer', 'extra'].
    """

    def __init__(
        self,
        model: FSDP,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        processing_class: PreTrainedTokenizer | ProcessorMixin = None,
        checkpoint_config: DictConfig = None,
        **kwargs,
    ):
        if processing_class is None:
            assert "tokenizer" in kwargs, "tokenizer or processor must be provided"
            warnings.warn(
                "`tokenizer` is deprecated. use `processing_class` instead.", DeprecationWarning, stacklevel=2
            )
            processing_class = kwargs.pop("tokenizer")

        super().__init__(
            model,
            optimizer,
            lr_scheduler=lr_scheduler,
            processing_class=processing_class,
            checkpoint_config=checkpoint_config,
        )

    def load_checkpoint(self, local_path: str, hdfs_path: str = None, del_local_after_load=False):
        """
        Load an FSDP checkpoint for this rank.

        Downloads and loads:
          - model and optimizer shards
          - extra state dict (scheduler + RNG)

        Args:
            local_path: Directory with per-rank checkpoint files.
            hdfs_path: Unused (for API compatibility).
            del_local_after_load: Remove local files after loading.
        """
        if local_path is None:
            return

        # check if the checkpoint_load_contents is valid
        if self.should_load_model:
            assert self.model is not None, "model must be provided when checkpoint_contents.load includes ['model']"
        if self.should_load_optimizer:
            assert self.optimizer is not None, (
                "optimizer must be provided when checkpoint_contents.load includes ['optimizer']"
            )

        # every rank download its own checkpoint
        state_dict_cfg = (
            ShardedStateDictConfig(offload_to_cpu=True if is_cuda_available else False)
            if self.should_load_model
            else None
        )
        optim_cfg = (
            ShardedOptimStateDictConfig(offload_to_cpu=True if is_cuda_available else False)
            if self.should_load_optimizer
            else None
        )
        with get_fsdp_state_ctx(self.model, StateDictType.SHARDED_STATE_DICT, state_dict_cfg, optim_cfg):
            if self.should_load_model:
                remote_model_path = os.path.join(local_path, f"model_world_size_{self.world_size}_rank_{self.rank}.pt")
                local_model_path = copy_to_local(remote_model_path)
                model_state_dict = torch.load(local_model_path, weights_only=False)
                self.model.load_state_dict(model_state_dict)
                log_with_rank(f"Loaded model from {remote_model_path}", rank=self.rank, logger=logger)

            if self.should_load_optimizer:
                remote_optim_path = os.path.join(local_path, f"optim_world_size_{self.world_size}_rank_{self.rank}.pt")
                local_optim_path = copy_to_local(remote_optim_path)
                optimizer_state_dict = torch.load(local_optim_path, weights_only=False)
                self.optimizer.load_state_dict(optimizer_state_dict)
                log_with_rank(f"Loaded optimizer from {remote_optim_path}", rank=self.rank, logger=logger)

        if self.should_load_extra:
            remote_extra_state_path = os.path.join(
                local_path, f"extra_state_world_size_{self.world_size}_rank_{self.rank}.pt"
            )
            local_extra_state_path = copy_to_local(remote_extra_state_path)
            extra_state_dict = torch.load(local_extra_state_path, weights_only=False)
            # recover random state
            if "rng" in extra_state_dict:
                # 'rng' may not exist for backward compatibility
                self.load_rng_state(extra_state_dict["rng"])
                log_with_rank(f"Loaded rng from {remote_extra_state_path}", rank=self.rank, logger=logger)

            lr_scheduler_state_dict = extra_state_dict["lr_scheduler"]
            if lr_scheduler_state_dict is not None and self.lr_scheduler is not None:
                self.lr_scheduler.load_state_dict(lr_scheduler_state_dict)
                log_with_rank(f"Loaded lr_scheduler from {remote_extra_state_path}", rank=self.rank, logger=logger)

        if self.rank == 0 and del_local_after_load:
            try:
                os.remove(local_model_path) if is_non_local(local_model_path) else None
                os.remove(local_optim_path) if is_non_local(local_optim_path) else None
                os.remove(local_extra_state_path) if is_non_local(local_extra_state_path) else None
            except Exception as e:
                log_with_rank(
                    f"remove local resume ckpt file after loading failed, exception {e} will be ignored",
                    rank=self.rank,
                    logger=logger,
                )

        # wait for everyone to load checkpoints
        torch.distributed.barrier()

    def save_checkpoint(self, local_path: str, hdfs_path: str = None, global_step: int = 0, max_ckpt_to_keep=None):
        """
        Save an FSDP checkpoint for this rank.

        Writes:
          - model & optimizer shard files
          - extra state dict (scheduler + RNG)
          - HF tokenizer/processor and model/config on rank 0
          - optional full HF model under 'huggingface/' if requested

        Rotates old checkpoints, keeping at most `max_ckpt_to_keep`.

        Args:
            local_path: Target directory for checkpoint files.
            hdfs_path: Unused (for API compatibility).
            global_step: Current training step (used for bookkeeping).
            max_ckpt_to_keep: Number of recent checkpoints to retain.
        """
        if local_path is None:
            return

        # record the previous global step
        self.previous_global_step = global_step

        # remove previous local_path, only rank 0 should do this
        if (
            self.rank == 0
            and max_ckpt_to_keep
            and isinstance(max_ckpt_to_keep, int)
            and max_ckpt_to_keep > 0
            and len(self.previous_saved_paths) >= max_ckpt_to_keep
        ):
            keep_start = len(self.previous_saved_paths) - max_ckpt_to_keep + 1
            self.remove_previous_save_local_path(self.previous_saved_paths[:keep_start])
            self.previous_saved_paths = self.previous_saved_paths[keep_start:]

        local_path = local_mkdir_safe(local_path)
        torch.distributed.barrier()

        # check if the checkpoint_save_contents is valid
        if self.should_save_model:
            assert self.model is not None, "model must be provided when checkpoint_contents.save includes ['model']"
        if self.should_save_optimizer:
            assert self.optimizer is not None, (
                "optimizer must be provided when checkpoint_contents.save includes ['optimizer']"
            )

        # every rank will save its own model and optim shard
        state_dict_cfg = ShardedStateDictConfig(offload_to_cpu=True if is_cuda_available else False)
        optim_cfg = ShardedOptimStateDictConfig(offload_to_cpu=True if is_cuda_available else False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with get_fsdp_state_ctx(self.model, StateDictType.SHARDED_STATE_DICT, state_dict_cfg, optim_cfg):
                model_path = os.path.join(local_path, f"model_world_size_{self.world_size}_rank_{self.rank}.pt")
                optim_path = os.path.join(local_path, f"optim_world_size_{self.world_size}_rank_{self.rank}.pt")
                extra_path = os.path.join(local_path, f"extra_state_world_size_{self.world_size}_rank_{self.rank}.pt")

                if self.should_save_model:
                    model_state_dict = self.model.state_dict()
                    torch.save(model_state_dict, model_path)
                    log_with_rank(f"Saved model to {os.path.abspath(model_path)}", rank=self.rank, logger=logger)

                if self.should_save_optimizer:
                    optimizer_state_dict = self.optimizer.state_dict()
                    torch.save(optimizer_state_dict, optim_path)
                    log_with_rank(f"Saved optim to {os.path.abspath(optim_path)}", rank=self.rank, logger=logger)

                if self.should_save_extra:
                    lr_scheduler_state_dict = self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None
                    extra_state_dict = {
                        "lr_scheduler": lr_scheduler_state_dict,
                        "rng": self.get_rng_state(),
                    }
                    torch.save(extra_state_dict, extra_path)
                    log_with_rank(f"Saved extra_state to {os.path.abspath(extra_path)}", rank=self.rank, logger=logger)

        if self.rank == 0:
            # Save HF tokenizer/processor and model config on rank 0 to huggingface/ directory, no matter whether
            # huggingface model is requested to be saved or not.

            if fsdp_version(self.model) == 1:
                unwrap_model = self.model._fsdp_wrapped_module
            else:
                unwrap_model = self.model

            hf_config_tokenizer_path = os.path.join(local_path, "huggingface")
            local_mkdir_safe(hf_config_tokenizer_path)
            model_config = unwrap_model.config
            generation_config = None
            if unwrap_model.can_generate() and hasattr(model_config, "name_or_path") and model_config.name_or_path:
                try:
                    # Some model's name_or_path is empty if not initialized from pretrained,
                    # in this cases, we don't save generation config.
                    generation_config = GenerationConfig.from_pretrained(model_config.name_or_path)
                    generation_config.save_pretrained(hf_config_tokenizer_path)
                except Exception:
                    # if the generation config isn't available, we don't save it
                    pass

            model_config.save_pretrained(hf_config_tokenizer_path)
            self.processing_class.save_pretrained(hf_config_tokenizer_path)
            log_with_rank(
                f"Saved model config and tokenizer class to {os.path.abspath(hf_config_tokenizer_path)}",
                rank=self.rank,
                logger=logger,
                log_only_rank_0=True,
            )

            # Also save runtime FSDP config
            fsdp_config_path = os.path.join(local_path, "fsdp_config.json")
            fsdp_config = FSDPConfig(
                FSDP_version=fsdp_version(self.model),
                world_size=self.world_size,
            )
            with open(fsdp_config_path, "w") as f:
                json.dump(asdict(fsdp_config), f, indent=4)

        # wait for everyone to dump to local
        torch.distributed.barrier()

        if self.should_save_hf_model:
            # Only rank 0 will save hf model and,
            # offload to cpu to save LLMs which may be too large to fit in one GPU
            state_dict = get_fsdp_full_state_dict(self.model, offload_to_cpu=True, rank0_only=True)

            if self.rank == 0:
                hf_local_path = os.path.join(local_path, "huggingface")
                os.makedirs(hf_local_path, exist_ok=True)

                if "ForTokenClassification" in model_config.architectures[0]:
                    from transformers import AutoModelForTokenClassification

                    auto_model_cls = AutoModelForTokenClassification
                elif "ForCausalLM" in model_config.architectures[0]:
                    from transformers import AutoModelForCausalLM

                    auto_model_cls = AutoModelForCausalLM
                elif "ForConditionalGeneration" in model_config.architectures[0]:
                    from transformers import AutoModelForVision2Seq

                    auto_model_cls = AutoModelForVision2Seq
                else:
                    raise NotImplementedError(f"Unknown architecture {model_config['architectures']}")

                with init_empty_weights():
                    save_model = auto_model_cls.from_config(model_config, torch_dtype=torch.bfloat16)
                save_model.to_empty(device="cpu")

                if save_model.can_generate():
                    if generation_config is not None:
                        save_model.generation_config = generation_config
                    else:
                        print(
                            f"Warning: {self.__class__.__name__}.save_checkpoint: Generation config file not found "
                            f"in, using a generation config created from the model config when saving hf_model."
                        )

                save_model.save_pretrained(hf_local_path, state_dict=state_dict)
                log_with_rank(
                    f"Saved hf_model to {os.path.abspath(hf_local_path)}",
                    rank=self.rank,
                    logger=logger,
                    log_only_rank_0=True,
                )
                del state_dict
                del save_model

            # wait for rank0 to dump hf_model to local
            torch.distributed.barrier()

        self.previous_saved_paths.append(local_path)


class KeepingFSDPCheckpointManager(FSDPCheckpointManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.keep_every_n_saves = self.checkpoint_config.get("keep_every_n_saves", 1)
        assert isinstance(self.keep_every_n_saves, int) and self.keep_every_n_saves > 0, (
            "keep_every_n_saves must be a positive integer"
        )
        self.keep_only_hf_in_previous_saves = self.checkpoint_config.get("keep_only_hf_in_previous_saves", False)
        self._all_previous_saved_paths = []
        self.hf_repo_id = self.checkpoint_config.get("hf_repo_id", None)
        self.push_to_hub_freq = self.checkpoint_config.get("push_to_hub_freq", 0)
        self.push_to_hub_enabled = self.checkpoint_config.get("push_to_hub_enabled", False)
        self.push_to_hub_ignore_errors = self.checkpoint_config.get("push_to_hub_ignore_errors", False)

    def _push_to_hub(self, hf_local_path, global_step):
        assert self.hf_repo_id is not None, "hf_repo_id must be set"

        from huggingface_hub import HfApi

        # Use a branch name related to this checkpoint path (e.g., last folder name)
        branch_name = f"ckpt-{global_step:06d}"

        # Create repo if it doesn't exist
        api = HfApi()
        repo_id = api.create_repo(self.hf_repo_id, repo_type="model", exist_ok=True).repo_id

        # Upload the folder to the hub under the branch
        print(f"Pushing {hf_local_path} to {repo_id} on branch {branch_name}")
        api.create_branch(repo_id, branch=branch_name, exist_ok=True)
        api.upload_folder(
            repo_id=repo_id,
            folder_path=hf_local_path,
            path_in_repo="",
            repo_type="model",
            commit_message=f"Checkpoint push for step {global_step:06d}",
            revision=branch_name,
        )
        print(f"Successfully pushed {hf_local_path} to {repo_id}@{branch_name}")

    def save_checkpoint(self, local_path: str, hdfs_path: str = None, global_step: int = 0, max_ckpt_to_keep=None):
        assert max_ckpt_to_keep is None, "max_ckpt_to_keep is not supported for KeepingFSDPCheckpointManager"
        super().save_checkpoint(
            local_path, hdfs_path=hdfs_path, global_step=global_step, max_ckpt_to_keep=max_ckpt_to_keep
        )
        self._all_previous_saved_paths.append(self.previous_saved_paths[-1])

        if self.rank != 0:
            return

        if self.keep_every_n_saves > 1:
            all_previous_saved_paths = self._all_previous_saved_paths.copy()

            candidates = all_previous_saved_paths[:-1]  # keep the last because we just saved it
            self.previous_saved_paths = candidates[:: self.keep_every_n_saves] + [all_previous_saved_paths[-1]]

            to_remove = [p for p in candidates if p not in self.previous_saved_paths]
            self.remove_previous_save_local_path(to_remove)

        if self.keep_only_hf_in_previous_saves:
            files_to_keep = [
                "huggingface",
                "fsdp_config.json",
            ]
            for path in self.previous_saved_paths[:-2]:  # keep the last two checkpoints fully intact
                if os.path.exists(path):
                    for item in os.listdir(path):
                        item_path = os.path.join(path, item)
                        if item not in files_to_keep:
                            if os.path.isfile(item_path):
                                os.remove(item_path)
                            elif os.path.isdir(item_path):
                                shutil.rmtree(item_path)
                            log_with_rank(
                                f"Removed {item} from checkpoint {path}",
                                rank=self.rank,
                                logger=logger,
                                log_only_rank_0=True,
                            )

        hf_local_path = os.path.join(self.previous_saved_paths[-1], "huggingface")
        should_push_to_hub = (
            self.hf_repo_id is not None
            and self.push_to_hub_enabled
            and self.push_to_hub_freq > 0
            and global_step % self.push_to_hub_freq == 0
            and os.path.exists(hf_local_path)
        )
        if should_push_to_hub:
            # Always spawn a Ray task pinned to the same node; allow concurrent pushes.
            current_node_id = ray.get_runtime_context().get_node_id()
            _push_to_hub_ray_task.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=current_node_id, soft=False)
            ).remote(self.hf_repo_id, hf_local_path, global_step, self.push_to_hub_ignore_errors)
            log_with_rank(
                f"Spawned background Ray push for step {global_step} to {self.hf_repo_id}",
                rank=self.rank,
                logger=logger,
                log_only_rank_0=True,
            )

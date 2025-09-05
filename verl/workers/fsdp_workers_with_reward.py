import logging
import os

import ray
import torch
from omegaconf import DictConfig

from verl import DataProto
from verl.single_controller.base.decorator import (
    Dispatch,
    collect_dp_compute_data_proto,
    dispatch_dp_compute_data_proto_with_indices,
    make_nd_compute_dataproto_dispatch_fn,
    register,
)
from verl.trainer.ppo.reward import compute_reward, load_reward_manager
from verl.utils.checkpoint.fsdp_checkpoint_manager import KeepingFSDPCheckpointManager
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.device import get_device_id, get_device_name, get_torch_device
from verl.utils.fs import copy_to_local
from verl.utils.profiler.performance import log_gpu_memory_usage, reduce_timing, simple_timer, topk_reduce_ratio_min_max
from verl.utils.profiler.profile import DistProfiler
from verl.workers.config.rollout import RolloutConfig
from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

device_name = get_device_name()


@ray.remote
class RewardWorker:
    def __init__(self, config, tokenizer):
        self._reward_fn = load_reward_manager(
            config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {})
        )

    def compute(self, data: DataProto):
        return compute_reward(data, self._reward_fn)


class MultiWorkerRewardManager:
    def __init__(self, config, tokenizer):
        current_node_id = ray.get_runtime_context().get_node_id()
        self._reward_workers = [
            RewardWorker.options(
                # Colocate reward workers with the actor and rollout to avoid cross-node communication
                scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                    node_id=current_node_id, soft=False
                )
            ).remote(config, tokenizer)
            for _ in range(config.reward_model.get("num_workers", 1))
        ]
        logger.info(f"Initialized {len(self._reward_workers)} reward workers")
        self._next_worker_idx = 0

    def compute_reward(self, data: DataProto) -> ray.ObjectRef:
        worker = self._reward_workers[self._next_worker_idx]
        self._next_worker_idx = (self._next_worker_idx + 1) % len(self._reward_workers)
        return worker.compute.remote(data)
    
    @property
    def num_workers(self):
        return len(self._reward_workers)


class ActorRolloutRefRewardWorker(ActorRolloutRefWorker):
    def __init__(self, config: DictConfig, role: str, **kwargs):
        super().__init__(config, role, **kwargs)

        if self._is_rollout:
            assert "global_config" in kwargs, "global_config is required"
            self._global_config = kwargs["global_config"]

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        super().init_model()

        if not self._is_rollout:
            return

        self._reward_manager = MultiWorkerRewardManager(self._global_config, self.tokenizer)
        if hasattr(self, "rollout"):
            self.rollout.reward_manager = self._reward_manager

        if self._is_actor:
            self.checkpoint_manager = KeepingFSDPCheckpointManager(
                model=self.actor_module_fsdp,
                optimizer=self.actor.actor_optimizer,
                lr_scheduler=self.actor_lr_scheduler,
                processing_class=self.processor if self.processor is not None else self.tokenizer,
                checkpoint_config=self.config.actor.checkpoint,
            )

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="rollout"))
    @DistProfiler.annotate(color="red", role="rollout_generate")
    def generate_sequences(self, prompts: DataProto):
        # Support all hardwares
        prompts = prompts.to(get_device_id())

        assert self._is_rollout

        meta_info = {
            "eos_token_id": self.generation_config.eos_token_id
            if self.generation_config is not None
            else self.tokenizer.eos_token_id,
            "pad_token_id": self.generation_config.pad_token_id
            if self.generation_config is not None
            else self.tokenizer.pad_token_id,
        }
        prompts.meta_info.update(meta_info)
        timing_generate = {}
        with self.rollout_sharding_manager:
            log_gpu_memory_usage("After entering rollout sharding manager", logger=logger)

            with simple_timer("generate_sequences", timing_generate):
                output = self.rollout.generate_sequences(prompts=prompts)

            log_gpu_memory_usage("After rollout generation", logger=logger)

        # compute per worker throughput
        if "response_lengths" in output.batch:
            total_response_tokens = output.batch["response_lengths"].sum().item()
            timing_generate["gen_throughput_per_worker"] = total_response_tokens / timing_generate["generate_sequences"]

        timing_generate.update(self.rollout_sharding_manager.timing)
        # We calculate the average timing across all ranks
        # to make sure meta_info["timing"] is the same
        timing_generate_topk_ratio, timing_generate_min, timing_generate_max = topk_reduce_ratio_min_max(
            timing_generate["generate_sequences"]
        )
        timing_generate = reduce_timing(timing_generate)
        timing_generate.update(
            {
                "generation_timing/max": timing_generate_max,
                "generation_timing/min": timing_generate_min,
                "generation_timing/topk_ratio": timing_generate_topk_ratio,
            }
        )
        output.meta_info["timing"] = timing_generate
        output = output.to("cpu")

        # clear kv cache
        get_torch_device().empty_cache()
        return output

    @register(
        dispatch_mode={
            "dispatch_fn": dispatch_dp_compute_data_proto_with_indices,
            "collect_fn": collect_dp_compute_data_proto,
        }
    )
    @DistProfiler.annotate(color="red", role="rollout_generate_manual")
    def generate_sequences_with_manual_control(self, prompts: DataProto):
        # The driver may ask us to repeat the prompts to save communication cost
        if "rollout_n" in prompts.batch:
            prompts = prompts.sample_level_repeat(prompts.batch["rollout_n"])
        elif prompts.meta_info.get("rollout_n", 1) > 1:
            prompts = prompts.repeat(repeat_times=prompts.meta_info["rollout_n"], interleave=True)

        # Support all hardwares
        prompts = prompts.to(get_device_id())

        assert self._is_rollout

        do_wake_up = prompts.meta_info.pop("do_wake_up", True)
        do_sleep = prompts.meta_info.pop("do_sleep", True)

        meta_info = {
            "eos_token_id": self.generation_config.eos_token_id
            if self.generation_config is not None
            else self.tokenizer.eos_token_id,
            "pad_token_id": self.generation_config.pad_token_id
            if self.generation_config is not None
            else self.tokenizer.pad_token_id,
        }
        prompts.meta_info.update(meta_info)
        timing_generate = {}

        # always reset timing
        self.rollout_sharding_manager.timing = {}

        if do_wake_up:
            self.rollout_sharding_manager.__enter__()
        else:
            logger.info("Skipping rollout wake up")

        log_gpu_memory_usage("After entering rollout sharding manager", logger=logger)

        with simple_timer("generate_sequences", timing_generate):
            output = self.rollout.generate_sequences(prompts=prompts)

        log_gpu_memory_usage("After rollout generation", logger=logger)

        if do_sleep:
            self.rollout_sharding_manager.__exit__(None, None, None)
        else:
            logger.info("Skipping rollout sleep")

        # compute per worker throughput
        if "response_lengths" in output.batch:
            total_response_tokens = output.batch["response_lengths"].sum().item()
            timing_generate["gen_throughput_per_worker"] = total_response_tokens / timing_generate["generate_sequences"]

        timing_generate.update(self.rollout_sharding_manager.timing)
        # We calculate the average timing across all ranks
        # to make sure meta_info["timing"] is the same
        timing_generate_topk_ratio, timing_generate_min, timing_generate_max = topk_reduce_ratio_min_max(
            timing_generate["generate_sequences"]
        )
        timing_generate = reduce_timing(timing_generate)
        timing_generate.update(
            {
                "generation_timing/max": timing_generate_max,
                "generation_timing/min": timing_generate_min,
                "generation_timing/topk_ratio": timing_generate_topk_ratio,
            }
        )
        output.meta_info["timing"] = timing_generate
        output = output.to("cpu")

        # clear kv cache
        get_torch_device().empty_cache()
        return output

    def _build_rollout(self, trust_remote_code=False):
        if self.config.rollout.name != "sglang_custom":
            return super()._build_rollout(trust_remote_code)

        from torch.distributed.device_mesh import init_device_mesh

        # TODO(sgm): support FSDP hybrid shard for larger model
        infer_tp = self.config.rollout.tensor_model_parallel_size
        dp = self.world_size // infer_tp
        assert self.world_size % infer_tp == 0, (
            f"rollout world_size: {self.world_size} is not divisible by infer_tp: {infer_tp}"
        )
        rollout_device_mesh = init_device_mesh(
            device_name, mesh_shape=(dp, infer_tp), mesh_dim_names=["dp", "infer_tp"]
        )
        rollout_name = self.config.rollout.name

        is_collect = rollout_device_mesh["infer_tp"].get_local_rank() == 0
        self._register_dispatch_collect_info(
            "rollout", dp_rank=rollout_device_mesh["dp"].get_local_rank(), is_collect=is_collect
        )

        rollout_config: RolloutConfig = omega_conf_to_dataclass(self.config.rollout)

        from verl.workers.rollout.sglang_rollout.sglang_rollout_custom import CustomSGLangRollout

        # NOTE(linjunrong): Due to recent fp8 support in SGLang. Now importing any symbol relate to
        # SGLang's model_runner would check CUDA device capability. However, due to verl's setting,
        # the main process of ray can not find any CUDA device, which would potentially lead to:
        # "RuntimeError: No CUDA GPUs are available".
        # For this reason, sharding_manager.__init__ should not import FSDPSGLangShardingManager and
        # we import it here use the abs path.
        # check: https://github.com/sgl-project/sglang/blob/00f42707eaddfc2c0528e5b1e0094025c640b7a0/python/sglang/srt/layers/quantization/fp8_utils.py#L76
        from verl.workers.sharding_manager.fsdp_sglang import FSDPSGLangShardingManager

        local_path = copy_to_local(self.config.model.path)
        log_gpu_memory_usage(f"Before building {rollout_name} rollout", logger=logger)
        rollout = CustomSGLangRollout(
            actor_module=local_path,
            config=rollout_config,
            processing_class=self.processor if self.processor is not None else self.tokenizer,
            model_hf_config=self.actor_model_config,
            trust_remote_code=trust_remote_code,
        )
        log_gpu_memory_usage(f"After building {rollout_name} rollout", logger=logger)

        if torch.distributed.get_world_size() == 1:
            self.config.rollout.load_format = "dummy_hf"
        rollout_sharding_manager = FSDPSGLangShardingManager(
            module=self.actor_module_fsdp,
            inference_engine=rollout._engine,
            model_config=self.actor_model_config,
            rollout_config=self.config.rollout,
            full_params="hf" in self.config.rollout.load_format,
            device_mesh=rollout_device_mesh,
            offload_param=self._is_offload_param,
            multi_stage_wake_up=self.config.rollout.multi_stage_wake_up,
        )
        log_gpu_memory_usage("After building sharding manager", logger=logger)

        return rollout, rollout_sharding_manager


class AsyncActorRolloutRefRewardWorker(AsyncActorRolloutRefWorker):
    def __init__(self, config: DictConfig, role: str, **kwargs):
        super().__init__(config, role, **kwargs)
        if self._is_rollout:
            assert "global_config" in kwargs, "global_config is required"
            self._global_config = kwargs["global_config"]

    def _build_rollout(self, trust_remote_code=False):
        if self.config.rollout.name != "sglang_custom":
            return super()._build_rollout(trust_remote_code)

        from torch.distributed.device_mesh import init_device_mesh

        # TODO(sgm): support FSDP hybrid shard for larger model
        infer_tp = self.config.rollout.tensor_model_parallel_size
        dp = self.world_size // infer_tp
        assert self.world_size % infer_tp == 0, (
            f"rollout world_size: {self.world_size} is not divisible by infer_tp: {infer_tp}"
        )
        rollout_device_mesh = init_device_mesh(
            device_name, mesh_shape=(dp, infer_tp), mesh_dim_names=["dp", "infer_tp"]
        )
        rollout_name = self.config.rollout.name

        is_collect = rollout_device_mesh["infer_tp"].get_local_rank() == 0
        self._register_dispatch_collect_info(
            "rollout", dp_rank=rollout_device_mesh["dp"].get_local_rank(), is_collect=is_collect
        )

        rollout_config: RolloutConfig = omega_conf_to_dataclass(self.config.rollout)

        from verl.workers.rollout.sglang_rollout.sglang_rollout_custom import CustomSGLangRollout

        # NOTE(linjunrong): Due to recent fp8 support in SGLang. Now importing any symbol relate to
        # SGLang's model_runner would check CUDA device capability. However, due to verl's setting,
        # the main process of ray can not find any CUDA device, which would potentially lead to:
        # "RuntimeError: No CUDA GPUs are available".
        # For this reason, sharding_manager.__init__ should not import FSDPSGLangShardingManager and
        # we import it here use the abs path.
        # check: https://github.com/sgl-project/sglang/blob/00f42707eaddfc2c0528e5b1e0094025c640b7a0/python/sglang/srt/layers/quantization/fp8_utils.py#L76
        from verl.workers.sharding_manager.fsdp_sglang import FSDPSGLangShardingManager

        local_path = copy_to_local(self.config.model.path)
        log_gpu_memory_usage(f"Before building {rollout_name} rollout", logger=logger)
        rollout = CustomSGLangRollout(
            actor_module=local_path,
            config=rollout_config,
            processing_class=self.processor if self.processor is not None else self.tokenizer,
            model_hf_config=self.actor_model_config,
            trust_remote_code=trust_remote_code,
        )
        log_gpu_memory_usage(f"After building {rollout_name} rollout", logger=logger)

        if torch.distributed.get_world_size() == 1:
            self.config.rollout.load_format = "dummy_hf"
        rollout_sharding_manager = FSDPSGLangShardingManager(
            module=self.actor_module_fsdp,
            inference_engine=rollout._engine,
            model_config=self.actor_model_config,
            rollout_config=self.config.rollout,
            full_params="hf" in self.config.rollout.load_format,
            device_mesh=rollout_device_mesh,
            offload_param=self._is_offload_param,
            multi_stage_wake_up=self.config.rollout.multi_stage_wake_up,
        )
        log_gpu_memory_usage("After building sharding manager", logger=logger)

        self.vllm_tp_size = self.config.rollout.tensor_model_parallel_size
        self.vllm_dp_rank = int(os.environ["RANK"]) // self.vllm_tp_size
        self.vllm_tp_rank = int(os.environ["RANK"]) % self.vllm_tp_size

        # used for sleep/wake_up
        rollout.sharding_manager = rollout_sharding_manager

        return rollout, rollout_sharding_manager

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        super().init_model()
        if not self._is_rollout:
            return

        self._reward_manager = MultiWorkerRewardManager(self._global_config, self.tokenizer)
        if hasattr(self, "rollout"):
            self.rollout.reward_manager = self._reward_manager

        if self._is_actor:
            self.checkpoint_manager = KeepingFSDPCheckpointManager(
                model=self.actor_module_fsdp,
                optimizer=self.actor.actor_optimizer,
                lr_scheduler=self.actor_lr_scheduler,
                processing_class=self.processor if self.processor is not None else self.tokenizer,
                checkpoint_config=self.config.actor.checkpoint,
            )

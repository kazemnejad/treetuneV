# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Callable, Optional

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from sglang.srt.managers.tokenizer_manager import (
    ReleaseMemoryOccupationReqInput,
    ResumeMemoryOccupationReqInput,
    UpdateWeightsFromTensorReqInput,
)
from sglang.srt.utils import get_ip, get_open_port
from tensordict import TensorDict
from tqdm import tqdm

import verl.third_party.sglang.engine
from verl import DataProto
from verl.utils.net_utils import is_ipv6
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.torch_functional import get_response_mask, pad_sequence_to_length
from verl.workers.fsdp_workers_with_reward import MultiWorkerRewardManager
from verl.workers.rollout.sglang_rollout.sglang_rollout import SGLangRollout, _post_process_outputs, _pre_process_inputs
from verl.workers.rollout.sglang_rollout.utils import broadcast_pyobj

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


# because chatCompletion is an async method, it makes the whole ray actor be an async actor
# which can not call loop.run_until_complete. So we need to make the engine to be an async class
class AsyncEngine(verl.third_party.sglang.engine.Engine):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # default to use dummy load format, which need to reload weights in first time
        self._need_reload = True

    async def release_memory_occupation(self, tags: Optional[list[str]] = None):
        """Release GPU occupation temporarily."""
        if tags is None:
            obj = ReleaseMemoryOccupationReqInput()
        else:
            obj = ReleaseMemoryOccupationReqInput(tags=tags)
        return await self.tokenizer_manager.release_memory_occupation(obj, None)

    async def resume_memory_occupation(self, tags: Optional[list[str]] = None):
        """Resume GPU occupation."""
        # because __init__ is a sync method, it can not call the async release_memory_occupation
        # have to move release_memory_occupation from __init__ to here
        # For multi-stage awake, we run release weight and kv_cache when we resume weights for the first time.
        if self._need_reload:
            await self.release_memory_occupation()
            self._need_reload = False

        if tags is None:
            obj = ResumeMemoryOccupationReqInput()
        else:
            obj = ResumeMemoryOccupationReqInput(tags=tags)
        return await self.tokenizer_manager.resume_memory_occupation(obj, None)

    async def update_weights_from_tensor(self, update_weights_request: UpdateWeightsFromTensorReqInput):
        return await self.tokenizer_manager.update_weights_from_tensor(update_weights_request, None)

    async def flush_cache(self):
        return await self.tokenizer_manager.flush_cache()

    async def abort_request(self, rid: str = "", abort_all: bool = False):
        """Abort a specific request or all requests.

        Args:
            rid: The request ID to abort. If empty and abort_all is False, no action is taken.
            abort_all: If True, abort all running requests regardless of rid.
        """
        return self.tokenizer_manager.abort_request(rid=rid, abort_all=abort_all)

    async def async_generate(
        self,
        *args,
        concurrent_reward_computation: bool = False,
        reward_manager: Optional[MultiWorkerRewardManager] = None,
        convert_to_data_proto: Callable[[list[dict[str, Any]], DataProto], DataProto] = None,
        prompts_data: Optional[DataProto] = None,
        **kwargs,
    ):
        if concurrent_reward_computation:
            assert convert_to_data_proto is not None
            assert reward_manager is not None
            assert prompts_data is not None

            kwargs["stream_once_complete"] = True
            kwargs["stream"] = False

            # Initialize progress bar
            progress_bar = tqdm(total=len(kwargs["input_ids"]), desc="Generating responses", unit="resp")

            rewards_futures = {}
            responses = {}
            generator = await super().async_generate(*args, **kwargs)
            async for resp in generator:
                req_id = resp["index"]
                responses[req_id] = resp

                # Update progress bar
                progress_bar.update(1)

                # Compute reward for the current request asynchronously
                sliced_prompts = prompts_data[req_id : req_id + 1]
                outputs = [resp]
                data_proto = convert_to_data_proto(outputs, sliced_prompts)

                rewards_futures[req_id] = reward_manager.compute_reward(data_proto)

            progress_bar.close()

            # Wait for all rewards to be computed
            reward_tensors = [None] * len(rewards_futures)
            reward_extra_infos_dicts = [None] * len(rewards_futures)
            for req_id, reward_future in rewards_futures.items():
                reward, reward_infos = await reward_future
                reward_tensors[req_id] = reward
                reward_extra_infos_dicts[req_id] = reward_infos

            # Concatenate rewards and reward extra infos and keep the same order as the responses
            reward_tensors = torch.cat(reward_tensors, dim=0)
            reward_extra_infos_dicts = {
                k: np.concatenate([d[k] for d in reward_extra_infos_dicts], axis=0)
                for k in reward_extra_infos_dicts[0].keys()
            }

            # Reconstruct the responses in the same order as the prompts
            responses = [responses[i] for i in range(len(responses))]

            return responses, reward_tensors, reward_extra_infos_dicts
        else:
            return await super().async_generate(*args, **kwargs)


class CustomSGLangRollout(SGLangRollout):
    def _init_inference_engine(self, trust_remote_code, actor_module, port):
        # initialize the inference engine
        nnodes = -(-self._tp_size // len(self.visible_devices_set))
        if nnodes > 1:
            ip = get_ip()
            port = get_open_port() if port is None else port
            [ip, port] = broadcast_pyobj(
                [ip, port],
                rank=self._rank,
                dist_group=self._device_mesh_cpu.get_group("tp"),
                src=self._device_mesh_cpu["tp"].mesh[0].item(),
                force_cpu_device=False,
            )
            dist_init_addr = f"[{ip}]:{port}" if is_ipv6(ip) else f"{ip}:{port}"
        else:
            dist_init_addr = None

        load_format = "dummy" if self.config.load_format.startswith("dummy") else self.config.load_format
        tp_size_per_node = self._tp_size // nnodes
        node_rank = self._tp_rank // tp_size_per_node
        first_rank_in_node = self._tp_rank % tp_size_per_node == 0
        enable_debug = self.config.get("enable_debug", False) and self._rank == 0
        engine_kwargs = self.config.get("engine_kwargs", {}).get("sglang", {}) or {}
        if isinstance(engine_kwargs, DictConfig):
            engine_kwargs = OmegaConf.to_container(engine_kwargs, resolve=True)
        # attention backend will be changed to fa3 if not specified
        attention_backend = engine_kwargs.pop("attention_backend", None)

        if attention_backend is not None:
            logger.info(f"Using attention backend: {attention_backend}")

        if first_rank_in_node:
            rank = dist.get_rank()
            os.environ["SGLANG_BLOCK_NONZERO_RANK_CHILDREN"] = "0"
            logger.info(f"engine_kwargs: {engine_kwargs}")
            self._engine = AsyncEngine(
                model_path=actor_module,
                dtype=self.config.dtype,
                mem_fraction_static=self.config.gpu_memory_utilization,
                enable_memory_saver=True,
                base_gpu_id=0,
                gpu_id_step=1,
                tp_size=self._tp_size,
                node_rank=node_rank,
                load_format=load_format,
                dist_init_addr=dist_init_addr,
                nnodes=nnodes,
                trust_remote_code=trust_remote_code,
                context_length=self.config.max_model_len,
                # NOTE(linjunrong): add rank to prevent SGLang generate same port inside PortArgs.init_new
                # when random.seed is being set during training
                port=30000 + rank,
                # NOTE(Chenyang): turn on log_level to see the decoding speed of SGLang Engine
                # log_level="INFO"
                # NOTE(Chenyang): turn the following lines to see the input and output of each request
                log_level="INFO" if enable_debug else "WARN",
                enable_metrics=True if enable_debug else False,
                # log_requests=True,
                # log_requests_level=2,
                # NOTE(Chenyang): turn on max_running_requests to set the max concurrent running requests
                # max_running_requests=1,
                mm_attention_backend="fa3",
                attention_backend=attention_backend if attention_backend is not None else "fa3",
                # In async mode for AgentLoop, SGLang support token in token out to avoid the tokenizer
                # inconsistency issue.
                skip_tokenizer_init=self.config.mode == "async",
                **engine_kwargs,
            )
        else:
            self._engine = None

        self.sharding_manager = None
        self.is_sleep = True

    @GPUMemoryLogger(role="sglang rollout", logger=logger)
    @torch.no_grad()
    def _batch_level_generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        """Generates single-turn sequences for a batch of prompts.
        For single-turn generation, all prompts are processed in one request.
        `_batch_level_generate_sequences` involves:
        1.  Extracting and pre-processing prompt token IDs from the input
            `prompts`. This includes handling padding and preparing raw
            token ID lists.
        2.  Preparing inputs for the SGLang engine, including multi-modal
            data if present.
        3.  Invoking the SGLang engine (`self._engine.async_generate`,
            an async coroutine) with the batch of processed inputs and
            specified sampling parameters on the master TP rank.
        4.  Broadcasting the results from the master TP rank to all
            other TP ranks.
        5.  Post-processing the engine's output to format the generated
            token IDs and (if applicable) log probabilities.
        6.  Constructing the final sequences by concatenating original
            prompts with the generated responses.
        7.  Updating attention masks and position IDs to reflect the full
            concatenated sequences.
        8.  If `self.config.free_cache_engine` is true, the SGLang engine's
            KV cache is flushed after generation on the master TP rank.
        Args:
            prompts: A `DataProto` object containing the batch of
              input prompts, including tensor data (like `input_ids`,
              `attention_mask`) and meta-information (like `eos_token_id`,
              `do_sample`).
            **kwargs: Additional keyword arguments that can override the
              default sampling parameters (e.g., `temperature`, `top_p`,
              `max_new_tokens`). These are temporarily applied using
              `update_sampling_params`.
        Returns:
            DataProto: A `DataProto` object containing the batch of
              generated sequences. This includes tensors for `prompts`
              (original input IDs), `responses` (generated token IDs),
              `input_ids` (concatenated prompt and response),
              `attention_mask`, and `position_ids` for the full
              sequences.
        Note that in GRPO, if the prompts are validated, we repeat the prompts for rollout.n times in ray_trainer.
        Thus we do not need to repeat the prompts here and set the sampling parameter n to 1.
        """
        # input ids: (bs, prompt_length), left-padded
        idx = prompts.batch["input_ids"]
        # attention_mask: (bs, seq_length), left-padded
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]

        # used to generate attention mask for the
        # response based on EOS token position
        eos_token_id = prompts.meta_info["eos_token_id"]

        concurrent_reward_computation = (
            prompts.meta_info.get("concurrent_reward_computation", False)
            and getattr(self, "reward_manager", None) is not None
        )
        return_rewards_extra_info = prompts.meta_info.get("return_rewards_extra_info", False)

        free_engine_cache = prompts.meta_info.get("free_engine_cache", True)
        return_response_lengths = prompts.meta_info.get("return_response_lengths", True)

        keys_to_keep = prompts.meta_info.get("keys_to_keep", None)

        batch_size = idx.size(0)

        # Extract non-tensor data
        non_tensor_batch = prompts.non_tensor_batch
        if "raw_prompt_ids" not in non_tensor_batch:
            non_tensor_batch["raw_prompt_ids"] = np.array(
                [_pre_process_inputs(self.pad_token_id, idx[i]).tolist() for i in range(batch_size)],
                dtype=object,
            )

        if "multi_modal_data" in non_tensor_batch:
            sglang_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(
                non_tensor_batch.pop("raw_prompt_ids"),
                non_tensor_batch.pop("multi_modal_data"),
                strict=True,
            ):
                sglang_inputs.append(
                    {
                        "prompt_token_ids": raw_prompt_ids,
                        "multi_modal_data": multi_modal_data,
                        "image_data": (
                            multi_modal_data.get("image", None) if isinstance(multi_modal_data, dict) else None
                        ),
                    }
                )
        else:
            sglang_inputs = [
                {"prompt_token_ids": raw_prompt_ids} for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")
            ]

        for input_data in sglang_inputs:
            # Ensure token IDs are lists or numpy arrays
            if not isinstance(input_data["prompt_token_ids"], list | np.ndarray):
                raise TypeError(
                    f"prompt_token_ids must be a list or numpy array, got {type(input_data['prompt_token_ids'])}"
                )

            input_data["prompt_token_ids"] = list(input_data["prompt_token_ids"])

        # Extract token IDs and image data for SGLang Engine
        idx_list = [input_data["prompt_token_ids"] for input_data in sglang_inputs]
        image_list = [input_data.get("image_data", None) for input_data in sglang_inputs]

        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)

        max_new_token_list = prompts.batch.get("max_new_tokens", None)

        # Create request-level sampling parameters
        request_sampling_params = self.sampling_params.copy()
        if not do_sample:
            request_sampling_params.update(
                {
                    "n": 1,
                    "presence_penalty": 0.0,
                    "frequency_penalty": 0.0,
                    "repetition_penalty": 1.0,
                    "temperature": 0,
                    "top_p": 1,
                    "top_k": -1,
                    "ignore_eos": False,
                    "min_new_tokens": 0,
                    "max_new_tokens": self.config.response_length,
                    "skip_special_tokens": True,
                    "spaces_between_special_tokens": True,
                }
            )
        elif is_validate:
            request_sampling_params.update(
                {
                    "top_k": self.config.val_kwargs.top_k,
                    "top_p": self.config.val_kwargs.top_p,
                    "temperature": self.config.val_kwargs.temperature,
                    "n": 1,  # if validate, already repeat in ray_trainer
                }
            )

        # Update with any additional kwargs
        request_sampling_params.update(kwargs)

        if max_new_token_list is not None:
            # We need to pass a sampling_params for each request
            request_sampling_params_list = []
            for max_new_token in max_new_token_list.tolist():
                sampling_params = request_sampling_params.copy()
                sampling_params["max_new_tokens"] = max_new_token
                request_sampling_params_list.append(sampling_params)
            request_sampling_params = request_sampling_params_list
            assert len(request_sampling_params) == len(idx_list), (
                "max_new_token_list and idx_list must have the same length"
            )

        if self._tp_rank == 0:
            loop = asyncio.get_event_loop()
            output = loop.run_until_complete(
                self._engine.async_generate(
                    prompt=None,  # because we have already convert it to prompt token id
                    sampling_params=request_sampling_params,
                    return_logprob=True,
                    input_ids=idx_list,
                    image_data=image_list,
                    concurrent_reward_computation=concurrent_reward_computation,
                    reward_manager=getattr(self, "reward_manager", None),
                    convert_to_data_proto=self._convert_output_data_proto,
                    prompts_data=prompts,
                )
            )
        else:
            output = None

        # Most naive implementation, can extract tensor and send via gloo if too slow
        dist.barrier()
        [output] = broadcast_pyobj(
            data=[output],
            rank=self._rank,
            dist_group=self._device_mesh_cpu["tp"].get_group(),
            src=self._device_mesh_cpu["tp"].mesh[0].item(),
            force_cpu_device=False,
        )
        if concurrent_reward_computation:
            output, rewards_tensor, reward_extra_infos_dict = output

        out = _post_process_outputs(self.processing_class, output)

        response = out[0].to(idx.device)
        rollout_log_probs = None
        if self.config.calculate_log_probs:
            rollout_log_probs = out[1].to(idx.device)

        if response.shape[1] < self.config.response_length:
            response = pad_sequence_to_length(response, self.config.response_length, self.pad_token_id)
            if self.config.calculate_log_probs:
                rollout_log_probs = pad_sequence_to_length(
                    rollout_log_probs, self.config.response_length, self.pad_token_id
                )

        seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(batch_size, 1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(
            response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        tensor_dict_data = {
            "prompts": idx,
            "responses": response,
            "input_ids": seq,  # here input_ids become the whole sentences
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }
        if concurrent_reward_computation:
            tensor_dict_data["__rewards"] = rewards_tensor

        if return_response_lengths:
            response_lengths = response_attention_mask.sum(-1)
            tensor_dict_data["response_lengths"] = response_lengths

        if self.config.calculate_log_probs:
            # we will recompute old log prob with actor
            tensor_dict_data["rollout_log_probs"] = rollout_log_probs

        batch = TensorDict(tensor_dict_data, batch_size=batch_size)

        # free cache engine
        if self._engine is not None and self._tp_rank == 0 and free_engine_cache:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(self._engine.flush_cache())

        data = DataProto(batch=batch, non_tensor_batch=non_tensor_batch)

        if keys_to_keep is not None:
            assert isinstance(keys_to_keep, dict), "keys_to_keep must be a dict"
            if "batch" not in keys_to_keep:
                keys_to_keep["batch"] = sorted(data.batch.keys())
            if "non_tensor_batch" not in keys_to_keep:
                keys_to_keep["non_tensor_batch"] = sorted(data.non_tensor_batch.keys())
            if "meta_info" not in keys_to_keep:
                keys_to_keep["meta_info"] = sorted(data.meta_info.keys())
            if return_response_lengths and "response_lengths" not in keys_to_keep["batch"]:
                keys_to_keep["batch"].append("response_lengths")

            data = data.pop(**{f"{k}_keys": v for k, v in keys_to_keep.items()})

        if return_rewards_extra_info:
            data.non_tensor_batch.update(reward_extra_infos_dict)

        return data

    def _convert_output_data_proto(self, output: list[dict[str, Any]], prompts: DataProto) -> DataProto:
        # input ids: (bs, prompt_length), left-padded
        idx = prompts.batch["input_ids"]
        # attention_mask: (bs, seq_length), left-padded
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]

        # used to generate attention mask for the
        # response based on EOS token position
        eos_token_id = prompts.meta_info["eos_token_id"]

        batch_size = idx.size(0)

        # Extract non-tensor data
        non_tensor_batch = prompts.non_tensor_batch
        out = _post_process_outputs(self.processing_class, output)

        response = out[0].to(idx.device)
        rollout_log_probs = None
        if self.config.calculate_log_probs:
            rollout_log_probs = out[1].to(idx.device)

        if response.shape[1] < self.config.response_length:
            response = pad_sequence_to_length(response, self.config.response_length, self.pad_token_id)
            if self.config.calculate_log_probs:
                rollout_log_probs = pad_sequence_to_length(
                    rollout_log_probs, self.config.response_length, self.pad_token_id
                )

        seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(batch_size, 1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(
            response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                "prompts": idx,
                "responses": response,
                "input_ids": seq,  # here input_ids become the whole sentences
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )
        if self.config.calculate_log_probs:
            # we will recompute old log prob with actor
            batch["rollout_log_probs"] = rollout_log_probs

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)

    async def generate(
        self,
        prompt_ids: torch.Tensor,
        sampling_params: dict[str, Any],
        request_id: str,
        image_data: Optional[list[Any]] = None,
    ) -> torch.Tensor | dict[str, Any]:
        """Generate sequence with token-in-token-out."""
        return_full_output = sampling_params.pop("return_full_output", False)
        request_sampling_params = self.sampling_params.copy()
        request_sampling_params.update(sampling_params)
        output = await self._handle_engine_generate(prompt_ids, request_sampling_params, image_data=image_data)
        if return_full_output:
            return output
        else:
            return output["output_ids"]

    async def _handle_engine_generate(
        self, generation_prompt_ids: list[int], sampling_params: dict, image_data: Optional[list[Any]] = None
    ) -> dict:
        requested_response_length = sampling_params.get("max_new_tokens", self.config.response_length)
        return_logprob = sampling_params.pop("return_logprob", False)
        max_new_tokens = min(requested_response_length, self.config.max_model_len - len(generation_prompt_ids) - 1)

        kwargs = sampling_params.copy()
        kwargs["max_new_tokens"] = max_new_tokens
        kwargs["n"] = 1  # group size is supported in preprocess

        output = await self._engine.async_generate(
            input_ids=generation_prompt_ids,
            sampling_params=kwargs,
            return_logprob=return_logprob,
            image_data=image_data,
        )
        return output

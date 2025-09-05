"""
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import uuid
from collections import defaultdict
from pprint import pprint
from typing import Any, Optional

import numpy as np
import ray
import torch
from tqdm import tqdm

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import DataProtoItem
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics_with_distribution,
    compute_throughout_metrics,
    compute_timing_and_throughput_metrics,
)
from verl.trainer.ppo.ray_trainer import apply_kl_penalty, compute_response_mask
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.registry import register as register_trainer
from verl.trainer.treetune_ppo.ray_trainer import RayTreetunePPOTrainer
from verl.trainer.vineppo.solution_splitters import get_solution_splitter_cls
from verl.utils.checkpoint.checkpoint_manager import should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import pad_tensor_list


def _hash_tensor_like(arr: np.ndarray | torch.Tensor) -> tuple[int, ...]:
    """Return a hashable, collision-safe key for an ndarray."""
    if isinstance(arr, torch.Tensor):
        arr = arr.cpu().numpy()
    if not arr.flags.c_contiguous:
        arr = np.ascontiguousarray(arr)
    return (arr.tobytes(), arr.shape, arr.dtype.str)


def _pre_process_inputs(
    pad_token_id,
    prompt_token_ids: torch.Tensor,
) -> torch.Tensor:
    # remove the left padding in the prompt token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    return prompt_token_ids[non_pad_index:]


def _deduplicate_mc_queries(
    queries: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[tuple[int, ...], list[int]]]:
    unique_queries = []
    unique_to_orig_map: dict[tuple[int, ...], list[int]] = defaultdict(list)

    for idx, query in enumerate(queries):
        prompt_ids = _hash_tensor_like(query["input_ids"])
        if prompt_ids not in unique_to_orig_map:
            unique_queries.append(query)

        unique_to_orig_map[prompt_ids].append(idx)

    # Ensure that the queries are unique
    assert len(set([tuple(q["input_ids"].tolist()) for q in unique_queries])) == len(unique_queries)

    return unique_queries, unique_to_orig_map


def _unpack_unique_mc_results(
    unique_results: list[dict[str, Any]],
    unique_to_orig_map: dict[tuple[int, ...], list[int]],
    original_queries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    results = [None] * len(original_queries)
    for res in unique_results:
        prompt_ids = _hash_tensor_like(res["input_ids"])
        for orig_idx in unique_to_orig_map[prompt_ids]:
            orig_query = original_queries[orig_idx]
            assert results[orig_idx] is None  # This is the first time we see this query
            results[orig_idx] = orig_query

            if orig_query is res:
                # This is the original query which was unique, so we don't need to do anything
                continue

            assert _hash_tensor_like(orig_query["input_ids"]) == prompt_ids

            for key in res.keys():
                if not key.startswith("__mc_"):
                    continue
                assert key not in orig_query, f"Key {key} already exists in the original query"
                orig_query[key] = res[key]

    # Ensure that all requests have been filled
    assert all([r is not None for r in results])

    return results


@register_trainer("vineppo")
class RayVinePPOTrainer(RayTreetunePPOTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_response_length = self.config.data.max_response_length
        self.vineppo_config = omega_conf_to_dataclass(self.config.trainer.vineppo)
        solution_splitter_config = omega_conf_to_dataclass(self.config.trainer.solution_splitter)
        self.solution_splitter = get_solution_splitter_cls(solution_splitter_config.name)(
            tokenizer=self.tokenizer, **solution_splitter_config.kwargs
        )

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        wake_up_engine_before_rollout = True

        # Dynamic Sampling (DAPO style)
        dynamic_sampling_state = {}
        do_filter_groups = self.config.algorithm.get("filter_groups", {}).get("enable", False)
        if do_filter_groups:
            assert not self.config.reward_model.launch_reward_fn_async, (
                "Dynamic Sampling (DAPO style) is not supported when reward_model.launch_reward_fn_async is True"
            )

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )

                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # add uid to batch
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                # pop those keys for generation
                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
                if "multi_modal_data" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("multi_modal_data")
                if "raw_prompt" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                if "interaction_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("interaction_kwargs")
                if "index" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("index")
                if "agent_name" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("agent_name")

                gen_batch = batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch.meta_info["free_engine_cache"] = False
                gen_batch.meta_info["do_wake_up"] = wake_up_engine_before_rollout
                gen_batch.meta_info["do_sleep"] = False  # We will sleep after estimating the values
                gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences_with_manual_control(gen_batch)
                        else:
                            raise NotImplementedError("Async rollout mode is not implemented in VinePPO.")
                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        raise NotImplementedError("REMAX advantage estimation is not implemented in VinePPO.")
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)

                    reward_extra_infos_keys: list[str] = []
                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(data=batch, reward_fn=self.reward_fn)
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)
                            if reward_extra_infos_dict:
                                batch.non_tensor_batch.update(
                                    {k: np.array(v) for k, v in reward_extra_infos_dict.items()}
                                )
                            reward_extra_infos_keys = list(reward_extra_infos_dict.keys())
                            batch.batch["token_level_scores"] = reward_tensor

                    # might need to generate more batches
                    if do_filter_groups:
                        batch, dynamic_sampling_state, more_generation_needed = self._filter_groups(
                            batch, dynamic_sampling_state
                        )
                        if more_generation_needed:
                            print("More generation needed. Skipping to the next batch.")
                            wake_up_engine_before_rollout = False
                            continue
                        metrics.update(dynamic_sampling_state["metrics"])

                    # compute MC value estimates
                    with marked_timer("e2s_mc_values", timing_raw, color="purple"):
                        batch, mc_values_metrics = self._compute_mc_values(batch, timing_raw)
                        metrics.update(mc_values_metrics)

                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    batch_mc_extra_info = batch.pop(non_tensor_batch_keys=["mc_extra_info"])

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # recompute old_log_probs
                    with marked_timer("old_log_prob", timing_raw, color="blue"):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                        if "rollout_log_probs" in batch.batch.keys():
                            # TODO: we may want to add diff of probs too.
                            from verl.utils.debug.metrics import calculate_debug_metrics

                            metrics.update(calculate_debug_metrics(batch))

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer("ref", timing_raw, color="olive"):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                            if reward_extra_infos_dict:
                                batch.non_tensor_batch.update(
                                    {k: np.array(v) for k, v in reward_extra_infos_dict.items()}
                                )
                                reward_extra_infos_keys = list(reward_extra_infos_dict.keys())
                            batch.batch["token_level_scores"] = reward_tensor

                        # Extract reward_extra_infos_dict from batch
                        reward_extra_infos_dict = {
                            k: batch.non_tensor_batch[k].tolist() for k in reward_extra_infos_keys
                        }

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # compute advantages, executed on the driver process
                        batch = self._compute_vineppo_advantage(batch)

                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    full_episodes_dump_freq = self.config.trainer.get("full_episodes_dump_freq", 0)
                    should_dump_full_episodes = (
                        rollout_data_dir is not None
                        and full_episodes_dump_freq > 0
                        and (self.global_steps % full_episodes_dump_freq == 0 or self.global_steps == 1)
                    )
                    if should_dump_full_episodes:
                        save_dir = os.path.join(rollout_data_dir, "full_episodes", f"iter_{self.global_steps:06d}")
                        os.makedirs(save_dir, exist_ok=True)
                        batch.meta_info["config"] = OmegaConf.to_container(self.config, resolve=True)
                        batch.save_to_disk(os.path.join(save_dir, "batch.pkl"))
                        batch_mc_extra_info.save_to_disk(os.path.join(save_dir, "mc_extra_info.pkl"))
                        batch.meta_info.pop("config", None)

                    # remove the batch keys that are not needed for training
                    batch.pop(non_tensor_batch_keys=["mc_values", "states"])

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_dump_freq = self.config.trainer.get("rollout_dump_freq", 0)
                    should_save_rollout = (
                        rollout_data_dir is not None
                        and rollout_dump_freq > 0
                        and (self.global_steps % rollout_dump_freq == 0 or self.global_steps == 1)
                    )
                    if should_save_rollout:
                        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            sample_gts = [
                                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None)
                                for item in batch
                            ]

                            if "request_id" in batch.non_tensor_batch:
                                reward_extra_infos_dict.setdefault(
                                    "request_id",
                                    batch.non_tensor_batch["request_id"].tolist(),
                                )

                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                gts=sample_gts,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )

                    # validate
                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                    ):
                        with marked_timer("testing", timing_raw, color="green"):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                    esi_close_to_expiration = should_save_ckpt_esi(
                        max_steps_duration=self.max_steps_duration,
                        redundant_time=self.config.trainer.esi_redundant_time,
                    )
                    # Check if the conditions for saving a checkpoint are met.
                    # The conditions include a mandatory condition (1) and
                    # one of the following optional conditions (2/3/4):
                    # 1. The save frequency is set to a positive value.
                    # 2. It's the last training step.
                    # 3. The current step number is a multiple of the save frequency.
                    # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                    if self.config.trainer.save_freq > 0 and (
                        is_last_step
                        or self.global_steps % self.config.trainer.save_freq == 0
                        or esi_close_to_expiration
                    ):
                        if esi_close_to_expiration:
                            print("Force saving checkpoint: ESI instance expiration approaching.")
                        with marked_timer("save_checkpoint", timing_raw, color="green"):
                            self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics_with_distribution(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_and_throughput_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1
                dynamic_sampling_state = {}
                wake_up_engine_before_rollout = True

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)

    @torch.no_grad()
    def _compute_vineppo_advantage(self, batch: DataProto) -> DataProto:
        advantages = torch.zeros_like(batch.batch["responses"], dtype=torch.float32)
        response_lengths = batch.batch["response_mask"].sum(dim=-1)

        for idx, item in enumerate(batch):
            states: np.ndarray = item.non_tensor_batch["states"]
            mc_values: dict[int, float] = item.non_tensor_batch["mc_values"]
            response_length = response_lengths[idx]

            assert np.all(states[:-1] <= states[1:]), f"States must be sorted: {states}"

            for i in range(len(states) - 1):
                curr_state, next_state = states[i], states[i + 1]
                advantages[idx, curr_state:next_state] = mc_values[next_state] - mc_values[curr_state]

            # Handle the final segment (last state till the end of the response)
            assert states[-1] < response_length, (
                f"Last state must be before the end of the response: {states[-1]} < {response_length}"
            )
            final_reward = batch.batch["token_level_scores"][idx, response_length - 1]
            advantages[idx, states[-1] : response_length] = final_reward - mc_values[states[-1]]

        # Zero out segments if specified (e.g. useful for low entropy segments)
        advantage_mask = batch.batch.get("advantage_mask", batch.batch["response_mask"])
        advantages *= advantage_mask

        batch.batch["advantages"] = advantages
        batch.batch["returns"] = (
            batch.batch["token_level_scores"].sum(dim=-1, keepdim=True) * batch.batch["response_mask"]
        )

        return batch

    def _compute_mc_values(self, batch: DataProto, timing_raw: dict) -> DataProto:
        """
        Compute MC value estimates for the batch.
        """
        metrics = {}

        # Assign a unique id to each trajectory
        batch.non_tensor_batch["_traj_uid"] = np.array(
            [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
        )

        with marked_timer("mcval_split_trajectories", timing_raw):
            states_list = self.solution_splitter(batch)

        mc_values, mc_extra_info = self._estimate_values(
            trajectories=batch,
            states_list=states_list,
            timing_raw=timing_raw,
            timing_and_metric_prefix="mcval_",
            metrics=metrics,
            wake_up_engine_before_rollout=False,  # Engine is already awake
            sleep_engine_after_rollout=True,
        )

        # States: List of state indices for each trajectory. Each item is a list of integers representing
        # the token positions where the trajectory was split for Monte Carlo value estimation.
        # Example: [0, 3, 5, ...] where each number is a token index in the trajectory
        batch.non_tensor_batch["states"] = states_list

        # MC values: List of dictionaries, one per trajectory. Each dict maps state indices to their
        # corresponding Monte Carlo value estimates. Example: {0: 0.1, 3: 0.2, 5: 0.3, ...}
        batch.non_tensor_batch["mc_values"] = mc_values

        # MC extra info: List of dictionaries, one per trajectory. Each dict maps state indices to
        # additional Monte Carlo estimation metadata (rollout returns, rollout texts, etc.).
        # Example: {0: {"__mc_returns": [0.1, 0.2, 0.3], "__mc_rollouts": ["...", "...", "..."], ...}, ...}
        batch.non_tensor_batch["mc_extra_info"] = mc_extra_info

        with marked_timer("mcval_stats", timing_raw):
            num_states_list = np.array([len(states) for states in states_list])
            state_length_list = np.array([np.mean(np.diff(states)) for states in states_list])

            mc_values_flat = np.concatenate(
                [list(mc_values[i].values()) for i in range(len(mc_values))], dtype=np.float32
            )

            stats = {
                # number of states
                "num_states/mean": np.mean(num_states_list),
                "num_states/max": np.max(num_states_list),
                "num_states/min": np.min(num_states_list),
                "num_states/dist": num_states_list,
                # average state length
                "avg_state_length/mean": np.mean(state_length_list),
                "avg_state_length/max": np.max(state_length_list),
                "avg_state_length/min": np.min(state_length_list),
                "avg_state_length/dist": state_length_list,
                # mc values
                "values/mean": np.mean(mc_values_flat),
                "values/dist": mc_values_flat,
            }
            metrics.update({f"mcval_{k}": v for k, v in stats.items()})

        return batch, metrics

    def _estimate_values(
        self,
        trajectories: DataProto,
        states_list: np.ndarray,
        timing_raw: dict,
        timing_and_metric_prefix: str = "",
        metrics: dict = None,
        wake_up_engine_before_rollout: Optional[bool] = None,
        sleep_engine_after_rollout: Optional[bool] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        assert "_traj_uid" in trajectories.non_tensor_batch, "Trajectories must have a unique id"
        assert len(trajectories) == len(states_list), "Trajectories and states list must have the same length"

        uid_to_traj = {}
        uid_to_states = {}
        for i, traj in enumerate(trajectories):
            uid_to_traj[traj.non_tensor_batch["_traj_uid"]] = traj
            uid_to_states[traj.non_tensor_batch["_traj_uid"]] = states_list[i]

        # Create MC queries
        with marked_timer(f"{timing_and_metric_prefix}create_queries", timing_raw):
            orig_mc_queries = self._create_mc_queries(trajectories=trajectories, states_list=states_list)
            unique_mc_queries, unique_to_orig_map = _deduplicate_mc_queries(orig_mc_queries)

        metrics.update(
            {
                f"{timing_and_metric_prefix}num_queries": len(orig_mc_queries),
                f"{timing_and_metric_prefix}num_unique_queries": len(unique_mc_queries),
            }
        )

        # Balance the MC queries based on vine cost
        with marked_timer(f"{timing_and_metric_prefix}balance_queries", timing_raw):
            # Reorder the mc queries on single controller such that each dp rank gets similar inference load
            seqlen_lst = [self._estimate_vine_cost(trajectories[i], states_list[i]) for i in range(len(trajectories))]
            world_size = self.actor_rollout_wg.world_size

            partition_lst = get_seqlen_balanced_partitions(seqlen_lst, k_partitions=world_size, equal_size=False)
            vine_balance_stats = log_seqlen_unbalance(seqlen_lst, partition_lst, f"{timing_and_metric_prefix}vine_cost")
            if metrics is not None:
                metrics.update(vine_balance_stats)

            # Assign each trajectory UID to its DP‑partition index
            traj_uid_to_partition = {
                trajectories[traj_idx].non_tensor_batch["_traj_uid"]: partition_idx
                for partition_idx, partition in enumerate(partition_lst)
                for traj_idx in partition
            }

            # Group Monte‑Carlo queries by partition
            partition_to_query_indices: dict[int, list] = defaultdict(list)
            for query_idx, query in enumerate(unique_mc_queries):
                partition = traj_uid_to_partition[query["traj_uid"]]
                partition_to_query_indices[partition].append(query_idx)

            # Within each partition sort by prompt (descending)
            def _query_key(q_idx: int) -> tuple[int, ...]:
                return tuple(
                    _pre_process_inputs(self.tokenizer.pad_token_id, unique_mc_queries[q_idx]["input_ids"]).tolist()
                )

            for indices in partition_to_query_indices.values():
                indices.sort(key=_query_key, reverse=True)

        # Attach uid to queries
        for q in unique_mc_queries:
            q["query_uid"] = str(uuid.uuid4())
        uid_to_query = {q["query_uid"]: q for q in unique_mc_queries}

        # Create gen_batch
        with marked_timer(f"{timing_and_metric_prefix}create_gen_batch", timing_raw):
            gen_batch = self._create_mc_gen_batch(unique_mc_queries, uid_to_traj)
            gen_batch.meta_info["concurrent_reward_computation"] = True
            gen_batch.meta_info["return_response_lengths"] = True

            if wake_up_engine_before_rollout is not None:
                gen_batch.meta_info["do_wake_up"] = wake_up_engine_before_rollout
            if sleep_engine_after_rollout is not None:
                gen_batch.meta_info["do_sleep"] = sleep_engine_after_rollout
                gen_batch.meta_info["free_engine_cache"] = sleep_engine_after_rollout

            gen_batch.meta_info["return_rewards_extra_info"] = False
            # To reduce communication overhead ask the rollout workers to only return necessary stuff
            gen_batch.meta_info["keys_to_keep"] = {
                "batch": ["__rewards", "responses"],
                "non_tensor_batch": ["query_uid"],
            }

        with marked_timer(f"{timing_and_metric_prefix}rollouts_gen", timing_raw):
            worker_to_indices = list(partition_to_query_indices.values())  # order doesn't matter here
            gen_batch_output = self.actor_rollout_wg.generate_sequences_with_manual_control(
                gen_batch, VERL_DISPATCH_INDICES=worker_to_indices
            )

        timing_raw.update(
            {
                f"{timing_and_metric_prefix}e2e_rollouts_gen_throughput": (
                    gen_batch_output.batch["response_lengths"].sum().item()
                    / timing_raw[f"{timing_and_metric_prefix}rollouts_gen"]
                ),
                **{f"{timing_and_metric_prefix}{k}": v for k, v in gen_batch_output.meta_info["timing"].items()},
            }
        )

        # Process the rollout results
        with marked_timer(f"{timing_and_metric_prefix}fill_state_values", timing_raw):
            for out in gen_batch_output:
                response_len = out.batch["response_lengths"]
                mc_return = out.batch["__rewards"][:response_len].sum().item()  # @TODO: Handle the discounting
                mc_rollout = out.batch["responses"][:response_len].cpu().numpy()

                query = uid_to_query[out.non_tensor_batch["query_uid"]]
                query.setdefault("__mc_returns", []).append(mc_return)
                query.setdefault("__mc_rollouts", []).append(mc_rollout)

            for query in unique_mc_queries:
                assert len(query["__mc_returns"]) == len(query["__mc_rollouts"]) == query["rollout_n"], (
                    "The number of rollouts should match the rollout n. "
                    f"Rollout n: {query['rollout_n']}, "
                    f"MC returns: {len(query['__mc_returns'])}, "
                    f"MC rollouts: {len(query['__mc_rollouts'])}"
                )
                query["__mc_returns"] = np.array(query["__mc_returns"])
                query["__mc_rollouts"] = np.array(query["__mc_rollouts"], dtype=object)
                query["__mc_value"] = np.mean(query["__mc_returns"])

            mc_queries = _unpack_unique_mc_results(unique_mc_queries, unique_to_orig_map, orig_mc_queries)

            traj_to_mc_values = defaultdict(dict)
            traj_to_mc_extra_info = defaultdict(dict)
            for query in mc_queries:
                traj_uid = query["traj_uid"]
                state = query["state"]
                traj_to_mc_values[traj_uid][state] = query["__mc_value"]
                traj_to_mc_extra_info[traj_uid][state] = {
                    k: v for k, v in query.items() if k.startswith("__mc_") and k != "__mc_value"
                }

            mc_values = np.array(
                [traj_to_mc_values[traj.non_tensor_batch["_traj_uid"]] for traj in trajectories], dtype=object
            )
            mc_extra_info = np.array(
                [traj_to_mc_extra_info[traj.non_tensor_batch["_traj_uid"]] for traj in trajectories], dtype=object
            )

            if metrics is not None:
                num_rollouts = np.array([query["rollout_n"] for query in unique_mc_queries])
                metrics.update(
                    {
                        f"{timing_and_metric_prefix}rollout_n/mean": np.mean(num_rollouts),
                        f"{timing_and_metric_prefix}rollout_n/max": np.max(num_rollouts),
                        f"{timing_and_metric_prefix}rollout_n/min": np.min(num_rollouts),
                        f"{timing_and_metric_prefix}rollout_n/dist": num_rollouts,
                    }
                )

        return mc_values, mc_extra_info

    def _create_mc_queries(self, trajectories: DataProto, states_list: np.ndarray) -> list[dict[str, Any]]:
        pad_token_id = self.tokenizer.pad_token_id

        prompt_len = trajectories.batch["prompts"].shape[-1]
        queries = []

        for i, traj in enumerate(trajectories):
            non_pad_prompt_ids = _pre_process_inputs(pad_token_id, traj.batch["prompts"])
            states = states_list[i]  # List of ints

            for state in states:
                state_length = state + prompt_len
                if self.vineppo_config.use_state_adjusted_max_new_tokens:
                    max_new_tokens = self.max_response_length - state
                else:
                    max_new_tokens = self.max_response_length
                queries.append(
                    {
                        "traj_uid": traj.non_tensor_batch["_traj_uid"],
                        "state": state,
                        "input_ids": traj.batch["input_ids"][:state_length],
                        "attention_mask": traj.batch["attention_mask"][:state_length],
                        "position_ids": traj.batch["position_ids"][:state_length],
                        "max_new_tokens": max_new_tokens,
                        "max_seq_len": len(non_pad_prompt_ids) + max_new_tokens,
                        "orig_prompt_len": len(non_pad_prompt_ids),
                        "rollout_n": self.vineppo_config.vineppo_k,
                    }
                )

        return queries

    def _create_mc_gen_batch(
        self, mc_queries: list[dict[str, Any]], uid_to_traj: dict[str, DataProtoItem]
    ) -> DataProto:
        batch = {
            "input_ids": [],
            "attention_mask": [],
            "position_ids": [],
            "max_new_tokens": [],
            "rollout_n": [],
        }
        non_batch = {"query_uid": [], "orig_prompt_len": []}

        keys_to_ignore_from_trajectory = {
            *batch.keys(),
            *non_batch.keys(),
            "responses",
            "response_mask",
            "response_lengths",
            "prompt_ids",
            "prompts",
        }

        for query in mc_queries:
            batch["input_ids"].append(query["input_ids"])
            batch["attention_mask"].append(query["attention_mask"])
            batch["position_ids"].append(query["position_ids"])
            batch["max_new_tokens"].append(query["max_new_tokens"])
            batch["rollout_n"].append(query["rollout_n"])
            non_batch["query_uid"].append(query["query_uid"])
            non_batch["orig_prompt_len"].append(query["orig_prompt_len"])

            # Copy the rest of stuff that might be needed for concurrent reward computation
            traj = uid_to_traj[query["traj_uid"]]
            for key in traj.batch.keys():
                if key not in keys_to_ignore_from_trajectory:
                    batch.setdefault(key, []).append(traj.batch[key])

            for key in traj.non_tensor_batch.keys():
                if key not in keys_to_ignore_from_trajectory:
                    non_batch.setdefault(key, []).append(traj.non_tensor_batch[key])

        meta_info = {
            "global_steps": self.global_steps,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        # Convert batch to tensors
        max_len = max(len(input_ids) for input_ids in batch["input_ids"])
        for k, tensors in batch.items():
            if k in ["max_new_tokens", "rollout_n"]:
                batch[k] = torch.tensor(tensors, device=batch["input_ids"].device)
            else:
                batch[k] = pad_tensor_list(
                    tensors,
                    padding_value=self.tokenizer.pad_token_id if k == "input_ids" else 0,
                    mode="left",
                    max_seq_length=max_len,
                )

        non_batch = {
            k: np.array(v, dtype=None if isinstance(v[0], np.ndarray | float | int) else object)
            for k, v in non_batch.items()
        }

        data = DataProto.from_dict(tensors=batch, non_tensors=non_batch, meta_info=meta_info, auto_padding=False)

        return data

    def _estimate_vine_cost(self, trajectory: DataProtoItem, states: list[int]) -> float:
        max_traj_len = trajectory.batch["response_mask"].sum(-1)
        if self.vineppo_config.use_max_response_len_for_vine_cost:
            max_traj_len = self.max_response_length

        total_cost = 0.0
        for state in states:
            total_cost += self.vineppo_config.prefill_cost * state
            total_cost += (
                self.vineppo_config.generation_cost * self.vineppo_config.vineppo_k * (max_traj_len - state) ** 2
                # Generation cost is quadratic in the remaining length of the trajectory
            )

        return total_cost

from collections import defaultdict

import torch

from verl import DataProto
from verl.protocol import DataProtoItem
from verl.workers.reward_manager import register
from verl.workers.reward_manager.naive import NaiveRewardManager


def maybe_recover_orig_prompt_and_response(
    data_item: DataProtoItem, valid_prompt_ids: torch.Tensor, valid_response_ids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Separate original prompt from partial response that may be included in the prompt.

    This method handles cases where the prompt contains partial response content.
    It uses the 'orig_prompt_len' information to extract the true original prompt
    and reconstruct the complete response by combining the partial response from
    the prompt with the actual response tokens.

    Args:
        data_item: A DataProtoItem containing batch data and metadata.
        valid_prompt_ids: Tensor containing the prompt token IDs (may include partial response).
        valid_response_ids: Tensor containing the response token IDs.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - orig_prompt_ids: The original prompt token IDs (without partial response)
            - orig_response_ids: The complete response token IDs (partial + actual response)

    Note:
        If 'orig_prompt_len' is not present in the batch data, the method returns
        the input valid_prompt_ids and valid_response_ids unchanged, assuming no
        partial response is included in the prompt.
    """

    orig_prompt_len = data_item.non_tensor_batch.get("orig_prompt_len", None)
    if orig_prompt_len is None:
        return valid_prompt_ids, valid_response_ids

    orig_prompt_ids = valid_prompt_ids[:orig_prompt_len]
    orig_response_ids = valid_prompt_ids[orig_prompt_len:]
    orig_response_ids = torch.cat([orig_response_ids, valid_response_ids], dim=0)
    return orig_prompt_ids, orig_response_ids


@register("vineppo")
class VineppoRewardManager(NaiveRewardManager):
    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            valid_prompt_ids, valid_response_ids = maybe_recover_orig_prompt_and_response(
                data_item, valid_prompt_ids, valid_response_ids
            )

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
            extra_info["num_turns"] = num_turns

            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )

            if isinstance(score, dict):
                reward = score["score"]
                # Store the information including original reward
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score

            reward_tensor[i, valid_response_length - 1] = reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor

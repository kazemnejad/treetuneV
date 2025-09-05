import logging
import os
from typing import Optional

import numpy as np
import torch

from verl import DataProto

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


SOLUTION_SPLITTER_REGISTRY = {}


def register(name):
    """Decorator to register a solution splitter class with a given name.

    Args:
        name: `(str)`
            The name of the solution splitter.
    """

    def decorator(cls):
        if name in SOLUTION_SPLITTER_REGISTRY and SOLUTION_SPLITTER_REGISTRY[name] != cls:
            raise ValueError(
                f"Solution splitter {name} has already been registered: {SOLUTION_SPLITTER_REGISTRY[name]} vs {cls}"
            )
        SOLUTION_SPLITTER_REGISTRY[name] = cls
        return cls

    return decorator


def get_solution_splitter_cls(name):
    """Get the solution splitter class with a given name.

    Args:
        name: `(str)`
            The name of the solution splitter.

    Returns:
        `(type)`: The solution splitter class.
    """
    if name not in SOLUTION_SPLITTER_REGISTRY:
        raise ValueError(f"Unknown solution splitter: {name}")
    return SOLUTION_SPLITTER_REGISTRY[name]


class SolutionSplitter:
    def __init__(self, tokenizer, merge_every_k_states: int = 1, max_num_states: Optional[int] = None):
        self.tokenizer = tokenizer
        self.merge_every_k_states = merge_every_k_states
        self.max_num_states = max_num_states
        assert self.merge_every_k_states > 0, "merge_every_k_states must be greater than 0"
        assert self.max_num_states is None or self.max_num_states > 0, "max_num_states must be greater than 0"

    def __call__(self, data: DataProto) -> np.ndarray:
        raise NotImplementedError

    def maybe_merge_and_truncate_states(self, states: np.ndarray) -> np.ndarray:
        states = states[:: self.merge_every_k_states]
        if self.max_num_states is not None:
            states = states[: self.max_num_states]
        return states


@register("every_n_token")
class EveryNTokenSolutionSplitter(SolutionSplitter):
    def __init__(self, n: int, **kwargs):
        super().__init__(**kwargs)
        self.n = n

    def __call__(self, trajectories: DataProto) -> np.ndarray:
        states_list: list[np.ndarray] = []

        for traj in trajectories:
            response_length = traj.batch["response_mask"].sum(-1)
            assert response_length > 0, "Response length must be greater than 0"
            states = np.arange(0, response_length, self.n, dtype=np.int32)
            states = self.maybe_merge_and_truncate_states(states)
            states_list.append(states)

        states_list = np.array(states_list, dtype=object)

        return states_list


@register("double_new_line_tokens")
class DoubleNewLineSolutionSplitter(SolutionSplitter):
    def __init__(
        self,
        double_new_line_str: str = "ĊĊ",
        enclosing_think_tag_str: Optional[str] = None,
        only_split_thinking: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.only_split_thinking = only_split_thinking

        double_new_line_tokens = []
        for tok_str in self.tokenizer.get_vocab():
            if double_new_line_str in tok_str:
                double_new_line_tokens.append(tok_str)
        assert len(double_new_line_tokens) > 0, "No double new line tokens found"
        logger.info(f"Found {len(double_new_line_tokens)} double new line tokens")
        double_new_line_token_ids = [
            [self.tokenizer.convert_tokens_to_ids(tok_str)] for tok_str in double_new_line_tokens
        ]

        self._max_delimiter_len = max(len(tok_ids) for tok_ids in double_new_line_token_ids)
        assert self._max_delimiter_len > 0, "Max delimiter length must be greater than 0"

        self._delimiters_hash_map = {
            tuple(tok_ids + [-1] * (self._max_delimiter_len - len(tok_ids))): len(tok_ids)
            for tok_ids in double_new_line_token_ids
        }
        assert len(self._delimiters_hash_map) == len(double_new_line_token_ids), "Duplicate delimiter found"

        self.think_tag_token_ids = None
        if enclosing_think_tag_str is not None:
            self.think_tag_token_ids = np.array(
                self.tokenizer.encode(enclosing_think_tag_str, add_special_tokens=False)
            )
            logger.info(f"Think tag token ids: {self.tokenizer.convert_ids_to_tokens(self.think_tag_token_ids)}")

    def __call__(self, trajectories: DataProto) -> np.ndarray:
        states_list: list[np.ndarray] = []
        response_lengths = trajectories.batch["response_mask"].sum(-1)

        for i, traj in enumerate(trajectories):
            response_length = response_lengths[i]
            response_token_ids = traj.batch["responses"][:response_length]

            if self.only_split_thinking:
                end_of_think = self._find_end_of_think(response_token_ids)
                if end_of_think is not None:
                    # We only split the thinking part
                    response_token_ids = response_token_ids[:end_of_think]

            states = self._split_to_double_new_line(response_token_ids)
            states = self.maybe_merge_and_truncate_states(states)
            states_list.append(states)

        states_list = np.array(states_list, dtype=object)

        return states_list

    def _split_to_double_new_line(self, response_token_ids: torch.Tensor) -> np.ndarray:
        response_token_ids = response_token_ids.tolist()

        states = [0]
        i = 0
        while i < len(response_token_ids):
            segment = tuple([response_token_ids[i]] + [-1] * (self._max_delimiter_len - 1))
            if segment in self._delimiters_hash_map:
                delimiter_len = self._delimiters_hash_map[segment]
                states.append(i + delimiter_len)
                i += delimiter_len
            else:
                i += 1

        return np.array(states)

    def _find_end_of_think(self, response_token_ids: torch.Tensor) -> Optional[int]:
        if self.think_tag_token_ids is None:
            return None

        response_token_ids = response_token_ids.cpu().numpy()
        delimiter = self.think_tag_token_ids
        delimiter_len = len(delimiter)

        # Find the first occurrence of the delimiter
        for i in range(len(response_token_ids) - delimiter_len + 1):
            if np.array_equal(response_token_ids[i : i + delimiter_len], delimiter):
                return i + delimiter_len

        return None

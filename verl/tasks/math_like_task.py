import re
from dataclasses import dataclass
from typing import Callable, Optional

from datasets import Dataset

from verl.tasks.registry import register
from verl.tasks.task import Split, Task, TaskConfig


@dataclass
class MathLikeTaskConfig(TaskConfig):
    problem_key: str = "problem"
    answer_key: str = "answer"
    filter_empty_answers: bool = True


@register("math_like_task")
class MathLikeTask(Task):
    def build_dataset(self, split: Split) -> Dataset:
        dataset = self._load_dataset_from_config(split)
        dataset = self._sample_dataset(dataset)
        data_source = self.config.data_source or "math_like_task"
        dataset = dataset.map(
            function=self.make_map_fn(split),
            with_indices=True,
            num_proc=self.config.num_dataset_workers,
            desc=f"Building {split} dataset for {data_source}",
        )
        filter_fn = self.make_filter_fn(split)
        if filter_fn is not None:
            dataset = dataset.filter(
                filter_fn,
                num_proc=self.config.num_dataset_workers,
                desc=f"Filtering out invalid items from {data_source}",
            )
        dataset = self._remove_useless_columns(dataset)
        return dataset

    def make_map_fn(self, split: Split) -> Callable[[dict, int], dict]:
        prompt_template = self.config.prompt_template or "{}"
        system_prompt = self.config.system_prompt

        problem_key = self.config.problem_key
        answer_key = self.config.answer_key

        data_source = self.config.data_source or "math_like_task"

        if split in [Split.VALIDATION, Split.TEST]:
            val_sampling_params = self.config.val_sampling_params or {}
            val_sampling_params = {f"val_sampling_params.{k}": v for k, v in val_sampling_params.items()}
        else:
            val_sampling_params = {}

        def process_fn(example, idx):
            problem = example[problem_key]
            answer = example[answer_key]

            if not isinstance(answer, str):
                answer = str(answer)

            # Enclose the answer in brackets if it is not already to support latex parsing
            enclosed_with_brackets = "\\[" in answer and "\\]" in answer  # This is a safe way of checking
            enclosed_with_dollars = answer.startswith("$") and answer.endswith("$")
            if not enclosed_with_brackets and not enclosed_with_dollars:
                answer = f"\\[{answer}\\]"

            prompt_messages = []
            if system_prompt:
                prompt_messages.append({"role": "system", "content": system_prompt})
            prompt_messages.append({"role": "user", "content": prompt_template.format(problem)})

            data = {
                "data_source": data_source,
                "prompt": prompt_messages,
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer,
                    "question": problem,
                    "_orig_answer": str(example[answer_key]),
                },
                **val_sampling_params,
            }
            return data

        return process_fn

    def make_filter_fn(self, split: Split) -> Optional[Callable[[dict], bool]]:
        def _keep_non_empty_answers(x):
            return x["reward_model"]["ground_truth"] is not None and len(x["reward_model"]["ground_truth"]) > 0

        if self.config.filter_empty_answers:
            return _keep_non_empty_answers
        else:
            return None

import re
from dataclasses import dataclass

from datasets import Dataset

from verl.tasks.registry import register
from verl.tasks.task import Split, Task, TaskConfig


@dataclass
class GSM8KConfig(TaskConfig):
    use_mathlike_answer_format: bool = False


@register("gsm8k")
class GSM8K(Task):
    def build_dataset(self, split: Split) -> Dataset:
        dataset = self._load_dataset_from_config(split)
        dataset = self._sample_dataset(dataset)
        dataset = dataset.map(
            function=self.make_map_fn(split),
            with_indices=True,
            num_proc=self.config.num_dataset_workers,
            desc=f"Building {split} dataset",
        )
        return dataset

    def make_map_fn(self, split: Split):
        use_mathlike_answer_format = self.config.use_mathlike_answer_format
        if use_mathlike_answer_format:
            instruction_following = "Let's think step by step and output the final answer within \\boxed{...}."
        else:
            instruction_following = "Let's think step by step and output the final answer after '####'."

        sol_regex_pattern = re.compile("#### (\\-?[0-9\\.\\,]+)")

        def extract_solution(solution_str: str) -> str:
            solution = sol_regex_pattern.search(solution_str)
            assert solution is not None
            final_solution = solution.group(0)
            final_solution = final_solution.split("#### ")[1].replace(",", "")
            return final_solution

        def process_fn(example, idx):
            question_raw = example.pop("question")

            question = question_raw + " " + instruction_following

            answer_raw = example.pop("answer")
            solution = extract_solution(answer_raw)
            if use_mathlike_answer_format:
                solution = f"\\[{solution}\\]"

            data = {
                "data_source": "openai/gsm8k",
                "prompt": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer_raw,
                    "question": question_raw,
                },
            }
            return data

        return process_fn

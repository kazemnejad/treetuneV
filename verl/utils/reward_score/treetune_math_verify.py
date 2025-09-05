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

import functools
import logging
import os

try:
    from math_verify import LatexExtractionConfig, LatexNormalizationConfig, parse, verify
except ImportError:
    print("To use Math-Verify, please install it first by running `pip install math-verify`.")

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


parse_gold_answer_fn = functools.partial(
    parse,
    extraction_config=[
        LatexExtractionConfig(
            LatexNormalizationConfig(
                basic_latex=True,
                boxed="all",
                malformed_operators=False,
                units=True,
            ),
            boxed_match_priority=0,
        )
    ],
    fallback_mode="no_fallback",
    extraction_mode=["first_match"],
)

parse_predicted_answer_fn = functools.partial(
    parse,
    extraction_config=[
        LatexExtractionConfig(
            LatexNormalizationConfig(
                basic_latex=True,
                boxed="all",
                malformed_operators=False,
                units=True,
            ),
            boxed_match_priority=0,
        )
    ],
    parsing_timeout=1,
    fallback_mode="no_fallback",
    extraction_mode=["first_match"],
)


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict = None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
    enclosing_think_tag_str: str = "</think>",
    extract_from_answer_only: bool = True,
    failure_score: float = 0.0,
    success_score: float = 1.0,
    timeout_seconds: int = 2,
) -> float:
    if extract_from_answer_only:
        if enclosing_think_tag_str not in solution_str:
            return failure_score
        parts = solution_str.split(enclosing_think_tag_str, 1)
        assert len(parts) == 2, f"Expected 2 parts, but got {len(parts)}: {parts}"
        solution_str = parts[1]

    if isinstance(ground_truth, str):
        ground_truth = parse_gold_answer_fn(ground_truth)

    try:
        predicted_answer = parse_predicted_answer_fn(solution_str)
    except Exception as e:
        logger.warning(f"Error parsing predicted answer {solution_str}: {e}")
        return failure_score

    try:
        is_correct = verify(
            gold=ground_truth,
            target=predicted_answer,
            timeout_seconds=timeout_seconds,
        )

        if is_correct:
            return success_score
        else:
            return failure_score
    except Exception as e:
        logger.warning(f"Error verifying answer {solution_str}: {e}")
        return failure_score

import asyncio
import logging
from contextlib import nullcontext
from copy import copy
from typing import Optional

import fastapi
from sglang.srt.managers.io_struct import EmbeddingReqInput, GenerateReqInput
from sglang.srt.managers.scheduler_input_blocker import input_blocker_guard_region
from sglang.srt.managers.tokenizer_manager import TokenizerManager as SGLangTokenizerManager
from sglang.srt.utils import get_bool_env_var

logger = logging.getLogger(__name__)


class TokenizerManager(SGLangTokenizerManager):
    async def _handle_batch_request(
        self,
        obj: GenerateReqInput | EmbeddingReqInput,
        request: Optional[fastapi.Request] = None,
        created_time: Optional[float] = None,
    ):
        batch_size = obj.batch_size

        generators = []
        rids = []
        if getattr(obj, "parallel_sample_num", 1) == 1:
            if self.server_args.enable_tokenizer_batch_encode:
                # Validate batch tokenization constraints
                self._validate_batch_tokenization_constraints(batch_size, obj)

                tokenized_objs = await self._batch_tokenize_and_process(batch_size, obj)

                for i, tokenized_obj in enumerate(tokenized_objs):
                    tmp_obj = obj[i]
                    state = self._send_one_request(tmp_obj, tokenized_obj, created_time)
                    generators.append(self._wait_one_response(tmp_obj, state, request))
                    rids.append(tmp_obj.rid)
            else:
                # Sequential tokenization and processing
                with (
                    input_blocker_guard_region(send_to_scheduler=self.send_to_scheduler)
                    if get_bool_env_var("SGLANG_ENABLE_COLOCATED_BATCH_GEN")
                    else nullcontext()
                ):
                    for i in range(batch_size):
                        tmp_obj = obj[i]
                        tokenized_obj = await self._tokenize_one_request(tmp_obj)
                        state = self._send_one_request(tmp_obj, tokenized_obj, created_time)
                        generators.append(self._wait_one_response(tmp_obj, state, request))
                        rids.append(tmp_obj.rid)
        else:
            # FIXME: When using batch and parallel_sample_num together, the perf is not optimal.
            if batch_size > 128:
                logger.warning(
                    "Sending a single large batch with parallel sampling (n > 1) has not been well optimized. "
                    "The performance might be better if you just duplicate the requests n times or use "
                    "many threads to send them one by one with parallel sampling (n > 1)."
                )

            # Tokenize all requests
            objs = [obj[i] for i in range(batch_size)]
            tokenized_objs = await asyncio.gather(*(self._tokenize_one_request(obj) for obj in objs))

            # Cache the common prefix for parallel sampling
            for i in range(batch_size):
                tmp_obj = copy.copy(objs[i])
                tokenized_obj = copy.copy(tokenized_objs[i])
                tokenized_obj.rid = tmp_obj.regenerate_rid()
                tokenized_obj.sampling_params = copy.copy(tokenized_obj.sampling_params)
                tokenized_obj.sampling_params.max_new_tokens = 0
                tokenized_obj.stream = False
                state = self._send_one_request(tmp_obj, tokenized_obj, created_time)
                await self._wait_one_response(tmp_obj, state, request).__anext__()

            # Expand requests, assign new rids for them, and send them
            for i in range(batch_size):
                for _ in range(obj.parallel_sample_num):
                    tmp_obj = copy.copy(objs[i])
                    tokenized_obj = copy.copy(tokenized_objs[i])
                    tokenized_obj.rid = tmp_obj.regenerate_rid()
                    state = self._send_one_request(tmp_obj, tokenized_obj, created_time)
                    generators.append(self._wait_one_response(tmp_obj, state, request))
                    rids.append(tmp_obj.rid)

        # Wait for all requests
        is_stream = hasattr(obj, "stream") and obj.stream
        is_stream_once_complete = hasattr(obj, "stream_once_complete") and obj.stream_once_complete
        if not is_stream and not is_stream_once_complete:
            outputs = await asyncio.gather(*(gen.__anext__() for gen in generators))
            yield outputs
        elif is_stream_once_complete:
            rid_to_index = {rid: i for i, rid in enumerate(rids)}
            tasks = [asyncio.create_task(gen.__anext__()) for gen in generators]

            # Process all tasks as they complete
            for completed_task in asyncio.as_completed(tasks):
                result = await completed_task
                result["index"] = rid_to_index[result["meta_info"]["id"]]
                yield result
        else:
            rid_to_index = {rid: i for i, rid in enumerate(rids)}
            task_map = {asyncio.create_task(gen.__anext__()): gen for gen in generators}
            while task_map:
                done, _ = await asyncio.wait(task_map.keys(), return_when=asyncio.FIRST_COMPLETED)

                for task in done:
                    gen = task_map.pop(task)
                    try:
                        result = task.result()
                        result["index"] = rid_to_index[result["meta_info"]["id"]]
                        yield result
                        new_task = asyncio.create_task(gen.__anext__())
                        task_map[new_task] = gen
                    except StopAsyncIteration:
                        pass

    def detokenize_logprob_tokens(
        self,
        token_logprobs_val: list[float],
        token_logprobs_idx: list[int],
        decode_to_text: bool,
    ):
        def clamp_neg_inf(val):
            return -999999999.0 if val == float("-inf") else val

        if not decode_to_text:
            return [
                (clamp_neg_inf(logprob), token_id, None)
                for logprob, token_id in zip(token_logprobs_val, token_logprobs_idx)
            ]
        else:
            assert self.tokenizer is not None
            token_texts = self.tokenizer.batch_decode(token_logprobs_idx)
            return list(
                zip(
                    list(map(clamp_neg_inf, token_logprobs_val)),
                    token_logprobs_idx,
                    token_texts,
                )
            )

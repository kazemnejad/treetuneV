import logging
import multiprocessing as mp
import os
from typing import AsyncIterator, Optional

import sglang.srt.entrypoints.engine as engine_module
from sglang.srt.managers.data_parallel_controller import (
    run_data_parallel_controller_process,
)
from sglang.srt.managers.detokenizer_manager import run_detokenizer_process
from sglang.srt.managers.io_struct import MultimodalDataInputFormat
from sglang.srt.managers.scheduler import run_scheduler_process
from sglang.srt.managers.template_manager import TemplateManager
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.torch_memory_saver_adapter import TorchMemorySaverAdapter
from sglang.srt.utils import (
    configure_logger,
    launch_dummy_health_check_server,
    prepare_model_and_tokenizer,
)

from verl.third_party.sglang.io_struct import GenerateReqInput
from verl.third_party.sglang.tokenizer_manager import TokenizerManager

logger = logging.getLogger(__name__)


class Engine(engine_module.Engine):
    async def async_generate(
        self,
        # The input prompt. It can be a single prompt or a batch of prompts.
        prompt: Optional[list[str] | str] = None,
        sampling_params: Optional[list[dict] | dict] = None,
        # The token ids for text; one can either specify text or input_ids.
        input_ids: Optional[list[list[int]] | list[int]] = None,
        # The image input. It can be an image instance, file name, URL, or base64 encoded string.
        # Can be formatted as:
        # - Single image for a single request
        # - List of images (one per request in a batch)
        # - List of lists of images (multiple images per request)
        # See also python/sglang/srt/utils.py:load_image for more details.
        image_data: Optional[MultimodalDataInputFormat] = None,
        return_logprob: Optional[list[bool] | bool] = False,
        logprob_start_len: Optional[list[int] | int] = None,
        top_logprobs_num: Optional[list[int] | int] = None,
        token_ids_logprob: Optional[list[list[int]] | list[int]] = None,
        lora_path: Optional[list[Optional[str]]] = None,
        custom_logit_processor: Optional[list[str] | str] = None,
        stream: bool = False,
        stream_once_complete: bool = False,
        bootstrap_host: Optional[list[str] | str] = None,
        bootstrap_port: Optional[list[int] | int] = None,
        bootstrap_room: Optional[list[int] | int] = None,
    ) -> dict | AsyncIterator[dict]:
        """
        The arguments of this function is the same as `sglang/srt/managers/io_struct.py::GenerateReqInput`.
        Please refer to `GenerateReqInput` for the documentation.
        """
        obj = GenerateReqInput(
            text=prompt,
            input_ids=input_ids,
            sampling_params=sampling_params,
            image_data=image_data,
            return_logprob=return_logprob,
            logprob_start_len=logprob_start_len,
            top_logprobs_num=top_logprobs_num,
            token_ids_logprob=token_ids_logprob,
            lora_path=lora_path,
            stream=stream,
            stream_once_complete=stream_once_complete,
            custom_logit_processor=custom_logit_processor,
            bootstrap_host=bootstrap_host,
            bootstrap_port=bootstrap_port,
            bootstrap_room=bootstrap_room,
        )
        generator = self.tokenizer_manager.generate_request(obj, None)

        if stream is True or stream_once_complete is True:
            return generator
        else:
            return await generator.__anext__()


def _launch_subprocesses(
    server_args: ServerArgs, port_args: Optional[PortArgs] = None
) -> tuple[TokenizerManager, TemplateManager, dict]:
    """
    Launch the TokenizerManager in the main process, the Scheduler in a subprocess, and the DetokenizerManager in another subprocess.
    """
    # Configure global environment
    configure_logger(server_args)
    server_args.check_server_args()
    engine_module._set_envs_and_config(server_args)

    # Allocate ports for inter-process communications
    if port_args is None:
        port_args = PortArgs.init_new(server_args)
        logger.info(f"{server_args=}")

    # If using model from www.modelscope.cn, first download the model.
    server_args.model_path, server_args.tokenizer_path = prepare_model_and_tokenizer(
        server_args.model_path, server_args.tokenizer_path
    )

    scheduler_procs = []
    if server_args.dp_size == 1:
        memory_saver_adapter = TorchMemorySaverAdapter.create(enable=server_args.enable_memory_saver)
        scheduler_pipe_readers = []

        nnodes_per_tp_group = max(server_args.nnodes // server_args.pp_size, 1)
        tp_size_per_node = server_args.tp_size // nnodes_per_tp_group
        tp_rank_range = range(
            tp_size_per_node * (server_args.node_rank % nnodes_per_tp_group),
            tp_size_per_node * (server_args.node_rank % nnodes_per_tp_group + 1),
        )

        pp_size_per_node = max(server_args.pp_size // server_args.nnodes, 1)
        pp_rank_range = range(
            pp_size_per_node * (server_args.node_rank // nnodes_per_tp_group),
            pp_size_per_node * (server_args.node_rank // nnodes_per_tp_group + 1),
        )

        for pp_rank in pp_rank_range:
            for tp_rank in tp_rank_range:
                reader, writer = mp.Pipe(duplex=False)
                gpu_id = (
                    server_args.base_gpu_id
                    + ((pp_rank % pp_size_per_node) * tp_size_per_node)
                    + (tp_rank % tp_size_per_node) * server_args.gpu_id_step
                )
                proc = mp.Process(
                    target=run_scheduler_process,
                    args=(
                        server_args,
                        port_args,
                        gpu_id,
                        tp_rank,
                        pp_rank,
                        None,
                        writer,
                    ),
                )

                with memory_saver_adapter.configure_subprocess():
                    proc.start()
                scheduler_procs.append(proc)
                scheduler_pipe_readers.append(reader)
    else:
        # Launch the data parallel controller
        reader, writer = mp.Pipe(duplex=False)
        scheduler_pipe_readers = [reader]
        proc = mp.Process(
            target=run_data_parallel_controller_process,
            args=(server_args, port_args, writer),
        )
        proc.start()
        scheduler_procs.append(proc)

    if server_args.node_rank >= 1:
        # In multi-node cases, non-zero rank nodes do not need to run tokenizer or detokenizer,
        # so they can just wait here.

        for reader in scheduler_pipe_readers:
            data = reader.recv()
            assert data["status"] == "ready"

        if os.getenv("SGLANG_BLOCK_NONZERO_RANK_CHILDREN") == "0":
            # When using `Engine` as a Python API, we don't want to block here.
            return None, None, None

        launch_dummy_health_check_server(server_args.host, server_args.port, server_args.enable_metrics)

        for proc in scheduler_procs:
            proc.join()
            logger.error(f"Scheduler or DataParallelController {proc.pid} terminated with {proc.exitcode}")
        return None, None, None

    # Launch detokenizer process
    detoken_proc = mp.Process(
        target=run_detokenizer_process,
        args=(
            server_args,
            port_args,
        ),
    )
    detoken_proc.start()

    # Launch tokenizer process
    tokenizer_manager = TokenizerManager(server_args, port_args)

    # Initialize templates
    template_manager = TemplateManager()
    template_manager.initialize_templates(
        tokenizer_manager=tokenizer_manager,
        model_path=server_args.model_path,
        chat_template=server_args.chat_template,
        completion_template=server_args.completion_template,
    )

    # Wait for the model to finish loading
    scheduler_infos = []
    for i in range(len(scheduler_pipe_readers)):
        try:
            data = scheduler_pipe_readers[i].recv()
        except EOFError:
            logger.error(f"Rank {i} scheduler is dead. Please check if there are relevant logs.")
            scheduler_procs[i].join()
            logger.error(f"Exit code: {scheduler_procs[i].exitcode}")
            raise

        if data["status"] != "ready":
            raise RuntimeError("Initialization failed. Please see the error messages above.")
        scheduler_infos.append(data)

    # Assume all schedulers have the same scheduler_info
    scheduler_info = scheduler_infos[0]
    tokenizer_manager.max_req_input_len = scheduler_info["max_req_input_len"]
    return tokenizer_manager, template_manager, scheduler_info


engine_module._launch_subprocesses = _launch_subprocesses

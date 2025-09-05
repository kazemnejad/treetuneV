from dataclasses import dataclass

from sglang.srt.managers.io_struct import GenerateReqInput as SGLangGenerateReqInput


@dataclass
class GenerateReqInput(SGLangGenerateReqInput):
    # Whether to add special tokens in tokenization
    add_special_tokens_in_tokenization: bool = True
    # Whether to stream output once complete (useful for batch requests)
    stream_once_complete: bool = False

    def __getitem__(self, i):
        return GenerateReqInput(
            text=self.text[i] if self.text is not None else None,
            input_ids=self.input_ids[i] if self.input_ids is not None else None,
            image_data=self.image_data[i],
            audio_data=self.audio_data[i],
            sampling_params=self.sampling_params[i],
            rid=self.rid[i],
            return_logprob=self.return_logprob[i],
            logprob_start_len=self.logprob_start_len[i],
            top_logprobs_num=self.top_logprobs_num[i],
            token_ids_logprob=self.token_ids_logprob[i],
            return_text_in_logprobs=self.return_text_in_logprobs,
            stream=self.stream,
            stream_once_complete=self.stream_once_complete,
            log_metrics=self.log_metrics,
            modalities=self.modalities[i] if self.modalities else None,
            lora_path=self.lora_path[i] if self.lora_path is not None else None,
            add_special_tokens_in_tokenization=self.add_special_tokens_in_tokenization,
            custom_logit_processor=(
                self.custom_logit_processor[i] if self.custom_logit_processor is not None else None
            ),
            return_hidden_states=self.return_hidden_states,
            # if `__getitem__` is called, the bootstrap_host, bootstrap_port, bootstrap_room must be a list
            bootstrap_host=(self.bootstrap_host[i] if self.bootstrap_host is not None else None),
            bootstrap_port=(self.bootstrap_port[i] if self.bootstrap_port is not None else None),
            bootstrap_room=(self.bootstrap_room[i] if self.bootstrap_room is not None else None),
        )

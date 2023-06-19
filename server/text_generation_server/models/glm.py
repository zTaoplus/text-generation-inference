from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch
from opentelemetry import trace
from text_generation_server.models import Model
from text_generation_server.models.types import Batch, GeneratedText, Generation
from text_generation_server.pb import generate_pb2
from text_generation_server.utils import NextTokenChooser, StoppingCriteria
from torch import nn
from torch.nn import Module
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
)
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.logits_process import LogitsProcessor
from transformers.generation.utils import GenerationMixin, LogitsProcessorList

tracer = trace.get_tracer(__name__)


class InvalidScoreLogitsProcessor(LogitsProcessor):
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            scores.zero_()
            scores[..., 5] = 5e4
        return scores


class ChatGLMNextTokenChooser(NextTokenChooser):
    def __init__(
        self,
        # input_ids_seq_length:int,
        watermark=False,
        temperature=1,
        repetition_penalty=1,
        top_k=None,
        top_p=None,
        typical_p=None,
        do_sample=False,
        seed=0,
        device="cpu",
    ):
        generation = GenerationMixin()
        self.device = device
        self.gcfg: GenerationConfig = GenerationConfig(
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            top_k=top_k,
            top_p=top_p,
            typical_p=typical_p,
            do_sample=do_sample,
        )
        logits_processor = LogitsProcessorList()
        logits_processor.append(InvalidScoreLogitsProcessor())
        self.logits_processor = generation._get_logits_processor(
            generation_config=self.gcfg,
            input_ids_seq_length=None,  # mintokens, can disable ?
            encoder_input_ids=None,  # encoder_repetition_penalty , chatglm can not use this.
            prefix_allowed_tokens_fn=None,
            logits_processor=logits_processor,
        )

        self.logits_warper = generation._get_logits_warper(self.gcfg)

    def __call__(
        self,
        input_ids,
        next_token_logits,
    ):
        next_token_scores = self.logits_processor(input_ids, next_token_logits)
        next_token_scores = self.logits_warper(input_ids, next_token_scores)

        probs = nn.functional.softmax(next_token_scores, dim=-1)
        if self.gcfg.do_sample:
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_tokens = torch.argmax(probs, dim=-1)
        return next_tokens, probs

    @classmethod
    def from_pb(
        cls,
        pb: generate_pb2.NextTokenChooserParameters,
        device: torch.device,
        # input_ids_seq_length:int
    ) -> "ChatGLMNextTokenChooser":
        return ChatGLMNextTokenChooser(
            # input_ids_seq_length,
            watermark=pb.watermark,
            temperature=pb.temperature,
            repetition_penalty=pb.repetition_penalty,
            top_k=pb.top_k,
            top_p=pb.top_p,
            typical_p=pb.typical_p,
            do_sample=pb.do_sample,
            seed=pb.seed,
            device=device,
        )


@dataclass
class ChatGLMBatch(Batch):
    batch_id: int
    requests: List[generate_pb2.Request]
    requests_idx_mapping: Dict[int, int]

    # Encoder values
    input_ids: torch.Tensor

    # ADD paading lengths, 可以通过pad token的长度 配合max lengths 将可能多余出来的pad token 提出在当前batch中. 减少输入的input ids
    # padding_lengths: List[int]

    # Lengths of all generations present in the batch
    input_lengths: List[int]

    prefix_offsets: List[int]
    read_offsets: List[int]

    # Generation helpers
    next_token_choosers: List[ChatGLMNextTokenChooser]
    stopping_criterias: List[StoppingCriteria]

    # Metadata used for padding
    max_input_length: int

    # Maximum number of tokens this batch will grow to
    max_tokens: int

    def to_pb(self) -> generate_pb2.CachedBatch:
        """Convert a ChatGLMBatch to a text_generation_server.v1.CachedBatch protobuf"""
        return generate_pb2.CachedBatch(
            id=self.batch_id,
            request_ids=[r.id for r in self.requests],
            size=len(self),
            max_tokens=self.max_tokens,
        )

    @classmethod
    def from_pb(
        cls,
        pb: generate_pb2.Batch,
        tokenizer: PreTrainedTokenizerBase,
        dtype: torch.dtype,
        device: torch.device,
    ) -> "ChatGLMBatch":
        """Convert a text_generation_server.v1.Batch protobuf to a ChatGLMBatch"""
        inputs = []

        next_token_choosers = []
        stopping_criterias = []

        requests_idx_mapping = {}
        prefix_offsets = []
        read_offsets = []

        # Parse batch
        max_truncation = 0
        for i, r in enumerate(pb.requests):
            inputs.append(r.inputs)
            requests_idx_mapping[r.id] = i
            # add input ids, input seq lengths model
            next_token_choosers.append(
                ChatGLMNextTokenChooser.from_pb(r.parameters, device)
            )
            stopping_criteria = StoppingCriteria.from_pb(
                r.stopping_parameters, tokenizer
            )
            stopping_criterias.append(stopping_criteria)
            max_truncation = max(max_truncation, r.truncate)

        # Tokenize batch
        tokenized_inputs = tokenizer(
            inputs,
            return_tensors="pt",
            padding=True,
            return_token_type_ids=False,
            truncation=True,
            max_length=max_truncation,
        )

        # input ids length - pad token ids length = input length
        # padding lengths of all input ids
        padding_lengths = np.count_nonzero(
            tokenized_inputs.input_ids == tokenizer.pad_token_id, axis=1
        )

        # input lengths and without pad tokens length

        input_lengths = tokenized_inputs.input_ids.shape[1] - padding_lengths

        tokenized_inputs.to(device)

        max_input_length = input_lengths.max()

        max_tokens = len(inputs) * max_input_length
        for _ in pb.requests:
            input_len = tokenized_inputs["input_ids"].shape[1]
            prefix_offsets.append(input_len - 5)
            read_offsets.append(input_len)

        # all_input_ids = tokenized_inputs["input_ids"].T.split(1, dim=1)

        return cls(
            batch_id=pb.id,
            requests=pb.requests,
            requests_idx_mapping=requests_idx_mapping,
            input_ids=tokenized_inputs["input_ids"],
            read_offsets=read_offsets,
            # padding_lengths=padding_lengths,
            # NOTE: But I don't know how to dynamically generate this value , so  bye!
            # past_key_values=None,
            input_lengths=input_lengths.tolist(),
            prefix_offsets=prefix_offsets,
            next_token_choosers=next_token_choosers,
            stopping_criterias=stopping_criterias,
            max_input_length=max_input_length.item(),
            max_tokens=max_tokens,
        )

    @tracer.start_as_current_span("filter")
    def filter(self, request_ids: List[int]) -> Optional["ChatGLMBatch"]:
        # NOTE: from rust request ids to get python cache in the python shard!!!

        # print("filter...")
        if len(request_ids) == 0:
            raise ValueError("Batch must have at least one request")
        if len(request_ids) == len(self):
            return self

        keep_indices = []

        # New values after filtering
        requests_idx_mapping = {}
        requests = []
        input_lengths = []

        next_token_choosers = []
        stopping_criterias = []
        # decoder_input_lengths = []
        prefix_offsets = []
        read_offsets = []
        # padding_lengths = []
        # all_input_ids = []
        total_remaining_decode_tokens = 0
        max_input_length = 0

        for i, request_id in enumerate(request_ids):
            idx = self.requests_idx_mapping[request_id]
            requests_idx_mapping[request_id] = i
            keep_indices.append(idx)
            # padding_lengths.append(self.padding_lengths[idx])
            prefix_offsets.append(self.prefix_offsets[idx])
            read_offsets.append(self.read_offsets[idx])

            requests.append(self.requests[idx])

            request_input_length = self.input_lengths[idx]
            input_lengths.append(request_input_length)
            max_input_length = max(max_input_length, request_input_length)

            next_token_choosers.append(self.next_token_choosers[idx])
            stopping_criteria = self.stopping_criterias[idx]
            stopping_criterias.append(stopping_criteria)

            remaining_decode_tokens = (
                stopping_criteria.max_new_tokens - stopping_criteria.current_tokens
            )

            total_remaining_decode_tokens += remaining_decode_tokens

        input_ids = self.input_ids[keep_indices]
        # max token ?
        # requests ids * max input length + total_remaining_decode_tokens
        max_tokens = (
            len(request_ids) * (max_input_length) + total_remaining_decode_tokens
        )

        self.requests = requests
        self.requests_idx_mapping = requests_idx_mapping
        self.input_ids = input_ids
        self.input_lengths = input_lengths
        self.prefix_offsets = prefix_offsets
        self.read_offsets = read_offsets
        # self.padding_lengths = padding_lengths
        # self.next_token_choosers = next_token_choosers
        self.stopping_criterias = stopping_criterias
        self.max_input_length = max_input_length
        self.max_tokens = max_tokens

        return self

    @classmethod
    @tracer.start_as_current_span("concatenate")
    def concatenate(cls, batches: List["ChatGLMBatch"]) -> "ChatGLMBatch":
        """Concatenate multiple batches together by padding internal torch tensors"""

        # Used for padding
        device = None

        total_batch_size = 0
        max_input_length = 0

        # NOTE: total batch size and max input length to update batch tensor , like add padding token

        # NOTE:
        # 1. 得到最大的input ids length
        # 2. 获取对应paading length
        # 3, 截断整个input length
        # max_input_length_batch_ids = 0

        for ids, batch in enumerate(batches):
            total_batch_size += len(batch)  # 最终整合之后这个batch input ids的高度.

            max_input_length = max(max_input_length, batch.input_ids.shape[-1])
            # 选择最小的pdding length
            # min_paading_length = min(0,min(batch.padding_lengths))

            if ids == 0:
                device = batch.input_ids.device

        # Batch attributes
        requests = []
        requests_idx_mapping = {}
        input_lengths = []
        # padding_lengths = []

        prefix_offsets = []
        read_offsets = []
        next_token_choosers = []
        stopping_criterias = []
        max_tokens = 0

        input_ids = []

        # Used for slicing correctly inside the tensors
        # Equivalent to a cumsum on batch sizes
        # chatglm: used for mapping request and batch id
        start_index = 0

        for i, batch in enumerate(batches):
            # Extend all list attributes
            requests.extend(batch.requests)
            input_lengths.extend(batch.input_lengths)
            prefix_offsets.extend(batch.prefix_offsets)
            read_offsets.extend(batch.read_offsets)
            # all_input_ids.extend(batch.all_input_ids)
            next_token_choosers.extend(batch.next_token_choosers)
            stopping_criterias.extend(batch.stopping_criterias)

            if i == 0:
                requests_idx_mapping = batch.requests_idx_mapping
            else:
                # We need to offset the mapping for each batch by the cumulative batch size
                for k, v in batch.requests_idx_mapping.items():
                    requests_idx_mapping[k] = v + start_index

            # Slicing end index for this batch
            end_index = start_index + len(batch)

            # NOTE: concat input ids
            # 生成max toaltal length 长度, input ids shape -1 的维度 然后cat
            # check the padding lengths is equal to 0

            # padding_length = batch.padding_lengths

            # max input length 是所有batches 中最大的一个input length
            # batch.input_ids.shape[0]当前input ids 的高度
            # 为了保持每个input ids 的长度一致. max_input_length - batch.max_input_length 使用所有batch中input ids length最大的一个 - 当前batch中最大的一个.
            # 生成左边的padding tokens tensor
            # then cat left pad token and batch input ids to new batch input ids
            # 可能会造成的问题:
            # 1. 当最长的input ids 已经推理完毕, 但是可能其他批次比他小的input
            # 还在推理. 导致, input ids 左侧会有太多的没用token, input ids length 长度未精简
            # 解决方案: 需要保存变量 记录当前batch中的padding length
            # 如果整个批次中最小的padding length 不为0  则往前添加的padding需要减少当前值. 若当前max length input length <  input length
            left_pad_tensor = torch.full(
                (
                    batch.input_ids.shape[0],
                    max_input_length - batch.input_ids.shape[-1],
                ),
                1,
            ).to(batch.input_ids.device)

            # 更新 padding length?

            # print(f"left tensor: {left_pad_tensor},shape:{left_pad_tensor.shape}")
            # print(f"input ids: {batch.input_ids},shape:{batch.input_ids.shape}")
            input_ids.append(torch.cat([left_pad_tensor, batch.input_ids], dim=1))

            # input_ids[start_index:end_index] = batch.input_ids

            # Add eventual padding tokens that were added while concatenating
            max_tokens += batch.max_tokens + (
                max_input_length - batch.max_input_length
            ) * len(batch)

            start_index = end_index
        # print(input_ids)
        return cls(
            batch_id=batches[0].batch_id,
            requests=requests,
            requests_idx_mapping=requests_idx_mapping,
            input_ids=torch.cat(input_ids).to(device),
            # all_input_ids=all_input_ids,
            input_lengths=input_lengths,
            prefix_offsets=prefix_offsets,
            read_offsets=read_offsets,
            next_token_choosers=next_token_choosers,
            stopping_criterias=stopping_criterias,
            max_input_length=max_input_length,
            max_tokens=max_tokens,
        )

    def __len__(self):
        return len(self.requests)


class ChatGLM(Model):
    def __init__(
        self,
        model_id: str,
        revision: Optional[str] = None,
        quantize: Optional[str] = None,
        trust_remote_code: bool = False,
    ):
        if torch.cuda.is_available():
            device = torch.device("cuda")
            dtype = torch.float16
        else:
            if quantize:
                raise ValueError("quantization is not available on CPU")

            device = torch.device("cpu")
            dtype = torch.float32

        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id,
            revision=revision,
            torch_dtype=dtype,
            device_map="auto"
            if torch.cuda.is_available() and torch.cuda.device_count() > 1
            else None,
            # load_in_8bit=quantize == "bitsandbytes",
            trust_remote_code=trust_remote_code,
        ).half()

        if torch.cuda.is_available() and torch.cuda.device_count() == 1:
            model = model.cuda()

        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
            padding_side="left",
            truncation_side="left",
            trust_remote_code=trust_remote_code,
        )
        # tokenizer.bos_token_id = model.config.decoder_start_token_id

        super(ChatGLM, self).__init__(
            model=model,
            tokenizer=tokenizer,
            requires_padding=True,
            dtype=dtype,
            device=device,
        )

    @property
    def batch_type(self) -> Type[ChatGLMBatch]:
        return ChatGLMBatch

    def decode(self, decoder_ids: List[int]) -> str:
        return self.tokenizer.decode(
            decoder_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

    @tracer.start_as_current_span("generate_token")
    def generate_token(
        self, batch: ChatGLMBatch
    ) -> Tuple[List[Generation], Optional[ChatGLMBatch]]:
        # slice the attention mask to the correct shape

        # Results
        # TODO: update the generations when the forward and logtis id chooser done
        # so i do not use the prefill token

        generations: List[Generation] = []

        stopped = True

        model_inputs = self.model.prepare_inputs_for_generation(batch.input_ids)
        outputs = self.model.forward(**model_inputs, return_dict=True)

        next_token_logits = outputs.logits[:, -1, :]

        # print(f"next token lgs:{next_token_logits},shape: {next_token_logits.shape}")

        all_input_ids = []

        for i, (
            input_ids,
            reqs,
            next_token_chooser,
            stopping_criteria,
            next_token_logits,
            prefix_offset,
            read_offset,
        ) in enumerate(
            zip(
                batch.input_ids,
                batch.requests,
                batch.next_token_choosers,
                batch.stopping_criterias,
                next_token_logits,
                batch.prefix_offsets,
                batch.read_offsets,
            )
        ):
            next_tokens, next_token_probs = next_token_chooser(
                torch.unsqueeze(input_ids, dim=-2),
                torch.unsqueeze(next_token_logits, dim=-2),
            )

            # print(f"next_token:{next_tokens},shape:{next_tokens.shape}")

            next_token_id_squeezed = next_tokens.squeeze()
            # print(f"input ids:{input_ids},next tokens:{next_tokens}")

            # TODO: this value set to batch input ids.
            new_input_ids = torch.cat([input_ids, next_tokens])

            next_token_text, prefix_offset, read_offset = self.decode_token(
                new_input_ids,
                prefix_offset,
                read_offset,
            )
            # print(f"next_token_text:{next_token_text}")
            all_input_ids.append(new_input_ids)
            # Evaluate stopping criteria
            stop, reason = stopping_criteria(
                next_token_id_squeezed,
                next_token_text,
            )

            if not stop:
                stopped = False

            # Shard generations
            # All generations will be appended in the rust sharded client
            if i % self.world_size == self.rank:
                if stop:
                    # Decode generated tokens
                    # remove the origin input text
                    output_text = self.decode(
                        new_input_ids[-stopping_criteria.current_tokens :]
                    )
                    # Get seed
                    # if isinstance(next_token_chooser.choice, Sampling):
                    #     seed = next_token_chooser.choice.seed
                    # else:
                    #     seed = None

                    generated_text = GeneratedText(
                        output_text,
                        stopping_criteria.current_tokens,
                        reason,
                        None,
                    )
                else:
                    generated_text = None

                generation = Generation(
                    reqs.id,
                    None,
                    next_token_id_squeezed,
                    next_token_probs[-1, next_tokens],
                    next_token_text,
                    next_token_id_squeezed.item() in self.all_special_ids,
                    generated_text,
                )

                generations.append(generation)
            batch.prefix_offsets[i] = prefix_offset
            batch.read_offsets[i] = read_offset

        if stopped:
            return generations, None

        # update the batch values
        # inputs_ids add next tokens
        # all_next_tokens = torch.tensor([next_tokens_cat]).to(self.device)
        # print(f"inputs ids:{batch.input_ids}, shape:{batch.input_ids.shape} \n next tokens :{all_next_tokens},shape:{all_next_tokens.shape}")
        # print(all_input_ids)
        batch.input_ids = torch.stack(all_input_ids).to(self.device)

        return generations, batch


class ChatGLMShard(ChatGLM):
    def __init__(
        self,
        model_id: str,
        revision: str | None = None,
        quantize: str | None = None,
        trust_remote_code: bool = False,
    ):
        rank, world_size = int(os.getenv("RANK", 0)), int(os.getenv("WORLD_SIZE", 0))

        if torch.cuda.is_available():
            device = torch.device(f"cuda:{rank}")
            dtype = torch.float16
        else:
            device = torch.device("cpu")
            dtype = torch.float16

        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
            padding_side="left",
            truncation_side="left",
            trust_remote_code=trust_remote_code,
        )

        model = self._load_model_on_gpus(
            model_id,
            world_size,
        )

        super(ChatGLM, self).__init__(
            model,
            tokenizer=tokenizer,
            requires_padding=True,
            device=device,
            dtype=dtype,
            rank=rank,
            world_size=world_size,
        )

    @staticmethod
    def _auto_configure_device_map(num_gpus: int) -> Dict[str, int]:
        # transformer.word_embeddings 占用1层
        # transformer.final_layernorm 和 lm_head 占用1层
        # transformer.layers 占用 28 层
        # 总共30层分配到num_gpus张卡上
        num_trans_layers = 28
        per_gpu_layers = 30 / num_gpus

        # bugfix: 在linux中调用torch.embedding传入的weight,input不在同一device上,导致RuntimeError
        # windows下 model.device 会被设置成 transformer.word_embeddings.device
        # linux下 model.device 会被设置成 lm_head.device
        # 在调用chat或者stream_chat时,input_ids会被放到model.device上
        # 如果transformer.word_embeddings.device和model.device不同,则会导致RuntimeError
        # 因此这里将transformer.word_embeddings,transformer.final_layernorm,lm_head都放到第一张卡上
        device_map = {
            "transformer.word_embeddings": 0,
            "transformer.final_layernorm": 0,
            "lm_head": 0,
        }

        used = 2
        gpu_target = 0
        for i in range(num_trans_layers):
            if used >= per_gpu_layers:
                gpu_target += 1
                used = 0
            assert gpu_target < num_gpus
            device_map[f"transformer.layers.{i}"] = gpu_target
            used += 1

        return device_map

    def _load_model_on_gpus(
        self,
        checkpoint_path: Union[str, os.PathLike],
        num_gpus: int = 2,
        device_map: Optional[Dict[str, int]] = None,
        **kwargs,
    ) -> Module:
        if num_gpus < 2 and device_map is None:
            model = (
                AutoModelForSeq2SeqLM.from_pretrained(
                    checkpoint_path, trust_remote_code=True, **kwargs
                )
                .half()
                .cuda()
            )
        else:
            from accelerate import dispatch_model

            model = AutoModelForSeq2SeqLM.from_pretrained(
                checkpoint_path, trust_remote_code=True, **kwargs
            ).half()

            if device_map is None:
                device_map = self._auto_configure_device_map(num_gpus)

            model = dispatch_model(model, device_map=device_map)

        return model

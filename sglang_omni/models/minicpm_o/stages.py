# SPDX-License-Identifier: Apache-2.0
"""Stage executor factories for MiniCPM-o text and speech pipelines."""

from __future__ import annotations

import asyncio
import logging
import os
import queue
import threading
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from sglang.srt.arg_groups.model_override_base import resolved_view
from transformers import AutoTokenizer

from sglang_omni.models.minicpm_o.bootstrap import (
    create_talker_scheduler,
    create_thinker_scheduler,
)
from sglang_omni.models.minicpm_o.components.audio_encoder import MiniCPMOAudioEncoder
from sglang_omni.models.minicpm_o.components.code2wav import MiniCPMOCode2Wav
from sglang_omni.models.minicpm_o.components.image_encoder import MiniCPMOImageEncoder
from sglang_omni.models.minicpm_o.components.preprocessor import MiniCPMOPreprocessor
from sglang_omni.models.minicpm_o.hf_config import register_minicpm_o_hf_config
from sglang_omni.models.minicpm_o.merge import build_decode_result
from sglang_omni.models.minicpm_o.native_config import TALKER_CONTEXT_LENGTH
from sglang_omni.models.minicpm_o.payload_types import MiniCPMOPipelineState
from sglang_omni.models.minicpm_o.request_builders import build_encoder_request
from sglang_omni.models.minicpm_o.routing import TALKER_STAGE, code2wav_reference_audio
from sglang_omni.platforms import current_platform
from sglang_omni.proto.request import StagePayload
from sglang_omni.scheduling.generation_batch_policy import (
    CudaGraphBackend,
    build_default_prefill_cuda_graph_bs,
    build_generation_batch_overrides,
    operator_selected_prefill_backend,
    validate_generation_batch_policy,
)
from sglang_omni.scheduling.message import IncomingMessage
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData
from sglang_omni.scheduling.sglang_backend.server_args_builder import (
    build_sglang_server_args,
)
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.scheduling.stage_cache import StageOutputCache
from sglang_omni.scheduling.streaming_detokenizer import StreamingDetokenizeScheduler
from sglang_omni.utils.audio_payload import audio_waveform_payload
from sglang_omni.utils.device import resolve_concrete_device
from sglang_omni.utils.misc import avail_gpu_mem

logger = logging.getLogger(__name__)


def create_preprocessing_executor(
    model_path: str,
    *,
    speech_enabled: bool = False,
    device: str | None = None,
    gpu_id: int | None = None,
) -> SimpleScheduler[StagePayload, StagePayload]:
    preprocessor = MiniCPMOPreprocessor(
        model_path,
        speech_enabled=speech_enabled,
        device=resolve_concrete_device(device, gpu_id),
    )

    return SimpleScheduler[StagePayload, StagePayload](preprocessor)


ENCODER_CACHE_MAX_ENTRIES = 64
ENCODER_CACHE_MAX_BYTES = 4 * 1024**3


def create_encoder_executor(
    encoder: nn.Module, *, stage_name: str
) -> SimpleScheduler[StagePayload, StagePayload]:
    cache = StageOutputCache(
        max_size=ENCODER_CACHE_MAX_ENTRIES,
        max_bytes=ENCODER_CACHE_MAX_BYTES,
        cache_device="cpu",
    )

    def _encode_stage(payload: StagePayload) -> StagePayload:
        state = MiniCPMOPipelineState.from_dict(payload.data)
        request = build_encoder_request(state, stage_name=stage_name)
        cached = (
            None if request.skip_result is not None else cache.get(request.cache_key)
        )
        if request.skip_result is not None:
            encoder_out = request.skip_result
        elif cached is not None:
            encoder_out = cached
        else:
            with torch.no_grad():
                encoder_out = encoder(**request.model_inputs)
            cache.put(request.cache_key, encoder_out)
        state.encoder_outs[stage_name] = encoder_out
        payload.data = state.to_dict()
        return payload

    return SimpleScheduler(_encode_stage)


def create_image_encoder_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    dtype: str | None = None,
) -> SimpleScheduler[StagePayload, StagePayload]:
    encoder = MiniCPMOImageEncoder(
        model_path, device=str(resolve_concrete_device(device, gpu_id)), dtype=dtype
    )
    return create_encoder_executor(encoder, stage_name="image_encoder")


def create_audio_encoder_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    dtype: str | None = None,
) -> SimpleScheduler[StagePayload, StagePayload]:
    encoder = MiniCPMOAudioEncoder(
        model_path, device=str(resolve_concrete_device(device, gpu_id)), dtype=dtype
    )
    return create_encoder_executor(encoder, stage_name="audio_encoder")


# note (0xtoward): the talker prefills unchunked, so no chunk bounds the capture;
# forwards above this many tokens run eager.
TALKER_PREFILL_CUDA_GRAPH_MAX_TOKENS = 2048


def create_sglang_talker_executor_from_config(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    tp_rank: int = 0,
    tp_size: int = 1,
    nccl_port: int | None = None,
    max_seq_len: int = TALKER_CONTEXT_LENGTH,
    server_args_overrides: Mapping[str, object] | None = None,
    total_gpu_memory_fraction: float | None = None,
    session_mode: bool = False,
) -> OmniScheduler[SGLangARRequestData]:
    """Returns OmniScheduler for the native sglang MiniCPM-o talker."""
    concrete_device = resolve_concrete_device(device, gpu_id)
    gpu_id = concrete_device.index or 0
    register_minicpm_o_hf_config()
    if (
        current_platform.is_cuda()
        and current_platform.enable_talker_graph()
        and not session_mode
    ):
        prefill_graph_backend = CudaGraphBackend.BREAKABLE
    else:
        prefill_graph_backend = CudaGraphBackend.DISABLED
    overrides = build_generation_batch_overrides(
        max_running_requests=32,
        server_args_overrides=server_args_overrides,
        disable_cuda_graph=False,
        # note (Chenyang): CI serves MiniCPM-o with SGLang torch compile off.
        enable_torch_compile=False,
        sampling_backend="pytorch",
        cuda_graph_backend_prefill=prefill_graph_backend,
        cuda_graph_bs_prefill=build_default_prefill_cuda_graph_bs(
            TALKER_PREFILL_CUDA_GRAPH_MAX_TOKENS
        ),
    )
    overrides.setdefault("trust_remote_code", False)
    if session_mode:
        overrides.update(
            enable_streaming_session=True,
            disable_overlap_schedule=True,
        )
    else:
        pass
    overrides["tp_size"] = tp_size
    # note (MayDomine): cap talker KV allocation so it does not starve the thinker.
    overrides.setdefault("max_total_tokens", 32 * max_seq_len)
    server_args = build_sglang_server_args(
        model_path,
        context_length=max_seq_len,
        **overrides,
    )
    validate_generation_batch_policy(
        model_name="MiniCPM-o talker",
        server_args=server_args,
    )

    logger.info(
        f"sglang_ar_startup stage=talker gpu_id={gpu_id} "
        f"tp_rank={tp_rank}/{tp_size} context_length={max_seq_len} "
        f"total_gpu_memory_fraction={total_gpu_memory_fraction} "
        f"mem_fraction_static={resolved_view(server_args).mem_fraction_static} "
        f"max_running_requests={resolved_view(server_args).max_running_requests} "
        f"max_total_tokens={resolved_view(server_args).max_total_tokens} "
        f"pre_load_avail_mem={avail_gpu_mem(gpu_id)} pid={os.getpid()}"
    )
    scheduler = create_talker_scheduler(
        server_args,
        gpu_id,
        tp_rank=tp_rank,
        nccl_port=nccl_port,
        total_gpu_memory_fraction=total_gpu_memory_fraction,
        session_mode=session_mode,
        operator_selected_prefill_backend=operator_selected_prefill_backend(
            server_args_overrides
        ),
    )
    logger.info(
        f"sglang_ar_started stage=talker gpu_id={gpu_id} "
        f"post_load_avail_mem={avail_gpu_mem(gpu_id)} pid={os.getpid()}"
    )
    return scheduler


def create_sglang_session_talker_executor_from_config(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    server_args_overrides: Mapping[str, object] | None = None,
    total_gpu_memory_fraction: float | None = None,
) -> OmniScheduler[SGLangARRequestData]:
    """Returns the talker that keeps native KV across the units of a duplex session."""
    return create_sglang_talker_executor_from_config(
        model_path,
        device=device,
        gpu_id=gpu_id,
        server_args_overrides=server_args_overrides,
        total_gpu_memory_fraction=total_gpu_memory_fraction,
        session_mode=True,
    )


HIFT_PENDING_BATCHES = 2


@dataclass(kw_only=True)
class Code2WavBatch:
    payloads: list[StagePayload]
    codec_tokens: list[list[int]]
    references: list[bytes | None]
    voiced_rows: list[int]

    @classmethod
    def from_payloads(cls, payloads: list[StagePayload]) -> Code2WavBatch:
        codec_tokens: list[list[int]] = []
        references: list[bytes | None] = []
        voiced_rows: list[int] = []
        for row, payload in enumerate(payloads):
            state = MiniCPMOPipelineState.from_dict(payload.data)
            tokens = (
                state.engine_outputs[TALKER_STAGE]["codec_tokens"].reshape(-1).tolist()
            )
            if tokens:
                voiced_rows.append(row)
                codec_tokens.append(tokens)
                references.append(code2wav_reference_audio(payload))
            else:
                pass
        logger.info(
            f"minicpm_code2wav_batch size={len(payloads)} "
            f"max_codec_tokens={max(map(len, codec_tokens), default=0)}"
        )
        return cls(
            payloads=payloads,
            codec_tokens=codec_tokens,
            references=references,
            voiced_rows=voiced_rows,
        )

    def outputs(
        self, voiced_waveforms: list[np.ndarray], sample_rate: int
    ) -> list[StagePayload]:
        waveforms = [np.zeros(0, dtype=np.float32) for _ in self.payloads]
        for row, waveform in zip(self.voiced_rows, voiced_waveforms, strict=True):
            waveforms[row] = waveform
        for payload, waveform in zip(self.payloads, waveforms, strict=True):
            payload.data = dict(
                audio_waveform_payload(
                    waveform,
                    sample_rate=sample_rate,
                    modality="audio",
                    source_hint="MiniCPM-o",
                )
            )
        return self.payloads


def vocode_code2wav_payloads(
    model: MiniCPMOCode2Wav, payloads: list[StagePayload]
) -> list[StagePayload]:
    """Keep the synchronous vocoder entry point for direct callers."""
    try:
        batch = Code2WavBatch.from_payloads(payloads)
        return batch.outputs(
            model.vocode(batch.codec_tokens, batch.references), model.sample_rate
        )
    finally:
        for payload in payloads:
            model.release_reference(payload.request_id)


@dataclass(kw_only=True)
class HiFTRequest:
    payload: StagePayload
    mel: torch.Tensor
    ready: torch.Event
    enqueue_order: int


class MiniCPMOCode2WavScheduler(SimpleScheduler[StagePayload, StagePayload]):
    """Regroup Flow outputs by length for a separate HiFT worker."""

    def __init__(
        self,
        model: MiniCPMOCode2Wav,
        *,
        max_batch_size: int,
        max_batch_wait_ms: float,
        batch_wait_when_idle: bool,
        max_batch_cost: int | None,
        decode_stream_priority: int,
    ) -> None:
        self.model: MiniCPMOCode2Wav = model
        self.hift_pending: dict[int, deque[HiFTRequest]] = {}
        self.hift_condition: threading.Condition = threading.Condition()
        self.hift_pending_count: int = 0
        self.hift_enqueue_order: int = 0
        self.hift_closed: bool = False
        with model.device_context:
            device_module = torch.get_device_module(model.token2wav.device)
            self.hift_stream: torch.Stream = device_module.Stream(
                priority=decode_stream_priority
            )
            self.hift_stream.wait_stream(device_module.current_stream())
        super().__init__(
            lambda payload: vocode_code2wav_payloads(model, [payload])[0],
            batch_compute_fn=lambda payloads: vocode_code2wav_payloads(model, payloads),
            max_batch_size=max_batch_size,
            max_batch_wait_ms=max_batch_wait_ms,
            batch_wait_when_idle=batch_wait_when_idle,
            request_cost_fn=self.codec_token_cost,
            max_batch_cost=max_batch_cost,
            abort_callback=model.release_reference,
            shutdown_callback=model.close_reference_pool,
            request_arrival_hook=self.prefetch_reference,
            request_readiness_hook=self.is_ready,
        )

    @staticmethod
    def codec_token_cost(payload: StagePayload) -> int:
        state = MiniCPMOPipelineState.from_dict(payload.data)
        return int(state.engine_outputs[TALKER_STAGE]["codec_tokens"].numel())

    def prefetch_reference(self, payload: StagePayload) -> None:
        try:
            reference = code2wav_reference_audio(payload)
        except ValueError:
            return
        future = self.model.prefetch_reference(payload.request_id, reference)
        if future is not None:
            future.add_done_callback(lambda completed: self.notify_readiness())
        else:
            pass

    def is_ready(self, payload: StagePayload) -> bool:
        return self.codec_token_cost(payload) == 0 or self.model.is_reference_ready(
            payload.request_id
        )

    def run_batch(
        self, batch: list[IncomingMessage], loop: asyncio.AbstractEventLoop
    ) -> None:
        payloads = [
            message.data
            for message in batch
            if not self.consume_if_aborted(message.request_id)
        ]
        if not payloads:
            return
        else:
            pass
        hift_submitted_request_ids: set[str] = set()
        try:
            requests = Code2WavBatch.from_payloads(payloads)
            payloads = [requests.payloads[row] for row in requests.voiced_rows]
            for row, payload in enumerate(requests.payloads):
                if row not in requests.voiced_rows:
                    if not self.consume_if_aborted(payload.request_id):
                        payload.data = dict(
                            audio_waveform_payload(
                                np.zeros(0, dtype=np.float32),
                                sample_rate=self.model.sample_rate,
                                modality="audio",
                                source_hint="MiniCPM-o",
                            )
                        )
                        self.emit_result(payload.request_id, payload, self.outbox)
                    else:
                        pass
                    self.model.release_reference(payload.request_id)
                else:
                    pass
            if not payloads:
                return
            else:
                pass
            speaker_prompts = self.model.prepare_references(requests.references)
            row_groups = self.model.flow_row_groups(
                requests.codec_tokens, speaker_prompts
            )
            device_module = torch.get_device_module(self.model.token2wav.device)
            for row_group in row_groups:
                with device_module.stream(self.model.flow_stream):
                    mel = self.model.generate_mel(
                        [requests.codec_tokens[index] for index in row_group],
                        [speaker_prompts[index] for index in row_group],
                    )
                    ready = device_module.Event()
                    ready.record(self.model.flow_stream)
                with self.hift_condition:
                    for row, index in enumerate(row_group):
                        payload = payloads[index]
                        self.hift_condition.wait_for(
                            lambda: self.hift_pending_count
                            < HIFT_PENDING_BATCHES * self.max_batch_size
                        )
                        token_length = len(requests.codec_tokens[index])
                        self.hift_pending.setdefault(token_length, deque()).append(
                            HiFTRequest(
                                payload=payload,
                                mel=mel[
                                    row : row + 1,
                                    :,
                                    : token_length * self.model.token2wav.flow.up_rate,
                                ],
                                ready=ready,
                                enqueue_order=self.hift_enqueue_order,
                            )
                        )
                        hift_submitted_request_ids.add(payload.request_id)
                        self.hift_enqueue_order += 1
                        self.hift_pending_count += 1
                        self.hift_condition.notify_all()
        except Exception as error:
            logger.exception("MiniCPM-o Flow batch failed")
            for payload in payloads:
                if payload.request_id in hift_submitted_request_ids:
                    continue
                else:
                    pass
                self.model.release_reference(payload.request_id)
                if self.consume_if_aborted(payload.request_id):
                    continue
                else:
                    self.emit_error(payload.request_id, error, self.outbox)

    def run_hift(self) -> None:
        device_module = torch.get_device_module(self.model.token2wav.device)
        while True:
            with self.hift_condition:
                self.hift_condition.wait_for(
                    lambda: self.hift_pending or self.hift_closed
                )
                if not self.hift_pending:
                    return
                else:
                    pass
                token_length = min(
                    self.hift_pending,
                    key=lambda length: self.hift_pending[length][0].enqueue_order,
                )
                group = self.hift_pending[token_length]
                batch_size = min(len(group), self.max_batch_size)
                if self.max_batch_cost is not None:
                    batch_size = min(
                        batch_size, max(1, self.max_batch_cost // token_length)
                    )
                else:
                    pass
                batch = [group.popleft() for _ in range(batch_size)]
                if not group:
                    self.hift_pending.pop(token_length)
                else:
                    pass
                self.hift_pending_count -= len(batch)
                self.hift_condition.notify_all()
            batch = [
                request
                for request in batch
                if not self.consume_if_aborted(request.payload.request_id)
            ]
            if not batch:
                continue
            else:
                pass
            try:
                with device_module.stream(self.hift_stream):
                    for request in batch:
                        self.hift_stream.wait_event(request.ready)
                        request.mel.record_stream(self.hift_stream)
                    mel = torch.cat([request.mel for request in batch])
                    waveforms = self.model.decode_mel(mel, [token_length] * len(batch))
                for request, waveform in zip(batch, waveforms, strict=True):
                    payload = request.payload
                    if self.consume_if_aborted(payload.request_id):
                        continue
                    else:
                        payload.data = dict(
                            audio_waveform_payload(
                                waveform,
                                sample_rate=self.model.sample_rate,
                                modality="audio",
                                source_hint="MiniCPM-o",
                            )
                        )
                        self.emit_result(payload.request_id, payload, self.outbox)
            except Exception as error:
                logger.exception("MiniCPM-o HiFT batch failed")
                for request in batch:
                    if self.consume_if_aborted(request.payload.request_id):
                        continue
                    else:
                        self.emit_error(request.payload.request_id, error, self.outbox)
            finally:
                for request in batch:
                    self.model.release_reference(request.payload.request_id)

    def start_serial(self) -> None:
        hift_worker = threading.Thread(target=self.run_hift, name="minicpmo-hift")
        hift_worker.start()
        try:
            super().start_serial()
        finally:
            with self.hift_condition:
                self.hift_closed = True
                self.hift_condition.notify_all()
            hift_worker.join()
            while True:
                if self.pending_messages:
                    message = self.pending_messages.popleft()
                else:
                    try:
                        message = self.inbox.get_nowait()
                    except queue.Empty:
                        break
                self.model.release_reference(message.request_id)


def create_code2wav_executor(
    model_path: str,
    *,
    max_batch_size: int,
    max_batch_wait_ms: float,
    batch_wait_when_idle: bool,
    enable_flow_variable_length: bool,
    reference_workers: int,
    prompt_cache_capacity: int,
    decode_stream_priority: int,
    enable_flow_block_compile: bool,
    enable_dit_torch_compile: bool,
    enable_hift_torch_compile: bool,
    device: str | None = None,
    gpu_id: int | None = None,
    dtype: str | None = None,
    max_batch_cost: int | None = None,
    enable_flow_cuda_graph: bool,
    flow_cuda_graph_capture_shapes: tuple[tuple[int, int], ...],
    packed_dit_cuda_graph_capture_shapes: tuple[tuple[int, int], ...],
) -> MiniCPMOCode2WavScheduler:
    model = MiniCPMOCode2Wav(
        model_path,
        device=str(resolve_concrete_device(device, gpu_id)),
        dtype=dtype,
        enable_dit_torch_compile=enable_dit_torch_compile,
        enable_hift_torch_compile=enable_hift_torch_compile,
        enable_flow_variable_length=enable_flow_variable_length,
        reference_workers=reference_workers,
        prompt_cache_capacity=prompt_cache_capacity,
        decode_stream_priority=decode_stream_priority,
        enable_flow_block_compile=enable_flow_block_compile,
        enable_flow_cuda_graph=enable_flow_cuda_graph,
        flow_cuda_graph_capture_shapes=flow_cuda_graph_capture_shapes,
        packed_dit_cuda_graph_capture_shapes=packed_dit_cuda_graph_capture_shapes,
    )

    return MiniCPMOCode2WavScheduler(
        model,
        max_batch_size=max_batch_size,
        max_batch_wait_ms=max_batch_wait_ms,
        batch_wait_when_idle=batch_wait_when_idle,
        max_batch_cost=max_batch_cost,
        decode_stream_priority=decode_stream_priority,
    )


def create_decode_executor(model_path: str) -> StreamingDetokenizeScheduler:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    eos_token_id = tokenizer.eos_token_id
    return StreamingDetokenizeScheduler(
        tokenizer,
        eos_token_id,
        build_result=lambda payload, is_streaming: build_decode_result(
            payload,
            tokenizer=tokenizer,
            eos_token_id=eos_token_id,
            is_streaming=is_streaming,
        ),
    )


def create_sglang_thinker_executor_from_config(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    tp_rank: int = 0,
    tp_size: int = 1,
    nccl_port: int | None = None,
    max_seq_len: int = 8192,
    server_args_overrides: Mapping[str, object] | None = None,
    total_gpu_memory_fraction: float | None = None,
    enable_async_decode: bool = True,
    async_decode_min_batch_size: int = 1,
    speech_enabled: bool = False,
) -> OmniScheduler[SGLangARRequestData]:
    """Returns OmniScheduler for the MiniCPM-o thinker."""
    concrete_device = resolve_concrete_device(device, gpu_id)
    gpu_id = concrete_device.index or 0
    register_minicpm_o_hf_config()
    overrides = build_generation_batch_overrides(
        max_running_requests=64,
        server_args_overrides=server_args_overrides,
        disable_cuda_graph=False,
        # note (Chenyang): CI serves MiniCPM-o with SGLang torch compile off.
        enable_torch_compile=False,
        enable_mixed_chunk=True,
        chunked_prefill_size=8192,
        sampling_backend="pytorch",
    )
    overrides.setdefault("trust_remote_code", False)
    overrides["tp_size"] = tp_size
    server_args = build_sglang_server_args(
        model_path,
        context_length=max_seq_len,
        **overrides,
    )
    validate_generation_batch_policy(
        model_name="MiniCPM-o thinker",
        server_args=server_args,
    )

    logger.info(
        f"sglang_ar_startup stage=thinker gpu_id={gpu_id} "
        f"tp_rank={tp_rank}/{tp_size} context_length={max_seq_len} "
        f"total_gpu_memory_fraction={total_gpu_memory_fraction} "
        f"mem_fraction_static={resolved_view(server_args).mem_fraction_static} "
        f"pre_load_avail_mem={avail_gpu_mem(gpu_id)} pid={os.getpid()}"
    )
    scheduler = create_thinker_scheduler(
        server_args,
        gpu_id,
        tp_rank=tp_rank,
        nccl_port=nccl_port,
        total_gpu_memory_fraction=total_gpu_memory_fraction,
        enable_async_decode=enable_async_decode,
        async_decode_min_batch_size=async_decode_min_batch_size,
        speech_enabled=speech_enabled,
    )
    logger.info(
        f"sglang_ar_started stage=thinker gpu_id={gpu_id} "
        f"post_load_avail_mem={avail_gpu_mem(gpu_id)} pid={os.getpid()}"
    )
    return scheduler

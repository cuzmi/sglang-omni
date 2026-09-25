# SPDX-License-Identifier: Apache-2.0
"""Run the MiniCPM-o text backbone on SGLang's Qwen3 model."""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Iterable, Iterator
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
from sglang.srt.layers.linear import MergedColumnParallelLinear, RowParallelLinear
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.qwen3 import Qwen3ForCausalLM
from transformers import PretrainedConfig

from sglang_omni.models.minicpm_o.components.gate_up_gemv import (
    gate_up_gemv_forward,
)
from sglang_omni.models.minicpm_o.hf_config import derive_text_config

NON_TEXT_PREFIXES = (
    "vpm.",
    "resampler.",
    "apm.",
    "audio_projection_layer.",
    "tts.",
)


class MiniCPMOThinkerForCausalLM(nn.Module):
    """MiniCPM-o text backbone without the multimodal towers."""

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.root_config = config
        self.config = derive_text_config(config)
        self.language_model = Qwen3ForCausalLM(
            self.config,
            quant_config,
            prefix=prefix,
        )
        if os.environ.get("SGLANG_OMNI_MINICPMO_GATEUP_GEMV") == "1":
            for layer in self.language_model.model.layers:
                projection = layer.mlp.gate_up_proj
                projection.gate_up_gemv_selected = False
                projection.forward = partial(
                    gate_up_gemv_forward, projection, projection.forward
                )
        shape_log_path = os.environ.get("SGLANG_OMNI_MINICPMO_MLP_SHAPES_PATH")
        if shape_log_path:
            shape_log = Path(shape_log_path)
            shape_log.parent.mkdir(parents=True, exist_ok=True)
            seen_shapes: set[tuple[str, tuple[int, ...]]] = set()
            mlp = self.language_model.model.layers[0].mlp
            for projection, layer in (
                ("gate_up_proj", mlp.gate_up_proj),
                ("down_proj", mlp.down_proj),
            ):

                def record_shape(
                    module: MergedColumnParallelLinear | RowParallelLinear,
                    inputs: tuple[torch.Tensor, ...],
                    output: tuple[torch.Tensor, torch.Tensor | None],
                    projection: str = projection,
                ) -> None:
                    input_tensor = inputs[0]
                    output_tensor = output[0]
                    key = (projection, tuple(input_tensor.shape))
                    if key not in seen_shapes:
                        seen_shapes.add(key)
                        weight = module.weight
                        record = {
                            "layer": 0,
                            "projection": projection,
                            "input_shape": list(input_tensor.shape),
                            "input_stride": list(input_tensor.stride()),
                            "input_dtype": str(input_tensor.dtype),
                            "weight_shape": list(weight.shape),
                            "weight_stride": list(weight.stride()),
                            "weight_dtype": str(weight.dtype),
                            "output_shape": list(output_tensor.shape),
                            "output_stride": list(output_tensor.stride()),
                            "output_dtype": str(output_tensor.dtype),
                            "tp_size": module.tp_size,
                            "pid": os.getpid(),
                        }
                        with shape_log.open("a", encoding="utf-8") as log_file:
                            log_file.write(f"{json.dumps(record)}\n")

                layer.register_forward_hook(record_shape)

            gate_up_projection = mlp.gate_up_proj
            gate_up_forward = gate_up_projection.forward
            gate_up_calls_log = shape_log.with_name("gate_up_calls.jsonl")

            def profiled_gate_up_forward(
                input_tensor: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor | None]:
                if input_tensor.shape[0] == 1:
                    call = {
                        "timestamp_ns": time.time_ns(),
                        "pid": os.getpid(),
                        "thread_ident": threading.get_ident(),
                        "native_thread_id": threading.get_native_id(),
                        "input_shape": list(input_tensor.shape),
                    }
                    with gate_up_calls_log.open("a", encoding="utf-8") as log_file:
                        log_file.write(f"{json.dumps(call)}\n")
                    with torch.profiler.record_function(
                        "minicpmo.layer0.gate_up_proj.m1"
                    ):
                        return gate_up_forward(input_tensor)
                else:
                    return gate_up_forward(input_tensor)

            gate_up_projection.forward = profiled_gate_up_forward

    @property
    def thinker(self) -> "MiniCPMOThinkerForCausalLM":
        # note (MayDomine): the shared thinker runner expects this backbone view.
        return self

    @property
    def model(self) -> nn.Module:
        return self.language_model.model

    @property
    def lm_head(self) -> nn.Module:
        return self.language_model.lm_head

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> LogitsProcessorOutput:
        return self.language_model(
            input_ids,
            positions,
            forward_batch,
            input_embeds=input_embeds,
            **kwargs,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        def _text_weights() -> Iterator[tuple[str, torch.Tensor]]:
            for name, loaded_weight in weights:
                if name.startswith(NON_TEXT_PREFIXES):
                    continue
                if name.startswith("llm."):
                    yield name[len("llm.") :], loaded_weight

        self.language_model.load_weights(_text_weights())


EntryClass = MiniCPMOThinkerForCausalLM

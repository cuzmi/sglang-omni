"""Experimental BF16 Gate/Up GEMV for MiniCPM-o decode."""

from __future__ import annotations

import logging
from collections.abc import Callable

import torch
import triton
import triton.language as tl
from sglang.srt.layers.linear import MergedColumnParallelLinear
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

INPUT_FEATURES = 4096
OUTPUT_FEATURES = 24576
BLOCK_N = 16
BLOCK_K = 512
NUM_WARPS = 4

logger = logging.getLogger(__name__)


@triton.jit
def gate_up_gemv_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    input_features: tl.constexpr,
    output_features: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    prefetch: tl.constexpr,
):
    output_rows = tl.program_id(0) * block_n + tl.arange(0, block_n)
    column_offsets = tl.arange(0, block_k)
    partial_sums = tl.full((block_n, block_k), 0, tl.float32)

    if prefetch:
        input_values = tl.load(input_ptr + column_offsets).to(tl.float32)
        weights = tl.load(
            weight_ptr
            + output_rows[:, None] * input_features
            + column_offsets[None, :],
            mask=output_rows[:, None] < output_features,
            other=0,
        )
        for block in range(1, input_features // block_k):
            next_columns = block * block_k + column_offsets
            next_input_values = tl.load(input_ptr + next_columns).to(tl.float32)
            next_weights = tl.load(
                weight_ptr
                + output_rows[:, None] * input_features
                + next_columns[None, :],
                mask=output_rows[:, None] < output_features,
                other=0,
            )
            partial_sums = tl.fma(
                weights.to(tl.float32), input_values[None, :], partial_sums
            )
            input_values = next_input_values
            weights = next_weights
        partial_sums = tl.fma(
            weights.to(tl.float32), input_values[None, :], partial_sums
        )
    else:
        for block in range(input_features // block_k):
            columns = block * block_k + column_offsets
            input_values = tl.load(input_ptr + columns).to(tl.float32)
            weights = tl.load(
                weight_ptr + output_rows[:, None] * input_features + columns[None, :],
                mask=output_rows[:, None] < output_features,
                other=0,
            ).to(tl.float32)
            partial_sums = tl.fma(weights, input_values[None, :], partial_sums)

    values = tl.sum(partial_sums, axis=1)
    tl.store(output_ptr + output_rows, values, mask=output_rows < output_features)


def gate_up_gemv(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor | None = None,
    block_n: int = BLOCK_N,
    block_k: int = BLOCK_K,
    num_warps: int = NUM_WARPS,
    prefetch: bool = True,
) -> torch.Tensor:
    assert input_tensor.shape == (1, INPUT_FEATURES)
    assert weight.shape == (OUTPUT_FEATURES, INPUT_FEATURES)
    assert input_tensor.dtype == weight.dtype == torch.bfloat16
    assert input_tensor.is_cuda and weight.is_cuda
    assert input_tensor.is_contiguous() and weight.is_contiguous()

    if output is None:
        output = torch.empty(
            (1, OUTPUT_FEATURES), device=input_tensor.device, dtype=input_tensor.dtype
        )
    else:
        assert output.shape == (1, OUTPUT_FEATURES)
        assert output.dtype == input_tensor.dtype
        assert output.device == input_tensor.device
        assert output.is_contiguous()
    gate_up_gemv_kernel[(triton.cdiv(OUTPUT_FEATURES, block_n),)](
        input_tensor,
        weight,
        output,
        INPUT_FEATURES,
        OUTPUT_FEATURES,
        block_n,
        block_k,
        prefetch,
        num_warps=num_warps,
    )
    return output


def gate_up_gemv_forward(
    projection: MergedColumnParallelLinear,
    fallback: Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor | None]],
    input_tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    weight = projection.weight
    if (
        input_tensor.shape == (1, INPUT_FEATURES)
        and input_tensor.dtype == torch.bfloat16
        and input_tensor.is_cuda
        and input_tensor.is_contiguous()
        and weight.shape == (OUTPUT_FEATURES, INPUT_FEATURES)
        and weight.dtype == torch.bfloat16
        and weight.is_cuda
        and weight.is_contiguous()
        and projection.tp_size == 1
        and projection.bias is None
        and isinstance(projection.quant_method, UnquantizedLinearMethod)
    ):
        if not projection.gate_up_gemv_selected:
            logger.warning("Selected MiniCPM-o Gate/Up GEMV for M=1")
            projection.gate_up_gemv_selected = True
        return gate_up_gemv(input_tensor, weight), None
    return fallback(input_tensor)

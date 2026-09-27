# SPDX-License-Identifier: Apache-2.0
"""Cross-request image encoding with cache reuse and ordered output splitting."""

from __future__ import annotations

import torch

from sglang_omni.models.minicpm_o.components.image_encoder import MiniCPMOImageEncoder
from sglang_omni.models.minicpm_o.payload_types import MiniCPMOPipelineState
from sglang_omni.models.minicpm_o.request_builders import build_encoder_request
from sglang_omni.profiler.event_recorder import emit as emit_event
from sglang_omni.proto.request import StagePayload
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.scheduling.stage_cache import StageOutputCache


def image_slice_cost(payload: StagePayload) -> int:
    state = MiniCPMOPipelineState.from_dict(payload.data)
    request = build_encoder_request(state, stage_name="image_encoder")
    return len(request.model_inputs.get("pixel_values", []))


def batch_image_encoder_payloads(
    payloads: list[StagePayload],
    *,
    encoder: MiniCPMOImageEncoder,
    cache: StageOutputCache,
) -> list[StagePayload]:
    """Encode unique misses together and restore each payload's slice order."""
    states = [MiniCPMOPipelineState.from_dict(payload.data) for payload in payloads]
    requests = [
        build_encoder_request(state, stage_name="image_encoder") for state in states
    ]
    outputs: dict[int, dict[str, torch.Tensor]] = {}
    leaders: dict[str, int] = {}
    duplicates: dict[int, int] = {}
    active_indices: list[int] = []
    cache_hits = 0
    for index, request in enumerate(requests):
        cached = cache.get(request.cache_key)
        if request.skip_result is not None:
            outputs[index] = request.skip_result
        elif cached is not None:
            outputs[index] = cached
            cache_hits += 1
        elif request.cache_key is not None and request.cache_key in leaders:
            duplicates[index] = leaders[request.cache_key]
        else:
            pixels = request.model_inputs.get("pixel_values")
            sizes = request.model_inputs.get("tgt_sizes")
            if not pixels or sizes is None:
                outputs[index] = {}
            else:
                active_indices.append(index)
                if request.cache_key is not None:
                    leaders[request.cache_key] = index
                else:
                    pass

    if active_indices:
        slice_counts = [
            len(requests[index].model_inputs["pixel_values"])
            for index in active_indices
        ]
        pixel_values: list[torch.Tensor] = [
            pixels
            for index in active_indices
            for pixels in requests[index].model_inputs["pixel_values"]
        ]
        target_sizes = torch.cat(
            [requests[index].model_inputs["tgt_sizes"] for index in active_indices],
            dim=0,
        )
        total_slices = sum(slice_counts)
        assert target_sizes.shape == (total_slices, 2)
        metadata = {
            "modality": "image",
            "request_batch_size": len(payloads),
            "batch_size": len(active_indices),
            "num_slices": total_slices,
            "cache_hits": cache_hits,
            "dedup_same_batch": len(duplicates),
        }
        for index in active_indices:
            emit_event(
                request_id=payloads[index].request_id,
                stage="image_encoder",
                event_name="encoder_start",
                metadata=metadata,
            )
        status = "error"
        try:
            with torch.no_grad():
                embeddings = encoder(pixel_values=pixel_values, tgt_sizes=target_sizes)[
                    "image_embeds"
                ]
            assert embeddings.ndim == 2
            assert embeddings.shape[0] > 0 and embeddings.shape[0] % total_slices == 0
            query_count = embeddings.shape[0] // total_slices
            cursor = 0
            for index, slice_count in zip(active_indices, slice_counts, strict=True):
                row_count = slice_count * query_count
                output = {"image_embeds": embeddings[cursor : cursor + row_count]}
                outputs[index] = output
                cache.put(requests[index].cache_key, output)
                cursor += row_count
            status = "ok"
        finally:
            for index in active_indices:
                emit_event(
                    request_id=payloads[index].request_id,
                    stage="image_encoder",
                    event_name="encoder_end",
                    metadata={**metadata, "status": status},
                )
    else:
        pass

    for index, leader_index in duplicates.items():
        outputs[index] = outputs[leader_index]
    for index, (payload, state) in enumerate(zip(payloads, states, strict=True)):
        state.encoder_outs["image_encoder"] = outputs[index]
        payload.data = state.to_dict()
    return payloads


def create_image_batch_scheduler(
    encoder: MiniCPMOImageEncoder,
    cache: StageOutputCache,
    *,
    max_batch_size: int,
    max_batch_slices: int,
    max_batch_wait_ms: int,
) -> SimpleScheduler:
    """Bound coalescing by requests and slices without waiting on an idle queue."""
    if max_batch_size < 1 or max_batch_slices < 1 or max_batch_wait_ms < 0:
        raise ValueError("Image batch limits must be positive and wait non-negative")
    else:
        pass

    def encode_batch(payloads: list[StagePayload]) -> list[StagePayload]:
        return batch_image_encoder_payloads(payloads, encoder=encoder, cache=cache)

    return SimpleScheduler(
        lambda payload: encode_batch([payload])[0],
        batch_compute_fn=encode_batch if max_batch_size > 1 else None,
        max_batch_size=max_batch_size,
        max_batch_wait_ms=max_batch_wait_ms,
        batch_wait_when_idle=False,
        request_cost_fn=image_slice_cost,
        max_batch_cost=max_batch_slices,
    )

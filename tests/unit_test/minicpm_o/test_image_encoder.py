# SPDX-License-Identifier: Apache-2.0
"""Tests for the srt-based MiniCPM-o image encoder.

The golden-parity test compares the srt-module encoder against the
checkpoint's remote-code path (modeling_navit_siglip.SiglipVisionTransformer
+ modeling_minicpmo.Resampler) on the real checkpoint weights, so it needs
a full checkpoint (weights included) and a CUDA device for the srt vision
attention. Set MINICPMO_CHECKPOINT or place MiniCPM-o-4_6 / MiniCPM-o-4_5
in the repo root; the test skips otherwise.
"""

from __future__ import annotations

import asyncio
import os
import threading
from pathlib import Path

import pytest
import torch

from sglang_omni.models.minicpm_o import stages
from sglang_omni.models.minicpm_o.components.image_encoder import MiniCPMOImageEncoder
from sglang_omni.models.minicpm_o.payload_types import MiniCPMOPipelineState
from sglang_omni.models.minicpm_o.stages import (
    batch_image_encoder_payloads,
    create_image_encoder_executor,
)
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.scheduling.message import IncomingMessage
from sglang_omni.scheduling.stage_cache import StageOutputCache

REPO_ROOT = Path(__file__).resolve().parents[3]


def checkpoint_dir() -> Path | None:
    env = os.environ.get("MINICPMO_CHECKPOINT")
    candidates = [Path(env)] if env else []
    candidates += [REPO_ROOT / "MiniCPM-o-4_6", REPO_ROOT / "MiniCPM-o-4_5"]
    for path in candidates:
        if (path / "model.safetensors.index.json").exists() and (
            path / "modeling_navit_siglip.py"
        ).exists():
            return path
    return None


def test_padding_does_not_change_image_embeddings() -> None:
    batch_size = 2
    encoder = object.__new__(MiniCPMOImageEncoder)
    torch.nn.Module.__init__(encoder)
    encoder.device = torch.device("cpu")
    encoder.dtype = torch.float32
    encoder.vision_batch_size = batch_size

    def run_vpm(pixel_values, patch_attn_mask, tgt_sizes, patch_counts_cpu):
        features = pixel_values.mean(dim=1)
        pooled = (features * patch_attn_mask).sum(dim=-1) / patch_attn_mask.sum(dim=-1)
        return pooled.unsqueeze(-1)

    encoder.run_vpm = run_vpm
    encoder.resampler = lambda features, tgt_sizes: features
    tgt_sizes = torch.tensor([[1, 6], [1, 1], [1, 4]], dtype=torch.int32)
    pixel_values = [
        torch.full((3, 1, count), float(i + 1)) for i, count in enumerate([6, 1, 4])
    ]

    batched = encoder(pixel_values=pixel_values, tgt_sizes=tgt_sizes)["image_embeds"]
    individual = torch.cat(
        [
            encoder(pixel_values=[pixels], tgt_sizes=tgt_sizes[i : i + 1])[
                "image_embeds"
            ]
            for i, pixels in enumerate(pixel_values)
        ]
    )

    torch.testing.assert_close(batched, individual)
    torch.testing.assert_close(batched[:, 0], torch.tensor([1.0, 2.0, 3.0]))


def build_remote_encoder(checkpoint: Path, device: torch.device, dtype: torch.dtype):
    """The pre-srt remote-code path this component replaced, as golden."""
    from transformers import AutoConfig
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    from sglang_omni.models.weight_loader import load_module

    model_dir = str(checkpoint)
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    siglip_cls = get_class_from_dynamic_module(
        "modeling_navit_siglip.SiglipVisionTransformer", model_dir
    )
    resampler_cls = get_class_from_dynamic_module(
        "modeling_minicpmo.Resampler", model_dir
    )
    vision_config = config.vision_config
    vision_config._attn_implementation = (
        "eager"  # noqa: leading-underscore  # production name
    )
    vpm = siglip_cls(vision_config)
    if getattr(config, "drop_vision_last_layer", False):
        vpm.encoder.layers = vpm.encoder.layers[:-1]
    vpm = load_module(vpm, model_dir, prefix=("vpm.",), dtype=dtype, device=str(device))

    embed_dim = config.hidden_size
    resampler = resampler_cls(
        num_queries=config.query_num,
        embed_dim=embed_dim,
        num_heads=embed_dim // 128,
        kv_dim=vision_config.hidden_size,
        adaptive=True,
    )
    resampler = load_module(
        resampler, model_dir, prefix=("resampler.",), dtype=dtype, device=str(device)
    )
    resampler._set_2d_pos_cache(
        resampler.max_size, device=str(device)
    )  # noqa: leading-underscore  # upstream name
    return config, vpm.eval(), resampler.eval()


def remote_forward(vpm, resampler, pixel_values, tgt_sizes, device, dtype):
    from torch.nn.utils.rnn import pad_sequence

    tgt_sizes = tgt_sizes.to(device, dtype=torch.int32)
    all_pixel_values = [
        v.to(device, dtype=dtype).flatten(end_dim=1).permute(1, 0) for v in pixel_values
    ]
    all_pixel_values = pad_sequence(
        all_pixel_values, batch_first=True, padding_value=0.0
    )
    B, L, _ = all_pixel_values.shape
    all_pixel_values = all_pixel_values.permute(0, 2, 1).reshape(B, 3, -1, L)
    patch_counts = tgt_sizes[:, 0] * tgt_sizes[:, 1]
    max_patches = int(patch_counts.max().item())
    patch_attn_mask = torch.zeros((B, 1, max_patches), dtype=torch.bool, device=device)
    for i in range(B):
        patch_attn_mask[i, 0, : patch_counts[i]] = True
    vision_embedding = vpm(
        all_pixel_values,
        patch_attention_mask=patch_attn_mask,
        tgt_sizes=tgt_sizes,
    ).last_hidden_state
    return resampler(vision_embedding, tgt_sizes)


def test_golden_parity_vs_remote_code() -> None:
    checkpoint = checkpoint_dir()
    if checkpoint is None:
        pytest.skip("no MiniCPM-o checkpoint with weights")
    if not torch.cuda.is_available():
        pytest.skip("srt vision attention requires CUDA")

    from sglang_omni.models.minicpm_o.components.image_encoder import (
        MiniCPMOImageEncoder,
    )

    # srt VisionAttention's flash-attn backend only supports fp16/bf16, so the
    # srt encoder cannot run an fp32 bitwise-parity pass. Instead, both the
    # srt path and the remote-code path run in bf16 against an fp32
    # remote-code golden, and the srt path's error must stay within the
    # remote path's own bf16 rounding error (plus slack) — i.e. the module
    # swap adds no error beyond dtype noise.
    device = torch.device("cuda")

    torch.manual_seed(0)
    config, vpm32, resampler32 = build_remote_encoder(checkpoint, device, torch.float32)
    patch = config.vision_config.patch_size
    # Variable-resolution slices (h, w) in patch units, incl. a 1-patch-high one.
    tgt_sizes = torch.tensor([[8, 12], [3, 5], [1, 9]], dtype=torch.int32)
    pixel_values = [
        torch.randn(3, patch, int(h * w) * patch) for h, w in tgt_sizes.tolist()
    ]
    with torch.no_grad():
        golden = remote_forward(
            vpm32, resampler32, pixel_values, tgt_sizes, device, torch.float32
        ).float()
    del vpm32, resampler32
    torch.cuda.empty_cache()

    config, vpm16, resampler16 = build_remote_encoder(
        checkpoint, device, torch.bfloat16
    )
    with torch.no_grad():
        remote_bf16 = remote_forward(
            vpm16, resampler16, pixel_values, tgt_sizes, device, torch.bfloat16
        ).float()
    del vpm16, resampler16
    torch.cuda.empty_cache()

    native = MiniCPMOImageEncoder(str(checkpoint), device="cuda", dtype="bfloat16")
    with torch.no_grad():
        got = (
            native(pixel_values=pixel_values, tgt_sizes=tgt_sizes)["image_embeds"]
            .float()
            .view(golden.shape)
        )

    remote_err = (remote_bf16 - golden).abs()
    native_err = (got - golden).abs()
    assert native_err.mean() <= remote_err.mean() * 1.5, (
        f"srt path error {native_err.mean():.6f} exceeds remote bf16 "
        f"rounding error {remote_err.mean():.6f}"
    )
    cos = torch.nn.functional.cosine_similarity(got, golden, dim=-1)
    remote_cos = torch.nn.functional.cosine_similarity(remote_bf16, golden, dim=-1)
    assert cos.min() >= remote_cos.min() - 0.01, (
        f"srt path cos_min {cos.min():.6f} below remote bf16 "
        f"cos_min {remote_cos.min():.6f}"
    )


class SliceEncoder(MiniCPMOImageEncoder):
    """Deterministic slice outputs for stage routing and cache contracts."""

    def __init__(self) -> None:
        torch.nn.Module.__init__(self)
        self.calls: list[int] = []

    def forward(
        self, *, pixel_values: list[torch.Tensor], tgt_sizes: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        self.calls.append(len(pixel_values))
        for pixels, size in zip(pixel_values, tgt_sizes.tolist(), strict=True):
            assert pixels.shape[-2:] == tuple(size)
        return {
            "image_embeds": torch.stack([pixels.mean() for pixels in pixel_values])
            .repeat_interleave(2)
            .reshape(-1, 1)
        }


def image_payload(
    request_id: str, values: list[float], cache_key: str | None = None
) -> StagePayload:
    state = MiniCPMOPipelineState(
        encoder_inputs={
            "image_encoder": {
                "pixel_values": [
                    torch.full((3, 1, i + 1), v) for i, v in enumerate(values)
                ],
                "tgt_sizes": torch.tensor([[1, i + 1] for i in range(len(values))]),
                "cache_key": cache_key,
            }
        },
        thinker_inputs={"marker": request_id},
    )
    return StagePayload(
        request_id=request_id, request=OmniRequest(inputs={}), data=state.to_dict()
    )


def test_image_batch_preserves_request_and_slice_order() -> None:
    encoder = SliceEncoder()
    payloads = [
        image_payload("a", [1.0, 2.0]),
        image_payload("b", [3.0]),
        image_payload("c", [4.0, 5.0, 6.0]),
    ]
    outputs = batch_image_encoder_payloads(
        payloads, encoder=encoder, cache=StageOutputCache(cache_device="cpu")
    )
    expected = {
        "a": [1.0, 1.0, 2.0, 2.0],
        "b": [3.0, 3.0],
        "c": [4.0, 4.0, 5.0, 5.0, 6.0, 6.0],
    }
    assert [payload.request_id for payload in outputs] == list(expected)
    for payload in outputs:
        assert payload.data["thinker_inputs"]["marker"] == payload.request_id
        torch.testing.assert_close(
            payload.data["encoder_outs"]["image_encoder"]["image_embeds"],
            torch.tensor(expected[payload.request_id]).reshape(-1, 1),
        )
    assert encoder.calls == [6]


def test_image_batch_cache_dedup_and_skip() -> None:
    encoder = SliceEncoder()
    cache = StageOutputCache(cache_device="cpu")
    cache.put("cached", {"image_embeds": torch.full((2, 1), 9.0)})
    skipped = StagePayload(request_id="skip", request=OmniRequest(inputs={}), data={})
    payloads = [
        image_payload("a", [1.0, 2.0], "same"),
        image_payload("hit", [9.0], "cached"),
        skipped,
        image_payload("duplicate", [1.0, 2.0], "same"),
        image_payload("unkeyed", [3.0]),
    ]
    batch_image_encoder_payloads(payloads, encoder=encoder, cache=cache)
    assert encoder.calls == [3]
    assert skipped.data["encoder_outs"]["image_encoder"] == {}
    for index, values in (
        (0, [1, 1, 2, 2]),
        (1, [9, 9]),
        (3, [1, 1, 2, 2]),
        (4, [3, 3]),
    ):
        torch.testing.assert_close(
            payloads[index].data["encoder_outs"]["image_encoder"]["image_embeds"],
            torch.tensor(values, dtype=torch.float32).reshape(-1, 1),
        )
    batch_image_encoder_payloads(
        [image_payload("later", [1.0, 2.0], "same"), skipped],
        encoder=encoder,
        cache=cache,
    )
    assert encoder.calls == [3]


def test_image_scheduler_shares_cache_between_single_and_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoder = SliceEncoder()
    monkeypatch.setattr(
        stages, "MiniCPMOImageEncoder", lambda model_path, device, dtype: encoder
    )
    scheduler = create_image_encoder_executor(
        "unused",
        device="cpu",
        max_batch_size=8,
        max_batch_slices=64,
        max_batch_wait_ms=0,
    )
    single = [image_payload("single-a", [1.0, 2.0], "a")]
    batch = [
        image_payload("batch-a", [1.0, 2.0], "a"),
        image_payload("batch-b", [3.0], "b"),
    ]
    groups = [single, batch, [image_payload("cached-b", [3.0], "b")]]
    expected = {
        "single-a": [1.0, 1.0, 2.0, 2.0],
        "batch-a": [1.0, 1.0, 2.0, 2.0],
        "batch-b": [3.0, 3.0],
        "cached-b": [3.0, 3.0],
    }
    loop = asyncio.new_event_loop()
    try:
        for group in groups:
            scheduler.run_batch(
                [
                    IncomingMessage(
                        type="new_request", request_id=payload.request_id, data=payload
                    )
                    for payload in group
                ],
                loop,
            )
            for payload in group:
                output = scheduler.outbox.get_nowait()
                assert output.type == "result"
                assert output.request_id == payload.request_id
                state = MiniCPMOPipelineState.from_dict(output.data.data)
                torch.testing.assert_close(
                    state.encoder_outs["image_encoder"]["image_embeds"],
                    torch.tensor(expected[payload.request_id]).reshape(-1, 1),
                )
    finally:
        loop.close()
    assert encoder.calls == [2, 1]


@pytest.mark.parametrize("max_batch_size,expected_calls", [(1, [2, 3, 1]), (8, [2, 4])])
def test_image_scheduler_respects_slice_budget(
    max_batch_size: int, expected_calls: list[int], monkeypatch: pytest.MonkeyPatch
) -> None:
    encoder = SliceEncoder()
    monkeypatch.setattr(
        stages, "MiniCPMOImageEncoder", lambda model_path, device, dtype: encoder
    )
    scheduler = create_image_encoder_executor(
        "unused",
        device="cpu",
        max_batch_size=max_batch_size,
        max_batch_slices=4,
        max_batch_wait_ms=0,
    )
    for index, values in enumerate(([1.0, 2.0], [3.0, 4.0, 5.0], [6.0])):
        payload = image_payload(str(index), values)
        scheduler.inbox.put(
            IncomingMessage(type="new_request", request_id=str(index), data=payload)
        )
    worker = threading.Thread(target=scheduler.start, daemon=True)
    worker.start()
    try:
        outputs = [scheduler.outbox.get(timeout=5) for _ in range(3)]
    finally:
        scheduler.stop()
        worker.join(timeout=5)
    assert not worker.is_alive()
    assert [output.request_id for output in outputs] == ["0", "1", "2"]
    assert all(output.type == "result" for output in outputs)
    assert encoder.calls == expected_calls


@pytest.fixture(scope="module")
def batching_native_encoder() -> MiniCPMOImageEncoder:
    checkpoint = checkpoint_dir()
    if checkpoint is None or not torch.cuda.is_available():
        pytest.skip("cross-request numerical parity requires a checkpoint and CUDA")
    else:
        pass
    return MiniCPMOImageEncoder(str(checkpoint), device="cuda", dtype="bfloat16")


@pytest.mark.accelerator
@pytest.mark.parametrize("chunk_size", [1, 2, 16])
@pytest.mark.parametrize("slice_counts", [(1, 1), (2, 3), (9, 8)])
def test_cross_request_checkpoint_parity(
    batching_native_encoder: MiniCPMOImageEncoder,
    chunk_size: int,
    slice_counts: tuple[int, int],
    request: pytest.FixtureRequest,
) -> None:
    encoder = batching_native_encoder
    encoder.vision_batch_size = chunk_size
    generator = torch.Generator().manual_seed(20260926)
    patch_size = encoder.vpm.embeddings.patch_size
    payloads = []
    expected = []
    for index, slice_count in enumerate(slice_counts):
        payload = image_payload(str(index), [1.0] * slice_count)
        inputs = payload.data["encoder_inputs"]["image_encoder"]
        sizes = torch.tensor(
            [(8, 12), (3, 5), (1, 9)] * slice_count, dtype=torch.int32
        )[:slice_count]
        inputs["tgt_sizes"] = sizes
        inputs["pixel_values"] = [
            torch.randn(
                3, patch_size, int(height * width) * patch_size, generator=generator
            )
            for height, width in sizes.tolist()
        ]
        expected.append(
            encoder(pixel_values=inputs["pixel_values"], tgt_sizes=sizes)[
                "image_embeds"
            ].clone()
        )
        repeat = encoder(pixel_values=inputs["pixel_values"], tgt_sizes=sizes)[
            "image_embeds"
        ]
        request.node.user_properties.append(
            (
                f"request_{index}_aa_max_abs",
                float((repeat.float() - expected[-1].float()).abs().max()),
            )
        )
        torch.testing.assert_close(repeat, expected[-1], rtol=0, atol=0)
        payloads.append(payload)
    batch_image_encoder_payloads(payloads, encoder=encoder, cache=StageOutputCache())
    combined = encoder(
        pixel_values=[
            pixels
            for payload in payloads
            for pixels in payload.data["encoder_inputs"]["image_encoder"][
                "pixel_values"
            ]
        ],
        tgt_sizes=torch.cat(
            [
                payload.data["encoder_inputs"]["image_encoder"]["tgt_sizes"]
                for payload in payloads
            ]
        ),
    )["image_embeds"]
    reconstructed = torch.cat(
        [
            payload.data["encoder_outs"]["image_encoder"]["image_embeds"]
            for payload in payloads
        ]
    )
    request.node.user_properties.append(
        ("split_max_abs", float((combined.float() - reconstructed.float()).abs().max()))
    )
    torch.testing.assert_close(reconstructed, combined, rtol=0, atol=0)
    for payload, serial in zip(payloads, expected, strict=True):
        batched = payload.data["encoder_outs"]["image_encoder"]["image_embeds"]
        assert torch.isfinite(batched).all()
        difference = (batched.float() - serial.float()).abs()
        cosine = torch.nn.functional.cosine_similarity(
            batched.float(), serial.float(), dim=-1
        )
        request.node.user_properties.extend(
            [
                (f"request_{payload.request_id}_max_abs", float(difference.max())),
                (f"request_{payload.request_id}_mean_abs", float(difference.mean())),
                (f"request_{payload.request_id}_cosine_min", float(cosine.min())),
            ]
        )
    for payload, serial in zip(payloads, expected, strict=True):
        torch.testing.assert_close(
            payload.data["encoder_outs"]["image_encoder"]["image_embeds"],
            serial,
            rtol=0.02,
            atol=0.02,
        )

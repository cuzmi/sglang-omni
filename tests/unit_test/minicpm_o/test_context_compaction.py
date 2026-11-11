# SPDX-License-Identifier: Apache-2.0
"""Compaction commits new history only after prefill and preserves queued input on rollback."""

from array import array
from queue import Queue
from unittest.mock import Mock

import pytest
import torch
from sglang.srt.managers.io_struct import SessionParams, TokenizedGenerateReqInput
from sglang.srt.managers.schedule_batch import FINISH_LENGTH, Req
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.session.session_controller import SessionController

from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.proto.session import SessionIdentity, SessionOperation, TimedChunk
from sglang_omni.scheduling.sglang_backend.ar_session import (
    ARSessionAdapter,
    ARSessionBridge,
    ARSessionCompactionPlan,
    BridgeSession,
)
from sglang_omni.scheduling.sglang_backend.request_data import (
    EmbeddingSpan,
    SGLangARRequestData,
)

IDENTITY = SessionIdentity("external")
HISTORY_IDS = [11, 0, 0, 12, 40, 0, 0, 41]
REBUILT_IDS = [11, 0, 0, 12, 31, 32]


def tokenized(request_id: str, token_ids: list[int]) -> TokenizedGenerateReqInput:
    return TokenizedGenerateReqInput(
        rid=request_id,
        input_text=None,
        input_ids=array("q", token_ids),
        input_embeds=None,
        mm_inputs=None,
        token_type_ids=None,
        sampling_params=SamplingParams(max_new_tokens=3),
        return_logprob=False,
        logprob_start_len=-1,
        top_logprobs_num=0,
        token_ids_logprob=None,
        stream=False,
        session_params=SessionParams(id=IDENTITY.id),
    )


def request_data(request_id: str, token_ids: list[int]) -> SGLangARRequestData:
    request = Req(
        request_id,
        "",
        array("q", token_ids),
        SamplingParams(max_new_tokens=3),
        vocab_size=128,
    )
    request.tokenizer = None
    return SGLangARRequestData(
        req=request,
        stage_payload=StagePayload(request_id, OmniRequest(None), None),
    )


def compaction_bridge() -> tuple[ARSessionBridge, Mock, StagePayload]:
    cache = Mock()
    cache.session_records.return_value = {}
    scheduler = Mock(max_req_len=128)
    scheduler.model_config.vocab_size = 128
    scheduler.session_controller = SessionController(cache)
    scheduler.tree_cache = cache
    scheduler.outbox = Queue()
    scheduler.deferred_request_payloads = {}
    scheduler.dirty_deferred_request_ids = set()
    adapter = ARSessionAdapter()
    adapter.compaction_enabled = True
    adapter.finish_compaction = Mock()
    bridge = ARSessionBridge(scheduler, adapter)
    assert bridge.open_streaming_session(IDENTITY.id)
    native = scheduler.session_controller.get(IDENTITY.id)
    last_request = native.create_req(tokenized("previous", HISTORY_IDS), None, 128)
    last_request.output_ids.extend([21, 22])
    last_request.finished_reason = FINISH_LENGTH(2)
    last_request.refresh_fill_ids()
    native.finish_req(last_request)
    last_request.origin_input_ids.append(99)
    spans = [
        EmbeddingSpan(start=1, end=3, input_embeds=torch.full((2, 2), 7.0)),
        EmbeddingSpan(start=5, end=7, input_embeds=torch.full((2, 2), 8.0)),
    ]
    bridge.sessions[IDENTITY.id] = BridgeSession(
        session_identity=IDENTITY, embedding_spans=spans
    )
    payload = StagePayload("pending", OmniRequest(None), None)
    unit = bridge.accept(
        payload,
        SessionOperation(
            "append",
            IDENTITY,
            ("thinker",),
            TimedChunk("audio", 1000, 1000, 1, b"\x00\x00"),
        ),
    )
    unit.is_prepared = True
    summary = request_data("pending:summary", [*HISTORY_IDS, 21, 22, 50])
    adapter.plan_compaction = Mock(
        return_value=ARSessionCompactionPlan(
            request_data=summary,
            timeout_seconds=20,
            continuation_tokens=6,
        )
    )
    rebuild = request_data("pending:summary:rebuild", REBUILT_IDS)
    rebuild.unit_embedding_spans = spans[:1]
    adapter.build_compacted_history = Mock(return_value=rebuild)
    assert bridge.start_compaction(unit, payload)
    assert scheduler.outbox.empty()
    return bridge, scheduler, payload


def finish_summary(bridge: ARSessionBridge) -> None:
    transaction = bridge.sessions[IDENTITY.id].compaction
    summary = transaction.request_data
    summary.output_ids = [31, 32]
    summary.finish_reason = "stop"
    bridge.finish_internal_request(summary.req.rid, summary)


def test_snapshot_keeps_latest_output_and_media_without_uncommitted_ids() -> None:
    bridge, scheduler, payload = compaction_bridge()
    session = bridge.sessions[IDENTITY.id]
    history = bridge.adapter.plan_compaction.call_args.args[2]
    assert history.token_ids == [*HISTORY_IDS, 21, 22]
    assert len(history.embedding_spans) == 2
    assert torch.equal(
        history.embedding_spans[1].input_embeds,
        torch.full((2, 2), 8.0),
    )
    assert session.unit.session_request is None
    assert scheduler.session_controller.get(IDENTITY.id) is not None
    assert payload.request_id not in scheduler.deferred_request_payloads
    assert set(bridge.units_by_request_id) == {payload.request_id}


def test_prefill_commit_swaps_history_and_resumes_pending_input_once() -> None:
    bridge, scheduler, payload = compaction_bridge()
    finish_summary(bridge)
    session = bridge.sessions[IDENTITY.id]
    transaction = session.compaction
    rebuild = transaction.request_data
    shadow = scheduler.session_controller.get(transaction.shadow_session_id)
    assert scheduler.session_controller.get(IDENTITY.id) is not None
    assert payload.request_id not in scheduler.deferred_request_payloads
    rebuild.req.output_ids.append(77)
    rebuild.req.finished_reason = FINISH_LENGTH(1)
    rebuild.finish_reason = "length"
    rebuild.req.refresh_fill_ids()
    shadow.finish_req(rebuild.req)

    bridge.finish_internal_request(rebuild.req.rid, rebuild)

    assert session.session_identity == IDENTITY
    assert scheduler.session_controller.get(IDENTITY.id) is None
    assert session.native_session_id == shadow.session_id
    assert session.embedding_spans == rebuild.unit_embedding_spans
    assert len(session.embedding_spans) == 1
    assert scheduler.deferred_request_payloads == {payload.request_id: payload}
    assert scheduler.dirty_deferred_request_ids == {payload.request_id}
    assert not bridge.compactions
    assert not bridge.start_compaction(session.unit, payload)
    pending = request_data(payload.request_id, [61, 62])
    bridge.create_session_request(payload, pending)
    assert list(pending.req.origin_input_ids) == [*REBUILT_IDS, 61, 62]
    assert [span.start for span in pending.session_embedding_spans] == [1]
    assert scheduler.outbox.empty()


@pytest.mark.parametrize("failure", ["prefill_not_committed", "prefill_failed"])
def test_failed_prefill_keeps_old_session_and_pending_input(
    failure: str,
) -> None:
    bridge, scheduler, payload = compaction_bridge()
    session = bridge.sessions[IDENTITY.id]
    old_session = scheduler.session_controller.get(IDENTITY.id)
    finish_summary(bridge)
    rebuild = session.compaction.request_data
    if failure == "prefill_failed":
        shadow = scheduler.session_controller.get(session.compaction.shadow_session_id)
        rebuild.req.output_ids.append(77)
        rebuild.req.finished_reason = FINISH_LENGTH(1)
        rebuild.req.refresh_fill_ids()
        shadow.finish_req(rebuild.req)
        rebuild.finish_reason = "abort"
    else:
        rebuild.finish_reason = "length"
    bridge.finish_internal_request(rebuild.req.rid, rebuild)
    assert session.native_session_id == IDENTITY.id
    assert scheduler.session_controller.get(IDENTITY.id) is old_session
    assert list(old_session.req_nodes["previous"].req.origin_input_ids) == [
        *HISTORY_IDS,
        99,
    ]
    assert session.compaction is None
    assert not bridge.compactions
    assert scheduler.deferred_request_payloads == {payload.request_id: payload}
    assert session.unit.is_prepared is False
    assert set(scheduler.session_controller.sessions) == {IDENTITY.id}
    assert scheduler.outbox.empty()

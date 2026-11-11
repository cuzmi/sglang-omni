# SPDX-License-Identifier: Apache-2.0
"""Compaction commits new history only after prefill and preserves queued input on rollback."""

from array import array
from queue import Queue
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from sglang.srt.managers.io_struct import SessionParams, TokenizedGenerateReqInput
from sglang.srt.managers.schedule_batch import FINISH_LENGTH, Req
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.session.session_controller import SessionController
from tokenizers import Regex, Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Split
from transformers import PreTrainedTokenizerFast

from sglang_omni.admission import ContextExhaustedError
from sglang_omni.models.minicpm_o.components.streaming_perception import (
    PerceptionStepPlan,
    SummaryAudioTask,
)
from sglang_omni.models.minicpm_o.context_compaction import MiniCPMOContextMemory
from sglang_omni.models.minicpm_o.native_config import (
    MiniCPMODuplexSampling,
    MiniCPMODuplexWindow,
)
from sglang_omni.models.minicpm_o.session_adapters import ThinkerAdapter
from sglang_omni.models.minicpm_o.special_tokens import REQUIRED_SPECIAL_TOKENS
from sglang_omni.models.minicpm_o.thinker_state import (
    MiniCPMOThinkerSessionState,
    ReplyUnit,
)
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.proto.session import SessionIdentity, SessionOperation, TimedChunk
from sglang_omni.proto.streams import CONTROL, INTERRUPT, SESSION_CONTROL
from sglang_omni.scheduling.sglang_backend.ar_session import (
    ARSessionAdapter,
    ARSessionBridge,
    ARSessionCompactionPlan,
    ARSessionHistory,
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


@pytest.mark.parametrize(
    "failure",
    [
        "prefill_not_committed",
        "prefill_failed",
        "control",
        "timeout",
        "pending_input_budget",
    ],
)
def test_failed_or_invalidated_compaction_keeps_old_session_and_pending_input(
    failure: str,
) -> None:
    bridge, scheduler, payload = compaction_bridge()
    session = bridge.sessions[IDENTITY.id]
    old_session = scheduler.session_controller.get(IDENTITY.id)
    if failure == "control":
        event = TimedChunk(
            CONTROL, 0, 0, 0, {"type": INTERRUPT}, format=SESSION_CONTROL
        )
        bridge.apply_operation(
            StagePayload("control", OmniRequest(None), None),
            SessionOperation("control", IDENTITY, ("thinker",), event),
        )
        finish_summary(bridge)
    elif failure == "timeout":
        session.compaction.deadline = 0
        finish_summary(bridge)
    elif failure == "pending_input_budget":
        session.compaction.continuation_tokens = scheduler.max_req_len
        finish_summary(bridge)
    else:
        finish_summary(bridge)
        rebuild = session.compaction.request_data
        if failure == "prefill_failed":
            shadow = scheduler.session_controller.get(
                session.compaction.shadow_session_id
            )
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


def test_summary_enqueue_exception_releases_transaction_before_input_error() -> None:
    """An enqueue exception left a compaction alive after its pending input was removed."""
    bridge, scheduler, payload = compaction_bridge()
    session = bridge.sessions[IDENTITY.id]
    bridge.fail_compaction(
        session.compaction.request_data.req.rid, "setup", resume=False
    )
    session.unit.compaction_attempted = False
    scheduler.enqueue_built_request.side_effect = RuntimeError(
        "request initialization failed"
    )

    with pytest.raises(RuntimeError, match="request initialization failed"):
        bridge.start_compaction(session.unit, payload)

    assert session.compaction is None
    assert not bridge.compactions
    assert session.native_session_id == IDENTITY.id
    assert scheduler.session_controller.get(IDENTITY.id) is not None
    assert payload.request_id not in scheduler.deferred_request_payloads
    assert payload.request_id not in scheduler.dirty_deferred_request_ids
    assert scheduler.outbox.empty()


@pytest.mark.parametrize("outcome", ["complete", "control", "timeout", "error"])
def test_summary_continuations_keep_input_parked_and_share_one_deadline(
    outcome: str,
) -> None:
    """A completed native chunk must not commit a partial summary or restart its timeout."""
    bridge, scheduler, payload = compaction_bridge()
    session = bridge.sessions[IDENTITY.id]
    continuation = request_data("pending:summary:unit:1", [*HISTORY_IDS, 31, 32, 50])
    bridge.adapter.continue_compaction = Mock(side_effect=[continuation, None])
    deadline = session.compaction.deadline

    finish_summary(bridge)

    assert session.compaction.request_data is continuation
    assert session.compaction.deadline == deadline
    assert session.compaction.phase == "summary"
    assert set(bridge.compactions) == {continuation.req.rid}
    bridge.adapter.build_compacted_history.assert_not_called()
    assert payload.request_id not in scheduler.deferred_request_payloads
    assert scheduler.outbox.empty()
    assert scheduler.enqueue_built_request.call_args.args == (
        continuation.stage_payload,
        False,
        continuation,
    )
    if outcome == "complete":
        bridge.finish_internal_request(continuation.req.rid, continuation)
        assert session.compaction.phase == "rebuild"
        assert session.compaction.deadline == deadline
        assert payload.request_id not in scheduler.deferred_request_payloads
        bridge.adapter.build_compacted_history.assert_called_once_with(
            IDENTITY, continuation, scheduler.max_req_len
        )
    else:
        if outcome == "control":
            event = TimedChunk(
                CONTROL, 0, 0, 0, {"type": INTERRUPT}, format=SESSION_CONTROL
            )
            bridge.apply_operation(
                StagePayload("control", OmniRequest(None), None),
                SessionOperation("control", IDENTITY, ("thinker",), event),
            )
        elif outcome == "timeout":
            session.compaction.deadline = 0
        else:
            bridge.adapter.continue_compaction.side_effect = ValueError(
                "incomplete summary"
            )
        bridge.finish_internal_request(continuation.req.rid, continuation)
        assert session.compaction is None
        assert not bridge.compactions
        assert session.native_session_id == IDENTITY.id
        assert scheduler.session_controller.get(IDENTITY.id) is not None
        assert scheduler.deferred_request_payloads == {payload.request_id: payload}
        assert session.unit.compaction_attempted is (outcome != "control")
        bridge.adapter.build_compacted_history.assert_not_called()
        assert scheduler.outbox.empty()


def test_summary_reserve_overflow_does_not_cut_history() -> None:
    """Prioritizing summary must still reject a snapshot that cannot fit its output."""
    adapter = thinker_adapter()
    adapter.window = MiniCPMODuplexWindow(
        sliding_window_mode="summary", summary_trigger_tokens=30, summary_max_tokens=5
    )
    state = adapter.states[IDENTITY]
    state.is_turn_ended = False
    payload = StagePayload("pending", OmniRequest(None), {"token_ids": [1, 2]})
    history = ARSessionHistory(token_ids=list(range(57)), embedding_spans=[])
    with pytest.raises(ContextExhaustedError, match="summary request would exceed"):
        adapter.plan_compaction(IDENTITY, payload, history, 64)
    assert history.token_ids == list(range(57))
    assert state.unit_starts == [4]


def summary_audio_task(
    adapter: ThinkerAdapter,
    *,
    unit_count: int = 3,
    instruction_units: int = 1,
    audio_tokens: int = 2,
) -> SummaryAudioTask:
    plans: list[PerceptionStepPlan] = [
        {
            "token_ids": [adapter.special.unit, *([0] * audio_tokens)],
            "input_embeds": torch.full((audio_tokens, 2), float(index + 20)),
            "embedding_spans": [
                {
                    "modality": "audio",
                    "token_start": 1,
                    "token_end": audio_tokens + 1,
                    "embedding_start": 0,
                    "embedding_end": audio_tokens,
                }
            ],
        }
        for index in range(unit_count)
    ]
    return {"instruction_units": instruction_units, "plans": plans}


def thinker_adapter() -> ThinkerAdapter:
    vocabulary = {
        token: index
        for index, token in enumerate(
            (
                "<unk>",
                *REQUIRED_SPECIAL_TOKENS,
                "<eos>",
                "<|im_start|>",
                "<|im_end|>",
                "<|endoftext|>",
                "raw_fact",
                "summary_fact",
                "new_fact",
                "\n",
            )
        )
    }
    backend = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
    backend.pre_tokenizer = Split(Regex(r"[^\S\n]+"), behavior="removed")
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        eos_token="<eos>",
        additional_special_tokens=[
            *REQUIRED_SPECIAL_TOKENS,
            "<|im_start|>",
            "<|im_end|>",
            "<|endoftext|>",
        ],
    )
    tokenizer.bad_token_ids = []
    adapter = ThinkerAdapter(
        tokenizer,
        len(vocabulary),
        window=MiniCPMODuplexWindow(
            sliding_window_mode="summary",
            summary_trigger_tokens=80,
            summary_max_tokens=8,
        ),
        rotary_embedding=SimpleNamespace(is_neox_style=True, rotary_dim=4, head_size=4),
    )
    state = MiniCPMOThinkerSessionState(
        sampling=MiniCPMODuplexSampling(), speech_detector=Mock()
    )
    state.unit_starts = [4]
    state.is_prefix_pending = False
    adapter.states[IDENTITY] = state
    adapter.context_memories[IDENTITY] = MiniCPMOContextMemory(
        summary_audio=summary_audio_task(adapter)
    )
    return adapter


@pytest.mark.parametrize("invalid_summary", [None, "empty", "length"])
def test_model_builders_replace_all_history_and_reuse_previous_summary(
    invalid_summary: str | None,
) -> None:
    adapter = thinker_adapter()
    state = adapter.states[IDENTITY]
    tokenizer = adapter.tokenizer
    vocabulary = tokenizer.get_vocab()
    prefix_ids = [
        vocabulary["<|im_start|>"],
        vocabulary["<|audio_start|>"],
        0,
        vocabulary["<|im_end|>"],
    ]
    prefix_span = EmbeddingSpan(start=2, end=3, input_embeds=torch.full((1, 2), 3.0))
    user_span = EmbeddingSpan(start=5, end=7, input_embeds=torch.full((2, 2), 9.0))
    history = ARSessionHistory(
        token_ids=[*prefix_ids, *([vocabulary["raw_fact"]] * 86)],
        embedding_spans=[prefix_span, user_span],
    )
    payload = StagePayload(
        "pending", OmniRequest(None), {"token_ids": [adapter.special.unit, 0]}
    )
    plan = adapter.plan_compaction(IDENTITY, payload, history, 1024)
    assert plan is not None
    summary = plan.request_data
    assert (
        set(summary.req.sampling_params.stop_token_ids)
        == adapter.special.chunk_terminators
    )
    assert not summary.generate_audio
    assert summary.thinker_state is not state
    assert summary.thinker_state.sampling is state.sampling
    assert list(summary.req.origin_input_ids) == [
        *history.token_ids,
        adapter.special.unit_end,
        *adapter.context_memories[IDENTITY].summary_audio["plans"][0]["token_ids"],
    ]
    assert summary.session_embedding_spans[:-1] == history.embedding_spans
    assert summary.unit_embedding_spans == summary.session_embedding_spans
    instruction_span = summary.session_embedding_spans[-1]
    assert (instruction_span.start, instruction_span.end) == (92, 94)
    torch.testing.assert_close(instruction_span.input_embeds, torch.full((2, 2), 20.0))
    summary.generated_unit_ids = [vocabulary["summary_fact"], adapter.special.turn_eos]
    summary.output_ids = [
        adapter.special.speak,
        *summary.generated_unit_ids,
        adapter.special.chunk_eos,
    ]
    summary.finish_reason = "stop"
    if invalid_summary == "empty":
        summary.generated_unit_ids = [adapter.special.turn_eos]
        summary.output_ids = [adapter.special.turn_eos, adapter.special.chunk_eos]
    elif invalid_summary == "length":
        summary.output_ids = summary.output_ids[:-1]
        summary.finish_reason = "length"
    else:
        pass
    if invalid_summary is not None:
        with pytest.raises(ValueError):
            adapter.continue_compaction(IDENTITY, summary, 1024)
            adapter.build_compacted_history(IDENTITY, summary, 1024)
        assert history.token_ids == [*prefix_ids, *([vocabulary["raw_fact"]] * 86)]
        return
    else:
        pass
    assert adapter.continue_compaction(IDENTITY, summary, 1024) is None
    rebuilt = adapter.build_compacted_history(IDENTITY, summary, 1024)
    rebuilt_ids = list(rebuilt.req.origin_input_ids)
    assert rebuilt_ids[:4] == prefix_ids
    assert rebuilt_ids.count(vocabulary["summary_fact"]) == 1
    assert vocabulary["raw_fact"] not in rebuilt_ids
    assert len(rebuilt.unit_embedding_spans) == 1
    torch.testing.assert_close(
        rebuilt.unit_embedding_spans[0].input_embeds, prefix_span.input_embeds
    )
    assert (
        rebuilt.unit_embedding_spans[0].input_embeds.data_ptr()
        != prefix_span.input_embeds.data_ptr()
    )
    adapter.finish_compaction(IDENTITY, status="completed")
    assert (
        state.is_prefix_pending
        and not state.generated_history
        and not state.unit_starts
    )
    state.is_prefix_pending = False
    state.unit_starts = [len(rebuilt_ids)]
    later_history = ARSessionHistory(
        token_ids=[*rebuilt_ids, *([vocabulary["new_fact"]] * 80)],
        embedding_spans=list(rebuilt.unit_embedding_spans),
    )
    later_plan = adapter.plan_compaction(IDENTITY, payload, later_history, 1024)
    assert later_plan is not None
    later_ids = list(later_plan.request_data.req.origin_input_ids)
    assert later_ids[: len(later_history.token_ids)] == later_history.token_ids
    assert (
        vocabulary["summary_fact"] in later_ids and vocabulary["new_fact"] in later_ids
    )
    assert vocabulary["raw_fact"] not in later_ids


@pytest.mark.parametrize("ending", ["complete", "budget"])
def test_native_summary_collects_chunks_without_mutating_live_state(
    ending: str,
) -> None:
    """Chunk completion must preserve all summary text and leave rollback state untouched."""
    adapter = thinker_adapter()
    state = adapter.states[IDENTITY]
    vocabulary = adapter.tokenizer.get_vocab()
    state.generated_history = [vocabulary["raw_fact"]]
    state.is_turn_ended = True
    user_span = EmbeddingSpan(start=5, end=7, input_embeds=torch.full((2, 2), 9.0))
    history = ARSessionHistory(token_ids=[0] * 90, embedding_spans=[user_span])
    payload = StagePayload(
        "pending", OmniRequest(None), {"token_ids": [adapter.special.unit, 0]}
    )
    plan = adapter.plan_compaction(IDENTITY, payload, history, 1024)
    first_chunk = plan.request_data
    first_chunk.generated_unit_ids = [vocabulary["summary_fact"]]
    first_chunk.output_ids = [
        adapter.special.speak,
        vocabulary["summary_fact"],
        adapter.special.chunk_eos,
    ]
    first_chunk.finish_reason = "stop"
    first_chunk.thinker_state.generated_history.append(vocabulary["summary_fact"])
    first_chunk.thinker_state.is_turn_ended = False

    second_chunk = adapter.continue_compaction(IDENTITY, first_chunk, 1024)

    assert second_chunk is not None
    assert second_chunk.thinker_state is first_chunk.thinker_state
    assert second_chunk.req.rid != first_chunk.req.rid
    assert not second_chunk.generate_audio
    assert list(second_chunk.req.origin_input_ids) == [
        *first_chunk.req.origin_input_ids,
        *first_chunk.output_ids,
        adapter.special.unit_end,
        *adapter.context_memories[IDENTITY].summary_audio["plans"][1]["token_ids"],
    ]
    assert second_chunk.session_embedding_spans[0] is user_span
    assert len(second_chunk.session_embedding_spans) == 3
    for index, span in enumerate(second_chunk.session_embedding_spans[1:]):
        torch.testing.assert_close(
            span.input_embeds, torch.full((2, 2), float(index + 20))
        )
        assert list(second_chunk.req.origin_input_ids[span.start : span.end]) == [0, 0]
    assert (
        second_chunk.session_embedding_spans[-1].start
        == len(first_chunk.req.origin_input_ids) + len(first_chunk.output_ids) + 2
    )
    assert second_chunk.max_new_tokens == adapter.window.summary_max_tokens - len(
        first_chunk.output_ids
    )
    assert not adapter.context_memories[IDENTITY].summary.is_complete
    assert state.generated_history == [vocabulary["raw_fact"]]
    assert state.is_turn_ended
    if ending == "complete":
        second_chunk.generated_unit_ids = [
            vocabulary["new_fact"],
            adapter.special.turn_eos,
        ]
        second_chunk.output_ids = [
            adapter.special.tts_bos,
            *second_chunk.generated_unit_ids,
            adapter.special.chunk_eos,
        ]
        second_chunk.finish_reason = "stop"
        assert adapter.continue_compaction(IDENTITY, second_chunk, 1024) is None
        rebuilt = adapter.build_compacted_history(IDENTITY, second_chunk, 1024)
        assert rebuilt.req.origin_input_ids.count(vocabulary["summary_fact"]) == 1
        assert rebuilt.req.origin_input_ids.count(vocabulary["new_fact"]) == 1
        assert not rebuilt.generate_audio
    else:
        second_chunk.generated_unit_ids = [vocabulary["new_fact"]] * 3
        second_chunk.output_ids = [
            adapter.special.tts_bos,
            *second_chunk.generated_unit_ids,
            adapter.special.chunk_eos,
        ]
        second_chunk.finish_reason = "stop"
        with pytest.raises(ValueError, match="budget"):
            adapter.continue_compaction(IDENTITY, second_chunk, 1024)
        adapter.finish_compaction(IDENTITY, status="failed")
        assert adapter.context_memories[IDENTITY].summary is None
        assert state.generated_history == [vocabulary["raw_fact"]]
        assert state.is_turn_ended


def test_first_input_caches_owned_summary_audio() -> None:
    """Later transport-buffer changes must not rewrite a future spoken summary instruction."""
    adapter = thinker_adapter()
    memory = adapter.context_memories[IDENTITY]
    memory.summary_audio = None
    audio_task = summary_audio_task(adapter, instruction_units=2)
    payload = StagePayload(
        "first",
        OmniRequest(None),
        {
            "token_ids": [adapter.special.unit, 0],
            "input_embeds": torch.ones((1, 2)),
            "embedding_spans": [
                {
                    "modality": "audio",
                    "token_start": 1,
                    "token_end": 2,
                    "embedding_start": 0,
                    "embedding_end": 1,
                }
            ],
            "summary_audio": audio_task,
        },
    )
    request = adapter.build(
        IDENTITY, TimedChunk("audio", 0, 1000, 0, b"\x00\x00"), payload
    )
    assert request.generate_audio
    assert len(request.unit_embedding_spans) == 1
    assert memory.summary_audio is not audio_task
    assert memory.summary_audio["instruction_units"] == 2
    for index, audio_plan in enumerate(audio_task["plans"]):
        audio_plan["input_embeds"].fill_(-1)
        torch.testing.assert_close(
            memory.summary_audio["plans"][index]["input_embeds"],
            torch.full((2, 2), float(index + 20)),
        )
    assert list(request.req.origin_input_ids) == [
        adapter.special.unit_end,
        adapter.special.unit,
        0,
    ]


@pytest.mark.parametrize("early_output", ["listen", "turn_end", "unfinished_reply"])
def test_spoken_instruction_finishes_before_accepting_summary(
    early_output: str,
) -> None:
    """Listening and old reply endings cannot skip the remainder of a spoken instruction."""
    adapter = thinker_adapter()
    adapter.window.summary_max_tokens = 20
    audio_task = summary_audio_task(adapter, instruction_units=3, unit_count=5)
    adapter.context_memories[IDENTITY].summary_audio = audio_task
    vocabulary = adapter.tokenizer.get_vocab()
    history = ARSessionHistory(token_ids=[0] * 90, embedding_spans=[])
    payload = StagePayload(
        "pending",
        OmniRequest(None),
        {"token_ids": [adapter.special.unit, vocabulary["new_fact"]]},
    )
    current = adapter.plan_compaction(IDENTITY, payload, history, 1024).request_data
    if early_output == "listen":
        current.generated_unit_ids = []
        current.output_ids = [adapter.special.listen]
    else:
        current.generated_unit_ids = [vocabulary["raw_fact"]]
        current.output_ids = [adapter.special.speak, vocabulary["raw_fact"]]
        if early_output == "turn_end":
            current.generated_unit_ids.append(adapter.special.turn_eos)
            current.output_ids.append(adapter.special.turn_eos)
        current.output_ids.append(adapter.special.chunk_eos)
    current.finish_reason = "stop"
    current = adapter.continue_compaction(IDENTITY, current, 1024)
    assert current is not None
    assert not adapter.context_memories[IDENTITY].summary.is_complete
    torch.testing.assert_close(
        current.session_embedding_spans[-1].input_embeds,
        audio_task["plans"][1]["input_embeds"],
    )
    current.generated_unit_ids = []
    current.output_ids = [adapter.special.listen]
    current.finish_reason = "stop"
    current = adapter.continue_compaction(IDENTITY, current, 1024)
    assert current is not None
    torch.testing.assert_close(
        current.session_embedding_spans[-1].input_embeds,
        audio_task["plans"][2]["input_embeds"],
    )
    current.generated_unit_ids = [vocabulary["summary_fact"]]
    current.output_ids = [
        adapter.special.speak,
        vocabulary["summary_fact"],
        adapter.special.chunk_eos,
    ]
    current.finish_reason = "stop"
    current = adapter.continue_compaction(IDENTITY, current, 1024)
    assert current is not None
    assert not adapter.context_memories[IDENTITY].summary.is_complete
    torch.testing.assert_close(
        current.session_embedding_spans[-1].input_embeds,
        audio_task["plans"][3]["input_embeds"],
    )
    assert vocabulary["new_fact"] not in current.req.origin_input_ids
    current.generated_unit_ids = [vocabulary["new_fact"], adapter.special.turn_eos]
    current.output_ids = [
        adapter.special.speak,
        *current.generated_unit_ids,
        adapter.special.chunk_eos,
    ]
    current.finish_reason = "stop"
    assert adapter.continue_compaction(IDENTITY, current, 1024) is None
    rebuilt = adapter.build_compacted_history(IDENTITY, current, 1024)
    assert vocabulary["raw_fact"] not in rebuilt.req.origin_input_ids
    assert rebuilt.req.origin_input_ids.count(vocabulary["summary_fact"]) == 1
    assert rebuilt.req.origin_input_ids.count(vocabulary["new_fact"]) == 1
    assert not rebuilt.generate_audio


def test_spoken_summary_closes_only_shadow_reply_and_listens_to_instruction() -> None:
    """An old open answer must not continue through or overwrite the spoken summary instruction."""
    adapter = thinker_adapter()
    state = adapter.states[IDENTITY]
    state.is_turn_ended = False
    audio_task = summary_audio_task(adapter, instruction_units=3, unit_count=4)
    adapter.context_memories[IDENTITY].summary_audio = audio_task
    history = ARSessionHistory(token_ids=[0] * 90, embedding_spans=[])
    payload = StagePayload(
        "pending", OmniRequest(None), {"token_ids": [adapter.special.unit, 0]}
    )
    current = adapter.plan_compaction(IDENTITY, payload, history, 1024).request_data
    assert current.thinker_state.is_turn_ended
    assert list(current.req.origin_input_ids) == [
        *history.token_ids,
        adapter.special.unit_end,
        *audio_task["plans"][0]["token_ids"],
        adapter.special.turn_eos,
    ]
    for unit_index in range(2):
        assert current.is_listen_forced
        assert not state.is_turn_ended
        current.generated_unit_ids = []
        current.output_ids = [adapter.special.listen]
        current.finish_reason = "stop"
        current = adapter.continue_compaction(IDENTITY, current, 1024)
        assert current is not None
        assert current.req.origin_input_ids.count(adapter.special.turn_eos) == 1
        torch.testing.assert_close(
            current.session_embedding_spans[-1].input_embeds,
            audio_task["plans"][unit_index + 1]["input_embeds"],
        )
    assert not current.is_listen_forced
    fact_token = adapter.tokenizer.convert_tokens_to_ids("summary_fact")
    current.generated_unit_ids = [fact_token, adapter.special.turn_eos]
    current.output_ids = [
        adapter.special.speak,
        *current.generated_unit_ids,
        adapter.special.chunk_eos,
    ]
    current.finish_reason = "stop"
    assert adapter.continue_compaction(IDENTITY, current, 1024) is None
    rebuilt = adapter.build_compacted_history(IDENTITY, current, 1024)
    assert rebuilt.req.origin_input_ids.count(fact_token) == 1
    assert not state.is_turn_ended


@pytest.mark.parametrize("exhaustion", ["audio_units", "output_tokens"])
def test_native_audio_summary_stops_at_input_or_output_budget(exhaustion: str) -> None:
    """Listening cannot run beyond available audio, and speaking cannot exceed its output cap."""
    adapter = thinker_adapter()
    adapter.window = MiniCPMODuplexWindow(
        sliding_window_mode="summary",
        summary_trigger_tokens=5500,
        summary_max_tokens=1000,
    )
    adapter.context_memories[IDENTITY].summary_audio = summary_audio_task(
        adapter,
        unit_count=128,
        instruction_units=8 if exhaustion == "audio_units" else 1,
        audio_tokens=10,
    )
    history = ARSessionHistory(token_ids=[0] * 5499, embedding_spans=[])
    payload = StagePayload(
        "pending", OmniRequest(None), {"token_ids": [adapter.special.unit, 0]}
    )
    assert adapter.plan_compaction(IDENTITY, payload, history, 8191) is None
    history.token_ids.append(0)
    plan = adapter.plan_compaction(IDENTITY, payload, history, 8191)
    assert plan is not None
    current = plan.request_data
    assert current.max_new_tokens == 20
    unit_count = 128 if exhaustion == "audio_units" else 50
    fact_token = adapter.tokenizer.convert_tokens_to_ids("summary_fact")
    for unit_index in range(unit_count):
        assert len(current.req.origin_input_ids) + current.max_new_tokens < 8191
        assert len(current.session_embedding_spans) == unit_index + 1
        if exhaustion == "audio_units":
            current.generated_unit_ids = []
            current.output_ids = [adapter.special.listen]
        else:
            current.generated_unit_ids = [fact_token] * 18
            current.output_ids = [
                adapter.special.speak,
                *current.generated_unit_ids,
                adapter.special.chunk_eos,
            ]
        current.finish_reason = "stop"
        if unit_index == unit_count - 1:
            with pytest.raises(ValueError, match="budget|exhaust|audio"):
                adapter.continue_compaction(IDENTITY, current, 8191)
        else:
            current = adapter.continue_compaction(IDENTITY, current, 8191)
            assert current is not None
    assert not adapter.context_memories[IDENTITY].summary.is_complete
    assert history.token_ids == [0] * 5500


def test_closing_a_parked_input_cancels_internal_work_without_resuming_it() -> None:
    bridge, scheduler, payload = compaction_bridge()
    session = bridge.sessions[IDENTITY.id]
    internal_request = session.compaction.request_data.req
    scheduler.chunked_req = internal_request
    bridge.cancel(payload.request_id)
    assert not bridge.compactions
    assert session.compaction is None and session.unit is None
    assert payload.request_id not in scheduler.deferred_request_payloads
    assert scheduler.chunked_req is None
    assert scheduler.session_controller.get(IDENTITY.id) is not None
    bridge.close_session(session)
    assert scheduler.session_controller.get(IDENTITY.id) is None
    assert not bridge.sessions


def test_pending_summary_survives_rewind_and_invalidation() -> None:
    """A control rewind must not let the next input overtake an already due summary."""
    adapter = thinker_adapter()
    state = adapter.states[IDENTITY]
    state.is_turn_ended = False
    payload = StagePayload(
        "pending", OmniRequest(None), {"token_ids": [adapter.special.unit, 0]}
    )
    history = ARSessionHistory(token_ids=[0] * 90, embedding_spans=[])
    plan = adapter.plan_compaction(IDENTITY, payload, history, 1024)

    assert plan is not None
    assert list(plan.request_data.req.origin_input_ids) == [
        *history.token_ids,
        adapter.special.unit_end,
        *adapter.context_memories[IDENTITY].summary_audio["plans"][0]["token_ids"],
        adapter.special.turn_eos,
    ]
    adapter.finish_compaction(IDENTITY, status="invalidated")
    state.is_turn_ended = True
    history = ARSessionHistory(
        token_ids=history.token_ids[:20], embedding_spans=history.embedding_spans
    )
    retry = adapter.plan_compaction(IDENTITY, payload, history, 1024)
    assert retry is not None
    assert list(retry.request_data.req.origin_input_ids) == [
        *history.token_ids,
        adapter.special.unit_end,
        *adapter.context_memories[IDENTITY].summary_audio["plans"][0]["token_ids"],
    ]
    summary = retry.request_data
    summary.generated_unit_ids = [
        adapter.tokenizer.convert_tokens_to_ids("summary_fact"),
        adapter.special.turn_eos,
    ]
    summary.output_ids = [
        adapter.special.speak,
        *summary.generated_unit_ids,
        adapter.special.chunk_eos,
    ]
    summary.finish_reason = "stop"
    assert adapter.continue_compaction(IDENTITY, summary, 1024) is None
    rebuilt = adapter.build_compacted_history(IDENTITY, summary, 1024)
    internal_request_ids: set[str] = set()
    for internal in (plan.request_data, summary, rebuilt):
        assert internal.stage_payload.request_id == internal.req.rid
        assert internal.req.rid not in internal_request_ids
        internal_request_ids.add(internal.req.rid)
    adapter.finish_compaction(IDENTITY, status="completed")
    assert adapter.plan_compaction(IDENTITY, payload, history, 1024) is None


def test_summary_reserve_can_trigger_before_the_history_threshold() -> None:
    """The pending input must wait when it would consume the summary reserve."""
    adapter = thinker_adapter()
    history = ARSessionHistory(token_ids=[0] * 70, embedding_spans=[])
    payload = StagePayload(
        "pending", OmniRequest(None), {"token_ids": [adapter.special.unit, 0]}
    )
    context_limit = 105

    assert (
        adapter.plan_compaction(IDENTITY, payload, history, context_limit) is not None
    )


def test_shortest_continuing_chunks_fit_reserved_context() -> None:
    """Early chunk endings must not consume more context than admission reserved."""
    adapter = thinker_adapter()
    payload = StagePayload(
        "pending", OmniRequest(None), {"token_ids": [adapter.special.unit, 0]}
    )
    history = ARSessionHistory(token_ids=[0] * 80, embedding_spans=[])
    plan = adapter.plan_compaction(IDENTITY, payload, history, 101)
    assert plan is not None
    current = plan.request_data
    text_token_id = adapter.tokenizer.convert_tokens_to_ids("summary_fact")
    for _ in range(2):
        assert len(current.req.origin_input_ids) + current.max_new_tokens < 101
        current.generated_unit_ids = [text_token_id]
        current.output_ids = [
            adapter.special.tts_bos,
            text_token_id,
            adapter.special.chunk_eos,
        ]
        current.finish_reason = "stop"
        current = adapter.continue_compaction(IDENTITY, current, 101)
        assert current is not None
    assert len(current.req.origin_input_ids) + current.max_new_tokens < 101
    current.generated_unit_ids = []
    current.output_ids = [adapter.special.turn_eos, adapter.special.chunk_eos]
    current.finish_reason = "stop"
    assert adapter.continue_compaction(IDENTITY, current, 101) is None
    rebuilt = adapter.build_compacted_history(IDENTITY, current, 101)
    assert list(rebuilt.req.origin_input_ids).count(text_token_id) == 2


@pytest.mark.parametrize("is_turn_ended", [False, True])
def test_next_input_waits_for_summary_even_when_previous_reply_is_open(
    is_turn_ended: bool,
) -> None:
    """An open reply or retained reply records must not admit another input ahead of summary."""
    adapter = thinker_adapter()
    state = adapter.states[IDENTITY]
    state.is_turn_ended = is_turn_ended
    state.drain_units_left = 3 if is_turn_ended else 0
    state.reply_units = [
        ReplyUnit(
            unit_index=7,
            is_spoken=True,
            kept_token_count=85,
            perception_token_ids=[adapter.special.unit, 0],
            perception_embedding_spans=[],
            is_turn_ended_at_start=True,
            generated_history_at_start=[],
        )
    ]
    state.is_latest_reply_ended = is_turn_ended
    history = ARSessionHistory(token_ids=[0] * 90, embedding_spans=[])
    payload = StagePayload(
        "pending",
        OmniRequest(None),
        {
            "token_ids": [
                adapter.special.unit,
                adapter.tokenizer.convert_tokens_to_ids("new_fact"),
            ]
        },
    )

    plan = adapter.plan_compaction(IDENTITY, payload, history, 1024)
    assert plan is not None
    assert list(plan.request_data.req.origin_input_ids) == [
        *history.token_ids,
        adapter.special.unit_end,
        *adapter.context_memories[IDENTITY].summary_audio["plans"][0]["token_ids"],
        *([] if is_turn_ended else [adapter.special.turn_eos]),
    ]
    state.speech_detector.score.assert_not_called()
    assert state.reply_units
    adapter.finish_compaction(IDENTITY, status="completed")

    assert state.drain_units_left == (3 if is_turn_ended else 0)
    assert not state.reply_units
    assert not state.is_latest_reply_ended


def test_pending_summary_does_not_force_listen_in_ordinary_build() -> None:
    """Summary priority must park an input rather than turn it into a listen request."""
    adapter = thinker_adapter()
    state = adapter.states[IDENTITY]
    state.force_listen_counter = state.sampling.force_listen_count
    adapter.context_memories[IDENTITY].is_compaction_pending = True
    payload = StagePayload(
        "pending",
        OmniRequest(None),
        {
            "token_ids": [adapter.special.unit, 0],
            "input_embeds": torch.empty((0, 2)),
            "embedding_spans": [],
        },
    )
    current = adapter.build(
        IDENTITY, TimedChunk("audio", 1000, 1000, 1, b"\xff\x7f" * 16000), payload
    )

    assert not current.is_listen_forced
    state.speech_detector.score.assert_not_called()


@pytest.mark.parametrize("is_turn_ended", [False, True])
def test_control_refill_is_summarized_before_pending_input(is_turn_ended: bool) -> None:
    """A truncate repair must enter the snapshot without admitting the queued input."""
    adapter = thinker_adapter()
    state = adapter.states[IDENTITY]
    state.is_interrupt_pending = True
    state.is_turn_ended = is_turn_ended
    state.refill_token_ids = [adapter.special.listen, adapter.special.unit, 0]
    state.refill_embedding_spans = [
        EmbeddingSpan(start=2, end=3, input_embeds=torch.full((1, 2), 9.0))
    ]
    history = ARSessionHistory(token_ids=[0] * 90, embedding_spans=[])
    payload = StagePayload(
        "pending",
        OmniRequest(None),
        {
            "token_ids": [adapter.special.unit, 1],
            "input_embeds": torch.empty((0, 2)),
            "embedding_spans": [],
        },
    )
    refill_token_ids = list(state.refill_token_ids)

    plan = adapter.plan_compaction(IDENTITY, payload, history, 1024)

    assert plan is not None
    assert list(plan.request_data.req.origin_input_ids) == [
        *history.token_ids,
        *refill_token_ids,
        adapter.special.unit_end,
        *adapter.context_memories[IDENTITY].summary_audio["plans"][0]["token_ids"],
        *([] if is_turn_ended else [adapter.special.turn_eos]),
    ]
    assert plan.request_data.thinker_state.is_turn_ended
    assert state.is_turn_ended is is_turn_ended
    assert state.is_interrupt_pending
    refill_span, instruction_span = plan.request_data.session_embedding_spans
    assert instruction_span.start == 95
    assert (refill_span.start, refill_span.end) == (92, 93)
    torch.testing.assert_close(refill_span.input_embeds, torch.full((1, 2), 9.0))
    assert state.refill_token_ids == refill_token_ids
    adapter.finish_compaction(IDENTITY, status="completed")
    assert not state.refill_token_ids
    assert not state.refill_embedding_spans
    assert state.is_turn_ended
    continuation = adapter.build(
        IDENTITY, TimedChunk("audio", 1000, 1000, 1, b"\x00\x00"), payload
    )
    assert plan.continuation_tokens == (
        len(continuation.req.origin_input_ids) + continuation.max_new_tokens
    )


@pytest.mark.parametrize("phase", ["summary", "rebuild"])
def test_invalidated_summary_retries_before_resuming_pending_input(phase: str) -> None:
    """A history change used to release the parked input before retrying summary."""
    bridge, scheduler, payload = compaction_bridge()
    session = bridge.sessions[IDENTITY.id]
    if phase == "rebuild":
        finish_summary(bridge)
    else:
        pass
    event = TimedChunk(CONTROL, 0, 0, 0, {"type": INTERRUPT}, format=SESSION_CONTROL)
    bridge.apply_operation(
        StagePayload("control", OmniRequest(None), None),
        SessionOperation("control", IDENTITY, ("thinker",), event),
    )
    internal = session.compaction.request_data
    bridge.finish_internal_request(internal.req.rid, internal)
    resumed = scheduler.deferred_request_payloads.pop(payload.request_id)
    scheduler.dirty_deferred_request_ids.remove(payload.request_id)
    bridge.prepare_unit(session.unit, resumed)

    assert bridge.start_compaction(session.unit, resumed)
    assert session.unit.session_request is None
    assert payload.request_id not in scheduler.deferred_request_payloads
    finish_summary(bridge)
    rebuild = session.compaction.request_data
    shadow = scheduler.session_controller.get(session.compaction.shadow_session_id)
    rebuild.req.output_ids.append(77)
    rebuild.req.finished_reason = FINISH_LENGTH(1)
    rebuild.finish_reason = "length"
    rebuild.req.refresh_fill_ids()
    shadow.finish_req(rebuild.req)
    bridge.finish_internal_request(rebuild.req.rid, rebuild)

    assert scheduler.deferred_request_payloads == {payload.request_id: payload}
    assert scheduler.dirty_deferred_request_ids == {payload.request_id}
    assert not bridge.start_compaction(session.unit, payload)
    pending = request_data(payload.request_id, [61, 62])
    bridge.create_session_request(payload, pending)
    assert list(pending.req.origin_input_ids) == [*REBUILT_IDS, 61, 62]

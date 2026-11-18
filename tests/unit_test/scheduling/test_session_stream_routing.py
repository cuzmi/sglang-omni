# SPDX-License-Identifier: Apache-2.0
"""A session unit's stream goes to the client only from its route's last stage; before that it flows downstream as usual."""

import queue
import threading
from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import pytest

from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.proto.session import SessionIdentity, TimedChunk
from sglang_omni.scheduling import omni_scheduler
from sglang_omni.scheduling.message import OutgoingMessage
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from sglang_omni.scheduling.sglang_backend.ar_session import (
    ARSessionAdapter,
    ARSessionBridge,
    InternalSessionRequest,
    SessionUnit,
)


def scheduler_with_unit(stages: tuple[str, ...]):
    scheduler = Mock(spec=OmniScheduler)
    scheduler.aborted_request_ids = set()
    scheduler.outbox = queue.Queue()
    scheduler.first_emit_done = set()
    unit = SessionUnit(
        request_id="unit-1",
        session_identity=SessionIdentity("tts"),
        stages=stages,
        chunk=TimedChunk("text", 0, 0, 0, {"text": "hi"}, format="text_delta"),
    )
    scheduler.session_bridge = Mock()
    scheduler.session_bridge.units_by_request_id = {"unit-1": unit}
    scheduler.session_bridge.internal_requests = {}
    scheduler.session_bridge.cancelling_request_id = None
    scheduler.session_bridge.stream_messages = Mock(
        return_value=[OutgoingMessage("unit-1", "stream", "from-bridge")]
    )
    scheduler.active_session_unit = MethodType(
        OmniScheduler.active_session_unit, scheduler
    )
    scheduler.stream_output_builder = Mock(
        return_value=[OutgoingMessage("unit-1", "stream", "from-builder")]
    )
    scheduler.stream_output_builder.flush.return_value = [
        OutgoingMessage("unit-1", "stream", "flushed-builder")
    ]
    scheduler.stage_stream_messages = MethodType(
        OmniScheduler.stage_stream_messages, scheduler
    )
    scheduler.put_stream_messages = MethodType(
        OmniScheduler.put_stream_messages, scheduler
    )
    sched_output = SimpleNamespace(
        requests=[SimpleNamespace(request_id="unit-1", data="data")]
    )
    mr_output = SimpleNamespace(outputs={"unit-1": "output"})
    return scheduler, sched_output, mr_output


@pytest.mark.parametrize(
    "stages, expected",
    [(("tts_engine", "vocoder"), "from-builder"), (("tts_engine",), "from-bridge")],
)
def test_a_session_stream_leaves_through_the_bridge_only_at_the_last_stage(
    monkeypatch, stages, expected
):
    monkeypatch.setattr(omni_scheduler, "_get_active_stage", lambda: "tts_engine")
    scheduler, sched_output, mr_output = scheduler_with_unit(stages)
    OmniScheduler.emit_stream_output(scheduler, sched_output, mr_output)
    assert scheduler.outbox.get_nowait() == OutgoingMessage(
        "unit-1", "stream", expected
    )
    assert scheduler.outbox.empty()
    OmniScheduler.flush_stream_output(scheduler, "unit-1", "data")
    flushed = scheduler.outbox.get_nowait()
    if expected == "from-bridge":
        assert flushed == OutgoingMessage("unit-1", "stream", "from-bridge")
    else:
        assert flushed == OutgoingMessage("unit-1", "stream", "flushed-builder")
    assert scheduler.outbox.empty()


def test_internal_compaction_stream_and_flush_never_leave_the_scheduler():
    scheduler, sched_output, mr_output = scheduler_with_unit(("thinker", "talker"))
    scheduler.session_bridge.internal_requests["unit-1"] = Mock()
    OmniScheduler.emit_stream_output(scheduler, sched_output, mr_output)
    OmniScheduler.flush_stream_output(scheduler, "unit-1", "data")
    assert scheduler.outbox.empty()
    scheduler.session_bridge.stream_messages.assert_not_called()
    scheduler.stream_output_builder.assert_not_called()
    scheduler.stream_output_builder.flush.assert_not_called()


def test_internal_completion_resumes_original_payload_after_cleanup(monkeypatch):
    scheduler, _, _ = scheduler_with_unit(("thinker", "talker"))
    scheduler.request_admission_lock = threading.Lock()
    scheduler.draft_worker = None
    scheduler.result_adapter = Mock()
    scheduler.deferred_request_payloads = {}
    scheduler.dirty_deferred_request_ids = set()
    payload = StagePayload("unit-1", OmniRequest(None), {})
    bridge = scheduler.session_bridge
    bridge.internal_requests = {
        "internal": SimpleNamespace(session_identity=SessionIdentity("tts"))
    }
    bridge.sessions = {
        "tts": SimpleNamespace(unit=bridge.units_by_request_id["unit-1"])
    }
    completed = []
    scheduler.close_completed_request.side_effect = (
        lambda request: completed.append(request.rid) or False
    )

    def complete_internal(request_id, request_data):
        assert completed == [request_id]
        assert request_data.output_ids == [7]
        return None, payload

    bridge.complete_internal_request.side_effect = complete_internal
    request_data = SimpleNamespace()
    request = SimpleNamespace(
        rid="internal",
        finished=lambda: True,
        finished_reason=SimpleNamespace(to_json=lambda: {"type": "length"}),
        omni_data=request_data,
        output_ids=[7],
        _omni_terminal_claimed=False,
    )
    monkeypatch.setattr(
        omni_scheduler, "get_serving", lambda: SimpleNamespace(weight_version="test")
    )
    scheduler.run_request_finished_callback.return_value = None
    OmniScheduler.stream_output(scheduler, [request])
    assert scheduler.deferred_request_payloads == {"unit-1": payload}
    assert scheduler.dirty_deferred_request_ids == {"unit-1"}
    assert scheduler.outbox.empty()
    scheduler.result_adapter.assert_not_called()
    bridge.adapter.result.assert_not_called()
    scheduler.flush_stream_output.assert_not_called()
    scheduler.fail_context_compaction.assert_not_called()


def test_internal_error_settles_suspended_input_and_session_read():
    scheduler, _, _ = scheduler_with_unit(("thinker", "talker"))
    scheduler.is_entry_rank = True
    bridge = scheduler.session_bridge
    bridge.internal_requests["internal"] = Mock()
    reader = StagePayload("session-read", OmniRequest(None), {})

    def fail_internal(request_id):
        assert request_id == "internal"
        bridge.internal_requests.pop(request_id)
        return "unit-1", [reader]

    bridge.fail_compaction.side_effect = fail_internal
    scheduler.fail_context_compaction = MethodType(
        OmniScheduler.fail_context_compaction, scheduler
    )
    scheduler.emit_request_error = MethodType(
        OmniScheduler.emit_request_error, scheduler
    )
    error = ValueError("summary token budget exhausted")
    scheduler.emit_request_error("internal", error)
    assert scheduler.outbox.get_nowait() == OutgoingMessage("unit-1", "error", error)
    assert scheduler.outbox.get_nowait() == OutgoingMessage(
        "session-read", "error", error
    )
    assert scheduler.outbox.empty()
    assert not bridge.internal_requests


def test_internal_cleanup_drains_abort_without_recursive_failure() -> None:
    """A terminal abort delivered while cleanup drains decode must not close the same compaction recursively or expose its internal ID."""
    scheduler, _, _ = scheduler_with_unit(("thinker", "talker"))
    scheduler.request_admission_lock = threading.RLock()
    scheduler.is_entry_rank = True
    scheduler.draft_worker = None
    scheduler.waiting_queue = []
    scheduler.chunked_req = None
    bridge = ARSessionBridge(scheduler, ARSessionAdapter())
    scheduler.session_bridge = bridge
    request = SimpleNamespace(
        rid="internal",
        finished=lambda: request.finished_reason is not None,
        finished_reason=None,
        omni_data=SimpleNamespace(),
    )
    internal_request = InternalSessionRequest(
        request_id=request.rid,
        session_identity=SessionIdentity("tts"),
        session_request=request,
        is_enqueued=True,
    )
    bridge.internal_requests[request.rid] = internal_request

    def deliver_pending_abort() -> None:
        assert bridge.cancelling_request_id == request.rid
        request.finished_reason = omni_scheduler.FINISH_ABORT()
        OmniScheduler.emit_request_error(scheduler, request.rid, ValueError("aborted"))
        OmniScheduler.stream_output(scheduler, [request])

    scheduler.synchronize_launched_decode.side_effect = deliver_pending_abort
    bridge.cancel_native_request(internal_request)
    scheduler.fail_context_compaction.assert_not_called()
    scheduler.release_request_kv_cache.assert_called_once_with(request)
    scheduler.run_abort_callback.assert_called_once_with(request.rid)
    assert scheduler.outbox.empty()
    assert request.omni_data is None
    assert bridge.cancelling_request_id is None
    assert bridge.internal_requests[request.rid] is internal_request

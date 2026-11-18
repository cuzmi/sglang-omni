# SPDX-License-Identifier: Apache-2.0
"""Bridge pipeline units to streaming sessions."""

from __future__ import annotations

import logging
from array import array
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Literal, Protocol
from uuid import uuid4

import torch
from sglang.srt.managers.io_struct import (
    CloseSessionReqInput,
    OpenSessionReqInput,
    SessionParams,
    TokenizedGenerateReqInput,
)
from sglang.srt.managers.schedule_batch import FINISH_ABORT, Req
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.session.session_controller import Session, SessionController

from sglang_omni.admission import (
    REQUEST_TO_TOKEN_SLOTS_RESERVED_FOR_RETAINED_KV,
    QueueFullError,
)
from sglang_omni.profiler.event_recorder import get_recorder
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.proto.session import (
    SessionIdentity,
    SessionOperation,
    TimedChunk,
    find_session_operation,
)
from sglang_omni.scheduling.message import OutgoingMessage
from sglang_omni.scheduling.sglang_backend.history_cut import (
    HistoryCut,
    LayerKVCache,
    RetainedKV,
    SessionRows,
    cut_retained_kv,
    cut_token_ids,
    retained_history_length,
)
from sglang_omni.scheduling.sglang_backend.request_data import (
    EmbeddingSpan,
    SGLangARRequestData,
)
from sglang_omni.scheduling.types import RequestOutput

SESSION_STRING_LENGTH_LIMIT_CHARACTERS = 0


logger = logging.getLogger(__name__)


class SessionKV(RetainedKV, Protocol):
    holds_kv: bool

    def get_evicted_seqlen(self, component_type: ComponentType) -> int:
        pass


class SessionTreeCache(SessionRows, Protocol):
    def session_records(self) -> Mapping[str, SessionKV]:
        pass

    def evictable_size(self) -> int:
        pass


class RequestToTokenPool(Protocol):
    free_slots: list[int]


class TokenToKVPoolAllocator(Protocol):
    size: int

    def available_size(self) -> int:
        pass

    def get_kvcache(self) -> LayerKVCache:
        pass


class ModelVocabulary(Protocol):
    vocab_size: int


class BridgeScheduler(Protocol):
    """The scheduler surface the bridge drives; OmniScheduler satisfies it."""

    session_controller: SessionController
    tree_cache: SessionTreeCache
    req_to_token_pool: RequestToTokenPool
    token_to_kv_pool_allocator: TokenToKVPoolAllocator
    model_config: ModelVocabulary
    waiting_queue: list[Req]
    chunked_req: Req | None
    max_running_requests: int
    max_req_len: int

    def abort(self, request_id: str) -> None:
        pass

    def emit_request_error(self, request_id: str, error: Exception) -> None:
        pass

    def release_request_kv_cache(self, req: Req) -> None:
        pass

    def run_abort_callback(self, request_id: str) -> None:
        pass

    def synchronize_launched_decode(self) -> None:
        pass

    def resolve_pending_async(self) -> None:
        pass


def is_close_request(payload: StagePayload) -> bool:
    """Return whether the payload carries a session close."""
    operation_metadata = find_session_operation(payload.request.metadata)
    if operation_metadata is None:
        return False
    else:
        return operation_metadata.operation == "close"


@dataclass(frozen=True, kw_only=True)
class ARSessionPreparation:
    reset_history: bool = False
    bypass_generation: bool = False
    # note (ruoyu): The native tokens to keep before the unit is appended, a prefix of the last unit's prompt that ends between units; the tokens past it leave the session's arrays, spans and KV.
    kept_token_count: int | None = None


class ARSessionAdapter:
    """Convert unit inputs and outputs without mutating streaming session state."""

    is_compaction_enabled: bool = False

    def open(self, session_identity: SessionIdentity, request: OmniRequest) -> None:
        """Initialize auxiliary model history after the streaming session is created."""

    def close(self, session_identity: SessionIdentity) -> None:
        """Release auxiliary history after streaming session work and KV are released."""

    def control(self, session_identity: SessionIdentity, event: TimedChunk) -> None:
        """Handle a control event; an adapter without control semantics lets it pass."""

    def prepare_unit(
        self,
        session_identity: SessionIdentity,
        chunk: TimedChunk,
        payload: StagePayload,
    ) -> ARSessionPreparation:
        """Select a fresh model turn or relay a unit that needs no generation."""
        return ARSessionPreparation()

    def complete_without_generation(
        self,
        session_identity: SessionIdentity,
        chunk: TimedChunk,
        payload: StagePayload,
    ) -> StagePayload:
        """The result of a unit its model cannot generate for."""
        return payload

    def finish_input(
        self, session_identity: SessionIdentity, payload: StagePayload
    ) -> StagePayload | None:
        """Return a relayed EOS payload, or None to use normal generation."""
        return None

    def bind_request(
        self,
        session_identity: SessionIdentity,
        request_data: SGLangARRequestData,
        unit: "SessionUnit",
    ) -> None:
        """Attach model-specific inputs to the native request once retained turns precede the unit's ids."""

    def on_abort(self, session_identity: SessionIdentity, unit: "SessionUnit") -> None:
        """Take what an aborted unit generated before its native request is dropped."""

    def build(
        self,
        session_identity: SessionIdentity,
        chunk: TimedChunk,
        payload: StagePayload,
    ) -> SGLangARRequestData:
        raise NotImplementedError

    def start_compaction(
        self,
        session_identity: SessionIdentity,
        chunk: TimedChunk,
        payload: StagePayload,
        history_length: int,
        context_limit: int,
    ) -> bool:
        return False

    def build_compaction_request(
        self,
        session_identity: SessionIdentity,
        request_id: str,
        payload: StagePayload,
        history_length: int,
        context_limit: int,
    ) -> SGLangARRequestData:
        raise NotImplementedError

    def complete_compaction_request(
        self, session_identity: SessionIdentity, request_data: SGLangARRequestData
    ) -> Literal["continue", "rebuild"]:
        raise NotImplementedError

    def build_rebuild_request(
        self,
        session_identity: SessionIdentity,
        request_id: str,
        payload: StagePayload,
        context_limit: int,
    ) -> SGLangARRequestData:
        raise NotImplementedError

    def finish_compaction(self, session_identity: SessionIdentity) -> None:
        raise NotImplementedError

    def cancel_compaction(self, session_identity: SessionIdentity) -> None:
        """Discard model-side temporary compaction state during session teardown."""

    def plan_history_cut(
        self,
        session_identity: SessionIdentity,
        request_data: SGLangARRequestData,
        history_length: int,
        history_capacity: int,
    ) -> HistoryCut | None:
        """Choose whole units of retained history to drop before this unit joins the session.

        history_length tokens come ahead of the unit now; it can still finish with at most history_capacity of them.
        """
        return None

    def shift_key_positions(
        self, keys: torch.Tensor, distances: torch.Tensor
    ) -> torch.Tensor:
        """Re-encode each cached key distances[i] positions earlier, in its dtype."""
        raise NotImplementedError

    def result(
        self, session_identity: SessionIdentity, request_data: SGLangARRequestData
    ) -> StagePayload:
        raise NotImplementedError

    def stream(
        self,
        session_identity: SessionIdentity,
        request_data: SGLangARRequestData,
        output: RequestOutput,
    ) -> Iterable[TimedChunk]:
        return ()

    def flush(
        self, session_identity: SessionIdentity, request_data: SGLangARRequestData
    ) -> Iterable[TimedChunk]:
        return ()


@dataclass(kw_only=True)
class NativeSessionRequest:
    request_id: str
    session_identity: SessionIdentity
    session_request: Req | None = None
    is_enqueued: bool = False
    embedding_spans: list[EmbeddingSpan] = field(default_factory=list)
    native_offset: int = 0
    prompt_length: int = 0


@dataclass(kw_only=True)
class SessionUnit(NativeSessionRequest):
    stages: tuple[str, ...]
    chunk: TimedChunk
    is_prepared: bool = False
    is_compaction_checked: bool = False


@dataclass(kw_only=True)
class InternalSessionRequest(NativeSessionRequest):
    pass


@dataclass(kw_only=True)
class SessionCompaction:
    payload: StagePayload
    phase: Literal["summary", "rebuild"] = "summary"
    request_id: str | None = None


@dataclass(kw_only=True)
class BridgeSession:
    session_identity: SessionIdentity
    unit: SessionUnit | None = None
    compaction: SessionCompaction | None = None
    # note (Junnan Li): KV rebuilds need embeddings from every completed unit.
    embedding_spans: list[EmbeddingSpan] = field(default_factory=list)
    # The session's read; an AR stage emits only inside appends, so the close answers it.
    output_read: StagePayload | None = None


class ARSessionBridge:
    """Map pipeline open, append, and close onto one SGLang streaming session."""

    def __init__(
        self, bridge_scheduler: BridgeScheduler, adapter: ARSessionAdapter
    ) -> None:
        self.bridge_scheduler = bridge_scheduler
        self.adapter = adapter
        self.sessions: dict[str, BridgeSession] = {}
        self.units_by_request_id: dict[str, SessionUnit] = {}
        self.internal_requests: dict[str, InternalSessionRequest] = {}
        self.cancelling_request_id: str | None = None

    def drain(self) -> None:
        self.bridge_scheduler.synchronize_launched_decode()
        self.bridge_scheduler.resolve_pending_async()

    def apply_operation(
        self, payload: StagePayload, operation: SessionOperation
    ) -> list[StagePayload]:
        """Apply an operation other than append; returns the operations it answers, none for a read the close answers."""
        session_identity = operation.session_identity
        operation_kind = operation.operation
        session = self.sessions.get(session_identity.id)
        if (
            session is not None
            and session_identity.open_index != session.session_identity.open_index
        ):
            raise ValueError("stale session open index")
        elif operation_kind == "open":
            if session is not None:
                raise ValueError("session already opened")
            elif len(self.sessions) >= self.bridge_scheduler.max_running_requests:
                raise QueueFullError()
            elif self.open_streaming_session(session_identity.id):
                self.sessions[session_identity.id] = BridgeSession(
                    session_identity=session_identity
                )
                self.adapter.open(session_identity, payload.request)
                payload.data = {"opened": True}
                return [payload]
            else:
                raise ValueError("streaming session open failed")
        elif operation_kind == "close":
            payload.data = {"closed": True}
            if session is not None:
                return [*self.close_session(session), payload]
            else:
                # note (chenyang): coding style, each if should be closed by else.
                return [payload]
        elif session is None:
            raise ValueError("unknown session open index")
        elif operation_kind == "read":
            if session.output_read is not None:
                raise ValueError("session already has an output reader")
            else:
                session.output_read = payload
                return []
        elif operation_kind == "control":
            assert operation.chunk is not None, "control operation carries no event"
            if operation.should_preempt and session.unit is not None:
                self.cancel(session.unit.request_id)
            else:
                pass
            if session_identity.id not in self.sessions:
                raise ValueError("context compaction interrupted; session closed")
            else:
                self.adapter.control(session_identity, operation.chunk)
            payload.data = {"controlled": True}
            return [payload]
        else:
            raise ValueError("unknown session operation")

    def accept(self, payload: StagePayload, operation: SessionOperation) -> SessionUnit:
        chunk = operation.chunk
        assert (
            chunk is not None
        ), f"session append {payload.request_id} requires a chunk"
        session_identity = operation.session_identity
        session = self.sessions.get(session_identity.id)
        if session is None or session_identity != session.session_identity:
            raise ValueError("unknown or stale session open index")
        else:
            if session.unit is None:
                session.unit = SessionUnit(
                    request_id=payload.request_id,
                    session_identity=session_identity,
                    stages=operation.stages,
                    chunk=chunk,
                )
                self.units_by_request_id[payload.request_id] = session.unit
                return session.unit
            elif session.unit.request_id != payload.request_id:
                raise ValueError("session already has an active request")
            else:
                return session.unit

    def prepare_unit(self, unit: SessionUnit, payload: StagePayload) -> bool:
        if unit.is_prepared:
            return False
        else:
            unit.is_prepared = True
        preparation = self.adapter.prepare_unit(
            unit.session_identity, unit.chunk, payload
        )
        if preparation.reset_history:
            session_id = unit.session_identity.id
            self.sessions[session_id].embedding_spans.clear()
            self.release_streaming_session(session_id)
            if not self.open_streaming_session(session_id):
                raise RuntimeError("native session reset failed")
            else:
                pass
        elif preparation.kept_token_count is not None:
            self.rewind_streaming_session(
                unit.session_identity.id, preparation.kept_token_count
            )
        else:
            pass
        return preparation.bypass_generation

    def open_streaming_session(self, session_id: str) -> bool:
        result = self.bridge_scheduler.session_controller.open(
            OpenSessionReqInput(
                session_id=session_id,
                capacity_of_str_len=SESSION_STRING_LENGTH_LIMIT_CHARACTERS,
                streaming=True,
                timeout=None,
            )
        )
        return result.success

    def release_streaming_session(self, session_id: str) -> None:
        self.drain()
        self.bridge_scheduler.session_controller.close(
            CloseSessionReqInput(session_id=session_id)
        )
        if (
            self.bridge_scheduler.session_controller.get(session_id) is not None
            or session_id in self.bridge_scheduler.tree_cache.session_records()
        ):
            raise RuntimeError("streaming session close is still pending")
        else:
            pass

    def rewind_streaming_session(self, session_id: str, kept_token_count: int) -> None:
        """Rewind the native session to its first kept_token_count tokens, a prefix of the last unit's prompt that ends between units: the last request's generated tokens, the session's committed array lengths, its KV record and the retained spans shrink together, so the next unit extends the kept tokens without a re-prefill."""
        native_session = self.bridge_scheduler.session_controller.get(session_id)
        [last_request_node] = native_session.req_nodes.values()
        last_request = last_request_node.req
        session = self.sessions[session_id]
        # note (ruoyu): The shared token array may still hold an aborted append's tokens until the next create_req trims it, so the finished prompt's length is the committed one.
        committed_token_count = native_session.committed_origin_len
        if committed_token_count is None or not (
            0 < kept_token_count <= committed_token_count
        ):
            raise ValueError(
                "a session rewind keeps a prefix of the last unit's prompt"
            )
        elif any(
            span.start < kept_token_count < span.end for span in session.embedding_spans
        ):
            raise ValueError("a session rewind lands between units, outside every span")
        else:
            # note (ruoyu): The streaming session trims its shared token arrays to the committed lengths on the next append and matches its KV by length alone, so all of them shrink to the kept count together or the next unit reuses rows computed for the dropped tokens.
            last_request.output_ids = last_request.output_ids[:0]
            native_session.committed_origin_len = kept_token_count
            native_session.committed_unpadded_len = kept_token_count
            native_session.committed_fill_len = kept_token_count
            record = self.bridge_scheduler.tree_cache.session_records()[session_id]
            record.kv_committed_len = min(record.kv_committed_len, kept_token_count)
            session.embedding_spans = [
                span for span in session.embedding_spans if span.end <= kept_token_count
            ]

    def create_session_request(
        self, payload: StagePayload, request_data: SGLangARRequestData
    ) -> None:
        unit = self.units_by_request_id[payload.request_id]
        assert (
            request_data.req.rid == payload.request_id
        ), "session adapter changed request ID"
        self.append_native_request(unit, request_data, should_cut_history=True)
        if request_data.stage_payload is None:
            request_data.stage_payload = payload
        else:
            pass
        self.adapter.bind_request(unit.session_identity, request_data, unit)

    def append_native_request(
        self,
        unit: NativeSessionRequest,
        request_data: SGLangARRequestData,
        *,
        should_cut_history: bool,
    ) -> None:
        self.drain()
        adapter_request = request_data.req
        if (
            adapter_request.input_embeds is not None
            or adapter_request.multimodal_inputs is not None
        ):
            raise ValueError(
                "a session unit carries its embeddings as spans, not as input_embeds or multimodal inputs"
            )
        elif request_data.prefill_input_embeds is not None:
            prompt = request_data.prefill_input_embeds
            if request_data.unit_embedding_spans or int(prompt.shape[0]) != len(
                adapter_request.origin_input_ids
            ):
                raise ValueError(
                    "a projected prompt is the unit's one span and covers all of its ids"
                )
            else:
                request_data.unit_embedding_spans = [
                    EmbeddingSpan(
                        start=0, end=int(prompt.shape[0]), input_embeds=prompt
                    )
                ]
                request_data.prefill_input_embeds = None
        else:
            pass
        native_session = self.bridge_scheduler.session_controller.get(
            unit.session_identity.id
        )
        session = self.sessions[unit.session_identity.id]
        history_length = retained_history_length(native_session)
        # note (0xtoward): SGLang grants the whole max_new_tokens only while input plus output stays below max_req_len.
        history_capacity = (
            self.bridge_scheduler.max_req_len
            - 1
            - len(adapter_request.origin_input_ids)
            - int(adapter_request.sampling_params.max_new_tokens or 0)
        )
        if should_cut_history:
            history_cut = self.adapter.plan_history_cut(
                unit.session_identity, request_data, history_length, history_capacity
            )
        elif history_length > history_capacity:
            raise ValueError("context compaction request exceeds the context budget")
        else:
            history_cut = None
        if history_cut is not None:
            self.drop_history(session, native_session, history_cut)
            history_length -= history_cut.length
        else:
            pass
        tokenized_input = TokenizedGenerateReqInput(
            rid=adapter_request.rid,
            input_text=None,
            input_ids=array("q", adapter_request.origin_input_ids),
            input_embeds=None,
            mm_inputs=None,
            token_type_ids=None,
            sampling_params=adapter_request.sampling_params,
            logprob_start_len=adapter_request.logprob_start_len,
            session_params=SessionParams(id=native_session.session_id),
            stream=adapter_request.stream,
            return_logprob=adapter_request.return_logprob,
            return_sampling_mask=adapter_request.return_sampling_mask,
            lora_id=adapter_request.lora_id,
            custom_logit_processor=adapter_request.custom_logit_processor,
            require_reasoning=adapter_request.require_reasoning,
            return_hidden_states=adapter_request.return_hidden_states,
            return_routed_experts=adapter_request.return_routed_experts,
            routed_experts_start_len=adapter_request.routed_experts_start_len,
            priority=adapter_request.priority,
            routing_key=adapter_request.routing_key,
            extra_key=adapter_request.extra_key,
            cache_salt=adapter_request.cache_salt,
            http_worker_ipc=adapter_request.http_worker_ipc,
            top_logprobs_num=adapter_request.logprob.top_logprobs_num,
            token_ids_logprob=adapter_request.logprob.token_ids_logprob,
        )
        session_request = native_session.create_req(
            tokenized_input,
            adapter_request.tokenizer,
            self.bridge_scheduler.model_config.vocab_size,
            eos_token_ids=adapter_request.eos_token_ids,
        )
        if session_request.to_finish is not None:
            raise ValueError("native session rejected append")
        else:
            pass
        unit.session_request = session_request
        # note (Junnan Li): Native requests prepend retained turns to this unit's ids.
        offset = len(session_request.origin_input_ids) - len(
            adapter_request.origin_input_ids
        )
        # note (0xtoward): A cut edits SGLang's session in place, so SGLang must put exactly the kept history ahead of the unit.
        assert (
            history_cut is None or offset == history_length
        ), f"SGLang put {offset} history tokens ahead of the unit after a cut, not {history_length}"
        unit.native_offset = offset
        unit.prompt_length = len(adapter_request.origin_input_ids)
        request_data.session_history_length = offset
        unit.embedding_spans = [
            EmbeddingSpan(
                start=span.start + offset,
                end=span.end + offset,
                input_embeds=span.input_embeds,
            )
            for span in request_data.unit_embedding_spans
        ]
        if any(span.start < 0 for span in unit.embedding_spans):
            raise ValueError("embedding span precedes the native sequence")
        else:
            pass
        # note (Junnan Li): Placeholder ids under embedding spans do not identify their rows, so such a unit stays out of the radix tree.
        session_request.skip_radix_cache_insert = bool(unit.embedding_spans)
        request_data.session_embedding_spans = [
            *session.embedding_spans,
            *unit.embedding_spans,
        ]
        session_request.logprob_start_len = adapter_request.logprob_start_len
        session_request._omni_prompt_cache_key = getattr(
            adapter_request, "_omni_prompt_cache_key", None
        )  # noqa: leading-underscore
        request_data.req = session_request

    def start_compaction(
        self, unit: SessionUnit, payload: StagePayload
    ) -> SGLangARRequestData | None:
        if not self.adapter.is_compaction_enabled or unit.is_compaction_checked:
            return None
        else:
            unit.is_compaction_checked = True
        session = self.sessions[unit.session_identity.id]
        if session.compaction is not None:
            return None
        else:
            pass
        native_session = self.bridge_scheduler.session_controller.get(
            unit.session_identity.id
        )
        history_length = retained_history_length(native_session)
        if not self.adapter.start_compaction(
            unit.session_identity,
            unit.chunk,
            payload,
            history_length,
            self.bridge_scheduler.max_req_len,
        ):
            return None
        else:
            session.compaction = SessionCompaction(payload=payload)
            request_data = self.adapter.build_compaction_request(
                unit.session_identity,
                uuid4().hex,
                payload,
                history_length,
                self.bridge_scheduler.max_req_len,
            )
            self.create_internal_request(session, request_data)
            return request_data

    def create_internal_request(
        self, session: BridgeSession, request_data: SGLangARRequestData
    ) -> None:
        compaction = session.compaction
        assert compaction is not None
        request_id = request_data.req.rid
        internal_request = InternalSessionRequest(
            request_id=request_id, session_identity=session.session_identity
        )
        self.internal_requests[request_id] = internal_request
        compaction.request_id = request_id
        self.append_native_request(
            internal_request, request_data, should_cut_history=False
        )

    def complete_internal_request(
        self, request_id: str, request_data: SGLangARRequestData
    ) -> tuple[SGLangARRequestData | None, StagePayload | None]:
        internal_request = self.internal_requests[request_id]
        session_identity = internal_request.session_identity
        session = self.sessions[session_identity.id]
        compaction = session.compaction
        assert compaction is not None and compaction.request_id == request_id
        native_session = self.bridge_scheduler.session_controller.get(
            session_identity.id
        )
        [last_node] = native_session.req_nodes.values()
        assert last_node.req is request_data.req, "internal request has not committed"
        session.embedding_spans.extend(internal_request.embedding_spans)
        if compaction.phase == "rebuild":
            prefix_token_count = internal_request.prompt_length
            request = request_data.req
            record = self.bridge_scheduler.tree_cache.session_records()[
                session_identity.id
            ]
            if (
                len(request.output_ids) != 1
                or native_session.committed_origin_len != prefix_token_count
                or native_session.committed_unpadded_len != prefix_token_count
                or native_session.committed_fill_len != prefix_token_count
                or record.kv_committed_len != prefix_token_count
                or list(request.full_untruncated_fill_ids)
                != list(request.origin_input_ids)
            ):
                raise RuntimeError("context rebuild committed more than its prefix")
            else:
                request.output_ids = request.output_ids[:0]
                assert retained_history_length(native_session) == prefix_token_count
            self.adapter.finish_compaction(session_identity)
            session.compaction = None
            self.internal_requests.pop(request_id)
            return None, compaction.payload
        else:
            outcome = self.adapter.complete_compaction_request(
                session_identity, request_data
            )
            if outcome == "continue":
                next_request_data = self.adapter.build_compaction_request(
                    session_identity,
                    uuid4().hex,
                    compaction.payload,
                    retained_history_length(native_session),
                    self.bridge_scheduler.max_req_len,
                )
            else:
                next_request_data = self.adapter.build_rebuild_request(
                    session_identity,
                    uuid4().hex,
                    compaction.payload,
                    self.bridge_scheduler.max_req_len,
                )
                self.release_streaming_session(session_identity.id)
                session.embedding_spans.clear()
                if not self.open_streaming_session(session_identity.id):
                    raise RuntimeError("native context rebuild session open failed")
                else:
                    compaction.phase = "rebuild"
            self.create_internal_request(session, next_request_data)
            self.internal_requests.pop(request_id)
            return next_request_data, None

    def fail_compaction(self, request_id: str) -> tuple[str, list[StagePayload]]:
        if request_id in self.internal_requests:
            session_identity = self.internal_requests[request_id].session_identity
        else:
            session_identity = self.units_by_request_id[request_id].session_identity
        session = self.sessions[session_identity.id]
        assert session.unit is not None
        external_request_id = session.unit.request_id
        self.discard_compaction(session)
        self.cancel(external_request_id)
        return external_request_id, self.close_session(session)

    def discard_compaction(self, session: BridgeSession) -> None:
        for internal_request in list(self.internal_requests.values()):
            if internal_request.session_identity == session.session_identity:
                self.cancel_native_request(internal_request)
                native_session = self.bridge_scheduler.session_controller.get(
                    session.session_identity.id
                )
                if native_session is not None:
                    native_session.abort_req()
                else:
                    pass
                self.internal_requests.pop(internal_request.request_id)
            else:
                pass
        self.adapter.cancel_compaction(session.session_identity)
        session.compaction = None

    def drop_history(
        self, session: BridgeSession, native_session: Session, cut: HistoryCut
    ) -> None:
        """Drop the cut from an idle session's ids, KV and embedding spans; later positions move down."""
        cut_token_ids(native_session, cut)
        record = self.bridge_scheduler.tree_cache.session_records().get(
            native_session.session_id
        )
        # note (0xtoward): After an abort the session holds no KV, and the next unit re-prefills the shortened ids.
        if record is not None and record.holds_kv:
            cut_retained_kv(
                self.bridge_scheduler.tree_cache,
                self.bridge_scheduler.token_to_kv_pool_allocator.get_kvcache(),
                record,
                cut,
                self.adapter.shift_key_positions,
            )
        else:
            pass
        session.embedding_spans = [
            EmbeddingSpan(
                start=cut.new_position(span.start),
                end=cut.new_position(span.end),
                input_embeds=span.input_embeds,
            )
            for span in session.embedding_spans
            if not cut.drops(span.start, span.end)
        ]
        logger.info(
            "Session %s: dropped %d history tokens in %d ranges, %d kept",
            native_session.session_id,
            cut.length,
            len(cut.ranges),
            native_session.committed_origin_len,
        )

    def release_append_unit(self, request_id: str) -> None:
        unit = self.units_by_request_id.pop(request_id, None)
        if unit is not None:
            self.adapter.on_abort(unit.session_identity, unit)
            streaming_session = self.bridge_scheduler.session_controller.get(
                unit.session_identity.id
            )
            if streaming_session is not None and unit.session_request is not None:
                streaming_session.abort_req()
            else:
                pass
            self.sessions[unit.session_identity.id].unit = None
        else:
            pass

    def cancel(self, request_id: str) -> None:
        unit = self.units_by_request_id[request_id]
        if self.sessions[unit.session_identity.id].compaction is not None:
            error = ValueError("context compaction interrupted; session closed")
            self.bridge_scheduler.emit_request_error(request_id, error)
            _, reads = self.fail_compaction(request_id)
            for payload in reads:
                self.bridge_scheduler.emit_request_error(payload.request_id, error)
        else:
            self.cancel_native_request(unit)
            self.release_append_unit(request_id)

    def cancel_native_request(self, unit: NativeSessionRequest) -> None:
        request_id = unit.request_id
        session_request = unit.session_request
        if (
            isinstance(unit, InternalSessionRequest)
            and session_request is not None
            and session_request.finished()
        ):
            return
        else:
            pass
        previous_cancelling_request_id = self.cancelling_request_id
        self.cancelling_request_id = request_id
        try:
            if not unit.is_enqueued:
                self.bridge_scheduler.abort(request_id)
                if (
                    isinstance(unit, InternalSessionRequest)
                    and session_request is not None
                ):
                    native_session = self.bridge_scheduler.session_controller.get(
                        unit.session_identity.id
                    )
                    native_session.abort_req()
                else:
                    pass
            else:
                assert session_request is not None
                is_queued = any(
                    queued_request is session_request
                    for queued_request in self.bridge_scheduler.waiting_queue
                )
                if is_queued:
                    session_request.detach_kv()
                    session_request.session = None
                    self.bridge_scheduler.abort(request_id)
                    self.drain()
                else:
                    self.bridge_scheduler.abort(request_id)
                    self.drain()
                    session_request.finished_reason = FINISH_ABORT()
                    self.bridge_scheduler.release_request_kv_cache(session_request)
                    if self.bridge_scheduler.chunked_req is session_request:
                        self.bridge_scheduler.chunked_req = None
                    else:
                        pass
                    if session_request.omni_data is not None:
                        self.bridge_scheduler.run_abort_callback(request_id)
                        session_request.omni_data = None
                    else:
                        pass
        finally:
            self.cancelling_request_id = previous_cancelling_request_id

    def check_session_capacity(self, request_id: str) -> str | None:
        if request_id in self.internal_requests:
            unit = self.internal_requests[request_id]
        else:
            unit = self.units_by_request_id[request_id]
        requests = [
            *self.units_by_request_id.values(),
            *self.internal_requests.values(),
        ]
        session_request = unit.session_request
        assert (
            session_request is not None
        ), f"session capacity check {request_id} requires a session request"
        cache = self.bridge_scheduler.tree_cache
        records = cache.session_records()
        record = records.get(unit.session_identity.id)
        retained_kv_tokens = record.kv_allocated_len if record is not None else 0
        free_request_slots = self.bridge_scheduler.req_to_token_pool.free_slots
        unallocated_request_count = sum(
            active_append_unit is not unit
            and active_append_unit.session_request is not None
            and not active_append_unit.session_request.kv.holds_kv
            and active_append_unit.session_identity.id not in records
            for active_append_unit in requests
        )
        if (
            not retained_kv_tokens
            and len(free_request_slots)
            <= unallocated_request_count
            + REQUEST_TO_TOKEN_SLOTS_RESERVED_FOR_RETAINED_KV
        ):
            capacity_message = (
                "session request-to-token slots exhausted "
                "(one slot kept for a session that already holds KV)"
            )
        else:
            required_kv_tokens = len(session_request.origin_input_ids) + int(
                session_request.sampling_params.max_new_tokens or 0
            )
            reserved_kv_tokens = 0
            for active_append_unit in requests:
                if (
                    active_append_unit is unit
                    or active_append_unit.session_request is None
                ):
                    continue
                else:
                    active_session_request = active_append_unit.session_request
                    allocated_kv_tokens = active_session_request.kv.kv_allocated_len
                    reserved_kv_tokens += max(
                        0,
                        len(active_session_request.origin_input_ids)
                        + int(
                            active_session_request.sampling_params.max_new_tokens or 0
                        )
                        - allocated_kv_tokens,
                    )
            available_kv_tokens = (
                self.bridge_scheduler.token_to_kv_pool_allocator.available_size()
                + cache.evictable_size()
            )
            if (
                required_kv_tokens - retained_kv_tokens + reserved_kv_tokens
                > available_kv_tokens
            ):
                capacity_message = "session KV capacity exhausted"
            else:
                capacity_message = None
        return capacity_message

    def count_row_reusing_requests(
        self, waiting_queue: list[Req], free_request_rows: int
    ) -> int:
        """Count queued units that reuse their slot's row, up to the first request no free row is left for."""
        records = self.bridge_scheduler.tree_cache.session_records()
        fresh_request_count = 0
        row_reusing_request_count = 0
        for queued_request in waiting_queue:
            session = queued_request.session
            if session is not None and session.streaming:
                record = records.get(session.session_id)
            else:
                record = None
            # note (Junnan Li): An aborted unit does not take its slot's row.
            if (
                record is not None
                and record.holds_kv
                and queued_request.to_finish is None
            ):
                row_reusing_request_count += 1
            elif fresh_request_count < free_request_rows:
                fresh_request_count += 1
            else:
                break
        return row_reusing_request_count

    def stream_messages(
        self,
        request_id: str,
        request_data: SGLangARRequestData,
        output: RequestOutput | None = None,
        *,
        should_flush: bool = False,
    ) -> Iterable[OutgoingMessage]:
        unit = self.units_by_request_id[request_id]
        if should_flush:
            chunks = self.adapter.flush(unit.session_identity, request_data)
        else:
            assert (
                output is not None
            ), f"session stream {request_id} requires request output"
            chunks = self.adapter.stream(unit.session_identity, request_data, output)
        for chunk in chunks:
            yield OutgoingMessage(
                request_id=request_id,
                type="stream",
                data=chunk.to_dict(),
                metadata={"modality": chunk.modality},
            )

    def complete(
        self, request_id: str, request_data: SGLangARRequestData | None = None
    ) -> None:
        unit = self.units_by_request_id.pop(request_id, None)
        if unit is not None:
            recorder = get_recorder()
            if recorder.is_active():
                record = self.bridge_scheduler.tree_cache.session_records().get(
                    unit.session_identity.id
                )
                allocator = self.bridge_scheduler.token_to_kv_pool_allocator
                available_kv_positions = int(allocator.available_size())
                kv_pool_capacity_positions = int(allocator.size)
                session_logical_length = (
                    0 if record is None else int(record.kv_allocated_len)
                )
                session_swa_evicted_length = (
                    0
                    if record is None
                    else int(record.get_evicted_seqlen(ComponentType.SWA))
                )
                recorder.emit(
                    request_id=request_id,
                    stage=None,
                    event_name="session_kv_snapshot",
                    metadata={
                        "session_id": unit.session_identity.id,
                        "session_open_index": unit.session_identity.open_index,
                        "input_seq": unit.chunk.seq,
                        "cache_type": type(self.bridge_scheduler.tree_cache).__name__,
                        "kv_allocator_type": type(
                            self.bridge_scheduler.token_to_kv_pool_allocator
                        ).__name__,
                        "session_logical_length": session_logical_length,
                        "session_swa_evicted_length": session_swa_evicted_length,
                        "session_window_kv_positions": (
                            session_logical_length - session_swa_evicted_length
                        ),
                        "kv_pool_capacity_positions": kv_pool_capacity_positions,
                        "kv_pool_allocated_positions": (
                            kv_pool_capacity_positions - available_kv_positions
                        ),
                        "available_kv_positions": available_kv_positions,
                        "evictable_kv_positions": int(
                            self.bridge_scheduler.tree_cache.evictable_size()
                        ),
                    },
                )
            else:
                pass
            session = self.sessions[unit.session_identity.id]
            session.unit = None
            session.embedding_spans.extend(unit.embedding_spans)
            if (
                request_data is not None
                and request_data.input_embeds_are_projected
                and request_data.decode_input_embeds
            ):
                # Note (Junnan Li): A projected talker feeds each generated token back as a computed row, so a KV rebuild needs those rows, not the token embeddings.
                start = unit.native_offset + unit.prompt_length
                session.embedding_spans.append(
                    EmbeddingSpan(
                        start=start,
                        end=start + len(request_data.decode_input_embeds),
                        input_embeds=torch.stack(request_data.decode_input_embeds),
                    )
                )
            else:
                pass
        else:
            pass

    def close_streaming_session(self, session: BridgeSession) -> None:
        if session.compaction is not None:
            self.discard_compaction(session)
        else:
            pass
        if session.unit is not None:
            self.bridge_scheduler.abort(session.unit.request_id)
        else:
            pass
        self.release_streaming_session(session.session_identity.id)

    def close_session(self, session: BridgeSession) -> list[StagePayload]:
        """Release the session; returns its read, answered."""
        self.close_streaming_session(session)
        self.adapter.close(session.session_identity)
        self.sessions.pop(session.session_identity.id)
        if session.output_read is None:
            return []
        else:
            session.output_read.data = {"closed": True}
            return [session.output_read]

    def close_open_sessions(self) -> None:
        adapter_failures: dict[str, Exception] = {}
        for session in list(self.sessions.values()):
            self.close_streaming_session(session)
            session_id = session.session_identity.id
            try:
                self.adapter.close(session.session_identity)
            except Exception as adapter_error:
                adapter_failures[session_id] = adapter_error
            else:
                self.sessions.pop(session_id)
        if not adapter_failures:
            return
        else:
            failure_text = "; ".join(
                f"{session_id}: {adapter_error}"
                for session_id, adapter_error in adapter_failures.items()
            )
            raise RuntimeError(
                f"AR session adapter cleanup failed for {failure_text}"
            ) from next(iter(adapter_failures.values()))

# SPDX-License-Identifier: Apache-2.0
"""MiniCPM-o unit conversion for the shared session runtime."""

from array import array
from dataclasses import dataclass, field
from typing import Literal
from uuid import uuid4

import torch
from sglang.srt.layers.rotary_embedding.base import RotaryEmbedding
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.sampling.sampling_params import SamplingParams
from transformers import PreTrainedTokenizerBase

from sglang_omni.admission import ContextExhaustedError
from sglang_omni.audio.vad import StatelessVAD
from sglang_omni.client.client import Client
from sglang_omni.models.minicpm_o.components.code2wav import OUTPUT_SAMPLE_RATE
from sglang_omni.models.minicpm_o.components.streaming_perception import (
    SAMPLE_RATE,
    UNIT_MS,
    PerceptionStepPlan,
    SummaryAudioTask,
)
from sglang_omni.models.minicpm_o.context_compaction import (
    MEMORY_HEADER,
    MiniCPMOContextMemory,
    MiniCPMOContextSummary,
    build_internal_request,
    build_summary_request,
)
from sglang_omni.models.minicpm_o.native_config import (
    MAX_DRAIN_UNITS,
    MiniCPMODuplexPipelineConfig,
    MiniCPMODuplexSampling,
    MiniCPMODuplexWindow,
)
from sglang_omni.models.minicpm_o.special_tokens import (
    REQUIRED_SPECIAL_TOKENS,
    MiniCPMOSpecialTokenIds,
    resolve_special_token_ids,
)
from sglang_omni.models.minicpm_o.thinker_state import (
    DuplexUnitRequestData,
    MiniCPMOThinkerSessionState,
    ReplyUnit,
)
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.proto.session import OutputChunk, SessionIdentity, TimedChunk
from sglang_omni.proto.streams import (
    CONTROL,
    INTERRUPT,
    PCM16_BYTES_PER_SAMPLE,
    SESSION_CONTROL,
    TRUNCATE,
    control_type,
)
from sglang_omni.scheduling.sglang_backend.ar_session import (
    ARSessionAdapter,
    ARSessionCompactionPlan,
    ARSessionHistory,
    ARSessionPreparation,
)
from sglang_omni.scheduling.sglang_backend.history_cut import HistoryCut
from sglang_omni.scheduling.sglang_backend.request_data import (
    EmbeddingSpan,
    SGLangARRequestData,
)
from sglang_omni.serve.realtime.adapters import CoordinatorAdapter
from sglang_omni.serve.realtime.manager import (
    RealtimeDeployment,
    runtime_limits_for_pipeline,
)
from sglang_omni.serve.realtime.output import (
    AudioDelta,
    AudioFinished,
    OutputEvent,
    ResponseFinished,
    ResponseStarted,
    ResponseStatus,
    TextDelta,
    TextFinished,
)
from sglang_omni.serve.realtime.types import Capabilities, samples_to_ms

# note (ruoyu): A client's truncate follows its stop within a unit or two; this many listen units after a reply's last spoken unit, a cut would re-feed more audio than any stop needs, so the reply is no longer cut.
TRUNCATE_WINDOW_UNITS = 10


def keep_through_unit_of(event: TimedChunk) -> int | None:
    """The unit a stop or truncate control keeps the reply through; None when the payload carries none, as any session client may send."""
    keep_through_unit = event.payload.get("keep_through_unit")
    if isinstance(keep_through_unit, int):
        return keep_through_unit
    else:
        return None


class ThinkerAdapter(ARSessionAdapter):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        vocab_size: int,
        *,
        window: MiniCPMODuplexWindow,
        rotary_embedding: RotaryEmbedding,
    ) -> None:
        assert (
            rotary_embedding.is_neox_style
            and rotary_embedding.rotary_dim == rotary_embedding.head_size
        ), "the thinker window shifts keys with full-width neox RoPE"
        self.tokenizer: PreTrainedTokenizerBase = tokenizer
        self.vocab_size: int = vocab_size
        self.window: MiniCPMODuplexWindow = window
        self.rotary_embedding: RotaryEmbedding = rotary_embedding
        self.special: MiniCPMOSpecialTokenIds = resolve_special_token_ids(
            tokenizer, bad_token_ids=tuple(tokenizer.bad_token_ids)
        )
        self.states: dict[SessionIdentity, MiniCPMOThinkerSessionState] = {}
        self.compaction_enabled = window.sliding_window_mode == "summary"
        self.context_memories: dict[SessionIdentity, MiniCPMOContextMemory] = {}
        self.summary_control_token_ids: set[int] = (
            set(tokenizer.all_special_ids)
            | set(tokenizer.convert_tokens_to_ids(list(REQUIRED_SPECIAL_TOKENS)))
            if self.compaction_enabled
            else set()
        )

    def open(self, session_identity: SessionIdentity, request: OmniRequest) -> None:
        self.states[session_identity] = MiniCPMOThinkerSessionState(
            sampling=MiniCPMODuplexSampling.model_validate(
                {
                    key: request.params[key]
                    for key in MiniCPMODuplexSampling.model_fields
                }
            ),
            speech_detector=StatelessVAD(),
        )

        if self.compaction_enabled:
            self.context_memories[session_identity] = MiniCPMOContextMemory()
        else:
            pass

    def close(self, session_identity: SessionIdentity) -> None:
        self.states.pop(session_identity, None)
        self.context_memories.pop(session_identity, None)

    def control(self, session_identity: SessionIdentity, event: TimedChunk) -> None:
        if (
            control_type(event) in (INTERRUPT, TRUNCATE)
            and "reply_start_unit" in event.payload
        ):
            reply_start_unit = event.payload["reply_start_unit"]
            state = self.states[session_identity]
            if type(reply_start_unit) is not int or reply_start_unit < 0:
                raise ValueError("reply_start_unit must be a nonnegative integer")
            elif self.compaction_enabled and (
                not state.reply_units
                or state.reply_units[0].unit_index != reply_start_unit
            ):
                return
            else:
                pass
        else:
            pass
        if control_type(event) == INTERRUPT:
            state = self.states[session_identity]
            state.is_interrupt_pending = True
            state.is_listen_held = True
            # note (luojiaxuan): The detector hears no unit between holds, so each hold starts it afresh.
            state.speech_detector.reset()
            state.drain_units_left = 0
            state.is_latest_reply_ended = True
            self.cut_reply(state, keep_through_unit_of(event))
        elif control_type(event) == TRUNCATE:
            state = self.states[session_identity]
            if state.is_latest_reply_ended:
                self.cut_reply(state, keep_through_unit_of(event))
            else:
                # note (ruoyu): The truncate names a reply that ended, but the model may already have begun the next one, whose units the client will hear.
                pass
        else:
            pass

    def cut_reply(
        self, state: MiniCPMOThinkerSessionState, keep_through_unit: int | None
    ) -> None:
        """Drop the latest reply's spoken units past keep_through_unit: the next unit rewinds the session to where the first of them began, refills that unit as the official forced listen, turn_eos first when the turn was open before it, and re-feeds every later unit's own ids followed by a listen, so the user's audio of those seconds stays in context."""
        dropped_positions = [
            position
            for position, unit in enumerate(state.reply_units)
            if keep_through_unit is not None
            and unit.is_spoken
            and unit.unit_index > keep_through_unit
        ]
        if not dropped_positions:
            pass
        else:
            first_dropped_unit = state.reply_units[dropped_positions[0]]
            refill_token_ids = (
                [self.special.listen]
                if first_dropped_unit.is_turn_ended_at_start
                else [self.special.turn_eos, self.special.listen]
            )
            refill_embedding_spans: list[EmbeddingSpan] = []
            for later_unit in state.reply_units[dropped_positions[0] + 1 :]:
                refill_token_ids.append(self.special.unit_end)
                refill_embedding_spans.extend(
                    EmbeddingSpan(
                        start=span.start + len(refill_token_ids),
                        end=span.end + len(refill_token_ids),
                        input_embeds=span.input_embeds,
                    )
                    for span in later_unit.perception_embedding_spans
                )
                refill_token_ids.extend(later_unit.perception_token_ids)
                refill_token_ids.append(self.special.listen)
            state.kept_token_count = first_dropped_unit.kept_token_count
            state.refill_token_ids = refill_token_ids
            state.refill_embedding_spans = refill_embedding_spans
            state.is_turn_ended = True
            state.generated_history = list(
                first_dropped_unit.generated_history_at_start
            )
            for position in dropped_positions:
                state.reply_units[position].is_spoken = False

    def prepare_unit(
        self,
        session_identity: SessionIdentity,
        chunk: TimedChunk,
        payload: StagePayload,
    ) -> ARSessionPreparation:
        state = self.states[session_identity]
        kept_token_count = state.kept_token_count
        state.kept_token_count = None
        if kept_token_count is not None:
            state.unit_starts = [
                start for start in state.unit_starts if start < kept_token_count
            ]
        else:
            pass
        return ARSessionPreparation(kept_token_count=kept_token_count)

    def finish_input(
        self, session_identity: SessionIdentity, payload: StagePayload
    ) -> StagePayload:
        payload.data = dict(
            talker_conditions=[],
            text="",
            is_listen=True,
            end_of_turn=True,
            is_interrupted=self.states[session_identity].is_interrupt_pending,
            hears_user=False,
        )
        return payload

    def build(
        self,
        session_identity: SessionIdentity,
        chunk: TimedChunk,
        payload: StagePayload,
    ) -> DuplexUnitRequestData:
        state = self.states[session_identity]
        plan: PerceptionStepPlan = payload.data
        if self.compaction_enabled and "summary_audio" in plan:
            memory = self.context_memories[session_identity]
            if memory.summary_audio is None:
                task = plan["summary_audio"]
                memory.summary_audio = SummaryAudioTask(
                    instruction_units=task["instruction_units"],
                    plans=[
                        PerceptionStepPlan(
                            token_ids=list(audio_plan["token_ids"]),
                            input_embeds=audio_plan["input_embeds"].detach().clone(),
                            embedding_spans=[
                                span.copy() for span in audio_plan["embedding_spans"]
                            ],
                        )
                        for audio_plan in task["plans"]
                    ],
                )
            else:
                pass
        else:
            pass
        is_interrupted = state.is_interrupt_pending
        state.is_interrupt_pending = False
        if state.is_listen_held or state.drain_units_left > 0:
            pcm = (
                chunk.payload["pcm"]
                if isinstance(chunk.payload, dict)
                else chunk.payload
            )
            hears_user = any(
                state.speech_detector.step(frame, in_speech=False) is not None
                for frame in state.speech_detector.score(
                    pcm, round(chunk.t_start_ms * SAMPLE_RATE / 1000)
                )
            )
        else:
            hears_user = False
        state.is_listen_held = state.is_listen_held and not hears_user
        state.drain_units_left = 0 if hears_user else max(state.drain_units_left - 1, 0)
        prefix = (
            []
            if state.is_prefix_pending
            else [*state.refill_token_ids, self.special.unit_end]
        )
        refill_embedding_spans = state.refill_embedding_spans
        state.refill_token_ids = []
        state.refill_embedding_spans = []
        if is_interrupted and not state.is_turn_ended:
            # note (ruoyu): Without turn_eos the model reads the stop as a pause in its own turn.
            suffix = [self.special.turn_eos]
        else:
            suffix = []
        token_ids = [*prefix, *plan["token_ids"], *suffix]
        perception_embedding_spans = [
            EmbeddingSpan(
                start=span["token_start"],
                end=span["token_end"],
                input_embeds=plan["input_embeds"][
                    span["embedding_start"] : span["embedding_end"]
                ],
            )
            for span in plan["embedding_spans"]
        ]
        spans = [
            *refill_embedding_spans,
            *(
                EmbeddingSpan(
                    start=span.start + len(prefix),
                    end=span.end + len(prefix),
                    input_embeds=span.input_embeds,
                )
                for span in perception_embedding_spans
            ),
        ]
        sampling = state.sampling
        sampling_params = SamplingParams(
            max_new_tokens=sampling.max_new_tokens_per_unit,
            temperature=1.0,
            top_p=1.0,
            top_k=-1,
            repetition_penalty=1.0,
            stop_token_ids=set(self.special.chunk_terminators),
            no_stop_trim=True,
            skip_special_tokens=False,
        )
        sampling_params.normalize(self.tokenizer)
        adapter_request = Req(
            payload.request_id,
            "",
            array("q", token_ids),
            sampling_params,
            vocab_size=self.vocab_size,
        )
        adapter_request.tokenizer = self.tokenizer
        adapter_request.return_hidden_states = True
        return DuplexUnitRequestData(
            req=adapter_request,
            input_ids=torch.tensor(token_ids),
            max_new_tokens=sampling.max_new_tokens_per_unit,
            stage_payload=payload,
            thinker_state=state,
            # note (0xtoward): The last <unit> is the unit's own; a system prompt ahead of it may spell one.
            unit_token_index=max(
                index
                for index, token_id in enumerate(token_ids)
                if token_id == self.special.unit
            ),
            unit_embedding_spans=spans,
            is_listen_forced=is_interrupted
            or state.is_listen_held
            or state.force_listen_counter < sampling.force_listen_count,
            is_interrupted=is_interrupted,
            hears_user=hears_user,
            unit_index=chunk.seq,
            perception_token_ids=plan["token_ids"],
            perception_embedding_spans=perception_embedding_spans,
            is_turn_ended_at_start=state.is_turn_ended,
            generated_history_at_start=list(state.generated_history),
        )

    def plan_history_cut(
        self,
        session_identity: SessionIdentity,
        request_data: DuplexUnitRequestData,
        history_length: int,
        history_capacity: int,
    ) -> HistoryCut | None:
        """The checkpoint's StreamDecoder.enforce_window: past the high watermark, the oldest units go until the low one.

        It also cuts early when the next unit would not fit in the context.
        """
        window = self.window
        # note (0xtoward): The checkpoint checks its cache length after </unit>, which here comes with the next unit, so that length is the next <unit>'s position.
        next_unit_start = history_length + request_data.unit_token_index
        fitting_start = history_capacity + request_data.unit_token_index
        if window.sliding_window_mode in {"off", "summary"} or (
            next_unit_start <= window.basic_window_high_tokens
            and next_unit_start <= fitting_start
        ):
            return None
        else:
            state = self.states[session_identity]
            return state.drop_units(
                range(len(state.unit_starts)),
                next_unit_start,
                min(window.basic_window_low_tokens, fitting_start),
            )

    def plan_compaction(
        self,
        session_identity: SessionIdentity,
        payload: StagePayload,
        history: ARSessionHistory,
        context_limit: int,
    ) -> ARSessionCompactionPlan | None:
        state = self.states[session_identity]
        memory = self.context_memories[session_identity]
        if memory.prefix is None:
            prefix_tokens = state.unit_starts[0]
            memory.prefix = ARSessionHistory(
                token_ids=history.token_ids[:prefix_tokens],
                embedding_spans=[
                    EmbeddingSpan(
                        start=span.start,
                        end=span.end,
                        input_embeds=span.input_embeds.detach().clone(),
                    )
                    for span in history.embedding_spans
                    if span.end <= prefix_tokens
                ],
            )
        else:
            pass
        plan: PerceptionStepPlan = payload.data
        next_unit_tokens = (
            len(plan["token_ids"])
            + (0 if state.is_prefix_pending else len(state.refill_token_ids) + 1)
            + int(state.is_interrupt_pending and not state.is_turn_ended)
            + state.sampling.max_new_tokens_per_unit
        )
        task = memory.summary_audio
        if task is None:
            raise ValueError("context summary requires encoded instruction audio")
        else:
            pass
        summary_reserve = (
            sum(1 + len(audio_plan["token_ids"]) for audio_plan in task["plans"])
            + int(not state.is_turn_ended)
            + self.window.summary_max_tokens
        )
        history_tokens = len(history.token_ids)
        if memory.has_failed:
            if history_tokens + next_unit_tokens >= context_limit:
                raise ContextExhaustedError(
                    "context_exhausted: compaction failed and retained history is full"
                )
            else:
                return None
        elif self.window.summary_trigger_tokens + summary_reserve >= context_limit:
            raise ValueError(
                "summary trigger plus instruction and output reserve must fit the context"
            )
        else:
            pass
        memory.is_compaction_pending = (
            memory.is_compaction_pending
            or history_tokens >= self.window.summary_trigger_tokens
            or history_tokens + next_unit_tokens + summary_reserve >= context_limit
        )
        if not memory.is_compaction_pending:
            return None
        else:
            pass
        repaired_history = ARSessionHistory(
            token_ids=[
                *history.token_ids,
                *state.refill_token_ids,
            ],
            embedding_spans=[
                *history.embedding_spans,
                *(
                    EmbeddingSpan(
                        start=history_tokens + span.start,
                        end=history_tokens + span.end,
                        input_embeds=span.input_embeds,
                    )
                    for span in state.refill_embedding_spans
                ),
            ],
        )
        if (
            history_tokens + len(state.refill_token_ids) + summary_reserve
            >= context_limit
        ):
            raise ContextExhaustedError(
                "context_exhausted: summary request would exceed context"
            )
        else:
            summary = MiniCPMOContextSummary(
                thinker_state=MiniCPMOThinkerSessionState(
                    sampling=state.sampling,
                    speech_detector=state.speech_detector,
                    generated_history=list(state.generated_history),
                    force_listen_counter=state.force_listen_counter,
                    is_turn_ended=state.is_turn_ended,
                    is_prefix_pending=False,
                ),
                audio_task=task,
            )
            memory.summary = summary
            request_data = build_summary_request(
                f"{payload.request_id}:summary:{uuid4().hex}",
                repaired_history,
                self.tokenizer,
                self.vocab_size,
                min(
                    state.sampling.max_new_tokens_per_unit,
                    self.window.summary_max_tokens,
                ),
                summary,
                self.special,
                context_limit,
                is_interrupted=state.is_interrupt_pending or not state.is_turn_ended,
            )
            return ARSessionCompactionPlan(
                request_data=request_data,
                timeout_seconds=self.window.summary_timeout_seconds,
                continuation_tokens=len(plan["token_ids"])
                + state.sampling.max_new_tokens_per_unit,
            )

    def continue_compaction(
        self,
        session_identity: SessionIdentity,
        summary: SGLangARRequestData,
        context_limit: int,
    ) -> SGLangARRequestData | None:
        assert isinstance(summary, DuplexUnitRequestData)
        collected = self.context_memories[session_identity].summary
        assert collected is not None
        output_token_ids = list(summary.output_ids)
        collected.generated_tokens += len(output_token_ids)
        if (
            summary.finish_reason not in {"stop", "length"}
            or not output_token_ids
            or output_token_ids[-1] not in self.special.chunk_terminators
        ):
            raise ValueError("context summary did not finish its native unit")
        elif collected.generated_tokens > self.window.summary_max_tokens:
            raise ValueError("context summary exceeded its output budget")
        else:
            collected.audio_units_consumed += 1
        instruction_units = collected.audio_task["instruction_units"]
        if collected.audio_units_consumed < instruction_units:
            collected.token_ids.clear()
        else:
            collected.token_ids.extend(
                token_id
                for token_id in summary.generated_unit_ids
                if token_id not in self.summary_control_token_ids
            )
        if (
            collected.audio_units_consumed >= instruction_units
            and self.special.turn_eos in output_token_ids
            and collected.token_ids
        ):
            collected.is_complete = True
            return None
        else:
            pass
        if collected.generated_tokens >= self.window.summary_max_tokens:
            raise ValueError(
                "context summary exhausted its output budget before turn end"
            )
        elif collected.audio_units_consumed >= len(collected.audio_task["plans"]):
            raise ValueError(
                "context summary exhausted its audio units before turn end"
            )
        else:
            pass
        remaining_tokens = self.window.summary_max_tokens - collected.generated_tokens
        return build_summary_request(
            f"{summary.req.rid}:continue:{uuid4().hex}",
            ARSessionHistory(
                token_ids=[*summary.req.origin_input_ids, *output_token_ids],
                embedding_spans=list(summary.session_embedding_spans),
            ),
            self.tokenizer,
            self.vocab_size,
            min(
                collected.thinker_state.sampling.max_new_tokens_per_unit,
                remaining_tokens,
            ),
            collected,
            self.special,
            context_limit,
        )

    def build_compacted_history(
        self,
        session_identity: SessionIdentity,
        summary: SGLangARRequestData,
        context_limit: int,
    ) -> SGLangARRequestData:
        memory = self.context_memories[session_identity]
        collected = memory.summary
        assert collected is not None
        if (
            not collected.is_complete
            or not self.tokenizer.decode(
                collected.token_ids, skip_special_tokens=True
            ).strip()
        ):
            raise ValueError("context summary did not finish with nonempty text")
        else:
            pass
        prefix = memory.prefix
        assert prefix is not None
        token_ids = [
            *prefix.token_ids,
            *self.tokenizer.encode(MEMORY_HEADER, add_special_tokens=False),
            *collected.token_ids,
            *self.tokenizer.encode("<|im_end|>\n", add_special_tokens=False),
        ]
        if len(token_ids) + 1 >= min(context_limit, self.window.summary_trigger_tokens):
            raise ValueError("compacted history does not leave continuation budget")
        else:
            return build_internal_request(
                f"{summary.req.rid}:rebuild",
                token_ids,
                prefix.embedding_spans,
                self.tokenizer,
                self.vocab_size,
                1,
                collected.thinker_state,
                self.special,
            )

    def finish_compaction(
        self,
        session_identity: SessionIdentity,
        *,
        status: Literal["completed", "invalidated", "failed"],
    ) -> None:
        memory = self.context_memories[session_identity]
        memory.summary = None
        if status == "completed":
            state = self.states[session_identity]
            memory.is_compaction_pending = False
            state.unit_starts.clear()
            state.generated_history.clear()
            # shortcut: A committed summary closes the old reply's truncate window; revising summarized speech requires retaining and replaying the old history.
            state.reply_units.clear()
            state.is_latest_reply_ended = False
            state.units_since_last_spoken = 0
            state.is_prefix_pending = True
            state.is_turn_ended = True
            state.refill_token_ids.clear()
            state.refill_embedding_spans.clear()
        elif status == "failed":
            memory.is_compaction_pending = False
            memory.has_failed = True
        else:
            pass

    def shift_key_positions(
        self, keys: torch.Tensor, distances: torch.Tensor
    ) -> torch.Tensor:
        cosine, sine = (
            self.rotary_embedding.cos_sin_cache[distances]
            .to(torch.float32)
            .unsqueeze(1)
            .chunk(2, dim=-1)
        )
        first_half, second_half = keys.to(torch.float32).chunk(2, dim=-1)
        return torch.cat(
            (
                first_half * cosine + second_half * sine,
                second_half * cosine - first_half * sine,
            ),
            dim=-1,
        ).to(keys.dtype)

    def result(
        self, session_identity: SessionIdentity, data: DuplexUnitRequestData
    ) -> StagePayload:
        state = self.states[session_identity]
        starts_new_context = state.is_prefix_pending
        state.is_prefix_pending = False
        state.unit_starts.append(data.session_history_length + data.unit_token_index)
        if data.is_interrupted:
            # note (ruoyu): Without the ended turn, the next sampled listen is rewritten into speech and the stopped reply resumes.
            state.is_turn_ended = True
        else:
            pass
        output_token_ids = [int(token_id) for token_id in data.output_ids]
        ends_own_turn = not data.is_interrupted and (
            self.special.turn_eos in output_token_ids
            or any(ends_turn for _, _, ends_turn in data.talker_conditions)
        )
        is_turn_ending = data.is_interrupted or ends_own_turn
        if ends_own_turn:
            state.drain_units_left = MAX_DRAIN_UNITS
            state.speech_detector.reset()
        else:
            pass
        if data.generated_unit_ids and data.is_turn_ended_at_start:
            # note (ruoyu): A new reply begins here, so the reply before it can no longer be cut.
            state.reply_units = []
            state.is_latest_reply_ended = False
        else:
            pass
        if data.generated_unit_ids or state.reply_units:
            state.reply_units.append(
                ReplyUnit(
                    unit_index=data.unit_index,
                    is_spoken=bool(data.generated_unit_ids),
                    # note (ruoyu): The native request's prompt is the retained turns plus this unit's ids, so dropping the unit's output keeps exactly that many tokens.
                    kept_token_count=len(data.req.origin_input_ids),
                    perception_token_ids=data.perception_token_ids,
                    perception_embedding_spans=data.perception_embedding_spans,
                    is_turn_ended_at_start=data.is_turn_ended_at_start,
                    generated_history_at_start=data.generated_history_at_start,
                )
            )
            if data.generated_unit_ids:
                state.units_since_last_spoken = 0
            elif state.units_since_last_spoken < TRUNCATE_WINDOW_UNITS:
                state.units_since_last_spoken += 1
            else:
                state.reply_units = []
                state.is_latest_reply_ended = False
        else:
            pass
        if is_turn_ending and state.reply_units:
            # note (ruoyu): A reply the model ended on hearing the user is cut by a truncate like a stopped one.
            state.is_latest_reply_ended = True
        else:
            pass
        data.stage_payload.data = dict(
            starts_new_context=starts_new_context,
            reply_start_unit=(
                state.reply_units[0].unit_index if state.reply_units else None
            ),
            talker_conditions=data.talker_conditions,
            text=self.tokenizer.decode(
                data.generated_unit_ids, skip_special_tokens=True
            ),
            is_listen=bool(
                output_token_ids and output_token_ids[0] == self.special.listen
            ),
            end_of_turn=is_turn_ending,
            is_interrupted=data.is_interrupted,
            hears_user=data.hears_user,
        )
        return data.stage_payload


@dataclass(frozen=True, kw_only=True)
class KeptReply:
    """A reply a client stop or truncate cuts: its turn, and the last unit it keeps, the last one the client received or the last one it played, whichever is earlier."""

    turn: int | None
    keep_through_unit: int
    reply_start_unit: int


@dataclass(frozen=True, kw_only=True)
class DeliveredUnit:
    """A unit whose output the client received, and where its audio ends in the reply's audio."""

    unit_index: int
    audio_end_ms: float


@dataclass(kw_only=True)
class Reply:
    """The latest reply: its ids and turn, what the client received of it, and the unit a stop or a truncate keeps it through."""

    response_id: str
    item_id: str
    turn: int | None
    reply_start_unit: int
    text: str = ""
    has_audio: bool = False
    audio_samples: int = 0
    is_open: bool = True
    is_cancelled: bool = False
    delivered_units: list[DeliveredUnit] = field(default_factory=list)
    keep_through_unit: int | None = None

    @property
    def last_delivered_unit(self) -> int:
        return self.delivered_units[-1].unit_index

    def played_through_unit(self, audio_end_ms: float) -> int:
        """The last delivered unit whose audio had fully ended at audio_end_ms, so a position inside a unit rounds down; the unit before the first one when none had."""
        played_unit_indices = [
            unit.unit_index
            for unit in self.delivered_units
            if unit.audio_end_ms <= audio_end_ms
        ]
        if played_unit_indices:
            return played_unit_indices[-1]
        else:
            return self.delivered_units[0].unit_index - 1


class OutputConverter:
    def __init__(self) -> None:
        self.reply: Reply | None = None
        self.serial: int = 0

    def cancel(self) -> KeptReply | None:
        """Mark the open reply cancelled and keep it through the last unit the client received; None when no reply is open or it is already cancelled."""
        reply = self.reply
        if reply is None or not reply.is_open or reply.is_cancelled:
            return None
        else:
            reply.is_cancelled = True
            reply.keep_through_unit = reply.last_delivered_unit
            return KeptReply(
                turn=reply.turn,
                keep_through_unit=reply.keep_through_unit,
                reply_start_unit=reply.reply_start_unit,
            )

    def truncate(self, item_id: str, audio_end_ms: float) -> KeptReply | None:
        """Keep the latest reply, when item_id names it and the client stopped it or the model ended it, through the last unit the client had played at audio_end_ms, never past a unit it received or a unit already kept; None when the position does not move back."""
        reply = self.reply
        if (
            reply is None
            or reply.item_id != item_id
            or (reply.is_open and not reply.is_cancelled)
        ):
            # note (luojiaxuan): Only the latest reply's units are still in reach of a cut, and an open reply the client has not stopped goes on past any cut.
            return None
        else:
            keep_through_unit = reply.played_through_unit(audio_end_ms)
            if (
                reply.keep_through_unit is not None
                and keep_through_unit >= reply.keep_through_unit
            ):
                return None
            else:
                reply.keep_through_unit = keep_through_unit
                return KeptReply(
                    turn=reply.turn,
                    keep_through_unit=keep_through_unit,
                    reply_start_unit=reply.reply_start_unit,
                )

    def finish(self, is_speech_cut: bool) -> list[OutputEvent]:
        """End the open response: cancelled when the client stopped it, incomplete when the talker's budget cut its speech."""
        reply = self.reply
        if reply is None or not reply.is_open:
            return []
        else:
            if reply.is_cancelled:
                status: ResponseStatus = "cancelled"
                reason = "interrupted"
            elif is_speech_cut:
                status = "incomplete"
                reason = "max_output_tokens"
            else:
                status = "completed"
                reason = "stop"
            events: list[OutputEvent] = [
                TextFinished(reply.response_id, reply.item_id, reply.text)
            ]
            if reply.has_audio:
                events.append(AudioFinished(reply.response_id, reply.item_id))
            else:
                pass
            events.append(
                ResponseFinished(
                    reply.response_id,
                    reply.item_id,
                    reply.text,
                    reply.has_audio,
                    status,
                    reason,
                )
            )
            reply.is_open = False
            return events

    def __call__(self, output: OutputChunk) -> list[OutputEvent]:
        value = output.payload
        reply = self.reply
        events: list[OutputEvent] = []
        if (
            reply is not None
            and reply.is_open
            and reply.is_cancelled
            and output.turn == reply.turn
        ):
            # note (ruoyu): The client stopped this reply, so what the pipeline still produces for its turn is not played.
            pass
        elif value["text"] or value["pcm"]:
            if reply is None or not reply.is_open:
                response_id = f"{output.session_identity.id}-response-{self.serial}"
                reply_start_unit = value.get("reply_start_unit", output.input_seq)
                assert isinstance(reply_start_unit, int)
                reply = Reply(
                    response_id=response_id,
                    item_id=response_id + "-message",
                    turn=output.turn,
                    reply_start_unit=reply_start_unit,
                )
                self.reply = reply
                self.serial += 1
                events.append(ResponseStarted(response_id))
            else:
                pass
            reply.audio_samples += len(value["pcm"]) // PCM16_BYTES_PER_SAMPLE
            delivered_unit = DeliveredUnit(
                unit_index=output.input_seq,
                audio_end_ms=samples_to_ms(reply.audio_samples, OUTPUT_SAMPLE_RATE),
            )
            if (
                reply.delivered_units
                and reply.delivered_units[-1].unit_index == output.input_seq
            ):
                # note (ruoyu): A unit vocoded in several chunks ends where its last chunk ends, so a position between its chunks still rounds down to the unit before it.
                reply.delivered_units[-1] = delivered_unit
            else:
                reply.delivered_units.append(delivered_unit)
            if value["text"]:
                reply.text += value["text"]
                events.append(
                    TextDelta(reply.response_id, reply.item_id, value["text"])
                )
            else:
                pass
            if value["pcm"]:
                reply.has_audio = True
                events.append(
                    AudioDelta(reply.response_id, reply.item_id, value["pcm"])
                )
            else:
                pass
        else:
            pass
        if value["end_of_turn"] or output.eos:
            events.extend(self.finish(value["is_speech_cut"]))
        else:
            pass
        return events


class RealtimeAdapter(CoordinatorAdapter):
    """The coordinator adapter with the client stop and the client truncate of a native duplex reply."""

    output_converter: OutputConverter
    is_compaction_enabled: bool = False

    async def cancel(self) -> None:
        """Interrupt the open reply's turn: the turn's later audio is dropped, and the thinker forgets the units the client did not receive, then listens until the user speaks again."""
        assert self.session_identity is not None
        kept_reply = self.output_converter.cancel()
        if kept_reply is None:
            pass
        else:
            await self.client.control_session(
                self.session_identity,
                TimedChunk(
                    CONTROL,
                    0,
                    0,
                    0,
                    {
                        "type": INTERRUPT,
                        "keep_through_unit": kept_reply.keep_through_unit,
                        "reply_start_unit": kept_reply.reply_start_unit,
                    },
                    format=SESSION_CONTROL,
                    turn=kept_reply.turn,
                ),
                is_ordered=not self.is_compaction_enabled,
            )

    async def truncate(self, item_id: str, audio_end_ms: float) -> None:
        """Cut the reply item_id names where the client stopped playing it: the thinker keeps it through the last unit whose audio had ended by then and forgets the rest, never more than the stop already kept."""
        assert self.session_identity is not None
        kept_reply = self.output_converter.truncate(item_id, audio_end_ms)
        if kept_reply is None:
            pass
        else:
            await self.client.control_session(
                self.session_identity,
                TimedChunk(
                    CONTROL,
                    0,
                    0,
                    0,
                    {
                        "type": TRUNCATE,
                        "t_ms": audio_end_ms,
                        "keep_through_unit": kept_reply.keep_through_unit,
                        "reply_start_unit": kept_reply.reply_start_unit,
                    },
                    format=SESSION_CONTROL,
                    turn=kept_reply.turn,
                ),
                is_ordered=not self.is_compaction_enabled,
            )


def build_realtime_deployment(
    client: Client, config: MiniCPMODuplexPipelineConfig
) -> RealtimeDeployment:

    def factory() -> RealtimeAdapter:
        adapter = RealtimeAdapter(
            client,
            request_builder=lambda session: OmniRequest(
                None,
                params={
                    "instructions": session.get("instructions")
                    or "Streaming Omni Conversation.",
                    **config.sampling.model_dump(),
                    **session.get("sglang", {}).get("sampling", {}),
                    **{
                        field: reference.data
                        for field in ("reference_audio", "tts_reference_audio")
                        if (reference := session.get("sglang", {}).get(field))
                        is not None
                    },
                    "max_slice_nums": session.get("sglang", {}).get(
                        "max_slice_nums", config.vision.max_slice_nums
                    ),
                },
            ),
            output_converter=OutputConverter(),
            input_sample_rate_hz=SAMPLE_RATE,
            atomic_consumption=True,
        )
        adapter.is_compaction_enabled = config.window.sliding_window_mode == "summary"
        return adapter

    return RealtimeDeployment(
        Capabilities(
            native_unit_ms=UNIT_MS,
            output_sample_rate_hz=OUTPUT_SAMPLE_RATE,
            output_modalities=("audio", "text"),
            input_modalities=("audio", "image"),
            image_frames_per_unit=tuple(
                min(
                    config.vision.max_frames_per_unit,
                    config.vision.max_tiles_per_unit
                    // (1 if slices == 1 else slices + 1),
                )
                for slices in range(1, config.vision.max_slice_nums_limit + 1)
            ),
            default_max_slice_nums=config.vision.max_slice_nums,
            tail_policy="pad",
            supports_reference_audio=True,
            supports_truncate=True,
            supports_cancel=True,
            sampling_parameters=tuple(MiniCPMODuplexSampling.model_fields),
        ),
        factory,
        limits=runtime_limits_for_pipeline(config),
        max_connections=config.max_sessions,
    )

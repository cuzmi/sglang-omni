# SPDX-License-Identifier: Apache-2.0
"""Opt-in counters for per-chunk pipeline orchestration overhead.

Set SGLANG_OMNI_STREAM_OVERHEAD_STATS=1 before launching the server.
Each stage logs one STREAM_OVERHEAD_STATS JSON record when it stops.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from collections import Counter
from collections.abc import Mapping

from sglang_omni.profiler.event_recorder import get_active_stage

logger = logging.getLogger(__name__)

STREAM_OVERHEAD_STATS_ENV = "SGLANG_OMNI_STREAM_OVERHEAD_STATS"
PENDING_TEXT_FALLBACK_STAGE = "qwen3_omni_pending_text"


def _enabled_from_env() -> bool:
    value = os.environ.get(STREAM_OVERHEAD_STATS_ENV, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


class StreamOverheadRecorder:
    """Thread-safe process-local integer counters grouped by stage."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._enabled = _enabled_from_env()
        self._counters: dict[str, Counter[str]] = {}

    @property
    def enabled(self) -> bool:
        return self._enabled

    def add(self, stage: str, counts: Mapping[str, int]) -> None:
        if not self._enabled:
            return
        normalized = {name: int(value) for name, value in counts.items() if value}
        if not normalized:
            return
        with self._lock:
            counters = self._counters.setdefault(stage, Counter())
            counters.update(normalized)

    def observe_max(self, stage: str, name: str, value: int) -> None:
        if not self._enabled:
            return
        value = int(value)
        with self._lock:
            counters = self._counters.setdefault(stage, Counter())
            counters[name] = max(int(counters.get(name, 0)), value)

    def log_stage(self, stage: str) -> None:
        if not self._enabled:
            return
        with self._lock:
            counters = dict(sorted(self._counters.get(stage, {}).items()))
        if not counters:
            return
        logger.info(
            "STREAM_OVERHEAD_STATS %s",
            json.dumps(
                {
                    "pid": os.getpid(),
                    "stage": stage,
                    "counters": counters,
                },
                sort_keys=True,
            ),
        )


_RECORDER = StreamOverheadRecorder()


def get_stream_overhead_recorder() -> StreamOverheadRecorder:
    return _RECORDER


def pending_text_stage() -> str:
    return get_active_stage() or PENDING_TEXT_FALLBACK_STAGE

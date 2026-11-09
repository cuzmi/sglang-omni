# MiniCPM context-memory audio fixture

Separate Mandarin user utterances for comparing MiniCPM's basic history window with a future summary-and-rebuild policy. Each WAV is one utterance. Play the clips in order within one session, waiting for each completed response before sending the next clip. Revision 3 contains individual clips and playback lists; it has no merged recording or fixed response gaps.

## Review the material

| File | Purpose |
| --- | --- |
| [clips/](audio_v1/clips/) | 38 individual WAVs: 6 initial messages, 30 filler questions, and 2 alternative probes |
| [long.playlist.json](audio_v1/long.playlist.json) | 70 turns: 6 initial messages, 63 filler turns, then the factual-recall probe |
| [short-control.playlist.json](audio_v1/short-control.playlist.json) | 7 turns: the same 6 initial messages and factual-recall probe |
| [scenario.json](audio_v1/scenario.json) | Clip transcripts, synthesis parameters, and filler order |
| [rubric.json](audio_v1/rubric.json) | Offline expected facts and validity checks; never send this file to the model |
| [manifest.json](audio_v1/manifest.json) | Clip durations, format, hashes, and verification status |
| [使用说明.txt](audio_v1/使用说明.txt) | Chinese usage notes |

Playlist file paths are relative to `audio_v1/`. The long list repeats the 30 filler clips in order for 63 turns; repeated entries reuse the same WAV bytes. To test applied planning, replace only the final probe with `clips/applied_planning.wav` and run it in a separate session. Running both probes in one session would refresh the facts through the first answer.

The clips use the macOS Tingting voice at a say rate of 180, in mono 16 kHz PCM16 WAV. They contain synthesized user speech. Their bytes are unchanged from revision 2, with no appended fixed response gaps. There is no fixed total session duration: it depends on the model's responses and the observed waiting time.

## Replay and comparison

Open one persistent `/v1/realtime` connection per run. Stream each clip at real-time pace, then wait for the current response's `response.done` with `status: completed` before sending the next clip. Match the response ID. If audible playback is part of the test, also wait for the local playback queue to empty. Record missing responses, timeouts, cancellations, and failures instead of silently continuing in a new session.

In the current MiniCPM native code path, generation advances per input unit. If the response is unfinished and queued units are exhausted, another audio unit is needed to advance generation. A replay driver can supply silence during the wait. This follows from the code and has not been verified in a model run. Do not call `run_session()` separately for each clip or send `sglang.input_audio.end` between questions; those operations open a new connection or end input for the whole session.

Both policies use the same clip list, sampling settings, response completion rule, and post-maintenance context budget. The summary consumes that budget too. Response durations can differ, producing different amounts of silence input. Record actual input times, silence durations, outputs, and context token counts; identical speech clips do not guarantee identical total input bytes or compute work.

First run the short control to check initial comprehension. Then use the long list for both policies. Confirm actual cut/rebuild events before the probe, including token counts, retained history boundaries, and maintenance latency. Verify that the initial facts and any assistant repetitions have left the basic window. Clip count or elapsed time alone does not prove a cut occurred. Report protocol failures separately from memory errors, and report correctness, information loss, and added latency separately.

This package does not implement the replay driver, compaction, or scoring. The existing `benchmark_duplex.py` CLI consumes a single WAV, has a 230-second limit, and does not consume these playlists. A playlist driver needs sufficient client and server session lifetimes. This checkout includes the basic history window.

## Observe history cuts

With INFO logging enabled, the session bridge emits one `history_cut` line after each successful cut. For example:

```text
history_cut session_id=native session_open_index=7 request_id=next input_seq=42 input_t_start_ms=42000 history_tokens_before=23 history_tokens_after=18 dropped_tokens=5 dropped_token_ranges=((2, 5), (10, 12))
```

The before/after counts include the previous unit's output retained for the next append and exclude the incoming unit. `input_seq` and `input_t_start_ms` identify the incoming audio unit that triggered the cut; they are not playlist question numbers or wall-clock timestamps. The half-open `dropped_token_ranges` use coordinates immediately before that cut, not coordinates in the original conversation. Join the session ID, open index, and input position with the replay record when investigating an answer.

This event confirms that the cut completed; the next append or generation can still fail. Check subsequent responses for continued interaction and judge answers against the fixture separately. No event is emitted when no cut is requested or when the cut raises an error.

## Integrity and evidence

The checksum list covers every file in `audio_v1/` except itself. Verify it before replay:

```bash
cd benchmarks/duplex/context_memory/audio_v1
shasum -a 256 -c SHA256SUMS
```

Clip format, sample bounds, unchanged audio bytes, playlist references, and turn order have been checked. No MiniCPM inference run, compaction result, recall result, or serving-latency measurement is included.

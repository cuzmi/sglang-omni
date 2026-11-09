# MiniCPM context-memory audio fixture

Mandarin user-side audio for comparing MiniCPM's basic history window with a future summary-and-rebuild policy. Revision 2 removes spoken answer-length limits and asks the filler questions to invite explanations and examples. This package contains input materials and an evaluation rubric; it does not implement compaction, a replay driver, or model scoring. Revision 1 remains available in commit `6212c0df9`.

## Review the material

| File | Purpose |
| --- | --- |
| [scenario.json](audio_v1/scenario.json) | Initial facts, an early budget correction, 30 filler prompts, and two alternative probes |
| [rubric.json](audio_v1/rubric.json) | Expected facts, constraint checks, and validity requirements; never send this file to the model |
| [clips/](audio_v1/clips/) | Individual utterances for replay that waits for each answer before sending the next question |
| [long.wav](audio_v1/long.wav) | Listening preview, 43 min 5.723 s: initial facts, 63 filler turns, and the factual-recall probe |
| [short-control.wav](audio_v1/short-control.wav) | Listening preview, 5 min 16.589 s: the identical initial facts followed immediately by the identical probe |
| [long.transcript.txt](audio_v1/long.transcript.txt) | Human-readable long-session script with timestamps |
| [long.timeline.json](audio_v1/long.timeline.json) | Speech, silence, phase, and PCM frame boundaries |
| [manifest.json](audio_v1/manifest.json) | Voice, audio format, measured file durations, hashes, and verification status |
| [使用说明.txt](audio_v1/使用说明.txt) | Chinese usage notes |

The recordings use the installed macOS Tingting voice at a say rate of 180. They contain only synthesized user speech and silence, not recorded assistant responses. The format is mono 16 kHz PCM16 WAV. The merged listening previews insert 30 seconds of silence after each non-final utterance and 60 seconds after the final probe. These gaps do not schedule the benchmark: playback must wait for each actual response. Preview timelines describe the merged WAV files, not the actual input times of a run.

The first 241.2993125 seconds and the final probe PCM are identical between the two previews. The long sequence repeats filler prompts in their listed order for exactly 63 turns; longer silence does not reduce the number of questions. The long preview probe starts at 2,510.433375 seconds. Run different probes in separate sessions to avoid refreshing the facts through an earlier probe's answer.

## Intended comparison

Use the same ordered clips for both A and B: all initial turns, 63 filler turns, then one probe. Stream each clip at real-time pace into one persistent `/v1/realtime` connection per run. During the wait, continue streaming silence so MiniCPM's native input-driven inference can advance. Send the next question only after the current response's `response.done` has `status: completed`; match its response ID, rather than accepting a stale response or a unit/chunk completion. If the test also requires audible playback, wait for the local playback queue to empty. Record a missing response, timeout, cancellation, or failure instead of silently continuing in a new session.

Splitting the audio does not reset history: the native adapter appends units to the same session identity. Do not call `run_session()` separately for each clip, because it opens a new connection. Do not send `sglang.input_audio.end` after each question, because it ends input for the whole session. This sequencing behavior is specified here but is not yet implemented in a replay driver.

The short control uses the same initial clips and probe, without fillers. It checks initial comprehension; it is not the B policy. First confirm the model can answer it, then check whether the long session preserves the early facts, the corrected budget, the no-open-flame constraint, and completed versus pending tasks.

Observe context length and actual cut/rebuild events on the server. Duration or output text length alone does not prove a cut occurred. Verify that all occurrences of the tested facts, including any assistant repetitions, have left the basic window before scoring distant recall. Record protocol failures, overlapping speech, and truncated answers separately from memory errors. Keep the model, speech clips, order, completion rule, sampling settings, and post-maintenance context budget comparable; the summary consumes that budget too. Different response durations can produce different amounts of silence input. Record speech, silence, output, and context token counts separately; this protocol does not guarantee identical total input bytes or compute work between policies.

The current checkout predates the basic-window implementation in PR #2973. Use a deployment that contains it for the baseline. The existing `benchmark_duplex.py` CLI in this checkout caps input duration at 230 seconds, so it cannot replay either revised preview unchanged. A clip replay driver will also need sufficient client and server session lifetimes; reconnecting for each question would invalidate the memory test. This fixture does not modify the CLI or launch a server.

## Integrity and evidence

The audio and metadata in `audio_v1/` describe fixture revision 2. The checksum list covers every file in that directory except itself. Verify it before replay:

```bash
cd benchmarks/duplex/context_memory/audio_v1
shasum -a 256 -c SHA256SUMS
```

WAV format, nonempty unclipped speech samples, timeline-to-clip equality, silence spans, and the shared prefix/probe have been checked. Revision 2 has not had a human listening review or MiniCPM inference run. There are no measured compaction, recall, or serving-latency results in this package.

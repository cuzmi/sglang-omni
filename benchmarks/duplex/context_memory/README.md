# MiniCPM context-memory audio fixture

Frozen Mandarin user-side audio for comparing MiniCPM's basic history window with a future summary-and-rebuild policy. This commit contains input materials and an evaluation rubric; it does not implement compaction, a replay driver, or model scoring.

## Review the material

| File | Purpose |
| --- | --- |
| [scenario.json](audio_v1/scenario.json) | Initial facts, an early budget correction, 30 filler prompts, and two alternative probes |
| [rubric.json](audio_v1/rubric.json) | Expected facts, constraint checks, and validity requirements; never send this file to the model |
| [long.wav](audio_v1/long.wav) | 20 min 50.363 s: initial facts, 63 filler turns, and the factual-recall probe |
| [short-control.wav](audio_v1/short-control.wav) | 3 min 6.863 s: the identical initial facts followed immediately by the identical probe |
| [long.transcript.txt](audio_v1/long.transcript.txt) | Human-readable long-session script with timestamps |
| [long.timeline.json](audio_v1/long.timeline.json) | Speech, silence, phase, and PCM frame boundaries |
| [manifest.json](audio_v1/manifest.json) | Voice, audio format, measured file durations, hashes, and verification status |
| [使用说明.txt](audio_v1/使用说明.txt) | Chinese usage notes |

The recordings use the installed macOS Tingting voice at a say rate of 180. They contain only synthesized user speech and silence, not recorded assistant responses. The format is mono 16 kHz PCM16 WAV. Each non-final utterance is followed by 12 seconds of silence; the final probe is followed by 35 seconds. These fixed gaps do not guarantee that a model has finished answering.

The first 136.573375 seconds and the final probe PCM are identical between the two recordings. Filler prompts repeat in their fixed order until the next probe can start at or after 1,200 seconds from the beginning. In this frozen version, the long probe starts at 1,200.07325 seconds. Individual clips, including an optional applied-planning probe, are retained in [clips/](audio_v1/clips/). Run different probes in separate sessions to avoid refreshing the facts through an earlier probe's answer.

## Intended comparison

Use the same long WAV bytes for both A and B, streamed at real-time pace into one persistent `/v1/realtime` session per run. The short control checks initial comprehension; it is not the B policy. First confirm the model can answer the short control, then check whether the long session preserves the early facts, the corrected budget, the no-open-flame constraint, and completed versus pending tasks.

Observe context length and actual cut/rebuild events on the server. Duration alone does not prove a cut occurred. Verify that all occurrences of the tested facts, including any assistant repetitions, have left the basic window before scoring distant recall. Record protocol failures, overlapping speech, and truncated answers separately from memory errors. Keep the model, input bytes, settings, and post-maintenance context budget comparable; the summary consumes that budget too.

The current checkout predates the basic-window implementation in PR #2973. Use a deployment that contains it for the baseline. The existing `benchmark_duplex.py` CLI in this checkout also caps input duration at 230 seconds, so it cannot replay the long file unchanged. This fixture does not modify that CLI or launch a server.

## Integrity and evidence

The audio and metadata in `audio_v1/` are byte-identical to the generated material package. The checksum list covers every file in that directory except itself. Verify it before replay:

```bash
cd benchmarks/duplex/context_memory/audio_v1
shasum -a 256 -c SHA256SUMS
```

WAV format, nonempty unclipped speech samples, timeline-to-clip equality, silence spans, and the shared prefix/probe have been checked. No human listening review or MiniCPM inference run has been performed. There are no measured compaction, recall, or serving-latency results in this commit.

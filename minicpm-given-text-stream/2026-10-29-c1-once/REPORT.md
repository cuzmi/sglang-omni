# MiniCPM-o C1 streaming single-pair comparison — 2026-10-29

Both modes completed one warmup and one measured English C1 request on the same frozen source and server. This is stream=false versus stream=true, not a comparison between code revisions. All four requests completed and the recorded measured input/codec gates passed: both measured requests generated the same 374 codec tokens.

| Mode | First audio (ms) | Meaningful audio (ms) | Completion (ms) | Audio duration (s) |
| --- | ---: | ---: | ---: | ---: |
| nonstream | 452.85 | 452.85 | 456.01 | 14.96 |
| stream | 768.66 | 768.66 | 1939.26 | 15.24 |

Streaming produced 15 generated audio chunks but reached first audio later in this measured pair. Output durations were 14.96 and 15.24 seconds (1.87% difference). Matching codec tokens and passing the duration screen do not establish waveform or perceptual-quality equivalence.

This run used no profiler. First-audio latency was 452.85 ms nonstream and 768.66 ms stream (+69.74% in this pair); completion was 456.01 ms and 1939.26 ms. These are individual observations, not repeatability or population estimates.

The sequence was warmup/nonstream, warmup/stream, measured/nonstream, measured/stream. There were no repetitions, ABBA ordering or C2 requests. SSE/base64 serialization is included in stream latency. C1 cannot establish multi-request fairness or GPU overlap. No ASR/MOS, speaker similarity, cancellation or fault-injection validation was performed.

The H100 run used pinned MiniCPM-o-4_5 revision 503e754207c94da6bb26850b4469f367c9ea3582 and mono 24 kHz PCM. Candidate base HEAD was 67476823ae81b398626b85e368d1c6f11a832496 with dirty snapshot archive SHA256 b57f7fadaa93f39b7f335fcbf1a908a24d9f2f25c0208becdab57075470c7afc. The full source package and audio remain local.

[Comparison](comparison.json) · [Protocol](protocol.json) · [Request summary](artifacts/stream-c1-once-20261029T183642Z/measurements/summary.json) · [Status](artifacts/stream-c1-once-20261029T183642Z/status.json) · [Resolved configuration](artifacts/stream-c1-once-20261029T183642Z/resolved_config.yaml)

Publication adds this report and validates saved artifacts; it does not run the experiment again.

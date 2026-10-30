# MiniCPM-o C1 streaming event diagnostics — 2026-10-29

Both modes completed one warmup and one measured English C1 request on the same frozen source and server. This is stream=false versus stream=true, not a comparison between code revisions. All four requests completed and the recorded measured input/codec gates passed: both measured requests generated the same 374 codec tokens.

| Mode | First audio (ms) | Meaningful audio (ms) | Completion (ms) | Audio duration (s) |
| --- | ---: | ---: | ---: | ---: |
| nonstream | 483.89 | 483.89 | 488.17 | 14.96 |
| stream | 928.71 | 928.71 | 2287.83 | 15.24 |

Streaming produced 15 generated audio chunks but reached first audio later in this measured pair. Output durations were 14.96 and 15.24 seconds (1.87% difference). Matching codec tokens and passing the duration screen do not establish waveform or perceptual-quality equivalence.

Both modes used the request event recorder. The recorded stream path has first codec availability at 66.37 ms, followed by request initialization of 400.17 ms and first Flow host interval of 410.20 ms. Initialization plus first Flow accounts for 87.26% of the recorded 928.71 ms first-audio time. Later chunk Flow host intervals range from 81.33 to 89.71 ms.

These host function intervals include CPU submission, thread competition and blocking. First Flow overlaps the Talker execution interval, whereas later chunks follow Talker completion; this correlation does not distinguish CPU/GIL contention from GPU contention. No Torch/CUPTI trace or added CUDA synchronization was collected. Do not treat the phase intervals as isolated GPU compute or subtract this run from the separate unprofiled run to estimate recorder overhead.

[Event analysis](artifacts/stream-c1-events-20261029T190650Z/event_analysis.json) contains the per-request spans, timelines and incomplete-span fields. Raw recorder JSONL files are retained in the run events directory.

The sequence was warmup/nonstream, warmup/stream, measured/nonstream, measured/stream. There were no repetitions, ABBA ordering or C2 requests. SSE/base64 serialization is included in stream latency. C1 cannot establish multi-request fairness or GPU overlap. No ASR/MOS, speaker similarity, cancellation or fault-injection validation was performed.

The H100 run used pinned MiniCPM-o-4_5 revision 503e754207c94da6bb26850b4469f367c9ea3582 and mono 24 kHz PCM. Candidate base HEAD was 67476823ae81b398626b85e368d1c6f11a832496 with dirty snapshot archive SHA256 b57f7fadaa93f39b7f335fcbf1a908a24d9f2f25c0208becdab57075470c7afc. The full source package and audio remain local.

[Comparison](comparison.json) · [Protocol](protocol.json) · [Request summary](artifacts/stream-c1-events-20261029T190650Z/measurements/summary.json) · [Status](artifacts/stream-c1-events-20261029T190650Z/status.json) · [Resolved configuration](artifacts/stream-c1-events-20261029T190650Z/resolved_config.yaml)

Publication adds this report and validates saved artifacts; it does not run the experiment again.

# MiniCPM-o given-text streaming on H100 — 2026-10-27

The run completed 8 warmup and 48 measured requests. All 24 ABBA pairs passed the recorded input/codec matching and duration gates. Streaming delivered interleaved generated audio in all 8 concurrent streaming groups, with both requests producing first audio before the peer completed. In this experiment streaming had higher meaningful-audio latency and completion time in every measured scenario/sample median.

This compares stream=false and stream=true on the same candidate snapshot, not an old/new scheduler comparison. One H100 served English and Chinese given-text requests, with C1, simultaneous C2 and 250 ms staggered C2 scenarios. Each row has four measured requests. Meaningful audio is the first received 10 ms PCM window with RMS >= 0.001 full scale; PCM is mono 24 kHz.

| Scenario | Sample | Mode | Meaningful TTFA median (ms) | Completion median (ms) |
| --- | --- | --- | ---: | ---: |
| c1_en | en | nonstream | 562.36 | 564.46 |
| c1_en | en | stream | 901.36 | 2717.18 |
| c1_zh | zh | nonstream | 661.34 | 663.69 |
| c1_zh | zh | stream | 888.32 | 3363.11 |
| c2_simultaneous | en | nonstream | 731.11 | 733.21 |
| c2_simultaneous | en | stream | 1291.63 | 3053.93 |
| c2_simultaneous | zh | nonstream | 833.28 | 835.27 |
| c2_simultaneous | zh | stream | 1290.87 | 3558.38 |
| c2_staggered | en | nonstream | 860.39 | 862.51 |
| c2_staggered | en | stream | 1254.80 | 3211.69 |
| c2_staggered | zh | nonstream | 906.44 | 908.75 |
| c2_staggered | zh | stream | 1005.93 | 3447.75 |

[Full analysis, individual values and ABBA pairs](artifacts/stream-20261027T192929Z/analysis.json) · [Request summary](artifacts/stream-20261027T192929Z/measurements/summary.json) · [Run status](artifacts/stream-20261027T192929Z/status.json) · [Protocol](protocol.json)

The source base is 67476823ae81b398626b85e368d1c6f11a832496 plus the recorded dirty snapshot; source package SHA256 is 00fbf2698c3f0685510f3b39b4f0997249893c1333c4bc309a123042da2c12dd. The model revision is 503e754207c94da6bb26850b4469f367c9ea3582. Environment and resolved configuration are retained beside the run.

Matched codec tokens do not establish identical Flow noise or waveform quality. Client-observed interleaving is not proof of simultaneous GPU execution. This run has no profiler, ASR/MOS/speaker-similarity validation or cancellation/fault testing. Original timings and negative results remain included.

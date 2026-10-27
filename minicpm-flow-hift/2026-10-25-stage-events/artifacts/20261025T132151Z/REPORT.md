# MiniCPM-o request-event stage comparison

## Setup and validation

Run `20261025T132151Z`, Modal workspace `stangameact`, one NVIDIA H100 80GB. Baseline `078d0dbaabc82da3e67fbdda038257cd9a391051`; optimized `758405416635738e8812cbddbe170dd94339f5e6`. Model `openbmb/MiniCPM-o-4_5` revision `503e754207c94da6bb26850b4469f367c9ea3582`. The pinned image, runtime versions, fixed inputs, sampling, native batching/Graph defaults, and engine capacities match the prior RADIX experiment; the host and CPU/memory provisioning differ. Full settings and hashes are in [PROTOCOL.md](https://github.com/cuzmi/RemoteFiles/blob/b141239f34546b0ee8c04f86f4c39e162d0ba3f1/research/minicpm_serving_event256_modal_h100_20261025/PROTOCOL.md) and [protocol.json](https://github.com/cuzmi/RemoteFiles/blob/b141239f34546b0ee8c04f86f4c39e162d0ba3f1/research/minicpm_serving_event256_modal_h100_20261025/protocol.json).

Each C1/C8/C16/C32 cell uses the same 256 measured SeedTTS inputs and 128 disjoint warmup inputs. Two repeats per version, baseline → optimized in the first pass and optimized → baseline in the second. Every visit starts a fresh server and fresh compiler caches. The maintained repository benchmark is reused; response decoding and audio saving happen after the timed cohort. Only the maintained request-event recorder is enabled, after warmup and before formal requests.

All 16 visits completed: 4,096/4,096 formal requests and 2,048/2,048 warmup requests passed. All formal HTTP response IDs match profiler request IDs, with one start/end interval per request for preprocessing, Thinker, Talker, and Code2Wav. There are 163,840 recorded events. The Modal app completed; the final GPU snapshot shows 4 MiB used and 0% utilization. [Modal run](https://modal.com/apps/stangameact/main/ap-uhPxIWLtOH4s1Hl9vpGwjl).

## Pooled stage results

Completed cells: 16/16. Means below are milliseconds per request; two repeats per arm when complete.

Stage residence includes waiting and execution. This run enables only the maintained request event recorder. It does not measure pure GPU execution or isolate resource contention. Modal timing cannot be subtracted from the RADIX run to estimate profiler overhead.

| C | Preprocessing B → O | Thinker B → O | Talker B → O | Code2Wav B → O | E2E B → O |
| --- | ---: | ---: | ---: | ---: | ---: |
| C1 | 1.26 → 1.24 | 124.04 → 124.44 | 97.84 → 97.98 | 75.15 → 75.50 | 308.33 → 310.03 |
| C8 | 2.25 → 1.39 | 240.75 → 244.00 | 313.73 → 317.62 | 289.20 → 225.93 | 862.80 → 801.41 |
| C16 | 3.29 → 1.84 | 263.28 → 273.15 | 468.69 → 570.60 | 465.64 → 311.91 | 1226.35 → 1173.29 |
| C32 | 6.67 → 2.71 | 300.30 → 295.41 | 562.22 → 1148.09 | 981.87 → 412.30 | 1890.81 → 1899.39 |

## Interpretation

The reduction in overall benefit at higher concurrency is accompanied by a shift in the longest stage from Code2Wav to Talker. Code2Wav residence continues to decrease in the optimized version, while Talker residence grows substantially at C16 and C32.

- C8: Code2Wav decreases by 63.27 ms (21.9%); E2E decreases by 61.39 ms (7.1%). Talker changes direction across repeats, so the small pooled increase is not a consistent slowdown at this concurrency.
- C16: Code2Wav decreases by 153.73 ms (33.0%), while Talker increases by 101.91 ms (21.7%). E2E improves by 53.07 ms (4.3%). Both repeats show the same opposing stage changes.
- C32: Code2Wav decreases by 569.57 ms (58.0%), while Talker increases by 585.88 ms (104.2%). E2E increases by 8.58 ms (0.45%), effectively losing the lower-concurrency benefit. The two repeat-level E2E changes are -0.19% and +1.11%.

This C32 Talker shift also appears in the median (521.27 → 1,109.54 ms) and p95 (940.59 → 1,852.55 ms), rather than being limited to a few extreme requests. Percentiles use linear interpolation over the 512 requests per version. Pooled generated audio at C32 is 2,149.68 → 2,148.40 seconds (-0.06%); aggregate output length is nearly unchanged, although individual requests and batch composition can still differ.

These measurements locate the offsetting time in Talker. They do not distinguish Talker queueing from execution, prove GPU resource contention, or measure batch shapes and Graph hit/fallback behavior. The next diagnostic question is what causes the extra Talker residence at C16/C32; contention, scheduling, and changed batching remain possible explanations. No additional instrumentation or experiment was run beyond this first stage-level comparison.

## Individual visits

| Repeat | Variant | C | Thinker ms | Talker ms | Code2Wav ms | E2E ms | Total audio seconds |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | baseline | 1 | 124.34 | 98.35 | 75.62 | 309.92 | 1060.24 |
| 1 | optimized | 1 | 124.33 | 97.53 | 75.27 | 309.40 | 1060.24 |
| 1 | baseline | 8 | 240.87 | 319.38 | 289.44 | 869.57 | 1057.04 |
| 1 | optimized | 8 | 248.03 | 339.19 | 219.53 | 821.01 | 1066.88 |
| 1 | baseline | 16 | 265.98 | 460.01 | 498.05 | 1249.96 | 1058.68 |
| 1 | optimized | 16 | 274.22 | 569.18 | 309.79 | 1170.83 | 1057.20 |
| 1 | baseline | 32 | 304.53 | 568.47 | 991.10 | 1910.35 | 1070.32 |
| 1 | optimized | 32 | 287.01 | 1165.18 | 413.55 | 1906.73 | 1076.48 |
| 2 | baseline | 1 | 123.74 | 97.32 | 74.68 | 306.73 | 1060.24 |
| 2 | optimized | 1 | 124.56 | 98.43 | 75.73 | 310.66 | 1060.24 |
| 2 | baseline | 8 | 240.63 | 308.07 | 288.96 | 856.02 | 1070.76 |
| 2 | optimized | 8 | 239.96 | 296.05 | 232.33 | 781.81 | 1053.92 |
| 2 | baseline | 16 | 260.58 | 477.37 | 433.24 | 1202.74 | 1061.08 |
| 2 | optimized | 16 | 272.08 | 572.02 | 314.03 | 1175.74 | 1065.56 |
| 2 | baseline | 32 | 296.07 | 555.96 | 972.65 | 1871.28 | 1079.36 |
| 2 | optimized | 32 | 303.82 | 1131.01 | 411.05 | 1892.06 | 1071.92 |

## Change in each repeat

Negative latency changes mean shorter residence or latency. Audio seconds measure generated work and are not a latency metric.

| Repeat | C | Thinker change | Talker change | Code2Wav change | E2E change | Audio seconds change |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | C1 | -0.01% | -0.84% | -0.46% | -0.17% | +0.00% |
| 1 | C8 | +2.97% | +6.20% | -24.15% | -5.59% | +0.93% |
| 1 | C16 | +3.10% | +23.73% | -37.80% | -6.33% | -0.14% |
| 1 | C32 | -5.75% | +104.97% | -58.27% | -0.19% | +0.58% |
| 2 | C1 | +0.66% | +1.14% | +1.40% | +1.28% | +0.00% |
| 2 | C8 | -0.28% | -3.90% | -19.60% | -8.67% | -1.57% |
| 2 | C16 | +4.41% | +19.83% | -27.51% | -2.25% | +0.42% |
| 2 | C32 | +2.62% | +103.43% | -57.74% | +1.11% | -0.69% |

## Evidence

- [Per-visit stage means and output work](stage_cells.csv)
- [Pooled means, p50, p95 and relative changes](stage_comparison.json)
- [Visit completion status](results/status.json)
- Raw events, joined per-request stages, maintained profiler reports, request/audio outputs, and server/client logs are retained under `results/<visit>/`.
- [Runtime environment](results/environment.json) and [final GPU state](gpu_final.csv).

All 13,061 downloaded files passed remote-size checks. All 384 reference hashes match the frozen input manifest, and every reported stage interval was independently recomputed from the 48 raw event files. See [validation.json](validation.json) and [SHA256SUMS.json](https://github.com/cuzmi/RemoteFiles/blob/b141239f34546b0ee8c04f86f4c39e162d0ba3f1/research/minicpm_serving_event256_modal_h100_20261025/artifacts/20261025T132151Z/SHA256SUMS.json).

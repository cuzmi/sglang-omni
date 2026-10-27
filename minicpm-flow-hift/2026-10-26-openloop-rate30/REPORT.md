# MiniCPM-o open-loop serving comparison

Baseline `078d0dba` versus optimized `75840541` on one H100. Each visit uses the same 256 English SeedTTS samples and 128 disjoint warmup samples. Formal arrivals use seed 20261026, rates 30 requests/s, and no client concurrency cap; warmup uses C32. Each rate runs baseline/optimized/optimized/baseline with fresh servers and compiler caches. No profiler is enabled.

## Validation

4 of 4 cells are complete and valid. Saved WAV headers, request identities, fixed schedules, seed/rate settings, and client-slot waits are checked. See validation.json for incomplete or invalid cells. Invalid cells are excluded in full; a pooled rate comparison requires all four ABBA cells.

## Pooled complete ABBA results

Each arm contains 512 outputs from 256 unique inputs repeated twice. Percentiles use the pooled request latencies, and rates use total work divided by total cohort wall time.

| Offered req/s | Arm | Mean ms | p50 ms | p95 ms | Mean RTF | Completed QPS | Audio s/s | Total audio s |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 30 | baseline | 1930.9 | 1979.5 | 2731.5 | 0.482 | 25.031 | 103.947 | 2126.20 |
| 30 | optimized | 2195.3 | 2286.8 | 3230.1 | 0.544 | 23.370 | 96.836 | 2121.48 |

## Repetition and arrival diagnostics

| Cell | Mean ms | p95 ms | Lateness p99 ms | Lateness max ms | Wall s | Drain s | Peak in flight | Audio s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| r1-baseline-rate30 | 1964.8 | 2748.8 | 2.031 | 6.694 | 10.265 | 1.725 | 86 | 1060.76 |
| r1-optimized-rate30 | 2210.4 | 3277.6 | 2.035 | 2.200 | 10.684 | 2.144 | 85 | 1065.52 |
| r2-optimized-rate30 | 2180.3 | 3175.6 | 2.057 | 15.694 | 11.224 | 2.685 | 87 | 1055.96 |
| r2-baseline-rate30 | 1896.9 | 2612.8 | 4.445 | 14.031 | 10.190 | 1.650 | 79 | 1065.44 |

## Paired changes

Positive latency reduction means lower optimized latency. Two repetitions expose variability but do not establish a confidence interval.

| Offered req/s | Pooled mean reduction | Repeat 1 | Repeat 2 | Pooled p95 reduction | Total audio change | Changed lengths / 512 |
|---:|---:|---:|---:|---:|---:|---:|
| 30 | -13.7% | -12.5% | -14.9% | -18.3% | -0.2% | 466/512 |

## 30 requests/s completion tail

Mean latency reductions are -12.50% and -14.94%; p95 reductions are -19.24% and -21.54%. Negative reductions mean regressions.

| Cell | Last-finishing sample | Audio s | Same sample in other arm, audio s | Completion offset s | Gap after preceding completion s |
|---|---|---:|---:|---:|---:|
| r1-baseline-rate30 | common_voice_en_19496235-common_voice_en_19496229 | 6.88 | 5.60 | 10.264 | 0.001 |
| r1-optimized-rate30 | common_voice_en_19496235-common_voice_en_19496266 | 5.72 | 5.08 | 10.683 | 0.006 |
| r2-baseline-rate30 | common_voice_en_19449066-common_voice_en_19449075 | 6.60 | 6.40 | 10.190 | 0.001 |
| r2-optimized-rate30 | common_voice_en_19496235-common_voice_en_19496229 | 5.44 | 4.96 | 11.224 | 0.008 |

All outputs remain included. These observations identify which requests set the completion tail and therefore the cohort wall time; they do not isolate the effects of their work on other concurrent requests.

## Measurement scope

At a fixed offered rate, completed QPS is constrained by arrivals and the finite drain tail; it is not a sustainable-capacity estimate. The 256-request sending windows are 30 req/s: 8.54 s. Mean RTF is the mean of each request's latency divided by its generated audio duration.

Latency starts at actual dispatch. Planned-arrival-to-completion also includes recorded dispatch lateness. In-flight and drain figures are reconstructed from planned offsets, dispatch lateness, and measured latency, with small client clock-boundary overhead; they do not measure a server queue or GPU occupancy.

The same scheduled inputs remove completion-driven client refill feedback. Internal batching, graph behavior, and generated audio lengths can still differ. This is a comparison of the full serving change, not an isolated overlap speedup. No WER or speaker-similarity result is inferred from these timing measurements.

# MiniCPM-o open-loop serving comparison

Baseline `078d0dba` versus optimized `75840541` on one H100. Each visit uses the same 256 English SeedTTS samples and 128 disjoint warmup samples. Formal arrivals use seed 20261026, rates 8/14/20 requests/s, and no client concurrency cap; warmup uses C32. Each rate runs baseline/optimized/optimized/baseline with fresh servers and compiler caches. No profiler is enabled.

## Validation

12 of 12 cells are complete and valid. Saved WAV headers, request identities, fixed schedules, seed/rate settings, and client-slot waits are checked. See validation.json for incomplete or invalid cells. Invalid cells are excluded in full; a pooled rate comparison requires all four ABBA cells.

## Pooled complete ABBA results

Each arm contains 512 outputs from 256 unique inputs repeated twice. Percentiles use the pooled request latencies, and rates use total work divided by total cohort wall time.

| Offered req/s | Arm | Mean ms | p50 ms | p95 ms | Mean RTF | Completed QPS | Audio s/s | Total audio s |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 8 | baseline | 652.2 | 644.7 | 978.1 | 0.161 | 7.892 | 32.665 | 2119.20 |
| 8 | optimized | 627.2 | 625.4 | 935.0 | 0.155 | 7.888 | 32.718 | 2123.80 |
| 14 | baseline | 882.4 | 872.1 | 1266.9 | 0.219 | 13.539 | 56.037 | 2119.20 |
| 14 | optimized | 852.5 | 851.9 | 1245.9 | 0.212 | 13.513 | 55.897 | 2117.84 |
| 20 | baseline | 1203.4 | 1180.9 | 1754.0 | 0.296 | 18.583 | 79.197 | 2182.00 |
| 20 | optimized | 1169.0 | 1148.3 | 1667.8 | 0.284 | 17.929 | 77.794 | 2221.52 |

## Repetition and arrival diagnostics

| Cell | Mean ms | p95 ms | Lateness p99 ms | Lateness max ms | Wall s | Drain s | Peak in flight | Audio s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| r1-baseline-rate8 | 651.5 | 974.1 | 2.067 | 2.400 | 32.443 | 0.420 | 13 | 1060.16 |
| r1-optimized-rate8 | 633.0 | 951.2 | 2.086 | 2.102 | 32.467 | 0.444 | 14 | 1064.08 |
| r2-optimized-rate8 | 621.4 | 913.2 | 2.068 | 2.100 | 32.445 | 0.422 | 14 | 1059.72 |
| r2-baseline-rate8 | 652.8 | 980.9 | 1.992 | 2.101 | 32.434 | 0.411 | 14 | 1059.04 |
| r1-baseline-rate14 | 874.3 | 1253.9 | 2.043 | 2.111 | 18.914 | 0.615 | 23 | 1058.52 |
| r1-optimized-rate14 | 845.8 | 1245.2 | 2.106 | 2.176 | 18.956 | 0.657 | 22 | 1049.40 |
| r2-optimized-rate14 | 859.2 | 1248.2 | 1.953 | 2.053 | 18.932 | 0.633 | 21 | 1068.44 |
| r2-baseline-rate14 | 890.5 | 1280.5 | 2.038 | 2.099 | 18.904 | 0.605 | 24 | 1060.68 |
| r1-baseline-rate20 | 1222.7 | 1859.0 | 2.079 | 2.101 | 13.708 | 0.897 | 39 | 1078.36 |
| r1-optimized-rate20 | 1193.7 | 1671.8 | 2.052 | 2.062 | 14.377 | 1.566 | 35 | 1114.84 |
| r2-optimized-rate20 | 1144.3 | 1651.9 | 1.992 | 2.065 | 14.180 | 1.370 | 34 | 1106.68 |
| r2-baseline-rate20 | 1184.1 | 1619.6 | 2.045 | 2.096 | 13.844 | 1.034 | 37 | 1103.64 |

## Paired changes

Positive latency reduction means lower optimized latency. Two repetitions expose variability but do not establish a confidence interval.

| Offered req/s | Pooled mean reduction | Repeat 1 | Repeat 2 | Pooled p95 reduction | Total audio change | Changed lengths / 512 |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | +3.8% | +2.8% | +4.8% | +4.4% | +0.2% | 282/512 |
| 14 | +3.4% | +3.3% | +3.5% | +1.7% | -0.1% | 348/512 |
| 20 | +2.9% | +2.4% | +3.4% | +4.9% | +1.8% | 409/512 |

## 20 requests/s completion tail

Mean latency improves in both repetitions (2.37% and 3.36%). The p95 changes differ: reductions of +10.07% and -1.99% (negative means a regression).

| Cell | Last-finishing sample | Audio s | Same sample in other arm, audio s | Completion offset s | Gap after preceding completion s |
|---|---|---:|---:|---:|---:|
| r1-baseline-rate20 | common_voice_en_19527999-common_voice_en_19528002 | 4.72 | 4.72 | 13.707 | 0.001 |
| r1-optimized-rate20 | common_voice_en_19277026-common_voice_en_19277029 | 32.88 | 5.24 | 14.376 | 0.605 |
| r2-baseline-rate20 | common_voice_en_19277026-common_voice_en_19277029 | 18.00 | 5.44 | 13.844 | 0.098 |
| r2-optimized-rate20 | common_voice_en_19083094-common_voice_en_19083091 | 31.72 | 6.44 | 14.180 | 0.427 |

Both optimized visits finish on long generated outputs. Their longer final drain lowers whole-cohort QPS even though average request latency improves. All outputs remain included. These observations identify which requests set the completion tail; they do not isolate the effects of their work on other concurrent requests.

## Measurement scope

At a fixed offered rate, completed QPS is constrained by arrivals and the finite drain tail; it is not a sustainable-capacity estimate. The 256-request sending windows are approximately 32.02, 18.30, and 12.81 seconds. Mean RTF is the mean of each request's latency divided by its generated audio duration.

Latency starts at actual dispatch. Planned-arrival-to-completion also includes recorded dispatch lateness. In-flight and drain figures are reconstructed from planned offsets, dispatch lateness, and measured latency, with small client clock-boundary overhead; they do not measure a server queue or GPU occupancy.

The same scheduled inputs remove completion-driven client refill feedback. Internal batching, graph behavior, and generated audio lengths can still differ. This is a comparison of the full serving change, not an isolated overlap speedup. No WER or speaker-similarity result is inferred from these timing measurements.

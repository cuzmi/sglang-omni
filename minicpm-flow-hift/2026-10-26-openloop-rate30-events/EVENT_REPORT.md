# MiniCPM-o 30 requests/s event-only comparison

Baseline `078d0dba` and optimized `75840541`; one H100, 256 fixed formal inputs, 128 disjoint warmup inputs, fixed Poisson arrivals at 30 requests/s, no client concurrency cap, fresh server per arm. Only maintained request events are enabled during the formal cohort. This diagnostic has one baseline/optimized pair.

Validated current visits: 2/2. Pending: none.

## Stage residence and Talker split

All values are mean milliseconds per request. Before prefill is stage input to first-forward start; after prefill is first-forward end to stage completion, including subsequent generation and intervening waits. These are host elapsed times, not pure GPU execution.

| Visit | E2E | Thinker | Talker | Code2Wav | Talker before prefill | Talker first forward | Talker after prefill | Audio s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| r1-baseline-rate30 | 1963.26 | 320.86 | 891.17 | 731.76 | 72.41 | 13.62 | 805.15 | 1074.72 |
| r1-optimized-rate30 | 2137.03 | 318.36 | 1507.29 | 293.52 | 446.38 | 19.76 | 1041.15 | 1075.76 |

## First-forward count versus duration

Overlapping per-request first-forward intervals are merged within the single Talker process. Each merged interval is counted once, avoiding multiplication of batch elapsed time by its request count. Requests per interval means newly admitted requests, not the active decode batch size. Percentiles use linear interpolation across merged intervals.

| Workload / visit | Intervals | Requests / interval | Mean ms | p50 ms | p95 ms | Host interval union ms |
|---|---:|---:|---:|---:|---:|---:|
| r1-baseline-rate30 | 221 | 1.158 | 13.300 | 12.242 | 22.297 | 2939.331 |
| r1-optimized-rate30 | 213 | 1.202 | 18.959 | 12.545 | 52.370 | 4038.366 |
| r1-baseline-c32 | 146 | 1.753 | 19.395 | 17.358 | 42.758 | 2831.716 |
| r1-optimized-c32 | 195 | 1.313 | 32.564 | 24.395 | 81.718 | 6350.066 |
| r2-baseline-c32 | 147 | 1.741 | 18.679 | 17.150 | 37.867 | 2745.826 |
| r2-optimized-c32 | 199 | 1.286 | 31.388 | 27.185 | 75.484 | 6246.248 |

The recorder also reports full forward batch size. The following cross-check keeps that field separate from the number of newly admitted requests in each merged interval.

| Visit | Mean reported batch size | Missing request metadata | Intervals with inconsistent batch metadata | Intervals whose new-request count differs from batch size | Extend-in-batch requests |
|---|---:|---:|---:|---:|---:|
| r1-baseline-rate30 | 1.158 | 0 | 0 | 0 | 0 |
| r1-optimized-rate30 | 1.202 | 0 | 0 | 0 | 0 |
| r1-baseline-c32 | 1.753 | 0 | 0 | 0 | 0 |
| r1-optimized-c32 | 1.313 | 0 | 0 | 0 | 0 |
| r2-baseline-c32 | 1.741 | 0 | 0 | 0 | 0 |
| r2-optimized-c32 | 1.286 | 0 | 0 | 0 | 0 |

For this open-loop pair, interval count changes by -3.62%, mean interval duration by +42.55%, and their union by +37.39%. Union equals interval count multiplied by mean interval duration. This separates the two observed factors arithmetically; it does not assign a causal explanation to longer intervals.

Mean Talker residence changes by +616.12 ms per request: stage entry to first-forward start changes by +373.97 ms, and first-forward end to stage completion by +236.00 ms. The remaining difference is the request-weighted first-forward interval. The interval before first forward includes input preparation and scheduling; it is not a pure queue-wait measurement.

This pair does not reproduce the older closed-loop increase in first-forward count. More prefill calls therefore cannot explain this pair's additional Talker time. The longer per-interval tail and increased time before and after the first forward remain observations requiring explanation.

## Closed-loop C32 event-only reference

The four older C32 visits are recomputed from their raw events using exactly the same definitions. Each has 256 formal requests. Their fixed concurrency differs from the new fixed arrival rate, so compare directions rather than subtracting absolute timings across experiments. Torch captures are excluded from these tables.

| Visit | Mean Talker ms | Before prefill ms | First forward ms, request-weighted | After prefill ms |
|---|---:|---:|---:|---:|
| r1-baseline-c32 | 568.47 | 25.18 | 18.04 | 525.24 |
| r1-optimized-c32 | 1165.18 | 28.94 | 31.27 | 1104.96 |
| r2-baseline-c32 | 555.96 | 25.80 | 17.69 | 512.47 |
| r2-optimized-c32 | 1131.01 | 28.61 | 29.26 | 1073.15 |

## Evidence boundary

HTTP response IDs join exactly to event IDs; warmup IDs are excluded; every formal request has one ordered stage pair and one Talker first-forward pair. The new arms share the precomputed arrival schedule. Raw intervals, request timing splits, checks, and event-file hashes are retained in event_analysis.json and event_validation.json.

Fixed external arrivals remove completion-driven client refill feedback. They do not fix internal stage arrivals, batch shapes, graph behavior, or generated audio lengths. Stage events cannot identify GPU saturation, stream contention, or the cause of slower individual prefill intervals. The repeated unprofiled runs remain the performance evidence; this single profiled pair is a diagnostic observation.

# MiniCPM-o stage comparison with request-event profiling

Baseline `90923df` versus optimized `917821a`; one H100 80GB, one visit per arm at C1/C8/C16/C32. All 256 measured and 256 warmup requests succeeded. Every measured request matched complete, unique stage boundaries. Only the measured phase enabled the existing request-event recorder; Torch/CUPTI, Nsight, NVTX and SM profiling remained disabled.

## Setup and measurement

Same source snapshots, pinned model revision `503e754207c94da6bb26850b4469f367c9ea3582`, dataset revision `27f4c1adee83b5b29b7c4b375f6b976324bda308`, dependencies and fixed samples as unprofiled run `20261022T132646Z`. Each fresh server receives 32 separate warmup samples before the same 32 evaluation samples. Warmup/evaluation IDs, target texts and reference bytes are disjoint. Evaluation references are not explicitly prewarmed. Ordinary speech defaults retain compilation and all 155 Flow graph shapes. Thinker temperature=0.7, seed=20261019, max_tokens=256; Talker defaults. Closed-loop arrival; C1 B/O, C8 O/B, C16 B/O, C32 O/B.

Stage residence is `stage_complete - stage_input_received`, including queueing, reference preparation where applicable, execution and completion handling within those boundaries. It is not GPU compute time. The text-decode stage is a branch; summing all stage intervals does not reconstruct E2E. No new fine-grained instrumentation was added.

## Mean stage residence (ms)

| Concurrency | Arm | E2E | Preprocessing | Thinker | Talker | Code2Wav | Text decode |
|---|---|---:|---:|---:|---:|---:|---:|
| C1 | baseline | 394.28 | 1.69 | 136.14 | 118.09 | 121.07 | 1.51 |
| C1 | optimized | 396.85 | 1.78 | 136.15 | 116.98 | 124.53 | 1.52 |
| C8 | baseline | 1134.96 | 5.08 | 237.64 | 318.76 | 536.26 | 1.65 |
| C8 | optimized | 996.47 | 2.54 | 258.37 | 322.09 | 389.78 | 1.63 |
| C16 | baseline | 1672.09 | 8.56 | 298.61 | 388.78 | 909.57 | 1.57 |
| C16 | optimized | 1769.76 | 10.03 | 300.11 | 645.66 | 769.74 | 2.52 |
| C32 | baseline | 2630.31 | 18.00 | 314.05 | 630.06 | 1568.22 | 2.12 |
| C32 | optimized | 2254.69 | 18.32 | 334.71 | 719.30 | 1111.07 | 2.29 |

## Stage changes (optimized relative to baseline)

| Concurrency | Stage | Baseline mean ms | Optimized mean ms | Change ms | Change % |
|---|---|---:|---:|---:|---:|
| C1 | preprocessing | 1.69 | 1.78 | +0.10 | +5.7% |
| C1 | thinker | 136.14 | 136.15 | +0.01 | +0.0% |
| C1 | talker | 118.09 | 116.98 | -1.11 | -0.9% |
| C1 | code2wav | 121.07 | 124.53 | +3.46 | +2.9% |
| C8 | preprocessing | 5.08 | 2.54 | -2.54 | -50.0% |
| C8 | thinker | 237.64 | 258.37 | +20.73 | +8.7% |
| C8 | talker | 318.76 | 322.09 | +3.33 | +1.0% |
| C8 | code2wav | 536.26 | 389.78 | -146.48 | -27.3% |
| C16 | preprocessing | 8.56 | 10.03 | +1.47 | +17.2% |
| C16 | thinker | 298.61 | 300.11 | +1.50 | +0.5% |
| C16 | talker | 388.78 | 645.66 | +256.87 | +66.1% |
| C16 | code2wav | 909.57 | 769.74 | -139.83 | -15.4% |
| C32 | preprocessing | 18.00 | 18.32 | +0.32 | +1.8% |
| C32 | thinker | 314.05 | 334.71 | +20.66 | +6.6% |
| C32 | talker | 630.06 | 719.30 | +89.25 | +14.2% |
| C32 | code2wav | 1568.22 | 1111.07 | -457.15 | -29.2% |

## Output-work differences

| Concurrency | Equal audio lengths | Baseline total audio s | Optimized total audio s | Total change | Largest request difference s |
|---|---:|---:|---:|---:|---:|
| C1 | 32/32 | 127.88 | 127.88 | +0.0% | 0.00 |
| C8 | 12/32 | 127.68 | 125.20 | -1.9% | 2.16 |
| C16 | 11/32 | 123.84 | 124.28 | +0.4% | 0.80 |
| C32 | 21/32 | 129.16 | 127.64 | -1.2% | 0.88 |

Audio frames/duration and response completion tokens are retained per sample. Completion tokens are not asserted to be codec-token counts. Matching output lengths do not imply matching codec contents or batch/Graph execution. Different outputs affect processing work and waiting for other requests.

## Runtime allocations

Resolved service configuration was identical. Automatic KV capacities are shown below because available memory and startup ordering can vary.

| Concurrency | Baseline Thinker KV tokens | Optimized Thinker KV tokens |
|---|---:|---:|
| C1 | 290410 | 290410 |
| C8 | 290410 | 290410 |
| C16 | 290410 | 290410 |
| C32 | 290410 | 280502 |

## Scope

This is a separate diagnostic run, not a controlled profiler-overhead experiment. Use the unprofiled comparison as the serving performance reference. Each condition has one 32-request evaluation; C32 is a finite burst, not sustained capacity. Event timings do not separate GPU execution from queueing, and do not establish resource contention or Graph hit rates. Changed batching, output work, initialization state and run-to-run variability remain possible contributors.

stage_summary_ms.csv includes mean/p50/p95 for all observed stages. joined_requests_all.csv includes each request's original sample ID, output length, stage intervals and raw boundary timestamps. Raw events and profiler start/stop responses remain under each measured/ directory. serving_summary.csv retains client latency, QPS, audio throughput and RTF from this diagnostic run.

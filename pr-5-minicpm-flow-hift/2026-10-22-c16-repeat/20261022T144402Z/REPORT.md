# MiniCPM-o stage comparison with request-event profiling

Baseline `90923df` versus optimized `917821a`; one H100 80GB, one visit per arm at C16. All 64 measured and 64 warmup requests succeeded. Every measured request matched complete, unique stage boundaries. Only the measured phase enabled the existing request-event recorder; Torch/CUPTI, Nsight, NVTX and SM profiling remained disabled.

## Setup and measurement

Same source snapshots, pinned model revision `503e754207c94da6bb26850b4469f367c9ea3582`, dataset revision `27f4c1adee83b5b29b7c4b375f6b976324bda308`, dependencies and fixed samples as unprofiled run `20261022T132646Z`. Each fresh server receives 32 separate warmup samples before the same 32 evaluation samples. Warmup/evaluation IDs, target texts and reference bytes are disjoint. Evaluation references are not explicitly prewarmed. Ordinary speech defaults retain compilation and all 155 Flow graph shapes. Thinker temperature=0.7, seed=20261019, max_tokens=256; Talker defaults. Closed-loop arrival; C16 baseline then optimized, matching the preceding C16 run.

Stage residence is `stage_complete - stage_input_received`, including queueing, reference preparation where applicable, execution and completion handling within those boundaries. It is not GPU compute time. The text-decode stage is a branch; summing all stage intervals does not reconstruct E2E. No new fine-grained instrumentation was added.

## Mean stage residence (ms)

| Concurrency | Arm | E2E | Preprocessing | Thinker | Talker | Code2Wav | Text decode |
|---|---|---:|---:|---:|---:|---:|---:|
| C16 | baseline | 1912.93 | 7.34 | 312.20 | 486.10 | 1044.21 | 1.67 |
| C16 | optimized | 1551.58 | 5.05 | 262.96 | 586.00 | 661.18 | 1.57 |

## Stage changes (optimized relative to baseline)

| Concurrency | Stage | Baseline mean ms | Optimized mean ms | Change ms | Change % |
|---|---|---:|---:|---:|---:|
| C16 | preprocessing | 7.34 | 5.05 | -2.29 | -31.1% |
| C16 | thinker | 312.20 | 262.96 | -49.24 | -15.8% |
| C16 | talker | 486.10 | 586.00 | +99.91 | +20.6% |
| C16 | code2wav | 1044.21 | 661.18 | -383.03 | -36.7% |

## Output-work differences

| Concurrency | Equal audio lengths | Baseline total audio s | Optimized total audio s | Total change | Largest request difference s |
|---|---:|---:|---:|---:|---:|
| C16 | 13/32 | 126.92 | 124.84 | -1.6% | 1.44 |

Audio frames/duration and response completion tokens are retained per sample. Completion tokens are not asserted to be codec-token counts. Matching output lengths do not imply matching codec contents or batch/Graph execution. Different outputs affect processing work and waiting for other requests.

## Runtime allocations

Resolved service configuration was identical. Automatic KV capacities are shown below because available memory and startup ordering can vary.

| Concurrency | Baseline Thinker KV tokens | Optimized Thinker KV tokens |
|---|---:|---:|
| C16 | 290410 | 280502 |

## Scope

This is a separate diagnostic run, not a controlled profiler-overhead experiment. Use the unprofiled comparison as the serving performance reference. Each condition has one 32-request evaluation; This short closed-loop run does not establish sustained capacity. Event timings do not separate GPU execution from queueing, and do not establish resource contention or Graph hit rates. Changed batching, output work, initialization state and run-to-run variability remain possible contributors.

stage_summary_ms.csv includes mean/p50/p95 for all observed stages. joined_requests_all.csv includes each request's original sample ID, output length, stage intervals and raw boundary timestamps. Raw events and profiler start/stop responses remain under each measured/ directory. serving_summary.csv retains client latency, QPS, audio throughput and RTF from this diagnostic run.

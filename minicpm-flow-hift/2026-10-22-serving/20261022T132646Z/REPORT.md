# MiniCPM-o full-serving comparison without profiling

Baseline `90923df` versus optimized `917821a`; one H100 80GB, one visit per arm and concurrency. All 256 measured and 256 warmup requests succeeded with finite, nonempty 24kHz audio. This measures the combined serving effect of the candidate, including changes to batching and scheduling.

## Setup

Same pinned model revision `503e754207c94da6bb26850b4469f367c9ea3582`, dataset revision `27f4c1adee83b5b29b7c4b375f6b976324bda308`, 32 fixed English evaluation samples, and separate 32 warmup samples. IDs, target texts and reference bytes do not overlap between warmup and evaluation. Each server starts fresh, captures 155 default Flow CUDA Graph shapes, and uses a fresh compiler-cache directory. Resolved configuration is identical between arms; no memory-fraction override. Actual KV allocations and startup details are retained in runtime_summary.json.

Closed-loop C1/C8/C16/C32. Sampling matches the October 19 baseline: Thinker temperature=0.7, seed=20261019, max_tokens=256; Talker defaults. Ordinary nonstreaming speech with DiT/HiFT compilation and Flow CUDA Graph enabled. No request-event, Torch/CUPTI, NVTX/Nsight or SM profiling. Both arms run on the same GPU in one container: C1 B/O, C8 O/B, C16 B/O, C32 O/B.

## Serving results

| Concurrency | Arm | QPS | Audio s/s | E2E mean ms | p50 ms | p95 ms | Mean RTF | Total audio s |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| C1 | baseline | 2.022 | 8.080 | 485.04 | 483.15 | 703.49 | 0.125 | 127.88 |
| C1 | optimized | 1.972 | 7.879 | 491.82 | 472.22 | 773.54 | 0.128 | 127.88 |
| C8 | baseline | 7.337 | 28.320 | 984.63 | 1042.07 | 1369.04 | 0.262 | 123.52 |
| C8 | optimized | 7.613 | 30.653 | 976.09 | 979.71 | 1325.16 | 0.251 | 128.84 |
| C16 | baseline | 10.438 | 41.020 | 1442.31 | 1280.68 | 1823.01 | 0.387 | 125.76 |
| C16 | optimized | 10.827 | 43.160 | 1313.30 | 1274.86 | 1789.86 | 0.339 | 127.56 |
| C32 | baseline | 10.098 | 39.621 | 2449.86 | 1928.24 | 3150.64 | 0.635 | 125.56 |
| C32 | optimized | 10.586 | 41.535 | 2155.62 | 2174.35 | 2908.32 | 0.560 | 125.56 |

## Paired changes (optimized relative to baseline)

| Concurrency | QPS change | Mean latency change | p95 change | Total audio change | Equal audio lengths | Requests with >10% length change |
|---|---:|---:|---:|---:|---:|---:|
| C1 | -2.5% | +1.4% | +10.0% | +0.0% | 32/32 | 0/32 |
| C8 | +3.8% | -0.9% | -3.2% | +4.3% | 11/32 | 8/32 |
| C16 | +3.7% | -8.9% | -1.8% | +1.4% | 17/32 | 3/32 |
| C32 | +4.8% | -12.0% | -7.7% | +0.0% | 32/32 | 0/32 |

## Largest per-request audio-length differences

| Concurrency | Sample | Baseline s | Optimized s | Difference |
|---|---|---:|---:|---:|
| C8 | common_voice_en_15735839-common_voice_en_15735837 | 2.48 | 3.72 | +50.0% |
| C8 | common_voice_en_125386-common_voice_en_125389 | 3.72 | 4.80 | +29.0% |
| C8 | common_voice_en_15903802-common_voice_en_15903815 | 3.28 | 4.24 | +29.3% |
| C16 | common_voice_en_15903802-common_voice_en_15903815 | 3.52 | 4.48 | +27.3% |
| C16 | common_voice_en_1416089-common_voice_en_1416093 | 3.84 | 4.68 | +21.9% |
| C8 | common_voice_en_153872-common_voice_en_153874 | 5.00 | 5.68 | +13.6% |
| C8 | common_voice_en_153872-common_voice_en_153873 | 2.80 | 3.40 | +21.4% |
| C8 | common_voice_en_15903802-common_voice_en_15903807 | 4.44 | 3.92 | -11.7% |
| C8 | common_voice_en_1416089-common_voice_en_1416090 | 4.08 | 4.56 | +11.8% |
| C8 | common_voice_en_155313-common_voice_en_155315 | 5.92 | 5.56 | -6.1% |

## Observed runtime differences

Resolved configuration was identical, but the default automatic KV budget differed in C1 and C8 because available memory at Thinker startup differed. These cells are not strictly matched for KV capacity. C16 and C32 had matching allocations. No capacity constraint is inferred from this difference alone.

| Concurrency | Baseline Thinker KV tokens | Optimized Thinker KV tokens |
|---|---:|---:|
| C1 | 290410 | 280502 |
| C8 | 280502 | 290410 |
| C16 | 290410 | 290410 |
| C32 | 290410 | 290410 |

Talker KV capacity was 131,072 tokens in every cell. Each C1 and C32 request pair had the same audio length; C8 and C16 had output-length drift. All paired response completion-token counts matched, with 475 total per cell. Matching lengths do not establish identical codec-token contents or audio quality.

C32 showed higher throughput and lower mean/p95 latency, while p50 increased from 1928.24 to 2174.35 ms. The observed improvement is not uniform across the request distribution.

## Interpretation and limits

QPS is 32 divided by batch wall time; audio throughput uses total output audio seconds over that same wall time. The maintained benchmark wall time includes arrival, drain and client WAV writes. E2E uses engine_time_s, which measures client request/response processing, not isolated engine execution, and excludes WAV writing. RTF is request time divided by audio duration.

This is one small-sample screening per condition, without repeated trials or A/A noise estimates. C32 is a finite 32-request burst; p95 is not a stable tail estimate. Default compilation may encounter unseen shapes despite warmup. Evaluation references were not explicitly prewarmed.

Identical requests and sampling seeds do not ensure identical generated work. Per-request audio frames, audio duration and response completion tokens are saved in output_lengths_paired.csv. Response completion tokens are not codec token counts. Audio throughput helps describe different output amounts but does not remove batching, padding or output-distribution confounding. No result here isolates overlap from batching or graph effects, and no quality-equivalence evaluation was performed.

Raw per-request results, WAV outputs, commands, standard logs, input hashes, runtime allocations and source revisions are retained alongside this report.

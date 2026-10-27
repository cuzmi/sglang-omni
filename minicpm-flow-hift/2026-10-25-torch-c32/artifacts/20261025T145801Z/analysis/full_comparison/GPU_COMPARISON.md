# C32 baseline vs optimized: full GPU trace comparison

This analysis covers all four original traces, not only the earlier two-second viewing windows. Each arm serves the same 256 formal inputs after 128 separate warmup inputs, on one H100. Baseline: `078d0db`; optimized: `7584054`. CPU/CUDA Torch profiling and request-event recording were enabled. The preceding unprofiled run remains the performance reference.

## Main finding

The Talker regression is concentrated after its first model forward. Its principal forward CUDA Graph execution spans are not uniformly longer; their submission cadence, graph-family distribution, and the work performed by the other stages change substantially. GPU resource competition remains possible, but these traces do not establish that SM saturation or uniformly slower kernels cause the approximately doubled Talker residence time.

## 1. Locate the additional Talker time

Mean milliseconds per request, using the 256 complete request-event sequences in each arm:

| Interval | Baseline | Optimized |
| --- | ---: | ---: |
| Stage input to request-build start | 41.14 | 27.70 |
| Request construction | 3.41 | 4.30 |
| Build end to scheduler queue entry | 0.38 | 0.39 |
| Scheduler queue entry to first forward start | 10.88 | 6.65 |
| First forward start to return | 31.96 | 43.52 |
| First forward return to model-path completion | 730.10 | 1541.13 |
| Model-path completion to stage completion | 1.39 | 1.50 |
| Total Talker residence | 819.26 | 1625.19 |

The first-forward markers come from `emit_prefill_start_for_batch` and `emit_prefill_end_for_batch` in `sglang_omni/scheduling/omni_scheduler.py`. The last long interval includes subsequent generation steps, scheduling, synchronization, and completion handling; it is not pure decode GPU time. The increase is not explained by a longer initial scheduler queue.

During these captures, Code2Wav residence decreases from 1451.77 to 672.63 ms and E2E changes from 2657.46 to 2689.49 ms. Total output audio changes from 1053.72 to 1057.20 seconds; reported completion-token totals are 3843 in both arms. These aggregates do not imply identical per-request codec sequences or batch membership.

## 2. Talker GPU execution versus submission cadence

Principal Talker forward graphs are selected from stream 7 by the 243/263-kernel families containing model attention and RMSNorm work. GPU graph execution span is the interval from its first kernel start to last kernel end, grouped by CUDA correlation ID. Host launch spacing uses the start timestamps of the corresponding `cudaGraphLaunch` calls.

| Measurement | Baseline | Optimized |
| --- | ---: | ---: |
| Principal forward graph replays | 2564 | 1499 |
| Mean GPU graph execution span | 1.282 ms | 1.306 ms |
| Median GPU graph execution span | 0.980 ms | 0.981 ms |
| Mean spacing between principal graph submissions | 8.373 ms | 14.502 ms |
| Median submission spacing | 4.725 ms | 6.295 ms |
| P95 submission spacing | 12.518 ms | 57.621 ms |

Submission spacing includes intervening prefill, sampling, other CPU/GPU operations, synchronization, and possible periods without this graph family. It is not all idle time, a per-request token interval, or a direct measurement of CPU scheduling overhead. The graph mix also changes, so the aggregate span comparison alone is not a matched-work experiment.

To check the graph-mix issue, graphs were grouped across runs by a fingerprint of their kernel names, launch grids, blocks, shared-memory sizes, and occurrence counts. All eight principal forward families appear in both arms. Seven families have shorter mean spans in optimized; one increases from 0.975 to 1.342 ms. For the most frequent optimized family (`0b27ff60f1b1fad42028`), mean span is 1.578 to 1.510 ms, median 1.047 to 1.046 ms, with 290 versus 764 observations. A matching launch fingerprint does not establish equal active batch membership, KV lengths, tensor values, or concurrent work.

A same-name/grid/block/shared-memory FlashAttention kernel signature has mean duration 10.42 to 14.46 microseconds and median 9.02 to 12.03 microseconds. Several matched GEMM signatures have nearly unchanged medians, while others decrease. Some kernel slowdowns exist, but they are not uniform and cannot be assigned to resource competition without matching the remaining workload dimensions.

## 3. Other stages perform a different sequence of GPU work

| Full-capture observation | Baseline | Optimized |
| --- | ---: | ---: |
| Thinker CUDA Graph replays | 492 | 776 |
| Thinker GPU kernel busy-time union | 5.555 s | 9.459 s |
| Flow-stream CUDA Graph replays | 170 | 1320 |
| Flow cuDNN SDPA kernel calls | 4480 | 21600 |

The Flow counters measure step-level replays and kernel calls, not requests, Flow batches, graph hit rates, or FLOPs. Individual work sizes differ: Flow SDPA summed kernel duration changes from 524.51 to 577.34 ms despite the much larger call count. A higher replay count therefore cannot by itself be called more useful computation or more graph misses.

Thinker token totals are unchanged while its launch counts and kernel busy time grow. This is consistent with changed batching and arrival patterns; exact batch/request attribution is needed to establish why. Comparing only the earlier two-second windows would obscure this full-run change.

## 4. Flow/HiFT overlap and the remaining default-stream dependency

Launch correlation links the baseline Code2Wav worker to stream 14 and some stream-7 kernels; its Talker worker launches model graphs on stream 7. In optimized, the Flow worker launches on stream 14, the HiFT worker launches on stream 18 and some stream-7 kernels, and the Talker worker uses stream 7. Reference-preparation threads also use stream 14, so that stream is not exclusively DiT.

Optimized stream-14 and stream-18 kernel intervals intersect for 350.70 ms over the full capture. This establishes concurrent GPU kernel activity on these two paths, not time saved or SM utilization. It includes stream-14 reference work and does not isolate every Flow batch. Baseline Flow/HiFT main work uses the same stream 14.

Both versions have the same `default_stream_for_cufft` helper in `components/token2wav/hift.py`: the default stream waits for the caller, STFT/ISTFT work runs on the default stream, then the caller waits for that stream. Correlation confirms that the Code2Wav/HiFT CPU worker launches FFT kernels on stream 7, which also carries Talker model work. Optimized therefore does not make all HiFT work independent of Talker.

This is an existing shared-stream dependency, not a newly introduced FFT helper. The FFT kernels themselves total only about 1.50 versus 1.96 ms; any important impact would have to come from the ordering and waits around them, not that small kernel sum alone. Current aggregate trace analysis does not establish their contribution to the Talker regression.

## Interpretation and next diagnostic target

Observed: Code2Wav completes sooner, Talker takes longer after its first forward, principal Talker graph submissions are more widely spaced, and the full serving run changes the GPU work distribution substantially. This supports investigating scheduling/batching feedback and cross-stream dependencies alongside GPU competition.

The next useful attribution is the long interval between Talker forwards: correlate each with CPU submission/synchronization, intervening prefill, Flow/HiFT readiness, and realized batch/Graph shape. Do not conclude GPU saturation from kernel busy-time unions or estimated per-kernel occupancy. These captures contain no measured SM Active/SM Issue counters, and CPU operator stacks/shapes were disabled.

## Reproduction and artifacts

- `event_decomposition.json`: complete-request event interval means and medians.
- `comparison.json`: graph counts, spans, submission gaps, and all eight matched-family comparisons.
- `baseline-preprocessing.json`, `baseline-talker.json`, `optimized-thinker.json`, `optimized-talker.json`: full-trace streaming summaries, per-signature duration statistics, runtime-thread statistics, graph correlation groups, and merged kernel intervals.
- Analysis scripts: `../../../../compare_gpu.py` and `../../../../summarize_gpu_comparison.py`, relative to this directory.

Kernel signatures use exact names/grid/block/shared-memory tuples. Graph fingerprints use the first captured launch for each graph ID. Kernel duration sums can overlap; busy-time union merges actual intervals. Source raw traces and prior reports are unchanged. No new GPU run was launched for this analysis.

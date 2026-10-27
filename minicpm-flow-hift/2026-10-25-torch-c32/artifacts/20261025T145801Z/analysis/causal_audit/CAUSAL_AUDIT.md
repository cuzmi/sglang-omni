# C32 causal audit from capture boundaries to execution dependencies

This re-analysis uses the same four raw Torch traces and archived request events. No GPU experiment or production change was made. It revises the priority of hypotheses: default-stream FFT coupling is a possible dependency, but more frequent and more expensive Talker first-forward processing has stronger direct evidence.

## Capture contract and limits

Baseline 078d0db and optimized 7584054 each served the same 256 formal inputs after 128 disjoint warmup inputs on one H100, at closed-loop concurrency 32. Each arm has one Torch capture, CPU/CUDA activities and request events. Tensor shapes, stacks and memory profiling were disabled. Warmup and capture/export are outside the formal cohort. All formal requests succeeded and the raw traces passed validation.

Fixed global concurrency is not a fixed batch size for any stage. Faster completion changes when subsequent requests are submitted, and hence the batch membership and arrival timing of upstream and downstream stages. Equal output-token totals and similar audio duration do not establish equal padded compute, plan/cache behavior, graph paths or numbers of forwards.

The profiler changes absolute timings and scheduling. Only the original event-only and unprofiled runs can establish whether a finding persists under lighter instrumentation. A GPU busy-time union is not measured SM utilization, and a kernel-signature match is not proof of equal active batch/KV lengths.

## Start with the request dependency chain

A slow Talker request can require more generation work, encounter slower GPU execution for comparable work, or wait longer between useful steps because of scheduling, dependency resolution or host work. These mechanisms can coexist. Identify which part lengthens before selecting a stream change.

The preceding event decomposition locates almost all of the additional Talker residence after the first forward. That interval includes subsequent generation plus intervening scheduling/first-forward work for other requests. It does not isolate pure decode computation.

## New observation: first-forward work is more fragmented

First-forward intervals are formed from scheduler_prefill_start to scheduler_prefill_end for each request. Overlapping intervals for requests executed together are merged, preventing batch time from being counted once per request. The 101 and 182 merged components are also corroborated by 2020 and 3640 non-graph Talker FlashAttention kernels, respectively: 20 layers per first forward.

| Full Torch capture | Baseline | Optimized |
| --- | ---: | ---: |
| Requests | 256 | 256 |
| Distinct first-forward intervals | 101 | 182 |
| New requests per first-forward interval | 2.53 | 1.41 |
| First-forward host elapsed-time union | 3.658 s | 8.433 s |
| Mean first-forward interval | 36.22 ms | 46.34 ms |

These are host-delimited intervals, including work and waits; they are not GPU compute times. New requests per interval is not the total active decode batch size. The increase combines more first-forward visits with longer per-visit elapsed time.

## Correlate long decode-graph submission gaps with those intervals

Select the middle 80% of formal elapsed time to exclude the first and last 10%. Measure start-to-start spacing of the principal Talker forward graph families, using CUDA correlation IDs. A long gap is greater than 20 ms. These gaps can contain other useful work.

| Middle-80% measurement | Baseline | Optimized |
| --- | ---: | ---: |
| All principal graph submission gaps | 2045 | 1113 |
| Gaps longer than 20 ms | 36 | 91 |
| Long gaps intersecting first-forward processing | 29 | 79 |
| Mean gap with no first-forward interval | 5.65 ms | 6.76 ms |
| Median gap with no first-forward interval | 4.81 ms | 5.85 ms |
| P95 gap with no first-forward interval | 11.32 ms | 13.97 ms |

The large aggregate gap increase is substantially associated with intervals containing first-forward work. Association does not mean all time inside those gaps was spent doing prefill: some long gaps include large unclassified host regions. Even without first-forward work, optimized still has longer gaps, so this is not a complete explanation.

OmniScheduler.event_loop_async_decode launches decode with lookahead, but flushes an in-flight decode before taking its synchronous prefill path. This gives a concrete mechanism by which frequent arrivals and fragmented prefill can interrupt progression of already-running requests. It does not establish why every first-forward interval is longer.

## Cross-check against two archived event-only C32 repetitions

The same interval-merging method is applied to the original 256-request runs without Torch profiling.

| Visit | First-forward intervals B → O | Host interval union B → O | Talker residence B → O |
| --- | ---: | ---: | ---: |
| Repeat 1 | 146 → 195 | 2.832 → 6.350 s | 568.47 → 1165.18 ms |
| Repeat 2 | 147 → 199 | 2.746 → 6.246 s | 555.96 → 1131.01 ms |

Thus the direction is not exclusive to the heavily instrumented Torch run. This still does not isolate batching, shared resources, synchronization, or arrival feedback as the initiating cause. All 16 event-only visits are retained in event_only_prefill.json.

## Stage population shows why C32 does not fix internal concurrency

Compute each stage's time-average request population as the sum of its 256 residence times divided by the full formal event interval. This is an exact finite-window residence-area calculation, not a stationarity assumption.

| Stage | Baseline | Optimized |
| --- | ---: | ---: |
| Thinker | 3.77 | 3.98 |
| Talker | 9.56 | 18.77 |
| Code2Wav | 16.95 | 7.77 |

Requests spend less of their lifetime in Code2Wav and more in Talker. This describes the changed system state; it is not independent evidence that moving waiting upstream caused the regression. It also does not measure simultaneously executing GPU batch size.

## CPU visibility is insufficient to close the causal chain

The Talker-process traces contain 7936 cpu_op events on trace thread 143 (baseline) or 982 (optimized). Correlated Talker CUDA launches appear on trace thread 1369430336 or 2141182272. There are no cpu_op events labeled with those CUDA-launch thread IDs. Without a verified mapping and worker-local CPU spans, the trace does not fully explain host work between Talker submissions. A blank region cannot be labeled Python/GIL contention, idle CPU, or removable overhead from these records alone.

In the entire principal-forward submission span, the union of Talker-thread cudaEventSynchronize/cudaStreamSynchronize host-call intervals decreases from about 0.973 s to 0.679 s. This does not exclude GPU-side cudaStreamWaitEvent dependencies, which can block GPU progress without a long host call. It does weaken an unqualified explanation that more visible host synchronization alone accounts for the regression.

## Hypothesis ranking and required next evidence

| Candidate | Current support | Still required |
| --- | --- | --- |
| Fragmented first-forward work interrupts ongoing decode | Direct event/trace alignment; repeats in event-only runs; scheduler has flush-before-prefill behavior | Mark request admission/build, prefill, decode launch and resolve to assign elapsed time and explain longer individual first-forward visits |
| More upstream/Flow work and host scheduling pressure | Thinker/Flow launch counts and work distribution change; remaining graph gaps increase | Realized batches/shapes, host-thread work and scheduling visibility |
| GPU execution becomes slower under shared-resource competition | Some same-launch-signature kernels slow; principal graph spans are not uniformly slower | Matched active work and overlapping-kernel context, not just same kernel names or grid |
| FFT/default-stream dependency blocks Talker | Shared stream and explicit dependencies exist in both versions | Show a ready/submitted Talker operation actually delayed by a specific HiFT dependency and quantify critical-path time |

## Kernel, overlap and fusion disposition

| Kernel/source attribution | Status |
| --- | --- |
| Talker principal graphs | Attention/RMSNorm families on stream 7; 243/263-kernel replay groups |
| Talker first forwards | Request events aligned with non-graph attention work |
| Flow/HiFT | Prior correlation-based stream attribution retained; stream 14 also includes reference preparation |

| Overlap opportunity | Status |
| --- | --- |
| Additional Talker stream | An ablation candidate only, not an established fix |
| Prefill/decode scheduling | First inspect why arrivals and synchronous visits fragment decode; preserve dependency correctness |

| Fusion pattern | Status |
| --- | --- |
| New kernel fusion | No evidence-backed proposal; current issue is execution organization and unexplained host time |

The next capture, if authorized, should preserve workload/defaults and add stage-specific worker spans for admission/build, prefill, decode launch/resolve, and Flow/HiFT readiness, with batch/Graph metadata. CPU scheduling visibility is needed if unexplained host regions remain. More kernel detail alone will not supply those missing facts.

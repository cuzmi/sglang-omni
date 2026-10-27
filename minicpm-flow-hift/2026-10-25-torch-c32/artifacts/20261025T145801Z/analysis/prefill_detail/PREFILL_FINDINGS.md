# Why Talker prefill becomes more fragmented and slower

This analysis revisits existing C32 evidence only: one full Torch capture per arm plus two archived event-only repetitions, each using 256 formal inputs. Baseline 078d0db and optimized 7584054. No production code changed and no GPU job was launched.

## 1. Fragmentation has a concrete source-level and observed explanation

Baseline returns results after the whole outer Code2Wav batch finishes. In baseline stages.py:259–278, model.vocode processes all voiced rows before outputs are constructed; SimpleScheduler.run_batch waits for batch_fn to return before emitting results. Baseline code2wav.py:503–516 completes all exact-length HiFT groups and copies the combined outputs to CPU before returning.

Optimized stages.py:480–528 takes one exact-length HiFT group, decodes it, and immediately emits its results before processing the next group. This removes the whole-outer-batch completion barrier.

The closed-loop benchmark replaces completed HTTP requests. Consequently, changing result-release granularity changes the next requests' arrival pattern even though concurrency remains 32.

```text
Baseline:   finish large outer batch → release a burst → refill a burst → larger ready groups
Optimized:  finish one HiFT group    → release a few   → refill a few   → smaller ready groups
```

Excluding the first 32 admissions, the remaining 224 refills provide 223 adjacent arrival gaps:

| Torch capture | Baseline | Optimized |
| --- | ---: | ---: |
| Audio-completion gap median | 0.54 ms | 62.41 ms |
| HTTP refill-admission gap median | 1.06 ms | 62.79 ms |
| HTTP refill-admission gap mean | 87.45 ms | 88.52 ms |
| Refill gaps at most 5 ms | 181/223 | 5/223 |

The nearly equal means with very different medians show that the major difference is burstiness, not average rate. Both event-only repetitions show the same pattern: refill medians about 1 ms in baseline versus 46–48 ms in optimized.

Across all six C32 visits and 970 first-forward groups, every group includes all requests already recorded as scheduler-ready at the group's first prefill-start marker. No ready-but-unselected request was found. Group size also matches batch_size metadata in all 970 groups. Thus the observed small prefills are explained by how many ready requests were available, rather than the scheduler leaving an existing larger ready queue unbatched. Proactively waiting for future arrivals would be a different policy and is not evaluated here.

Full membership, readiness and timing records: [arrival analysis](arrival/ARRIVAL_ANALYSIS.md).

## 2. What a prefill interval contains

OmniScheduler._run_batch (omni_scheduler.py:2091–2110) brackets model_runner.execute with scheduler_prefill_start/end. This includes forward-batch preparation, before_prefill, model execution, prefill sampling setup, sampling, and host result extraction. The output processor ultimately calls ids.tolist(), so the interval can include waiting for GPU results.

Request construction and the preceding async-decode drain occur before prefill_start. They can affect the time between decode steps but must not be counted as inside the measured prefill interval. MiniCPM's prefill sampling resets slot state and stages data; smaller, more frequent prefills repeat these per-visit costs.

## 3. Correlation-based decomposition of one prefill visit

For each of the 101 baseline or 182 optimized prefill intervals, select CUDA Runtime/Driver calls on the verified Talker launch thread whose entry lies inside that interval. Link GPU kernels by correlation ID. Every resulting interval has exactly 20 Talker FlashAttention kernels, consistent with one 20-layer forward. GPU work from the separate HiFT thread is excluded even when it uses the same default stream.

The table partitions each host interval into disjoint elapsed regions. All values are means per prefill visit, not per request and not matched-batch performance estimates.

| Elapsed region | Baseline | Optimized |
| --- | ---: | ---: |
| Marker start to first associated GPU kernel | 2.466 ms | 2.352 ms |
| Associated GPU kernel activity union | 1.361 ms | 1.500 ms |
| Between kernels, before next launch API entry | 17.443 ms | 17.999 ms |
| Between kernels, while next launch API is active | 1.941 ms | 2.967 ms |
| Between kernels, after next launch API returns | 12.128 ms | 21.055 ms |
| Last associated kernel to marker end | 0.882 ms | 0.464 ms |
| **Total prefill host interval** | **36.221 ms** | **46.336 ms** |

The principal increase in the aggregate mean is after the next kernel's launch API has already returned. Thus the increase cannot simply be called more CPU time before submitting that kernel. The post-return intervals can include prior queued operations, stream/event dependencies, GPU scheduling and shared-resource effects. They do not uniquely identify an FFT dependency or hardware saturation.

GPU memcpy/memset activity on stream 7 within these prefill windows averages only 0.033 ms and 0.038 ms, respectively, including any other owner on that stream. It does not account for the roughly 10 ms increase. Kernel-gap time is not whole-device idle time: other work can be active.

During these post-return gaps, interval intersection with Thinker kernel activity averages 9.264→13.487 ms per visit; with the Flow/reference stream, 1.190→4.139 ms. Optimized HiFT stream intersection is 0.539 ms. These overlap measures are not additive causal contributions and may overlap one another. They establish concurrent other-work intervals, not which event or resource prevented this kernel from starting.

## 4. The average increase is not universal per-input slowdown

The grouping and serving phase differ between arms. There are 26 sample IDs that form singleton prefills in both captures. For this restricted matched-input subset:

| Mean per singleton prefill | Baseline | Optimized |
| --- | ---: | ---: |
| Host interval | 39.884 ms | 40.877 ms |
| Associated GPU kernel union | 1.449 ms | 1.589 ms |
| Kernel count | 365.27 | 365.27 |

This subset has matching input IDs and one-request grouping but not matching concurrent activity or launch time. It demonstrates that the aggregate 36→46 ms change must not be described as every identical prefill becoming 10 ms slower. It does not replace all-request performance results or prove an absence of contention.

## 5. Connection to Talker decode delay

Smaller ready groups cause more prefill visits. Each visit repeats preparation, launches and synchronous result handling, and the async-decode scheduler drains an in-flight decode before its synchronous prefill path. Existing decoding requests therefore encounter more interruptions. The first-forward event-only counts also increase (146→195 and 147→199), with total intervals increasing from 2.83→6.35 s and 2.75→6.25 s.

The observed behavior can be explained in plain terms: the earlier release of small audio groups changes the next arrivals, so Talker handles more small intake batches. Those visits repeat substantial fixed and waiting costs. In the Torch capture, the larger mean visit time is concentrated in gaps after kernel submission.

The fragmentation mechanism is well supported by source plus repeated events. The precise cause of the post-submission wait remains unresolved. Default-stream waits, cross-process GPU scheduling and other dependencies remain candidates. The existing CUDA wait API records do not provide a complete producer-event/consumer dependency chain, and worker CPU attribution remains incomplete. No stream migration or scheduling change is yet justified as a proven fix.

## Analysis artifacts and checks

- [Arrival analysis](arrival/ARRIVAL_ANALYSIS.md), per-visit memberships and refill distributions.
- gpu/baseline.json and gpu/optimized.json: per-prefill kernels, host launch times, runtime calls, signatures and elapsed decomposition.
- gpu/wait_context.json: other-stream intersections during post-return gaps.
- gpu/matched_single_requests.json: the 26 matched singleton sample pairs.
- gpu/memory_activity.json: default-stream copy/set timing upper bound within prefill intervals.

Scripts at the experiment root: analyze_prefill_gpu.py, prefill_gap_breakdown.py, prefill_wait_context.py. All batch partitions reconstruct their kernel-gap totals to floating-point precision; all 283 Torch first-forward intervals contain exactly 20 correlated attention kernels. Raw captures and earlier reports are retained unchanged.

## Attribution and optimization disposition

| Kernel/source | Evidence |
| --- | --- |
| Talker first-forward kernels | Launch-thread correlation inside request-event boundaries; 20 attention layers in every interval |
| Default-stream FFT work | Launched by a different HiFT worker; excluded from the associated prefill kernel union |

| Overlap or scheduling candidate | Status |
| --- | --- |
| Coalescing future arrivals | Possible policy experiment, with added waiting; not a validated fix |
| Isolating Talker's stream | Requires an identified blocking dependency; not established by these post-return gaps alone |

| Fusion pattern | Disposition |
| --- | --- |
| Prefill kernel fusion | No change proposed; first attribute post-submission waits and repeated per-visit overhead |

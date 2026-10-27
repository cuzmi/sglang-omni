# Talker time before the first forward at 30 requests/s

Reanalysis of the existing event-only baseline/optimized pair: 256 formal requests each, same input and arrival schedule. No source changes or new GPU runs. All event pairs are unique and ordered, and the four intervals sum exactly to each request's previously reported prefill-entry time.

| Interval | Baseline mean ms | Optimized mean ms | Change ms |
|---|---:|---:|---:|
| Stage entry to request build | 7.985 | 13.040 | +5.055 |
| Request build | 4.317 | 4.645 | +0.328 |
| Build completion to queue entry | 0.059 | 0.062 | +0.003 |
| Queue entry to first forward | 60.046 | 428.635 | +368.589 |
| Total | 72.408 | 446.382 | +373.974 |

Queue entry to first forward accounts for 98.56% of the additional prefill-entry time. Request construction itself adds only 0.328 ms per request.

| Queue-entry-to-first-forward distribution | Baseline ms | Optimized ms |
|---|---:|---:|
| p50_ms | 1.571 | 366.943 |
| p95_ms | 308.932 | 1070.445 |
| max_ms | 380.787 | 1151.880 |

The build span brackets the model-specific request builder, including condition embeddings and request construction. The queue-entry event is emitted just before waiting_queue.append. Prefill start is emitted at the first model-runner entry. Their gap includes scheduling, work on other requests, batch selection/preparation, and possible pending-decode resolution; it is not a pure measurement of waiting for a capacity slot or GPU compute.

Source anchors at optimized commit 758405416635738e8812cbddbe170dd94339f5e6: sglang_omni/scheduling/omni_scheduler.py:1328 (builder events), :1733 (queue entry), :2091 (model-runner entry), :3669 (async decode loop); sglang_omni/models/minicpm_o/talker_request.py:129 (condition and request construction).

The result localizes the extra elapsed time after request construction and queue admission. These event boundaries alone do not determine which scheduling activity or resource dependency caused that delay.

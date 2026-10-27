# Arrival pattern and Talker prefill fragmentation

Read-only analysis of archived coordinator, Thinker and Talker events from the Torch C32 pair and both event-only C32 pairs. No new GPU run or production change.

## Main observation

Talker receives more dispersed requests in optimized. The difference is already visible at the coordinator's HTTP admission boundary, and the replenishment pattern closely follows Code2Wav completion bursts. The sampled scheduler did not split a larger already-ready queue into smaller prefills: at every observed first-forward group in all six visits, every request with scheduler_queue_enter before the group's first scheduler_prefill_start was included in that group, after excluding requests whose first forward had already started.

## From Code2Wav completion to new request arrival

Exclude the initial 32 admissions. The remaining 224 admissions have 223 adjacent intervals; compare the first 224 audio completions that replenish those slots. Audio completion here is coordinator terminal_response whose metadata.from_stage is code2wav, not the separate decode/text terminal response.

| Capture | Median audio-completion gap B → O | Median refill-admission gap B → O | Refill gaps ≤5 ms B → O |
| --- | ---: | ---: | ---: |
| Torch | 0.54 → 62.41 ms | 1.06 → 62.79 ms | 181/223 → 5/223 |
| Event-only repeat 1 | 0.33 → 48.07 ms | 1.13 → 48.16 ms | 173/223 → 7/223 |
| Event-only repeat 2 | 0.31 → 46.17 ms | 1.03 → 46.32 ms | 184/223 → 5/223 |

The mean refill interval is much closer across arms (Torch 87.45→88.52 ms; event-only59.43→62.10 and58.60→60.35 ms). Thus the key difference is burstiness, not merely a slower average request arrival rate. Baseline has many near-simultaneous completions/refills separated by long gaps; optimized has more evenly spaced completions/refills.

Sorted completion-to-refill offsets are positive and small relative to inter-burst gaps, supporting the closed-loop feedback interpretation. These offsets pair temporal ranks, not a per-request causal successor identity. Their full distributions are retained in refill.json.

## The fragmentation persists through Thinker into Talker

All256 inputs are included for this table (255 adjacent arrival intervals).

| Capture | Talker input median gap B → O | Talker queue-entry median gap B → O | Prefill groups B → O | Singleton groups B → O |
| --- | ---: | ---: | ---: | ---: |
| Torch | 12.94 → 44.73 ms | 4.90 → 56.43 ms | 101 → 182 | 54 → 134 |
| Event-only repeat 1 | 13.44 → 35.10 ms | 16.22 → 44.48 ms | 146 → 195 | 86 → 149 |
| Event-only repeat 2 | 13.57 → 39.27 ms | 15.45 → 43.52 ms | 147 → 199 | 92 → 159 |

## Was the scheduler leaving ready requests behind?

For each prefill group, collect all unprocessed requests with scheduler_queue_enter timestamp at or before the group's earliest scheduler_prefill_start. Compare that set with the group's members.

- Ready-but-unselected requests: zero in every group in all six visits (970 groups total).
- Each group's recorded batch_size equals the number of grouped new requests in every group.
- Some requests had arrived at the Talker stage but were not yet queued:44/101 baseline and34/182 optimized Torch groups;52/146 vs33/195 in event-only repeat1;50/147 vs37/199 in repeat2. Stage receipt therefore cannot be equated with scheduler-ready input.
- In the Torch pair, mean Talker input-to-build-start decreases41.14→27.70 ms, while build time increases3.41→4.30 ms and queue-to-prefill decreases10.88→6.65 ms. The more fragmented ready queue is not explained by a longer mean initial queue wait.

This supports an arrival-limited batching explanation for the observed group sizes. It does not prove that an alternative scheduler policy could not coalesce future arrivals; that would trade current waiting time against batch efficiency and needs its own evaluation.

## Method and limits

Each request's first-forward interval is scheduler_prefill_start→scheduler_prefill_end. Overlapping intervals are grouped as a batch, as in the preceding audit. All256 requests have the required boundaries. The resulting batch_size metadata matches every group exactly. Selection occurs shortly before the recorded prefill_start marker, so queue availability is measured at the marker, not at an independently recorded selector instant. There are nevertheless no already-queued excluded requests even at this later boundary.

These observations identify the changed arrival pattern and its propagation. They do not alone prove that all added prefill elapsed time is caused by smaller batches, and do not resolve why a single prefill is slower. GPU/host decomposition is a separate investigation.

Artifacts: summary.json contains six compact summaries; per-visit JSON files retain group memberships, marker times, queue availability and stage boundary distributions; refill.json contains the initial32-excluded comparison. Analysis script was executed from /private/tmp/prefill_arrival_analysis.py and /private/tmp/refill_stats.py.

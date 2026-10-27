# MiniCPM-o given-text streaming results

Experiments for text-to-speech streaming and chunk decoding. Future results belong in dated directories under this topic; directory names describe the subject and do not depend on PR numbers.

| Experiment | Result and limits |
| --- | --- |
| [Streaming modes](2026-10-27-streaming/REPORT.md) | 48 measured requests completed. Concurrent generated audio interleaves, but meaningful TTFA and completion medians regress. |
| [Chunk graph](2026-10-27-chunk-graph/REPORT.md) | Comparison invalid: enabled warmup produced nonfinite audio. Interrupted diagnosis and later preparation do not establish a successful fix. |

Only reports and evidence are stored here. Each experiment links to a fixed RemoteFiles commit for scripts and source context.

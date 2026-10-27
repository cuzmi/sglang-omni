# MiniCPM-o streaming chunk graph — 2026-10-27

The graph-enabled comparison is INVALID. The graph-disabled condition completed 8 warmup and 48 measured requests, but the enabled condition failed during warmup with nonfinite audio, an int16-conversion warning and a silent Chinese request. There is no valid graph-on/off timing comparison and no speedup conclusion.

| Material | Status |
| --- | --- |
| [Invalidity record](artifacts/stream-chunk-20261027T195127Z/INVALID.json) | Authoritative failure boundary; original snapshot SHA256 5476b0b886a689995117c775ba8f0cf90846de525734d8977b4ab5e78aefd964 |
| [Disabled condition](artifacts/stream-chunk-20261027T195127Z/without_chunk_graph/analysis.json) | Completed condition only; not a substitute for the failed comparison |
| [Enabled condition](artifacts/stream-chunk-20261027T195127Z/with_chunk_graph/status.json) | Failed during warmup; partial summary and logs retained |
| [Earlier preparation attempt](preparation_attempts/stream-chunk-20261027T194954Z/status.json) | Stopped during CPU preflight for capture-width correction |
| [Follow-up diagnostic log](diagnostic_modal.log) | Started bad-checkpoint diagnostic, then stopped from CLI; no completed diagnostic result available |

The [executed protocol](artifacts/stream-chunk-20261027T195127Z/protocol.json) and its source manifest remain inside the run directory. Top-level protocol/source/patch files describe later preparation and differ from the executed snapshot. The planned timestep-buffer correction and stricter finite-waveform checks are preparation, not evidence of a successful corrected run. No completed corrected comparison was present in this archive snapshot.

[Local comparison checker receipt](local_comparison_validation.json) and [probe checker receipt](local_probe_validation.json) concern local validation tools. The historical change manifest records 98 passed/14 deselected and a later 8-case wiring check; precommit logs also record unavailable cargo for the unchanged Rust formatting hook. These receipts are not a passing end-to-end GPU comparison, and were not rerun during publication.

The disabled condition uses the same fixed English/Chinese workload, 24 kHz PCM, C1/C2 scenarios and stream/nonstream ABBA measurements. Its standalone analysis has matching input and codec gates, but must not be compared against another historical run to infer chunk-graph gains. Audio and complete source packages remain local; hashes and the failed-run evidence are preserved.

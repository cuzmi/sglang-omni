# MiniCPM-o Flow/HiFT Results

Benchmark, accuracy and profiling results for MiniCPM-o readiness-aware batching
and decoupled Flow/HiFT execution. Each dated directory preserves one experiment
and its measurement context.

## October 25–26 follow-up results

Unprofiled open-loop runs measure performance at fixed offered arrival rates. Event-only and Torch runs diagnose stage residence and execution; their elapsed times do not establish unprofiled speedup.

| Experiment | Result and scope |
| --- | --- |
| [Request-event stage comparison](2026-10-25-stage-events/README.md) | 16 completed visits; stage residence includes waiting and execution. The diagnostics/ directory is an earlier 8/16-visit snapshot, not the final report. |
| [C32 Torch traces and follow-up analyses](2026-10-25-torch-c32/README.md) | One capture per arm; Torch timings are diagnostic. Later full-trace, prefill and arrival analyses refine the initial capture report. Raw full traces and viewing windows remain distinct. |

## Earlier performance, accuracy and mechanism results

| Evidence | Result | Scope |
| --- | --- | --- |
| [256-input serving, two passes](2026-10-24-serving/REPORT.md) | C8 QPS +11.3%; C16 +6.0%; C1 -0.3%; C32 -1.3% | Full serving; generated work varies; two observations per arm |
| [Frozen-input accuracy](2026-10-23-frozen-accuracy/REPORT.md) | 256/256 mel outputs and 246/256 waveforms bit-exact; identical corpus WER | Compiled non-graph Flow path: 0 replays, 16 misses; no graph-hit correctness claim |
| [Same-graph control](2026-10-20-same-graph/20261020T171213Z/REPORT.md) | Unprofiled 153.093 → 131.118 ms (-14.35%); GPU kernel intersection 15.268 ms | Isolated ready-input Code2Wav; not an E2E throughput result |

Full-serving quality is not equivalent to frozen-input numerical accuracy. Overall serving WER changed from 2.781% to 2.932%, and mean speaker similarity from 48.672 to 48.619. Concurrent output-length outliers and all quality scores remain included. No formal quality-equivalence claim is made.

## Supporting and historical results

| Experiment | Evidence boundary |
| --- | --- |
| [Fixed batches](2026-10-20-fixed-batches/README.md) | Original mixed-shape comparison, including the failed initial RNG check |
| [Kernel overlap trace](2026-10-20-kernel-overlap/README.md) | GPU kernel attribution; separate from unprofiled timing |
| [SM metrics](2026-10-20-sm-metrics/README.md) | Device-wide Active/Issue samples; not remaining compute capacity or serving speedup |
| [Earlier serving comparison](2026-10-22-serving/README.md) | 32 measured requests per cell, earlier revisions and runtime |
| [Stage comparison](2026-10-22-stage-comparison/README.md) | Host-observed request residence; not pure GPU compute |
| [C16 repeat](2026-10-22-c16-repeat/README.md) | Retains the E2E direction reversal and work drift |

The October 20 controls use `917821a95770c89f875fc0e4c4456f85c221d441`. October 22 compares `90923df1057a7ab69a3c7e124d18bb4e6d70c304` with `917821a95770c89f875fc0e4c4456f85c221d441`. October 23–24 compares `078d0dbaabc82da3e67fbdda038257cd9a391051` with `758405416635738e8812cbddbe170dd94339f5e6`. These are different experiments, not repeated measurements of one unchanged build.

## Provenance

[PROVENANCE.json](PROVENANCE.json) pins every copied result to its original RemoteFiles commit and hash. Raw result bytes are unchanged. The frozen-accuracy report omits reproduction instructions; the documented transformation preserves its results and limitations.

[EXTERNAL_EVIDENCE.json](EXTERNAL_EVIDENCE.json) links large kernel tables and Nsight captures at a fixed commit. Audio, tensors, source bundles, reproduction scripts, patches and PR drafts are not copied into this branch. Original run protocols and scripts remain in the linked RemoteFiles archives. SHA256SUMS.json at the branch root covers exactly the published files, excluding itself. No GPU experiment was rerun for publication.

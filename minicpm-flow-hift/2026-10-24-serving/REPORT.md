# MiniCPM-o unprofiled serving: RADIX H100

Two complete passes reproduce gains at C8 and C16. C1 is effectively flat, and C32 shows no speedup. Per-request audio length varies substantially at concurrent settings; quality results below retain all outputs, including long outliers.

Baseline `078d0dba` versus optimized `75840541`; one H100 80GB; fixed 256 English SeedTTS inputs per cell and 128 disjoint warmup inputs. See PROTOCOL.md for pinned sources, runtime, timing, and ownership boundaries.

The maintained benchmark constructs requests, dispatches closed-loop concurrency, and computes speed metrics. An identical experiment-only patch defers response decoding and audio writes until the entire timed cohort finishes. Both engines have a controlled 131,072-token KV limit; other compile, Graph, and batching defaults are preserved.

## Complete serving cells

| Repeat | Arm | C | Success | QPS | Audio s/s | Mean ms | p50 ms | p95 ms | Total audio s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | baseline | 1 | 256/256 | 3.486 | 14.438 | 286.8 | 286.7 | 393.2 | 1060.24 |
| 1 | optimized | 1 | 256/256 | 3.500 | 14.497 | 285.7 | 285.1 | 391.7 | 1060.24 |
| 1 | baseline | 8 | 256/256 | 10.808 | 44.899 | 731.7 | 722.5 | 984.9 | 1063.52 |
| 1 | optimized | 8 | 256/256 | 11.949 | 50.366 | 662.0 | 642.5 | 899.3 | 1079.04 |
| 1 | baseline | 16 | 256/256 | 16.115 | 67.159 | 976.5 | 979.5 | 1260.8 | 1066.84 |
| 1 | optimized | 16 | 256/256 | 17.106 | 70.311 | 920.7 | 923.3 | 1244.0 | 1052.24 |
| 1 | baseline | 32 | 256/256 | 21.941 | 91.938 | 1421.5 | 1382.4 | 1968.6 | 1072.72 |
| 1 | optimized | 32 | 256/256 | 21.466 | 89.535 | 1450.8 | 1397.4 | 1984.9 | 1067.80 |
| 2 | baseline | 1 | 256/256 | 3.541 | 14.664 | 282.4 | 282.7 | 382.3 | 1060.24 |
| 2 | optimized | 1 | 256/256 | 3.502 | 14.503 | 285.5 | 286.5 | 387.4 | 1060.24 |
| 2 | baseline | 8 | 256/256 | 10.918 | 45.086 | 724.7 | 714.0 | 980.2 | 1057.20 |
| 2 | optimized | 8 | 256/256 | 12.242 | 51.242 | 646.8 | 635.9 | 858.6 | 1071.60 |
| 2 | baseline | 16 | 256/256 | 15.809 | 65.848 | 996.5 | 1004.4 | 1292.6 | 1066.32 |
| 2 | optimized | 16 | 256/256 | 16.734 | 69.834 | 939.5 | 938.8 | 1233.8 | 1068.36 |
| 2 | baseline | 32 | 256/256 | 22.249 | 92.961 | 1400.9 | 1360.5 | 1933.7 | 1069.64 |
| 2 | optimized | 32 | 256/256 | 22.172 | 91.531 | 1399.8 | 1394.6 | 1889.4 | 1056.84 |

## Matched completed repetitions

Rates below use total completed requests or audio seconds divided by total measured wall time across paired repetitions. Unpaired cells remain visible above but are excluded from the comparison.

| C | Paired repeats | Baseline QPS | Optimized QPS | QPS change | Audio throughput change | Generated audio change |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 3.513 | 3.501 | -0.3% | -0.3% | +0.0% |
| 8 | 2 | 10.862 | 12.094 | +11.3% | +12.9% | +1.4% |
| 16 | 2 | 15.961 | 16.918 | +6.0% | +5.4% | -0.6% |
| 32 | 2 | 22.094 | 21.813 | -1.3% | -2.1% | -0.8% |

## Interpretation boundaries

These measurements cover the combined serving change, including native batch and Graph behavior. They do not isolate overlap. Identical prompts and seeds do not guarantee equal generated work; total and per-request audio duration are retained. Audio throughput accounts for output duration but does not remove all batching or shape confounders. Returned completion tokens are not assumed to be codec tokens. Per-request paired lengths are available in request_length_comparison.csv.

This is a finite 256-request workload at each concurrency. A single pair is preliminary; complete ABBA pairs supply only two observations per arm. The shared host may introduce CPU or storage variability despite matching CPU affinity. No event or GPU profiler was enabled.

## Per-request output-work variation

Aggregate totals hide substantial per-request length changes. Every paired returned text completion is identical in text_output_parity.json; the audio work is not fixed. Long outputs remain included in all performance and quality statistics.

| C | Changed audio lengths / pairs | Mean absolute length difference | Maximum absolute difference |
|---:|---:|---:|---:|
| 1 | 0/512 | 0.000 s | 0.00 s |
| 8 | 444/512 | 0.412 s | 14.68 s |
| 16 | 464/512 | 0.401 s | 3.68 s |
| 32 | 440/512 | 0.409 s | 11.96 s |

## Offline quality on the actual serving outputs

All saved formal outputs are scored after serving stops. The maintained SeedTTS pipelines use pinned Qwen3-ASR weights and the verified WavLM speaker models. WER is corpus-level; Similarity is the mean score on the 0–100 scale. Each per-concurrency arm includes 512 outputs from 256 unique inputs repeated twice.

| C | Baseline WER | Optimized WER | Baseline Similarity | Optimized Similarity |
|---|---:|---:|---:|---:|
| 1 | 2.440% | 2.367% | 48.490 | 48.490 |
| 8 | 2.495% | 3.114% | 48.795 | 48.649 |
| 16 | 3.460% | 2.932% | 48.517 | 48.802 |
| 32 | 2.731% | 3.314% | 48.884 | 48.533 |
| all | 2.781% | 2.932% | 48.672 | 48.619 |

Overall WER changes by +0.150 percentage points and mean Similarity by -0.053. Per-concurrency quality changes differ, and the two repetitions also vary; see quality_cells.csv. These observations do not establish quality equivalence or identify the cause of the changes.

The two largest positive length differences are +14.68 s at C8 and +11.96 s at C32. Their optimized-output ASR alignments contain 13 and 31 insertions, respectively. All long outputs remain in the full-corpus scores. The ten largest absolute length differences, including cases where baseline is longer, are saved in outlier_examples.json. This serving result warrants separate follow-up from the earlier frozen-input numerical accuracy check.

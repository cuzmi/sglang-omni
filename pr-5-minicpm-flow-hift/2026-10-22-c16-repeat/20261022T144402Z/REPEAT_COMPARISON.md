# C16 request-event profiling repeat

Previous run `20261022T141048Z` versus repeat `20261022T144402Z`. Both use baseline `90923df` then optimized `917821a`, a fresh server per arm, the same 32 evaluation samples, 32 separate unprofiled warmup samples and unchanged sampling. Only measured requests enable request-event profiling. Each pair uses one H100; separate runs are not assumed to share the same physical GPU.

Stage values below are mean residence times, including waiting and execution. E2E is measured at the client. Results from the two repetitions are kept separate.

| Run | Arm | Thinker ms | Talker ms | Code2Wav ms | E2E mean ms | E2E p95 ms | QPS | Total audio s |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| previous | baseline | 298.61 | 388.78 | 909.57 | 1672.09 | 2105.39 | 8.910 | 123.84 |
| previous | optimized | 300.11 | 645.66 | 769.74 | 1769.76 | 2489.33 | 7.632 | 124.28 |
| repeat | baseline | 312.20 | 486.10 | 1044.21 | 1912.93 | 2650.85 | 7.787 | 126.92 |
| repeat | optimized | 262.96 | 586.00 | 661.18 | 1551.58 | 1962.29 | 8.783 | 124.84 |

## Within-run changes (optimized relative to baseline)

| Run | Talker change | Code2Wav change | E2E mean change |
|---|---:|---:|---:|
| previous | +66.1% | -15.4% | +5.8% |
| repeat | +20.6% | -36.7% | -18.9% |

## Output work and scope

Across repetitions, 14/32 baseline requests and 7/32 optimized requests had identical audio lengths. The full paired records are in output_lengths_across_repeats.csv; within-repeat baseline/optimized comparisons are in output_lengths_paired.csv.

Matching input samples and seeds does not ensure identical generated work, batch shapes or Graph paths. Refer to REPORT.md for current output-length differences and actual KV allocations. Event spans alone do not identify GPU contention or isolate queueing from execution. Two short observations do not establish a stable effect or isolate profiler overhead. No Torch/CUPTI, NVTX, Nsight or SM profiling was added.

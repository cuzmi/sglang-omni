# Same-graph SM Active and SM Issue comparison

Modal H100 GPU metric collection succeeded. The dedicated probe and the same-node gate both produced nonzero SM Active and SM Issue samples. The full run used Nsight Systems 2026.4.1, the general GH100 metric set and 10 kHz sampling.

## Setup and validity

- Source: unchanged commit `917821a95770c89f875fc0e4c4456f85c221d441`.
- Model: MiniCPM-o-4_5 revision `503e754207c94da6bb26850b4469f367c9ea3582`.
- A/B: the same four real inputs replayed twice, eight executions, one shared Flow graph `(4, 576)`; unchanged per-batch HiFT grouping.
- Four captures per mode in two serial/overlap/overlap/serial blocks after warmup and correctness validation.
- Input tokens, reference manifest, effective configuration, noise hash and runtime settings match the preceding same-graph experiment.
- Every formal pass: twenty graph replays, zero misses, zero new compilation. All graph launches use graph ID 188 in this capture.
- 264 expected NVTX ranges, 160 graph launches, 108,192 observed kernels; zero unmatched launch correlations. Corresponding kernel signatures match across all eight passes.
- Mel tensors were bit-exact in validation. Six of eight waveforms were bit-exact; maximum absolute difference 0.000125736 and maximum relative RMS 0.000033981, below the predeclared 0.001 threshold.
- 13,388 samples for each retained metric. Actual sample spacing: 0.099998–0.100000 ms. No gap over 0.5 ms occurred in any measured window.

## Primary comparison

Values are percentages, averaged within each pass and then equally across four passes. Parentheses show the standard deviation across the four pass means. Zero-valued samples are included.

| Window | Mode | SM Active | SM Issue |
| --- | --- | ---: | ---: |
| full_gpu | serial | 45.68% (2.54) | 19.57% (1.09) |
| full_gpu | overlap | 59.22% (0.77) | 25.25% (0.30) |
| flow_b_graph | serial | 72.88% (0.10) | 32.23% (0.09) |
| flow_b_graph | overlap | 78.76% (0.65) | 33.53% (0.22) |
| overlap_span | overlap | 81.92% (0.51) | 34.11% (0.17) |

`full_gpu` spans the first Flow A kernel start through the last observed kernel. `flow_b_graph` spans the first through last Flow B graph kernel. `overlap_span` is the intersection of the Flow B graph and HiFT A first-to-last spans; it includes gaps between kernels. These device-wide metrics include both components during overlap and cannot be attributed to either one separately.

## Supporting context from the same metric set

| Window | Mode | Tensor Active | DRAM read | DRAM write |
| --- | --- | ---: | ---: | ---: |
| full_gpu | serial | 11.82% | 2.30% | 4.94% |
| full_gpu | overlap | 15.15% | 3.44% | 7.29% |
| flow_b_graph | serial | 21.88% | 2.93% | 7.45% |
| flow_b_graph | overlap | 21.37% | 4.86% | 10.50% |
| overlap_span | overlap | 20.95% | 6.02% | 12.31% |

## Interpretation

Overlap increases average SM activity and instruction issue over the whole execution window. During the Flow B graph window, Active rises from 72.88% to 78.76%, while Issue rises only from 32.23% to 33.53%. Its first-to-last graph span grows from 39.55 to 45.26 ms in this diagnostic capture. Higher activity therefore coexists with a slower component.

The narrower overlap span averages 81.92% SM Active and 34.11% SM Issue. This does not establish that the GPU is full or that 65.89% of compute capacity remains available. Active measures cycles with resident warps; Issue measures cycles with instruction issue by warp schedulers. Neither is a direct fraction of peak FLOP/s, and the two metrics are not additive. Resource pressure within a subset of kernels can coexist with modest whole-window averages.

Use unprofiled completion time to judge performance, and these counters to explain scheduling behavior. The prior independent unprofiled result was 153.09 -> 131.12 ms (-14.35%). Do not recompute a speedup from this capture or compare its durations directly against an older capture on a different GPU allocation.

Tensor Active and DRAM percentages provide additional context, but their window averages do not exclude transient kernel-level bottlenecks. This experiment does not identify the specific resource causing the overlap slowdown or establish serving capacity.

## Kernel/source attribution

| Region | Source | Kernels per pass |
| --- | --- | ---: |
| Flow/A/graph | `flow_cuda_graph.py:372 run_step / captured.graph.replay` | 4020 |
| HiFT/A | `code2wav.py:468 decode_mel / token2wav.hift` | 2180 |
| Flow/B/graph | `flow_cuda_graph.py:372 run_step / captured.graph.replay` | 4020 |
| HiFT/B | `code2wav.py:468 decode_mel / token2wav.hift` | 2180 |

## Overlap opportunity

| Work | Dependency | Observation |
| --- | --- | --- |
| HiFT A and Flow B | HiFT A waits on Flow A readiness; Flow B is independent of HiFT A | Actual graph-B / HiFT-A kernel intersection is 0 in all serial passes, and 13.03–13.21 ms in the four overlap passes |

## Fusion patterns

| Existing execution | Change | Scope |
| --- | --- | --- |
| Compiled DiT/HiFT and Flow graphs | No fusion or compiler change | Scheduling and device metrics only |

## Collection limits

Nsight reported four unsupported auxiliary CTC transport counters, which were skipped; the requested SM counters and the listed Tensor/DRAM counters were present. Generic possible-incomplete CUDA/NVTX-event warnings were also retained. Marker, launch, kernel-correlation and signature validation passed for observed events; this does not guarantee unobserved events were collected.

The screenshot shows the first serial/overlap pair by execution order, not the four-pass average. Curves use 0.5 ms bins only for display; numerical tables use the raw 10 kHz samples. Both apps stopped with zero remaining tasks.

## Evidence

- `trace/timeline.nsys-rep` and `trace/timeline.sqlite`: original capture and export.
- `trace/gpu_metrics.csv`: all raw samples for the five retained metrics.
- `trace/window_metrics.csv` and `trace/metrics_analysis.json`: per-pass windows and aggregation.
- `trace/attributed_kernels.csv` and `trace/analysis.json`: launch-attributed kernels and validity.
- `trace/sm_metrics_timeline.png` and `.svg`: visualization.

Definitions and collection requirements: [NVIDIA Nsight Systems GPU metrics](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#gpu-metrics).

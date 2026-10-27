# MiniCPM-o C32 Torch traces

Both full-serving captures completed on one H100 in Modal workspace `stangameact`. Baseline `078d0db`, optimized `7584054`, the same fixed 256 formal inputs and 128 independent warmup inputs per fresh server, concurrency 32. The pinned image/model, benchmark, engine capacities and default compilation/CUDA Graph settings match the preceding event-only run. See [capture protocol](https://github.com/cuzmi/RemoteFiles/blob/428bf60817c2c57e8443982bba591161b0868603/research/minicpm_serving_torch_c32_modal_h100_20261025/PROTOCOL.md) and [frozen configuration and hashes](https://github.com/cuzmi/RemoteFiles/blob/428bf60817c2c57e8443982bba591161b0868603/research/minicpm_serving_torch_c32_modal_h100_20261025/protocol.json).

All 512 formal and 256 warmup requests succeeded. Four valid raw Chrome traces contain CPU/CUDA events, and all 512 formal requests have complete stage-event boundaries. The GPU app has completed. [Modal capture](https://modal.com/apps/stangameact/main/ap-PhUOlhUyIXwoEqLzTqaAPB).

## Start with these viewing files

| Version | Two-second combined GPU/runtime overview | Window statistics |
| --- | --- | --- |
| Baseline | [Open/download trace](https://github.com/cuzmi/RemoteFiles/blob/428bf60817c2c57e8443982bba591161b0868603/research/minicpm_serving_torch_c32_modal_h100_20261025/artifacts/20261025T145801Z/analysis/baseline/gpu_overview_2s.trace.json.gz) | [Summary](analysis/baseline/summary.json) |
| Optimized | [Open/download trace](https://github.com/cuzmi/RemoteFiles/blob/428bf60817c2c57e8443982bba591161b0868603/research/minicpm_serving_torch_c32_modal_h100_20261025/artifacts/20261025T145801Z/analysis/optimized/gpu_overview_2s.trace.json.gz) | [Summary](analysis/optimized/summary.json) |

The viewing interval starts at 40% of each capture's formal stage span and lasts two seconds. These are independently selected interior windows, not identical batches or matched amounts of GPU work. The combined overviews contain GPU kernels, copies and CUDA runtime calls. Source worker process IDs are namespaced to avoid collisions, timestamps are aligned using each raw trace's baseTimeNanoseconds and the request-event wall clock, and boundary spans are clipped. These are viewing copies, not replacements for raw evidence. Per-worker `.window.trace.json.gz` files in the same directories retain the original CPU operators, correlation events, timestamps and durations for events intersecting the window.

## Full original traces

| Version | Process contents | Raw trace | GPU kernels | Active kernel streams |
| --- | --- | --- | ---: | --- |
| Baseline | Thinker and colocated pipeline stages | [Full trace](https://github.com/cuzmi/RemoteFiles/blob/428bf60817c2c57e8443982bba591161b0868603/research/minicpm_serving_torch_c32_modal_h100_20261025/artifacts/20261025T145801Z/results/r1-baseline-c32/traces/preprocessing_pid142_rank0.trace.json.gz) | 305,186 | 7 |
| Baseline | Talker and Code2Wav | [Full trace](https://github.com/cuzmi/RemoteFiles/blob/428bf60817c2c57e8443982bba591161b0868603/research/minicpm_serving_torch_c32_modal_h100_20261025/artifacts/20261025T145801Z/results/r1-baseline-c32/traces/talker_pid143_rank0.trace.json.gz) | 1,235,723 | 7, 14 |
| Optimized | Thinker and colocated pipeline stages | [Full trace](https://github.com/cuzmi/RemoteFiles/blob/428bf60817c2c57e8443982bba591161b0868603/research/minicpm_serving_torch_c32_modal_h100_20261025/artifacts/20261025T145801Z/results/r1-optimized-c32/traces/thinker_pid981_rank0.trace.json.gz) | 558,083 | 7 |
| Optimized | Talker and Code2Wav | [Full trace](https://github.com/cuzmi/RemoteFiles/blob/428bf60817c2c57e8443982bba591161b0868603/research/minicpm_serving_torch_c32_modal_h100_20261025/artifacts/20261025T145801Z/results/r1-optimized-c32/traces/talker_pid982_rank0.trace.json.gz) | 1,394,930 | 7, 14, 18 |

The filename prefix is the stage that first started a process-wide profiler. `preprocessing` therefore does not mean that the baseline trace excludes Thinker. Stream numbers are local observations and are not yet assigned individually to Flow, HiFT or Talker by verified call-site attribution. Kernel-count differences do not establish matched-work performance differences.

## Stage events recorded during these Torch captures

Mean milliseconds per request, including waiting and execution:

| Version | Thinker | Talker | Code2Wav | E2E | Total generated audio seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 323.25 | 819.26 | 1,451.77 | 2,657.46 | 1,053.72 |
| Optimized | 344.44 | 1,625.19 | 672.63 | 2,689.49 | 1,057.20 |

The same opposing stage direction remains visible: shorter Code2Wav residence and longer Talker residence. Torch profiling materially changes absolute timings and can affect scheduling; these numbers are diagnostic, not a replacement for unprofiled performance measurements.

## Initial kernel/source attribution

| Observed kernel family | Observed launch context | Attribution limit |
| --- | --- | --- |
| GEMM, FlashAttention, fused RMSNorm | Many kernels correlate with cudaGraphLaunch | Exact model/block Python call sites remain unresolved in this low-overhead capture. |
| cuDNN convolution and SDPA kernels | Direct CUDA launches also appear | Kernel names alone do not identify a particular request or Flow/HiFT batch. |

## Overlap and waiting questions

| Observation | What it supports | What remains unresolved |
| --- | --- | --- |
| Talker/Code2Wav uses two active kernel streams in baseline and three in optimized | Inspect the additional stream and its temporal relationship with the existing work | Assign stream roles from launch/call-site evidence before claiming which stages overlap. |
| CUDA copy, launch and synchronization calls are retained | Inspect host-side gaps and synchronization around GPU work | Runtime-call duration can include waiting and overlap across host threads; it is not a direct measure of GPU computation. |
| Thinker is in a separate process on the same GPU | Its activity must be included when inspecting competition | A GPU-kernel busy interval is not SM occupancy or proof of compute saturation. |

## Fusion observations

| Pattern | Evidence status | Action |
| --- | --- | --- |
| Fused residual-add/RMSNorm kernels | Observed kernel names | Already fused; no new fusion proposed. |
| FP8 scaled GEMM | Not established | The automatic analyzer matches some nvjet names to an FP8 family, but this does not prove this capture uses that FP8 dispatch. Do not reuse that heuristic label as a finding. |

The skill-generated kernel/overlap/fusion tables are preserved as `.triage.raw.txt` files under `analysis/`. They are automated hypotheses; their source/fusion labels have not all been verified. Shapes, Python stacks, memory profiling and FLOP estimation were disabled, and no graph-off mapping run was added. This capture is ready for timeline inspection, not a final resource-contention diagnosis.

Native Torch profiling logs a second export attempt after its successful callback export. All four first exports passed gzip/JSON validation and contain GPU activity. No dropped-event or CUPTI-overflow diagnostic was found in the saved server logs.

# Same-graph Code2Wav replay

Run `20261020T171213Z` on commit `917821a`. One H100; A/B each contain the same four fixed real inputs and reuse graph `(4, 576)`.

## Unprofiled performance

| Metric | Serial | Overlap |
| --- | ---: | ---: |
| Mean (ms) | 153.093 | 131.118 |
| Median (ms) | 153.027 | 130.988 |
| Standard deviation (ms) | 1.922 | 0.973 |
| Minimum (ms) | 149.196 | 129.692 |
| Maximum (ms) | 156.243 | 133.871 |

Twenty passes per mode in ten ABBA blocks; 10/10 blocks improved. Mean completion-time reduction: 14.35%.

## Trace component timings

Each span runs from the first GPU kernel start to the last GPU kernel end in that component. The sum adds individual kernel durations. These diagnostic metrics are not the unprofiled completion time.

| Component | Serial kernel sum (ms) | Overlap kernel sum (ms) | Change | Serial GPU span (ms) | Overlap GPU span (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Flow/A/graph | 43.394 | 43.133 | -0.6% | 46.588 | 46.336 |
| HiFT/A | 10.866 | 17.732 | +63.2% | 37.150 | 37.286 |
| Flow/B/graph | 43.089 | 48.622 | +12.8% | 46.288 | 52.752 |
| HiFT/B | 10.842 | 10.984 | +1.3% | 36.852 | 34.260 |

Flow B graph / HiFT A actual kernel intersection: serial 0.000 ms; overlap 15.268 ms.

## Kernel/source attribution

| Region | Source | Kernels per mode |
| --- | --- | ---: |
| Flow/A/graph | `flow_cuda_graph.py:372 run_step -> captured.graph.replay` | 4020 |
| HiFT/A | `code2wav.py:468 decode_mel -> token2wav.hift` | 2180 |
| Flow/B/graph | `flow_cuda_graph.py:372 run_step -> captured.graph.replay` | 4020 |
| HiFT/B | `code2wav.py:468 decode_mel -> token2wav.hift` | 2180 |

## Overlap opportunity

| Ready work | Dependency | Measured evidence |
| --- | --- | --- |
| HiFT A and Flow B | HiFT A waits for Flow A; Flow B is independent of HiFT A | Actual GPU kernel intersections; unchanged input work and grouping |

## Fusion patterns

| Existing execution | Change | Conclusion |
| --- | --- | --- |
| Compiled DiT/HiFT and Flow CUDA Graph | No compiler or fusion change | This experiment evaluates scheduling only |

## Correctness and validity

Configuration, input tokens, references, runtime graph shapes and Flow noise match across the separate unprofiled and trace processes. All formal passes had 20 graph replays, zero misses and no new compilation. A/B graph IDs and graph-kernel signatures match within each trace; corresponding kernel signatures also match across modes.

- unprofiled: 2/2 mel tensors and 7/8 waveforms are bit-exact; maximum waveform absolute error 8.3446503e-05; maximum waveform relative RMS 2.8208318e-05.
- trace: 2/2 mel tensors and 7/8 waveforms are bit-exact; maximum waveform absolute error 2.9753894e-05; maximum waveform relative RMS 7.4407904e-06.

## Limits

- Four unique real samples replayed twice; not eight unique utterances.
- One diagnostic trace per mode; component slowdowns are observations, not repeatability estimates.
- Nsight possible-incomplete-event warnings retained; observed marker, launch, and correlation gates passed.
- No SM, cache, bandwidth, clock or power counters captured; the specific resource causing slowdown is unresolved.
- Isolated ready-input Code2Wav result; no serving throughput claim.

Raw evidence: `trace/timeline.nsys-rep`, `trace/timeline.sqlite`, `trace/analysis.json`, `trace/attributed_kernels.csv`, `unprofiled/passes.json`, `measured_passes.csv`, and `paired_blocks.json`. See the parent `PROTOCOL.md` for preparation and timing boundaries.

# MiniCPM-o fixed-input accuracy: v4, 256 samples

On this fixed-input Code2Wav comparison, all 256 mel outputs and 246/256 waveforms are bit-exact. Corpus WER is identical; mean speaker similarity differs by -0.000545 points. This supports quality preservation on the tested workload, but does not establish universal bit-exact output.

## Results

| Metric | Baseline | Optimized |
| --- | ---: | ---: |
| Successful paired outputs | 256/256 | 256/256 |
| Corpus WER (lower is better) | 2.512746% | 2.512746% |
| Mean WavLM speaker similarity (0–100, higher is better) | 48.461238 | 48.460693 |
| Requests with WER > 50% | 4 | 4 |

| Exactness check | Result |
| --- | ---: |
| Mel byte equality | 256/256 |
| Waveform byte equality | 246/256 (96.09375%) |
| Identical waveform lengths | 256/256 |
| Identical normalized ASR transcripts | 256/256 |
| Maximum waveform absolute difference | 0.365217447 |
| Maximum waveform relative RMS difference | 0.303545332 |

All outputs are finite and have the expected length. The waveforms range from 2.00 to 8.08 seconds at 24 kHz. Bit equality was checked on raw float32 arrays before WAV serialization and independently confirmed on the downloaded NPY files.

## Setup

| Item | Configuration |
| --- | --- |
| Baseline | `078d0dbaabc82da3e67fbdda038257cd9a391051` |
| Optimized | `758405416635738e8812cbddbe170dd94339f5e6` |
| Modal image | `sgl-omni-dev:v4`, published image `im-yYxAITlKItUpEHV2MFm7kC` |
| GPU | One NVIDIA H100 80 GB |
| Runtime | SGLang 0.5.22; Torch 2.14.1+cu130; Transformers 5.19.0 |
| Model | `openbmb/MiniCPM-o-4_5`, revision `503e754207c94da6bb26850b4469f367c9ea3582` |
| Dataset | English `zhaochenyang20/seed-tts-eval-arrow`, revision `27f4c1adee83b5b29b7c4b375f6b976324bda308` |
| Samples | 256 fixed evaluation samples; 32 separate warmup samples, disjoint by ID, target text and reference bytes |
| Input capture | Baseline serving at C16, then freeze the actual Talker codec tokens and reference audio |
| Replay | All 256 inputs ready before native Code2Wav scheduling; default max batch 16, FP16 Flow, DiT/HiFT compile and Flow Graph enabled |
| Randomness | Same Flow noise-buffer hash; HiFT source noise seeded per request, `20261023 + index` |
| WER scorer | Qwen3-ASR-1.7B, revision `7278e1e70fe206f11671096ffdd38061171dd6e5`, greedy decoding, maintained English normalization |
| Similarity scorer | Maintained WavLM speaker scorer; reference audio versus generated audio; checkpoint hashes in `20261023T214600Z/speaker_assets.json` |

The HiFT stochastic source is evaluated per row with matched per-request seeds and concatenated before native decoder execution. This controls noise assignment when batch grouping changes. The native HiFT decoder batching is preserved. Both sides use the same configuration, prompt-feature hashes, Flow noise buffer and per-request source seeds.

## Observed batching and differences

Both sides used the same 16 Flow batches of 16 requests. The configured Flow Graph table has no B16 entry: both executions recorded **0 graph replays and 16 graph misses**. No Flow budget subdivision occurred on these samples. Thus this run covers the compiled non-graph path and does not validate graph-hit correctness or oversized-batch splitting.

Baseline made 237 HiFT calls (219 B1, 17 B2, 1 B3); optimized made 233 (213 B1, 17 B2, 3 B3). Exactly 10 requests changed HiFT batch membership, and these are exactly the 10 non-bit-exact waveforms. The other 246 requests are bit-exact. This associates the differences with HiFT regrouping; it does not by itself isolate the precise numerical cause within HiFT.

All 10 affected requests have WER 0% in both variants. Their mean Similarity delta is -0.013956 points; individual deltas range from -0.541538 to +0.270866 points. Request 99 has the largest waveform discrepancy and Similarity drop, so its waveform change should not be described as negligible rounding even though its transcript is unchanged.

| Index | HiFT group, baseline → optimized | Max waveform abs. diff. | Similarity, baseline → optimized | Similarity delta |
| --- | --- | ---: | --- | ---: |
| 39 | `[39]` → `[39, 50]` | 0.001585905 | 40.830353 → 40.854847 | +0.024494 |
| 50 | `[50]` → `[39, 50]` | 0.000753254 | 59.370632 → 59.369217 | -0.001415 |
| 74 | `[74]` → `[74, 80]` | 0.001096934 | 45.621014 → 45.636089 | +0.015076 |
| 80 | `[80]` → `[74, 80]` | 0.000766244 | 50.789131 → 50.772942 | -0.016190 |
| 93 | `[93]` → `[93, 96, 109]` | 0.000411564 | 56.799171 → 56.773018 | -0.026154 |
| 94 | `[94]` → `[94, 99, 110]` | 0.034254283 | 59.958839 → 60.229706 | +0.270866 |
| 96 | `[96, 109]` → `[93, 96, 109]` | 0.000115171 | 72.316261 → 72.321709 | +0.005447 |
| 99 | `[99, 110]` → `[94, 99, 110]` | 0.365217447 | 54.185780 → 53.644241 | -0.541538 |
| 109 | `[96, 109]` → `[93, 96, 109]` | 0.001446806 | 61.353558 → 61.361336 | +0.007778 |
| 110 | `[99, 110]` → `[94, 99, 110]` | 0.016281670 | 65.133347 → 65.255424 | +0.122078 |

## Interpretation and scope

The tested optimization preserves every Flow mel output and every normalized ASR transcript. Average speaker similarity remains effectively unchanged in this paired sample. There is no measured WER regression, and no broad speaker-similarity regression is apparent. No formal non-inferiority margin was specified, and the largest individual Similarity decrease is reported above rather than hidden by the mean.

This is a frozen-input Code2Wav accuracy check, not an end-to-end serving accuracy test. It removes upstream generation drift and arrival timing; tensor observations synchronize GPU work and may change scheduling, so it does not establish uninstrumented stream race freedom or any performance gain. These results also do not cover graph hits, long-input Flow splitting, other random seeds or the full dataset. No listening-based perceptual evaluation was performed.

## Evidence

See [the result-file index](README.md) and [the original archived report](https://github.com/cuzmi/RemoteFiles/blob/fff0e25b3afd443e02ea682ce30a33a8a81ff145/research/minicpm_accuracy_frozen256_h100_20261023/README.md) for the full provenance.

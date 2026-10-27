# MiniCPM-o open-loop serving results

One H100; 256 fixed measured requests and 128 separate warmup requests per run; no profiler. **main** = baseline `078d0dba`; **branch** = optimized `75840541`. **a/b** are repetitions 1/2; execution order at each rate is main a → branch a → branch b → main b.

Change = (branch / main − 1) × 100%, calculated before rounding. Negative latency and RTF changes mean reductions.

## 8 req/s open loop · 256 requests per run

| Metric | main a | branch a | Change | main b | branch b | Change |
|---|---:|---:|---:|---:|---:|---:|
| Request throughput (req/s) | 7.891 | 7.885 | -0.07% | 7.893 | 7.890 | -0.03% |
| Mean RTF | 0.1605 | 0.1557 | -2.98% | 0.1612 | 0.1534 | -4.85% |
| Mean latency (ms) | 651.5 | 633.0 | -2.84% | 652.8 | 621.4 | -4.81% |
| Median latency (ms) | 643.7 | 630.0 | -2.12% | 649.2 | 619.0 | -4.64% |
| p95 latency (ms) | 974.1 | 951.2 | -2.35% | 980.9 | 913.2 | -6.90% |
| p99 latency (ms) | 1,044.1 | 1,050.9 | +0.64% | 1,070.1 | 1,035.7 | -3.22% |
| Audio throughput (audio s/s) | 32.677 | 32.774 | +0.30% | 32.652 | 32.662 | +0.03% |
| Generated audio (s) | 1,060.16 | 1,064.08 | +0.37% | 1,059.04 | 1,059.72 | +0.06% |

## 14 req/s open loop · 256 requests per run

| Metric | main a | branch a | Change | main b | branch b | Change |
|---|---:|---:|---:|---:|---:|---:|
| Request throughput (req/s) | 13.535 | 13.505 | -0.22% | 13.542 | 13.522 | -0.15% |
| Mean RTF | 0.2170 | 0.2121 | -2.27% | 0.2207 | 0.2121 | -3.91% |
| Mean latency (ms) | 874.3 | 845.8 | -3.26% | 890.5 | 859.2 | -3.51% |
| Median latency (ms) | 862.2 | 838.3 | -2.78% | 881.9 | 867.0 | -1.68% |
| p95 latency (ms) | 1,253.9 | 1,245.2 | -0.70% | 1,280.5 | 1,248.2 | -2.53% |
| p99 latency (ms) | 1,395.3 | 1,420.2 | +1.79% | 1,444.3 | 1,375.6 | -4.76% |
| Audio throughput (audio s/s) | 55.965 | 55.361 | -1.08% | 56.109 | 56.434 | +0.58% |
| Generated audio (s) | 1,058.52 | 1,049.40 | -0.86% | 1,060.68 | 1,068.44 | +0.73% |

## 20 req/s open loop · 256 requests per run

| Metric | main a | branch a | Change | main b | branch b | Change |
|---|---:|---:|---:|---:|---:|---:|
| Request throughput (req/s) | 18.676 | 17.807 | -4.65% | 18.492 | 18.054 | -2.37% |
| Mean RTF | 0.3022 | 0.2900 | -4.04% | 0.2898 | 0.2782 | -4.00% |
| Mean latency (ms) | 1,222.7 | 1,193.7 | -2.37% | 1,184.1 | 1,144.3 | -3.36% |
| Median latency (ms) | 1,182.3 | 1,172.8 | -0.80% | 1,177.7 | 1,130.5 | -4.01% |
| p95 latency (ms) | 1,859.0 | 1,671.8 | -10.07% | 1,619.6 | 1,651.9 | +1.99% |
| p99 latency (ms) | 2,152.1 | 2,014.5 | -6.39% | 1,980.8 | 1,964.1 | -0.84% |
| Audio throughput (audio s/s) | 78.669 | 77.545 | -1.43% | 79.720 | 78.045 | -2.10% |
| Generated audio (s) | 1,078.36 | 1,114.84 | +3.38% | 1,103.64 | 1,106.68 | +0.28% |

## 30 req/s open loop · 256 requests per run

| Metric | main a | branch a | Change | main b | branch b | Change |
|---|---:|---:|---:|---:|---:|---:|
| Request throughput (req/s) | 24.940 | 23.962 | -3.92% | 25.122 | 22.808 | -9.21% |
| Mean RTF | 0.4903 | 0.5462 | +11.41% | 0.4743 | 0.5417 | +14.22% |
| Mean latency (ms) | 1,964.8 | 2,210.4 | +12.50% | 1,896.9 | 2,180.3 | +14.94% |
| Median latency (ms) | 2,004.1 | 2,278.4 | +13.69% | 1,966.0 | 2,344.6 | +19.26% |
| p95 latency (ms) | 2,748.8 | 3,277.6 | +19.24% | 2,612.8 | 3,175.6 | +21.54% |
| p99 latency (ms) | 3,069.3 | 3,570.6 | +16.33% | 2,940.8 | 3,318.8 | +12.86% |
| Audio throughput (audio s/s) | 103.342 | 99.734 | -3.49% | 104.556 | 94.078 | -10.02% |
| Generated audio (s) | 1,060.76 | 1,065.52 | +0.45% | 1,065.44 | 1,055.96 | -0.89% |

Request throughput is completed requests divided by total cohort wall time, including the final drain; fixed offered rates constrain it, so these figures are not sustainable-capacity estimates. Mean RTF averages each request's latency/audio-duration ratio. Generated audio can differ between runs. Requests are nonstreaming; first-audio and inter-chunk latency were not measured.

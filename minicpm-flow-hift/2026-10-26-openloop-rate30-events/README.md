# 30 requests/s event diagnostics

One baseline/optimized pair; extra Talker time before first forward is mainly after queue entry. Event boundaries do not identify the causal scheduler activity or GPU dependency.

## Reports

- [EVENT_REPORT.md](EVENT_REPORT.md)
- [PRE_PREFILL_BREAKDOWN.md](PRE_PREFILL_BREAKDOWN.md)

This directory contains selected results, logs and measured configuration evidence. Raw result bytes are unchanged; report links to omitted artifacts use immutable archive URLs.

[Full archive and reproduction material](https://github.com/cuzmi/RemoteFiles/tree/955ebb1582d6c003e35b9c198ae7c9502991d259/research/minicpm_serving_openloop256_rate30_event_radix_h100_20261026). Large traces and analysis files are listed in [external evidence](../EXTERNAL_EVIDENCE.json). Historical preparation manifests remain in RemoteFiles; this branch uses its own SHA256SUMS.json.

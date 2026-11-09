# Realtime caption and response display — 2026-11-07

The orb caption now displays the latest received text in its two-line area. The Qwen reply panel stays within 70dvh and permits internal scrolling while preserving the reader's position as text arrives. These results document the display fixes associated with [PR #13](https://github.com/cuzmi/sglang-omni/pull/13).

| Browser fixture | Before | After | Coverage |
| --- | --- | --- | --- |
| Orb caption | 7/12 passed | 12/12 passed | English/Chinese at 600px and 320px, latest text, full transcript, clearing, short reply reset, resize after streaming |
| Qwen wide viewport | 20/32 passed | 32/32 passed | 1280 × 757; English/Chinese, text and text + audio presentation modes, bounded height, scroll position and full reply retention |
| Qwen narrow iframe | 20/32 passed | 32/32 passed | 390 × 844 iframe in desktop Chrome; the same display checks |

[Orb comparison](orb/comparison.json), [before results](orb/before-results.json), [after results](orb/after-results.json) and [Qwen results](qwen/manual-results.json) retain both baseline failures and passing results. The [before screenshot](orb/before-browser.png) and [after screenshot](orb/after-browser.png) provide the saved orb browser views.

These are synthetic DOM/layout checks, not model accuracy, GPU performance or speech-alignment measurements. Qwen's text + audio fixture exercises the text event path; it does not synthesize or play audio and cannot establish that playback completes. The narrow layout is an iframe, not a physical phone. The orb fixture extracts rendering logic and CSS; the Qwen fixture embeds the page and injects server events. HTML fixtures contain executable test/frontend code and are available only in the [fixed RemoteFiles archive](https://github.com/cuzmi/RemoteFiles/tree/eda7843a69336e507275e90e3380aeb8af99782d/research/realtime_display_20261107). No browser or GPU tests were rerun for this publication.

Both archived pre-commit logs have one failed hook because Cargo was unavailable for Rust formatting; all other hooks passed. See [orb log](orb/pre-commit-python312.log) and [Qwen log](qwen/pre-commit-display-python312.log).

Source revisions are orb baseline 67476823ae81b398626b85e368d1c6f11a832496, orb fix/Qwen baseline 2b57b14060896a1a2686dab48e3219b86672d3bd, and final display fix 4910ba41667b538329912fd2ce5ca8b3e6ee0c7b. [MiniCPM source metadata](modal/minicpm/source.json) and [Qwen source metadata](modal/qwen/source.json) retain the historical bundle and frontend hashes. The Qwen after-demo index.html hash differs because deployment rewrote asset URLs; CSS, app and playback hashes match the final source revision, as recorded in the source archive report.

[MiniCPM readiness](modal/minicpm/ready-verification.json), [WebSocket smoke](modal/minicpm/websocket-verification.json), [Qwen before readiness](modal/qwen/ready-verification.json), [after readiness](modal/qwen/after-ready-verification.json) and [page reopen](modal/qwen/reopen-verification.json) are historical deployment receipts. MiniCPM's WebSocket smoke sent no microphone audio. Deployment addresses and expiry values are historical, not live service pointers. [Closeout](CLOSEOUT.json) records both demo apps stopped with zero active tasks on November 7; this is not a live status check.

This directory contains results, screenshots, logs and provenance only. HTML fixtures, original archive manifests and the full narrative with user recording links remain in the [immutable source archive](https://github.com/cuzmi/RemoteFiles/tree/eda7843a69336e507275e90e3380aeb8af99782d/research/realtime_display_20261107). Raw access-token logs, source bundles, browser caches, media recordings and the deferred audio-stop investigation are excluded. PROVENANCE.json maps every copied file to its exact source and SHA256. The branch-level SHA256SUMS.json includes all published files except itself.

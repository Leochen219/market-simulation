# Architecture

The `capclaims.domain` modules implement auction, settlement and integer accounting. `agents` defines observations, candidates and rules. `engine` coordinates the legacy single-order market and replay. The independent `paper` profile supports procurement, up to five mixed orders, grouped agents, clearing, settlement and replay. `backends` supplies rule, local Laya and remote Jev adapters. `cli` selects the profile and writes run artifacts.

The research modules implement standalone prediction and evaluation paths: full-order portfolio prediction, conditional behavior and behavioral memory, and later fixed-direction one-order grid and long-history diagnostics. These prediction studies do not perform market clearing. A submitted order is an intention; only matching and settlement change holdings and cash. Numeric empty-support STOP is a separate client policy and is not model-initiated STOP.

A completed synthetic paper-profile run can be reconstructed with `paper-replay` without model inference. Replay checks recorded events; hashes detect accidental changes but are not cryptographic signatures. See the quickstart in the [README](../README.md). Full historical observations, raw requests and participant data are not included in this release.

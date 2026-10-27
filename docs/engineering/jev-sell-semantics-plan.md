# SELL Semantics Implementation Plan

> **For agentic workers:** Use subagent-driven-development for scoped implementation; no subagent may delegate. Root integrates, verifies and owns live execution.

**Goal:** Run a frozen64-request paired SELL-price intervention and compare behavior distributions stratified by personal SELL history.

**Architecture:** Add an isolated `sell_semantics` study; reuse immutable flat-grid and transport helpers without modifying frozen baseline files. Separate deterministic request construction, one-shot runner/replay and scoring. Keep raw evidence private.

**Tech Stack:** Existing Python3.11, pinned TypeSafe SDK, asyncio, SQLite cost ledger, pytest; existing Matplotlib for public CDFs.

**Spec:** `docs/research/jev-sell-semantics-design-2026-10-27.md`.

## Global constraints and rulings

- Existing user approval covers the neutral-definition proposal and real small comparison; do not add a redundant approval question.
- Quota routing remains paused by user/project AGENTS. Use local Terra-high implementation and fresh local Sol-high review; prior Opus/Astra semantic advice remains evidence, not execution authority.
- Original private research feature branch is reused additively because frozen untracked source/evidence is required; no old source is changed. Safe publication uses the existing separate codex/flat-price-grid5 checkout.
- Same eight SELL people,16 contexts,32 paths, two contemporaneous arms64 calls, USD0.25 hard cap,10inflight,40RPS,50k reserved bytes/s,12k body,600s/batch,no retries,no20market.
- Root primary metric is pooled person-balanced W1; an adviser's proposed mean per-person primary is not adopted because the user selected behavior distributions. Per-person errors are auxiliary.

## Review focus

- ORIGINAL exactly reproduces old bytes; RULES adds only one field, even at longest ten-round history.
- Zero personal SELL history is derived before target access; seven versus one is not a subgroup causal test.
- Empty supports remain abstentions, invalid probabilities/unknown fees halt and retain reservations.
- Arm and completion ordering cannot change draws or pair identities.
- Strict replay catches body/response/ledger tampering; partial pairs cannot become empty orders or be replaced by old responses.

## Tasks

- [x] Worker: implement `src/capclaims/sell_semantics/{__init__,design,runner}.py`, `tools/probes/jev_sell_semantics.py`, and focused synthetic tests. `replay` returns `(plan,predictions,status)`; predictions carry person/formal/window/repeat/arm/fixed_quantity/price/outcome/raw_probabilities/decision. No live calls or edits to frozen files.
- [x] Root: implement `src/capclaims/sell_semantics/analysis.py` and synthetic distribution/stratum/coverage tests; render reports with independent tools outside frozen primary code.
- [x] Worker/root: run appropriate new and shared-regression tests, offline all64-request preview and independent pre-live review. Freeze final code and inputs only after analysis exists.
- [x] Root: obtain independent PASS on frozen hashes/body/budget contract; execute exactly once; strict replay and independent arithmetic review. No automatic restart if partial.
- [ ] Root: deliver safe report/figures and prompt, update original Notion, mirror whitelisted public files, verify publication tests and push existing PR branch. Record outcomes and limitations; keep private original evidence.

## Progress

2026-10-27: accepted proposal recorded, scope/rulings fixed, prior baseline preserved. Implementation not yet run; no calls in this study.

2026-10-27 execution: COMPLETE64; strict replay and independent frozen-body/sampling/arithmetic review PASS. Short ALL W1 21.875->19.0625, long16.5625 unchanged; all64quotes, no technical failure. Full subgroup data private; public single/complement suppression verified. Original Notion homepage updated. Public targeted58PASS/6private-evidenceSKIP. Safe publication prepared; final remote commit verification remains the delivery step.

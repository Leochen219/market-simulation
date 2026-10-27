# Flat 5-dollar Price Grid Implementation Plan

> For agentic workers: implement this scoped plan with tests first; do not run a live model experiment or change frozen historical protocols.

**Goal:** Replace hierarchical price selection in a new diagnostic version with one Choice over 199 complete unit prices: 5, 10, ..., 995 dollars.

**Architecture:** Add a separate flat-grid protocol and a runnable offline preview/mock/replay entry. Reuse the existing generic top-3/raw-3% decoder and validated probability contract. Preserve the historical 2.5-dollar hierarchical studies; the new single-order diagnostic takes a supplied side and selected quantity, with no band request or band state.

**Tech stack:** Python 3.11, the frozen TypeSafe SDK serialization and existing bounded client interface, pytest.

**Spec:** User request on 2026-10-27: change the grid spacing to five dollars so price bands are unnecessary. This document records routine implementation choices, not a request to run a new study.

## Constraints and acceptance

- Quantity remains 10..990 in steps of ten; cumulative legal SELL capacity remains bounded by initial inventory and existing submitted SELL intentions.
- Price is a complete per-unit limit price in dollars, minimum 5, maximum 995, with exactly 199 numeric choices. No zero, 1000, band stage, inferred price cap or history-derived candidate subset.
- Candidate IDs must map explicitly to dollar values. Never reuse the old tick-index-times-2.5 conversion for this version.
- State is copied, contains the full supplied history and ledger, and carries the selected quantity to price. Strip obsolete band state and replace stale 2.5-dollar instructions. Submitted intentions do not change actual Q/D or realized profit.
- Client top-3/raw-3% sampling, deterministic seeds and candidate-order tie breaking remain explicit. Valid empty support stops appending and discards the current draft; invalid probabilities and transport errors remain errors. No explicit model STOP choice is added by this numeric-only change.
- Distinct agents can run concurrently with a ten-request limit; each agent's quantity precedes its price. Record requests, raw probabilities, decoded selections and failures for strict offline replay.
- Use the existing 255-option and 12,000-byte checks. Fail before sending rather than truncating history or silently increasing caps.
- No new Jev calls, no market experiments, no banks/credit, no reinterpretation of old branch probabilities as a new flat distribution.

## Implementation task

Files:
- Add `src/capclaims/flat_price_grid.py`: candidate/state construction, validated decoding, two-step async prediction, batch limit and replay.
- Add `tools/probes/jev_flat_grid5.py`: synthetic preview, SDK mock run and strict replay; no automatic live run.
- Add `tests/probes/test_jev_flat_grid5.py`: grid coverage, stale-band removal, state forwarding/immutability, units, boundary/cutoff/error handling, strict replay and ten distinct asynchronous paths.
- Update public documentation to distinguish the new flat protocol from the preserved historical two-stage protocol.

Steps:
- [x] Write focused failing tests and verify the failure.
- [x] Implement the minimal flat protocol and runnable offline entry.
- [x] Verify SDK request shape/size and ten-way mocked concurrency; verify replay rejects tampering.
- [x] Root checks all authorized local short/long history contexts fit without modification, without sending them externally.
- [x] Run appropriate regression checks and obtain independent review.
- [x] Mirror additive files into the private research workspace, leaving frozen dependencies intact; prepare the reviewed GitHub publication branch.

## Interpretation

The new protocol changes both grid spacing and generation procedure. No improvement in human fidelity is established by implementation or mocked tests. Fresh frozen model calls would be required to compare behavior. Old hierarchical results and failures keep their original labels and hashes.

## Completed offline verification

- Full regression: 585 passed, 13 private-data/evidence skips, 12 subtests passed.
- Focused new/decoder/abstention regression: 24 passed; independent new-protocol review: PASS.
- Synthetic preview/mock/replay: ten concurrent paths, twenty requests, replay passed.
- Local serialization of 18 participants and 36 saved short/long history states: 72 bodies, maximum 10,250 bytes (short 7,380; long 10,250), all histories retained, 199 prices per price request. Actual states and payloads remain private; no Jev calls were made.
- The 23 frozen dependencies of the prior long-history experiment remain unchanged.
- Replay checks internal consistency; coordinated rewriting of all evidence requires an independently trusted hash to detect.

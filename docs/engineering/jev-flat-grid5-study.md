# Matched flat-grid price study

This study compares a direct $5 price grid with the earlier hierarchical price procedure. It fixes each of the 72 baseline paths' participant, history, side, and previously selected quantity; it makes one price request per path. The comparison changes the choice hierarchy and grid together, so it does not isolate grid spacing. Private baseline artifacts and participant labels must already exist locally and pass strict baseline replay. They are not included in this repository.

From the repository root, use a Python environment with the project's dependencies installed and run `python -m tools.probes.jev_flat_grid5_study ACTION`, where ACTION is:

- `preview`: verify the private baseline and construct and size the 72 requests offline.
- `freeze`: create the private frozen plan, labels and body hashes in the study output directory. An existing output directory is never overwritten.
- `run`: verify the frozen contract before loading the local provider credential and make the one-shot bounded run. It permits at most 72 requests and reserves against a $0.25 budget. A started run cannot be resumed or automatically retried; technical failures halt new dispatch.
- `replay`: verify the frozen inputs, recorded wire evidence, decoder and ledger; incomplete technical runs remain partial.
- `report`: build aggregate comparisons from replayed private inputs. Keep participant records and wire evidence private.

The tests use synthetic inputs and mock transports. If the project's normal Python environment is unavailable, use a compatible Python 3.11 environment with the pinned project dependencies installed; select it through your shell or environment manager before running the commands.

## Recorded 2026-10-27 result

The authorized price-only run completed 72/72 actual Jev calls. Strict replay passed; no new abstentions or technical failures occurred. The measured HTTP peak was 7 under the unchanged limit of 10; elapsed 17.969 seconds, input 463,444 tokens, frozen-price cost USD 0.019464648. Prices >=700 fell from 15 old paths to zero new paths, with full new quote coverage. SELL prices still differ from humans. See [the aggregate report](../analysis/jev-flat-grid5-comparison-2026-10-27.md), its JSON, full and zoomed CDFs, and post-hoc tail audit. This is a development conditional-price result, not a complete market policy.

Read-only local verification, after installing the package in the selected Python environment:

```powershell
python -m tools.probes.jev_flat_grid5_study replay
python -m tools.probes.jev_flat_grid5_study report
python -m tools.analysis.flat_grid5_tail_diagnostic
python -m tools.analysis.report_jev_flat_grid5
```

The report renderer also requires Matplotlib. Running it from already-published aggregate JSON does not need private data or an API key. Replay and the tail audit need the local private frozen evidence. Do not run `freeze` or `run` again against the completed output.

Publication verification: 597 tests passed, 13 private-data/evidence tests skipped in 393.91 seconds; no failures or errors. The 47-test pre-live subset, independent frozen-body check, strict actual-wire replay, independent exact-weight W1 recomputation, and aggregate privacy-field audit also passed.

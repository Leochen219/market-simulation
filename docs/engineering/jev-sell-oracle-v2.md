# SELL oracle version 2

Version 2 uses the existing oracle design and source with deterministic recursive key ordering of the model state before SDK serialization. Candidate order and history arrays are unchanged. The v1 attempt halted before any physical request because frozen plan serialization changed nested historical-order key order; v2 validates serialized bodies again from the saved and reloaded plan for every query. No request is permitted before that validation. Preparation mismatches have private non-wire evidence and remain incomplete. Version 2 has a separate one-shot marker, ledger and output directory; the failed version remains preserved.

The previous USD 0.30 ceiling, 106-request cap, pinned SDK 0.7.5, model and sampling contracts remain in force. Freeze is an offline preparation step. Live execution requires authorization for the new version. This is a human-quantity oracle diagnostic, not a deployable full agent.

## Study coverage and interpretation

The unchanged study specification is [the expanded oracle design](../research/jev-expanded-sell-oracle-design-2026-10-28.md). It screens all 35 previously exposed GA10 participants over formal rounds 11–15 using current Q>D only: 76 eligible person-rounds from 30 people. Of these, 71 events from 28 people contain 106 SELL components. Five events without SELL remain in coverage. Deduplication of the same person's same-round identical quantities yields 101 physical requests, not 106 independent human observations. The primary price analysis uses the 49 single-SELL events. Multiple-SELL events are reported separately.

Each request uses the previous ten true rounds, the current p/w/Q/D, and the exact human SELL quantity. The target price is scoring-only. This diagnoses price prediction conditional on human quantity; it does not evaluate a deployable complete agent or prove that quantity mismatch caused earlier errors. The actual retained price distribution is raw top-3 with 3% cutoff over all 199 prices $5..$995. Raw human prices are the primary reference, with grid projection as sensitivity. No market simulation is run.

## Serialization repair and acceptance evidence

The previous plan SHA256 is `086e50f34c61929bd04d4ed3e9462eb53de6b030bd08c8541d3b3115199f0624`. Its first attempted batch stopped during request preparation: zero ledger attempts, zero wire requests, and ten pre-dispatch technical records. No Jev inference cost was incurred. All 39 frozen code-file hashes remain unchanged. The failure is not a model result.

Version 2 canonicalizes object keys in model state before serialization, while keeping array order and candidate order unchanged. It reconstructs every request from the saved/reloaded plan and compares its bytes before execution. It also records a future preparation mismatch under `preparation-evidence`, distinct from posted wire evidence. Strict replay verifies the intended request, constructed mismatch, zero-dispatch status and retained evidence.

Offline acceptance includes the v1 failure reproducer, 101/101 real-source request round trips, a full synthetic freeze/reload/run/replay cycle with genuinely different state/history inputs, ten simultaneous requests held at a transport barrier, no-key and client-start failures, unknown fee retention, invalid probabilities, duplicate component multiplicity, replay tampering and fixed sampling rules. Synthetic responses are not evidence of Jev price behavior.

## Local commands

From `C:\Users\hongy\Desktop\Plan\ai-market-v1`, use the approved Python 3.11 runtime; the venv executable may be blocked by local application control. This wrapper imports the existing environment without changing it:

```powershell
@'
import sys, runpy
sys.path[:0] = [r'.venv-m0/Lib/site-packages', 'src', '.']
sys.argv = ['jev_sell_oracle_v2', 'preview']
runpy.run_module('tools.probes.jev_sell_oracle_v2', run_name='__main__')
'@ | & 'C:/Users/hongy/AppData/Roaming/uv/python/cpython-3.11-windows-x86_64-none/python.exe' -X utf8 -
```

`preview` validates source policy and constructs all payloads offline. Replace `preview` with `freeze` once to save the private plan and labels, then validate its reload; the output folder must be new. After explicit approval of this replacement execution, `run` performs one live execution using the local `.env`. `replay` independently validates its evidence. `report` writes private detailed analysis and privacy-filtered aggregate JSON, and refuses a complete report for a technical-partial run. Never delete the one-shot `started` marker, resume the failed v1, retry provider calls, or expand the budget automatically.

Run the focused offline checks without provider credentials:

```powershell
@'
import sys, runpy
sys.path[:0] = [r'.venv-m0/Lib/site-packages', 'src', '.']
sys.argv = ['pytest', '-p', 'no:faulthandler',
            'tests/probes/test_jev_sell_oracle_v2.py',
            'tests/probes/test_jev_sell_oracle_analysis.py', '-q',
            '--basetemp=artifacts/pytest-sell-oracle-v2-local']
runpy.run_module('pytest', run_name='__main__')
'@ | & 'C:/Users/hongy/AppData/Roaming/uv/python/cpython-3.11-windows-x86_64-none/python.exe' -X utf8 -
```

The new plan freezes this guide along with its code dependency closure. Store later execution/review results separately; do not edit frozen inputs. Keep participant inputs, labels, raw probabilities, component-level records and small subgroups local. Only safe summaries belong on the existing Notion homepage and GitHub.

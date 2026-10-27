# Controlled three- versus ten-round history study

The authorized development-only target is formal round 11 (game round 16), for the same 18 people and observed target state. `short3` includes formal rounds 8–10 (game 13–15); `long10` includes formal rounds 1–10 (game 6–15). Source accounting and policy/registry hashes are rechecked; target orders and profits are kept in a separate private scoring file. There is no target order or future outcome in an input. Conditions share the same current observation and fixed candidate/prompt rules; only their history and derived M1 facts differ. Equal Q and D are excluded before side selection; current audited source has 10 BUY, 8 SELL and no exclusions. No held-out people are used.

Each condition has two repeats. Maximum 72 paths × (quantity + band + four price branches) = 432 requests. A separate USD 1.50 ledger reserves 65,536 × 42 nanodollars per request; maximum reserved USD 1.189085184. The shared transport enforces the existing size, concurrency, rate, timeout and zero-retry controls. A missing 3% quantity skips downstream calls on that path; other empty supports are logged per decision and do not halt unrelated paths. Invalid probabilities, transport failure or ambiguous usage halt new dispatch. The one-time start marker prevents automatic restart.

The 3% quantity conditions all four queried price branches. Thus 3% is the complete pipeline; argmax and 5% price controls are conditional on the 3% quantity, not separate full pipelines. Draw coordinates include person, target round, repeat and stage but omit history condition, yielding common draws; request IDs and states retain condition identity. An exact shared draw does not make the service responses identical. Short and long requests must both fit the 12,000-byte cap without truncation; an offline SDK preview on 36 contexts found short maximum 6,373 bytes and long maximum 9,243 bytes.

Primary numeric human comparisons use each person's observed target orders on the given side and balanced person/order or person/repeat weights. Quantity and 3% price comparisons use their respective two-history common-success masks; full eligible price bounds show the impact of missing predictions. This evaluates distributions, not individual action matching, calibration or direction prediction. The secondary six-way price mask compares both conditions and three decoders, with argmax and 5% labelled conditional. Last same-side and nearest-state historical baselines preserve all side-matching orders in the chosen round; historical SELL totals exceeding current inventory are ineligible. Missing candidates remain missing, and baseline/model W1 is computed on the same covered people. Historical order components differ from a single model order.

After independent review, from the repository root:

```powershell
.venv-m0/Scripts/python.exe tools/probes/jev_long_history.py preview
.venv-m0/Scripts/python.exe tools/probes/jev_long_history.py freeze
.venv-m0/Scripts/python.exe tools/probes/jev_long_history.py run
.venv-m0/Scripts/python.exe tools/probes/jev_long_history.py replay
.venv-m0/Scripts/python.exe tools/probes/jev_long_history.py report --output docs/analysis/jev-long-history-v1-report.json
```

Preview is read-only; freeze makes a new private plan, separate target file, schedule and code/source hashes. A pre-existing directory is never overwritten. Replay verifies plan and source, chain identity, candidate order, wire evidence, request condition, probability decoding, attempt/timing evidence and settled cost ledger. Technical partial runs are not reported as completed comparisons. The output directory remains private; only aggregates belong in public reports.
The comparison estimates the total effect of supplying a longer window under this fixed interface. Earlier rounds may contain initial-learning information as well as person-specific patterns; improvement would not by itself establish personalization, and no improvement would not establish that humans are unusual or that the model is incapable. The pooled-history baseline uses legal same-side components, weighted per person then historical round then order; it uses a common covered person set across windows and model outcomes and never replaces missing histories with a cross-person fallback.

# SELL semantics local commands and verification

Run from the repository root in a Python3.11 environment with the project and its Jev SDK dependencies installed. Use module invocation so research helpers under `tools` resolve correctly:

```powershell
python -m tools.probes.jev_sell_semantics preview
python -m tools.probes.jev_sell_semantics replay
python -m tools.probes.jev_sell_semantics report
python -m tools.analysis.report_jev_sell_semantics
```

The completed study already has its one-time start marker. Do not run `freeze` or `run` again for that directory. Preview/replay/report are offline; only `run` can load the local key and call Jev. No key or private data is needed to render the published aggregate report using the optional Matplotlib analysis dependencies.

If the old local virtual-environment executable is blocked by Windows application control, use an approved compatible Python3.11 interpreter. For the existing research environment, its installed libraries can be selected in that shell before module invocation:

```powershell
$env:PYTHONPATH = "src;.;.venv-m0/Lib/site-packages"
# Invoke the approved Python3.11 interpreter with -m tools.probes.jev_sell_semantics replay.
```

Full local subgroup output is `artifacts/participant-data/jev-sell-semantics-v1-2026-10-27/comparison-private.md`, generated from strictly replayed `analysis-private.json`. It includes the one-person no-SELL-history group and the seven-person complement; do not publish those files. Public JSON and figures show only the ALL eight-person numeric distributions and stratum counts.

Pre-live evidence:63 shared/new regression tests passed;17 final new tests passed after lifecycle/privacy updates; the final eight analysis tests passed again after removing the public per-person history-count vector. Independent review verified the frozen64 bodies, each original body matching the old request and each intervention adding only one definition field. Completed wire, ledger, sampling, distributions and arithmetic were independently replayed and recomputed. A report wording correction distinguishes unchanged W1 from unchanged CDF; the frozen experiment is unchanged.

The contemporaneous control is re-called rather than reusing the previous flat5 output. Interpret these as development results for fixed old model quantities, not a complete market policy or an independently confirmed digital human.

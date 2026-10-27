# AI Capacity Claims Market

A research simulator for capacity procurement and secondary trading, with deterministic accounting, replay, baseline policies, and optional model backends. It contains a legacy single-order market and a paper-aligned profile with up to five mixed orders per participant. This is a research implementation and does not establish human-equivalent behavior or reproduce all findings of the paper.

## Offline quickstart

Python 3.11 or newer is required. The following synthetic baseline uses CPU and requires no API key or model weights. Install dependencies once while online; the simulation itself runs offline. From a shell on Windows, macOS, or Linux:

```sh
git clone https://github.com/Leochen219/market-simulation.git
cd market-simulation
python -m venv .venv
```

Activate with `.venv\Scripts\Activate.ps1` in PowerShell or `source .venv/bin/activate` in a POSIX shell, then:

```sh
python -m pip install -e ".[dev]"
python -m capclaims.cli paper-run --config configs/paper-random-legal-ga-w10.yaml --output artifacts/runs/synthetic-001 --device cpu
python -m capclaims.cli paper-replay --run artifacts/runs/synthetic-001
python -m capclaims.cli paper-summarize --run artifacts/runs/synthetic-001
python -m pytest tests -q
```

Choose a fresh output directory for each run. The example is a synthetic random-legal policy with 10 agents and 20 rounds, including five warmup rounds. Generated logs can contain private observations; `artifacts/` is ignored. The dev extra installs the small SDK and spreadsheet dependencies used by offline tests, without GPU PyTorch. Tests use synthetic fixtures and do not require live credentials. Ten tests marked `requires_participant_data` skip when the private participant source, authorization policy, or exposure registries are absent; synthetic and other data-free tests still run. Research tooling may also require private inputs or evidence not distributed here. The full suite has not yet been verified in a fresh environment. Frozen design and cost-gate documents preserve historical experimental context and are not authorization for new API calls or data use.

Optional Laya dependencies can be installed with `python -m pip install -e ".[laya]"`; model weights and a suitable PyTorch installation are separate. The Jev backend requires `python -m pip install -e ".[jev]"` and explicit provider configuration. API calls are opt-in and may incur charges. The optional `analysis` extra installs matplotlib for plotting tools; original study data needed to regenerate some historical figures is not included. `.env.example` contains empty placeholders; keep your own `.env` private. Do not enter real participant data without appropriate authorization.

## Research and scope

The full-order J1 interface generates BUY/SELL/STOP one order at a time; J2 chooses a count first and then generates up to five orders. Recent grid diagnostics instead fix direction and one order, then sample quantity and price; they do not predict direction, count, or model-initiated STOP. A separate client-side empty-support STOP policy has only been evaluated by offline replay and is not integrated into a live multi-order market. These are different action spaces and should not be compared as interchangeable complete strategies.

Longer personal history improved some quantity distributions in the evaluated development sample, while a substantial price gap remained. Results are not evidence of calibrated human behavior. Recent Jev diagnostics did not run a new 20-round Jev market. See [current research summary](docs/current-state.md) and [architecture](docs/architecture.md).

Source code licensing has not yet been selected. Participant data, original logs, private policies, and model weights are not redistributed.

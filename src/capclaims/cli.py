"""Command line entry points for running, replaying, and summarizing sessions."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from capclaims.backends.jev import JevBackend
from capclaims.backends.laya import LayaBackend
from capclaims.backends.rules import RuleBackend
from capclaims.config import RunConfig
from capclaims.engine.replay import replay
from capclaims.engine.session import run_session
from capclaims.paper.analysis import summarize_events
from capclaims.paper.baselines import PaperBaselineBackend
from capclaims.paper.config import PaperConfig
from capclaims.paper.replay import replay_paper
from capclaims.paper.session import run_paper_session


def _backend(config: RunConfig, model_path: str | None, device: str) -> Any:
    if config.policy == "laya":
        return LayaBackend(model_path or ".cache/models/laya-multilingual", device)
    if config.policy == "jev":
        return JevBackend()
    return RuleBackend(config)


def _summary(run_dir: Path, result: Any | None = None) -> dict[str, Any]:
    result = result or replay(run_dir)
    return {
        "status": getattr(result, "status", "COMPLETE"),
        "rounds": result.rounds,
        "trades": result.trades,
        "volume": result.volume,
        "profit_cents": result.profit_cents,
        "final_hash": result.final_hash,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="capclaims", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run one session")
    run.add_argument("--config", required=True, type=Path)
    run.add_argument("--output", required=True, type=Path)
    run.add_argument("--model-path", default=None)
    run.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    run.add_argument("--wave-timeout", type=float, default=60.0)
    replay_parser = commands.add_parser("replay", help="recompute and verify a completed run")
    replay_parser.add_argument("--run", required=True, dest="run_dir", type=Path)
    summarize = commands.add_parser("summarize", help="summarize a verified completed run")
    summarize.add_argument("--run", required=True, dest="run_dir", type=Path)
    paper_run = commands.add_parser("paper-run", help="run the versioned paper-aligned profile")
    paper_run.add_argument("--config", required=True, type=Path)
    paper_run.add_argument("--output", required=True, type=Path)
    paper_run.add_argument("--model-path", default=None)
    paper_run.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    paper_run.add_argument("--wave-timeout", type=float, default=120.0)
    paper_replay = commands.add_parser("paper-replay", help="verify a schema-2 paper-profile run")
    paper_replay.add_argument("--run", required=True, dest="run_dir", type=Path)
    paper_summary = commands.add_parser("paper-summarize", help="summarize verified formal paper rounds")
    paper_summary.add_argument("--run", required=True, dest="run_dir", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    import os
    if os.environ.get("CAPCLAIMS_DIAGNOSTICS") == "1":
        import faulthandler
        faulthandler.dump_traceback_later(30, repeat=True)
    args = build_parser().parse_args(argv)
    try:
        if args.command == "run":
            config = RunConfig.load(args.config)
            backend = _backend(config, args.model_path, args.device)
            result = asyncio.run(run_session(
                config, backend, args.output,
                wave_timeout_seconds=args.wave_timeout,
            ))
            print(json.dumps(_summary(args.output, result), sort_keys=True, indent=2))
            manifest = json.loads((args.output / "manifest.json").read_text(encoding="utf-8"))
            if manifest.get("observed_batch_sizes"):
                print(json.dumps({"observed_batch_sizes": manifest["observed_batch_sizes"]}, sort_keys=True))
            return 0 if result.status == "COMPLETE" else 1
        if args.command == "paper-run":
            config = PaperConfig.load(args.config)
            if config.policy == "laya":
                backend = LayaBackend(args.model_path or ".cache/models/laya-multilingual", args.device)
            elif config.policy == "jev":
                backend = JevBackend(instruction_profile=config.instruction_profile)
            else:
                backend = PaperBaselineBackend(config.policy, config.master_seed + config.seed_index)
            result = asyncio.run(run_paper_session(config, backend, args.output,
                                                   wave_timeout_seconds=args.wave_timeout))
            summary = summarize_events(result.events, config.wholesale_price)
            print(json.dumps({"status": result.status, "rounds": result.rounds,
                              "formal_rounds": result.formal_rounds, "final_hash": result.final_hash,
                              "analysis": summary}, sort_keys=True, indent=2))
            return 0
        if args.command == "paper-replay":
            result = replay_paper(args.run_dir)
            print(json.dumps({"status": "COMPLETE", "rounds": result.rounds,
                "formal_rounds": result.formal_rounds, "trades": result.trades,
                "volume": result.volume, "profit_cents": result.profit_cents,
                "final_hash": result.final_hash}, sort_keys=True, indent=2))
            return 0
        if args.command == "paper-summarize":
            replay_result = replay_paper(args.run_dir)
            manifest = json.loads((args.run_dir / "manifest.json").read_text(encoding="utf-8"))
            summary = summarize_events(replay_result.events, manifest["config"]["wholesale_price"])
            print(json.dumps(summary, sort_keys=True, indent=2))
            return 0
        if args.command == "replay":
            result = replay(args.run_dir)
            print(json.dumps(_summary(args.run_dir, result), sort_keys=True, indent=2))
            return 0
        result = replay(args.run_dir)
        print(json.dumps(_summary(args.run_dir, result), sort_keys=True, indent=2))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "ERROR", "error": f"{type(exc).__name__}: {exc}"}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

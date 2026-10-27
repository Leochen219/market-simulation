"""Preview-first local interface for the behavioral-memory JEV study."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any

from .analysis import chinese_report, public_report, score_trajectories, write_report
from .lifecycle import ROOT, freeze_preview, verify_preview
from .runner import execute_preview, replay_preview, _read_events


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="capclaims-jev-behavioral-memory")
    commands = parser.add_subparsers(dest="command", required=True)
    preview = commands.add_parser("preview", help="freeze authorized source, prompts, schedule and labels")
    preview.add_argument("--source", type=Path, default=None)
    preview.add_argument("--output", required=True, type=Path)
    validate = commands.add_parser("validate", help="check the immutable preview digest")
    validate.add_argument("preview_dir", type=Path)
    validate.add_argument("--reviewed-sha256", required=True)
    for name in ("run", "replay", "score", "report"):
        cmd = commands.add_parser(name, help=f"{name} the frozen study")
        cmd.add_argument("preview_dir", type=Path)
        cmd.add_argument("run_dir", type=Path, nargs="?")
        cmd.add_argument("--reviewed-sha256", required=True)
        if name == "run":
            cmd.add_argument("--live", action="store_true")
            cmd.add_argument("--cost-attestation", type=Path)
        if name == "report":
            cmd.add_argument("--output", type=Path)
    return parser


def _private_run_path(path: Path) -> Path:
    root = (ROOT / "artifacts" / "participant-data" / "jev-behavioral-memory-v1").resolve()
    result = path.resolve()
    if not result.is_relative_to(root):
        raise ValueError("runs containing participant rows and raw model evidence must stay in participant-data")
    return result


def _attestation(path: Path, manifest: dict[str, Any], ledger_path: Path) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    contract = ROOT / "docs" / "engineering" / "behavioral-memory-cost-gate.md"
    expected = {"schema": "jev_behavioral_memory_cost_attestation_v1", "status": "VERIFIED",
        "preview_sha256": manifest["preview_sha256"], "source_sha256": manifest["metadata"]["source_sha256"],
        "code_sha256": manifest["code_sha256"],
        "cost_contract_sha256": hashlib.sha256(contract.read_bytes()).hexdigest(),
        "global_ledger_path": str(ledger_path.resolve()),
        "max_billable_input_tokens_per_attempt": 65536, "budget_usd": 12,
        "request_cap": 3888, "shared_ledger_empty_required": True}
    if not isinstance(value, dict) or any(value.get(k) != v for k, v in expected.items()):
        raise ValueError("live cost attestation is not bound to the preview, source, code and fixed USD12 ledger")


def _save_private_score(run_dir: Path, score: dict[str, Any]) -> dict[str, str]:
    data = (json.dumps(score, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    digest = hashlib.sha256(data).hexdigest()
    path = run_dir / "score-private.json"
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0), 0o600)
    try:
        view = memoryview(data)
        while view:
            view = view[os.write(fd, view):]
        os.fsync(fd)
    finally:
        os.close(fd)
    try:
        os.chmod(path, stat.S_IREAD | stat.S_IWRITE)
    except OSError:
        pass
    return {"path": str(path), "sha256": digest}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "preview":
        from .source import DEFAULT_SOURCE, build_authorized_cases
        output = _private_run_path(args.output)
        cases, labels, metadata = build_authorized_cases(args.source or DEFAULT_SOURCE)
        if labels:
            for case in cases:
                target = labels[case["person_id"]]
                if target.get("round") != 13:
                    raise ValueError("source adapter returned a target label for the wrong formal round")
                case["label"] = {key: target[key] for key in ("orders", "profit_cents")}
        from .wire_preview import build_wire_previews
        wire = asyncio.run(build_wire_previews(cases))
        result = freeze_preview(cases, metadata, output, exact_wire_previews=wire)
    elif args.command == "validate":
        manifest = verify_preview(args.preview_dir, args.reviewed_sha256)
        result = {"status": manifest["status"], "preview_sha256": manifest["preview_sha256"],
                  "people_n": manifest["people_n"], "trajectory_n": manifest["trajectory_n"],
                  "request_cap": manifest["request_cap"]}
    else:
        if args.run_dir is None:
            raise ValueError(f"{args.command} requires run_dir")
        run_dir = _private_run_path(args.run_dir)
        manifest = verify_preview(args.preview_dir, args.reviewed_sha256)
        if args.command == "run":
            if run_dir.exists():
                raise FileExistsError("one-shot run directory already exists")
            if not args.live:
                result = asyncio.run(execute_preview(args.preview_dir, run_dir,
                    args.reviewed_sha256, fake=True))
            else:
                if args.cost_attestation is None:
                    raise ValueError("live execution requires --cost-attestation")
                if (manifest.get("wire_preview_mode") != "SDK_EXACT"
                        or not manifest["metadata"].get("source_paths")):
                    raise ValueError("live execution requires a complete SDK-exact preview and refreshable source paths")
                from .transport import CostLedger, DEFAULT_LEDGER, JevWireClient
                _attestation(args.cost_attestation, manifest, DEFAULT_LEDGER)
                ledger = CostLedger(DEFAULT_LEDGER, preview_hash=manifest["preview_sha256"])
                summary = ledger.summary()
                if summary["physical_requests"] != 0 or summary["committed_nanodollars"] != 0:
                    raise ValueError("shared cost ledger must be empty before this one-shot study")
                evidence = run_dir.parent / f"{run_dir.name}.transport-evidence"
                if evidence.exists():
                    raise FileExistsError("transport evidence directory already exists")
                marker = DEFAULT_LEDGER.parent / "live-started.json"
                marker.parent.mkdir(parents=True, exist_ok=True)
                with marker.open("xb") as stream:
                    stream.write((json.dumps({"preview_sha256": manifest["preview_sha256"]}) + "\n").encode())
                    stream.flush()
                    os.fsync(stream.fileno())
                from capclaims.backends.jev import load_local_env
                load_local_env()
                api_key = os.getenv("TYPESAFE_API_KEY")
                if not api_key:
                    raise ValueError("TYPESAFE_API_KEY is unavailable")
                async def live_run():
                    async with JevWireClient(api_key=api_key, ledger=ledger,
                                             evidence_dir=evidence, category="human") as client:
                        return await execute_preview(args.preview_dir, run_dir,
                            args.reviewed_sha256, fake=False, client=client,
                            preflight=lambda _: True)
                result = asyncio.run(live_run())
        elif args.command == "replay":
            result = asyncio.run(replay_preview(args.preview_dir, run_dir, args.reviewed_sha256))
        else:
            replayed = asyncio.run(replay_preview(args.preview_dir, run_dir, args.reviewed_sha256))
            labels = json.loads((args.preview_dir / "scoring-labels.json").read_text(encoding="utf-8"))["labels"]
            results = [e["data"] for e in _read_events(run_dir) if e["type"] == "TRAJECTORY_COMPLETE"]
            if replayed["status"] == "REPLAY_VALID":
                score = score_trajectories(manifest["schedule"], labels, results)
            else:
                score = {"status": "INCOMPLETE", "planned_trajectories": 108,
                    "completed_trajectories": len(results),
                    "missing_trajectory_n": 108 - len(results),
                    "scoring_note": "No reduced-K or complete-case conclusion is reported."}
            private_path = run_dir / "score-private.json"
            if private_path.exists():
                expected_private = (json.dumps(score, ensure_ascii=False, sort_keys=True,
                    indent=2, allow_nan=False) + "\n").encode()
                existing = private_path.read_bytes()
                if existing != expected_private:
                    raise ValueError("existing private score differs from the recomputed replayable score")
                private = {"path": str(private_path), "sha256": hashlib.sha256(existing).hexdigest()}
            else:
                private = _save_private_score(run_dir, score)
            if args.command == "report":
                report_path = write_report(run_dir, score, args.output)
                (run_dir / "report.zh.md").write_text(chinese_report(score) + "\n", encoding="utf-8")
                result = {"status": score["status"], "report_path": str(report_path),
                    "private_score_artifact": private, "replay": replayed}
            else:
                result = public_report(score) | {"private_score_artifact": private, "replay": replayed}
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

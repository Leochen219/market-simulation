"""Preview-first CLI for `jev_conditional_behavior_v1`."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any

from .analysis import score_run
from .runner import execute_preview, replay_run
from .study import build_preview, verify_preview


def _safe_score(score: dict[str, Any]) -> dict[str, Any]:
    safe = {key: score[key] for key in ("status", "preview_sha256", "execution_mode",
        "planned_targets", "completed_sample_rows", "per_arm_participant_first",
        "physical_attempted_requests", "completed_trace_requests",
        "case_state_counts", "provider_attempt_evidence",
        "per_arm_participant_first_bounds", "per_arm_complete_case",
        "portfolio_diagnostics_by_arm", "weighted_order_totals_w1_by_arm",
        "weighted_price_w1_by_arm", "weighted_feature_moments_by_arm",
        "score_registry", "baseline_score_registry",
        "baseline_participant_first", "metric_definitions",
        "primary_E_vs_state", "paired_energy_contrasts", "limitations") if key in score}
    private_keys = {"person_id", "participant_id", "case_id", "target_id", "donor_id",
        "participants_by_arm", "participant_differences", "leave_one_participant_out_intervals",
        "per_person", "person_values", "target_rows", "baseline_target_rows"}
    def scrub(value):
        if isinstance(value, dict):
            return {key: scrub(child) for key, child in value.items() if key not in private_keys}
        if isinstance(value, list):
            return [scrub(child) for child in value]
        if isinstance(value, tuple):
            return [scrub(child) for child in value]
        return value
    return scrub(safe)


def _save_private_score(run_dir: Path, score: dict[str, Any]) -> dict[str, str]:
    """Persist complete per-context evidence with owner-only file permissions."""
    data = (json.dumps(score, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    digest = hashlib.sha256(data).hexdigest()
    path = run_dir / "score-private.json"
    if path.exists():
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise FileExistsError("private score artifact already exists with different contents")
        return {"path": str(path), "sha256": digest}
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="capclaims-jev-conditional", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preview = commands.add_parser("preview", help="freeze the source, roles, labels, and 680 inference rows")
    preview.add_argument("--source", type=Path, default=None)
    preview.add_argument("--output", required=True, type=Path)
    validate = commands.add_parser("validate", help="verify an immutable preview by reviewed digest")
    validate.add_argument("preview_dir", type=Path)
    validate.add_argument("--reviewed-sha256", required=True)
    run = commands.add_parser("run", help="run the frozen rows; local fake mode is the default")
    run.add_argument("preview_dir", type=Path)
    run.add_argument("run_dir", type=Path)
    run.add_argument("--reviewed-sha256", required=True)
    run.add_argument("--live", action="store_true", help="use Jev with the verified cost attestation")
    run.add_argument("--cost-attestation", type=Path)
    replay = commands.add_parser("replay", help="strictly verify event hashes and semantic portfolios")
    replay.add_argument("preview_dir", type=Path)
    replay.add_argument("run_dir", type=Path)
    score = commands.add_parser("score", help="score a strictly replayable run")
    score.add_argument("preview_dir", type=Path)
    score.add_argument("run_dir", type=Path)
    report = commands.add_parser("report", help="write an aggregate report without participant rows")
    report.add_argument("preview_dir", type=Path)
    report.add_argument("run_dir", type=Path)
    report.add_argument("--output", type=Path)
    return parser


def _attestation(path: Path, manifest: dict[str, Any], ledger: Any) -> tuple[dict[str, Any], Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    required = {"schema", "status", "preview_sha256", "source_sha256", "cost_contract_sha256",
                "global_ledger_path", "max_billable_input_tokens_per_attempt", "budget_usd"}
    if not isinstance(value, dict) or not required <= set(value):
        raise ValueError("cost attestation is missing required frozen fields")
    from .study import _code_hashes, _sha_bytes
    from .runner import MAX_BODY_BYTES, MAX_REQUESTS
    contract_path = Path(__file__).resolve().parents[3] / "docs" / "engineering" / "conditional-behavior-cost-gate.md"
    expected = {"schema": "jev_conditional_cost_attestation_v1", "status": "VERIFIED",
        "preview_sha256": manifest["preview_sha256"], "source_sha256": manifest["source_sha256"],
        "cost_contract_sha256": _sha_bytes(contract_path.read_bytes()),
        "global_ledger_path": str(ledger.path.resolve()), "max_billable_input_tokens_per_attempt": 65536,
        "budget_usd": 20}
    if any(value.get(key) != expected[key] for key in required):
        raise ValueError("cost attestation is not bound to this frozen preview and global ledger")

    def verify(candidate: dict[str, Any], frozen: dict[str, Any]) -> bool:
        if candidate != value or frozen["code_sha256"] != _code_hashes():
            return False
        # Reserved rows are already included in committed_nanodollars.
        return ledger.summary()["committed_nanodollars"] <= 20_000_000_000
    return value, verify


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "preview":
        from .source import DEFAULT_SOURCE, build_authorized_plan
        plan, metadata = build_authorized_plan(args.source or DEFAULT_SOURCE)
        result = build_preview(plan, args.output, source_sha256=metadata["source_sha256"],
            exclusion_manifest_sha256=metadata["exclusion_manifest_sha256"],
            source_policy_sha256=metadata["source_policy_sha256"],
            source_archive_sha256=metadata["source_archive_sha256"], source_policy_verified=True)
        public = {"status": "FROZEN", "preview_sha256": result["preview_sha256"],
            "people_n": result["people_n"], "target_n": result["target_n"],
            "sample_rows": result["sample_rows"], "request_cap": result["request_cap"]}
    elif args.command == "validate":
        manifest = verify_preview(args.preview_dir, args.reviewed_sha256)
        public = {"status": "PREVIEW_VALID", "preview_sha256": manifest["preview_sha256"],
            "people_n": manifest["people_n"], "target_n": manifest["target_n"],
            "sample_rows": manifest["sample_rows"]}
    elif args.command == "run":
        manifest = verify_preview(args.preview_dir, args.reviewed_sha256)
        if not args.live:
            execution = asyncio.run(execute_preview(args.preview_dir, args.run_dir,
                args.reviewed_sha256, fake=True))
        else:
            if args.cost_attestation is None:
                raise ValueError("live execution requires --cost-attestation from root")
            from .sampling import sample_portfolio
            from .transport import CostLedger, JevWireClient
            root = Path(__file__).resolve().parents[3]
            ledger = CostLedger(root / "artifacts" / "participant-data" / "jev-conditional-v1" / "cost-ledger.sqlite",
                preview_hash=manifest["preview_sha256"])
            attestation, verifier = _attestation(args.cost_attestation, manifest, ledger)
            # Verify the frozen source-policy and cost gate before reading env files.
            if not verifier(attestation, manifest):
                raise ValueError("live attestation failed preflight")
            from capclaims.backends.jev import load_local_env
            load_local_env()
            api_key = os.getenv("TYPESAFE_API_KEY")
            if not api_key:
                raise ValueError("TYPESAFE_API_KEY is unavailable")
            evidence = args.run_dir.parent / f"{args.run_dir.name}.transport-evidence"

            async def live_run():
                async with JevWireClient(api_key=api_key, ledger=ledger, evidence_dir=evidence, category="human") as client:
                    sampler = lambda observation, row: sample_portfolio(client, row, seed=manifest["seed"], category="human")
                    return await execute_preview(args.preview_dir, args.run_dir, args.reviewed_sha256,
                        fake=False, sampler=sampler, cost_attestation=attestation,
                        attestation_verifier=verifier, transport_evidence_dir=evidence,
                        cost_ledger_path=ledger.path)
            execution = asyncio.run(live_run())
        public = {key: execution.get(key) for key in ("status", "samples", "provider_requests",
            "physical_attempted_requests", "completed_trace_requests", "mode")}
    elif args.command == "replay":
        replayed = replay_run(args.preview_dir, args.run_dir)
        public = {key: replayed[key] for key in ("status", "sample_n", "planned_sample_n", "mode",
            "provider_requests", "physical_attempted_requests", "completed_trace_requests")}
    else:
        score = score_run(args.preview_dir, args.run_dir)
        private_score = _save_private_score(args.run_dir, score)
        public = _safe_score(score)
        public["private_full_score_artifact"] = private_score
        if args.command == "report":
            output = args.output or (args.run_dir / "report.json")
            if output.exists():
                raise FileExistsError("aggregate report is immutable")
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(public, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            public = {"status": score["status"], "report_path": str(output),
                "preview_sha256": score["preview_sha256"],
                "private_full_score_artifact": private_score}
    print(json.dumps(public, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

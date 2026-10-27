"""Thin offline entry point for the five-arm history-injection pilot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from capclaims.injection_study import replay_run, verify_preview
from capclaims.injection_study import (build_preview, execute_preview, resolve_prior_exclusions,
    score_run, SOURCE_SHA)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    preview = sub.add_parser("preview", help="freeze an authorized, source-bound five-arm preview")
    preview.add_argument("--source", type=Path, required=True)
    preview.add_argument("--policy", type=Path, required=True)
    preview.add_argument("--procurement-manifest", type=Path, required=True)
    preview.add_argument("--portfolio-manifest", type=Path, required=True)
    preview.add_argument("--output-dir", type=Path, required=True)
    preview = sub.add_parser("verify-preview", help="verify reviewed preview digest and fixed envelope")
    preview.add_argument("preview_dir", type=Path)
    preview.add_argument("--reviewed-sha256", required=True)
    replay = sub.add_parser("replay", help="strictly replay a private run against its frozen preview")
    replay.add_argument("preview_dir", type=Path)
    replay.add_argument("run_dir", type=Path)
    execute = sub.add_parser("execute", help="execute one reviewed run under the frozen shared budget")
    execute.add_argument("preview_dir", type=Path)
    execute.add_argument("run_dir", type=Path)
    execute.add_argument("--reviewed-sha256", required=True)
    execute.add_argument("--mock", action="store_true", help="use a local mock transport for offline validation")
    score = sub.add_parser("score", help="score a strictly replayed completed run offline")
    score.add_argument("preview_dir", type=Path)
    score.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    if args.command == "preview":
        from capclaims.personalization_xlsx import load_paper_episodes
        policy = json.loads(args.policy.read_text(encoding="utf-8"))
        source, episodes = load_paper_episodes(args.source,
            intended_use_authorized=policy.get("intended_use_authorized") is True,
            provider_transmission_permitted=policy.get("provider_transmission_permitted") is True,
            codebook_verified=policy.get("codebook_verified") is True)
        if (policy.get("source_kind") != "human" or policy.get("source_sha256") != source.source_sha256
                or source.source_sha256 != SOURCE_SHA
                or not all((source.codebook_verified, source.intended_use_authorized,
                            source.provider_transmission_permitted))):
            raise ValueError("source policy, checksum, or use attestations do not authorize this preview")
        roster = sorted({episode.participant_id for episode in episodes
            if episode.scope == "GA 10" and episode.treatment == "GA"})
        procurement_manifest = json.loads(args.procurement_manifest.read_text(encoding="utf-8"))
        portfolio_manifest = json.loads(args.portfolio_manifest.read_text(encoding="utf-8"))
        excluded, exclusion_hashes = resolve_prior_exclusions(procurement_manifest, portfolio_manifest, roster,
            manifest_sha256={"procurement": __import__("hashlib").sha256(
                args.procurement_manifest.read_bytes()).hexdigest(),
                "portfolio": __import__("hashlib").sha256(args.portfolio_manifest.read_bytes()).hexdigest()})
        result = build_preview(source, episodes, roster, excluded, args.output_dir,
            source_policy_sha256=__import__("hashlib").sha256(args.policy.read_bytes()).hexdigest(),
            exclusion_manifest_sha256=exclusion_hashes)
        result = {"status": "FROZEN", "preview_sha256": result["preview_sha256"],
            "session_n": len(result["sessions"]), "selected_person_n": len(result["selected_people"]),
            "preflight_max_body_bytes": result["preflight_max_body_bytes"]}
    elif args.command == "verify-preview":
        manifest = verify_preview(args.preview_dir, args.reviewed_sha256)
        result = {"status": "PREVIEW_VALID", "preview_sha256": manifest["preview_sha256"],
            "session_n": len(manifest["sessions"]),
            "preflight_max_body_bytes": manifest["preflight_max_body_bytes"],
            "request_cap": manifest["request_cap"]}
    elif args.command == "replay":
        replay = replay_run(args.preview_dir, args.run_dir)
        kinds = [row["status"] for row in replay["statuses"]]
        result = {"status": "REPLAY_VALID", "session_n": replay["session_n"],
            "complete_n": kinds.count("complete"), "verified_tie_n": kinds.count("verified_tie")}
    elif args.command == "execute":
        execution = __import__("asyncio").run(execute_preview(args.preview_dir, args.run_dir,
            reviewed_preview_sha256=args.reviewed_sha256, fake=args.mock))
        result = {key: execution.get(key) for key in
            ("status", "reason", "planned_sessions", "attempted_sessions", "unattempted_sessions",
             "requests_charged", "reserved_bytes")}
    else:
        result = score_run(args.preview_dir, args.run_dir)
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

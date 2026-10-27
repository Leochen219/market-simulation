"""Frozen same-history, same-prompt ablation of four current-state field names."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    preview = sub.add_parser("preview")
    preview.add_argument("parent_dir", type=Path)
    preview.add_argument("output_dir", type=Path)
    verify = sub.add_parser("verify-preview")
    verify.add_argument("preview_dir", type=Path)
    verify.add_argument("--reviewed-sha256", required=True)
    execute = sub.add_parser("execute")
    execute.add_argument("preview_dir", type=Path)
    execute.add_argument("run_dir", type=Path)
    execute.add_argument("--reviewed-sha256", required=True)
    execute.add_argument("--mock", action="store_true")
    for name in ("replay", "score"):
        command = sub.add_parser(name)
        command.add_argument("preview_dir", type=Path)
        command.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    from capclaims import field_alignment_study as study
    if args.command == "preview":
        manifest = study.build_preview(args.parent_dir, args.output_dir)
        result = {"status": "FROZEN", "preview_sha256": manifest["preview_sha256"],
                  "session_n": len(manifest["sessions"]),
                  "preflight_max_body_bytes": manifest["preflight_max_body_bytes"]}
    elif args.command == "verify-preview":
        manifest = study.verify_preview(args.preview_dir, args.reviewed_sha256)
        result = {"status": "PREVIEW_VALID", "preview_sha256": manifest["preview_sha256"],
                  "session_n": len(manifest["sessions"])}
    elif args.command == "execute":
        execution = asyncio.run(study.execute_preview(args.preview_dir, args.run_dir,
            reviewed_preview_sha256=args.reviewed_sha256, fake=args.mock))
        result = {key: execution.get(key) for key in ("status", "reason", "planned_sessions",
            "attempted_sessions", "unattempted_sessions", "requests_charged", "reserved_bytes")}
    elif args.command == "replay":
        replay = study.replay_run(args.preview_dir, args.run_dir)
        result = {"status": "REPLAY_VALID", "session_n": replay["session_n"],
                  "complete_n": len(replay["portfolios"])}
    else:
        scored = study.score_run(args.preview_dir, args.run_dir)
        public_keys = ("status", "n_people", "repeat_n_per_person_arm", "per_arm",
                       "paired_aligned_minus_control", "repeat_consistency", "interpretation",
                       "transport_audit", "coverage", "session_n")
        result = {key: scored[key] for key in public_keys if key in scored}
    print(json.dumps(result, ensure_ascii=True, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

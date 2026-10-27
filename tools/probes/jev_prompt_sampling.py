"""Safe CLI for previewing and validating the Jev prompt-sampling study."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preview = commands.add_parser("preview")
    preview.add_argument("parent_dir", type=Path)
    preview.add_argument("output_dir", type=Path)
    verify = commands.add_parser("verify")
    verify.add_argument("preview_dir", type=Path)
    verify.add_argument("--reviewed-sha256", required=True)
    execute = commands.add_parser("execute")
    execute.add_argument("preview_dir", type=Path)
    execute.add_argument("run_dir", type=Path)
    execute.add_argument("--prompt", choices=("original", "explicit", "structured"), required=True)
    execute.add_argument("--reviewed-sha256", required=True)
    execute.add_argument("--mock", action="store_true")
    for name in ("replay", "score"):
        command = commands.add_parser(name)
        command.add_argument("preview_dir", type=Path)
        command.add_argument("run_dir", type=Path)
        command.add_argument("--prompt", choices=("original", "explicit", "structured"))
    compare = commands.add_parser("compare3runs")
    compare.add_argument("preview_dir", type=Path)
    compare.add_argument("original_run", type=Path)
    compare.add_argument("explicit_run", type=Path)
    compare.add_argument("structured_run", type=Path)
    args = parser.parse_args(argv)
    from capclaims import prompt_sampling_study as study
    if args.command == "preview":
        manifest = study.build_preview(args.parent_dir, args.output_dir)
        result = {"status": "FROZEN", "preview_sha256": manifest["preview_sha256"],
            "session_n_per_prompt": manifest["session_n_per_prompt"], "people_n": len(manifest["people"]),
            "prompts": manifest["prompts"], "decoders": manifest["decoders"]}
    elif args.command == "verify":
        manifest = study.verify_preview(args.preview_dir, args.reviewed_sha256)
        result = {"status": "PREVIEW_VALID", "preview_sha256": manifest["preview_sha256"],
            "session_n_per_prompt": manifest["session_n_per_prompt"], "people_n": len(manifest["people"])}
    elif args.command == "execute":
        import asyncio
        result = asyncio.run(study.execute_preview(args.preview_dir, args.run_dir, args.prompt,
            args.reviewed_sha256, mock=args.mock))
        result = {key: result[key] for key in ("status", "planned_sessions", "attempted_sessions",
            "requests_charged", "reserved_bytes", "max_overlap") if key in result}
    elif args.command == "replay":
        replayed = study.replay_run(args.preview_dir, args.run_dir, args.prompt)
        result = {"status": replayed["status"], "prompt": replayed["prompt"],
            "session_n": replayed["session_n"], "request_n": replayed["request_n"]}
    elif args.command == "score":
        score = study.score_run(args.preview_dir, args.run_dir, args.prompt)
        result = {key: score[key] for key in ("status", "prompt", "execution_mode", "configs",
            "paired_differences", "repeat_consistency", "limitations")}
    else:
        prompts = ("original", "explicit", "structured")
        runs = dict(zip(prompts, (args.original_run, args.explicit_run, args.structured_run)))
        result = study.compare_three_runs(args.preview_dir, runs)
    print(json.dumps(result, ensure_ascii=True, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

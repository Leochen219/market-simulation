"""Preview or execute bounded Jev paper diagnostics, without loading Laya."""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from capclaims.backends.jev import JevBackend, estimate_input_tokens, load_local_env
from capclaims.backends.jev_instructions import build_instruction
from capclaims.engine.protocol import DecisionRequest, digest_json, validate_reply
from capclaims.paper.interface import build_candidates
from tools.probes.economic_understanding_probe import (
    economic_requests, numeric_requests, execute_pairs, evaluate_readiness,
)
from tools.probes.paper_capability_probe import build_paper_states

MODEL = "jev-1.14.0"


def production_requests():
    states = [json.loads(s) for s in build_paper_states("economic_v2")]
    return [DecisionRequest(f"agent_{i:02d}", 18, "paper:slot4:price:u", state,
                            build_candidates(tuple(range(10)), state))
            for i, state in enumerate(states)]


def fixture_pairs(suite):
    numeric = numeric_requests("economic_v2")
    economic = economic_requests("economic_v2")
    if suite == "smoke":
        names = {"no_need_overpay", "fill_shortage", "shortage_overpay",
                 "sell_surplus", "sell_needed_premium"}
        return [], [pair for pair in economic if pair[1]["case"] in names and pair[1]["past"] == 10]
    confirmation = suite == "confirmation"
    if suite not in {"development", "confirmation"}:
        raise ValueError("unknown suite")
    return ([p for p in numeric if p[1]["confirmation"] == confirmation],
            [p for p in economic if p[1]["confirmation"] == confirmation])


def build_plan(suite, profile, include_history=False):
    if include_history:
        raise ValueError("free-running history sweeps are separate from this bounded diagnostic")
    numeric, economic = fixture_pairs(suite)
    production = [] if suite == "smoke" else production_requests()
    requests = []
    for request in [*production, *[r for r, _ in numeric], *[r for r, _ in economic]]:
        instruction = build_instruction(request, profile)
        wire = {"model": MODEL, "state": dict(request.observation), "questions": {
            "decision": {"type": "choice", "instructions": instruction,
                         "criteria": {c.candidate_id: c.label for c in request.candidates}}}}
        requests.append({"request_id": request.request_id, "wire": wire,
                         "estimated_input_tokens": estimate_input_tokens(request, instruction)})
    return {"suite": suite, "instruction_profile": profile, "model": MODEL,
            "numeric_count": len(numeric), "economic_count": len(economic),
            "request_count": len(requests), "requests": requests,
            "estimated_input_tokens": sum(r["estimated_input_tokens"] for r in requests),
            "fixture_scoring_hash": digest_json([m for _, m in [*numeric, *economic]]),
            "wire_plan_hash": digest_json(requests)}


def overlap_summary(windows):
    events = [(row["started_at"], 1) for row in windows]
    events += [(row["finished_at"], -1) for row in windows]
    active = peak = 0
    for _, delta in sorted(events):  # finishes before starts at an identical timestamp
        active += delta
        peak = max(peak, active)
    return {"request_count": len(windows), "peak_in_flight": peak,
            "overlap_demonstrated": peak >= 2}


def qualify_readiness(numbers, economics, transport_parallelism):
    return (evaluate_readiness(numbers, economics)
            and transport_parallelism["request_count"] == 10
            and transport_parallelism["peak_in_flight"] == 10)


async def execute(args, plan, output):
    backend = None
    result = {"status": "BLOCKED", "suite": args.suite,
              "instruction_profile": args.instruction_profile,
              "wire_plan_hash": plan["wire_plan_hash"],
              "fixture_scoring_hash": plan["fixture_scoring_hash"],
              "model": MODEL, "real_api_calls": 0}
    try:
        backend = JevBackend(instruction_profile=args.instruction_profile)
        if plan["request_count"] > backend.max_requests or plan["estimated_input_tokens"] > backend.max_estimated_input_tokens:
            result["reason"] = "Planned diagnostic exceeds configured request/token budget; zero requests sent."
            return result
        if args.suite != "smoke":
            requests = production_requests()
            replies = await asyncio.wait_for(backend.decide_many(requests), 120)
            by_id = {r.request_id: r for r in replies}
            if len(by_id) != len(requests):
                raise ValueError("production reply identity coverage mismatch")
            for request in requests:
                validate_reply(request, by_id[request.request_id])
            result["sdk_coroutine_parallelism"] = overlap_summary([dict(r.timing) for r in replies])
            result["http_transport_parallelism"] = overlap_summary(list(backend.transport_timings))
            (output / "production-raw.json").write_text(json.dumps({"requests": [asdict(r) for r in requests],
                "replies": [asdict(r) for r in replies]}, indent=2) + "\n", encoding="utf-8")
        numeric, economic = fixture_pairs(args.suite)
        numbers = await execute_pairs(backend, numeric, output, "jev-numeric") if numeric else []
        economics = await execute_pairs(backend, economic, output, "jev-economic")
        result.update(numeric_correct=sum(r["correct"] for r in numbers), numeric_total=len(numbers),
                      economic_correct=sum(r["correct"] for r in economics), economic_total=len(economics))
        if args.suite == "smoke":
            result["status"] = "SMOKE_COMPLETE"
            result["readiness"] = "NOT_EVALUATED: smoke subset cannot certify readiness"
            result["sdk_coroutine_parallelism"] = overlap_summary([dict(r["reply"]["timing"]) for r in economics])
            result["http_transport_parallelism"] = overlap_summary(list(backend.transport_timings))
        else:
            result["status"] = "PASS" if qualify_readiness(numbers, economics, result["http_transport_parallelism"]) else "FAIL"
        return result
    except Exception as exc:
        # SDK exceptions can contain raw bodies; never persist exception text or credentials.
        result["status"] = "ERROR" if backend and backend.window_timings else "BLOCKED"
        result["error_type"] = type(exc).__name__
        result["reason"] = "Check local key, client limits, request/token bounds and API availability; details intentionally redacted."
        return result
    finally:
        if backend:
            result["budget_usage"] = backend.budget_usage
            result["request_windows"] = list(backend.window_timings)
            result["http_transport_windows"] = list(backend.transport_timings)
            result["real_api_calls"] = len(backend.window_timings)
            await backend.aclose()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("smoke", "development", "confirmation"), default="smoke")
    parser.add_argument("--instruction-profile", choices=("matched_v2", "explicit_v1"), default="explicit_v1")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true", help="send real bounded API calls; default only previews")
    args = parser.parse_args(argv)
    load_local_env()
    if args.output_dir.exists():
        raise FileExistsError("output directory must be new")
    args.output_dir.mkdir(parents=True)
    plan = build_plan(args.suite, args.instruction_profile)
    (args.output_dir / "request-plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    if args.execute:
        result = asyncio.run(execute(args, plan, args.output_dir))
    else:
        result = {"status": "PREVIEW", "real_api_calls": 0,
                  **{k: v for k, v in plan.items() if k != "requests"}}
    result["probe_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 2 if result["status"] == "BLOCKED" else 1 if result["status"] in {"ERROR", "FAIL"} else 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Independent, read-only whole-pilot acceptance gate; --execute runs once first.

This does not repair or certify the library's general post-reply failure audit.
It rejects every operational-failure run and never selects a subset of targets.
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

from capclaims.portfolio_study import execute_preview, replay_run, verify_preview


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def lines(path):
    return [json.loads(row) for row in Path(path).read_text(encoding="utf-8").splitlines() if row]


def check_statuses(statuses, planned):
    pairs = [(row["target_id"], row["arm"]) for row in statuses]
    expected = [(row["target_id"], row["arm"]) for row in planned]
    require(len(expected) == len(set(expected)) == 12 and pairs == expected,
            "all twelve original sessions must be present in order")
    complete = ties = 0
    for row in statuses:
        require(row.get("failure_stage") is None, "operational failure invalidates the entire pilot")
        if row.get("status") == "complete":
            complete += 1
        else:
            require(row.get("status") == "incomplete" and row.get("reason") == "unresolved_tie",
                    "only independently verified model ties may be incomplete")
            ties += 1
    return complete, ties


def check_events(trace):
    ids = []
    for chained in trace:
        row = chained["event"]
        require(row.get("failure_stage") is None and row.get("kind") in {"forced", "model"},
                "an operational failure event invalidates the entire pilot")
        event = row["event"]
        if row["kind"] == "forced":
            require(event.get("kind") == "forced" and event.get("origin") == "constraint",
                    "forced steps must be explicit constraints")
            continue
        require(event.get("origin") == "model", "model event origin is invalid")
        if event.get("kind") == "failure":
            require(event.get("reason") == "unresolved_tie", "non-tie model failure is forbidden")
            values = list(row["reply"]["probabilities"].values())
            require(values and all(type(x) in (int, float) and math.isfinite(x) for x in values),
                    "tie has invalid probability evidence")
            require(sum(x == max(values) for x in values) > 1, "recorded tie is not a tie")
        else:
            require(event.get("kind") == "choice", "unexpected model event kind")
        ids.append(event["binding"]["request_id"])
    require(len(ids) == len(set(ids)), "model request IDs must be unique")
    return ids


def validate_static(preview, gate_file, gate_sha):
    require(digest(gate_file) == gate_sha, "frozen external gate changed")
    gate = load(gate_file)
    require(digest(__file__) == gate["checker_sha256"], "frozen checker changed")
    manifest, _ = verify_preview(preview, gate["preview_sha256"])
    require(manifest["sdk_version"] == "typesafe-sdk==0.7.5" and manifest["source_kind"] == "human",
            "pilot must use the reviewed SDK and human-source preview")
    return gate, manifest


def audit(preview, run, gate_file, gate_sha):
    gate, manifest = validate_static(preview, gate_file, gate_sha)
    receipt = load(gate_file.parent / "root-execution-receipt.json")
    require(receipt["gate_sha256"] == gate_sha and receipt["checker_sha256"] == gate["checker_sha256"],
            "pre-provider receipt does not bind the gate and checker")
    require(receipt["preview_sha256"] == gate["preview_sha256"], "receipt preview differs")
    result, terminal = load(run / "result.json"), load(run / "terminal.json")
    require(result["execution_mode"] == terminal["execution_mode"] == "provider",
            "mock outputs cannot satisfy real-provider acceptance")
    require(load(run / "run-manifest.json")["execution_mode"] == "provider", "run mode mismatch")
    complete, ties = check_statuses(terminal["statuses"], manifest["sessions"])
    require(result["attempted_portfolios"] == 12 and result["unattempted_portfolios"] == 0,
            "all original sessions must have been attempted")
    trace = lines(run / "trace.jsonl")
    ids = check_events(trace)
    ledger = lines(run / "attempts.jsonl")
    sent = lines(run / "transport-requests.jsonl")
    responses = lines(run / "transport-bodies.jsonl")
    timings = load(run / "transport-timings.json")
    for records in (ledger, sent, responses, timings):
        actual = [row["request_id"] for row in records]
        require(len(actual) == len(set(actual)) == len(ids) and set(actual) == set(ids),
                "independent HTTP, ledger, and transcript counts/IDs differ")
    ledger_by_id = {row["request_id"]: row for row in ledger}
    for chained in trace:
        row = chained["event"]
        if row["kind"] == "model":
            request_id = row["event"]["binding"]["request_id"]
            require(row["expected_wire_sha256"] == ledger_by_id[request_id]["expected_sha256"],
                    "transcript's expected wire hash differs from independent ledger")
    require(all(row["status"] == 200 for row in responses + timings), "every HTTP reply must be 200")
    require(all(math.isfinite(row["started_at"]) and math.isfinite(row["finished_at"])
                and row["started_at"] >= receipt["before_provider_monotonic"]
                and row["finished_at"] >= row["started_at"] for row in timings),
            "gate receipt must precede every provider request")
    count = len(ids)
    require(0 < count <= 420 and count == result["actual_http_request_n"], "request ceiling/count failed")
    require(result["budget"]["requests_charged"] == count
            and result["budget"]["reserved_bytes"] == count * 7500 <= 3_150_000,
            "global charged budget does not reconcile")
    require(all(row["reserved_bytes"] == 7500 for row in ledger), "reservation bound changed")
    body_bytes = sum(len(row["body_utf8"].encode("utf-8")) for row in sent)
    require(all(len(row["body_utf8"].encode("utf-8")) == row["body_bytes"] <= 7500 for row in sent),
            "actual body bound failed")
    require(body_bytes == result["actual_request_body_bytes"], "actual body-byte count differs")
    require(1 <= result["actual_http_max_overlap"] <= 10, "HTTP concurrency bound failed")
    # This rebuilds every normal/forced choice and compares the raw response
    # through the installed SDK/backend; it never reads the scoring-target file.
    replayed = replay_run(preview, run)
    require(replayed["audit"]["ledger_line_n"] == count, "strict replay accounting differs")
    require(digest(gate_file) == gate_sha and digest(__file__) == gate["checker_sha256"],
            "verification artifacts changed during audit")
    return {"status": "ACCEPTED_UNDER_FROZEN_GATE", "scope": "all twelve original portfolios",
            "complete_portfolios": complete, "verified_tie_portfolios": ties,
            "operational_failures": 0, "actual_http_requests": count,
            "actual_http_max_overlap": result["actual_http_max_overlap"],
            "actual_request_body_bytes": body_bytes, "gate_sha256": gate_sha,
            "checker_sha256": gate["checker_sha256"], "preview_sha256": gate["preview_sha256"],
            "library_limitation_resolved": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--gate-file", type=Path, required=True)
    parser.add_argument("--gate-sha256", required=True)
    parser.add_argument("--execute", action="store_true", help="perform the single real-provider run before audit")
    args = parser.parse_args(argv)
    gate, _ = validate_static(args.preview_dir, args.gate_file, args.gate_sha256)
    if args.execute:
        require(not args.run_dir.exists(), "execution never resumes or overwrites a run")
        receipt = {"gate_sha256": args.gate_sha256, "checker_sha256": digest(__file__),
                   "preview_sha256": gate["preview_sha256"],
                   "before_provider_utc": datetime.now(timezone.utc).isoformat(),
                   "before_provider_monotonic": time.monotonic()}
        with (args.gate_file.parent / "root-execution-receipt.json").open("xb") as handle:
            handle.write(json.dumps(receipt, sort_keys=True).encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
        # Normal ties return PARTIAL; an escaped exception never reaches audit.
        # No retry, replacement run, source workbook, or scoring labels are read.
        asyncio.run(execute_preview(args.preview_dir, args.run_dir,
                                   reviewed_preview_sha256=gate["preview_sha256"], fake=False))
    outcome = audit(args.preview_dir, args.run_dir, args.gate_file, args.gate_sha256)
    (args.run_dir / "root-acceptance-audit.json").write_text(json.dumps(outcome, indent=2), encoding="utf-8")
    print(json.dumps(outcome, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

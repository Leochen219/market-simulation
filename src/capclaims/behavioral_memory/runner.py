"""One-shot trajectory execution, append-only evidence, and strict semantic replay."""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sqlite3
from collections import defaultdict
from pathlib import Path
import time
from datetime import datetime, timezone
from typing import Any

from capclaims.conditional_behavior.protocol import replay_choices
from .lifecycle import _canonical, _sha, verify_preview
from .sampling import sample_trajectory

MAX_BATCH = 10
MAX_BATCH_SECONDS = 600


def _append(path: Path, value: dict[str, Any]) -> None:
    data = _canonical(value) + b"\n"
    fd = os.open(path, os.O_CREAT | os.O_APPEND | os.O_WRONLY | getattr(os, "O_BINARY", 0), 0o600)
    try:
        view = memoryview(data)
        while view:
            view = view[os.write(fd, view):]
        os.fsync(fd)
    finally:
        os.close(fd)


def _event(events_path: Path, previous: str, sequence: int,
           kind: str, data: dict[str, Any]) -> tuple[str, int]:
    event = {"sequence": sequence,
        "previous_sha256": previous, "type": kind, "data": data}
    digest = _sha(_canonical(event))
    _append(events_path, {**event, "sha256": digest})
    return digest, sequence + 1


class ProbabilityMock:
    """Offline deterministic uniform-probability harness, explicitly not model evidence."""
    async def choose(self, *, request_id: str, state: Any, instructions: str,
                     criteria: dict[str, str]) -> dict[str, Any]:
        ids = list(criteria)
        selected = ids[0]
        return {"model": "jev-1.14.0", "usage": {"input_tokens": 1, "output_tokens": 1},
            "answers": {"decision": {"type": "choice", "choice": selected,
                "probabilities": {key: 1 / len(ids) for key in ids}}}}


async def execute_preview(preview_dir: str | Path, run_dir: str | Path, reviewed_sha256: str, *,
                          fake: bool = True, client: Any = None,
                          preflight: Any = None) -> dict[str, Any]:
    manifest = verify_preview(preview_dir, reviewed_sha256)
    if fake:
        if client is not None:
            raise ValueError("fake mode owns its local deterministic mock")
        client = ProbabilityMock()
    else:
        if manifest.get("wire_preview_mode") != "SDK_EXACT":
            raise ValueError("live execution requires SDK-exact preview serialization")
        if not manifest["metadata"].get("source_paths"):
            raise ValueError("live execution requires refreshable authorized source paths")
        if not callable(preflight) or preflight(manifest) is not True:
            raise ValueError("live execution requires a reviewed cost-contract preflight")
        if client is None:
            raise ValueError("live execution needs the root-provided bounded Jev client")
    output = Path(run_dir)
    output.mkdir(parents=True, exist_ok=False)
    events_path = output / "events.jsonl"
    previous = "0" * 64
    sequence = 0
    previous, sequence = _event(events_path, previous, sequence, "RUN_START", {"mode": "MOCK" if fake else "LIVE",
        "preview_sha256": reviewed_sha256, "planned_trajectories": manifest["trajectory_n"],
        "seed": manifest["seed"], "schedule_sha256": _sha(_canonical(manifest["schedule"]))})
    rows = manifest["schedule"]
    completed_keys: set[tuple[str, int, str]] = set()
    failed_keys: set[tuple[str, int, str]] = set()
    complete_n = failed_n = 0
    batch_index = 0
    peak_concurrency = 0
    stop = False
    for repeat in (0, 1):
        repeat_rows = [row for row in rows if row["repeat"] == repeat]
        for offset in range(0, len(repeat_rows), MAX_BATCH):
            batch = repeat_rows[offset:offset + MAX_BATCH]
            batch_started = time.monotonic()
            batch_start_utc = datetime.now(timezone.utc).isoformat()
            attempts_before = (client.ledger.summary()["physical_requests"] if not fake else 0)
            previous, sequence = _event(events_path, previous, sequence, "BATCH_START", {"batch_index": batch_index,
                "repeat": repeat, "repeat_block_offset": offset, "batch_n": len(batch),
                "started_utc": batch_start_utc, "limit_seconds": MAX_BATCH_SECONDS})
            batch_failed = False
            def step_writer(row):
                def save(step):
                    nonlocal previous, sequence
                    previous, sequence = _event(events_path, previous, sequence, "TRAJECTORY_STEP", {
                        "person_id": row["person_id"], "repeat": row["repeat"], "arm": row["arm"],
                        "step": step})
                return save
            def request_writer(row):
                def save(request):
                    nonlocal previous, sequence
                    previous, sequence = _event(events_path, previous, sequence, "TRAJECTORY_REQUEST", {
                        "person_id": row["person_id"], "repeat": row["repeat"], "arm": row["arm"],
                        "request": request})
                return save
            async def run_row(row):
                key = (row["person_id"], row["repeat"], row["arm"])
                try:
                    result = await sample_trajectory(client, row, seed=manifest["seed"],
                        on_step=step_writer(row), on_request=request_writer(row))
                except BaseException as exc:
                    failed_keys.add(key)
                    _write_failure(row, type(exc).__name__)
                    raise
                completed_keys.add(key)
                _write_complete(result)
                return result
            def _write_failure(row, error_type):
                nonlocal previous, sequence
                previous, sequence = _event(events_path, previous, sequence, "TRAJECTORY_FAILURE", {
                    "person_id": row["person_id"], "repeat": row["repeat"], "arm": row["arm"],
                    "error_type": error_type})
            def _write_complete(result):
                nonlocal previous, sequence
                previous, sequence = _event(events_path, previous, sequence, "TRAJECTORY_COMPLETE", result)
            try:
                tasks = [run_row(row) for row in batch]
                results = await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True),
                                                 timeout=MAX_BATCH_SECONDS)
            except asyncio.TimeoutError:
                elapsed = time.monotonic() - batch_started
                previous, sequence = _event(events_path, previous, sequence, "BATCH_TIMEOUT", {"batch_index": batch_index,
                    "repeat": repeat, "batch_n": len(batch), "elapsed_seconds": elapsed,
                    "finished_utc": datetime.now(timezone.utc).isoformat(),
                    "physical_attempted_requests": ((client.ledger.summary()["physical_requests"] - attempts_before)
                        if not fake else sum(len(event["data"].get("trace", [])) for event in
                            _read_events(output) if event["type"] == "TRAJECTORY_COMPLETE")),
                    "limit_seconds": MAX_BATCH_SECONDS})
                complete_n, failed_n = len(completed_keys), len(failed_keys)
                stop = True
                break
            complete_n, failed_n = len(completed_keys), len(failed_keys)
            batch_failed = any(isinstance(result, BaseException) for result in results)
            if hasattr(client, "active"):
                peak_concurrency = max(peak_concurrency, getattr(client, "peak", 0))
            else:
                peak_concurrency = max(peak_concurrency, len(batch))
            previous, sequence = _event(events_path, previous, sequence, "BATCH_END", {"batch_index": batch_index,
                "repeat": repeat, "batch_n": len(batch), "completed_n": sum(
                    not isinstance(result, BaseException) for result in results),
                "failed_n": sum(isinstance(result, BaseException) for result in results),
                "elapsed_seconds": time.monotonic() - batch_started,
                "started_utc": batch_start_utc, "finished_utc": datetime.now(timezone.utc).isoformat(),
                "peak_inflight_requests": getattr(client, "peak", peak_concurrency),
                "physical_attempted_requests": ((client.ledger.summary()["physical_requests"] - attempts_before)
                    if not fake else sum(len(result.get("trace", [])) for result in results
                        if not isinstance(result, BaseException)))})
            batch_index += 1
            if batch_failed:
                previous, sequence = _event(events_path, previous, sequence, "STOP_AFTER_TECHNICAL_FAILURE",
                                  {"next_batch_not_started": True, "repeat": repeat})
                stop = True
                break
        if stop:
            break
    status = "INCOMPLETE" if failed_n or complete_n != len(rows) else "COMPLETE"
    cost_summary = None
    if not fake:
        cost_summary = client.ledger.summary()
        expected_requests = sum(len(event["data"].get("trace", []))
            for event in _read_events(output) if event["type"] == "TRAJECTORY_COMPLETE")
        if (cost_summary["physical_requests"] != expected_requests or
                cost_summary["physical_requests"] > 3_888 or
                cost_summary["committed_nanodollars"] > 12_000_000_000 or
                (status == "COMPLETE" and (cost_summary["states"].get("settled") != expected_requests
                    or cost_summary["states"].get("reserved", 0) or cost_summary["states"].get("unknown", 0)))):
            status = "INCOMPLETE"
    summary = {"schema": "behavioral-memory-run-v1", "status": status,
        "mode": "MOCK" if fake else "LIVE", "preview_sha256": reviewed_sha256,
        "planned_trajectories": len(rows), "completed_trajectories": complete_n,
        "failed_trajectories": failed_n, "peak_inflight_requests": peak_concurrency,
        "cost_summary": cost_summary}
    previous, sequence = _event(events_path, previous, sequence, "RUN_END", summary)
    summary["final_event_sha256"] = previous
    (output / "run.json").write_bytes(_canonical(summary) + b"\n")
    return summary


def _read_events(run_dir: Path) -> list[dict[str, Any]]:
    lines = (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    events, previous = [], "0" * 64
    for sequence, line in enumerate(lines):
        event = json.loads(line)
        if event.get("sequence") != sequence or event.get("previous_sha256") != previous:
            raise ValueError("event sequence or hash chain is invalid")
        content = {key: event[key] for key in ("sequence", "previous_sha256", "type", "data")}
        digest = _sha(_canonical(content))
        if event.get("sha256") != digest:
            raise ValueError("event was changed after execution")
        previous = digest
        events.append(event)
    return events


def _verify_live_ledger(preview_hash: str, expected_request_ids: set[str],
                        reported: Mapping[str, Any]) -> set[str]:
    from .transport import DEFAULT_LEDGER, BUDGET, MAX_REQUESTS, MODEL, RESERVATION
    path = DEFAULT_LEDGER.resolve()
    if not path.is_file():
        raise ValueError("fixed behavioral-memory global ledger is missing")
    db = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=5)
    try:
        config = dict(db.execute("SELECT key,value FROM config"))
        if config != {"preview_hash": preview_hash, "budget": str(BUDGET), "model": MODEL,
                      "reservation": str(RESERVATION), "study": "jev_behavioral_memory_v1",
                      "request_cap": str(MAX_REQUESTS)}:
            raise ValueError("global cost ledger configuration differs from the frozen contract")
        rows = list(db.execute("SELECT request_id,state,committed,input_tokens FROM attempts"))
    finally:
        db.close()
    if len(rows) > MAX_REQUESTS or len({row[0] for row in rows}) != len(rows):
        raise ValueError("global ledger physical request count or identities are invalid")
    if any(request_id not in expected_request_ids for request_id, *_ in rows):
        raise ValueError("global ledger has an attempt absent from durable request events")
    states: dict[str, int] = {}
    committed = 0
    for request_id, state, amount, tokens in rows:
        states[state] = states.get(state, 0) + 1
        committed += amount
        if state == "settled":
            if type(tokens) is not int or not 0 <= tokens <= 65_536 or amount != tokens * 42:
                raise ValueError("ledger settlement is not authoritative pinned-model input usage")
        elif state in {"reserved", "unknown"}:
            if amount != RESERVATION:
                raise ValueError("in-flight or unresolved attempts must retain the full reservation")
        else:
            raise ValueError("global ledger contains an unknown attempt state")
    actual = {"physical_requests": len(rows), "states": states,
        "committed_nanodollars": committed,
        "settled_nanodollars": sum(row[2] for row in rows if row[1] == "settled"),
        "reserved_nanodollars": sum(row[2] for row in rows if row[1] in {"reserved", "unknown"}),
        "request_cap": MAX_REQUESTS, "budget_nanodollars": BUDGET}
    if committed > BUDGET or any(reported.get(key) != value for key, value in actual.items()):
        raise ValueError("run cost summary differs from the immutable global ledger")
    return {row[0] for row in rows}


async def _verify_raw_requests(run_dir: Path,
                              requests: Mapping[tuple[str, int, str], list[dict[str, Any]]],
                              physical_ids: set[str]) -> None:
    from capclaims.conditional_behavior.transport import serialize_body, strict_json
    evidence_root = run_dir.parent / f"{run_dir.name}.transport-evidence"
    request_ids_seen = set()
    for rows in requests.values():
        for request in rows:
            request_id = request["request_id"]
            if request_id in request_ids_seen:
                raise ValueError("request id was reused across trajectories")
            request_ids_seen.add(request_id)
            body = await serialize_body(state=request["state"], instructions=request["instructions"],
                                        criteria=request["criteria"])
            folder = evidence_root / hashlib.sha256(request_id.encode()).hexdigest()
            if request_id not in physical_ids:
                if folder.exists():
                    raise ValueError("unreserved request has raw evidence of a physical attempt")
                continue
            if not folder.is_dir():
                raise ValueError("global ledger attempt has no per-request raw evidence directory")
            request_path, attempt_path = folder / "request.json", folder / "attempt.json"
            if not request_path.is_file() or not attempt_path.is_file():
                raise ValueError("raw attempted request or attempt metadata is missing")
            raw_body = request_path.read_bytes()
            if strict_json(raw_body) != strict_json(body):
                raise ValueError("raw attempted request differs semantically from replayed SDK body")
            attempt = strict_json(attempt_path.read_bytes())
            if (attempt.get("request_id") != request_id or attempt.get("request_sha256") != _sha(raw_body)
                    or attempt.get("retry_count") != 0 or attempt.get("compression") is not False):
                raise ValueError("raw failed attempt metadata violates the frozen transport contract")
            if (folder / "response.json").exists():
                if not (folder / "timing.json").is_file():
                    raise ValueError("raw response has no timing evidence")
            elif not (folder / "failure.json").is_file():
                raise ValueError("attempt with no provider response is missing failure evidence")


class ReplayClient:
    def __init__(self, expected: list[dict[str, Any]], *, allow_exhaustion: bool = False,
                 require_wire: bool = False):
        self.expected = expected
        self.index = 0
        self.allow_exhaustion = allow_exhaustion
        self.require_wire = require_wire

    async def choose(self, *, request_id: str, state: Any, instructions: str,
                     criteria: dict[str, str]) -> dict[str, Any]:
        if self.index >= len(self.expected):
            if self.allow_exhaustion:
                raise ReplayExhausted
            raise ValueError("replay produced an unexpected extra model request")
        row = self.expected[self.index]
        self.index += 1
        for key, value in (("request_id", request_id), ("state", state),
                           ("instructions", instructions), ("criteria", criteria)):
            if row.get(key) != value:
                raise ValueError(f"replay request field changed: {key}")
        response = row.get("response")
        if not isinstance(response, dict):
            raise ValueError("replay response evidence is missing")
        if self.require_wire and not response.get("_wire"):
            raise ValueError("live replay step is missing its exact wire evidence")
        if response.get("_wire"):
            from capclaims.conditional_behavior.transport import serialize_body, strict_json
            body = await serialize_body(state=state, instructions=instructions, criteria=criteria)
            wire = response["_wire"]
            folder = Path(wire.get("directory", ""))
            evidence_files = {"request_sha256": "request.json", "response_sha256": "response.json",
                              "timing_sha256": "timing.json", "attempt_sha256": "attempt.json"}
            for digest_key, filename in evidence_files.items():
                evidence = folder / filename
                if not evidence.is_file() or _sha(evidence.read_bytes()) != wire.get(digest_key):
                    raise ValueError(f"raw transport evidence changed or is missing: {filename}")
            raw_body = (folder / "request.json").read_bytes()
            if (strict_json(raw_body) != strict_json(body) or (folder / "failure.json").exists()
                    or _sha(raw_body) != wire.get("request_sha256")):
                raise ValueError("transport request bytes differ or failed evidence was marked")
            stored_response = strict_json((folder / "response.json").read_bytes())
            if any(stored_response.get(key) != response.get(key) for key in ("model", "usage", "answers")):
                raise ValueError("recorded provider response differs from raw transport evidence")
            attempt = strict_json((folder / "attempt.json").read_bytes())
            timing = strict_json((folder / "timing.json").read_bytes())
            if (attempt.get("request_id") != request_id or attempt.get("request_sha256") != _sha(raw_body)
                    or attempt.get("retry_count") != 0 or attempt.get("compression") is not False
                    or timing.get("status_code") != 200):
                raise ValueError("transport attempt or timing facts violate the frozen protocol")
        return response


class ReplayExhausted(Exception):
    """Partial run has durable requests but no response for its final attempted stage."""


async def replay_preview(preview_dir: str | Path, run_dir: str | Path,
                         reviewed_sha256: str) -> dict[str, Any]:
    run_dir = Path(run_dir)
    manifest = verify_preview(preview_dir, reviewed_sha256)
    events = _read_events(run_dir)
    if (not events or events[0]["type"] != "RUN_START"
            or events[0]["data"].get("preview_sha256") != reviewed_sha256
            or events[0]["data"].get("mode") not in {"MOCK", "LIVE"}
            or events[0]["data"].get("seed") != manifest["seed"]
            or events[0]["data"].get("schedule_sha256") != _sha(_canonical(manifest["schedule"]))):
        raise ValueError("run evidence has no start event")
    if events[-1]["type"] != "RUN_END":
        raise ValueError("run evidence is missing its terminal summary")
    run_summary = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    terminal = events[-1]
    if (run_summary.get("final_event_sha256") != terminal["sha256"] or
            _canonical({key: value for key, value in run_summary.items() if key != "final_event_sha256"})
            != _canonical(terminal["data"])):
        raise ValueError("run summary differs from terminal event evidence")
    complete = [event["data"] for event in events if event["type"] == "TRAJECTORY_COMPLETE"]
    mode = events[0]["data"]["mode"]
    expected_batches = [{"batch_index": index, "repeat": repeat,
        "repeat_block_offset": offset, "batch_n": min(MAX_BATCH, 54 - offset)}
        for index, (repeat, offset) in enumerate((
            (repeat, offset) for repeat in (0, 1) for offset in range(0, 54, MAX_BATCH)))]
    starts = [e["data"] for e in events if e["type"] == "BATCH_START"]
    if len(starts) > len(expected_batches):
        raise ValueError("run has more batch starts than the frozen schedule")
    for actual, expected in zip(starts, expected_batches):
        if any(actual.get(field) != value for field, value in expected.items()):
            raise ValueError("run batch boundary or repeat block differs from frozen schedule")
    ends = [e["data"] for e in events if e["type"] == "BATCH_END"]
    timeouts = [e["data"] for e in events if e["type"] == "BATCH_TIMEOUT"]
    if len(ends) + len(timeouts) != len(starts) or len(timeouts) > 1:
        raise ValueError("each started batch must have exactly one end or timeout record")
    terminal_batches = ends + timeouts
    terminal_batches.sort(key=lambda e: e["batch_index"])
    for actual, expected in zip(terminal_batches, expected_batches):
        if any(actual.get(field) != value for field, value in expected.items()
               if field != "repeat_block_offset"):
            raise ValueError("batch terminal record differs from its frozen block")
    stop_positions = [i for i, event in enumerate(events)
                      if event["type"] in {"STOP_AFTER_TECHNICAL_FAILURE", "BATCH_TIMEOUT"}]
    if stop_positions and any(event["type"] == "TRAJECTORY_COMPLETE"
                              for event in events[stop_positions[0] + 1:]):
        raise ValueError("trajectory completed after technical stop")
    schedule = {(r["person_id"], r["repeat"], r["arm"]): r for r in manifest["schedule"]}
    requests: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    steps: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        if event["type"] in {"TRAJECTORY_REQUEST", "TRAJECTORY_STEP"}:
            data = event["data"]
            key = (data["person_id"], data["repeat"], data["arm"])
            if key not in schedule:
                raise ValueError("request event references a nonfrozen trajectory")
            if event["type"] == "TRAJECTORY_REQUEST":
                requests[key].append(data["request"])
            else:
                steps[key].append(data["step"])
    seen = set()
    failed = set()
    stopped = False
    for result in complete:
        if stopped:
            raise ValueError("trajectory completed after technical stop")
        key = (result["person_id"], result["repeat"], result["arm"])
        if key not in schedule or key in seen:
            raise ValueError("run contains an unknown or duplicate trajectory")
        seen.add(key)
        expected = result["trace"]
        if steps[key] != expected:
            raise ValueError("durable per-step events differ from the completed trajectory trace")
        expected_requests = [{field: step[field] for field in
            ("coordinate", "request_id", "state", "instructions", "criteria", "candidate_order", "draw")}
            for step in expected]
        if requests[key] != expected_requests:
            raise ValueError("durable pre-send request events differ from the trajectory")
        client = ReplayClient(expected, require_wire=mode == "LIVE")
        recomputed_requests: list[dict[str, Any]] = []
        replayed = await sample_trajectory(client, schedule[key], seed=manifest["seed"],
                                           on_request=recomputed_requests.append)
        if client.index != len(expected) or _canonical(replayed) != _canonical(result):
            raise ValueError("trajectory probability draw or portfolio fails strict replay")
        if _canonical(recomputed_requests) != _canonical(requests[key]):
            raise ValueError("candidate masks or pre-send request states fail replay")
        semantic = replay_choices({"Q": schedule[key]["current_state"]["Q"]},
            result["actions"], interface="J2")
        if semantic["orders"] != result["orders"]:
            raise ValueError("portfolio differs from strict J2 action replay")
    for event in events:
        if event["type"] == "STOP_AFTER_TECHNICAL_FAILURE":
            stopped = True
        if event["type"] == "TRAJECTORY_FAILURE":
            data = event["data"]
            key = (data["person_id"], data["repeat"], data["arm"])
            if key not in schedule or key in failed or key in seen:
                raise ValueError("failure event has unknown or duplicate trajectory identity")
            failed.add(key)
        if event["type"] == "BATCH_TIMEOUT":
            stopped = True
    if seen & failed or (seen | failed) > set(schedule):
        raise ValueError("completion/failure event sets conflict with the frozen schedule")
    failed_terminal_request_ids: set[str] = set()
    for key in failed:
        client = ReplayClient(steps[key], allow_exhaustion=True, require_wire=mode == "LIVE")
        recomputed_requests: list[dict[str, Any]] = []
        recomputed_steps: list[dict[str, Any]] = []
        try:
            await sample_trajectory(client, schedule[key], seed=manifest["seed"],
                                    on_request=recomputed_requests.append,
                                    on_step=recomputed_steps.append)
        except ReplayExhausted:
            if client.index != len(steps[key]):
                raise ValueError("failure occurred before the recorded successful response prefix")
        else:
            raise ValueError("failed trajectory replays to completion without a final failed request")
        if (not requests[key] or len(requests[key]) != len(steps[key]) + 1
                or _canonical(recomputed_steps) != _canonical(steps[key])
                or _canonical(recomputed_requests) != _canonical(requests[key])):
            raise ValueError("partial trajectory requests or successful prefixes fail strict replay")
        failed_terminal_request_ids.add(requests[key][-1]["request_id"])
    if mode == "LIVE":
        request_ids = {request["request_id"] for rows in requests.values() for request in rows}
        physical_ids = _verify_live_ledger(reviewed_sha256, request_ids,
                                           run_summary.get("cost_summary") or {})
        await _verify_raw_requests(run_dir, requests, physical_ids)
        terminal_request_status_counts = {
            "PHYSICAL_ATTEMPT": sum(request_id in physical_ids for request_id in failed_terminal_request_ids),
            "NOT_RESERVED/NOT_DISPATCHED": sum(request_id not in physical_ids
                                                for request_id in failed_terminal_request_ids)}
    else:
        terminal_request_status_counts = None
    complete_status = len(seen) == 108 and not failed and run_summary["status"] == "COMPLETE"
    if run_summary["status"] == "COMPLETE" and not complete_status:
        raise ValueError("run claims completion without 108 validated trajectories")
    return {"status": "REPLAY_VALID" if complete_status else "INCOMPLETE",
        "mode": mode, "planned_trajectories": 108, "completed_trajectories": len(seen),
        "failed_trajectories": len(failed), "cost_summary": run_summary.get("cost_summary"),
        "failed_terminal_request_status_counts": terminal_request_status_counts}

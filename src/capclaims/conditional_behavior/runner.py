"""One-shot execution ledger with a closed live-cost gate and strict replay."""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import sqlite3
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import VERSION
from .protocol import validate_orders
from .study import _canonical, _sha_bytes, verify_preview

MAX_REQUESTS = 29_888
MAX_REAL_REQUESTS = 23_936
MAX_SYNTHETIC_REQUESTS = 5_952
MAX_BODY_BYTES = 12_000
MAX_IN_FLIGHT = 10
MAX_RPS = 40
MAX_BYTES_PER_SECOND = 50_000
MAX_WALL_SECONDS = 600


def _append(path: Path, value: Any, *, durable: bool = True, stream: Any = None) -> bytes:
    data = _canonical(value) + b"\n"
    if stream is not None:
        stream.write(data)
        if durable:
            stream.flush()
            os.fsync(stream.fileno())
        return data
    fd = os.open(path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
    try:
        view = memoryview(data)
        while view:
            view = view[os.write(fd, view):]
        if durable:
            os.fsync(fd)
    finally:
        os.close(fd)
    return data


def _write(path: Path, value: Any) -> bytes:
    data = _canonical(value) + b"\n"
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    return data


def _provider_attempt_summary(evidence_dir: str | Path | None,
                              ledger_path: str | Path | None) -> dict[str, Any]:
    """Count ambiguous requests from durable per-request and global-ledger evidence."""
    folders = []
    if evidence_dir is not None and Path(evidence_dir).exists():
        folders = [path for path in Path(evidence_dir).iterdir() if path.is_dir()]
    requests = []
    response_n = unknown_n = 0
    for folder in folders:
        attempt_path = folder / "attempt.json"
        if not attempt_path.is_file():
            continue
        try:
            attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            unknown_n += 1
            continue
        request_id = attempt.get("request_id")
        requests.append(request_id)
        if ((folder / "response.json").is_file() and (folder / "timing.json").is_file()
                and not (folder / "failure.json").exists()):
            response_n += 1
        else:
            unknown_n += 1
    ledger_states: dict[str, int] = {}
    if ledger_path is not None and Path(ledger_path).is_file() and requests:
        # Use a read-only connection; never create or mutate the shared budget ledger here.
        uri = Path(ledger_path).resolve().as_uri() + "?mode=ro"
        db = sqlite3.connect(uri, uri=True, timeout=5)
        try:
            for request_id in requests:
                row = db.execute("SELECT state FROM attempts WHERE request_id=?", (request_id,)).fetchone()
                if row is None:
                    ledger_states["missing"] = ledger_states.get("missing", 0) + 1
                else:
                    ledger_states[row[0]] = ledger_states.get(row[0], 0) + 1
        finally:
            db.close()
    unfinished = (ledger_states.get("reserved", 0) + ledger_states.get("unknown", 0)
        + ledger_states.get("missing", 0)
        if ledger_states else unknown_n)
    return {"durable_attempt_files": len(requests), "responses_recorded": response_n,
        "ambiguous_or_unfinished_attempts": unfinished, "ledger_states": ledger_states}


async def _call(sampler: Any, row: Mapping[str, Any]) -> Any:
    result = sampler(row["observation"], row)
    result = await result if hasattr(result, "__await__") else result
    if not isinstance(result, Mapping):
        return result
    return {**result, "case_id": row["case_id"]}


async def execute_preview(preview_dir: str | Path, run_dir: str | Path,
                          reviewed_preview_sha256: str, *, fake: bool = True,
                          sampler: Any = None, cost_attestation: Mapping[str, Any] | None = None,
                          attestation_verifier: Any = None,
                          transport_evidence_dir: str | Path | None = None,
                          cost_ledger_path: str | Path | None = None) -> dict[str, Any]:
    """Execute every frozen row once; append evidence immediately; never retry/resume."""
    if os.environ.get("CAPCLAIMS_DIAGNOSTICS") == "1":
        import faulthandler
        faulthandler.dump_traceback_later(10, repeat=True)
    manifest = verify_preview(preview_dir, reviewed_preview_sha256)
    if not fake:
        if (not isinstance(cost_attestation, Mapping) or not callable(attestation_verifier)
                or not attestation_verifier(cost_attestation, manifest)):
            raise ValueError("live execution requires verified cost-bound attestation")
        if not callable(sampler):
            raise ValueError("live execution requires the verified provider adapter")
    if fake and sampler is None:
        # Deliberately labelled local harness only; this is not a model result.
        sampler = lambda observation, coordinates: {"orders": [], "provider_requests": 0,
            "raw_wires": [], "trace": [{"kind": "synthetic_empty_portfolio"}]}
    if not callable(sampler):
        raise ValueError("a sampler/provider adapter is required")
    output = Path(run_dir)
    if output.exists():
        raise FileExistsError("runs are immutable; automatic resumption is disabled")
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    _write(output / "receipt.json", {"schema": VERSION, "preview_sha256": manifest["preview_sha256"],
        "mode": "fake" if fake else "provider", "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "started_monotonic": started, "status": "RUNNING",
        "transport_evidence_dir": str(Path(transport_evidence_dir).resolve()) if transport_evidence_dir else None,
        "cost_ledger_path": str(Path(cost_ledger_path).resolve()) if cost_ledger_path else None})
    inference = json.loads((Path(preview_dir) / "inputs.json").read_text(encoding="utf-8"))
    chain = "0" * 64
    total_requests = 0
    rows_done = 0
    failure_type = None
    events_path = output / "events.jsonl"
    wires_path = output / "raw-wires.jsonl"
    attempts_path = output / "attempts.jsonl"
    case_states_path = output / "case-states.jsonl"
    for path in (events_path, wires_path, attempts_path, case_states_path):
        path.touch(exist_ok=False)
    streams = [path.open("ab", buffering=64 * 1024) for path in
               (events_path, wires_path, attempts_path, case_states_path)]
    event_stream, wire_stream, attempt_stream, case_state_stream = streams
    case_latest: dict[str, dict[str, Any]] = {}

    def case_state(row: Mapping[str, Any], status_value: str, category: str | None = None) -> None:
        value = {"case_id": row["case_id"], "person_id": row["person_id"],
            "target_round": row["target_round"], "arm": row["arm"], "sample": row["sample"],
            "status": status_value}
        if category is not None:
            value["failure_category"] = category
        _append(case_states_path, value, durable=True, stream=case_state_stream)
        case_latest[row["case_id"]] = value

    def classify_failure(exc: BaseException) -> str:
        if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
            return "timeout"
        if isinstance(exc, asyncio.CancelledError):
            return "cancellation"
        message = str(exc).lower()
        if "budget" in message or "cap exhausted" in message:
            return "budget"
        return "technical"

    for row in inference["inputs"]:
        case_state(row, "planned")
    def commit(row: Mapping[str, Any], result: Mapping[str, Any]) -> None:
        nonlocal chain, total_requests, rows_done
        if not isinstance(result, Mapping):
            raise ValueError("sample provider returned a malformed result")
        observation = row["observation"]
        q = observation.get("Q", observation.get("own_q_units"))
        orders = list(validate_orders(result.get("orders", []), initial_q=q))
        physical = result.get("provider_requests", 0)
        if type(physical) is not int or physical < 0:
            raise ValueError("provider request count is invalid")
        if total_requests + physical > MAX_REQUESTS:
            raise ValueError("physical request cap would be exceeded")
        total_requests += physical
        wires = result.get("raw_wires", [])
        if not isinstance(wires, list):
            raise ValueError("raw wire evidence must be a list")
        wire_hashes = []
        for wire in wires:
            if not isinstance(wire, Mapping):
                raise ValueError("raw wire row must be a mapping")
            body = wire.get("request_body")
            if isinstance(body, str) and len(body.encode("utf-8")) > MAX_BODY_BYTES:
                raise ValueError("provider request exceeds the 12,000 byte cap")
            raw_bytes = _append(wires_path, {"case_id": row["case_id"], **dict(wire)}, durable=not fake, stream=wire_stream)
            wire_hashes.append(_sha_bytes(raw_bytes))
            _append(attempts_path, {"case_id": row["case_id"], "request_index": len(wire_hashes),
                "reserved_bytes": MAX_BODY_BYTES, "status": wire.get("status", "recorded")}, durable=not fake, stream=attempt_stream)
        interface = row.get("interface", observation.get("protocol_interface", observation.get("interface")))
        event = {"sequence": rows_done, "case_id": row["case_id"], "person_id": row["person_id"],
            "target_round": row["target_round"], "arm": row["arm"], "sample": row["sample"],
            "interface": interface, "status": "complete", "orders": orders,
            "provider_requests": physical, "wire_sha256": wire_hashes,
            "trace": result.get("trace", []), "previous_event_sha256": chain}
        event["event_sha256"] = _sha_bytes(_canonical(event))
        raw_event = _append(events_path, event, durable=not fake, stream=event_stream)
        chain = _sha_bytes(raw_event)
        rows_done += 1
        case_state(row, "completed")

    async def execute_batch(batch: list[Mapping[str, Any]]) -> None:
        for row in batch:
            case_state(row, "started")
        tasks = [asyncio.create_task(_call(sampler, row)) for row in batch]
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
        except asyncio.CancelledError:
            for task in tasks:
                if not task.done():
                    task.cancel()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            # Preserve every sibling that finished before the batch timeout.
            for row, result in zip(batch, results, strict=True):
                if isinstance(result, BaseException):
                    continue
                try:
                    if not isinstance(result, Mapping) or result.get("case_id") != row["case_id"]:
                        raise ValueError("sample task result is missing its frozen case binding")
                    commit(row, result)
                except BaseException:
                    pass
            raise
        failures = []
        for row, result in zip(batch, results, strict=True):
            if isinstance(result, BaseException):
                category = classify_failure(result)
                case_state(row, "cancelled" if isinstance(result, asyncio.CancelledError) else "failed", category)
                failures.append(result)
                continue
            try:
                if not isinstance(result, Mapping) or result.get("case_id") != row["case_id"]:
                    raise ValueError("sample task result is missing its frozen case binding")
                commit(row, result)
            except BaseException as exc:
                case_state(row, "failed", classify_failure(exc))
                failures.append(exc)
        if failures:
            raise failures[0]

    try:
        for start in range(0, len(inference["inputs"]), MAX_IN_FLIGHT):
            batch = inference["inputs"][start:start + MAX_IN_FLIGHT]
            # Batches are sequential and each receives the full frozen 600s window.
            await asyncio.wait_for(execute_batch(batch), timeout=MAX_WALL_SECONDS)
    except BaseException as exc:
        failure_type = type(exc).__name__
        category = classify_failure(exc)
        for row in inference["inputs"]:
            current = case_latest[row["case_id"]]["status"]
            if current == "planned":
                case_state(row, "not_started", "study_stopped")
            elif current == "started":
                case_state(row, "failed" if category == "timeout" else "cancelled", category)
        for stream in streams:
            stream.close()
        failed = {"sequence": rows_done, "status": "failed", "failure_type": failure_type,
            "completed_before_failure": rows_done, "previous_event_sha256": chain}
        failed["event_sha256"] = _sha_bytes(_canonical(failed))
        failure_bytes = _append(events_path, failed)
        chain = _sha_bytes(failure_bytes)
        _write(output / "invalid.json", {"status": "INVALID", "completed_before_failure": rows_done,
            "failure_type": failure_type})
    else:
        for stream in streams:
            stream.close()
    receipt_sha = _sha_bytes((output / "receipt.json").read_bytes())
    status = "COMPLETE" if rows_done == 680 and failure_type is None else "INCOMPLETE"
    attempt_summary = _provider_attempt_summary(transport_evidence_dir, cost_ledger_path)
    physical_attempted = attempt_summary["durable_attempt_files"]
    terminal = {"status": status, "mode": "fake" if fake else "provider",
        "preview_sha256": manifest["preview_sha256"], "completed_rows": rows_done,
        "failure_type": failure_type,
        "provider_requests": physical_attempted, "physical_attempted_requests": physical_attempted,
        "completed_trace_requests": total_requests,
        "event_chain_head": chain, "receipt_sha256": receipt_sha,
        "events_sha256": _sha_bytes(events_path.read_bytes()),
        "raw_wires_sha256": _sha_bytes(wires_path.read_bytes()),
        "attempts_sha256": _sha_bytes(attempts_path.read_bytes()),
        "case_states_sha256": _sha_bytes(case_states_path.read_bytes()),
        "case_state_counts": {status_value: sum(1 for value in case_latest.values()
            if value["status"] == status_value) for status_value in
            ("planned", "started", "completed", "failed", "cancelled", "not_started")},
        "provider_attempt_evidence": attempt_summary}
    _write(output / "terminal.json", terminal)
    return {"status": status, "samples": rows_done, "provider_requests": physical_attempted,
        "physical_attempted_requests": physical_attempted, "completed_trace_requests": total_requests,
        "mode": terminal["mode"], "case_state_counts": terminal["case_state_counts"],
        "provider_attempt_evidence": terminal["provider_attempt_evidence"]}


def execute_preview_sync(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return asyncio.run(execute_preview(*args, **kwargs))


def _strict_provider_replay(run_root: Path, provider_events: list[tuple[Mapping[str, Any], Mapping[str, Any]]],
                            manifest: Mapping[str, Any]) -> None:
    """Rebuild every provider stage from exact saved bytes and replay its draw."""
    import base64
    from .sampling import sample_portfolio
    from .transport import serialize_body, strict_json

    receipt = json.loads((run_root / "receipt.json").read_text(encoding="utf-8"))
    evidence_value = receipt.get("transport_evidence_dir")
    if not isinstance(evidence_value, str) or not evidence_value:
        raise ValueError("provider receipt is missing its durable transport evidence root")
    evidence_root = Path(evidence_value).resolve(strict=True)
    replay_tasks = []

    class ReplayClient:
        def __init__(self, event: Mapping[str, Any]):
            self.event = event
            self.trace = [row for row in event.get("trace", []) if row.get("origin") != "synthetic_oracle"]
            self.index = 0

        async def choose(self, *, request_id: str, state: Any, instructions: str,
                         criteria: dict[str, str]) -> dict[str, Any]:
            if self.index >= len(self.trace):
                raise ValueError("sample replay generated an unrecorded provider stage")
            row = self.trace[self.index]
            self.index += 1
            if (row.get("request_id") != request_id or row.get("state") != state
                    or row.get("instructions") != instructions or row.get("criteria") != criteria):
                raise ValueError("sample replay state, prompt, candidate set, or request ID differs")
            wire = row.get("raw_response", {}).get("_wire")
            if not isinstance(wire, Mapping) or wire.get("request_id") != request_id:
                raise ValueError("provider stage lacks bound durable wire evidence")
            folder = Path(wire.get("directory", "")).resolve(strict=True)
            if folder.parent != evidence_root or folder.name != hashlib.sha256(request_id.encode()).hexdigest():
                raise ValueError("provider wire evidence path escapes its run evidence root")
            files = {"request.json": "request_sha256", "response.json": "response_sha256",
                     "timing.json": "timing_sha256", "attempt.json": "attempt_sha256"}
            contents = {}
            for filename, field in files.items():
                data = (folder / filename).read_bytes()
                if _sha_bytes(data) != wire.get(field):
                    raise ValueError("durable provider evidence digest mismatch")
                contents[filename] = data
            rebuilt = await serialize_body(state=state, instructions=instructions, criteria=criteria)
            if rebuilt != contents["request.json"]:
                raise ValueError("reconstructed SDK request bytes differ from the captured request")
            attempt = strict_json(contents["attempt.json"])
            if (attempt.get("request_id") != request_id or attempt.get("request_sha256") != wire["request_sha256"]
                    or attempt.get("retry_count") != 0 or attempt.get("body_bytes") != len(rebuilt)):
                raise ValueError("provider attempt reservation metadata is inconsistent")
            actual = strict_json(contents["response.json"])
            saved = dict(row["raw_response"])
            saved.pop("_wire", None)
            if actual != saved:
                raise ValueError("saved provider response differs from durable raw response bytes")
            return {**actual, "_wire": dict(wire)}

    async def validate_all() -> None:
        for event, expected in provider_events:
            replay = ReplayClient(event)
            row = dict(expected)
            actual = await sample_portfolio(replay, row, seed=int(manifest["seed"]), category="human")
            if replay.index != len(replay.trace):
                raise ValueError("provider event contains unused stage traces")
            if actual.get("orders") != event.get("orders"):
                raise ValueError("strict probability and draw replay differs from saved portfolio")
            actual_trace = actual.get("trace", [])
            if len(actual_trace) != len(replay.trace):
                raise ValueError("strict provider replay has an unexpected stage count")
            for observed, saved in zip(actual_trace, replay.trace):
                for field in ("stage", "study_version", "random_draw", "state", "instructions", "criteria",
                              "candidate_order", "raw_probabilities", "selected", "model", "usage", "request_id"):
                    if observed.get(field) != saved.get(field):
                        raise ValueError("strict stage replay differs from the recorded trace")
    asyncio.run(validate_all())


def replay_run(preview_dir: str | Path, run_dir: str | Path) -> dict[str, Any]:
    manifest = verify_preview(preview_dir, json.loads((Path(preview_dir) / "manifest.json").read_text())["preview_sha256"])
    root = Path(run_dir)
    terminal = json.loads((root / "terminal.json").read_text(encoding="utf-8"))
    if terminal.get("status") not in {"COMPLETE", "INCOMPLETE"} or terminal.get("preview_sha256") != manifest["preview_sha256"]:
        raise ValueError("run terminal is invalid or belongs to another preview")
    expected_hashes = {"events.jsonl": "events_sha256", "raw-wires.jsonl": "raw_wires_sha256",
                       "attempts.jsonl": "attempts_sha256", "receipt.json": "receipt_sha256",
                       "case-states.jsonl": "case_states_sha256"}
    for name, field in expected_hashes.items():
        if _sha_bytes((root / name).read_bytes()) != terminal.get(field):
            raise ValueError("run evidence digest mismatch")
    inputs = json.loads((Path(preview_dir) / "inputs.json").read_text(encoding="utf-8"))["inputs"]
    lines = [line for line in (root / "events.jsonl").read_text(encoding="utf-8").splitlines() if line]
    if not 0 < len(lines) <= len(inputs):
        raise ValueError("run has no events or exceeds the frozen sample schedule")
    previous = "0" * 64
    events = []
    expected_by_case = {row["case_id"]: row for row in inputs}
    seen: set[str] = set()
    provider_events = []
    for seq, line in enumerate(lines):
        event = json.loads(line)
        digest = event.pop("event_sha256", None)
        if digest != _sha_bytes(_canonical(event)) or event.get("previous_event_sha256") != previous:
            raise ValueError("run event hash chain is invalid")
        if event.get("sequence") != seq:
            raise ValueError("run event sequence is invalid")
        if event.get("status") == "failed":
            if seq != len(lines) - 1 or event.get("completed_before_failure") != len(seen):
                raise ValueError("failure marker must be the final event and preserve completed work")
        else:
            case_id = event.get("case_id")
            expected = expected_by_case.get(case_id)
            if expected is None or case_id in seen or event.get("status") != "complete":
                raise ValueError("run event is unexpected, duplicated, or incomplete")
            expected_q = expected["observation"].get("Q", expected["observation"].get("own_q_units"))
            orders = validate_orders(event.get("orders", []), initial_q=expected_q)
            if list(orders) != event.get("orders"):
                raise ValueError("saved portfolio differs from strict order replay")
            if terminal["mode"] == "provider":
                provider_events.append((event, expected))
            seen.add(case_id)
        event["event_sha256"] = digest
        events.append(event)
        previous = _sha_bytes(_canonical(event) + b"\n")
    if previous != terminal.get("event_chain_head"):
        raise ValueError("terminal event chain head mismatch")
    if terminal["status"] == "COMPLETE" and len(seen) != 680:
        raise ValueError("complete terminal omitted frozen samples")
    if terminal["status"] == "INCOMPLETE" and len(seen) == 680:
        raise ValueError("incomplete terminal has all sample events")
    lifecycle = [json.loads(line) for line in (root / "case-states.jsonl").read_text(encoding="utf-8").splitlines() if line]
    latest: dict[str, dict[str, Any]] = {}
    allowed = {"planned": {"started", "not_started"}, "started": {"completed", "failed", "cancelled"}}
    for row in lifecycle:
        case_id = row.get("case_id")
        if case_id not in expected_by_case:
            raise ValueError("case lifecycle contains an unknown frozen case")
        expected = expected_by_case[case_id]
        if any(row.get(key) != expected[key] for key in ("person_id", "target_round", "arm", "sample")):
            raise ValueError("case lifecycle identity differs from the frozen schedule")
        status_value = row.get("status")
        previous_status = latest.get(case_id, {}).get("status")
        if previous_status is None:
            if status_value != "planned":
                raise ValueError("every frozen case must start as planned")
        elif status_value not in allowed.get(previous_status, set()):
            raise ValueError("case lifecycle transition is invalid")
        if status_value in {"failed", "cancelled", "not_started"} and not row.get("failure_category"):
            raise ValueError("noncomplete case lifecycle needs a failure category")
        latest[case_id] = row
    if set(latest) != set(expected_by_case):
        raise ValueError("case lifecycle omits frozen samples")
    state_counts = {value: sum(1 for row in latest.values() if row["status"] == value)
        for value in ("planned", "started", "completed", "failed", "cancelled", "not_started")}
    if state_counts != terminal.get("case_state_counts") or state_counts["completed"] != len(seen):
        raise ValueError("terminal case lifecycle counts differ from persisted case evidence")
    if state_counts["planned"] or state_counts["started"]:
        raise ValueError("run terminal has unresolved case-level work")
    if terminal["status"] == "COMPLETE" and any(value for key, value in state_counts.items()
            if key != "completed"):
        raise ValueError("complete run contains noncompleted frozen cases")
    receipt = json.loads((root / "receipt.json").read_text(encoding="utf-8"))
    attempt_summary = _provider_attempt_summary(receipt.get("transport_evidence_dir"),
                                               receipt.get("cost_ledger_path"))
    if attempt_summary != terminal.get("provider_attempt_evidence"):
        raise ValueError("provider attempt summary differs from durable transport evidence")
    trace_requests = sum(event.get("provider_requests", 0) for event in events
        if event.get("status") == "complete")
    physical_attempts = attempt_summary["durable_attempt_files"]
    if (trace_requests != terminal.get("completed_trace_requests")
            or physical_attempts != terminal.get("physical_attempted_requests")
            or terminal.get("provider_requests") != physical_attempts
            or (terminal["mode"] == "provider" and physical_attempts < trace_requests)):
        raise ValueError("physical provider attempts do not reconcile with completed traces")
    if provider_events:
        _strict_provider_replay(root, provider_events, manifest)
    elif terminal["mode"] == "provider" and terminal["status"] == "COMPLETE":
        raise ValueError("provider run has no strict request trace")
    return {"status": "REPLAY_VALID" if terminal["status"] == "COMPLETE" else "REPLAY_INCOMPLETE",
        "sample_n": len(seen), "planned_sample_n": len(inputs), "events": events,
        "mode": terminal["mode"], "provider_requests": terminal["provider_requests"],
        "physical_attempted_requests": terminal["physical_attempted_requests"],
        "completed_trace_requests": terminal["completed_trace_requests"],
        "case_state_counts": state_counts, "provider_attempt_evidence": attempt_summary}

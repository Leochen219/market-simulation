"""Frozen paired study for named complete-order history and fixed explanation."""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any, Mapping

from capclaims.order_presentation import ARMS, CONTROL, EXPLICIT, order_prompt, render_order_state
from capclaims.paired_portfolio_engine import ExplanationBackend, PairedPortfolioEngine
from capclaims.paired_portfolio_engine import serialize_explanation_wire

ROOT = Path(__file__).resolve().parents[2]
PARENT_SHA256 = "3034f77db130fd4b235cf956651fb276849392dd673cc196d7ab9d5d85d43bdf"
PARENT_RELATIVE = Path("artifacts/participant-data/field-alignment-preview-root-25-r0")
SOURCE_POLICY_RELATIVE = Path("artifacts/participant-data/quote-perspective-24/source-policy-authorized.json")
SEED = 20261025
REPEATS = (0, 1)
SESSION_N = 24
MAX_REQUESTS = 840
MAX_RESERVED_BYTES = 6_300_000
MAX_BODY = 7_500
MAX_IN_FLIGHT = 10
MAX_RPS = 40
MAX_BYTES_PER_SECOND = 50_000
MAX_WALL_SECONDS = 600
MAX_ORDERS = 5


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha(value: Any) -> str:
    from capclaims.randomness import canonical_json
    return _sha_bytes(canonical_json(value).encode("utf-8"))


def _write(path: Path, value: Any) -> None:
    from capclaims.randomness import canonical_json
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write((canonical_json(value) + "\n").encode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())


def _write_idempotent(path: Path, value: Any) -> None:
    from capclaims.randomness import canonical_json
    payload = (canonical_json(value) + "\n").encode("utf-8")
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError("private derived scores already exist with different bytes")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _private(path: str | Path) -> Path:
    resolved = Path(path).resolve()
    private = (ROOT / "artifacts" / "participant-data").resolve()
    if resolved != private and private not in resolved.parents:
        raise ValueError("order-explanation artifacts must remain under ignored participant-data")
    return resolved


def _session_schedule(people: list[str]) -> list[dict[str, Any]]:
    rng = random.Random(SEED)
    order = sorted(people)
    rng.shuffle(order)
    repeats = list(REPEATS)
    rng.shuffle(repeats)
    rows = []
    for repeat in repeats:
        for person in order:
            arms = list(ARMS)
            rng.shuffle(arms)
            for arm in arms:
                rows.append({"session_id": f"{person}:{arm}:r{repeat}", "person_id": person,
                             "arm": arm, "repeat": repeat})
    return rows


def _code_hashes() -> dict[str, str]:
    from capclaims.field_alignment_study import _code_hashes as frozen_hashes
    hashes = frozen_hashes()
    for name in ("src/capclaims/order_presentation.py", "src/capclaims/paired_portfolio_engine.py",
                 "src/capclaims/order_explanation_study.py",
                 "docs/superpowers/specs/2026-10-25-order-explanation-design.md",
                 "tools/probes/jev_order_explanation.py"):
        path = ROOT / name
        if not path.is_file():
            raise ValueError(f"study freeze file is missing: {name}")
        hashes[name] = _sha_bytes(path.read_bytes())
    return hashes


def build_preview(parent_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Freeze the 24 paired sessions and validate maximum-width SDK bodies."""
    parent, output = _private(parent_dir), _private(output_dir)
    from capclaims.field_alignment_study import verify_preview as verify_parent
    parent_manifest = verify_parent(parent, PARENT_SHA256)
    if output.exists():
        raise FileExistsError("preview is immutable")
    inputs = json.loads((parent / "inputs.json").read_text(encoding="utf-8"))
    labels_raw = (parent / "scoring-targets.json").read_bytes()
    labels = json.loads(labels_raw.decode("utf-8"))
    if _sha_bytes(labels_raw) != parent_manifest.get("scoring_targets_sha256"):
        raise ValueError("parent score targets fail their frozen checksum")
    from capclaims.portfolio_prediction import _validate_base
    bases = {key: _validate_base(value) for key, value in inputs.get("bases", {}).items()}
    if len(bases) != 6 or any([x["round"] for x in b["history"]] != [6, 7, 8] for b in bases.values()):
        raise ValueError("parent must provide six people with the same formal rounds 6 through 8")
    sessions = _session_schedule(list(bases))
    # Exercise all legal orders prefixes and all renderer widths before execution.
    from capclaims.injection_study import _maximal_legal_requests
    maximum = 0
    for base in bases.values():
        for request in _maximal_legal_requests(base["history"]):
            for arm in ARMS:
                wire = asyncio.run(serialize_explanation_wire(request, arm))
                maximum = max(maximum, len(wire.body))
    if maximum > MAX_BODY:
        raise ValueError("maximum-width order-explanation request exceeds body cap")
    policy_path = ROOT / SOURCE_POLICY_RELATIVE
    policy_bytes = policy_path.read_bytes()
    source = parent_manifest.get("source", {})
    policy_sha = parent_manifest.get("source_policy_sha256")
    if not isinstance(policy_sha, str) or _sha_bytes(policy_bytes) != policy_sha:
        raise ValueError("source policy differs from frozen parent authorization")
    _check_source_policy(policy_sha, source)
    output.mkdir(parents=True, exist_ok=False)
    _write(output / "inputs.json", {"bases": bases, "sessions": sessions})
    _write(output / "scoring-targets.json", labels)
    first = sessions[0]
    from capclaims.provider_choice_trace import build_provider_order_request
    first_base = {**bases[first["person_id"]], "agent_id": bases[first["person_id"]]["agent_id"] + "__freeze"}
    first_req = build_provider_order_request(first_base, ())
    if first_req is None:
        raise ValueError("first session has no initial provider request")
    fixed_prompt = order_prompt(first_req, CONTROL)
    manifest = {"schema": "jev_order_explanation_preview_v1", "parent_preview_sha256": PARENT_SHA256,
        "parent_manifest_sha256": _sha_bytes((parent / "manifest.json").read_bytes()),
        "parent_inputs_sha256": _sha_bytes((parent / "inputs.json").read_bytes()),
        "parent_labels_sha256": _sha_bytes(labels_raw), "source_policy_path": SOURCE_POLICY_RELATIVE.as_posix(),
        "source_policy_sha256": policy_sha, "source": source,
        "source_policy_verified": parent_manifest.get("source_policy_verified") is True,
        "input_sha256": _sha_bytes((output / "inputs.json").read_bytes()),
        "scoring_targets_sha256": _sha_bytes((output / "scoring-targets.json").read_bytes()),
        "sessions": sessions, "session_n": SESSION_N, "people": sorted(bases), "arms": list(ARMS),
        "repeats": list(REPEATS), "schedule_seed": SEED, "model": "jev-1.14.0", "sdk_version": "0.7.5",
        "instruction_profile": "portfolio_third_v1", "control_prompt": fixed_prompt,
        "control_prompt_sha256": _sha_bytes(fixed_prompt.encode()), "preflight_max_body_bytes": maximum,
        "request_cap": MAX_REQUESTS, "reserved_bytes_cap": MAX_RESERVED_BYTES, "body_bytes_cap": MAX_BODY,
        "max_in_flight": MAX_IN_FLIGHT, "rps_cap": MAX_RPS,
        "bytes_per_second_cap": MAX_BYTES_PER_SECOND, "wall_seconds_cap": MAX_WALL_SECONDS,
        "timeout_seconds": 30, "retry_cap": 0, "code_sha256": _code_hashes()}
    manifest["preview_sha256"] = _sha(manifest)
    _write(output / "manifest.json", manifest)
    return manifest


def verify_preview(preview_dir: str | Path, reviewed_sha256: str) -> dict[str, Any]:
    root = Path(preview_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    digest = manifest.pop("preview_sha256", None)
    if digest != reviewed_sha256 or digest != _sha(manifest):
        raise ValueError("preview is not bound to the reviewed digest")
    manifest["preview_sha256"] = digest
    if manifest.get("parent_preview_sha256") != PARENT_SHA256 or manifest.get("code_sha256") != _code_hashes():
        raise ValueError("frozen parent or code changed")
    inputs_raw = (root / "inputs.json").read_bytes()
    if _sha_bytes(inputs_raw) != manifest.get("input_sha256"):
        raise ValueError("inference inputs changed")
    inputs = json.loads(inputs_raw.decode("utf-8"))
    if inputs.get("sessions") != manifest.get("sessions") or inputs.get("sessions") != _session_schedule(manifest.get("people", [])):
        raise ValueError("session schedule differs from frozen balanced design")
    if len(inputs.get("bases", {})) != 6 or len(manifest.get("sessions", [])) != SESSION_N:
        raise ValueError("preview roster or session count is invalid")
    from capclaims.portfolio_prediction import _validate_base
    if set(inputs["bases"]) != set(manifest.get("people", [])):
        raise ValueError("preview person roster is inconsistent")
    for base in inputs["bases"].values():
        checked = _validate_base(base)
        if [episode["round"] for episode in checked["history"]] != [6, 7, 8]:
            raise ValueError("every preview input must preserve history rounds 6 through 8")
    counts = {(person, arm, repeat): 0 for person in manifest["people"] for arm in ARMS for repeat in REPEATS}
    for row in manifest["sessions"]:
        key = (row.get("person_id"), row.get("arm"), row.get("repeat"))
        if key not in counts or row.get("session_id") != f"{key[0]}:{key[1]}:r{key[2]}":
            raise ValueError("preview session factor is malformed")
        counts[key] += 1
    if any(value != 1 for value in counts.values()):
        raise ValueError("preview does not have exactly one session per person/arm/repeat")
    from capclaims.provider_choice_trace import build_provider_order_request
    first = manifest["sessions"][0]
    first_base = inputs["bases"][first["person_id"]]
    first_request = build_provider_order_request({**first_base, "agent_id": first_base["agent_id"] + "__freeze"}, ())
    if (first_request is None or order_prompt(first_request, CONTROL) != manifest.get("control_prompt")
            or _sha_bytes(manifest.get("control_prompt", "").encode()) != manifest.get("control_prompt_sha256")):
        raise ValueError("frozen C_joint control prompt differs from the unchanged prompt builder")
    _check_source_policy(manifest.get("source_policy_sha256", ""), manifest.get("source", {}))
    if manifest.get("preflight_max_body_bytes", MAX_BODY + 1) > MAX_BODY:
        raise ValueError("preflight request exceeded the body cap")
    from importlib.metadata import version
    if version("typesafe-sdk") != "0.7.5":
        raise ValueError("pinned SDK changed")
    return manifest


def _check_source_policy(expected_sha: str, source: Mapping[str, Any]) -> None:
    raw = (ROOT / SOURCE_POLICY_RELATIVE).read_bytes()
    if _sha_bytes(raw) != expected_sha:
        raise ValueError("source-policy hash differs from the frozen preview")
    policy = json.loads(raw.decode("utf-8"))
    if not all(policy.get(key) is True for key in (
            "codebook_verified", "intended_use_authorized", "provider_transmission_permitted")):
        raise ValueError("source policy does not authorize this use")
    if policy.get("source_kind") != "human" or policy.get("source_sha256") != source.get("source_sha256"):
        raise ValueError("source-policy identity differs from the frozen source")


async def execute_preview(preview_dir: str | Path, run_dir: str | Path, reviewed_preview_sha256: str,
                          fake: bool = False, fake_transport: Any = None) -> dict[str, Any]:
    """Execute all 24 frozen sessions once, invalidating the entire run on failure."""
    manifest = verify_preview(preview_dir, reviewed_preview_sha256)
    output = _private(run_dir)
    if output.exists():
        raise FileExistsError("run cannot be resumed or overwritten")
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    _write(output / "receipt.json", {"preview_sha256": manifest["preview_sha256"],
        "execution_mode": "mock" if fake else "provider", "status": "RUNNING",
        "started_at_utc": datetime.now(timezone.utc).isoformat(), "started_monotonic": started,
        "started_perf_ns": time.perf_counter_ns(), "source_policy_sha256": manifest["source_policy_sha256"]})
    inputs = json.loads((Path(preview_dir) / "inputs.json").read_text(encoding="utf-8"))
    if not fake:
        _check_source_policy(manifest["source_policy_sha256"], manifest["source"])
        from capclaims.backends.jev import load_local_env
        load_local_env()
        if not os.getenv("TYPESAFE_API_KEY"):
            raise ValueError("TYPESAFE_API_KEY is unavailable")
    import httpx2
    if fake_transport is None:
        if fake:
            async def mock(req: Any) -> Any:
                body = json.loads((await req.aread()).decode("utf-8"))
                criteria = body["questions"]["decision"]["criteria"]
                ids = list(criteria)
                selected = next((key for key, label in criteria.items() if label.startswith("STOP:")), ids[-1])
                probs = {key: float(key == selected) for key in ids}
                return httpx2.Response(200, json={"model": "jev-1.14.0", "usage": {"input_tokens": 3, "output_tokens": 1},
                    "answers": {"decision": {"type": "choice", "choice": selected,
                    "confidence": 1.0, "probabilities": probs}}})
            transport = httpx2.MockTransport(mock)
        else:
            transport = httpx2.AsyncHTTPTransport(retries=0)
    else:
        transport = fake_transport
    from capclaims.continuation_transport_audit import HttpOverlapTransport, build_event_chain
    from capclaims.injection_study import InMemoryWireRecorder
    from capclaims.portfolio_transport import DEFAULT_URL, GuardTransport, SharedBudget
    from capclaims.backends.jev import MODEL
    from typesafe_sdk import AsyncTypeSafeClient
    from typesafe_sdk._core.retry import RetryPolicy
    observer = HttpOverlapTransport(transport)
    budget = SharedBudget(output / "attempts.jsonl", max_requests=MAX_REQUESTS,
        max_bytes=MAX_RESERVED_BYTES, rps=10_000 if fake else MAX_RPS,
        bytes_per_second=100_000_000 if fake else MAX_BYTES_PER_SECOND, max_in_flight=MAX_IN_FLIGHT)
    guard = GuardTransport(observer, DEFAULT_URL, budget, max_body_bytes=MAX_BODY)
    recorder = InMemoryWireRecorder(guard)
    api_key = "offline-dummy-key" if fake else os.environ["TYPESAFE_API_KEY"]
    http = httpx2.AsyncClient(transport=recorder, follow_redirects=False)
    client = AsyncTypeSafeClient(api_key=api_key, model=MODEL, retry=RetryPolicy(max_retries=0),
        timeout=30, http_client=http, base_url="https://api.typesafe.ai", compression=False)
    backends = {arm: ExplanationBackend(api_key=api_key, rps=1000 if fake else MAX_RPS,
        tps=10_000_000 if fake else MAX_BYTES_PER_SECOND, max_requests=MAX_REQUESTS,
        max_estimated_input_tokens=MAX_RESERVED_BYTES, instruction_profile="portfolio_third_v1",
        objective="portfolio_prediction_v1", client=client, request_timeout=30,
        max_in_flight=MAX_IN_FLIGHT, load_environment=False, arm=arm) for arm in ARMS}
    engine = PairedPortfolioEngine()
    from capclaims.provider_choice_trace import (advance_provider_trace, build_provider_order_request,
        next_provider_trace_action, replay_provider_trace)
    from capclaims.portfolio_study import _replay_backend

    async def one(session: Mapping[str, Any]) -> dict[str, Any]:
        base0 = inputs["bases"][session["person_id"]]
        base = {**base0, "agent_id": f"{base0['agent_id']}__{session['arm']}_{session['repeat']}"}
        result = await engine.run_session(base, session["arm"], backends[session["arm"]],
            next_provider_trace_action, build_provider_order_request, advance_provider_trace,
            replay_provider_trace, guard.register, recorder)
        return {**session, "status": "complete", **result}

    rows: list[dict[str, Any]] = []
    try:
        for start in range(0, SESSION_N, MAX_IN_FLIGHT):
            remaining = MAX_WALL_SECONDS - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError("study exceeded wall-clock cap before next wave")
            result = await asyncio.wait_for(asyncio.gather(*(one(s) for s in inputs["sessions"][start:start + MAX_IN_FLIGHT]),
                return_exceptions=True), timeout=remaining)
            errors = [value for value in result if isinstance(value, BaseException)]
            if errors:
                raise RuntimeError(f"study wave invalidated ({len(errors)} failures; {type(errors[0]).__name__})")
            rows.extend(result)
    except BaseException:
        _write(output / "invalid.json", {"status": "INVALID", "completed_before_failure": len(rows)})
        raise
    finally:
        for backend in backends.values():
            backend._closed = True
        await client.aclose()
    _write(output / "events.json", rows)
    _write(output / "events-chain.json", build_event_chain(rows))
    _write(output / "transport.json", list(guard.http_timings))
    _write(output / "transport-precise.json", observer.rows)
    for filename in ("attempts.jsonl", "transport-requests.jsonl", "transport-bodies.jsonl"):
        path = output / filename
        if not path.exists():
            path.touch(exist_ok=False)
    if observer.peak > MAX_IN_FLIGHT:
        _write(output / "invalid.json", {"status": "INVALID", "reason": "concurrency_cap_exceeded"})
        raise RuntimeError("global HTTP concurrency limit was exceeded")
    terminal = {"status": "COMPLETE", "execution_mode": "mock" if fake else "provider",
        "preview_sha256": manifest["preview_sha256"], "receipt_sha256": _sha_bytes((output / "receipt.json").read_bytes()),
        "events_chain_sha256": _sha_bytes((output / "events-chain.json").read_bytes()),
        "event_sha256": _sha_bytes((output / "events.json").read_bytes()),
        "attempts_sha256": _sha_bytes((output / "attempts.jsonl").read_bytes()),
        "requests_sha256": _sha_bytes((output / "transport-requests.jsonl").read_bytes()),
        "bodies_sha256": _sha_bytes((output / "transport-bodies.jsonl").read_bytes()),
        "timings_sha256": _sha_bytes((output / "transport.json").read_bytes()),
        "precise_timings_sha256": _sha_bytes((output / "transport-precise.json").read_bytes()), "budget": budget.usage}
    _write(output / "terminal.json", terminal)
    return {"status": "COMPLETE", "planned_sessions": SESSION_N, "attempted_sessions": len(rows),
        "unattempted_sessions": SESSION_N - len(rows), "requests_charged": budget.usage["requests_charged"],
        "reserved_bytes": budget.usage["reserved_bytes"], "max_overlap": observer.peak}


def replay_run(preview_dir: str | Path, run_dir: str | Path) -> dict[str, Any]:
    """Strictly bind each trace choice to its rendered SDK request and raw response."""
    preview = Path(preview_dir)
    root = Path(run_dir)
    pmanifest = json.loads((preview / "manifest.json").read_text(encoding="utf-8"))
    verify_preview(preview, pmanifest.get("preview_sha256", ""))
    if (root / "invalid.json").exists():
        raise ValueError("study is INVALID and cannot be replayed or scored")
    terminal = json.loads((root / "terminal.json").read_text(encoding="utf-8"))
    if terminal.get("status") != "COMPLETE" or terminal.get("preview_sha256") != pmanifest["preview_sha256"]:
        raise ValueError("run terminal is not complete or belongs to another preview")
    artifacts = (("events.json", "event_sha256"), ("events-chain.json", "events_chain_sha256"),
        ("attempts.jsonl", "attempts_sha256"), ("transport-requests.jsonl", "requests_sha256"),
        ("transport-bodies.jsonl", "bodies_sha256"), ("transport.json", "timings_sha256"),
        ("transport-precise.json", "precise_timings_sha256"))
    for filename, key in artifacts:
        if _sha_bytes((root / filename).read_bytes()) != terminal.get(key):
            raise ValueError("run evidence digest mismatch")
    inputs = json.loads((preview / "inputs.json").read_text(encoding="utf-8"))
    rows = json.loads((root / "events.json").read_text(encoding="utf-8"))
    sent_rows = [json.loads(line) for line in (root / "transport-requests.jsonl").read_text().splitlines() if line]
    body_rows = [json.loads(line) for line in (root / "transport-bodies.jsonl").read_text().splitlines() if line]
    sent = {row["request_id"]: row for row in sent_rows}
    bodies = {row["request_id"]: row for row in body_rows}
    timings = {row["request_id"]: row for row in json.loads((root / "transport.json").read_text())}
    if len(rows) != SESSION_N or [{k: row.get(k) for k in ("session_id", "person_id", "arm", "repeat")} for row in rows] != pmanifest["sessions"]:
        raise ValueError("run rows do not match the frozen sessions in order")
    from capclaims.provider_choice import validate_provider_choice
    from capclaims.provider_choice_trace import build_provider_order_request, replay_provider_trace
    from capclaims.portfolio_transport import DEFAULT_URL, MAX_BODY_BYTES, _url_tuple
    global_ids: set[str] = set()
    portfolios: dict[str, Any] = {}
    for row in rows:
        base0 = inputs["bases"][row["person_id"]]
        base = {**base0, "agent_id": f"{base0['agent_id']}__{row['arm']}_{row['repeat']}"}
        trace = tuple(row["trace"])
        result = replay_provider_trace(base, trace)
        if row.get("status") != "complete" or result["status"] != "complete":
            raise ValueError("successful session has an incomplete provider trace")
        reconstructed_orders = [order.__dict__ for order in result["orders"]]
        if row.get("orders") != reconstructed_orders:
            raise ValueError("saved portfolio differs from the reconstructed completed trace")
        portfolios[row["session_id"]] = result["orders"]
        prefix: tuple[Any, ...] = ()
        used: set[str] = set()
        for event in trace:
            if event.get("origin") == "constraint":
                prefix = tuple([*prefix, event])
                continue
            if event.get("origin") != "provider":
                raise ValueError("successful trace has a non-provider event")
            rid = event.get("request_id")
            request_row, body_row, timing = sent.get(rid), bodies.get(rid), timings.get(rid)
            if not isinstance(rid, str) or rid in used or not request_row or not body_row or not timing or timing.get("status") != 200:
                raise ValueError("provider choice lacks one unique successful HTTP evidence join")
            used.add(rid)
            if rid in global_ids:
                raise ValueError("request id was reused across sessions")
            global_ids.add(rid)
            if timing.get("request_id") != rid:
                raise ValueError("HTTP timing identity differs from request id")
            raw_request = request_row.get("body_utf8", "").encode("utf-8")
            if _sha_bytes(raw_request) != request_row.get("body_sha256") or body_row.get("request_body_sha256") != request_row.get("body_sha256"):
                raise ValueError("sent request and response evidence hashes differ")
            if body_row.get("status") != 200:
                raise ValueError("saved provider response was not successful")
            request = build_provider_order_request(base, prefix)
            if request is None or request.request_id != rid:
                raise ValueError("request id differs from the reconstructed provider-choice state")
            expected = asyncio.run(serialize_explanation_wire(request, row["arm"]))
            if expected.body != raw_request or expected.body_sha256 != request_row.get("body_sha256"):
                raise ValueError("saved SDK body differs from reconstructed state and prompt")
            raw = json.loads(body_row["response_body_utf8"])
            answer = raw.get("answers", {}).get("decision")
            if raw.get("model") != "jev-1.14.0" or not isinstance(answer, Mapping) or answer.get("type") != "choice":
                raise ValueError("saved provider response has invalid model or decision shape")
            if event.get("raw_answer") != answer or answer.get("choice") != event.get("choice"):
                raise ValueError("trace selection differs from request-bound provider response")
            validate_provider_choice(answer, [choice.candidate_id for choice in request.candidates])
            prefix = tuple([*prefix, event])
    if global_ids != set(sent) or global_ids != set(bodies) or len(timings) != len(global_ids):
        raise ValueError("unmatched transport request, response, or timing evidence")
    attempts = [json.loads(line) for line in (root / "attempts.jsonl").read_text().splitlines() if line]
    if len(attempts) != len(global_ids) or {row.get("request_id") for row in attempts} != global_ids or len(attempts) > MAX_REQUESTS:
        raise ValueError("attempt ledger does not bind each provider call exactly once")
    for seq, attempt in enumerate(attempts, 1):
        request = sent[attempt["request_id"]]
        body = request["body_utf8"].encode()
        if (attempt.get("seq") != seq or attempt.get("method") != "POST" or _url_tuple(attempt.get("url", "")) != _url_tuple(DEFAULT_URL)
                or attempt.get("expected_sha256") != request.get("body_sha256") or attempt.get("actual_sha256") != request.get("body_sha256")
                or attempt.get("body_bytes") != len(body) or attempt.get("reserved_bytes") != MAX_BODY_BYTES or attempt.get("state") != "attempted"):
            raise ValueError("attempt ledger row differs from sent SDK request")
    if len(global_ids) * MAX_BODY > MAX_RESERVED_BYTES:
        raise ValueError("reserved bytes exceed the frozen cap")
    from capclaims.continuation_transport_audit import audit_transport_evidence
    audit = audit_transport_evidence(root, terminal, global_ids, pmanifest["preview_sha256"])
    return {"status": "REPLAY_VALID", "session_n": SESSION_N, "portfolios": portfolios,
        "statuses": [{"session_id": row["session_id"], "status": row["status"]} for row in rows],
        "transport_audit": audit}


def score_run(preview_dir: str | Path, run_dir: str | Path) -> dict[str, Any]:
    """Score per case first, save private rows, and expose only paired aggregates."""
    replayed = replay_run(preview_dir, run_dir)
    preview, run = Path(preview_dir), Path(run_dir)
    target_path = preview / "scoring-targets.json"
    manifest = json.loads((preview / "manifest.json").read_text(encoding="utf-8"))
    target_bytes = (preview / "scoring-targets.json").read_bytes()
    if _sha_bytes(target_bytes) != manifest.get("scoring_targets_sha256"):
        raise ValueError("scoring labels changed after preview")
    targets = json.loads(target_bytes.decode("utf-8"))
    rows = json.loads((run / "events.json").read_text(encoding="utf-8"))
    from capclaims.paper.orders import PaperOrder
    from capclaims.portfolio_metrics import portfolio_comparison
    private_rows = []
    for row in rows:
        target = targets[row["person_id"]]
        human = []
        for index, order in enumerate(target["orders"]):
            side = order.get("side", order.get("type"))
            quantity = order.get("qty", order.get("amount"))
            price = order.get("limit_dollars", order.get("price"))
            human.append(PaperOrder(f"{row['person_id']}-truth-{index}", row["person_id"], str(side).lower(), quantity, price))
        predicted = [PaperOrder(f"{row['person_id']}-pred-{index}", row["person_id"], order.side, order.qty, order.limit_dollars)
                     for index, order in enumerate(replayed["portfolios"][row["session_id"]])]
        comp = portfolio_comparison(predicted, human)
        predicted_buy, human_buy = comp["predicted_gross_buy"], comp["human_gross_buy"]
        signatures = sorted((o.side, o.qty, o.limit_dollars) for o in predicted)
        private_rows.append({"person_id": row["person_id"], "arm": row["arm"], "repeat": row["repeat"],
            "primary_curve_error": comp["primary_curve_error"],
            "retail_window_1_100_curve_error": comp["retail_window_1_100_curve_error"],
            "first_order_only_curve_error": portfolio_comparison(predicted[:1], human)["primary_curve_error"],
            "duplicate_order_n": len(predicted) - len(set((o.side, o.qty, o.limit_dollars) for o in predicted)),
            "duplicate_order_rate": ((len(predicted) - len(set((o.side, o.qty, o.limit_dollars) for o in predicted)))/len(predicted) if predicted else 0.0),
            "predicted_order_count": len(predicted), "human_order_count": len(human),
            "predicted_gross_buy": predicted_buy, "human_gross_buy": human_buy,
            "signed_buy_bias": predicted_buy - human_buy, "absolute_buy_error": abs(predicted_buy - human_buy),
            "positive_excess_buy": max(predicted_buy - human_buy, 0),
            "predicted_gross_sell": comp["predicted_gross_sell"], "human_gross_sell": comp["human_gross_sell"],
            "empty_agreement": comp["empty_agreement"], "exact_order_multiset_match": comp["exact_order_multiset_match"],
            "merged_same_side_price_match": comp["merged_same_side_price_match"],
            "side_class_agreement": comp["predicted_side_class"] == comp["human_side_class"],
            "five_order_rate": float(len(predicted) == 5), "order_multiset_signature": signatures,
            "provider_choice_n": sum(event.get("origin") == "provider" for event in row["trace"]),
            "provider_resolved_tie_n": sum(event.get("origin") == "provider" and len(event.get("near_top_candidate_ids", [])) > 1 for event in row["trace"]),
            "provider_choice_differs_unique_argmax_n": sum(event.get("origin") == "provider" and event.get("legacy_argmax_status") == "unique" and event.get("legacy_argmax_agrees") is False for event in row["trace"])})
    from capclaims.field_alignment_analysis import summarize_repeated_pairs
    adapted = [{**row, "arm": "C_joint_control" if row["arm"] == CONTROL else "C_joint_aligned"} for row in private_rows]
    analysis = summarize_repeated_pairs(adapted)
    per_arm = {CONTROL: analysis["per_arm"]["C_joint_control"],
               EXPLICIT: analysis["per_arm"]["C_joint_aligned"]}
    private_people = [{**row, "arm": CONTROL if row["arm"] == "C_joint_control" else EXPLICIT}
                      for row in analysis["private_person_scores"]]
    private_repeats = [{**row, "arm": CONTROL if row["arm"] == "C_joint_control" else EXPLICIT}
                       for row in analysis["private_repeat_rows"]]
    private_path = _private(run / "private-scores.json")
    _write_idempotent(private_path, {"preview_sha256": manifest["preview_sha256"], "rows": private_rows,
        "private_person_scores": private_people, "private_repeat_rows": private_repeats})
    paired = {}
    for metric, result in analysis["paired_aligned_minus_control"].items():
        paired[metric] = {key.replace("aligned", "treatment").replace("control", "control"): value for key, value in result.items()}
    return {"status": "DESCRIPTIVE_DEVELOPMENT_ONLY", "n_people": analysis["n_people"],
        "repeat_n_per_person_arm": analysis["repeat_n_per_person_arm"], "per_arm": per_arm,
        "paired_treatment_minus_control": paired, "repeat_consistency": analysis["repeat_consistency"],
        "interpretation": analysis["interpretation"], "transport_audit": replayed["transport_audit"],
        "coverage": {"session_n": SESSION_N, "people_n": analysis["n_people"]}, "session_n": SESSION_N}


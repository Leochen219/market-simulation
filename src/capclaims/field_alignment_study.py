"""Frozen, paired current-field-name study on top of the verified continuation preview."""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Mapping

from capclaims.backends.jev import MODEL, JevBackend, estimate_input_tokens, load_local_env
from capclaims.backends.jev_instructions import build_instruction
from capclaims.engine.protocol import DecisionReply
from capclaims.field_alignment_context import ALIGNED, CONTROL, RENAMES, render_field_state
from capclaims.injection_study import InMemoryWireRecorder
from capclaims.portfolio_prediction import _validate_base
from capclaims.portfolio_transport import (DEFAULT_URL, MAX_BODY_BYTES, ExpectedWire,
    GuardTransport, SharedBudget, _url_tuple)
from capclaims.provider_choice import validate_provider_choice
from capclaims.provider_choice_trace import (advance_provider_trace, build_provider_order_request,
    next_provider_trace_action, replay_provider_trace)
from capclaims.randomness import canonical_json

ROOT = Path(__file__).resolve().parents[2]
PARENT_SHA256 = "1bf9825ce1b9e2f331fb5d7568359941a2b69ffc95bce2abbaa7a3c6bfe77376"
PARENT_RELATIVE = Path("artifacts/participant-data/continuation-preview-root-25-r1")
SOURCE_POLICY_RELATIVE = Path("artifacts/participant-data/quote-perspective-24/source-policy-authorized.json")
SEED = 20261025
ARMS = (CONTROL, ALIGNED)
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
    return _sha_bytes(canonical_json(value).encode("utf-8"))


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write((canonical_json(value) + "\n").encode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())



def _write_derived_idempotent(path: Path, value: Any) -> None:
    """Write reproducible private scores once; allow an identical repeat score pass."""
    payload = (canonical_json(value) + "\n").encode("utf-8")
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError("derived private score file already exists with different bytes")
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
        raise ValueError("field-alignment study artifacts must remain under ignored participant-data")
    return resolved


def _code_hashes() -> dict[str, str]:
    from capclaims.continuation_study import _paths as frozen_paths
    paths = frozen_paths()
    own = (
        "src/capclaims/field_alignment_context.py",
        "src/capclaims/field_alignment_study.py",
        "src/capclaims/field_alignment_analysis.py",
        "docs/superpowers/specs/2026-10-25-field-alignment-design.md",
        "tools/probes/jev_field_alignment.py",
    )
    missing = [name for name in own if not (ROOT / name).is_file()]
    if missing:
        raise ValueError("field-alignment freeze files are missing: " + ", ".join(missing))
    paths.update({name: _sha_bytes((ROOT / name).read_bytes()) for name in own})
    return paths


def _check_source_policy(expected_sha: str, source: Mapping[str, Any]) -> dict[str, Any]:
    path = ROOT / SOURCE_POLICY_RELATIVE
    raw = path.read_bytes()
    if _sha_bytes(raw) != expected_sha:
        raise ValueError("source-policy hash differs from frozen preview")
    policy = json.loads(raw.decode("utf-8"))
    if not all(policy.get(k) is True for k in (
        "codebook_verified", "intended_use_authorized", "provider_transmission_permitted")):
        raise ValueError("source policy does not authorize this use")
    if policy.get("source_kind") != "human" or policy.get("source_sha256") != source.get("source_sha256"):
        raise ValueError("source-policy identity differs from source preview")
    return policy


def _session_schedule(people: list[str]) -> list[dict[str, Any]]:
    rng = random.Random(SEED)
    person_order = sorted(people)
    rng.shuffle(person_order)
    repeat_order = list(REPEATS)
    rng.shuffle(repeat_order)
    sessions = []
    for repeat in repeat_order:
        for person in person_order:
            arms = list(ARMS)
            rng.shuffle(arms)
            for arm in arms:
                sessions.append({"session_id": f"{person}:{arm}:r{repeat}", "person_id": person,
                    "arm": arm, "repeat": repeat})
    return sessions


def _same_schedule(rows: Any, people: list[str]) -> bool:
    return rows == _session_schedule(people)


def _prompt(request: Any) -> str:
    # Exact frozen C_joint prompt. Treatment changes only effective state keys.
    from capclaims.continuation_study import JOINT_APPEND
    return build_instruction(request, "portfolio_third_v1") + "\n\n" + JOINT_APPEND


async def serialize_alignment_wire(request: Any, arm: str) -> ExpectedWire:
    """Serialize the exact pinned SDK request after one explicit field renderer."""
    if arm not in ARMS:
        raise ValueError("unknown field-alignment arm")
    import httpx2
    from typesafe_sdk import AsyncTypeSafeClient, Choice
    from typesafe_sdk._core.retry import RetryPolicy

    captured: dict[str, Any] = {}
    ids = [choice.candidate_id for choice in request.candidates]

    async def capture(outgoing: Any) -> Any:
        captured.update(method=outgoing.method, url=str(outgoing.url), body=await outgoing.aread())
        probabilities = {key: float(key == ids[0]) for key in ids}
        return httpx2.Response(200, json={"model": MODEL, "usage": {"input_tokens": 1, "output_tokens": 1},
            "answers": {"decision": {"type": "choice", "choice": ids[0], "confidence": 1,
                "probabilities": probabilities}}})

    http = httpx2.AsyncClient(transport=httpx2.MockTransport(capture), follow_redirects=False)
    client = AsyncTypeSafeClient(api_key="dry-run-dummy", model=MODEL, retry=RetryPolicy(max_retries=0),
        http_client=http, base_url="https://api.typesafe.ai", compression=False)
    try:
        await client.system_one(state=render_field_state(request, arm),
            questions={"decision": Choice(instructions=_prompt(request),
                criteria={choice.candidate_id: choice.label for choice in request.candidates})}, model=MODEL)
    finally:
        await client.aclose()
    body = captured["body"]
    return ExpectedWire(request.request_id, captured["method"], _url_tuple(captured["url"]), body,
        _sha_bytes(body))


class _AlignmentBackend(JevBackend):
    def __init__(self, *args: Any, arm: str, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if arm not in ARMS:
            raise ValueError("unknown field-alignment arm")
        self.arm = arm

    async def decide_many(self, requests: Any) -> list[DecisionReply]:
        requests = tuple(requests)
        rendered = [(req, _prompt(req), render_field_state(req, self.arm)) for req in requests]
        estimates = [estimate_input_tokens(req, prompt, state) for req, prompt, state in rendered]
        await self._reserve(requests, estimates)
        return list(await asyncio.gather(*(self._one(req, estimate, prompt, state)
            for (req, prompt, state), estimate in zip(rendered, estimates))))


def build_preview(parent_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Freeze 6×2×2 cases; scoring labels are copied only at this boundary."""
    from capclaims.continuation_study import verify_preview as verify_parent
    parent = _private(parent_dir)
    output = _private(output_dir)
    pm = verify_parent(parent, PARENT_SHA256)
    if output.exists():
        raise FileExistsError("preview is immutable")
    parent_inputs_bytes = (parent / "inputs.json").read_bytes()
    parent_labels_bytes = (parent / "scoring-targets.json").read_bytes()
    parent_manifest_bytes = (parent / "manifest.json").read_bytes()
    inputs = json.loads(parent_inputs_bytes.decode("utf-8"))
    parent_manifest = json.loads(parent_manifest_bytes.decode("utf-8"))
    bases = {person: _validate_base(base) for person, base in inputs.get("bases", {}).items()}
    if len(bases) != 6:
        raise ValueError("parent preview must contain six verified development bases")
    if any([entry["round"] for entry in base["history"]] != [6, 7, 8] for base in bases.values()):
        raise ValueError("all cases must retain the same named three-round history (rounds 6 through 8)")
    labels = json.loads(parent_labels_bytes.decode("utf-8"))
    if _sha_bytes(parent_labels_bytes) != pm.get("scoring_targets_sha256"):
        raise ValueError("parent scoring target digest does not match")
    policy_path = ROOT / SOURCE_POLICY_RELATIVE
    policy_bytes = policy_path.read_bytes()
    source = pm.get("parent_source", pm.get("source", {}))
    policy_sha = pm.get("source_policy_sha256")
    _check_source_policy(policy_sha, source)
    sessions = _session_schedule(list(bases))
    # Build a maximum-width check on every source base and both renderer arms.
    from capclaims.injection_study import _maximal_legal_requests
    maximum = 0
    for person, base in bases.items():
        for request in _maximal_legal_requests(base["history"]):
            # Request IDs are synthetic preflight IDs; prompt contents/state exercise exact renderer.
            maximum = max(maximum, *(len(asyncio.run(serialize_alignment_wire(request, arm)).body) for arm in ARMS))
    if maximum > MAX_BODY:
        raise ValueError("preflight request exceeds hard body cap")
    output.mkdir(parents=True, exist_ok=False)
    frozen_inputs = {"bases": bases, "sessions": sessions}
    _write(output / "inputs.json", frozen_inputs)
    _write(output / "scoring-targets.json", labels)
    prompts = {}
    first_base = bases[sessions[0]["person_id"]]
    first_req = build_provider_order_request({**first_base, "agent_id": first_base["agent_id"] + "__freeze"}, ())
    if first_req is None:
        raise ValueError("parent base has no first provider action")
    prompts["C_joint"] = _prompt(first_req)
    manifest = {
        "schema": "jev_field_alignment_preview_v1", "parent_preview_sha256": PARENT_SHA256,
        "parent_manifest_sha256": _sha_bytes(parent_manifest_bytes), "parent_inputs_sha256": _sha_bytes(parent_inputs_bytes),
        "parent_labels_sha256": _sha_bytes(parent_labels_bytes),
        "source_policy_path": SOURCE_POLICY_RELATIVE.as_posix(), "source_policy_sha256": _sha_bytes(policy_bytes),
        "source": source, "source_policy_verified": True,
        "input_sha256": _sha_bytes((output / "inputs.json").read_bytes()),
        "scoring_targets_sha256": _sha_bytes((output / "scoring-targets.json").read_bytes()),
        "sessions": sessions, "session_n": SESSION_N, "people": sorted(bases), "arms": list(ARMS),
        "repeats": list(REPEATS), "schedule_seed": SEED, "model": MODEL, "sdk_version": "0.7.5",
        "instruction_profile": "portfolio_third_v1", "joint_prompt": prompts["C_joint"],
        "joint_prompt_sha256": _sha_bytes(prompts["C_joint"].encode("utf-8")),
        "renames": list(RENAMES.items()), "preflight_max_body_bytes": maximum,
        "request_cap": MAX_REQUESTS, "reserved_bytes_cap": MAX_RESERVED_BYTES, "body_bytes_cap": MAX_BODY,
        "max_in_flight": MAX_IN_FLIGHT, "rps_cap": MAX_RPS, "bytes_per_second_cap": MAX_BYTES_PER_SECOND,
        "wall_seconds_cap": MAX_WALL_SECONDS, "timeout_seconds": 30, "retry_cap": 0,
        "code_sha256": _code_hashes(),
    }
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
    if (manifest.get("parent_preview_sha256") != PARENT_SHA256 or manifest.get("code_sha256") != _code_hashes()
        or manifest.get("renames") != [[old, new] for old, new in RENAMES.items()] or manifest.get("source_policy_verified") is not True
        or manifest.get("session_n") != SESSION_N):
        raise ValueError("frozen parent, code, mapping or schedule contract changed")
    raw = (root / "inputs.json").read_bytes()
    if _sha_bytes(raw) != manifest.get("input_sha256"):
        raise ValueError("frozen inference inputs changed")
    inputs = json.loads(raw.decode("utf-8"))
    if inputs.get("sessions") != manifest.get("sessions") or not _same_schedule(inputs.get("sessions"), manifest["people"]):
        raise ValueError("schedule is not the frozen balanced randomized two-repeat schedule")
    if len(inputs.get("bases", {})) != 6 or set(inputs["bases"]) != set(manifest["people"]):
        raise ValueError("preview base roster is invalid")
    for base in inputs["bases"].values():
        checked = _validate_base(base)
        if [entry["round"] for entry in checked["history"]] != [6, 7, 8]:
            raise ValueError("frozen current/history input does not contain exactly rounds 6 through 8")
    counts = {(p, a, r): 0 for p in manifest["people"] for a in ARMS for r in REPEATS}
    for row in manifest["sessions"]:
        key = (row.get("person_id"), row.get("arm"), row.get("repeat"))
        if key not in counts:
            raise ValueError("unexpected session factor")
        counts[key] += 1
    if any(count != 1 for count in counts.values()):
        raise ValueError("incomplete or duplicated person/arm/repeat design")
    prompt = manifest.get("joint_prompt", "")
    if not prompt or _sha_bytes(prompt.encode("utf-8")) != manifest.get("joint_prompt_sha256"):
        raise ValueError("frozen full C_joint prompt digest is invalid")
    first_session = manifest["sessions"][0]
    first_base = inputs["bases"][first_session["person_id"]]
    first_request = build_provider_order_request({**first_base, "agent_id": first_base["agent_id"] + "__freeze"}, ())
    if first_request is None or prompt != _prompt(first_request):
        raise ValueError("frozen full C_joint prompt differs from the unchanged prompt builder")
    from importlib.metadata import version
    if version("typesafe-sdk") != "0.7.5" or manifest.get("sdk_version") != "0.7.5":
        raise ValueError("pinned SDK version changed")
    if type(manifest.get("preflight_max_body_bytes")) is not int or manifest["preflight_max_body_bytes"] > MAX_BODY:
        raise ValueError("closed worst-case request preflight failed")
    return manifest


def _session_base(inputs: Mapping[str, Any], session: Mapping[str, Any]) -> dict[str, Any]:
    base = inputs["bases"][session["person_id"]]
    return {**base, "agent_id": f"{base['agent_id']}__fa_{session['arm']}_{session['repeat']}"}


async def execute_preview(preview_dir: str | Path, run_dir: str | Path, reviewed_preview_sha256: str,
                          fake: bool = False, fake_transport: Any = None) -> dict[str, Any]:
    manifest = verify_preview(preview_dir, reviewed_preview_sha256)
    output = _private(run_dir)
    if output.exists():
        raise FileExistsError("run cannot be overwritten or resumed")
    output.mkdir(parents=True, exist_ok=False)
    start_mono, start_perf = time.monotonic(), time.perf_counter_ns()
    _write(output / "receipt.json", {"preview_sha256": manifest["preview_sha256"],
        "execution_mode": "mock" if fake else "provider", "status": "RUNNING",
        "started_at_utc": datetime.now(timezone.utc).isoformat(), "started_monotonic": start_mono,
        "started_perf_ns": start_perf, "source_policy_sha256": manifest["source_policy_sha256"]})
    inputs = json.loads((Path(preview_dir) / "inputs.json").read_text(encoding="utf-8"))
    # Permission artifact and source identity are checked before env/key/client access.
    if not fake:
        _check_source_policy(manifest["source_policy_sha256"], manifest["source"])
        load_local_env()
        if not os.getenv("TYPESAFE_API_KEY"):
            raise ValueError("TYPESAFE_API_KEY is unavailable")
    import httpx2
    fake_probe: dict[str, Any] | None = None
    if fake_transport is None:
        if fake:
            fake_probe = {"initial_request_bodies": []}
            first_wave_gate = asyncio.Event()
            first_wave_lock = asyncio.Lock()
            initial_count = 0
            async def fake_handler(req: Any) -> Any:
                nonlocal initial_count
                request_body = await req.aread()
                async with first_wave_lock:
                    if initial_count < MAX_IN_FLIGHT:
                        initial_count += 1
                        fake_probe["initial_request_bodies"].append(_sha_bytes(request_body))
                        if initial_count == MAX_IN_FLIGHT:
                            first_wave_gate.set()
                        gate_this = True
                    else:
                        gate_this = False
                if gate_this:
                    await first_wave_gate.wait()
                body = json.loads(request_body.decode("utf-8"))
                criteria = body["questions"]["decision"]["criteria"]
                ids = list(criteria)
                selected = next((key for key, label in criteria.items() if label.startswith("STOP:")), ids[-1])
                probs = {key: float(key == selected) for key in ids}
                return httpx2.Response(200, json={"model": MODEL, "usage": {"input_tokens": 3, "output_tokens": 1},
                    "answers": {"decision": {"type": "choice", "choice": selected, "confidence": 1.0,
                        "probabilities": probs}}})
            inner_transport = httpx2.MockTransport(fake_handler)
        else:
            inner_transport = httpx2.AsyncHTTPTransport(retries=0)
    else:
        inner_transport = fake_transport
    from capclaims.continuation_transport_audit import HttpOverlapTransport, build_event_chain
    observer = HttpOverlapTransport(inner_transport)
    budget = SharedBudget(output / "attempts.jsonl", max_requests=MAX_REQUESTS, max_bytes=MAX_RESERVED_BYTES,
        rps=10_000 if fake else MAX_RPS, bytes_per_second=100_000_000 if fake else MAX_BYTES_PER_SECOND,
        max_in_flight=MAX_IN_FLIGHT)
    guard = GuardTransport(observer, DEFAULT_URL, budget, max_body_bytes=MAX_BODY)
    recorder = InMemoryWireRecorder(guard)
    from typesafe_sdk import AsyncTypeSafeClient
    from typesafe_sdk._core.retry import RetryPolicy
    key = "offline-dummy-key" if fake else os.environ["TYPESAFE_API_KEY"]
    http = httpx2.AsyncClient(transport=recorder, follow_redirects=False)
    client = AsyncTypeSafeClient(api_key=key, model=MODEL, retry=RetryPolicy(max_retries=0), timeout=30,
        http_client=http, base_url="https://api.typesafe.ai", compression=False)
    backends = {arm: _AlignmentBackend(api_key=key, rps=10_000 if fake else MAX_RPS,
        tps=10_000_000 if fake else MAX_BYTES_PER_SECOND, max_requests=MAX_REQUESTS,
        max_estimated_input_tokens=MAX_RESERVED_BYTES, instruction_profile="portfolio_third_v1",
        objective="portfolio_prediction_v1", client=client, request_timeout=30,
        max_in_flight=MAX_IN_FLIGHT, load_environment=False, arm=arm) for arm in ARMS}
    rows: list[dict[str, Any]] = []

    async def run_session(session: Mapping[str, Any]) -> dict[str, Any]:
        base = _session_base(inputs, session)
        trace: tuple[dict[str, Any], ...] = ()
        events = []
        while True:
            action = next_provider_trace_action(base, trace)
            if action["kind"] == "complete":
                orders = replay_provider_trace(base, trace)["orders"]
                return {**session, "status": "complete", "trace": events,
                    "orders": [order.__dict__ for order in orders]}
            if action["kind"] == "forced":
                trace = advance_provider_trace(base, trace, None)
                events.append(trace[-1])
                continue
            if action["kind"] != "request":
                raise RuntimeError("provider trace became incomplete")
            request = build_provider_order_request(base, trace)
            wire = await serialize_alignment_wire(request, session["arm"])
            guard.register(wire)
            (reply,) = await backends[session["arm"]].decide_many([request])
            record = recorder.rows.get(request.request_id)
            if not record or record["status"] != 200:
                raise RuntimeError("request-bound raw HTTP response missing or unsuccessful")
            answer = json.loads(record["response_body_utf8"])["answers"]["decision"]
            trace = advance_provider_trace(base, trace, answer, reply)
            events.append(trace[-1])

    try:
        sessions = manifest["sessions"]
        for offset in range(0, len(sessions), MAX_IN_FLIGHT):
            remain = MAX_WALL_SECONDS - (time.monotonic() - start_mono)
            if remain <= 0:
                raise TimeoutError("study exceeded 600 second wall cap")
            wave = await asyncio.wait_for(asyncio.gather(*(
                run_session(session) for session in sessions[offset:offset + MAX_IN_FLIGHT]),
                return_exceptions=True), timeout=remain)
            failures = [result for result in wave if isinstance(result, BaseException)]
            if failures:
                detail = f"; {str(failures[0])[:160]}" if fake else ""
                raise RuntimeError(f"experiment invalidated by {len(failures)} session failure(s): {type(failures[0]).__name__}{detail}")
            rows.extend(wave)
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
    for file in ("attempts.jsonl", "transport-requests.jsonl", "transport-bodies.jsonl"):
        path = output / file
        if not path.exists():
            path.touch(exist_ok=False)
    receipt_bytes = (output / "receipt.json").read_bytes()
    event_bytes = (output / "events.json").read_bytes()
    terminal = {"status": "COMPLETE", "execution_mode": "mock" if fake else "provider",
        "preview_sha256": manifest["preview_sha256"], "receipt_sha256": _sha_bytes(receipt_bytes),
        "event_sha256": _sha_bytes(event_bytes),
        "events_chain_sha256": _sha_bytes((output / "events-chain.json").read_bytes()),
        "attempts_sha256": _sha_bytes((output / "attempts.jsonl").read_bytes()),
        "requests_sha256": _sha_bytes((output / "transport-requests.jsonl").read_bytes()),
        "bodies_sha256": _sha_bytes((output / "transport-bodies.jsonl").read_bytes()),
        "timings_sha256": _sha_bytes((output / "transport.json").read_bytes()),
        "precise_timings_sha256": _sha_bytes((output / "transport-precise.json").read_bytes()),
        "budget": budget.usage}
    if observer.peak > MAX_IN_FLIGHT:
        _write(output / "invalid.json", {"status": "INVALID", "reason": "concurrency_cap_exceeded"})
        raise RuntimeError("HTTP concurrency exceeded ten")
    if fake_probe is not None and (observer.peak != MAX_IN_FLIGHT
            or len(fake_probe["initial_request_bodies"]) != MAX_IN_FLIGHT
            or len(set(fake_probe["initial_request_bodies"])) < MAX_IN_FLIGHT):
        _write(output / "invalid.json", {"status": "INVALID", "reason": "mock_parallelism_or_input_diversity_not_demonstrated"})
        raise RuntimeError("offline barrier did not demonstrate ten distinct simultaneous first requests")
    _write(output / "terminal.json", terminal)
    return {"status": "COMPLETE", "planned_sessions": SESSION_N, "attempted_sessions": len(rows),
        "unattempted_sessions": SESSION_N-len(rows), "session_n": len(rows), "requests_charged": budget.usage["requests_charged"],
        "reserved_bytes": budget.usage["reserved_bytes"], "max_overlap": observer.peak}


def replay_run(preview_dir: str | Path, run_dir: str | Path) -> dict[str, Any]:
    pm = json.loads((Path(preview_dir) / "manifest.json").read_text(encoding="utf-8"))
    manifest = verify_preview(preview_dir, pm.get("preview_sha256", ""))
    root = Path(run_dir)
    if (root / "invalid.json").exists():
        raise ValueError("invalid study cannot be replayed or scored")
    terminal = json.loads((root / "terminal.json").read_text(encoding="utf-8"))
    for file, digest_key in (("events.json", "event_sha256"), ("events-chain.json", "events_chain_sha256"),
        ("attempts.jsonl", "attempts_sha256"), ("transport-requests.jsonl", "requests_sha256"),
        ("transport-bodies.jsonl", "bodies_sha256"), ("transport.json", "timings_sha256"),
        ("transport-precise.json", "precise_timings_sha256")):
        if _sha_bytes((root / file).read_bytes()) != terminal.get(digest_key):
            raise ValueError("run evidence digest mismatch: " + file)
    receipt = (root / "receipt.json").read_bytes()
    if terminal.get("receipt_sha256") != _sha_bytes(receipt) or terminal.get("preview_sha256") != manifest["preview_sha256"]:
        raise ValueError("run receipt/preview binding mismatch")
    from capclaims.continuation_transport_audit import audit_transport_evidence, build_event_chain
    rows = json.loads((root / "events.json").read_text(encoding="utf-8"))
    if len(rows) != SESSION_N or [dict((k, row.get(k)) for k in ("session_id", "person_id", "arm", "repeat")) for row in rows] != manifest["sessions"]:
        raise ValueError("run does not bind exact frozen session order and factors")
    if json.loads((root / "events-chain.json").read_text(encoding="utf-8")) != build_event_chain(rows):
        raise ValueError("event chain mismatch")
    requests = {x["request_id"]: x for x in (json.loads(line) for line in (root / "transport-requests.jsonl").read_text().splitlines() if line)}
    responses = {x["request_id"]: x for x in (json.loads(line) for line in (root / "transport-bodies.jsonl").read_text().splitlines() if line)}
    timings = {x["request_id"]: x for x in json.loads((root / "transport.json").read_text(encoding="utf-8"))}
    portfolios, used = {}, set()
    inputs = json.loads((Path(preview_dir) / "inputs.json").read_text(encoding="utf-8"))
    for row in rows:
        base = _session_base(inputs, row)
        trace = tuple(row["trace"])
        result = replay_provider_trace(base, trace)
        if result["status"] != "complete" or [o.__dict__ for o in result["orders"]] != row["orders"]:
            raise ValueError("saved trace does not reproduce complete portfolio")
        portfolios[row["session_id"]] = result["orders"]
        prefix = ()
        for event in trace:
            if event.get("origin") == "constraint":
                prefix = (*prefix, event)
                continue
            rid = event.get("request_id")
            if not rid or rid in used or rid not in requests or rid not in responses or rid not in timings:
                raise ValueError("trace lacks a unique request/response/timing join")
            used.add(rid)
            sent, response, timing = requests[rid], responses[rid], timings[rid]
            if timing.get("status") != 200 or response.get("status") != 200:
                raise ValueError("unsuccessful response exists in completed study")
            body = sent["body_utf8"].encode("utf-8")
            if _sha_bytes(body) != sent.get("body_sha256") or response.get("request_body_sha256") != sent.get("body_sha256"):
                raise ValueError("request and raw response binding invalid")
            request = build_provider_order_request(base, prefix)
            if request is None or request.request_id != rid:
                raise ValueError("reconstructed request identity differs")
            expected = asyncio.run(serialize_alignment_wire(request, row["arm"]))
            if expected.body != body:
                raise ValueError("effective SDK wire differs from reconstructed arm state")
            raw_body = response["response_body_utf8"].encode("utf-8")
            if _sha_bytes(raw_body) != response.get("response_body_sha256"):
                raise ValueError("raw response hash mismatch")
            raw = json.loads(raw_body.decode("utf-8"))
            answer = raw["answers"]["decision"]
            if raw.get("model") != MODEL or answer != event.get("raw_answer"):
                raise ValueError("literal model answer differs from trace choice")
            validate_provider_choice(answer, [candidate.candidate_id for candidate in request.candidates])
            prefix = (*prefix, event)
    if used != set(requests) or used != set(responses):
        raise ValueError("unjoined or extra transport evidence")
    # Independent raw audit checks exact caps, high-resolution causal overlap and chains.
    audit = audit_transport_evidence(root, terminal, used, manifest["preview_sha256"])
    return {"status": "REPLAY_VALID", "session_n": len(rows), "portfolios": portfolios,
        "transport_audit": audit}


def _score_components(predicted: Any, human: Any) -> dict[str, Any]:
    from capclaims.portfolio_metrics import portfolio_comparison
    comparison = portfolio_comparison(predicted, human)
    signatures = [(o.side, o.qty, o.limit_dollars) for o in predicted]
    predicted_buy = sum(o.qty for o in predicted if o.side == "buy")
    human_buy = sum(o.qty for o in human if o.side == "buy")
    return {**comparison, "predicted_order_count": len(predicted), "human_order_count": len(human),
        "duplicate_order_n": len(signatures)-len(set(signatures)),
        "duplicate_order_rate": (len(signatures)-len(set(signatures)))/len(signatures) if signatures else 0.0,
        "signed_buy_bias": predicted_buy-human_buy, "absolute_buy_error": abs(predicted_buy-human_buy),
        "positive_excess_buy": max(0, predicted_buy-human_buy),
        "first_order_only_curve_error": portfolio_comparison(predicted[:1], human)["primary_curve_error"],
        "five_order_rate": float(len(predicted) == MAX_ORDERS),
        "order_multiset_signature": sorted(signatures)}


def score_run(preview_dir: str | Path, run_dir: str | Path) -> dict[str, Any]:
    replayed = replay_run(preview_dir, run_dir)
    root = Path(preview_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    target_bytes = (root / "scoring-targets.json").read_bytes()
    if _sha_bytes(target_bytes) != manifest.get("scoring_targets_sha256"):
        raise ValueError("scoring targets changed after freeze")
    targets = json.loads(target_bytes.decode("utf-8"))
    rows = json.loads((Path(run_dir) / "events.json").read_text(encoding="utf-8"))
    from capclaims.paper.orders import PaperOrder
    score_rows = []
    for row in rows:
        person = row["person_id"]
        target = targets[person]
        truth = [PaperOrder(f"truth-{person}-{i}", person, str(o.get("side", o.get("type"))).lower(),
            o.get("qty", o.get("amount")), o.get("limit_dollars", o.get("price"))) for i, o in enumerate(target["orders"])]
        predicted = [PaperOrder(f"pred-{person}-{i}", person, o.side, o.qty, o.limit_dollars)
            for i, o in enumerate(replayed["portfolios"][row["session_id"]])]
        score = _score_components(predicted, truth)
        score_rows.append({"person_id": person, "arm": row["arm"], "repeat": row["repeat"], **score})
    from capclaims.field_alignment_analysis import summarize_repeated_pairs
    summary = summarize_repeated_pairs(score_rows)
    private_scores = {"private_case_scores": score_rows,
        "private_person_scores": summary["private_person_scores"],
        "private_repeat_rows": summary["private_repeat_rows"]}
    _write_derived_idempotent(Path(run_dir) / "private-scores.json", private_scores)
    public_summary = {key: value for key, value in summary.items()
        if key not in {"private_person_scores", "private_repeat_rows"}}
    return {**public_summary, "status": "SCORED_DESCRIPTIVE_DEVELOPMENT_ONLY",
        "transport_audit": replayed["transport_audit"], "coverage": {"planned_cases": SESSION_N,
            "completed_cases": len(score_rows), "complete": len(score_rows) == SESSION_N},
        "session_n": len(score_rows)}


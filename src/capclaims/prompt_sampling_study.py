"""Frozen three-prompt, two-decoder Jev portfolio forecasting study."""
from __future__ import annotations

import hashlib
import json
import os
import random
import asyncio
import time
import base64
import collections
import math
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from capclaims.prompt_sampling_presentation import PROMPTS, sampling_prompt, render_prompt_state
from capclaims.prompt_sampling_decoder import DECODERS

ROOT = Path(__file__).resolve().parents[2]
PARENT_SHA256 = "52f31949fbe4bda3a6c748a048252f1978b5c61c462dd1e777533fae7b4f5530"
PARENT_RELATIVE = Path("artifacts/participant-data/order-explanation-preview-root-25-r0")
SOURCE_POLICY_RELATIVE = Path("artifacts/participant-data/quote-perspective-24/source-policy-authorized.json")
SEED = 20261026
SESSION_N = 24
MAX_REQUESTS = 840
MAX_RESERVED_BYTES = 6_300_000
MAX_BODY = 7_500
MAX_IN_FLIGHT = 10
MAX_RPS = 40
MAX_BYTES_PER_SECOND = 50_000
MAX_WALL_SECONDS = 600
TIMEOUT_SECONDS = 30
RETRY_CAP = 0


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
            raise FileExistsError("immutable derived scores already exist with different content")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _append_jsonl(path: Path, value: Any) -> None:
    from capclaims.randomness import canonical_json
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (canonical_json(value) + "\n").encode("utf-8")
    fd = os.open(path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
    try:
        offset = 0
        while offset < len(data):
            offset += os.write(fd, data[offset:])
        os.fsync(fd)
    finally:
        os.close(fd)


def _strict_json(text: str) -> Any:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("JSON evidence contains a duplicate key")
            result[key] = value
        return result
    return json.loads(text, object_pairs_hook=unique, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("invalid JSON number")))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [_strict_json(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


class _DurablePhysicalTransport:
    """Rate-bound dispatch wrapper with durable raw response and clock evidence."""
    def __init__(self, inner: Any, path: Path, *, raw_path: Path, event_path: Path, mock: bool) -> None:
        self.inner, self.path = inner, path
        self.raw_path, self.event_path, self.mock = raw_path, event_path, mock
        self.rows: list[dict[str, Any]] = []
        self._starts: collections.deque[int] = collections.deque()
        self._pace_lock = asyncio.Lock()

    @property
    def peak(self) -> int:
        return int(getattr(self.inner, "peak", 0))

    async def handle_async_request(self, request: Any) -> Any:
        from capclaims.backends.jev import _CURRENT_REQUEST_ID
        rid = _CURRENT_REQUEST_ID.get()
        request_body = await request.aread()
        req_sha = _sha_bytes(request_body)
        # Make the request identity and fixed reservation durable before entering
        # the rate window. Slow storage must not postpone forwarding after a
        # rate slot has already been consumed.
        _append_jsonl(self.event_path, {"request_id": rid, "kind": "dispatch_start",
            "request_body_sha256": req_sha, "reserved_bytes": MAX_BODY,
            "rate_policy": "mock_bypassed" if self.mock else "6_reserved_bodies_per_rolling_second"})
        if not self.mock:
            while True:
                await self._pace_lock.acquire()
                now_ns = time.perf_counter_ns()
                while self._starts and now_ns - self._starts[0] > 1_005_000_000:
                    self._starts.popleft()
                if len(self._starts) < 6:
                    self._starts.append(now_ns)
                    self._pace_lock.release()
                    break
                wait = max(0.001, (1_005_000_000 - (now_ns - self._starts[0])) / 1e9)
                self._pace_lock.release()
                await asyncio.sleep(wait)
        # Do not add an awaited or durable operation between rate admission and
        # this forwarding call. HttpOverlapTransport records the actual start.
        status: Any = "transport_error"
        response_body: bytes | None = None
        try:
            response = await self.inner.handle_async_request(request)
            response_body = await response.aread()
            status = response.status_code
            return response
        except BaseException as exc:
            status = type(exc).__name__
            raise
        finally:
            physical = next((item for item in reversed(getattr(self.inner, "rows", []))
                if item.get("request_id") == rid), None)
            finish_perf_ns = physical.get("finished_perf_ns") if physical else time.perf_counter_ns()
            started_perf_ns = physical.get("started_perf_ns") if physical else finish_perf_ns
            receipt = None
            try:
                receipt = _strict_json((self.event_path.parent / "receipt.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                pass
            receipt_mono = receipt.get("started_monotonic") if isinstance(receipt, Mapping) else None
            receipt_perf = receipt.get("started_perf_ns") if isinstance(receipt, Mapping) else None
            start_mono = receipt_mono + (started_perf_ns - receipt_perf) / 1e9 \
                if isinstance(receipt_mono, (int, float)) and isinstance(receipt_perf, int) else time.monotonic()
            finish_mono = receipt_mono + (finish_perf_ns - receipt_perf) / 1e9 \
                if isinstance(receipt_mono, (int, float)) and isinstance(receipt_perf, int) else time.monotonic()
            row = {"request_id": rid, "started_perf_ns": started_perf_ns,
                "finished_perf_ns": finish_perf_ns, "dispatch_monotonic": start_mono,
                "finished_monotonic": finish_mono, "request_body_sha256": req_sha,
                "reserved_bytes": MAX_BODY, "status": status,
                "rate_policy": "mock_bypassed" if self.mock else "6_reserved_bodies_per_rolling_second",
                "start_event_seq": physical.get("start_event_seq") if physical else None,
                "end_event_seq": physical.get("end_event_seq") if physical else None,
                "response_body_bytes": len(response_body) if response_body is not None else None,
                "response_body_sha256": _sha_bytes(response_body) if response_body is not None else None,
                "response_body_base64": base64.b64encode(response_body).decode("ascii") if response_body is not None else None}
            self.rows.append(row)
            _append_jsonl(self.path, row)
            _append_jsonl(self.raw_path, {"request_id": rid, "status": status,
                "request_body_sha256": req_sha,
                "response_body_bytes": row["response_body_bytes"],
                "response_body_sha256": row["response_body_sha256"],
                "response_body_base64": row["response_body_base64"]})
            _append_jsonl(self.event_path, {"request_id": rid, "kind": "dispatch_end",
                "started_monotonic": start_mono, "dispatch_perf_ns": started_perf_ns,
                "finished_monotonic": finish_mono, "finished_perf_ns": finish_perf_ns,
                "status": status, "response_body_sha256": row["response_body_sha256"]})

    async def aclose(self) -> None:
        await self.inner.aclose()


def _private(path: str | Path) -> Path:
    resolved = Path(path).resolve()
    private = (ROOT / "artifacts" / "participant-data").resolve()
    if resolved != private and private not in resolved.parents:
        raise ValueError("prompt-sampling artifacts must remain under ignored participant-data")
    return resolved


def _schedule(people: list[str]) -> list[dict[str, Any]]:
    rng = random.Random(SEED)
    order = sorted(people)
    rng.shuffle(order)
    factor_rows = [(person, decoder, repeat) for repeat in (0, 1) for person in order
        for decoder in DECODERS]
    rng.shuffle(factor_rows)
    return [{"session_id": f"{person}:{decoder}:r{repeat}", "person_id": person,
        "decoder": decoder, "repeat": repeat} for person, decoder, repeat in factor_rows]


def _code_hashes() -> dict[str, str]:
    from capclaims.order_explanation_study import _code_hashes as frozen_hashes
    result = dict(frozen_hashes())
    files = ["src/capclaims/order_presentation.py", "src/capclaims/paired_portfolio_engine.py",
        "src/capclaims/order_explanation_study.py", "src/capclaims/portfolio_prediction.py",
        "src/capclaims/provider_choice_trace.py", "src/capclaims/prompt_sampling_presentation.py",
        "src/capclaims/prompt_sampling_decoder.py", "src/capclaims/prompt_sampling_trace.py",
        "src/capclaims/prompt_sampling_study.py", "src/capclaims/prompt_sampling_analysis.py",
        "docs/superpowers/specs/2026-10-26-prompt-sampling-design.md",
        "tools/probes/jev_prompt_sampling.py"]
    for name in files:
        path = ROOT / name
        if not path.is_file():
            raise ValueError(f"study freeze file is missing: {name}")
        result[name] = _sha_bytes(path.read_bytes())
    return result


def _preflight_max_body(bases: Mapping[str, Mapping[str, Any]]) -> int:
    from capclaims.injection_study import _maximal_legal_requests
    async def calculate() -> int:
        maximum = 0
        for base in bases.values():
            for request in _maximal_legal_requests(base["history"]):
                for name in PROMPTS:
                    wire = await _serialize_wire(request, sampling_prompt(request, name))
                    maximum = max(maximum, len(wire.body))
        return maximum
    coroutine = calculate()
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coroutine).result()


def build_preview(parent_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Freeze shared participant inputs, all three prompts, and the per-prompt schedule."""
    parent, output = _private(parent_dir), _private(output_dir)
    from capclaims.order_explanation_study import verify_preview as verify_parent
    pmanifest = verify_parent(parent, PARENT_SHA256)
    parent_labels_raw = (parent / "scoring-targets.json").read_bytes()
    if _sha_bytes(parent_labels_raw) != pmanifest.get("scoring_targets_sha256"):
        raise ValueError("parent labels fail their frozen checksum")
    parent_inputs_raw = (parent / "inputs.json").read_bytes()
    parent_inputs = _strict_json(parent_inputs_raw.decode("utf-8"))
    bases = parent_inputs.get("bases")
    from capclaims.portfolio_prediction import _validate_base
    if not isinstance(bases, dict) or len(bases) != 6:
        raise ValueError("approved parent must contain exactly six people")
    bases = {key: _validate_base(value) for key, value in bases.items()}
    if any(base["round"] != 9 or [item["round"] for item in base["history"]] != [6, 7, 8]
            for base in bases.values()):
        raise ValueError("parent must retain target formal round 9 and own formal rounds 6 through 8")
    policy_path = ROOT / SOURCE_POLICY_RELATIVE
    policy_raw = policy_path.read_bytes()
    if _sha_bytes(policy_raw) != pmanifest.get("source_policy_sha256"):
        raise ValueError("source policy differs from authorized parent")
    from capclaims.order_explanation_study import _check_source_policy
    _check_source_policy(pmanifest["source_policy_sha256"], pmanifest.get("source", {}))
    if output.exists():
        raise FileExistsError("preview is immutable")
    sessions = _schedule(sorted(bases))
    inputs = {"bases": bases, "sessions": sessions}
    from capclaims.provider_choice_trace import build_provider_order_request
    first = sessions[0]
    first_base = bases[first["person_id"]]
    first_request = build_provider_order_request({**first_base, "agent_id": first_base["agent_id"] + "__prompt_freeze"}, ())
    prompt_texts = {name: sampling_prompt(first_request, name) for name in PROMPTS}
    prompt_hashes = {name: _sha_bytes(text.encode("utf-8")) for name, text in prompt_texts.items()}
    maximum = _preflight_max_body(bases)
    if maximum > MAX_BODY:
        raise ValueError("maximum-width body across all prompts exceeds the frozen 7500-byte cap")
    output.mkdir(parents=True, exist_ok=False)
    _write(output / "inputs.json", inputs)
    _write(output / "scoring-targets.json", _strict_json(parent_labels_raw.decode("utf-8")))
    manifest = {"schema": "jev_prompt_sampling_preview_v1", "parent_preview_sha256": PARENT_SHA256,
        "parent_manifest_sha256": _sha_bytes((parent / "manifest.json").read_bytes()),
        "parent_inputs_sha256": _sha_bytes(parent_inputs_raw),
        "parent_labels_sha256": _sha_bytes(parent_labels_raw), "source_policy_path": SOURCE_POLICY_RELATIVE.as_posix(),
        "source_policy_sha256": _sha_bytes(policy_raw), "source": pmanifest.get("source", {}),
        "people": sorted(bases), "prompts": list(PROMPTS), "decoders": list(DECODERS),
        "repeats": [0, 1], "schedule_seed": SEED, "sessions": sessions, "session_n_per_prompt": SESSION_N,
        "prompts_text": prompt_texts, "prompt_sha256": prompt_hashes,
        "preflight_max_body_bytes": maximum,
        "model": "jev-1.14.0", "sdk_version": "0.7.5", "max_requests_per_prompt": MAX_REQUESTS,
        "request_cap": MAX_REQUESTS,
        "reserved_bytes_cap": MAX_RESERVED_BYTES, "body_bytes_cap": MAX_BODY,
        "max_in_flight": MAX_IN_FLIGHT, "rps_cap": MAX_RPS, "bytes_per_second_cap": MAX_BYTES_PER_SECOND,
        "wall_seconds_cap": MAX_WALL_SECONDS, "timeout_seconds": TIMEOUT_SECONDS, "retry_cap": RETRY_CAP,
        "input_sha256": _sha_bytes((output / "inputs.json").read_bytes()),
        "scoring_targets_sha256": _sha_bytes((output / "scoring-targets.json").read_bytes()),
        "code_sha256": _code_hashes()}
    manifest["preview_sha256"] = _sha(manifest)
    _write(output / "manifest.json", manifest)
    return manifest


def verify_preview(preview_dir: str | Path, reviewed_sha256: str) -> dict[str, Any]:
    """Validate frozen inference inputs without opening score labels."""
    root = _private(preview_dir)
    manifest = _strict_json((root / "manifest.json").read_text(encoding="utf-8"))
    digest = manifest.pop("preview_sha256", None)
    if digest != reviewed_sha256 or digest != _sha(manifest):
        raise ValueError("preview does not match reviewed digest")
    manifest["preview_sha256"] = digest
    if manifest.get("parent_preview_sha256") != PARENT_SHA256 or manifest.get("code_sha256") != _code_hashes():
        raise ValueError("frozen parent or study code changed")
    raw = (root / "inputs.json").read_bytes()
    if _sha_bytes(raw) != manifest.get("input_sha256"):
        raise ValueError("frozen inference inputs changed")
    inputs = _strict_json(raw.decode("utf-8"))
    from capclaims.portfolio_prediction import _validate_base
    if sorted(inputs.get("bases", {})) != manifest.get("people") or len(inputs.get("sessions", [])) != SESSION_N:
        raise ValueError("preview roster or session count changed")
    if inputs["sessions"] != manifest.get("sessions") or inputs["sessions"] != _schedule(manifest["people"]):
        raise ValueError("balanced session schedule differs from frozen design")
    for base in inputs["bases"].values():
        checked = _validate_base(base)
        if checked["round"] != 9 or [item["round"] for item in checked["history"]] != [6, 7, 8]:
            raise ValueError("inference base differs from approved target and history rounds")
    from capclaims.provider_choice_trace import build_provider_order_request
    first_session = inputs["sessions"][0]
    first_base = inputs["bases"][first_session["person_id"]]
    first_request = build_provider_order_request({**first_base,
        "agent_id": first_base["agent_id"] + "__prompt_freeze"}, ())
    prompts = {name: sampling_prompt(first_request, name) for name in PROMPTS}
    if (prompts != manifest.get("prompts_text")
            or {name: _sha_bytes(value.encode("utf-8")) for name, value in prompts.items()} != manifest.get("prompt_sha256")):
        raise ValueError("frozen prompt text or prompt hash differs from the prompt builder")
    if (manifest.get("model") != "jev-1.14.0" or manifest.get("sdk_version") != "0.7.5"
            or manifest.get("prompts") != list(PROMPTS) or manifest.get("decoders") != list(DECODERS)
            or manifest.get("request_cap") != MAX_REQUESTS):
        raise ValueError("frozen model, SDK or factor levels differ from study constants")
    maximum = _preflight_max_body(inputs["bases"])
    if (manifest.get("preflight_max_body_bytes") != maximum or maximum > MAX_BODY
            or manifest.get("body_bytes_cap") != MAX_BODY or manifest.get("request_cap") != MAX_REQUESTS
            or manifest.get("reserved_bytes_cap") != MAX_RESERVED_BYTES
            or manifest.get("max_in_flight") != MAX_IN_FLIGHT or manifest.get("rps_cap") != MAX_RPS
            or manifest.get("bytes_per_second_cap") != MAX_BYTES_PER_SECOND
            or manifest.get("wall_seconds_cap") != MAX_WALL_SECONDS
            or manifest.get("timeout_seconds") != TIMEOUT_SECONDS or manifest.get("retry_cap") != RETRY_CAP):
        raise ValueError("preflight body width or transport caps differ from the frozen constants")
    # Bind labels by the previewed digest, but do not read the file here.
    if manifest.get("scoring_targets_sha256") is None:
        raise ValueError("preview lacks frozen score-label digest")
    from importlib.metadata import version
    if version("typesafe-sdk") != "0.7.5":
        raise ValueError("pinned SDK changed")
    return manifest


async def _serialize_wire(request: Any, prompt: str) -> Any:
    import httpx2
    from typesafe_sdk import AsyncTypeSafeClient, Choice
    from typesafe_sdk._core.retry import RetryPolicy
    from capclaims.backends.jev import MODEL
    from capclaims.portfolio_transport import ExpectedWire, _url_tuple
    captured: dict[str, Any] = {}
    async def capture(outgoing: Any) -> Any:
        captured["method"] = outgoing.method
        captured["url"] = str(outgoing.url)
        captured["body"] = await outgoing.aread()
        criteria = {c.candidate_id: c.label for c in request.candidates}
        answer = {"type": "choice", "choice": next(iter(criteria)), "confidence": 1.0,
            "probabilities": {key: float(key == next(iter(criteria))) for key in criteria}}
        return httpx2.Response(200, json={"model": MODEL,
            "usage": {"input_tokens": 1, "output_tokens": 1}, "answers": {"decision": answer}})
    http = httpx2.AsyncClient(transport=httpx2.MockTransport(capture), follow_redirects=False)
    client = AsyncTypeSafeClient(api_key="dry-run-dummy", model=MODEL,
        retry=RetryPolicy(max_retries=0), http_client=http, base_url="https://api.typesafe.ai", compression=False)
    try:
        await client.system_one(state=render_prompt_state(request, "explicit"),
            questions={"decision": Choice(instructions=prompt,
                criteria={c.candidate_id: c.label for c in request.candidates})}, model=MODEL)
    finally:
        await client.aclose()
    body = captured["body"]
    return ExpectedWire(request.request_id, captured["method"], _url_tuple(captured["url"]), body,
        _sha_bytes(body))


class _SamplingBackend:
    def __new__(cls, prompt: str, **kwargs: Any) -> Any:
        from capclaims.paired_portfolio_engine import ExplanationBackend
        return ExplanationBackend(*kwargs.pop("args", ()), arm="C_joint_explicit",
            render=lambda request, _arm: render_prompt_state(request, "explicit"),
            prompt=lambda request, _arm: sampling_prompt(request, prompt), **kwargs)


async def _run_prompt(preview: Path, output: Path, prompt_name: str, reviewed_preview_sha256: str,
                      fake: bool, fake_transport: Any = None) -> dict[str, Any]:
    import httpx2
    from capclaims.backends.jev import MODEL, load_local_env
    from capclaims.continuation_transport_audit import HttpOverlapTransport, build_event_chain
    from capclaims.injection_study import InMemoryWireRecorder
    from capclaims.portfolio_prediction import _validate_base
    from capclaims.portfolio_transport import DEFAULT_URL, GuardTransport, SharedBudget
    from capclaims.backends.jev import JevBackend
    from typesafe_sdk import AsyncTypeSafeClient
    from typesafe_sdk._core.retry import RetryPolicy
    from capclaims.prompt_sampling_trace import next_action, build_request, advance

    manifest = _strict_json((preview / "manifest.json").read_text(encoding="utf-8"))
    verify_preview(preview, reviewed_preview_sha256)
    if output.exists():
        raise FileExistsError("run cannot be resumed or overwritten")
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    _write(output / "receipt.json", {"preview_sha256": manifest["preview_sha256"],
        "prompt": prompt_name, "execution_mode": "mock" if fake else "provider", "status": "RUNNING",
        "started_at_utc": datetime.now(timezone.utc).isoformat(), "started_monotonic": started,
        "started_perf_ns": time.perf_counter_ns(), "source_policy_sha256": manifest["source_policy_sha256"]})
    # Verify source use before reading local environment or touching a key.
    from capclaims.order_explanation_study import _check_source_policy
    _check_source_policy(manifest["source_policy_sha256"], manifest["source"])
    if not fake:
        load_local_env()
        if not os.getenv("TYPESAFE_API_KEY"):
            raise ValueError("TYPESAFE_API_KEY is unavailable")
    if fake_transport is None:
        if fake:
            async def mock(request: Any) -> Any:
                body = json.loads((await request.aread()).decode("utf-8"))
                candidates = body["questions"]["decision"]["criteria"]
                ordered = list(candidates)
                selected = ordered[0]
                return httpx2.Response(200, json={"model": MODEL,
                    "usage": {"input_tokens": 3, "output_tokens": 1},
                    "answers": {"decision": {"type": "choice", "choice": selected, "confidence": 1.0,
                        "probabilities": {key: float(key == selected) for key in ordered}}}})
            inner = httpx2.MockTransport(mock)
        else:
            inner = httpx2.AsyncHTTPTransport(retries=0)
    else:
        inner = fake_transport
    observer = _DurablePhysicalTransport(HttpOverlapTransport(inner), output / "transport-physical.jsonl",
        raw_path=output / "transport-raw.jsonl", event_path=output / "transport-physical-events.jsonl", mock=fake)
    budget = SharedBudget(output / "attempts.jsonl", max_requests=MAX_REQUESTS,
        max_bytes=MAX_RESERVED_BYTES, rps=100_000 if fake else MAX_RPS,
        bytes_per_second=10_000_000 if fake else MAX_BYTES_PER_SECOND, max_in_flight=MAX_IN_FLIGHT)
    guard = GuardTransport(observer, DEFAULT_URL, budget, max_body_bytes=MAX_BODY)
    recorder = InMemoryWireRecorder(guard)
    api_key = "offline-dummy-key" if fake else os.environ["TYPESAFE_API_KEY"]
    http = httpx2.AsyncClient(transport=recorder, follow_redirects=False)
    client = AsyncTypeSafeClient(api_key=api_key, model=MODEL, retry=RetryPolicy(max_retries=RETRY_CAP),
        timeout=TIMEOUT_SECONDS, http_client=http, base_url="https://api.typesafe.ai", compression=False)
    # ExplanatonBackend shares the pinned Jev SDK parser and _one implementation;
    # its injected prompt and state renderer are the only arm-specific behavior.
    backend = _SamplingBackend(prompt_name, api_key=api_key, rps=1000 if fake else MAX_RPS,
        tps=10_000_000 if fake else MAX_BYTES_PER_SECOND, max_requests=MAX_REQUESTS,
        max_estimated_input_tokens=MAX_RESERVED_BYTES, instruction_profile="portfolio_third_v1",
        objective="portfolio_prediction_v1", client=client, request_timeout=TIMEOUT_SECONDS,
        max_in_flight=MAX_IN_FLIGHT, load_environment=False)
    if not fake:
        claim_dir = preview / "used-prompts"
        claim_dir.mkdir(parents=True, exist_ok=True)
        _write(claim_dir / f"{prompt_name}.json", {"prompt": prompt_name,
            "preview_sha256": manifest["preview_sha256"],
            "claimed_at_utc": datetime.now(timezone.utc).isoformat()})
    inputs = _strict_json((preview / "inputs.json").read_text(encoding="utf-8"))
    sessions = inputs["sessions"]
    rows: list[dict[str, Any]] = []
    event_log = output / "session-events.jsonl"
    status_log = output / "session-status.jsonl"
    event_log.touch(exist_ok=False)
    status_log.touch(exist_ok=False)
    event_log_lock = asyncio.Lock()

    async def persist_new_events(session_id: str, trace: tuple[dict[str, Any], ...], start: int) -> None:
        async with event_log_lock:
            for index, event in enumerate(trace[start:], start):
                _append_jsonl(event_log, {"session_id": session_id, "event_index": index, "event": event})

    async def one(session: Mapping[str, Any]) -> dict[str, Any]:
        base0 = _validate_base(inputs["bases"][session["person_id"]])
        opaque_suffix = _sha([session["person_id"], session["decoder"], session["repeat"]])[:16]
        base = {**base0, "agent_id": f"{base0['agent_id']}__{opaque_suffix}"}
        trace: tuple[dict[str, Any], ...] = ()
        pending_binding: Mapping[str, Any] | None = None
        try:
            while True:
                action = next_action(base, trace, person=session["person_id"], repeat=session["repeat"])
                if action["kind"] == "complete":
                    from capclaims.prompt_sampling_trace import replay
                    result = replay(base, trace, person=session["person_id"], repeat=session["repeat"])
                    row = {**session, "prompt": prompt_name, "status": "complete", "trace": list(trace),
                        "orders": [order.__dict__ for order in result["orders"]],
                        "termination": result.get("terminal_reason")}
                    _append_jsonl(status_log, {"session_id": session["session_id"], "status": "complete",
                        "event_count": len(trace)})
                    return row
                if action["kind"] == "forced":
                    before = len(trace)
                    trace = advance(base, trace, person=session["person_id"], repeat=session["repeat"])
                    await persist_new_events(session["session_id"], trace, before)
                    continue
                if action["kind"] != "request":
                    raise RuntimeError("prompt sampling trace became incomplete")
                request = build_request(base, trace, person=session["person_id"], repeat=session["repeat"])
                pending_binding = __import__("capclaims.portfolio_prediction", fromlist=["_binding"])._binding(request)
                prompt_text = sampling_prompt(request, prompt_name)
                wire = await _serialize_wire(request, prompt_text)
                guard.register(wire)
                (reply,) = await backend.decide_many([request])
                record = recorder.rows.get(request.request_id)
                if not record or record.get("status") != 200:
                    raise RuntimeError("request-ID-bound HTTP response is missing or unsuccessful")
                raw = _strict_json(record["response_body_utf8"])
                answer = raw.get("answers", {}).get("decision")
                if (raw.get("model") != MODEL or not isinstance(answer, Mapping)
                    or set(answer) != {"type", "choice", "confidence", "probabilities"}
                    or answer.get("type") != "choice"):
                    raise ValueError("provider response has invalid model or decision shape")
                usage = {**reply.usage, "service_choice": answer.get("choice"),
                    "service_confidence": answer.get("confidence")}
                from capclaims.engine.protocol import DecisionReply
                reply = DecisionReply.from_request(request, reply.probabilities, reply.raw_scores,
                    backend=reply.backend, model_revision=reply.model_revision, usage=usage, timing=reply.timing)
                before = len(trace)
                trace = advance(base, trace, reply, probabilities=answer.get("probabilities"),
                    decoder=session["decoder"], person=session["person_id"], repeat=session["repeat"])
                await persist_new_events(session["session_id"], trace, before)
                pending_binding = None
        except BaseException as exc:
            if pending_binding is not None:
                failure_event = {"kind": "failure", "origin": "client", "binding": dict(pending_binding),
                    "stage": "provider_or_validation", "error_type": type(exc).__name__}
                async with event_log_lock:
                    _append_jsonl(event_log, {"session_id": session["session_id"],
                        "event_index": len(trace), "event": failure_event})
            _append_jsonl(status_log, {"session_id": session["session_id"], "status": "failed",
                "event_count": len(trace) + int(pending_binding is not None), "error_type": type(exc).__name__})
            raise

    try:
        for start in range(0, SESSION_N, MAX_IN_FLIGHT):
            remaining = MAX_WALL_SECONDS - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError("prompt batch exceeded the frozen wall-clock cap")
            results = await asyncio.wait_for(asyncio.gather(*(one(s) for s in sessions[start:start + MAX_IN_FLIGHT]),
                return_exceptions=True), timeout=remaining)
            errors = [item for item in results if isinstance(item, BaseException)]
            rows.extend(item for item in results if not isinstance(item, BaseException))
            if errors:
                raise RuntimeError(f"prompt batch invalidated ({len(errors)} failures; {type(errors[0]).__name__})")
    except BaseException:
        _write(output / "invalid.json", {"status": "INVALID", "completed_before_failure": len(rows)})
        raise
    finally:
        backend._closed = True
        await client.aclose()
        # Durable snapshots are written even when a session fails.
        _write(output / "events.json", rows)
        _write(output / "events-chain.json", build_event_chain(rows))
        _write(output / "transport.json", list(guard.http_timings))
        _write(output / "transport-precise.json", observer.inner.rows)
        for filename in ("attempts.jsonl", "transport-requests.jsonl", "transport-bodies.jsonl",
                         "transport-raw.jsonl", "transport-physical-events.jsonl"):
            if not (output / filename).exists():
                (output / filename).touch(exist_ok=False)
        if not (output / "transport-physical.jsonl").exists():
            (output / "transport-physical.jsonl").touch(exist_ok=False)
    if observer.peak > MAX_IN_FLIGHT:
        _write(output / "invalid-cap.json", {"status": "INVALID", "reason": "concurrency_cap_exceeded"})
        raise RuntimeError("observed HTTP concurrency exceeded the frozen cap")
    terminal = {"status": "COMPLETE", "prompt": prompt_name,
        "execution_mode": "mock" if fake else "provider", "preview_sha256": manifest["preview_sha256"],
        "receipt_sha256": _sha_bytes((output / "receipt.json").read_bytes()),
        "events_chain_sha256": _sha_bytes((output / "events-chain.json").read_bytes()),
        "event_sha256": _sha_bytes((output / "events.json").read_bytes()),
        "attempts_sha256": _sha_bytes((output / "attempts.jsonl").read_bytes()),
        "requests_sha256": _sha_bytes((output / "transport-requests.jsonl").read_bytes()),
        "bodies_sha256": _sha_bytes((output / "transport-bodies.jsonl").read_bytes()),
        "timings_sha256": _sha_bytes((output / "transport.json").read_bytes()),
        "precise_timings_sha256": _sha_bytes((output / "transport-precise.json").read_bytes()),
        "physical_timings_sha256": _sha_bytes((output / "transport-physical.jsonl").read_bytes()),
        "raw_transport_sha256": _sha_bytes((output / "transport-raw.jsonl").read_bytes()),
        "physical_events_sha256": _sha_bytes((output / "transport-physical-events.jsonl").read_bytes()),
        "session_events_sha256": _sha_bytes((event_log).read_bytes()),
        "session_status_sha256": _sha_bytes((status_log).read_bytes()),
        "budget": budget.usage, "started_monotonic": started,
        "started_perf_ns": json.loads((output / "receipt.json").read_text(encoding="utf-8"))["started_perf_ns"],
        "completed_monotonic": time.monotonic(), "completed_perf_ns": time.perf_counter_ns()}
    _write(output / "terminal.json", terminal)
    return {"status": "COMPLETE", "planned_sessions": SESSION_N, "attempted_sessions": len(rows),
        "requests_charged": budget.usage["requests_charged"], "reserved_bytes": budget.usage["reserved_bytes"],
        "max_overlap": observer.peak}


async def execute_preview(preview_dir: str | Path, run_dir: str | Path, prompt: str,
                          reviewed_preview_sha256: str, mock: bool = False,
                          fake_transport: Any = None) -> dict[str, Any]:
    if prompt not in PROMPTS:
        raise ValueError("prompt must be one of original, explicit, structured")
    output = _private(run_dir)
    run_preexisted = output.exists()
    try:
        return await _run_prompt(_private(preview_dir), output, prompt,
            reviewed_preview_sha256, mock, fake_transport)
    except BaseException as exc:
        if not run_preexisted:
            _finalize_invalid_run(output, prompt, reviewed_preview_sha256, mock, type(exc).__name__)
        raise RuntimeError(f"prompt batch invalidated ({type(exc).__name__})") from None


def _finalize_invalid_run(output: Path, prompt: str, preview_sha256: str, mock: bool,
                          error_type: str) -> None:
    """After a receipt exists, make setup and execution failures durably INVALID."""
    receipt_path = output / "receipt.json"
    if not receipt_path.is_file():
        return
    terminal_path = output / "terminal.json"
    if terminal_path.is_file():
        try:
            if _strict_json(terminal_path.read_text(encoding="utf-8")).get("status") in {"COMPLETE", "INVALID"}:
                return
        except Exception:
            pass
    receipts = _strict_json(receipt_path.read_text(encoding="utf-8"))
    from capclaims.continuation_transport_audit import build_event_chain
    for filename, default in (("events.json", []), ("events-chain.json", []),
        ("transport.json", []), ("transport-precise.json", [])):
        path = output / filename
        if not path.exists():
            _write(path, default)
    for filename in ("attempts.jsonl", "transport-requests.jsonl", "transport-bodies.jsonl",
        "transport-raw.jsonl", "transport-physical.jsonl", "transport-physical-events.jsonl",
        "session-events.jsonl", "session-status.jsonl"):
        path = output / filename
        if not path.exists():
            path.touch(exist_ok=False)
    events_path = output / "events.json"
    events = _strict_json(events_path.read_text(encoding="utf-8"))
    chain = build_event_chain(events)
    chain_path = output / "events-chain.json"
    if _strict_json(chain_path.read_text(encoding="utf-8")) != chain:
        # A partially written chain is replaced only for this incomplete run.
        chain_path.unlink()
        _write(chain_path, chain)
    invalid = output / "invalid.json"
    if not invalid.exists():
        _write(invalid, {"status": "INVALID", "error_type": error_type})
    terminal = {"status": "INVALID", "prompt": prompt,
        "execution_mode": "mock" if mock else "provider", "preview_sha256": preview_sha256,
        "receipt_sha256": _sha_bytes(receipt_path.read_bytes()), "error_type": error_type,
        "event_sha256": _sha_bytes(events_path.read_bytes()),
        "events_chain_sha256": _sha_bytes(chain_path.read_bytes()),
        "attempts_sha256": _sha_bytes((output / "attempts.jsonl").read_bytes()),
        "requests_sha256": _sha_bytes((output / "transport-requests.jsonl").read_bytes()),
        "bodies_sha256": _sha_bytes((output / "transport-bodies.jsonl").read_bytes()),
        "timings_sha256": _sha_bytes((output / "transport.json").read_bytes()),
        "precise_timings_sha256": _sha_bytes((output / "transport-precise.json").read_bytes()),
        "physical_timings_sha256": _sha_bytes((output / "transport-physical.jsonl").read_bytes()),
        "raw_transport_sha256": _sha_bytes((output / "transport-raw.jsonl").read_bytes()),
        "physical_events_sha256": _sha_bytes((output / "transport-physical-events.jsonl").read_bytes()),
        "session_events_sha256": _sha_bytes((output / "session-events.jsonl").read_bytes()),
        "session_status_sha256": _sha_bytes((output / "session-status.jsonl").read_bytes()),
        "started_monotonic": receipts.get("started_monotonic"),
        "started_perf_ns": receipts.get("started_perf_ns"),
        "completed_monotonic": time.monotonic(), "completed_perf_ns": time.perf_counter_ns()}
    if terminal_path.exists():
        terminal_path.unlink()
    _write(terminal_path, terminal)


def replay_run(preview_dir: str | Path, run_dir: str | Path, prompt: str | None = None) -> dict[str, Any]:
    """Rebuild every client trace and join it to the captured raw SDK wire evidence."""
    preview, run = _private(preview_dir), _private(run_dir)
    manifest0 = _strict_json((preview / "manifest.json").read_text(encoding="utf-8"))
    manifest = verify_preview(preview, manifest0.get("preview_sha256", ""))
    if (run / "invalid.json").exists() or (run / "invalid-cap.json").exists():
        raise ValueError("prompt batch is INVALID and cannot be scored")
    terminal = _strict_json((run / "terminal.json").read_text(encoding="utf-8"))
    prompt_name = prompt or terminal.get("prompt")
    if prompt_name not in PROMPTS or terminal.get("prompt") != prompt_name:
        raise ValueError("run prompt differs from requested prompt")
    if terminal.get("status") != "COMPLETE" or terminal.get("preview_sha256") != manifest["preview_sha256"]:
        raise ValueError("run terminal is incomplete or belongs to another preview")
    receipt_bytes = (run / "receipt.json").read_bytes()
    receipt = _strict_json(receipt_bytes.decode("utf-8"))
    if (terminal.get("receipt_sha256") != _sha_bytes(receipt_bytes)
            or receipt.get("preview_sha256") != manifest["preview_sha256"]
            or receipt.get("prompt") != prompt_name
            or receipt.get("execution_mode") != terminal.get("execution_mode")
            or receipt.get("status") != "RUNNING"):
        raise ValueError("pre-provider receipt binding is invalid")
    hashes = {"events.json": "event_sha256", "events-chain.json": "events_chain_sha256",
        "attempts.jsonl": "attempts_sha256", "transport-requests.jsonl": "requests_sha256",
        "transport-bodies.jsonl": "bodies_sha256", "transport.json": "timings_sha256",
        "transport-precise.json": "precise_timings_sha256",
        "transport-physical.jsonl": "physical_timings_sha256",
        "transport-raw.jsonl": "raw_transport_sha256",
        "transport-physical-events.jsonl": "physical_events_sha256",
        "session-events.jsonl": "session_events_sha256", "session-status.jsonl": "session_status_sha256"}
    for filename, key in hashes.items():
        if _sha_bytes((run / filename).read_bytes()) != terminal.get(key):
            raise ValueError("run evidence digest mismatch")
    inputs = _strict_json((preview / "inputs.json").read_text(encoding="utf-8"))
    rows = _strict_json((run / "events.json").read_text(encoding="utf-8"))
    chain = _strict_json((run / "events-chain.json").read_text(encoding="utf-8"))
    from capclaims.continuation_transport_audit import build_event_chain
    if chain != build_event_chain(rows):
        raise ValueError("session event chain differs from reconstructed terminal events")
    expected_rows = [{**session, "prompt": prompt_name} for session in inputs["sessions"]]
    expected_ids = [row["session_id"] for row in expected_rows]
    if len(rows) != SESSION_N or [row.get("session_id") for row in rows] != expected_ids:
        raise ValueError("run rows do not match the frozen per-prompt session schedule")
    sent_rows = _jsonl(run / "transport-requests.jsonl")
    response_rows = _jsonl(run / "transport-bodies.jsonl")
    attempts = _jsonl(run / "attempts.jsonl")
    sent = {row.get("request_id"): row for row in sent_rows}
    responses = {row.get("request_id"): row for row in response_rows}
    timing_rows = _strict_json((run / "transport.json").read_text(encoding="utf-8"))
    timings = {row.get("request_id"): row for row in timing_rows}
    if None in sent or len(sent) != len(sent_rows) or len(responses) != len(response_rows) or len(timings) != len(timing_rows):
        raise ValueError("duplicate or missing HTTP request identity")
    from capclaims.prompt_sampling_trace import replay as replay_trace, build_request
    from capclaims.prompt_sampling_trace import next_action
    from capclaims.backends.jev import MODEL
    used: set[str] = set()
    portfolios = {}
    for row, frozen in zip(rows, expected_rows):
        for key in ("person_id", "decoder", "repeat", "prompt"):
            if row.get(key) != frozen[key]:
                raise ValueError("run factor differs from frozen schedule")
        if row.get("status") != "complete":
            raise ValueError("completed run contains an incomplete session")
        suffix = _sha([row["person_id"], row["decoder"], row["repeat"]])[:16]
        base0 = inputs["bases"][row["person_id"]]
        base = {**base0, "agent_id": f"{base0['agent_id']}__{suffix}"}
        trace = tuple(row["trace"])
        if any(event.get("origin") == "client" and (event.get("decoder") != row["decoder"]
                or event.get("private_person") != row["person_id"] or event.get("repeat") != row["repeat"])
                for event in trace):
            raise ValueError("client trace decoder or random coordinates differ from frozen session factors")
        replayed = replay_trace(base, trace, person=row["person_id"], repeat=row["repeat"])
        if replayed["status"] != "complete" or [o.__dict__ for o in replayed["orders"]] != row.get("orders"):
            raise ValueError("session portfolio differs from client trace replay")
        portfolios[row["session_id"]] = replayed["orders"]
        prefix: tuple[dict[str, Any], ...] = ()
        for event in trace:
            if event["kind"] == "forced":
                prefix = tuple([*prefix, event])
                continue
            rid = event.get("request_id")
            if not isinstance(rid, str) or rid in used or rid not in sent or rid not in responses or rid not in timings:
                raise ValueError("client choice lacks one unique raw HTTP join")
            used.add(rid)
            request = build_request(base, prefix, person=row["person_id"], repeat=row["repeat"])
            if request is None or request.request_id != rid:
                raise ValueError("raw HTTP request id differs from reconstructed request")
            prompt_text = sampling_prompt(request, prompt_name)
            expected_wire = asyncio.run(_serialize_wire(request, prompt_text))
            sent_row, response_row = sent[rid], responses[rid]
            body = sent_row.get("body_utf8", "").encode("utf-8")
            if body != expected_wire.body or _sha_bytes(body) != sent_row.get("body_sha256"):
                raise ValueError("captured SDK request differs from reconstructed state and prompt")
            if response_row.get("request_body_sha256") != sent_row.get("body_sha256") or response_row.get("status") != 200:
                raise ValueError("captured response does not bind to the sent request")
            raw = _strict_json(response_row["response_body_utf8"])
            answer = raw.get("answers", {}).get("decision")
            if raw.get("model") != MODEL or not isinstance(answer, Mapping) \
                    or set(answer) != {"type", "choice", "confidence", "probabilities"} or answer != {key: event_value for key, event_value in (
                    ("type", "choice"), ("choice", event.get("raw_choice")),
                    ("confidence", event.get("service_confidence")),
                    ("probabilities", event.get("raw_probabilities")))}:
                raise ValueError("saved provider answer differs from trace raw evidence")
            prefix = tuple([*prefix, event])
    if used != set(sent) or used != set(responses) or used != set(timings):
        raise ValueError("captured request/response/timing evidence does not exactly join client trace")
    attempt_ids = [row.get("request_id") for row in attempts]
    if len(attempts) != len(used) or len(attempts) > MAX_REQUESTS or set(attempt_ids) != used or len(set(attempt_ids)) != len(attempt_ids):
        raise ValueError("attempt ledger count differs from reconstructed calls or exceeds cap")
    from capclaims.portfolio_transport import DEFAULT_URL, _url_tuple
    for index, attempt in enumerate(attempts, 1):
        rid = attempt.get("request_id")
        sent_row = sent.get(rid, {})
        body = sent_row.get("body_utf8", "").encode("utf-8")
        body_sha = _sha_bytes(body)
        if (rid not in used or attempt.get("seq") != index or attempt.get("method") != "POST"
                or _url_tuple(attempt.get("url", "")) != _url_tuple(DEFAULT_URL)
                or attempt.get("body_bytes") != len(body) or len(body) > MAX_BODY
                or attempt.get("expected_sha256") != body_sha or attempt.get("actual_sha256") != body_sha
                or sent_row.get("body_sha256") != body_sha
                or attempt.get("reserved_bytes") != MAX_BODY or attempt.get("state") != "attempted"):
            raise ValueError("attempt ledger violates request ordering or frozen transport caps")
    # Independent durable event/status/clock audit; raw records cannot be
    # silently omitted from the terminal session snapshots.
    session_events = _jsonl(run / "session-events.jsonl")
    by_session: dict[str, list[dict[str, Any]]] = {}
    for item in session_events:
        by_session.setdefault(item.get("session_id"), []).append(item)
    for row in rows:
        saved = sorted(by_session.get(row["session_id"], []), key=lambda item: item.get("event_index", -1))
        if [item.get("event") for item in saved] != row["trace"] or [item.get("event_index") for item in saved] != list(range(len(row["trace"]))):
            raise ValueError("durable per-event log differs from session trace snapshot")
    status_rows = _jsonl(run / "session-status.jsonl")
    if len(status_rows) != SESSION_N or {item.get("session_id") for item in status_rows} != set(expected_ids) \
            or any(item.get("status") != "complete" for item in status_rows):
        raise ValueError("durable session status log is incomplete")
    status_by_id = {item["session_id"]: item for item in status_rows}
    if len(status_by_id) != len(status_rows) or any(status_by_id[row["session_id"]].get("event_count") != len(row["trace"])
            for row in rows):
        raise ValueError("durable session status counts differ from reconstructed traces")
    precise = _strict_json((run / "transport-precise.json").read_text(encoding="utf-8"))
    precise_by_id = {item.get("request_id"): item for item in precise}
    physical = _jsonl(run / "transport-physical.jsonl")
    if {r.get("request_id") for r in physical} != used or len(physical) != len(used) \
            or len(precise) != len(used) or set(precise_by_id) != used:
        raise ValueError("physical HTTP timing rows do not exactly cover requests")
    for item in physical:
        if (not isinstance(item.get("started_perf_ns"), int) or not isinstance(item.get("finished_perf_ns"), int)
                or item["finished_perf_ns"] < item["started_perf_ns"] or item.get("status") != 200
                or not isinstance(item.get("dispatch_monotonic"), (int, float))
                or not isinstance(item.get("finished_monotonic"), (int, float))
                or item["finished_monotonic"] < item["dispatch_monotonic"]
                or item.get("reserved_bytes") != MAX_BODY):
            raise ValueError("physical HTTP start/end evidence is invalid")
    intervals = sorted((item["started_perf_ns"], item["finished_perf_ns"]) for item in physical)
    active = peak = 0
    points = sorted([(start, 1) for start, _ in intervals] + [(end, -1) for _, end in intervals])
    for _, delta in points:
        active += delta
        peak = max(peak, active)
    if peak > MAX_IN_FLIGHT:
        raise ValueError("physical HTTP overlap exceeded the frozen concurrency cap")
    if (terminal.get("budget", {}).get("requests_charged") != len(used)
            or terminal.get("budget", {}).get("reserved_bytes") != len(used) * MAX_BODY
            or len(used) * MAX_BODY > MAX_RESERVED_BYTES):
        raise ValueError("terminal request/byte budget differs from reconstructed attempts")
    receipt_utc = datetime.fromisoformat(receipt["started_at_utc"])
    if receipt_utc.utcoffset() is None:
        raise ValueError("receipt UTC clock lacks a timezone")
    receipt_mono, receipt_perf = receipt.get("started_monotonic"), receipt.get("started_perf_ns")
    if (isinstance(receipt_mono, bool) or not isinstance(receipt_mono, (int, float))
            or not isinstance(receipt_perf, int) or receipt_perf < 0
            or terminal.get("started_monotonic") != receipt_mono
            or terminal.get("started_perf_ns") != receipt_perf):
        raise ValueError("terminal and receipt clocks are not bound")
    physical_events = _jsonl(run / "transport-physical-events.jsonl")
    start_events = [row for row in physical_events if row.get("kind") == "dispatch_start"]
    end_events = [row for row in physical_events if row.get("kind") == "dispatch_end"]
    if len(start_events) != len(used) or len(end_events) != len(used):
        raise ValueError("physical dispatch event ledger has missing start or end rows")
    start_by_id = {row.get("request_id"): row for row in start_events}
    end_by_id = {row.get("request_id"): row for row in end_events}
    if set(start_by_id) != used or set(end_by_id) != used:
        raise ValueError("physical dispatch event identities are not unique and complete")
    observed_starts: set[str] = set()
    observed_ends: set[str] = set()
    previous_end_ns = -1
    for item in physical_events:
        rid = item.get("request_id")
        if item.get("kind") == "dispatch_start":
            if rid in observed_starts or rid in observed_ends:
                raise ValueError("physical dispatch starts violate unique causal order")
            observed_starts.add(rid)
        elif item.get("kind") == "dispatch_end":
            finish_ns = item.get("finished_perf_ns")
            if (rid not in observed_starts or rid in observed_ends or not isinstance(finish_ns, int)
                    or finish_ns < previous_end_ns or finish_ns < item.get("dispatch_perf_ns", math.inf)):
                raise ValueError("physical dispatch end precedes or duplicates its start")
            observed_ends.add(rid)
            previous_end_ns = finish_ns
        else:
            raise ValueError("physical dispatch event has an unknown event kind")
    if observed_starts != used or observed_ends != used:
        raise ValueError("physical dispatch event sequence is incomplete")
    raw_rows = _jsonl(run / "transport-raw.jsonl")
    raw_by_id = {row.get("request_id"): row for row in raw_rows}
    if len(raw_rows) != len(used) or set(raw_by_id) != used:
        raise ValueError("raw response ledger does not uniquely cover every dispatched call")
    dispatches = []
    for rid in used:
        start_event, end_event = start_by_id[rid], end_by_id[rid]
        phys = next(row for row in physical if row["request_id"] == rid)
        precise_row = precise_by_id[rid]
        raw_row = raw_by_id[rid]
        if (start_event.get("request_body_sha256") != sent[rid].get("body_sha256")
                or start_event.get("reserved_bytes") != MAX_BODY
                or end_event.get("status") != 200 or end_event.get("response_body_sha256") != raw_row.get("response_body_sha256")
                or end_event.get("finished_perf_ns") != phys.get("finished_perf_ns")
                or end_event.get("dispatch_perf_ns") != phys.get("started_perf_ns")
                or end_event.get("started_monotonic") != phys.get("dispatch_monotonic")
                or precise_row.get("started_perf_ns") != phys.get("started_perf_ns")
                or precise_row.get("finished_perf_ns") != phys.get("finished_perf_ns")
                or precise_row.get("start_event_seq") != phys.get("start_event_seq")
                or precise_row.get("end_event_seq") != phys.get("end_event_seq")
                or precise_row.get("status") != phys.get("status")
                or phys.get("request_body_sha256") != sent[rid].get("body_sha256")):
            raise ValueError("dispatch clock or body evidence differs across durable ledgers")
        try:
            response_bytes = base64.b64decode(raw_row["response_body_base64"], validate=True)
        except (KeyError, ValueError, TypeError) as exc:
            raise ValueError("raw response evidence is malformed") from exc
        if (_sha_bytes(response_bytes) != raw_row.get("response_body_sha256")
                or len(response_bytes) != raw_row.get("response_body_bytes")
                or responses[rid].get("response_body_sha256") != raw_row.get("response_body_sha256")
                or raw_row.get("status") != responses[rid].get("status")):
            raise ValueError("raw response body does not match transport response evidence")
        if phys.get("started_perf_ns", 0) < receipt_perf \
                or phys.get("dispatch_monotonic", 0) < receipt_mono \
                or phys.get("finished_monotonic", math.inf) - receipt_mono > MAX_WALL_SECONDS:
            raise ValueError("physical dispatch clock precedes receipt or exceeds wall cap")
        actual_mono = receipt_mono + (precise_row["started_perf_ns"] - receipt_perf) / 1e9
        if abs(actual_mono - phys["dispatch_monotonic"]) > 1e-6:
            raise ValueError("physical clock is not bound to the actual transport start")
        dispatches.append((actual_mono, MAX_BODY))
    if terminal.get("execution_mode") == "provider":
        dispatches.sort()
        dispatches.sort()
        for index in range(6, len(dispatches)):
            delta = dispatches[index][0] - dispatches[index - 6][0]
            if delta <= 1.0 or 6 * MAX_BODY > MAX_BYTES_PER_SECOND:
                raise ValueError("physical rolling dispatch rate or reserved-byte rate exceeds cap")
    started = terminal.get("started_monotonic")
    completed = terminal.get("completed_monotonic")
    if not isinstance(started, (int, float)) or not isinstance(completed, (int, float)) \
            or completed < started or completed - started > MAX_WALL_SECONDS:
        raise ValueError("physical batch elapsed time exceeds frozen wall cap")
    from capclaims.continuation_transport_audit import audit_transport_evidence
    transport_audit = audit_transport_evidence(run, terminal, used, manifest["preview_sha256"])
    if (completed < max(item["finished_monotonic"] for item in physical)
            or terminal.get("completed_perf_ns", 0) < max(item["finished_perf_ns"] for item in physical)):
        raise ValueError("terminal completion clock precedes physical transport completion")
    return {"status": "REPLAY_VALID", "prompt": prompt_name, "session_n": len(rows),
        "request_n": len(used), "portfolios": portfolios, "rows": rows,
        "execution_mode": terminal.get("execution_mode"), "transport_audit": transport_audit}


def score_run(preview_dir: str | Path, run_dir: str | Path, prompt: str | None = None) -> dict[str, Any]:
    """Score only after full replay validation; persist private cases and expose aggregates."""
    replayed = replay_run(preview_dir, run_dir, prompt)
    preview, run = _private(preview_dir), _private(run_dir)
    manifest = _strict_json((preview / "manifest.json").read_text(encoding="utf-8"))
    label_bytes = (preview / "scoring-targets.json").read_bytes()
    if _sha_bytes(label_bytes) != manifest.get("scoring_targets_sha256"):
        raise ValueError("score labels differ from previewed immutable digest")
    targets = _strict_json(label_bytes.decode("utf-8"))
    inputs = _strict_json((preview / "inputs.json").read_text(encoding="utf-8"))
    from capclaims.paper.orders import PaperOrder
    from capclaims.portfolio_metrics import portfolio_comparison
    cases = []
    for session in replayed["rows"]:
        person = session["person_id"]
        base = inputs["bases"][person]
        prediction = replayed["portfolios"][session["session_id"]]
        target = targets[person]
        human = [PaperOrder(f"{person}:truth:{i}", person, str(row.get("side", row.get("type"))).lower(),
            row.get("qty", row.get("amount")), row.get("limit_dollars", row.get("price")))
            for i, row in enumerate(target["orders"])]
        pred = [PaperOrder(f"{person}:pred:{i}", person, order.side, order.qty, order.limit_dollars)
            for i, order in enumerate(prediction)]
        comp = portfolio_comparison(pred, human)
        trace = session["trace"]
        buys = [order.limit_dollars for order in pred if order.side == "buy"]
        sells = [order.limit_dollars for order in pred if order.side == "sell"]
        retained = [event["retained_raw_mass"] for event in trace if event.get("origin") == "client"]
        cutoff_retained = [event["cutoff_retained_raw_mass"] for event in trace if event.get("origin") == "client"]
        cases.append({"session_id": session["session_id"], "person_id": person, "prompt": session["prompt"],
            "decoder": session["decoder"], "repeat": session["repeat"], **comp,
            "order_count_error": abs(len(pred) - len(human)),
            "buy_quantity_error": abs(sum(o.qty for o in pred if o.side == "buy") - sum(o.qty for o in human if o.side == "buy")),
            "sell_quantity_error": abs(sum(o.qty for o in pred if o.side == "sell") - sum(o.qty for o in human if o.side == "sell")),
            "duplicate_order_n": len(pred) - len(set((o.side, o.qty, o.limit_dollars) for o in pred)),
            "positive_buy_excess": max(sum(o.qty for o in pred if o.side == "buy") - sum(o.qty for o in human if o.side == "buy"), 0),
            "five_order": len(pred) == 5,
            "termination": ("stop" if any(event.get("origin") == "client" and event.get("value") == "STOP"
                for event in trace) else "max_orders" if len(pred) == 5 else "other_terminal"),
            "buy_limit_price_mean": (sum(buys) / len(buys) if buys else None),
            "sell_limit_price_mean": (sum(sells) / len(sells) if sells else None),
            "retained_mass_mean": (sum(retained) / len(retained) if retained else None),
            "cutoff_retained_mass_mean": (sum(cutoff_retained) / len(cutoff_retained) if cutoff_retained else None),
            "cutoff_dropped_mass_mean": (sum(e["cutoff_dropped_mass"] for e in trace if e.get("origin") == "client") / len(retained) if retained else None),
            "topk_dropped_mass_mean": (sum(e["topk_dropped_mass"] for e in trace if e.get("origin") == "client") / len(retained) if retained else None),
            "decision_n": len(retained),
            "nonmax_provider_choice_n": sum(e.get("nonmax_provider_choice") is True for e in trace),
            "service_choice_tie_order_disagreement_n": sum(e.get("service_choice_tie_order_disagreement") is True for e in trace),
            "order_multiset_signature": sorted((o.side, o.qty, o.limit_dollars) for o in pred)})
    from capclaims.prompt_sampling_analysis import summarize_cells
    summary = summarize_cells(cases)
    _write_idempotent(run / "private-scores.json", {"status": "SCORED_OFFLINE", "prompt": replayed["prompt"],
        "preview_sha256": manifest["preview_sha256"], "cases": cases, "summary": summary})
    configs = {}
    for decoder in DECODERS:
        decoder_rows = [row for row in cases if row["decoder"] == decoder]
        configs[decoder] = {"mean_primary_curve_error": summary["cell_means"][f"{replayed['prompt']}/{decoder}"],
            "forecast_n": len(decoder_rows), "person_n": 6,
            "diagnostics": _participant_diagnostics(decoder_rows)}
    safe = {"status": "SCORED_OFFLINE", "prompt": replayed["prompt"],
        "execution_mode": replayed["execution_mode"], "preview_sha256": manifest["preview_sha256"],
        "configs": configs, "paired_differences": _public_paired(summary["paired_differences"]),
        "repeat_consistency": summary["repeat_consistency"],
        "limitations": ["Six selected people; no population-equivalence claim.",
            "Two repeats support descriptive sampling diagnostics only.",
            "Primary curve distance couples quantity and price; raw side-specific price means are diagnostics only."]}
    _write_idempotent(run / "safe-summary.json", safe)
    return safe


def _public_paired(paired: Mapping[str, Any]) -> dict[str, Any]:
    return {group: {key: {name: value for name, value in result.items() if name != "person_differences"}
        for key, result in items.items()} for group, items in paired.items()}


def _participant_diagnostics(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    numeric = ("order_count_error", "empty_agreement", "side_class_agreement", "exact_order_multiset_match",
        "buy_quantity_error", "sell_quantity_error", "duplicate_order_n", "positive_buy_excess",
        "five_order", "buy_limit_price_mean", "sell_limit_price_mean", "retained_mass_mean", "cutoff_retained_mass_mean",
        "cutoff_dropped_mass_mean", "topk_dropped_mass_mean", "nonmax_provider_choice_n",
        "service_choice_tie_order_disagreement_n")
    people = sorted({row["person_id"] for row in rows})
    result = {}
    for metric in numeric:
        per_person = {}
        for person in people:
            values = [float(row[metric]) for row in rows if row["person_id"] == person and row[metric] is not None]
            if values:
                per_person[person] = sum(values) / len(values)
        result[metric] = {"participant_averaged_mean": (sum(per_person.values()) / len(per_person) if per_person else None),
            "participants_with_value_n": len(per_person)}
    result["termination_counts"] = {kind: sum(row["termination"] == kind for row in rows)
        for kind in ("stop", "max_orders", "other_terminal")}
    result["decision_n"] = sum(int(row["decision_n"]) for row in rows)
    result["limitations"] = ["Primary curve distance couples quantity and price.",
        "Raw order-price means are side-specific diagnostics, not matched-price accuracy."]
    return result


def compare_three_runs(preview_dir: str | Path, runs: Mapping[str, str | Path]) -> dict[str, Any]:
    if set(runs) != set(PROMPTS):
        raise ValueError("comparison requires all three named prompt runs")
    safe = {prompt: score_run(preview_dir, run, prompt) for prompt, run in runs.items()}
    if len({item["preview_sha256"] for item in safe.values()}) != 1:
        raise ValueError("prompt batches do not share the same frozen preview")
    private_rows = []
    for prompt, run_dir in runs.items():
        path = _private(run_dir) / "private-scores.json"
        score_record = _strict_json(path.read_text(encoding="utf-8"))
        if (score_record.get("status") != "SCORED_OFFLINE" or score_record.get("prompt") != prompt
                or score_record.get("preview_sha256") != safe[prompt]["preview_sha256"]):
            raise ValueError("private score rows do not belong to the requested prompt and shared preview")
        private_rows.extend(score_record.get("cases", []))
    from capclaims.prompt_sampling_analysis import summarize_three_prompt_cases
    return summarize_three_prompt_cases(private_rows)

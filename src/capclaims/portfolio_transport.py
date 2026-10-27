"""One-shot exact-wire HTTP guard and shared portfolio-study budget."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any
from urllib.parse import urlsplit

import httpx2

from capclaims.backends.jev import _CURRENT_REQUEST_ID

MAX_REQUESTS = 420
MAX_BODY_BYTES = 7_500
MAX_RESERVED_BYTES = 3_200_000
MAX_RPS = 40.0
MAX_BYTES_PER_SECOND = 50_000.0
MAX_IN_FLIGHT = 10
DEFAULT_URL = "https://api.typesafe.ai/v1/systemone"


def _url_tuple(url: str) -> tuple[str, str, int, str, str]:
    parsed = urlsplit(url)
    if not parsed.scheme or not parsed.hostname:
        raise ValueError("expected URL must be absolute")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    return parsed.scheme, parsed.hostname.lower(), port, parsed.path, parsed.query


@dataclass(frozen=True)
class ExpectedWire:
    request_id: str
    method: str
    url: tuple[str, str, int, str, str]
    body: bytes
    body_sha256: str


async def serialize_sdk_wire(request: Any, profile: str, effective_state: dict[str, Any] | None = None) -> ExpectedWire:
    """Ask the pinned SDK to serialize one request through an offline MockTransport."""
    from typesafe_sdk import AsyncTypeSafeClient, Choice
    from typesafe_sdk._core.retry import RetryPolicy
    from capclaims.backends.jev import MODEL
    from capclaims.backends.jev_instructions import build_instruction

    captured: dict[str, Any] = {}
    first_candidate = request.candidates[0].candidate_id

    async def capture(outgoing: httpx2.Request) -> httpx2.Response:
        captured["method"] = outgoing.method
        captured["url"] = str(outgoing.url)
        captured["body"] = await outgoing.aread()
        return httpx2.Response(200, json={"model": MODEL,
            "usage": {"input_tokens": 1, "output_tokens": 1},
            "answers": {"decision": {"type": "choice", "choice": first_candidate,
                "confidence": 1.0, "probabilities": {c.candidate_id: float(c.candidate_id == first_candidate)
                    for c in request.candidates}}}})

    http = httpx2.AsyncClient(transport=httpx2.MockTransport(capture), follow_redirects=False)
    client = AsyncTypeSafeClient(api_key="dry-run-dummy", model=MODEL,
        retry=RetryPolicy(max_retries=0),
        http_client=http, base_url="https://api.typesafe.ai", compression=False)
    try:
        await client.system_one(state=dict(effective_state if effective_state is not None else request.observation),
            questions={"decision": Choice(instructions=build_instruction(request, profile),
                criteria={c.candidate_id: c.label for c in request.candidates})}, model=MODEL)
    finally:
        await client.aclose()
    body = captured["body"]
    return ExpectedWire(request.request_id, captured["method"], _url_tuple(captured["url"]), body,
        hashlib.sha256(body).hexdigest())


class SharedBudget:
    """A global process-wide scheduler; all requests reserve the full byte ceiling."""

    def __init__(self, ledger_path: str | Path, *, max_requests: int = MAX_REQUESTS,
                 max_bytes: int = MAX_RESERVED_BYTES, rps: float = MAX_RPS,
                 bytes_per_second: float = MAX_BYTES_PER_SECOND,
                 max_in_flight: int = MAX_IN_FLIGHT) -> None:
        if not 1 <= max_in_flight <= MAX_IN_FLIGHT:
            raise ValueError("global in-flight limit must be 1 through 10")
        self.ledger_path = Path(ledger_path)
        self.max_requests, self.max_bytes = max_requests, max_bytes
        self.rps, self.bytes_per_second = rps, bytes_per_second
        self._requests = 0
        self._reserved_bytes = 0
        self._reservation_lock = asyncio.Lock()
        self._pace_lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(max_in_flight)
        self._next_request = 0.0
        self._next_bytes = 0.0

    @property
    def usage(self) -> dict[str, int]:
        return {"requests_charged": self._requests, "reserved_bytes": self._reserved_bytes,
            "max_requests": self.max_requests, "max_reserved_bytes": self.max_bytes}

    async def _charge_and_persist(self, wire: ExpectedWire, actual: bytes, method: str, url: str) -> None:
        async with self._reservation_lock:
            if self._requests + 1 > self.max_requests:
                raise RuntimeError("portfolio global request budget exhausted")
            if self._reserved_bytes + MAX_BODY_BYTES > self.max_bytes:
                raise RuntimeError("portfolio global reserved-byte budget exhausted")
            row = {"seq": self._requests + 1, "request_id": wire.request_id,
                "expected_sha256": wire.body_sha256,
                "actual_sha256": hashlib.sha256(actual).hexdigest(), "method": method,
                "url": url, "body_bytes": len(actual), "reserved_bytes": MAX_BODY_BYTES,
                "state": "attempted"}
            self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(self.ledger_path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
            try:
                payload = (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
                offset = 0
                while offset < len(payload):
                    offset += os.write(fd, payload[offset:])
                os.fsync(fd)
            finally:
                os.close(fd)
            self._requests += 1
            self._reserved_bytes += MAX_BODY_BYTES

    async def acquire(self, estimated_bytes: int) -> None:
        async with self._pace_lock:
            now = time.monotonic()
            start = max(now, self._next_request, self._next_bytes)
            self._next_request = start + 1.0 / self.rps
            self._next_bytes = start + estimated_bytes / self.bytes_per_second
        delay = start - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)


class GuardTransport(httpx2.AsyncBaseTransport):
    """Validate exact uncompressed SDK wire bytes before charging and forwarding."""

    def __init__(self, inner: httpx2.AsyncBaseTransport, expected_url: str,
                 budget: SharedBudget, *, max_body_bytes: int = MAX_BODY_BYTES) -> None:
        self.inner = inner
        self.expected_url = _url_tuple(expected_url)
        self.budget = budget
        self.max_body_bytes = max_body_bytes
        self._expected: dict[str, ExpectedWire] = {}
        self._consumed: set[str] = set()
        self.http_timings: list[dict[str, Any]] = []

    def register(self, wire: ExpectedWire) -> None:
        if wire.request_id in self._expected or wire.request_id in self._consumed:
            raise ValueError("duplicate one-shot wire registration")
        self._expected[wire.request_id] = wire

    async def handle_async_request(self, request: httpx2.Request) -> httpx2.Response:
        request_id = _CURRENT_REQUEST_ID.get()
        if request_id in self._consumed:
            raise RuntimeError("consumed one-shot request context")
        if not request_id or request_id not in self._expected:
            raise RuntimeError("missing, unknown, or consumed request context")
        wire = self._expected[request_id]
        try:
            body = await request.aread()
            actual_url = str(request.url)
            if (request.method != wire.method or _url_tuple(actual_url) != wire.url
                    or wire.url != self.expected_url):
                raise RuntimeError("HTTP destination or method differs from frozen wire")
            if len(body) > self.max_body_bytes or len(body) > MAX_BODY_BYTES:
                raise RuntimeError("uncompressed HTTP body exceeds the frozen byte bound")
            if request.headers.get("content-encoding"):
                raise RuntimeError("compressed HTTP requests are forbidden")
            if body != wire.body or hashlib.sha256(body).hexdigest() != wire.body_sha256:
                raise RuntimeError("serialized SDK body differs from the frozen wire")
            self._consumed.add(request_id)
            await self.budget.acquire(MAX_BODY_BYTES)
            await self.budget._charge_and_persist(wire, body, request.method, actual_url)
            async with self.budget._semaphore:
                self._append_request_record(request_id, body)
                started = time.monotonic()
                response = await self.inner.handle_async_request(request)
                response_body = await response.aread()
                finished = time.monotonic()
                self._append_private_wire_record(request_id, body, response_body, response.status_code)
        except BaseException as exc:
            if "started" in locals():
                self.http_timings.append({"request_id": request_id, "started_at": started,
                    "finished_at": time.monotonic(), "status": type(exc).__name__})
            raise
        finally:
            self._expected.pop(request_id, None)
        self.http_timings.append({"request_id": request_id, "started_at": started,
            "finished_at": finished, "status": response.status_code})
        return response

    def _append_private_wire_record(self, request_id: str, request_body: bytes,
                                    response_body: bytes, status: int) -> None:
        path = self.budget.ledger_path.with_name("transport-bodies.jsonl")
        row = {"request_id": request_id, "status": status,
            "request_body_sha256": hashlib.sha256(request_body).hexdigest(),
            "response_body_sha256": hashlib.sha256(response_body).hexdigest(),
            "request_body_utf8": request_body.decode("utf-8", errors="replace"),
            "response_body_utf8": (response_body.decode("utf-8", errors="replace")
                if 200 <= status < 300 else None)}
        payload = (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
        fd = os.open(path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
        try:
            offset = 0
            while offset < len(payload):
                offset += os.write(fd, payload[offset:])
            os.fsync(fd)
        finally:
            os.close(fd)

    def _append_request_record(self, request_id: str, body: bytes) -> None:
        path = self.budget.ledger_path.with_name("transport-requests.jsonl")
        row = {"request_id": request_id, "body_sha256": hashlib.sha256(body).hexdigest(),
            "body_bytes": len(body), "body_utf8": body.decode("utf-8", errors="replace")}
        payload = (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
        fd = os.open(path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
        try:
            offset = 0
            while offset < len(payload):
                offset += os.write(fd, payload[offset:])
            os.fsync(fd)
        finally:
            os.close(fd)

    async def aclose(self) -> None:
        await self.inner.aclose()

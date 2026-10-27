"""Independent durable evidence checks for the bounded continuation experiment."""
from __future__ import annotations

from datetime import datetime
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from capclaims.portfolio_transport import DEFAULT_URL, _url_tuple
from capclaims.randomness import canonical_json


class HttpOverlapTransport:
    """Observe actual forwarded HTTP work below the semaphore, without coarse-clock ties."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.rows: list[dict[str, Any]] = []
        self.active = self.peak = self.sequence = 0

    async def handle_async_request(self, request: Any) -> Any:
        from capclaims.backends.jev import _CURRENT_REQUEST_ID
        rid = _CURRENT_REQUEST_ID.get()
        if not rid:
            raise RuntimeError("physical transport has no guarded request identity")
        self.sequence += 1
        start_seq = self.sequence
        started = time.perf_counter_ns()
        self.active += 1
        self.peak = max(self.peak, self.active)
        status: int | str = "incomplete"
        try:
            response = await self.inner.handle_async_request(request)
            await response.aread()
            status = response.status_code
            return response
        except BaseException as exc:
            status = type(exc).__name__
            raise
        finally:
            finished = time.perf_counter_ns()
            self.sequence += 1
            self.rows.append({"request_id": rid, "status": status,
                "start_event_seq": start_seq, "end_event_seq": self.sequence,
                "started_perf_ns": started, "finished_perf_ns": finished})
            self.active -= 1

    async def aclose(self) -> None:
        await self.inner.aclose()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_event_chain(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    previous = "0" * 64
    chain = []
    for index, row in enumerate(rows):
        item = {"index": index, "previous_sha256": previous,
                "event_sha256": _sha(canonical_json(row).encode("utf-8"))}
        previous = _sha(canonical_json(item).encode("utf-8"))
        chain.append({**item, "sha256": previous})
    return chain


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _index(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    result = {}
    for row in rows:
        rid = row.get("request_id")
        if not isinstance(rid, str) or not rid or rid in result:
            raise ValueError(f"missing or duplicate request id in {label}")
        result[rid] = row
    return result


def audit_transport_evidence(run_dir: str | Path, terminal: Mapping[str, Any],
                             used_request_ids: set[str], preview_sha256: str) -> dict[str, Any]:
    """Check raw ledgers, clocks, byte caps and session chain independently of traces."""
    root = Path(run_dir)
    receipt_bytes = (root / "receipt.json").read_bytes()
    receipt = json.loads(receipt_bytes)
    if (terminal.get("status") != "COMPLETE"
            or terminal.get("preview_sha256") != preview_sha256
            or terminal.get("receipt_sha256") != _sha(receipt_bytes)
            or receipt.get("preview_sha256") != preview_sha256
            or receipt.get("execution_mode") not in {"mock", "provider"}
            or receipt.get("execution_mode") != terminal.get("execution_mode")):
        raise ValueError("terminal or pre-provider receipt binding is invalid")
    try:
        utc = datetime.fromisoformat(receipt["started_at_utc"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("receipt lacks a valid wall-clock timestamp") from exc
    if utc.utcoffset() is None:
        raise ValueError("receipt wall-clock timestamp must include its UTC offset")
    started = receipt.get("started_monotonic")
    if isinstance(started, bool) or not isinstance(started, (int, float)) or not math.isfinite(started):
        raise ValueError("receipt monotonic timestamp is invalid")

    rows = _load(root / "events.json")
    chain_bytes = (root / "events-chain.json").read_bytes()
    if (terminal.get("events_chain_sha256") != _sha(chain_bytes)
            or json.loads(chain_bytes) != build_event_chain(rows)):
        raise ValueError("session event chain is inconsistent with saved events")

    def jsonl(name: str) -> list[dict[str, Any]]:
        return [json.loads(line) for line in (root / name).read_text(encoding="utf-8").splitlines() if line]

    attempts = jsonl("attempts.jsonl")
    requested = jsonl("transport-requests.jsonl")
    returned = jsonl("transport-bodies.jsonl")
    timed = _load(root / "transport.json")
    mappings = [_index(values, name) for values, name in
                ((attempts, "attempts"), (requested, "requests"), (returned, "responses"), (timed, "timings"))]
    if any(set(mapping) != used_request_ids for mapping in mappings):
        raise ValueError("raw transport ledgers do not fully join the reconstructed calls")
    n = len(attempts)
    if not 1 <= n <= 840 or [row.get("seq") for row in attempts] != list(range(1, n + 1)):
        raise ValueError("request count or attempt sequence violates the frozen budget")
    budget = terminal.get("budget", {})
    if (budget.get("requests_charged") != n or budget.get("reserved_bytes") != n * 7500
            or budget.get("max_requests") != 840 or budget.get("max_reserved_bytes") != 6_300_000):
        raise ValueError("terminal counters disagree with independently charged requests")

    sweep = []
    maximum_body = 0
    starts, finishes = [], []
    for rid, attempt in mappings[0].items():
        req, res, timing = (mapping[rid] for mapping in mappings[1:])
        try:
            body = req["body_utf8"].encode("utf-8")
            response = res["response_body_utf8"].encode("utf-8")
            url = _url_tuple(attempt.get("url", ""))
        except (KeyError, AttributeError, TypeError, ValueError) as exc:
            raise ValueError("raw HTTP evidence is malformed") from exc
        digest = _sha(body)
        if (attempt.get("method") != "POST" or url != _url_tuple(DEFAULT_URL)
                or attempt.get("state") != "attempted" or attempt.get("reserved_bytes") != 7500
                or not 0 < len(body) <= 7500 or attempt.get("body_bytes") != len(body)
                or req.get("body_bytes") != len(body) or req.get("body_sha256") != digest
                or attempt.get("actual_sha256") != digest or attempt.get("expected_sha256") != digest
                or res.get("request_body_sha256") != digest or res.get("request_body_utf8") != req["body_utf8"]
                or res.get("response_body_sha256") != _sha(response)
                or res.get("status") != 200 or timing.get("status") != 200):
            raise ValueError("raw HTTP hashes, status, byte bounds or destination are inconsistent")
        start, finish = timing.get("started_at"), timing.get("finished_at")
        if any(isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t)
               for t in (start, finish)):
            raise ValueError("HTTP clock evidence is non-finite or missing")
        if start < started or finish < start or finish - started > 600:
            raise ValueError("HTTP event precedes receipt or exceeds the hard wall-clock limit")
        starts.append(start)
        finishes.append(finish)
        sweep.extend(((start, 1), (finish, -1)))
        maximum_body = max(maximum_body, len(body))
    active = peak = 0
    for _when, delta in sorted(sweep, key=lambda item: (item[0], item[1])):
        active += delta
        peak = max(peak, active)
    if active or peak > 10:
        raise ValueError("HTTP overlap exceeds the global ten-request limit")
    coarse_peak = peak
    precise_bytes = (root / "transport-precise.json").read_bytes()
    if terminal.get("precise_timings_sha256") != _sha(precise_bytes):
        raise ValueError("precise HTTP timing digest is invalid")
    precise = _index(json.loads(precise_bytes), "precise timings")
    if set(precise) != used_request_ids:
        raise ValueError("precise HTTP events do not join the charged requests")
    initial_ns = receipt.get("started_perf_ns")
    if type(initial_ns) is not int or initial_ns < 0:
        raise ValueError("receipt lacks the precise independent clock origin")
    sequence = []
    for rid, row in precise.items():
        begin, end = row.get("start_event_seq"), row.get("end_event_seq")
        start_ns, finish_ns = row.get("started_perf_ns"), row.get("finished_perf_ns")
        if (any(type(t) is not int for t in (begin, end, start_ns, finish_ns))
                or not 1 <= begin < end <= 2 * n or row.get("status") != 200
                or not initial_ns <= start_ns <= finish_ns
                or finish_ns - initial_ns > 600_000_000_000):
            raise ValueError("precise HTTP sequence, clock or status is invalid")
        sequence.extend(((begin, 1, start_ns), (end, -1, finish_ns)))
    if sorted(s for s, _, _ in sequence) != list(range(1, 2 * n + 1)):
        raise ValueError("precise HTTP event sequence has duplicates or missing events")
    precise_clock_order = [ns for _, _, ns in sorted(sequence)]
    if precise_clock_order != sorted(precise_clock_order):
        raise ValueError("precise HTTP event ordering contradicts the high-resolution clock")
    active = peak = 0
    for _, delta, _ in sorted(sequence):
        active += delta
        if not 0 <= active <= 10:
            raise ValueError("physical HTTP overlap violates the global concurrency limit")
        peak = max(peak, active)
    if active:
        raise ValueError("physical HTTP events have outstanding requests")
    return {"status": "TRANSPORT_AUDIT_PASS", "request_n": n, "reserved_bytes": n * 7500,
            "max_http_overlap": peak, "coarse_interval_overlap_lower_bound": coarse_peak,
            "overlap_source": "guarded transport causal event sequence and perf_counter_ns",
            "maximum_body_bytes": maximum_body,
            "http_window_seconds": max(finishes) - min(starts),
            "last_response_after_receipt_seconds": max(finishes) - started,
            "execution_mode": receipt["execution_mode"]}

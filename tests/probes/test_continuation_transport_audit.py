"""Independent durable transport audit, including rehashed evidence attacks."""
import hashlib
import json
import pytest

from capclaims.continuation_transport_audit import build_event_chain, audit_transport_evidence


def fixture(tmp_path):
    sha = lambda b: hashlib.sha256(b).hexdigest()
    dump = lambda v: json.dumps(v, sort_keys=True, separators=(",", ":")).encode()
    rows = [{"session_id": "synthetic", "status": "complete"}]
    (tmp_path / "events.json").write_bytes(dump(rows))
    (tmp_path / "events-chain.json").write_bytes(dump(build_event_chain(rows)))
    receipt = {"preview_sha256": "a" * 64, "execution_mode": "mock", "started_monotonic": 100.0,
               "started_at_utc": "2026-10-25T10:00:00+00:00", "started_perf_ns": 100_000_000_000}
    (tmp_path / "receipt.json").write_bytes(dump(receipt))
    attempts, requests, bodies, timings = [], [], [], []
    for i in range(2):
        rid = f"synthetic-{i}"
        req, response = dump({"synthetic_input": i}), dump({"synthetic_response": i})
        requests.append({"request_id": rid, "body_utf8": req.decode(), "body_sha256": sha(req), "body_bytes": len(req)})
        bodies.append({"request_id": rid, "status": 200, "request_body_utf8": req.decode(),
                       "request_body_sha256": sha(req), "response_body_utf8": response.decode(),
                       "response_body_sha256": sha(response)})
        attempts.append({"request_id": rid, "seq": i + 1, "method": "POST", "url": "https://api.typesafe.ai/v1/systemone",
                         "reserved_bytes": 7500, "body_bytes": len(req), "state": "attempted",
                         "expected_sha256": sha(req), "actual_sha256": sha(req)})
        timings.append({"request_id": rid, "status": 200, "started_at": 100.1 + i / 10, "finished_at": 100.4 + i / 10})
    for name, values in (("attempts.jsonl", attempts), ("transport-requests.jsonl", requests), ("transport-bodies.jsonl", bodies)):
        (tmp_path / name).write_bytes(b"\n".join(dump(v) for v in values) + b"\n")
    (tmp_path / "transport.json").write_bytes(dump(timings))
    precise=[{"request_id":f"synthetic-{i}","status":200,"start_event_seq":i+1,
              "end_event_seq":i+3,"started_perf_ns":100_100_000_000+i*100_000_000,
              "finished_perf_ns":100_400_000_000+i*100_000_000} for i in range(2)]
    (tmp_path / "transport-precise.json").write_bytes(dump(precise))
    terminal = {"status": "COMPLETE", "execution_mode": "mock", "preview_sha256": "a" * 64,
                "receipt_sha256": sha((tmp_path / "receipt.json").read_bytes()),
                "events_chain_sha256": sha((tmp_path / "events-chain.json").read_bytes()),
                "precise_timings_sha256":sha((tmp_path/"transport-precise.json").read_bytes()),
                "budget": {"requests_charged": 2, "reserved_bytes": 15000, "max_requests": 840, "max_reserved_bytes": 6300000}}
    return terminal


def test_transport_audit_validates_independent_evidence_and_overlap(tmp_path):
    terminal = fixture(tmp_path)
    result = audit_transport_evidence(tmp_path, terminal, {"synthetic-0", "synthetic-1"}, "a" * 64)
    assert result["request_n"] == 2 and result["max_http_overlap"] == 2
    assert result["reserved_bytes"] == 15000


def test_precise_causal_events_disambiguate_equal_coarse_clock_values(tmp_path):
    terminal = fixture(tmp_path)
    path = tmp_path / "transport.json"
    rows = json.loads(path.read_bytes())
    for row in rows:
        row["started_at"] = row["finished_at"] = 100.0
    path.write_text(json.dumps(rows), encoding="utf-8")
    result = audit_transport_evidence(tmp_path, terminal, {"synthetic-0", "synthetic-1"}, "a" * 64)
    assert result["coarse_interval_overlap_lower_bound"] == 0
    assert result["max_http_overlap"] == 2


@pytest.mark.parametrize("attack", ["sequence", "clock"])
def test_precise_event_tamper_is_rejected_after_rehash(tmp_path, attack):
    terminal = fixture(tmp_path)
    path = tmp_path / "transport-precise.json"
    rows = json.loads(path.read_bytes())
    if attack == "sequence":
        rows[1]["start_event_seq"] = rows[0]["start_event_seq"]
    else:
        rows[1]["started_perf_ns"] = 100_450_000_000
    path.write_text(json.dumps(rows), encoding="utf-8")
    terminal["precise_timings_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="sequence|ordering"):
        audit_transport_evidence(tmp_path, terminal, {"synthetic-0", "synthetic-1"}, "a" * 64)


@pytest.mark.parametrize("attack", ["duplicate", "response_hash", "timing_before_receipt", "over_wall", "budget", "chain", "receipt"])
def test_transport_audit_rejects_tamper_even_with_outer_hash_rewritten(tmp_path, attack):
    terminal = fixture(tmp_path)
    if attack == "duplicate":
        p = tmp_path / "transport-bodies.jsonl"
        p.write_bytes(p.read_bytes() + p.read_bytes().splitlines()[0] + b"\n")
    elif attack == "response_hash":
        p = tmp_path / "transport-bodies.jsonl"
        rows = [json.loads(x) for x in p.read_bytes().splitlines()]
        rows[0]["response_body_utf8"] = "{}"
        p.write_text("\n".join(json.dumps(x) for x in rows) + "\n", encoding="utf-8")
    elif attack in {"timing_before_receipt", "over_wall"}:
        p = tmp_path / "transport.json"
        rows = json.loads(p.read_bytes())
        rows[0]["started_at" if attack == "timing_before_receipt" else "finished_at"] = 99 if attack == "timing_before_receipt" else 701
        p.write_text(json.dumps(rows), encoding="utf-8")
    elif attack == "budget":
        terminal["budget"]["requests_charged"] = 1
    elif attack == "chain":
        p = tmp_path / "events-chain.json"
        p.write_text("[]", encoding="utf-8")
        terminal["events_chain_sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
    else:
        (tmp_path / "receipt.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError):
        audit_transport_evidence(tmp_path, terminal, {"synthetic-0", "synthetic-1"}, "a" * 64)

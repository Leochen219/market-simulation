import pytest
import hashlib
import json
from pathlib import Path

from tools.probes.accept_portfolio_pilot import check_events, check_statuses
from tools.probes import accept_portfolio_pilot as gate_module


def sessions():
    return [{"target_id": f"case-{i // 2}", "arm": f"view-{i % 2}"} for i in range(12)]


def test_all_twelve_cases_are_retained_and_verified_tie_is_explicit():
    planned = sessions()
    rows = [{**row, "status": "complete"} for row in planned]
    assert check_statuses(rows, planned) == (12, 0)
    rows[3] = {**planned[3], "status": "incomplete", "reason": "unresolved_tie"}
    assert check_statuses(rows, planned) == (11, 1)
    with pytest.raises(ValueError):
        check_statuses(rows[:-1], planned)


@pytest.mark.parametrize("failure", ["pre_send", "charged_transport", "transition_failure", "recording_failure"])
def test_one_technical_failure_invalidates_whole_pilot(failure):
    planned = sessions()
    rows = [{**row, "status": "complete"} for row in planned]
    rows[2]["failure_stage"] = failure
    with pytest.raises(ValueError, match="operational failure"):
        check_statuses(rows, planned)


def test_tie_requires_actual_probability_tie_and_no_operational_stage():
    event = {"kind": "model", "event": {"kind": "failure", "reason": "unresolved_tie",
             "origin": "model", "binding": {"request_id": "unique"}},
             "reply": {"probabilities": {"a": .5, "b": .5}}}
    assert check_events([{"event": event}]) == ["unique"]
    event["reply"]["probabilities"] = {"a": .6, "b": .4}
    with pytest.raises(ValueError, match="not a tie"):
        check_events([{"event": event}])
    event["failure_stage"] = "transition_failure"
    with pytest.raises(ValueError, match="operational failure"):
        check_events([{"event": event}])


def _audit_fixture(tmp_path, monkeypatch):
    preview, run = tmp_path / "preview", tmp_path / "run"
    preview.mkdir(); run.mkdir()
    manifest = {"sdk_version": "typesafe-sdk==0.7.5", "source_kind": "human", "sessions": sessions()}
    monkeypatch.setattr(gate_module, "verify_preview", lambda *args: (manifest, {}))
    gate_file = preview / "gate.json"
    gate = {"preview_sha256": "frozen-fixture", "checker_sha256": gate_module.digest(gate_module.__file__)}
    gate_file.write_text(json.dumps(gate), encoding="utf-8")
    sha = gate_module.digest(gate_file)
    def write(name, value):
        (run / name).write_text(json.dumps(value), encoding="utf-8")
    def rows(name, values):
        (run / name).write_text("\n".join(json.dumps(value) for value in values), encoding="utf-8")
    (preview / "root-execution-receipt.json").write_text(json.dumps({
        **gate, "gate_sha256": sha, "before_provider_monotonic": 1.0}), encoding="utf-8")
    statuses = [{**row, "status": "complete"} for row in sessions()]
    write("terminal.json", {"execution_mode": "provider", "statuses": statuses})
    write("run-manifest.json", {"execution_mode": "provider"})
    write("result.json", {"execution_mode": "provider", "attempted_portfolios": 12,
        "unattempted_portfolios": 0, "actual_http_request_n": 12, "actual_http_max_overlap": 1,
        "actual_request_body_bytes": 24, "budget": {"requests_charged": 12, "reserved_bytes": 90000}})
    body_sha = hashlib.sha256(b"{}").hexdigest()
    ledger = [{"request_id": str(i), "reserved_bytes": 7500, "expected_sha256": body_sha} for i in range(12)]
    rows("attempts.jsonl", ledger)
    rows("transport-requests.jsonl", [{"request_id": str(i), "body_utf8": "{}", "body_bytes": 2} for i in range(12)])
    rows("transport-bodies.jsonl", [{"request_id": str(i), "status": 200} for i in range(12)])
    write("transport-timings.json", [{"request_id": str(i), "status": 200,
        "started_at": 2.0+i, "finished_at": 2.5+i} for i in range(12)])
    rows("trace.jsonl", [{"event": {"kind": "model", "expected_wire_sha256": body_sha,
        "event": {"kind": "choice", "origin": "model", "binding": {"request_id": str(i)}}}} for i in range(12)])
    calls = []
    def strict_replay(*args):
        calls.append(args)
        return {"audit": {"ledger_line_n": 12}}
    monkeypatch.setattr(gate_module, "replay_run", strict_replay)
    return preview, run, gate_file, sha, manifest, calls


def test_auditor_accepts_producer_sdk_label_and_invokes_strict_replay(tmp_path, monkeypatch):
    preview, run, gate_file, sha, _, calls = _audit_fixture(tmp_path, monkeypatch)
    result = gate_module.audit(preview, run, gate_file, sha)
    assert result["status"] == "ACCEPTED_UNDER_FROZEN_GATE"
    assert result["complete_portfolios"] == 12 and len(calls) == 1


def test_incompatible_preview_is_rejected_before_execution(tmp_path, monkeypatch):
    preview, _, gate_file, sha, manifest, _ = _audit_fixture(tmp_path, monkeypatch)
    manifest["sdk_version"] = "unsupported"
    attempted = []
    async def forbidden(*args, **kwargs):
        attempted.append(True)
        raise AssertionError("provider execution preceded static validation")
    monkeypatch.setattr(gate_module, "execute_preview", forbidden)
    with pytest.raises(ValueError, match="reviewed SDK"):
        gate_module.main(["--execute", "--preview-dir", str(preview), "--run-dir", str(tmp_path / "fresh-run"),
            "--gate-file", str(gate_file), "--gate-sha256", sha])
    assert attempted == []

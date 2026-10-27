import asyncio
import hashlib
import json
from pathlib import Path

import pytest

from capclaims import field_alignment_study as study
from capclaims.continuation_transport_audit import build_event_chain
from capclaims.randomness import canonical_json


def _fixture(root, monkeypatch):
    private = root / "artifacts" / "participant-data"
    private.mkdir(parents=True)
    monkeypatch.setattr(study, "ROOT", root)
    people = [f"P{i}" for i in range(1, 7)]
    bases = {person: {"agent_id": f"opaque-{person}", "round": 9, "treatment": "GA",
        "wholesale_price_dollars": 10, "p": 51 + index * 9, "Q": 0 if index == 0 else 100 + index,
        "D": index * 20, "history": [{"round": r, "p": 60 + index, "Q": 100, "D": 70,
            "orders": [{"type": "BUY", "amount": 1, "price": 60}]} for r in (6, 7, 8)]}
        for index, person in enumerate(people)}
    sessions = study._session_schedule(people)
    inputs = {"bases": bases, "sessions": sessions}
    targets = {person: {"orders": [{"type": "BUY", "amount": 1, "price": 10}]} for person in people}
    preview = private / "preview"
    preview.mkdir()
    study._write(preview / "inputs.json", inputs)
    target_bytes = (canonical_json(targets) + "\n").encode()
    (preview / "scoring-targets.json").write_bytes(target_bytes)
    manifest = {"preview_sha256": "a" * 64, "sessions": sessions, "people": people,
        "scoring_targets_sha256": hashlib.sha256(target_bytes).hexdigest(),
        "source_policy_sha256": "b" * 64, "source": {"source_kind": "human", "source_sha256": "c" * 64}}
    study._write(preview / "manifest.json", manifest)
    monkeypatch.setattr(study, "verify_preview", lambda *args: manifest)
    return preview, manifest


def test_twenty_four_case_mock_run_replays_and_scores_privately(tmp_path, monkeypatch):
    preview, manifest = _fixture(tmp_path, monkeypatch)
    run = tmp_path / "artifacts" / "participant-data" / "run"
    original_targets = (preview / "scoring-targets.json").read_bytes()
    (preview / "scoring-targets.json").unlink()
    execution = asyncio.run(study.execute_preview(preview, run, manifest["preview_sha256"], fake=True))
    assert execution["session_n"] == 24 and execution["requests_charged"] == 24
    assert execution["max_overlap"] == 10
    attempts = [json.loads(line) for line in (run / "transport-requests.jsonl").read_text().splitlines()]
    assert len({row["body_sha256"] for row in attempts[:10]}) == 10
    replay = study.replay_run(preview, run)
    assert replay["status"] == "REPLAY_VALID" and replay["session_n"] == 24

    # Rehashing edited schedule metadata cannot detach a response from its frozen case.
    events_path, chain_path, terminal_path = run / "events.json", run / "events-chain.json", run / "terminal.json"
    original_events, original_chain, original_terminal = events_path.read_bytes(), chain_path.read_bytes(), terminal_path.read_bytes()
    events = json.loads(original_events)
    events[0]["repeat"] = 1 - events[0]["repeat"]
    event_bytes = (canonical_json(events) + "\n").encode()
    chain_bytes = (canonical_json(build_event_chain(events)) + "\n").encode()
    terminal = json.loads(original_terminal)
    terminal["event_sha256"] = hashlib.sha256(event_bytes).hexdigest()
    terminal["events_chain_sha256"] = hashlib.sha256(chain_bytes).hexdigest()
    events_path.write_bytes(event_bytes); chain_path.write_bytes(chain_bytes)
    terminal_path.write_bytes((canonical_json(terminal) + "\n").encode())
    with pytest.raises(ValueError, match="exact frozen session order"):
        study.replay_run(preview, run)
    events_path.write_bytes(original_events); chain_path.write_bytes(original_chain); terminal_path.write_bytes(original_terminal)

    # A rehashed raw answer still cannot disagree with the recorded provider choice.
    response_path = run / "transport-bodies.jsonl"
    original_responses = response_path.read_bytes()
    responses = [json.loads(line) for line in original_responses.decode().splitlines()]
    response = json.loads(responses[0]["response_body_utf8"])
    old_answer = response["answers"]["decision"]
    sent_rows = [json.loads(line) for line in (run / "transport-requests.jsonl").read_text().splitlines()]
    sent = next(row for row in sent_rows if row["request_id"] == responses[0]["request_id"])
    criteria = json.loads(sent["body_utf8"])["questions"]["decision"]["criteria"]
    alternative = next(key for key in criteria if key != old_answer["choice"])
    old_answer["choice"] = alternative
    old_answer["probabilities"] = {key: float(key == alternative) for key in criteria}
    changed_body = json.dumps(response, separators=(",", ":"))
    responses[0]["response_body_utf8"] = changed_body
    responses[0]["response_body_sha256"] = hashlib.sha256(changed_body.encode()).hexdigest()
    response_bytes = ("\n".join(json.dumps(row, separators=(",", ":")) for row in responses) + "\n").encode()
    terminal = json.loads(original_terminal)
    terminal["bodies_sha256"] = hashlib.sha256(response_bytes).hexdigest()
    response_path.write_bytes(response_bytes)
    terminal_path.write_bytes((canonical_json(terminal) + "\n").encode())
    with pytest.raises(ValueError, match="literal model answer"):
        study.replay_run(preview, run)
    response_path.write_bytes(original_responses); terminal_path.write_bytes(original_terminal)

    (preview / "scoring-targets.json").write_bytes(original_targets)
    score = study.score_run(preview, run)
    assert score["status"] == "SCORED_DESCRIPTIVE_DEVELOPMENT_ONLY"
    assert score["session_n"] == 24 and score["n_people"] == 6
    serialized = json.dumps(score)
    assert "private_case_scores" not in serialized and "private_person_scores" not in serialized
    assert (run / "private-scores.json").is_file()


def test_provider_permission_check_precedes_env_load(tmp_path, monkeypatch):
    preview, manifest = _fixture(tmp_path, monkeypatch)
    run = tmp_path / "artifacts" / "participant-data" / "provider-guard"
    calls = []
    def denied(*_args):
        calls.append("policy")
        raise ValueError("denied")
    monkeypatch.setattr(study, "_check_source_policy", denied)
    monkeypatch.setattr(study, "load_local_env", lambda: calls.append("env"))
    with pytest.raises(ValueError, match="denied"):
        import asyncio
        asyncio.run(study.execute_preview(preview, run, manifest["preview_sha256"], fake=False))
    assert calls == ["policy"]
    assert (run / "receipt.json").exists()
    assert not (run / "attempts.jsonl").exists()




def test_http_failure_invalidates_entire_run_without_retry_or_next_wave(tmp_path, monkeypatch):
    import httpx2
    preview, manifest = _fixture(tmp_path, monkeypatch)
    run = tmp_path / "artifacts" / "participant-data" / "failed-run"
    observed = {"calls": 0}
    async def fail(req):
        observed["calls"] += 1
        await req.aread()
        return httpx2.Response(500, json={"error": "offline test failure"})
    with pytest.raises(RuntimeError, match="experiment invalidated"):
        asyncio.run(study.execute_preview(preview, run, manifest["preview_sha256"], fake=True,
            fake_transport=httpx2.MockTransport(fail)))
    attempts = [json.loads(line) for line in (run / "attempts.jsonl").read_text().splitlines()]
    assert observed["calls"] == len(attempts) == 10
    assert (run / "invalid.json").is_file()
    assert not (run / "terminal.json").exists()

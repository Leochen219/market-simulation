import asyncio
import hashlib
import json
import shutil
import uuid

from capclaims.portfolio_study import (ARMS, EXPECTED_SOURCE_SHA, build_synthetic_preview,
    execute_preview, main, _order_rows, prior_pilot_exclusions, replay_run, score_run,
    select_participants, verify_preview)
from capclaims.personalization import Episode


def test_participant_selection_excludes_prior_three_and_uses_only_ids():
    roster = [f"p{i}" for i in range(35)]
    excluded_people = {"p2", "p11", "p29"}
    excluded = {hashlib.sha256(p.encode()).hexdigest() for p in excluded_people}
    first = select_participants(roster, excluded)
    second = select_participants(list(reversed(roster)), excluded)
    assert first == second and len(first) == 3
    assert not set(first) & excluded_people


def test_prior_exclusions_are_exact_known_roster_hashes_and_frozen_episodes_are_supported():
    roster = [f"p{i}" for i in range(35)]
    excluded = {hashlib.sha256(p.encode()).hexdigest() for p in roster[:3]}
    assert prior_pilot_exclusions({"source_sha256": EXPECTED_SOURCE_SHA,
        "selected_participants": sorted(excluded)}, roster) == excluded
    import pytest
    with pytest.raises(ValueError, match="exactly three known"):
        prior_pilot_exclusions({"source_sha256": EXPECTED_SOURCE_SHA,
            "selected_participants": sorted(excluded | {"f" * 64})}, roster)
    episode = Episode("p0", "GA 10", "GA", 1, "orders",
        {"p": 70, "w": 10, "Q": 10, "D": 5},
        {"orders": [{"type": "BUY", "amount": 2, "price": 60}]})
    assert isinstance(episode.action["orders"], tuple)
    assert _order_rows(episode.action) == [{"type": "BUY", "amount": 2, "price": 60}]


def test_synthetic_preview_inference_replay_and_offline_score(tmp_path, monkeypatch, capsys):
    from capclaims.portfolio_study import _project_root
    root = _project_root()
    token = uuid.uuid4().hex
    preview = root / "artifacts" / "participant-data" / f"test-portfolio-preview-{token}"
    run = root / "artifacts" / "participant-data" / f"test-portfolio-run-{token}"
    try:
        plan = build_synthetic_preview(preview)
        saved, inputs = verify_preview(preview)
        assert len(saved["sessions"]) == 12
        assert saved["design_spec_text"]
        assert saved["design_spec_sha256"] == hashlib.sha256(
            saved["design_spec_text"].encode("utf-8")).hexdigest()
        assert saved["preflight_max_body_bytes"] <= 7_500
        assert set(inputs) == {"bases", "baseline_training"}
        assert not any("orders" in base for base in inputs["bases"].values())
        # Inference succeeds while the scoring file is unreadable to it.
        from pathlib import Path
        original = Path.read_bytes
        import capclaims.portfolio_study as study
        def forbidden_source_read(*args, **kwargs):
            raise AssertionError("execute CLI attempted to load the raw source workbook")
        monkeypatch.setattr(study, "load_paper_episodes", forbidden_source_read)
        def block_targets(path):
            if path.name == "scoring-targets.json":
                raise AssertionError("inference opened held-out labels")
            return original(path)
        Path.read_bytes = block_targets
        try:
            cli_status = main(["--execute", "--mock", "--preview-dir", str(preview), "--run-dir", str(run)])
            output = json.loads(capsys.readouterr().out)
            result = json.loads((run / "result.json").read_text())
        finally:
            Path.read_bytes = original
        assert cli_status == 0 and output["status"] == "COMPLETE"
        assert result["status"] == "COMPLETE"
        assert result["execution_mode"] == "mock"
        assert json.loads((run / "terminal.json").read_text())["execution_mode"] == "mock"
        assert json.loads((run / "run-manifest.json").read_text())["execution_mode"] == "mock"
        assert result["planned_portfolios"] == 12
        assert result["budget"]["requests_charged"] == 420
        body_rows = [json.loads(line) for line in (run / "transport-bodies.jsonl").read_text().splitlines()]
        assert len(body_rows) == 420
        assert max(len(row["request_body_utf8"].encode("utf-8")) for row in body_rows) <= saved["preflight_max_body_bytes"]
        assert max(len(row["request_body_utf8"].encode("utf-8")) for row in body_rows) <= 7_500
        replayed = replay_run(preview, run)
        assert replayed["audit"]["status_n"] == 12
        assert main(["--replay", "--preview-dir", str(preview), "--run-dir", str(run)]) == 0
        replay_output = capsys.readouterr().out
        assert "REPLAY_VALID" in replay_output and "orders" not in replay_output
        scored = score_run(preview, run)
        assert scored["status"] == "SCORED_OFFLINE"
        assert set(scored["arms"]) == set(ARMS)
        # Ledger and terminal tampering fail closed.
        attempts = run / "attempts.jsonl"
        old_attempts = attempts.read_bytes()
        first_line, remainder = old_attempts.splitlines(keepends=True)[0], old_attempts.splitlines(keepends=True)[1:]
        changed = json.loads(first_line)
        changed["actual_sha256"] = "0" * 64
        attempts.write_text(json.dumps(changed, sort_keys=True, separators=(",", ":")) + "\n"
            + b"".join(remainder).decode())
        import pytest
        with pytest.raises(ValueError):
            replay_run(preview, run)
        attempts.write_bytes(old_attempts)
        terminal_path = run / "terminal.json"
        old_terminal = terminal_path.read_bytes()
        terminal = json.loads(old_terminal)
        terminal["statuses"][0]["status"] = "unattempted"
        terminal_path.write_text(json.dumps(terminal, sort_keys=True, separators=(",", ":")))
        with pytest.raises(ValueError):
            replay_run(preview, run)
        terminal_path.write_bytes(old_terminal)
        terminal = json.loads(old_terminal)
        terminal["statuses"][0]["trace_head"] = "0" * 64
        terminal_path.write_text(json.dumps(terminal, sort_keys=True, separators=(",", ":")))
        with pytest.raises(ValueError):
            replay_run(preview, run)
        terminal_path.write_bytes(old_terminal)

        # A fabricated all-STOP terminal with no requests cannot claim completion.
        forged = root / "artifacts" / "participant-data" / f"test-portfolio-forged-{token}"
        forged.mkdir(parents=True, exist_ok=False)
        for name, payload in (("trace.jsonl", b""), ("attempts.jsonl", b""),
                ("transport-requests.jsonl", b""), ("transport-bodies.jsonl", b""),
                ("transport-timings.json", b"[]")):
            (forged / name).write_bytes(payload)
        forged_preview_sha = hashlib.sha256((preview / "frozen-plan.json").read_bytes()).hexdigest()
        forged_run_manifest = json.dumps({"execution_mode": "mock", "preview_sha256": forged_preview_sha,
            "status": "COMPLETE"}, sort_keys=True, separators=(",", ":")).encode()
        (forged / "run-manifest.json").write_bytes(forged_run_manifest)
        forged_statuses = [{"target_id": row["target_id"], "arm": row["arm"],
            "status": "complete", "orders": [], "call_count": 0, "trace_head": None}
            for row in plan["sessions"]]
        terminal = {"schema": "jev_portfolio_terminal_v1",
            "preview_sha256": forged_preview_sha,
            "statuses": forged_statuses, "trace_chain_head": "0" * 64, "trace_event_n": 0,
            "transport_bodies_sha256": hashlib.sha256(b"").hexdigest(), "transport_body_line_n": 0,
            "transport_requests_sha256": hashlib.sha256(b"").hexdigest(), "transport_request_line_n": 0,
            "transport_timings_sha256": hashlib.sha256(b"[]").hexdigest(), "transport_timing_n": 0,
            "attempts_sha256": hashlib.sha256(b"").hexdigest(), "ledger_line_n": 0,
            "execution_mode": "mock",
            "run_manifest_sha256": hashlib.sha256(forged_run_manifest).hexdigest(),
            "budget": {"requests_charged": 0}}
        (forged / "terminal.json").write_text(json.dumps(terminal, sort_keys=True, separators=(",", ":")))
        with pytest.raises(ValueError):
            replay_run(preview, forged)
        shutil.rmtree(forged)
    finally:
        if preview.exists():
            shutil.rmtree(preview)
        if run.exists():
            shutil.rmtree(run)


def _private_dirs(prefix):
    from capclaims.portfolio_study import _project_root
    root = _project_root()
    token = uuid.uuid4().hex
    return (root / "artifacts" / "participant-data" / f"{prefix}-preview-{token}",
        root / "artifacts" / "participant-data" / f"{prefix}-run-{token}")


def test_zero_send_guard_failure_is_replayable_and_scores_conservative_upper_bound(monkeypatch):
    import dataclasses
    import httpx2
    from capclaims.portfolio_study import GuardTransport
    preview, run = _private_dirs("test-portfolio-presend")
    try:
        build_synthetic_preview(preview)
        original_register = GuardTransport.register
        def register_wrong_wire(self, wire):
            if "__first" not in wire.request_id:
                return original_register(self, wire)
            wrong_body = b"{}"
            original_register(self, dataclasses.replace(wire, body=wrong_body,
                body_sha256=hashlib.sha256(wrong_body).hexdigest()))
        monkeypatch.setattr(GuardTransport, "register", register_wrong_wire)
        async def choose_stop(request):
            body = json.loads((await request.aread()).decode())
            criteria = body["questions"]["decision"]["criteria"]
            chosen = next(candidate for candidate in criteria if "STOP" in candidate)
            return httpx2.Response(200, json={"model": "jev-1.14.0", "usage": {},
                "answers": {"decision": {"type": "choice", "choice": chosen,
                    "confidence": 1.0, "probabilities": {key: float(key == chosen) for key in criteria}}}})
        result = asyncio.run(execute_preview(preview, run, fake=True,
            fake_transport=httpx2.MockTransport(choose_stop)))
        assert result["status"] == "PARTIAL"
        assert result["budget"]["requests_charged"] == 5
        replayed = replay_run(preview, run)
        assert replayed["audit"]["ledger_line_n"] == 5
        statuses = json.loads((run / "terminal.json").read_text())["statuses"]
        assert all(row.get("failure_stage") == "pre_send"
            for row in statuses if row["arm"] == ARMS[0] and row["status"] != "unattempted"), statuses
        assert all(row["status"] in {"complete", "unattempted"}
            for row in statuses if row["arm"] == ARMS[1])
        safe = score_run(preview, run)
        first_arm_rows = json.loads((run / "private-scores.json").read_text())["arms"][ARMS[0]]["targets"].values()
        assert all(row["score_kind"] == "conservative_upper_bound" for row in first_arm_rows)
        assert safe["execution_mode"] == "mock"
    finally:
        if preview.exists():
            shutil.rmtree(preview)
        if run.exists():
            shutil.rmtree(run)


def test_invalid_http_200_is_replayed_as_backend_reply_validation_failure():
    import httpx2
    preview, run = _private_dirs("test-portfolio-invalid-reply")
    try:
        build_synthetic_preview(preview)
        async def invalid_model(_request):
            return httpx2.Response(200, json={"model": "wrong-model", "usage": {},
                "answers": {"decision": {"type": "choice", "choice": "digit1",
                    "confidence": 1.0, "probabilities": {"digit1": 1.0}}}})
        result = asyncio.run(execute_preview(preview, run, fake=True,
            fake_transport=httpx2.MockTransport(invalid_model)))
        assert result["status"] == "PARTIAL"
        replay_run(preview, run)
        statuses = json.loads((run / "terminal.json").read_text())["statuses"]
        assert any(row.get("failure_stage") == "reply_validation" for row in statuses)
        safe = score_run(preview, run)
        private = json.loads((run / "private-scores.json").read_text())
        assert all(row["score_kind"] == "conservative_upper_bound"
            for arm in ARMS for row in private["arms"][arm]["targets"].values())
        assert safe["execution_mode"] == "mock"
    finally:
        if preview.exists():
            shutil.rmtree(preview)
        if run.exists():
            shutil.rmtree(run)


def test_charged_transport_timeout_without_response_is_explicit_and_replayable():
    import httpx2
    preview, run = _private_dirs("test-portfolio-no-response")
    try:
        build_synthetic_preview(preview)
        async def no_response(_request):
            raise httpx2.ConnectError("fixture connection failure")
        result = asyncio.run(execute_preview(preview, run, fake=True,
            fake_transport=httpx2.MockTransport(no_response)))
        assert result["status"] == "PARTIAL"
        replayed = replay_run(preview, run)
        assert replayed["audit"]["ledger_line_n"] > 0
        statuses = json.loads((run / "terminal.json").read_text())["statuses"]
        assert any(row.get("failure_stage") == "charged_transport" for row in statuses)
    finally:
        if preview.exists():
            shutil.rmtree(preview)
        if run.exists():
            shutil.rmtree(run)

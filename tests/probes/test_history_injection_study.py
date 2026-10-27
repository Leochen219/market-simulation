from pathlib import Path

from capclaims.injection_study import choose_targets, match_donors, make_schedule, build_preview, resolve_prior_exclusions
from capclaims.personalization import Episode, SourceManifest
from capclaims.injection_study import SOURCE_SHA, _ROOT
from capclaims.injection_study import InjectionBackend, create_global_budget, MAX_REQUESTS, MAX_RESERVED_BYTES
from capclaims.backends.jev import MODEL
from capclaims.portfolio_prediction import build_order_request
import asyncio
from types import SimpleNamespace
import shutil
from capclaims.injection_study import execute_preview, replay_run, score_run


def test_donor_matching_is_deterministic_unique_and_uses_only_p_and_d():
    hist = lambda p, d: [{"round": r, "p": p, "D": d, "Q": 123, "orders": []} for r in (6, 7, 8)]
    targets = [{"participant_id": f"t{i}", "history": hist(70 + i * 4, 20 + i * 5)} for i in range(6)]
    donors = [{"participant_id": f"d{i}", "history": hist(70 + i * 2, 20 + i * 3)} for i in range(23)]
    assert match_donors(targets, donors, seed=20261025) == match_donors(targets, donors, seed=20261025)
    pairs = match_donors(targets, donors, seed=20261025)
    assert len({row["donor_id"] for row in pairs}) == 6
    assert all(row["distance"] >= 0 for row in pairs)


def test_selection_excludes_exact_ids_and_deterministically_selects_six():
    roster = [f"person-{i}" for i in range(35)]
    selected = choose_targets(roster, {"person-0", "person-1", "person-2", "person-3", "person-4", "person-5"})
    assert len(selected) == 6 and not set(selected) & set(roster[:6])
    assert selected == choose_targets(roster, set(roster[:6]))


def test_prior_exclusion_manifests_resolve_exact_disjoint_ids_against_current_roster():
    roster = [f"person-{i:02d}" for i in range(35)]
    procurement = {"source_sha256": SOURCE_SHA, "selected_participants": roster[:3]}
    portfolio = {"source_sha256": SOURCE_SHA,
        "selected_participant_hashes": [__import__("hashlib").sha256(x.encode()).hexdigest() for x in roster[3:6]]}
    excluded, manifest_hashes = resolve_prior_exclusions(procurement, portfolio, roster,
        manifest_sha256={"procurement": "a" * 64, "portfolio": "b" * 64})
    assert excluded == set(roster[:6])
    assert set(manifest_hashes) == {"procurement", "portfolio"}


def test_prior_exclusion_manifests_resolve_hashed_source_ids_and_reject_overlap():
    import hashlib
    import pytest
    roster = [f"person-{i:02d}" for i in range(35)]
    hashed = lambda person: hashlib.sha256(person.encode()).hexdigest()
    procurement = {"source_sha256": SOURCE_SHA,
        "selected_participants": [hashed(x) for x in roster[:3]]}
    portfolio = {"source_sha256": SOURCE_SHA,
        "selected_participant_hashes": [hashed(x) for x in roster[3:6]]}
    excluded, _ = resolve_prior_exclusions(procurement, portfolio, roster,
        manifest_sha256={"procurement": "a" * 64, "portfolio": "b" * 64})
    assert excluded == set(roster[:6])
    overlapping = {**portfolio, "selected_participant_hashes": [hashed(x) for x in roster[2:5]]}
    with pytest.raises(ValueError, match="disjoint"):
        resolve_prior_exclusions(procurement, overlapping, roster,
            manifest_sha256={"procurement": "a" * 64, "portfolio": "b" * 64})


def test_source_order_validation_uses_historical_q_without_clipping():
    import pytest
    from capclaims.injection_study import _episode_order
    valid = {"orders": [{"type": "SELL", "amount": 10, "price": 999}]}
    assert _episode_order(valid, 10, owner="fixture") == valid["orders"]
    with pytest.raises(ValueError, match="gross sell"):
        _episode_order({"orders": [{"type": "SELL", "amount": 11, "price": 999}]},
            10, owner="fixture")
    for invalid_order in (
        {"type": "SELL", "amount": 1, "price": 1000},
        {"type": "BUY", "amount": 1000, "price": 1},
    ):
        with pytest.raises(ValueError):
            _episode_order({"orders": [invalid_order]}, 10, owner="fixture")


def test_preview_cli_prints_only_safe_summary(tmp_path, monkeypatch, capsys):
    import json
    import sys
    import tools.probes.jev_history_injection as cli
    import capclaims.personalization_xlsx as personalization_xlsx
    from capclaims.personalization import SourceManifest

    source = SourceManifest("human", "fixture", SOURCE_SHA, True, True, True)
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({"source_kind": "human", "source_sha256": SOURCE_SHA,
        "intended_use_authorized": True, "provider_transmission_permitted": True,
        "codebook_verified": True}), encoding="utf-8")
    procurement = tmp_path / "procurement.json"
    procurement.write_text("{}", encoding="utf-8")
    portfolio = tmp_path / "portfolio.json"
    portfolio.write_text("{}", encoding="utf-8")
    private_sentinel = "PRIVATE_PERSON_IDENTIFIER_SENTINEL"
    monkeypatch.setattr(personalization_xlsx, "load_paper_episodes", lambda *a, **kw: (source, []))
    monkeypatch.setattr(cli, "resolve_prior_exclusions", lambda *a, **kw: (set(), {
        "procurement": "a" * 64, "portfolio": "b" * 64}))
    monkeypatch.setattr(cli, "build_preview", lambda *a, **kw: {"preview_sha256": "c" * 64,
        "sessions": [{}] * 40, "selected_people": [private_sentinel], "preflight_max_body_bytes": 7000})
    monkeypatch.setattr(sys, "argv", ["jev_history_injection.py", "preview", "--source", "unused.zip",
        "--policy", str(policy), "--procurement-manifest", str(procurement),
        "--portfolio-manifest", str(portfolio), "--output-dir", str(tmp_path / "out")])
    assert cli.main() == 0
    output = capsys.readouterr().out
    assert private_sentinel not in output
    assert '"selected_person_n": 1' in output
    assert '"session_n": 40' in output


def test_schedule_has_40_cases_and_two_repeats_without_extra_person_weight():
    people = [f"p{i}" for i in range(6)]
    schedule = make_schedule(people, seed=20261025)
    assert len(schedule) == 40
    for person in people:
        assert sum(row["person_id"] == person for row in schedule) == (10 if person in people[:2] else 5)
    assert {row["arm"] for row in schedule} == {"A", "B", "C", "D", "E"}


def test_synthetic_preview_separates_target_labels_and_freezes_40_sessions(monkeypatch):
    roster = [f"person-{i:02d}" for i in range(35)]
    excluded = set(roster[:6])
    eligible = roster[6:]
    episodes = []
    for index, person in enumerate(eligible):
        for round_no in range(1, 10):
            orders = [] if round_no < 9 else [{"type": "SELL", "amount": 1 + index, "price": 71}]
            episodes.append(Episode(person, "GA 10", "GA", round_no, "orders",
                {"p": 70 + index % 20, "w": 10, "Q": 100, "D": 30 + index % 80},
                {"orders": orders}))
    source = SourceManifest("human", "synthetic unit fixture", SOURCE_SHA, True, True, True)
    output = _ROOT / "artifacts" / "participant-data" / "test-history-injection-preview"
    if output.exists():
        raise AssertionError("fixture path already exists")
    manifest = build_preview(source, episodes, roster, excluded, output,
        source_policy_sha256="a" * 64,
        exclusion_manifest_sha256={"procurement": "b" * 64, "portfolio": "c" * 64})
    assert len(manifest["sessions"]) == 40
    assert len(manifest["selected_people"]) == 6
    assert len(manifest["donor_matches"]) == 6
    assert (output / "labels.json").exists() and (output / "inputs.json").exists()
    inputs = (output / "inputs.json").read_text(encoding="utf-8")
    assert '"orders":[{"amount":7,"price":71,"type":"SELL"}]' not in inputs
    assert manifest["retry_count"] == 0 and manifest["request_cap"] == 1400
    run_output = _ROOT / "artifacts" / "participant-data" / "test-history-injection-run"
    if run_output.exists():
        raise AssertionError("fixture run path already exists")
    original_read_text = Path.read_text
    original_read_bytes = Path.read_bytes
    def no_label_text(self, *args, **kwargs):
        if self.name == "labels.json":
            raise AssertionError("target labels must be unreadable during inference and replay")
        return original_read_text(self, *args, **kwargs)
    def no_label_bytes(self, *args, **kwargs):
        if self.name == "labels.json":
            raise AssertionError("target labels must be unreadable during inference and replay")
        return original_read_bytes(self, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", no_label_text)
    monkeypatch.setattr(Path, "read_bytes", no_label_bytes)
    result = asyncio.run(execute_preview(output, run_output,
        reviewed_preview_sha256=manifest["preview_sha256"], fake=True))
    assert result["status"] == "COMPLETE"
    replay = replay_run(output, run_output)
    assert replay["session_n"] == 40
    # Rehash the event evidence so the replay reaches the semantic check for
    # an otherwise well-formed but unlinked provider call.
    import json
    from capclaims.injection_study import _sha, _sha_bytes, canonical_json
    import pytest
    event_path = run_output / "events.json"
    chain_path = run_output / "events-chain.json"
    terminal_path = run_output / "terminal.json"
    pristine = {path: path.read_bytes() for path in (event_path, chain_path, terminal_path)}
    events = json.loads(pristine[event_path])
    events[0]["wire_calls"].append({"request_id": "unlinked-provider-call"})
    event_path.write_text(canonical_json(events), encoding="utf-8")
    chain, previous = [], "0" * 64
    for event in events:
        digest = _sha({"previous_sha256": previous, "event": event})
        chain.append({"previous_sha256": previous, "event": event, "sha256": digest})
        previous = digest
    chain_path.write_text(canonical_json(chain), encoding="utf-8")
    terminal = json.loads(pristine[terminal_path])
    terminal.update(events_sha256=_sha_bytes(event_path.read_bytes()),
        events_chain_sha256=_sha_bytes(chain_path.read_bytes()), events_chain_head=previous)
    terminal_path.write_text(canonical_json(terminal), encoding="utf-8")
    with pytest.raises(ValueError, match="unlinked provider call"):
        replay_run(output, run_output)
    for path, raw in pristine.items():
        path.write_bytes(raw)

    # Likewise, changing a raw response while updating the outer file digest
    # must still fail the independent per-response digest check.
    bodies_path = run_output / "transport-bodies.jsonl"
    pristine_bodies = bodies_path.read_bytes()
    pristine_terminal = terminal_path.read_bytes()
    bodies = [json.loads(line) for line in bodies_path.read_text(encoding="utf-8").splitlines() if line]
    bodies[0]["response_body_utf8"] = "{}"
    bodies_path.write_text("\n".join(canonical_json(row) for row in bodies) + "\n", encoding="utf-8")
    terminal = json.loads(pristine_terminal)
    terminal["bodies_sha256"] = _sha_bytes(bodies_path.read_bytes())
    terminal_path.write_text(canonical_json(terminal), encoding="utf-8")
    with pytest.raises(ValueError, match="raw HTTP request or response"):
        replay_run(output, run_output)
    # Restore the saved original bytes captured before the second mutation.
    bodies_path.write_bytes(pristine_bodies)
    terminal_path.write_bytes(pristine_terminal)
    monkeypatch.undo()
    scored = score_run(output, run_output)
    assert scored["session_n"] == 40 and "person_collapsed_mean_loss_by_arm" in scored
    assert "complete_case_person_collapsed_mean_loss_by_arm" in scored
    assert "repeat_consistency" in scored
    assert "copies_any_personal_training_portfolio" in scored["person_collapsed_metric_means_by_arm"]
    assert "sell_curve_error_as_share_of_Q" in scored["person_collapsed_metric_means_by_arm"]
    assert "sell_window_1_100_error_as_share_of_Q" in scored["person_collapsed_metric_means_by_arm"]

    # Exercise ties through the pinned SDK, HTTP mock transport, durable run,
    # strict replay, and conservative-U scoring path.
    import httpx2
    tie_run = _ROOT / "artifacts" / "participant-data" / "test-history-injection-tie-run"
    async def tied_response(request):
        body = json.loads((await request.aread()).decode("utf-8"))
        criteria = body["questions"]["decision"]["criteria"]
        probabilities = {key: 1 / len(criteria) for key in criteria}
        return httpx2.Response(200, json={"model": MODEL,
            "usage": {"input_tokens": 1, "output_tokens": 1},
            "answers": {"decision": {"type": "choice", "choice": next(iter(criteria)),
                "confidence": 1 / len(criteria), "probabilities": probabilities}}})
    ties = asyncio.run(execute_preview(output, tie_run,
        reviewed_preview_sha256=manifest["preview_sha256"], fake=True,
        fake_transport=httpx2.MockTransport(tied_response)))
    assert ties["status"] == "COMPLETE" and ties["attempted_sessions"] == 40
    tie_audit = replay_run(output, tie_run)
    assert tie_audit["session_n"] == 40
    assert sum(row["status"] == "verified_tie" for row in tie_audit["statuses"]) == 40
    tie_scores = score_run(output, tie_run)
    assert tie_scores["verified_tie_n"] == 40 and tie_scores["complete_portfolio_n"] == 0

    # A first-wave HTTP failure invalidates the full run, records only the
    # failed wave, and cannot be promoted to a replayable/scorable result.
    failed_run = _ROOT / "artifacts" / "participant-data" / "test-history-injection-failure-run"
    http_attempts = []
    async def fail_http(request):
        http_attempts.append(request)
        return httpx2.Response(500, json={"error": "synthetic failure"})
    failed = asyncio.run(execute_preview(output, failed_run,
        reviewed_preview_sha256=manifest["preview_sha256"], fake=True,
        fake_transport=httpx2.MockTransport(fail_http)))
    assert failed["status"] == "INVALID"
    assert failed["attempted_sessions"] == 10 and failed["unattempted_sessions"] == 30
    assert failed["requests_charged"] == len(http_attempts) == 10
    with pytest.raises(ValueError):
        replay_run(output, failed_run)
    with pytest.raises(ValueError):
        score_run(output, failed_run)
    for folder in (tie_run, failed_run):
        for file in folder.iterdir():
            file.unlink()
        folder.rmdir()
    assert "person-" not in str(scored)
    for file in run_output.iterdir():
        file.unlink()
    run_output.rmdir()
    for file in output.iterdir():
        file.unlink()
    output.rmdir()


def test_injection_backend_runs_ten_requests_concurrently_with_original_envelopes():
    class Client:
        def __init__(self):
            self.calls, self.active, self.maximum = [], 0, 0
        async def system_one(self, state, questions, **kwargs):
            self.calls.append((state, questions, kwargs))
            self.active += 1
            self.maximum = max(self.maximum, self.active)
            await __import__("asyncio").sleep(0.01)
            ids = list(questions["decision"].criteria)
            return SimpleNamespace(model=MODEL,
                choices={"decision": SimpleNamespace(probabilities={key: float(i == 0) for i, key in enumerate(ids)})},
                usage=SimpleNamespace(input_tokens=1, output_tokens=1))

    async def run():
        client = Client()
        backend = InjectionBackend(api_key="synthetic", rps=10000, tps=10_000_000,
            max_requests=40, max_estimated_input_tokens=10_000_000, max_in_flight=10,
            instruction_profile="portfolio_third_v1", objective="portfolio_prediction_v1",
            client=client, arm="B", load_environment=False)
        history = [{"round": r, "p": 70, "Q": 40, "D": 20, "orders": []} for r in (6, 7, 8)]
        requests = [build_order_request({"agent_id": f"agent-{i}", "round": 9,
            "treatment": "GA", "wholesale_price_dollars": 10, "p": 80,
            "Q": 40, "D": 20, "history": history}, ()) for i in range(10)]
        replies = await backend.decide_many(requests)
        assert len(replies) == 10 and client.maximum == 10
        assert all(reply.request_id == req.request_id and reply.observation_hash == req.observation_hash
                   for reply, req in zip(replies, requests))
        assert len({tuple(call[0]["own_history"][0]) for call in client.calls}) == 1
        assert client.calls[0][0]["own_history"][0]["game_round"] == 11
        assert "agent_id" not in client.calls[0][0]["own_history"][0]

    asyncio.run(run())


def test_one_global_mock_budget_accepts_exact_1400_calls_then_rejects_more(tmp_path):
    from capclaims.portfolio_transport import ExpectedWire, _url_tuple
    async def run():
        budget = create_global_budget(tmp_path / "attempts.jsonl", fake=True)
        assert budget.max_requests == 1400 and budget.max_bytes == 10_500_000
        for index in range(MAX_REQUESTS):
            request_id = f"request-{index}"
            wire = ExpectedWire(request_id, "POST", _url_tuple("https://api.typesafe.ai/v1/systemone"),
                                b"{}", "a" * 64)
            await budget.acquire(7500)
            await budget._charge_and_persist(wire, b"{}", "POST", "https://api.typesafe.ai/v1/systemone")
        assert budget.usage["requests_charged"] == MAX_REQUESTS
        assert budget.usage["reserved_bytes"] == MAX_RESERVED_BYTES
        wire = ExpectedWire("over-limit", "POST", _url_tuple("https://api.typesafe.ai/v1/systemone"),
                            b"{}", "b" * 64)
        try:
            await budget._charge_and_persist(wire, b"{}", "POST", "https://api.typesafe.ai/v1/systemone")
        except RuntimeError as error:
            assert "budget exhausted" in str(error)
        else:
            raise AssertionError("a 1401st provider request must be rejected by the shared budget")
    asyncio.run(run())

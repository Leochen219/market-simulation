import asyncio
import json
from pathlib import Path

import httpx2
import pytest

from capclaims import prompt_sampling_study as study
from capclaims.backends.jev import MODEL
from capclaims.order_presentation import EXPLICIT, order_prompt, render_order_state
from capclaims.portfolio_prediction import build_order_request
from capclaims.prompt_sampling_analysis import summarize_cells, summarize_three_prompt_cases
from capclaims.prompt_sampling_decoder import DECODERS
from capclaims.prompt_sampling_presentation import PROMPTS, sampling_prompt, render_prompt_state


def synthetic_base(agent_id="synthetic"):
    return {"agent_id": agent_id, "round": 9, "treatment": "GA", "wholesale_price_dollars": 10,
        "p": 80, "Q": 6, "D": 30, "history": []}


def test_prompt_arms_share_identical_state_and_structured_reorders_same_rules():
    request = build_order_request(synthetic_base(), ())
    explicit_state = render_order_state(request, EXPLICIT)
    assert all(render_prompt_state(request, prompt) == explicit_state for prompt in PROMPTS)
    assert sampling_prompt(request, "explicit") == order_prompt(request, EXPLICIT)
    original = sampling_prompt(request, "original")
    structured = sampling_prompt(request, "structured")
    assert original != structured
    for phrase in ("tie allocation", "self-match handling", "visible pretrade projection",
                   "Historical examples show choices", "Gross submitted sells cannot exceed initial Q",
                   "Repeated prices", "partial or zero fills"):
        if phrase == "Repeated prices":
            phrase = "repeated prices"
        if phrase == "partial or zero fills":
            phrase = "partial or zero fills"
        assert phrase.lower() in structured.lower()
    assert "avoid duplicate" not in structured.lower()
    assert "cap buy" not in structured.lower()


def test_analysis_requires_exact_six_by_two_repeats_by_two_decoders():
    rows = []
    for person in [f"p{i}" for i in range(6)]:
        for decoder in DECODERS:
            for repeat in (0, 1):
                rows.append({"person_id": person, "prompt": "original", "decoder": decoder,
                    "repeat": repeat, "primary_curve_error": 10 + repeat,
                    "order_multiset_signature": [(decoder, repeat)]})
    result = summarize_cells(rows)
    assert result["rows_n"] == 24
    assert result["cell_means"]["original/client_argmax"] == 10.5
    with pytest.raises(ValueError, match="duplicate"):
        summarize_cells(rows[:-1] + [rows[0]])


def test_three_prompt_analysis_uses_all_72_private_cases_and_returns_aggregates_only():
    rows = []
    for pindex, prompt in enumerate(PROMPTS):
        for person_index in range(6):
            for decoder_index, decoder in enumerate(DECODERS):
                for repeat in (0, 1):
                    value = person_index * 10 + pindex * 3 + decoder_index + repeat
                    rows.append({"session_id": f"{person_index}:{decoder}:r{repeat}",
                        "person_id": f"private-person-{person_index}", "prompt": prompt,
                        "decoder": decoder, "repeat": repeat, "primary_curve_error": value,
                        "order_multiset_signature": [(decoder, repeat)]})
    result = summarize_three_prompt_cases(rows)
    assert result["forecast_n"] == 72
    assert len(result["configs"]) == 6
    assert result["configs"]["original/client_argmax"]["mean_absolute_repeat_primary_error"] == 1
    assert result["repeat_consistency_by_config"]["original/client_argmax"]["same_portfolio_rate"] == 0
    assert "person_differences" not in json.dumps(result)
    assert "private-person" not in json.dumps(result)
    with pytest.raises(ValueError, match="72"):
        summarize_three_prompt_cases(rows[:-1])


def test_ten_synthetic_sdk_requests_reach_a_real_concurrency_barrier():
    async def run():
        active = peak = 0
        bodies = set()
        barrier = asyncio.Event()

        async def handler(request):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            body_bytes = await request.aread()
            bodies.add(body_bytes)
            if active == 10:
                barrier.set()
            await asyncio.sleep(.05)
            body = json.loads(body_bytes)
            ids = list(body["questions"]["decision"]["criteria"])
            selected = ids[0]
            try:
                return httpx2.Response(200, json={"model": MODEL, "usage": {"input_tokens": 1, "output_tokens": 1},
                    "answers": {"decision": {"type": "choice", "choice": selected,
                        "confidence": 1.0, "probabilities": {key: float(key == selected) for key in ids}}}})
            finally:
                active -= 1

        from typesafe_sdk import AsyncTypeSafeClient
        from typesafe_sdk._core.retry import RetryPolicy
        from capclaims.prompt_sampling_study import _SamplingBackend
        client_http = httpx2.AsyncClient(transport=httpx2.MockTransport(handler), follow_redirects=False, timeout=3)
        client = AsyncTypeSafeClient(api_key="offline-synthetic-key", model=MODEL,
            retry=RetryPolicy(max_retries=0), timeout=3, http_client=client_http,
            base_url="https://api.typesafe.ai", compression=False)
        backend = _SamplingBackend("original", api_key="offline-synthetic-key", rps=100000,
            tps=100000000, max_requests=10, max_estimated_input_tokens=1_000_000,
            instruction_profile="portfolio_third_v1", objective="portfolio_prediction_v1",
            client=client, request_timeout=3, max_in_flight=10, load_environment=False)
        requests = [build_order_request({**synthetic_base(f"synthetic-{i}"), "p": 51 + i,
            "Q": i + 1, "D": 20 + i}, ()) for i in range(10)]
        try:
            replies = await asyncio.wait_for(backend.decide_many(requests), timeout=5)
            assert len(replies) == 10 and barrier.is_set()
        finally:
            backend._closed = True
            await client.aclose()
        return peak, bodies

    peak, bodies = asyncio.run(run())
    assert peak == 10
    assert len(bodies) == 10


def test_live_dispatch_wrapper_enforces_rolling_reserved_byte_rate_after_slow_durable_prepare(monkeypatch, tmp_path):
    async def run():
        from capclaims.backends.jev import _CURRENT_REQUEST_ID
        from capclaims.continuation_transport_audit import HttpOverlapTransport
        inner = httpx2.MockTransport(lambda request: httpx2.Response(200, content=b"ok"))
        wrapped = study._DurablePhysicalTransport(HttpOverlapTransport(inner), tmp_path / "physical.jsonl",
            raw_path=tmp_path / "raw.jsonl", event_path=tmp_path / "events.jsonl", mock=False)
        append = study._append_jsonl
        def slow_event_append(path, value):
            if Path(path) == tmp_path / "events.jsonl" and value.get("kind") == "dispatch_start":
                import time
                time.sleep(.2)
            append(path, value)
        monkeypatch.setattr(study, "_append_jsonl", slow_event_append)
        client = httpx2.AsyncClient(transport=wrapped, timeout=4)
        async def submit(index):
            token = _CURRENT_REQUEST_ID.set(f"synthetic-rate-{index}")
            try:
                response = await client.post("https://api.typesafe.ai/synthetic", json={"index": index})
                assert response.status_code == 200
            finally:
                _CURRENT_REQUEST_ID.reset(token)
        try:
            await asyncio.gather(*(submit(i) for i in range(7)))
        finally:
            await client.aclose()
        return wrapped.rows, wrapped.inner.rows
    rows, precise = asyncio.run(run())
    assert len(rows) == len(precise) == 7
    dispatches = sorted(row["started_perf_ns"] for row in precise)
    assert dispatches[6] - dispatches[0] > 1_000_000_000
    for index in range(6, len(dispatches)):
        assert dispatches[index] - dispatches[index - 6] > 1_000_000_000
    assert 6 * study.MAX_BODY <= study.MAX_BYTES_PER_SECOND


def test_source_policy_check_precedes_local_environment_loading(monkeypatch, tmp_path):
    root = tmp_path / "study-root"
    private = root / "artifacts" / "participant-data"
    preview = private / "synthetic-preview"
    preview.mkdir(parents=True)
    (preview / "inputs.json").write_text(json.dumps({"bases": {}, "sessions": []}))
    manifest = {"preview_sha256": "f" * 64, "source_policy_sha256": "a" * 64,
        "source": {"source_sha256": "b" * 64}}
    (preview / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(study, "ROOT", root)
    monkeypatch.setattr(study, "verify_preview", lambda *_: manifest)
    def deny(*_):
        raise PermissionError("policy denied")
    monkeypatch.setattr("capclaims.order_explanation_study._check_source_policy", deny)
    called = []
    monkeypatch.setattr("capclaims.backends.jev.load_local_env", lambda: called.append("env"))
    with pytest.raises(RuntimeError, match="invalidated"):
        asyncio.run(study.execute_preview(preview, private / "run-policy", "original", "f" * 64, mock=False))
    assert not called
    invalid = json.loads((private / "run-policy" / "terminal.json").read_text())
    assert invalid["status"] == "INVALID" and invalid["error_type"] == "PermissionError"

    monkeypatch.setattr("capclaims.order_explanation_study._check_source_policy", lambda *_: None)
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="invalidated"):
        asyncio.run(study.execute_preview(preview, private / "run-key", "original", "f" * 64, mock=False))
    invalid_key_run = json.loads((private / "run-key" / "terminal.json").read_text())
    assert invalid_key_run["status"] == "INVALID" and invalid_key_run["error_type"] == "ValueError"


def test_cli_preview_stdout_uses_allowlisted_summary(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(study, "build_preview", lambda *_: {"preview_sha256": "c" * 64,
        "session_n_per_prompt": 24, "people": ["private-one", "private-two"],
        "prompts": list(PROMPTS), "decoders": list(DECODERS)})
    from tools.probes.jev_prompt_sampling import main
    assert main(["preview", "parent", str(tmp_path / "preview")]) == 0
    stdout = capsys.readouterr().out
    assert "private-one" not in stdout
    assert "private-two" not in stdout
    assert '"people_n": 2' in stdout


def test_compare_three_runs_aggregates_private_cases_from_all_runs(monkeypatch, tmp_path):
    preview = tmp_path / "preview"
    preview.mkdir()
    monkeypatch.setattr(study, "_private", lambda path: Path(path))
    monkeypatch.setattr(study, "score_run", lambda *_: {"preview_sha256": "a" * 64})
    runs = {}
    for prompt_index, prompt in enumerate(PROMPTS):
        run = tmp_path / prompt
        run.mkdir()
        runs[prompt] = run
        cases = []
        for person_index in range(6):
            for decoder_index, decoder in enumerate(DECODERS):
                for repeat in (0, 1):
                    cases.append({"session_id": f"{prompt}/{person_index}/{decoder}/{repeat}",
                        "person_id": f"private-{person_index}", "prompt": prompt, "decoder": decoder,
                        "repeat": repeat, "primary_curve_error": person_index + prompt_index + repeat,
                        "order_multiset_signature": [decoder, repeat]})
        (run / "private-scores.json").write_text(json.dumps({"status": "SCORED_OFFLINE", "prompt": prompt,
            "preview_sha256": "a" * 64, "cases": cases}))
    result = study.compare_three_runs(preview, runs)
    assert result["forecast_n"] == 72
    assert set(result["configs"]) == {f"{p}/{d}" for p in PROMPTS for d in DECODERS}
    assert "person_differences" not in json.dumps(result)
    assert "private-" not in json.dumps(result)


def test_synthetic_preview_execute_replay_score_hides_labels_until_score(monkeypatch, tmp_path):
    import hashlib
    root = tmp_path / "study-root"
    private = root / "artifacts" / "participant-data"
    parent = private / "synthetic-parent"
    parent.mkdir(parents=True)
    people = [f"synthetic-person-{i}" for i in range(6)]
    bases = {person: {"agent_id": person, "round": 9, "treatment": "GA",
        "wholesale_price_dollars": 10, "p": 60 + i, "Q": 6, "D": 20 + i,
        "history": [{"round": hround, "p": 60 + i, "Q": 6, "D": 20 + i, "orders": []}
            for hround in (6, 7, 8)]} for i, person in enumerate(people)}
    labels = {person: {"orders": []} for person in people}
    input_bytes = json.dumps({"bases": bases, "sessions": []}).encode()
    label_bytes = json.dumps(labels).encode()
    (parent / "inputs.json").write_bytes(input_bytes)
    (parent / "scoring-targets.json").write_bytes(label_bytes)
    (parent / "manifest.json").write_text("{}")
    source_policy = private / "synthetic-policy.json"
    source_policy.write_text("{}")
    source_policy_raw = source_policy.read_bytes()
    pmanifest = {"source_policy_sha256": hashlib.sha256(source_policy_raw).hexdigest(),
        "source": {"source_sha256": "synthetic-source"},
        "scoring_targets_sha256": hashlib.sha256(label_bytes).hexdigest()}
    monkeypatch.setattr(study, "ROOT", root)
    monkeypatch.setattr(study, "PARENT_RELATIVE", Path("artifacts/participant-data/synthetic-parent"))
    monkeypatch.setattr(study, "SOURCE_POLICY_RELATIVE", Path("artifacts/participant-data/synthetic-policy.json"))
    monkeypatch.setattr(study, "_code_hashes", lambda: {"synthetic-runtime": "0" * 64})
    monkeypatch.setattr("capclaims.order_explanation_study.verify_preview", lambda *_: pmanifest)
    monkeypatch.setattr("capclaims.order_explanation_study._check_source_policy", lambda *_: None)
    preview = private / "synthetic-preview"
    frozen = study.build_preview(parent, preview)

    original_read_bytes = Path.read_bytes
    labels_allowed = False
    def guarded_read_bytes(path):
        if path.name == "scoring-targets.json" and not labels_allowed:
            raise AssertionError("inference or replay opened score labels")
        return original_read_bytes(path)
    monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)
    assert study.verify_preview(preview, frozen["preview_sha256"])["preview_sha256"] == frozen["preview_sha256"]
    run = private / "synthetic-run-original"
    result = asyncio.run(study.execute_preview(preview, run, "original", frozen["preview_sha256"], mock=True))
    assert result["status"] == "COMPLETE" and result["attempted_sessions"] == 24
    replayed = study.replay_run(preview, run, "original")
    assert replayed["status"] == "REPLAY_VALID" and replayed["session_n"] == 24
    labels_allowed = True
    scored = study.score_run(preview, run, "original")
    assert scored["status"] == "SCORED_OFFLINE"
    assert scored["configs"]["client_argmax"]["forecast_n"] == 12
    assert study.score_run(preview, run, "original") == scored

    all_runs = {"original": run}
    for prompt_name in ("explicit", "structured"):
        other_run = private / f"synthetic-run-{prompt_name}"
        result = asyncio.run(study.execute_preview(preview, other_run, prompt_name,
            frozen["preview_sha256"], mock=True))
        assert result["status"] == "COMPLETE"
        assert study.replay_run(preview, other_run, prompt_name)["status"] == "REPLAY_VALID"
        all_runs[prompt_name] = other_run
    for prompt_name, other_run in all_runs.items():
        study.score_run(preview, other_run, prompt_name)
    comparison = study.compare_three_runs(preview, all_runs)
    assert comparison["forecast_n"] == 72
    assert len(comparison["configs"]) == 6
    assert "person_differences" not in json.dumps(comparison)

    failure_run = private / "synthetic-run-failed"
    async def bad_response(_request):
        return httpx2.Response(503, json={"error": "synthetic outage"})
    with pytest.raises(RuntimeError, match="invalidated"):
        asyncio.run(study.execute_preview(preview, failure_run, "explicit", frozen["preview_sha256"],
            mock=True, fake_transport=httpx2.MockTransport(bad_response)))
    assert (failure_run / "invalid.json").is_file()
    failure_events = [json.loads(line) for line in (failure_run / "session-events.jsonl").read_text().splitlines()]
    assert any(row["event"].get("kind") == "failure" for row in failure_events)
    assert (failure_run / "transport-physical.jsonl").stat().st_size > 0
    raw_failures = [json.loads(line) for line in (failure_run / "transport-raw.jsonl").read_text().splitlines()]
    assert raw_failures and all(row["status"] == 503 for row in raw_failures)
    import base64
    assert base64.b64decode(raw_failures[0]["response_body_base64"]) == b'{"error":"synthetic outage"}'
    assert any(json.loads(line)["status"] == "failed"
        for line in (failure_run / "session-status.jsonl").read_text().splitlines())
    invalid_terminal = json.loads((failure_run / "terminal.json").read_text())
    assert invalid_terminal["status"] == "INVALID"

    import shutil
    tampered = private / "synthetic-run-clock-tampered"
    shutil.copytree(run, tampered)
    timing_path = tampered / "transport.json"
    timing_rows = json.loads(timing_path.read_text())
    timing_rows[0]["started_at"] -= 1
    timing_path.write_text(json.dumps(timing_rows, separators=(",", ":")) + "\n")
    terminal_path = tampered / "terminal.json"
    terminal = json.loads(terminal_path.read_text())
    terminal["timings_sha256"] = hashlib.sha256(timing_path.read_bytes()).hexdigest()
    terminal_path.write_text(json.dumps(terminal, separators=(",", ":")) + "\n")
    with pytest.raises(ValueError, match="receipt|clock|transport"):
        study.replay_run(preview, tampered, "original")

import asyncio
import math
from types import SimpleNamespace

import pytest

from capclaims.backends.jev import JevBackend, RecordingTransport, load_local_env, normalize_jev_probabilities
from capclaims.backends.jev_instructions import build_instruction
from capclaims.domain.types import Candidate
from capclaims.engine.protocol import DecisionRequest


def req(agent="agent_00", stage="paper:purchase:h", observation=None, values=(0, 1, 2)):
    return DecisionRequest(agent, 0, stage, observation or {
        "decision_interface": "economic_v2", "decision_field": "purchase",
        "numeric_prefix": "", "choice_minimum": 0, "choice_maximum": 400,
        "own_p": 80, "own_q_units": 20, "own_d_units": None,
        "wholesale_price_dollars": 10,
    }, tuple(Candidate(f"digit{x}", x, f"{x*100}..{x*100+99} units") for x in values))


class Client:
    def __init__(self, *, latency=0, wrong_model=None, bad_scores=None):
        self.latency, self.wrong_model, self.bad_scores = latency, wrong_model, bad_scores
        self.active = 0
        self.maximum = 0
        self.calls = []
    async def system_one(self, state, questions, **kwargs):
        self.active += 1
        self.maximum = max(self.maximum, self.active)
        self.calls.append((state, questions, kwargs))
        try:
            await asyncio.sleep(self.latency)
            ids = list(questions["decision"].criteria)
            probs = self.bad_scores or {k: (0.8 if i == 0 else 0.1) for i, k in enumerate(ids)}
            return SimpleNamespace(model=self.wrong_model or "jev-1.14.0",
                choices={"decision": SimpleNamespace(probabilities=probs)},
                usage=SimpleNamespace(input_tokens=12, output_tokens=2))
        finally:
            self.active -= 1
    async def close(self): pass


def backend(client, **kwargs):
    return JevBackend(api_key="fake", rps=kwargs.pop("rps", 10000), tps=kwargs.pop("tps", 1_000_000_000),
        max_requests=100, max_estimated_input_tokens=100000, client=client, **kwargs)


def test_explicit_instruction_explains_role_economics_and_uncertainty_without_oracle():
    text = build_instruction(req(), "explicit_v1")
    for phrase in ("retailer", "wholesale", "demand", "profit", "fulfilled", "uncertain", "expire"):
        assert phrase in text.lower()
    assert "333" not in text and "optimal" not in text.lower()


def test_matched_profile_uses_existing_paper_instruction():
    from capclaims.paper.interface import instruction
    assert build_instruction(req(), "matched_v2") == instruction("paper:purchase:h", "economic_v2")


def test_missing_credentials_and_rate_or_budget_bounds_fail_closed(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.setattr("capclaims.backends.jev.load_local_env", lambda: {})
    with pytest.raises(ValueError, match="TYPESAFE_API_KEY"):
        JevBackend(rps=1, tps=100, max_requests=1, max_estimated_input_tokens=100)
    limited = JevBackend(api_key="fake", tps=1, max_requests=1, max_estimated_input_tokens=100, client=Client())
    assert limited.rps == 40 and limited.max_requests == 1


@pytest.mark.parametrize("limit", [True, 1.5, 0, 11, "2"])
def test_max_in_flight_must_be_an_integer_from_one_through_ten(limit):
    with pytest.raises(ValueError, match="max_in_flight"):
        JevBackend(api_key="fake", rps=1, tps=1, max_requests=1,
            max_estimated_input_tokens=100, max_in_flight=limit, client=Client())


def test_ten_private_calls_overlap_and_keep_state_isolated():
    async def run():
        client = Client(latency=0.03)
        be = backend(client)
        requests = [req(f"agent_{i:02}", observation={"private_agent": i}) for i in range(10)]
        replies = await be.decide_many(requests)
        assert len(replies) == 10
        assert client.maximum == 10
        assert len({id(call[0]) for call in client.calls}) == 10
        assert be.budget_usage["requests_reserved"] == 10
        assert all(r.backend == "jev" and r.model_revision == "jev-1.14.0" for r in replies)
        assert all(r.usage["actual_input_tokens"] == 12 for r in replies)
        assert all("instruction_text" in r.timing and "instruction_sha256" in r.timing for r in replies)
        await be.aclose()
    asyncio.run(run())


def test_budget_preflight_is_atomic_and_never_calls_provider():
    async def run():
        c = Client()
        be = JevBackend(api_key="fake", rps=10, tps=10000, max_requests=1,
            max_estimated_input_tokens=100000, client=c)
        with pytest.raises(ValueError, match="request budget"):
            await be.decide_many([req("agent_00"), req("agent_01")])
        assert not c.calls
        assert be.budget_usage["requests_reserved"] == 0
    asyncio.run(run())


@pytest.mark.parametrize("model,scores", [("jev-other", None), (None, {"digit0": 1.0})])
def test_rejects_wrong_model_and_incomplete_probability_map(model, scores):
    async def run():
        be = backend(Client(wrong_model=model, bad_scores=scores))
        with pytest.raises(ValueError):
            await be.decide_many([req()])
    asyncio.run(run())


def test_rejects_nonfinite_probabilities():
    async def run():
        be = backend(Client(bad_scores={"digit0": math.nan, "digit1": 0.0, "digit2": 0.0}))
        with pytest.raises(ValueError, match="probabilit"):
            await be.decide_many([req()])
    asyncio.run(run())


def test_jev_two_decimal_provider_rounding_normalizes_small_residual_and_preserves_raw_scores():
    ids = [f"c{i}" for i in range(10)]
    raw = {key: value for key, value in zip(ids, (0, 0, 0, 0, .01, .03, .93, .01, 0, .01))}
    normalized, metadata = normalize_jev_probabilities(raw, ids)
    assert math.isclose(math.fsum(normalized.values()), 1.0, abs_tol=1e-12)
    assert max(normalized, key=normalized.get) == "c6"
    assert raw["c6"] == .93 and math.isclose(math.fsum(raw.values()), .99)
    assert metadata["rounding_normalized"] is True
    assert metadata["source_precision_decimals"] == 2
    assert math.isclose(metadata["raw_probability_total"], .99)


@pytest.mark.parametrize("raw", [
    {"a": 0.0, "b": 0.0},
    {"a": 0.2, "b": 0.2},
    {"a": 1.01, "b": 0.0, "c": 0.0},
    {"a": 0.97, "b": 0.01},
    {"a": 1.0001, "b": 0.0},
    {"a": -0.01, "b": 1.0},
    {"a": math.nan, "b": 0.5},
])
def test_jev_probability_normalizer_rejects_invalid_totals_or_values(raw):
    with pytest.raises(ValueError):
        normalize_jev_probabilities(raw, tuple(raw))


def test_reply_keeps_raw_probability_scores_and_records_normalization_and_usage():
    async def run():
        raw = {f"digit{i}": value for i, value in enumerate((0, 0, 0, 0, .01, .03, .93, .01, 0, .01))}
        request = req(values=tuple(range(10)))
        be = backend(Client(bad_scores=raw))
        (reply,) = await be.decide_many([request])
        assert dict(reply.raw_scores) == raw
        assert math.isclose(math.fsum(reply.probabilities.values()), 1.0, abs_tol=1e-12)
        assert reply.probabilities["digit6"] > reply.probabilities["digit5"]
        assert reply.usage["rounding_normalized"] is True
        assert reply.usage["source_precision_decimals"] == 2
        assert math.isclose(reply.usage["raw_probability_total"], .99)
        assert be.window_timings[0]["actual_input_tokens"] == 12
        await be.aclose()
    asyncio.run(run())


def test_billed_usage_is_counted_when_response_probabilities_are_invalid():
    async def run():
        raw = {f"digit{i}": 0.0 for i in range(10)}
        be = backend(Client(bad_scores=raw))
        with pytest.raises(ValueError, match="positive"):
            await be.decide_many([req(values=tuple(range(10)))])
        assert be.budget_usage["actual_input_tokens_reported"] == 12
        assert be.budget_usage["actual_output_tokens_reported"] == 2
        assert be.window_timings[0]["actual_input_tokens"] == 12
        assert be.budget_usage["requests_completed"] == 0
        await be.aclose()
    asyncio.run(run())


def test_timeout_and_batch_failure_cancel_and_await_siblings():
    async def run():
        c = Client(latency=0.05)
        be = backend(c, request_timeout=0.001)
        with pytest.raises((TimeoutError, asyncio.TimeoutError)):
            await be.decide_many([req(f"agent_{i:02}") for i in range(3)])
        assert c.active == 0
    asyncio.run(run())


def test_local_env_does_not_override_parent_env(monkeypatch):
    from pathlib import Path
    envfile = Path(".test-jev-env")
    try:
        envfile.write_text("TYPESAFE_API_KEY=from-file\nJEV_RPS=3\n", encoding="utf-8")
        monkeypatch.setenv("TYPESAFE_API_KEY", "inherited")
        monkeypatch.setenv("JEV_RPS", "44")
        loaded = load_local_env(envfile)
        assert loaded["TYPESAFE_API_KEY"] == "inherited"
        assert loaded["JEV_RPS"] == "44"
    finally:
        envfile.unlink(missing_ok=True)


def test_installed_sdk_serializes_private_state_choice_prompt_and_probabilities():
    async def run():
        import httpx2
        captured = []
        def handler(request):
            payload = __import__("json").loads(request.content)
            captured.append(payload)
            criteria = payload["questions"]["decision"]["criteria"]
            keys = list(criteria)
            return httpx2.Response(200, json={"model": "jev-1.14.0",
                "usage": {"input_tokens": 27, "output_tokens": 1},
                "answers": {"decision": {"type": "choice", "choice": keys[0],
                    "confidence": 0.8, "probabilities": {key: (0.8 if i == 0 else 0.1)
                        for i, key in enumerate(keys)}}}})
        from typesafe_sdk import AsyncTypeSafeClient
        recorder = RecordingTransport(httpx2.MockTransport(handler))
        sdk = AsyncTypeSafeClient(api_key="unit-test-secret", model="jev-1.14.0", transport=recorder)
        be = backend(sdk, transport_recorder=recorder)
        request = req(observation={"decision_interface": "economic_v2", "decision_field": "purchase",
            "private_agent": "agent-unique", "choice_minimum": 0, "choice_maximum": 400,
            "numeric_prefix": "", "own_p": 72})
        replies = await be.decide_many([request])
        assert len(replies) == 1 and replies[0].usage["actual_input_tokens"] == 27
        assert len(captured) == 1
        wire = captured[0]
        assert wire["model"] == "jev-1.14.0"
        assert wire["state"]["private_agent"] == "agent-unique"
        assert set(wire["questions"]["decision"]["criteria"]) == {c.candidate_id for c in request.candidates}
        assert "retailer" in wire["questions"]["decision"]["instructions"].lower()
        assert be.transport_timings[0]["request_id"] == request.request_id
        assert "state" not in be.transport_timings[0] and "headers" not in be.transport_timings[0]
        await be.aclose()
    asyncio.run(run())


def test_http_transport_proves_ten_independent_requests_overlap_without_payload_capture():
    async def run():
        import httpx2
        import json
        from typesafe_sdk import AsyncTypeSafeClient
        seen_states = []
        async def handler(request):
            await asyncio.sleep(0.03)
            wire = json.loads(request.content)
            seen_states.append(wire["state"])
            ids = list(wire["questions"]["decision"]["criteria"])
            probs = {key: (1.0 if i == 0 else 0.0) for i, key in enumerate(ids)}
            return httpx2.Response(200, json={"model": "jev-1.14.0",
                "usage": {"input_tokens": 8, "output_tokens": 1},
                "answers": {"decision": {"type": "choice", "choice": ids[0],
                    "confidence": 1.0, "probabilities": probs}}})
        recorder = RecordingTransport(httpx2.MockTransport(handler))
        sdk = AsyncTypeSafeClient(api_key="unit-test-secret", model="jev-1.14.0", transport=recorder)
        be = backend(sdk, transport_recorder=recorder)
        requests = [req(f"agent_{i:02}", observation={"private_agent": i}) for i in range(10)]
        await be.decide_many(requests)
        windows = be.transport_timings
        assert len(windows) == 10
        assert {row["request_id"] for row in windows} == {r.request_id for r in requests}
        assert len({row["private_agent"] for row in seen_states}) == 10
        assert all(not ({"state", "headers", "body"} & set(row)) for row in windows)
        assert max(sum(a["started_at"] <= t <= a["finished_at"] for a in windows)
            for t in (row["started_at"] for row in windows)) > 1
        await be.aclose()
    asyncio.run(run())


def test_environment_limits_override_safe_defaults_and_failed_calls_are_redacted(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "unit-test")
    monkeypatch.setenv("JEV_RPS", "11")
    monkeypatch.setenv("JEV_TPS", "2222")
    monkeypatch.setenv("JEV_MAX_REQUESTS", "3")
    monkeypatch.setenv("JEV_MAX_ESTIMATED_INPUT_TOKENS", "4444")
    be = JevBackend(client=Client())
    assert (be.rps, be.tps, be.max_requests, be.max_estimated_input_tokens) == (11, 2222, 3, 4444)

    class LeakyClient(Client):
        async def system_one(self, *args, **kwargs):
            raise RuntimeError("sensitive provider response text")
    async def run():
        failing = backend(LeakyClient())
        with pytest.raises(RuntimeError) as error:
            await failing.decide_many([req()])
        assert "sensitive provider response text" not in str(error.value)
        assert failing.window_timings[0]["status"] == "provider_error"
        assert failing.budget_usage["requests_started"] == 1
        await failing.aclose()
    asyncio.run(run())


def test_jev_sdk_mock_transport_completes_ga_buy_sell_round_and_replay(tmp_path):
    async def run():
        import httpx2
        import json
        from typesafe_sdk import AsyncTypeSafeClient
        from capclaims.paper.config import PaperConfig
        from capclaims.paper.session import run_paper_session
        from capclaims.paper.replay import replay_paper

        def handler(wire_request):
            payload = json.loads(wire_request.content)
            state = payload["state"]
            obs = state if isinstance(state, dict) else json.loads(state)
            criteria = payload["questions"]["decision"]["criteria"]
            field = obs.get("decision_field")
            if field == "side":
                if obs.get("slot", 0) > 0:
                    wanted = "action-STOP"
                else:
                    wanted = "action-buy" if obs["own_d_units"] > obs["own_q_units"] else "action-sell"
            elif field == "purchase":
                target = 100
                wanted = f"digit{str(target).zfill(3)[len(obs.get('numeric_prefix', ''))]}"
            elif field == "quantity":
                target = 40
                wanted = f"digit{str(target).zfill(3)[len(obs.get('numeric_prefix', ''))]}"
            elif field == "price":
                target = 999 if obs.get("own_side") == "buy" else 1
                wanted = f"digit{str(target).zfill(3)[len(obs.get('numeric_prefix', ''))]}"
            else:
                raise AssertionError(f"unexpected decision field {field!r}")
            assert wanted in criteria
            probs = {key: (1.0 if key == wanted else 0.0) for key in criteria}
            return httpx2.Response(200, json={"model": "jev-1.14.0",
                "usage": {"input_tokens": 100, "output_tokens": 1},
                "answers": {"decision": {"type": "choice", "choice": wanted,
                    "confidence": 1.0, "probabilities": probs}}})

        output = tmp_path / "jev-ga-run"
        config = PaperConfig(treatment="GA", n_agents=10, warmup_rounds=0,
            analyzed_rounds=1, policy="jev", decision_interface="economic_v2")
        sdk = AsyncTypeSafeClient(api_key="unit-test-secret", model="jev-1.14.0",
            transport=httpx2.MockTransport(handler))
        be = JevBackend(api_key="unit-test-secret", rps=10000, tps=1_000_000_000,
            max_requests=500, max_estimated_input_tokens=1_000_000,
            instruction_profile="explicit_v1", client=sdk)
        try:
            result = await run_paper_session(config, be, output)
            assert result.status == "COMPLETE" and result.rounds == 1
            replayed = replay_paper(output)
            assert replayed.final_hash == result.final_hash
            trades = [trade for market in result.events[0]["markets"] for trade in market["trades"]]
            assert trades and sum(trade["qty"] for trade in trades) > 0
            sides = {order["side"] for order in result.events[0]["orders"]}
            assert sides == {"buy", "sell"}
        finally:
            await be.aclose()
    asyncio.run(run())


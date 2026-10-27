"""Integration contracts use actual economic_v2 requests, never invented shapes."""
import asyncio

from capclaims.engine.protocol import DecisionReply
from capclaims.injection_context import render_state
from capclaims.portfolio_prediction import advance_trace, build_order_request, next_trace_action
from capclaims.portfolio_transport import serialize_sdk_wire


def base():
    return {"agent_id": "synthetic", "round": 9, "treatment": "GA",
            "wholesale_price_dollars": 10, "p": 80, "Q": 100, "D": 40,
            "history": [{"round": r, "p": 70, "Q": 90, "D": 60,
                         "orders": [{"type": "BUY", "amount": 7, "price": 12}]}
                        for r in (6, 7, 8)]}


def add_values(facts, trace, values):
    for value in values:
        action = next_trace_action(facts, trace)
        if action["kind"] == "forced":
            assert action["candidate"].value == value
            trace = advance_trace(facts, trace, None)
        else:
            request = build_order_request(facts, trace)
            candidate = next(c for c in request.candidates if c.value == value)
            probs = {c.candidate_id: float(c == candidate) for c in request.candidates}
            trace = advance_trace(facts, trace, DecisionReply.from_request(request, probs))
    return trace


def test_named_history_replaces_actual_compact_rows_and_preserves_facts():
    request = build_order_request(base(), ())
    original = request.observation
    assert isinstance(original["own_history"][0], list)
    named = render_state(request, "B")
    row = named["own_history"][0]
    assert isinstance(row, dict)
    assert row["game_round"] == 11
    assert row["retail_price_dollars_per_unit"] == 70
    assert row["initial_inventory_units"] == 90
    assert row["realized_demand_units"] == 60
    assert row["submitted_orders"][0]["quantity_units"] == 7
    assert row["submitted_orders"][0]["limit_price_dollars_per_unit"] == 12
    assert "own_history_fields" not in named
    assert isinstance(request.observation["own_history"][0], list)


def test_no_history_arm_does_not_retain_actual_own_history_elsewhere():
    request = build_order_request(base(), ())
    state = render_state(request, "E")
    assert state["own_history"] == []
    assert "history" not in state
    assert "own_history_fields" not in state


def test_ledger_uses_actual_arrays_and_does_not_count_pending_order():
    facts = base()
    trace = add_values(facts, (), ["buy", 0, 0, 7, 0, 1, 2])
    trace = add_values(facts, trace, ["sell", 0, 2, 0, 0, 7, 7])
    trace = add_values(facts, trace, ["buy", 0, 1, 1])
    request = build_order_request(facts, trace)
    assert request.observation["decision_field"] == "price"
    assert request.observation["own_qty_units"] == 11
    ledger = render_state(request, "C")["submitted_order_ledger"]
    assert ledger == {"submitted_order_count": 2, "remaining_order_slots": 3,
                      "submitted_buy_quantity_units": 7, "submitted_sell_quantity_units": 20,
                      "remaining_sell_submission_capacity_units": 80}


def test_a_sdk_wire_is_exact_old_third_person_payload():
    request = build_order_request(base(), ())
    async def compare():
        old = await serialize_sdk_wire(request, "portfolio_third_v1")
        new = await serialize_sdk_wire(request, "portfolio_third_v1", render_state(request, "A"))
        assert old.body == new.body and old.url == new.url
    asyncio.run(compare())


def test_d_keeps_donor_state_and_orders_together_and_current_state_unchanged():
    request = build_order_request(base(), ())
    donor = [{"round": r, "p": 55, "Q": 10, "D": 20,
              "orders": [{"type": "SELL", "amount": 10, "price": 99}]}
             for r in (6, 7, 8)]
    state = render_state(request, "D", donor_history=donor)
    assert state["own_q_units"] == 100 and state["own_d_units"] == 40
    assert [r["game_round"] for r in state["own_history"]] == [11, 12, 13]
    assert all(r["initial_inventory_units"] == 10 for r in state["own_history"])
    assert all(r["submitted_orders"][0]["quantity_units"] == 10 for r in state["own_history"])
    assert state["submitted_order_ledger"]["remaining_sell_submission_capacity_units"] == 100
    assert state["submitted_order_ledger"]["submitted_sell_quantity_units"] == 0


def test_zero_inventory_and_empty_true_history_are_kept_as_zero_and_empty():
    facts = {**base(), "Q": 0, "D": 0, "history": []}
    request = build_order_request(facts, ())
    for arm in ("C", "E"):
        state = render_state(request, arm)
        assert state["own_history"] == []
        assert state["own_q_units"] == 0 and state["own_d_units"] == 0
        assert state["submitted_order_ledger"]["remaining_sell_submission_capacity_units"] == 0
        assert state["submitted_order_ledger"]["remaining_order_slots"] == 5


def test_unresolved_probability_tie_is_an_incomplete_trace_with_conservative_bound():
    from capclaims.injection_analysis import score_portfolio
    from capclaims.paper.orders import PaperOrder
    facts = base()
    request = build_order_request(facts, ())
    tied = {candidate.candidate_id: 1 / len(request.candidates) for candidate in request.candidates}
    trace = advance_trace(facts, (), DecisionReply.from_request(request, tied))
    outcome = next_trace_action(facts, trace)
    assert outcome["kind"] == "incomplete" and outcome["reason"] == "unresolved_tie"
    truth = (PaperOrder("truth", "synthetic", "sell", 10, 99),)
    scored = score_portfolio(None, truth, facts["Q"])
    assert scored["status"] == "unresolved"
    assert scored["score_kind"] == "conservative_upper_bound"
    assert scored["loss"] >= 0


def test_provider_failure_stays_a_failure_and_cannot_complete_portfolio():
    facts = base()
    trace = advance_trace(facts, (), None)
    outcome = next_trace_action(facts, trace)
    assert outcome["kind"] == "incomplete" and outcome["reason"] == "provider_failure"


def test_ten_actual_sdk_guarded_requests_preserve_distinct_effective_states(tmp_path):
    import json
    import httpx2
    from typesafe_sdk import AsyncTypeSafeClient
    from typesafe_sdk._core.retry import RetryPolicy
    from capclaims.backends.jev import MODEL
    from capclaims.injection_study import InjectionBackend, create_global_budget, create_guarded_transport

    async def run():
        reached = asyncio.Event()
        active = peak = 0
        seen = []

        async def handler(outgoing):
            nonlocal active, peak
            body = json.loads((await outgoing.aread()).decode("utf-8"))
            seen.append(body["state"]["own_p"])
            active += 1
            peak = max(peak, active)
            if active == 10:
                reached.set()
            await asyncio.wait_for(reached.wait(), timeout=5)
            criteria = body["questions"]["decision"]["criteria"]
            selected = next(k for k in criteria if "stop" in k.lower())
            active -= 1
            return httpx2.Response(200, json={"model": MODEL,
                "usage": {"input_tokens": 1, "output_tokens": 1},
                "answers": {"decision": {"type": "choice", "choice": selected,
                    "confidence": 1.0, "probabilities": {k: float(k == selected) for k in criteria}}}})

        budget = create_global_budget(tmp_path / "attempts.jsonl", fake=True)
        guard = create_guarded_transport(httpx2.MockTransport(handler), budget)
        http = httpx2.AsyncClient(transport=guard, follow_redirects=False)
        client = AsyncTypeSafeClient(api_key="offline-dummy", model=MODEL,
            retry=RetryPolicy(max_retries=0), http_client=http,
            base_url="https://api.typesafe.ai", compression=False)
        backend = InjectionBackend(api_key="offline-dummy", client=client, arm="C",
            instruction_profile="portfolio_third_v1", objective="portfolio_prediction_v1",
            rps=10000, tps=100000000, max_requests=10, max_estimated_input_tokens=1000000,
            max_in_flight=10, load_environment=False)
        requests = [build_order_request({**base(), "agent_id": f"parallel-{i}", "p": 51 + i}, ())
                    for i in range(10)]
        try:
            wires = await asyncio.gather(*(serialize_sdk_wire(q, "portfolio_third_v1", render_state(q, "C"))
                                           for q in requests))
            assert len({w.body_sha256 for w in wires}) == 10
            for wire in wires:
                guard.register(wire)
            replies = await backend.decide_many(requests)
            assert peak == 10 and sorted(seen) == list(range(51, 61))
            assert len(guard.http_timings) == budget.usage["requests_charged"] == 10
            assert all(q.request_id == r.request_id and q.observation_hash == r.observation_hash
                       for q, r in zip(requests, replies))
        finally:
            backend._closed = True
            await client.aclose()
    asyncio.run(run())

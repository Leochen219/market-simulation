from capclaims.engine.protocol import DecisionReply
from capclaims.paper.orders import PaperOrder
from capclaims.portfolio_prediction import (
    advance_trace, build_order_request, final_portfolio, next_trace_action,
    replay_trace,
)
from capclaims.backends.jev_instructions import build_instruction


def base(*, q=100, round_no=9):
    return {"agent_id": "opaque-agent-0", "round": round_no, "treatment": "GA",
        "wholesale_price_dollars": 10, "p": 70, "Q": q, "D": 40,
        "history": [{"round": 8, "p": 75, "Q": 90, "D": 50,
            "orders": [{"type": "BUY", "amount": 10, "price": 71},
                {"type": "SELL", "amount": 4, "price": 80}]}]}


def choose(req, value):
    selected = [candidate for candidate in req.candidates if candidate.value == value]
    assert len(selected) == 1
    probabilities = {candidate.candidate_id: float(candidate == selected[0]) for candidate in req.candidates}
    return DecisionReply.from_request(req, probabilities, backend="test", model_revision="fixture")


def add_model_choice(base_state, trace, value):
    req = build_order_request(base_state, trace)
    assert req is not None
    return advance_trace(base_state, trace, choose(req, value))


def add_digit_component(base_state, trace, number):
    for digit in f"{number:03d}":
        action = next_trace_action(base_state, trace)
        if action["kind"] == "forced":
            assert action["value"] == int(digit)
            trace = advance_trace(base_state, trace, None)
            continue
        assert action["kind"] == "request"
        trace = add_model_choice(base_state, trace, int(digit))
    return trace


def add_order(base_state, trace, side, qty, price):
    trace = add_model_choice(base_state, trace, side)
    trace = add_digit_component(base_state, trace, qty)
    trace = add_digit_component(base_state, trace, price)
    return trace


def test_stop_finishes_immediately_with_empty_portfolio_and_visible_market_facts():
    state = base(q=0)
    request = build_order_request(state, ())
    assert request is not None
    assert {candidate.value for candidate in request.candidates} == {"buy", "STOP"}
    assert "sell" not in {candidate.value for candidate in request.candidates}
    assert request.observation["market_size"] == 35
    assert request.observation["own_q_units"] == 0 and request.observation["own_d_units"] == 40
    assert request.observation["own_pretrade_profit_dollars"] == "0.00"
    assert request.observation["phase"] == "orders"
    assert request.observation["public_instructions"]["profit"].startswith("Accounting identity:")
    assert "Maximize" not in str(request.observation["public_instructions"])
    assert "seeded slot ties" not in str(request.observation["public_instructions"])
    assert "self_match_policy" not in request.observation
    trace = advance_trace(state, (), choose(request, "STOP"))
    assert next_trace_action(state, trace)["kind"] == "complete"
    assert final_portfolio(state, trace) == ()


def test_forecast_profiles_differ_only_in_perspective_sentence_and_objective_is_not_profit():
    request = build_order_request(base(), ())
    first = build_instruction(request, "portfolio_first_v1")
    third = build_instruction(request, "portfolio_third_v1")
    assert first.replace("You are this participant, deciding what orders to submit. ", "") == \
        third.replace("You are an observer predicting what orders this participant would submit. ", "")
    assert "not a recommendation or profit-maximization task" in first


def test_formal_rounds_map_to_paper_game_clock_and_training_history_clock():
    from capclaims.portfolio_prediction import _validate_base
    state = base(round_no=9)
    request = build_order_request(state, ())
    assert request.round_index == 13
    assert request.observation["round"] == 14
    assert request.observation["horizon"] == 20
    assert request.observation["own_history"][-1][0] == 13
    state10 = _validate_base({**state, "round": 10})
    request10 = build_order_request(state10, ())
    assert request10.round_index == 14
    assert request10.observation["round"] == 15


def test_full_five_order_portfolio_preserves_order_mixed_sides_duplicates_and_boundaries():
    state = base(q=40)
    trace = ()
    decisions = [
        ("buy", 999, 999),
        ("sell", 20, 1),
        ("buy", 5, 1),
        ("buy", 5, 1),  # Legal duplicate quote is preserved.
        ("sell", 20, 999),
    ]
    for side, qty, price in decisions:
        trace = add_order(state, trace, side, qty, price)
    assert next_trace_action(state, trace)["kind"] == "complete"
    portfolio = final_portfolio(state, trace)
    assert [(order.side, order.qty, order.limit_dollars) for order in portfolio] == decisions
    assert len(portfolio) == 5
    assert sum(order.qty for order in portfolio if order.side == "sell") == 40


def test_buys_do_not_replenish_exhausted_gross_sell_capacity():
    state = base(q=5)
    trace = add_order(state, (), "sell", 5, 50)
    trace = add_order(state, trace, "buy", 999, 999)
    request = build_order_request(state, trace)
    assert request is not None and request.stage == "paper:slot2:side"
    assert {candidate.value for candidate in request.candidates} == {"buy", "STOP"}


def test_sell_quantity_singletons_are_constraints_not_model_calls():
    state = base(q=1)
    trace = add_model_choice(state, (), "sell")
    for digit in (0, 0, 1):
        action = next_trace_action(state, trace)
        assert action["kind"] == "forced"
        trace = advance_trace(state, trace, None)
    trace = add_digit_component(state, trace, 1)  # singleton quantity is already complete; price=001 follows
    trace = add_model_choice(state, trace, "STOP")
    portfolio = final_portfolio(state, trace)
    assert [(order.side, order.qty, order.limit_dollars) for order in portfolio] == [("sell", 1, 1)]
    forced = [event for event in trace if event["kind"] == "forced"]
    assert len(forced) == 3
    assert all(event["origin"] == "constraint" for event in forced)


def test_leading_zero_digits_are_preserved_and_partial_order_is_not_committed():
    state = base()
    trace = add_model_choice(state, (), "buy")
    trace = add_digit_component(state, trace, 5)
    trace = add_model_choice(state, trace, 0)  # price hundreds
    trace = add_model_choice(state, trace, 0)  # price tens
    request = build_order_request(state, trace)
    assert request is not None and request.stage == "paper:slot0:price:u"
    assert request.observation["partial_orders"] == []
    result = replay_trace(state, trace)
    assert result["status"] == "incomplete" and result["reason"] == "truncated_trace"
    try:
        final_portfolio(state, trace)
    except ValueError as error:
        assert "incomplete" in str(error)
    else:
        raise AssertionError("truncated price prefix was treated as a completed portfolio")


def test_unresolved_tie_and_provider_failure_make_whole_portfolio_incomplete():
    state = base()
    request = build_order_request(state, ())
    probabilities = {candidate.candidate_id: 0. for candidate in request.candidates}
    for candidate in request.candidates[:2]:
        probabilities[candidate.candidate_id] = .5
    tied = DecisionReply.from_request(request, probabilities, backend="test", model_revision="fixture")
    trace = advance_trace(state, (), tied)
    assert replay_trace(state, trace)["reason"] == "unresolved_tie"
    request = build_order_request(state, ())
    failed = advance_trace(state, (), None)
    assert replay_trace(state, failed)["reason"] == "provider_failure"


def test_invalid_reply_and_tampered_choice_trace_fail_closed():
    state = base()
    request = build_order_request(state, ())
    wrong_request = request.__class__("another-agent", request.round_index, request.stage,
        request.observation, request.candidates)
    bad_reply = choose(wrong_request, "buy")
    failed = advance_trace(state, (), bad_reply)
    assert replay_trace(state, failed)["reason"] == "invalid_reply"
    chosen = advance_trace(state, (), choose(request, "buy"))
    tampered = ({**chosen[0], "value": "STOP"},)
    try:
        replay_trace(state, tampered)
    except ValueError as error:
        assert "does not match" in str(error)
    else:
        raise AssertionError("tampered trace value passed replay")


def test_changed_own_choice_changes_later_request_and_current_truth_is_not_a_base_field():
    state = base(q=100)
    buy_trace = add_model_choice(state, (), "buy")
    sell_trace = add_model_choice(state, (), "sell")
    buy_request = build_order_request(state, buy_trace)
    sell_request = build_order_request(state, sell_trace)
    assert buy_request.observation["own_side"] == "buy"
    assert sell_request.observation["own_side"] == "sell"
    assert buy_request.observation_hash != sell_request.observation_hash
    bad = {**state, "target_orders": []}
    try:
        build_order_request(bad, ())
    except ValueError as error:
        assert "unknown or missing" in str(error)
    else:
        raise AssertionError("target truth was admitted into a frozen model base")

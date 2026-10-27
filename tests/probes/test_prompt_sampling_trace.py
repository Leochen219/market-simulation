from capclaims.prompt_sampling_trace import advance, replay, next_action as next_trace_action, build_request as build_order_request
from capclaims.engine.protocol import DecisionReply


def reply_for(request, probabilities):
    chosen = next(key for key in probabilities)
    return DecisionReply.from_request(request, probabilities, probabilities, backend="synthetic",
        model_revision="synthetic", usage={"service_choice": chosen, "service_confidence": 1.0})


def base(q=6):
    return {"agent_id": "synthetic", "round": 9, "treatment": "GA", "wholesale_price_dollars": 10,
        "p": 80, "Q": q, "D": 30, "history": []}


def test_client_argmax_trace_replays_and_detects_tampering():
    b = base()
    trace = ()
    while next_trace_action(b, trace)["kind"] not in {"complete", "incomplete"}:
        action = next_trace_action(b, trace)
        if action["kind"] == "forced":
            trace = advance(b, trace)
            continue
        req = build_order_request(b, trace)
        ids = [c.candidate_id for c in req.candidates]
        if req.stage.endswith(":side"):
            raw = {key: float(key == next((c.candidate_id for c in req.candidates if c.value == "STOP"), ids[0])) for key in ids}
        else:
            raw = {key: 1 / len(ids) for key in ids}
        trace = advance(b, trace, reply_for(req, raw), probabilities=raw, decoder="client_argmax")
    result = replay(b, trace)
    assert result["status"] == "complete"
    broken = list(trace)
    broken[-1] = {**broken[-1], "value": 999}
    try:
        replay(b, broken)
    except ValueError:
        pass
    else:
        raise AssertionError("tampered client trace replayed")


def test_sell_capacity_never_exceeds_q():
    b = base(1)
    trace = ()
    while next_trace_action(b, trace)["kind"] not in {"complete", "incomplete"}:
        action = next_trace_action(b, trace)
        if action["kind"] == "forced":
            trace = advance(b, trace)
            continue
        request = build_order_request(b, trace)
        chosen = next((c.candidate_id for c in request.candidates if c.value == "sell"),
            next((c.candidate_id for c in request.candidates if c.value == "STOP"), request.candidates[0].candidate_id))
        probs = {c.candidate_id: float(c.candidate_id == chosen) for c in request.candidates}
        trace = advance(b, trace, reply_for(request, probs), probabilities=probs, decoder="client_argmax")
        if len(trace) > 12:
            break
    replayed = replay(b, trace)
    assert all(o.qty <= 1 for o in replayed["orders"] if o.side == "sell")


def test_mixed_five_order_maxslot_and_forced_sell_digit_transitions():
    b = base(6)
    trace = ()
    def request_at_current():
        nonlocal trace
        while next_trace_action(b, trace)["kind"] == "forced":
            trace = advance(b, trace)
        return build_order_request(b, trace)
    expected_sides = ["buy", "sell", "buy", "buy", "buy"]
    quantities = [1, 6, 2, 2, 2]
    for slot in range(5):
        request = request_at_current()
        ids = [c.candidate_id for c in request.candidates]
        desired = expected_sides[slot]
        chosen = next(c.candidate_id for c in request.candidates if c.value == desired)
        probs = {key: float(key == chosen) for key in ids}
        trace = advance(b, trace, reply_for(request, probs), probabilities=probs, decoder="client_argmax")
        for field, number in (("qty", quantities[slot]), ("price", 150 + slot)):
            digits = f"{number:03d}"
            while True:
                action = next_trace_action(b, trace)
                if action["kind"] == "forced":
                    trace = advance(b, trace)
                    continue
                if action["kind"] in {"complete", "incomplete"}:
                    break
                request = build_order_request(b, trace)
                if not request.stage.startswith(f"paper:slot{slot}:{field}:"):
                    break
                digit = int(digits["htu".index(request.stage[-1])])
                chosen = next(c.candidate_id for c in request.candidates if c.value == digit)
                probs = {c.candidate_id: float(c.candidate_id == chosen) for c in request.candidates}
                trace = advance(b, trace, reply_for(request, probs), probabilities=probs, decoder="client_argmax")
    result = replay(b, trace, person="synthetic", repeat=0)
    assert result["status"] == "complete"
    assert len(result["orders"]) == 5
    assert [o.side for o in result["orders"]] == expected_sides
    assert sum(o.qty for o in result["orders"] if o.side == "sell") == 6
    assert next_trace_action(b, trace)["kind"] == "complete"


def test_nonmax_provider_choice_is_recorded_but_does_not_fail():
    b = base()
    req = build_order_request(b, ())
    ids = [c.candidate_id for c in req.candidates]
    raw = {ids[0]: .1, ids[1]: .7, ids[2]: .2}
    reply = DecisionReply.from_request(req, raw, raw, backend="synthetic", model_revision="synthetic",
        usage={"service_choice": ids[0], "service_confidence": .4})
    trace = advance(b, (), reply, probabilities=raw, decoder="client_argmax")
    assert trace[0]["nonmax_provider_choice"] is True
    assert replay(b, trace, person="synthetic", repeat=0)["status"] in {"complete", "incomplete"}


def test_provider_choice_on_other_equal_maximum_is_not_nonmax_anomaly():
    b = base()
    req = build_order_request(b, ())
    ids = [c.candidate_id for c in req.candidates]
    raw = {ids[0]: .5, ids[1]: .5, ids[2]: 0.0}
    reply = DecisionReply.from_request(req, raw, raw, backend="synthetic", model_revision="synthetic",
        usage={"service_choice": ids[1], "service_confidence": .5})
    event = advance(b, (), reply, probabilities=raw, decoder="client_argmax")[0]
    assert event["nonmax_provider_choice"] is False
    assert event["service_choice_tie_order_disagreement"] is True


def test_invalid_confidence_and_service_choice_fail_closed():
    b = base()
    req = build_order_request(b, ())
    ids = [c.candidate_id for c in req.candidates]
    raw = {ids[0]: 1.0, ids[1]: 0.0, ids[2]: 0.0}
    bad_confidence = DecisionReply.from_request(req, raw, raw, backend="synthetic", model_revision="synthetic",
        usage={"service_choice": ids[0], "service_confidence": float("nan")})
    try:
        advance(b, (), bad_confidence, probabilities=raw, decoder="client_argmax")
    except ValueError:
        pass
    else:
        raise AssertionError("NaN provider confidence was accepted")

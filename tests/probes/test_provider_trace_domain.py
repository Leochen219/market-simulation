"""New readout must preserve the entire original legal state machine."""
import pytest

from capclaims import portfolio_prediction as old
from capclaims import provider_choice_trace as new
from capclaims.engine.protocol import DecisionReply


@pytest.mark.parametrize("q", [0, 1, 9, 10, 99, 100, 399, 400])
@pytest.mark.parametrize("mode", ["minimum", "maximum", "mixed"])
def test_unique_provider_choices_preserve_requests_masks_and_final_portfolios(q, mode):
    base = {"agent_id": "synthetic-domain-parity", "round": 9, "treatment": "GA",
            "wholesale_price_dollars": 10, "p": 100, "Q": q, "D": 0,
            "history": [{"round": r, "p": 51, "Q": 400, "D": 200,
                         "orders": [{"type": "SELL", "amount": 400, "price": 999},
                                    {"type": "BUY", "amount": 999, "price": 1}]}
                        for r in (6, 7, 8)]}
    legacy, current = (), ()
    for _ in range(40):
        prior_action = old.next_trace_action(base, legacy)
        action = new.next_provider_trace_action(base, current)
        assert action == prior_action
        if action["kind"] == "complete":
            break
        assert action["kind"] in {"forced", "request"}
        if action["kind"] == "forced":
            legacy = old.advance_trace(base, legacy, None)
            current = new.advance_provider_trace(base, current, None)
            continue
        original = old.build_order_request(base, legacy)
        request = new.build_provider_order_request(base, current)
        assert request == original
        if request.stage.endswith(":side"):
            sides = {c.value for c in request.candidates}
            slot = request.observation["slot"]
            desired = "sell" if mode == "maximum" or mode == "mixed" and slot % 2 == 0 else "buy"
            value = desired if desired in sides else "buy"
        else:
            digits = [c.value for c in request.candidates]
            value = min(digits) if mode == "minimum" else max(digits)
        selected = next(c for c in request.candidates if c.value == value)
        probabilities = {c.candidate_id: float(c == selected) for c in request.candidates}
        reply = DecisionReply.from_request(request, probabilities, probabilities,
            backend="synthetic-test", model_revision="jev-1.14.0")
        answer = {"type": "choice", "choice": selected.candidate_id, "confidence": 1,
                  "probabilities": probabilities}
        legacy = old.advance_trace(base, legacy, reply)
        current = new.advance_provider_trace(base, current, answer, reply)
    else:
        pytest.fail("five full-domain orders exceeded the proven trace bound")
    a, b = old.replay_trace(base, legacy), new.replay_provider_trace(base, current)
    assert a == b and a["status"] == "complete" and len(a["orders"]) == 5
    assert sum(o.qty for o in b["orders"] if o.side == "sell") <= q
    assert all(1 <= o.qty <= 999 and 1 <= o.limit_dollars <= 999 for o in b["orders"])
    with pytest.raises(ValueError, match="complete|terminal"):
        new.advance_provider_trace(base, current, answer, reply)


def test_stop_uses_literal_provider_choice_with_unmodified_distribution():
    base = {"agent_id": "synthetic-stop", "round": 9, "treatment": "GA",
            "wholesale_price_dollars": 10, "p": 51, "Q": 400, "D": 200, "history": []}
    request = new.build_provider_order_request(base, ())
    selected = next(c for c in request.candidates if c.value == "STOP")
    probabilities = {c.candidate_id: 1 / len(request.candidates) for c in request.candidates}
    answer = {"type": "choice", "choice": selected.candidate_id, "confidence": 0, "probabilities": probabilities}
    reply = DecisionReply.from_request(request, probabilities, probabilities)
    trace = new.advance_provider_trace(base, (), answer, reply)
    assert trace[0]["raw_probabilities"] == probabilities
    assert trace[0]["normalized_probabilities"] == reply.probabilities
    result = new.replay_provider_trace(base, trace)
    assert result["status"] == "complete" and result["orders"] == ()

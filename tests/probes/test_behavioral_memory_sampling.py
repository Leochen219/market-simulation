import asyncio
import json

from capclaims.behavioral_memory.sampling import sample_trajectory


class FirstCandidateClient:
    def __init__(self, count="N5"):
        self.count = count
        self.calls = []

    async def choose(self, *, request_id, state, instructions, criteria):
        self.calls.append((request_id, state, criteria))
        if state["decision_field"] == "order_count":
            selected = self.count
        else:
            selected = next(iter(criteria))
        return {"model": "jev-1.14.0", "usage": {"input_tokens": 3, "output_tokens": 1},
                "answers": {"decision": {"type": "choice", "choice": selected,
                    "probabilities": {key: float(key == selected) for key in criteria}}}}


def run_inline(coroutine):
    """Run these immediate fake-client coroutines without opening network-capable sockets."""
    try:
        coroutine.send(None)
    except StopIteration as result:
        return result.value
    raise AssertionError("mock request unexpectedly yielded asynchronous I/O")


def row(arm):
    observation = {"protocol_interface": "J2", "decision_interface": "economic_v2", "horizon": 20, "round": 18,
        "own_p": 100, "wholesale_price_dollars": 10, "own_q_units": 3,
        "own_d_units": 8, "history": [], "partial_orders": [],
        "submitted_order_ledger": {"submitted_order_count": 0,
            "remaining_order_slots": 5, "submitted_buy_quantity_units": 0,
            "submitted_sell_quantity_units": 0,
            "remaining_sell_submission_capacity_units": 3}}
    return {"person_id": "dev-abc", "target_round": 13, "repeat": 0, "arm": arm,
        "request_prefix": f"req-{arm}", "prompt": json.dumps({"system_prompt": "predict",
            "observation": observation}), "current_state": {"round": 13, "p": 100, "w": 10, "Q": 3, "D": 8}}


def test_maximum_legal_j2_trace_uses_36_sequential_requests_and_respects_sell_capacity():
    client = FirstCandidateClient("N5")
    result = run_inline(sample_trajectory(client, row("M1")))
    assert result["physical_requests"] == 36
    assert len(result["orders"]) == 5
    assert result["orders"] == [{"side": "buy", "qty": 1, "price": 1}] * 5
    assert len({call[0] for call in client.calls}) == 36


def test_arm_change_does_not_change_semantic_common_draws():
    a = FirstCandidateClient("N1")
    b = FirstCandidateClient("N1")
    one = run_inline(sample_trajectory(a, row("M0")))
    two = run_inline(sample_trajectory(b, row("M2")))
    assert [step["draw"] for step in one["trace"]] == [step["draw"] for step in two["trace"]]
    assert one["orders"] == two["orders"]

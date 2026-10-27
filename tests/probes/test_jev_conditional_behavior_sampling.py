import asyncio

from capclaims.conditional_behavior.sampling import sample_portfolio


class ScriptedClient:
    def __init__(self, chooser):
        self.calls = []
        self.chooser = chooser

    async def choose(self, *, request_id, state, instructions, criteria):
        self.calls.append({"request_id": request_id, "state": state,
            "instructions": instructions, "criteria": criteria})
        selected = self.chooser(state, criteria)
        return {"model": "jev-1.14.0", "usage": {"input_tokens": 4, "output_tokens": 1},
            "answers": {"decision": {"type": "choice", "choice": selected,
                "confidence": 1.0, "probabilities": {key: float(key == selected) for key in criteria}}}}


def _row(arm="C", q=10):
    interface = "J2" if arm == "E" else "J1"
    return {"case_id": "a" * 64, "person_id": "opaque", "target_round": 14,
        "sample": 0, "arm": arm, "interface": interface,
        "observation": {"round": 14, "market_size": 35, "own_p": 70,
            "wholesale_price_dollars": 10, "own_q_units": q, "own_d_units": 100,
            "phase": "orders", "decision_interface": "economic_v2", "protocol_interface": interface,
            "history": [], "partial_orders": [], "submitted_order_ledger": {
                "submitted_order_count": 0, "remaining_order_slots": 5,
                "submitted_buy_quantity_units": 0, "submitted_sell_quantity_units": 0,
                "remaining_sell_submission_capacity_units": q},
            "own_pretrade_profit_dollars": 700 - 10 * q,
            "public_instructions": {"sell_cap": "Gross sells cannot exceed initial Q."}}}


def test_j1_samples_one_complete_order_then_stop_with_economic_v2_named_state():
    def chooser(state, criteria):
        if state["decision_field"] == "side":
            return "BUY" if state["slot"] == 0 else "STOP"
        return next(iter(criteria))
    client = ScriptedClient(chooser)
    result = asyncio.run(sample_portfolio(client, _row(), seed=20261026))
    assert result["orders"] == [{"side": "buy", "qty": 1, "price": 1}]
    assert len(client.calls) == result["provider_requests"] == 8
    numeric_state = next(call["state"] for call in client.calls if call["state"].get("decision_field") == "quantity")
    assert numeric_state["decision_interface"] == "economic_v2"
    assert numeric_state["partial_orders"] == []
    assert "selected_order_count" not in numeric_state
    assert all("selected_order_count" not in call["state"] for call in client.calls)
    assert client.calls[0]["criteria"] == {"BUY": client.calls[0]["criteria"]["BUY"],
        "STOP": client.calls[0]["criteria"]["STOP"], "SELL": client.calls[0]["criteria"]["SELL"]}
    assert numeric_state["submitted_order_ledger"]["submitted_order_count"] == 0
    assert "partial_order_fields" not in numeric_state
    assert result["trace"][0]["selected"] == "BUY"


def test_j2_fixes_order_count_before_side_and_never_chooses_stop():
    def chooser(state, criteria):
        if state.get("decision_field") == "order_count":
            return "N1"
        if state.get("decision_field") == "side":
            return "SELL" if "SELL" in criteria else "BUY"
        return next(iter(criteria))
    client = ScriptedClient(chooser)
    result = asyncio.run(sample_portfolio(client, _row("E"), seed=20261026))
    assert result["orders"] == [{"side": "sell", "qty": 1, "price": 1}]
    assert client.calls[0]["state"]["decision_field"] == "order_count"
    assert "selected_order_count" not in client.calls[0]["state"]
    assert client.calls[0]["state"]["slots_remaining"] == 5
    assert len(client.calls) == 8
    assert all(call["state"]["selected_order_count"] == 1 for call in client.calls[1:])
    assert all(call["state"]["remaining_planned_order_slots"] == 1 for call in client.calls[1:])
    assert all("STOP" not in call["criteria"] for call in client.calls)


def test_j1_zero_inventory_never_offers_sell_and_stop_preserves_empty_portfolio():
    client = ScriptedClient(lambda state, criteria: "STOP")
    result = asyncio.run(sample_portfolio(client, _row(q=0), seed=7))
    assert result["orders"] == [] and len(client.calls) == 1
    assert "SELL" not in client.calls[0]["criteria"]


def test_synthetic_prompt_describes_actual_history_packet_count():
    def chooser(state, criteria):
        if state["decision_field"] == "side":
            return "STOP"
        return next(iter(criteria))
    row = _row("C")
    row["observation"]["history"] = [{"round": 16, "p": 50, "Q": 10, "D": 20, "orders": []},
                                      {"round": 17, "p": 51, "Q": 10, "D": 20, "orders": []}]
    synthetic = ScriptedClient(chooser)
    asyncio.run(sample_portfolio(synthetic, row, seed=1, category="synthetic"))
    human = ScriptedClient(chooser)
    asyncio.run(sample_portfolio(human, row, seed=1, category="human"))
    assert "2 historical example(s)" in synthetic.calls[0]["instructions"]
    assert "formal rounds 6 through 13" not in synthetic.calls[0]["instructions"]
    assert "formal rounds 6 through 13" in human.calls[0]["instructions"]


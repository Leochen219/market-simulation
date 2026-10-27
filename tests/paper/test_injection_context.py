from types import SimpleNamespace

import pytest

from capclaims.injection_context import render_state, semantic_history


def _request(history=None, partial=None, **overrides):
    observation = {
        "agent_id": "opaque", "round": 14, "round_index": 13, "treatment": "GA",
        "wholesale_price_dollars": 10, "own_p": 80, "own_q_units": 100, "own_d_units": 40,
        "own_history": [[row["round"], row["p"], row["Q"], row["D"], None, None, None,
            [[i, order["side"], order["qty"], order["limit_dollars"]]
             for i, order in enumerate(row["orders"])]] for row in history or []],
        "own_history_fields": "legacy fields",
        "partial_orders": [[i, row["side"], row["qty"], row["limit_dollars"]]
                            for i, row in enumerate(partial or [])],
        "slot": len(partial or []), "phase": "orders",
    }
    observation.update(overrides)
    return SimpleNamespace(observation=observation)


def test_named_history_is_semantically_lossless_and_does_not_sort_orders():
    history = [{"round": 11, "p": 80, "Q": 100, "D": 40,
                "orders": [{"side": "sell", "qty": 20, "limit_dollars": 77},
                           {"side": "buy", "qty": 10, "limit_dollars": 31}]}]
    named = render_state(_request(history), "B")
    assert semantic_history(named) == semantic_history(_request(history).observation)
    assert named["own_history"][0]["submitted_orders"][0]["side"] == "SELL"
    assert named["own_history"][0]["submitted_orders"][1]["side"] == "BUY"
    assert named["own_history"][0]["outcomes"] == "not_provided_in_input"
    assert "history" not in named


def test_ledger_counts_submitted_orders_without_double_counting_pending_order():
    partial = [{"side": "sell", "qty": 20, "limit_dollars": 50},
               {"side": "buy", "qty": 10, "limit_dollars": 60}]
    request = _request([], partial, own_q_units=100, slot=2)
    state = render_state(request, "C")
    ledger = state["submitted_order_ledger"]
    assert ledger == {"submitted_order_count": 2, "remaining_order_slots": 3,
                      "submitted_buy_quantity_units": 10, "submitted_sell_quantity_units": 20,
                      "remaining_sell_submission_capacity_units": 80}
    assert state["own_history"] == []


@pytest.mark.parametrize("arm", ["A", "B", "C", "D", "E"])
def test_each_arm_keeps_full_observation_action_space_and_does_not_leak_arm_label(arm):
    history = [{"round": 11, "p": 51, "Q": 0, "D": 0, "orders": []}]
    request = _request(history, own_p=51, own_q_units=0, own_d_units=0)
    rendered = render_state(request, arm, donor_history=history if arm == "D" else None)
    assert rendered["own_p"] == 51 and rendered["own_q_units"] == 0 and rendered["own_d_units"] == 0
    assert not {"arm", "arm_id", "donor_id", "donor_identity"} & set(rendered)
    assert "legal_actions" not in rendered or rendered["legal_actions"] == ["buy", "STOP"]


def test_a_preserves_old_compact_state_exactly():
    history = [{"round": 11, "p": 99, "Q": 400, "D": 200, "orders": []}]
    base = _request(history, [{"side": "sell", "qty": 400, "limit_dollars": 999}]).observation
    assert render_state(SimpleNamespace(observation=base), "A") == base


def test_d_requires_complete_donor_history_and_never_relabels_it_as_self():
    with pytest.raises(ValueError):
        render_state(_request([]), "D")


def test_donor_history_maps_formal_rounds_to_the_game_clock_without_touching_q_or_quotes():
    donor = [{"round": r, "p": 77, "Q": 24, "D": 10,
              "orders": [{"type": "SELL", "amount": 20, "price": 44}]} for r in (6, 7, 8)]
    state = render_state(_request([]), "D", donor_history=donor)
    assert [row["game_round"] for row in state["own_history"]] == [11, 12, 13]
    assert [row["initial_inventory_units"] for row in state["own_history"]] == [24, 24, 24]
    assert [row["submitted_orders"][0]["quantity_units"] for row in state["own_history"]] == [20, 20, 20]


def test_non_integer_decimal_price_is_rejected_instead_of_rounded():
    bad_history = [[11, "12.50", 100, 40, None, None, None, []]]
    request = _request([])
    request.observation["own_history"] = bad_history
    with pytest.raises(ValueError, match="integer-dollar"):
        render_state(request, "B")

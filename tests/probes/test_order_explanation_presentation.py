from types import SimpleNamespace

import pytest

from capclaims.order_presentation import (
    CONTROL,
    EXPLICIT,
    inverse_explicit_state,
    order_prompt,
    render_order_state,
)
from capclaims.injection_context import render_state


def request(state):
    return SimpleNamespace(observation=state)


def test_explicit_render_names_only_partial_order_values_and_roundtrips_key_position():
    state = {
        "round": 14, "own_p": 43, "own_q_units": 400, "own_d_units": 90,
        "history": [],
        "partial_order_fields": "slot,side,qty units,limit $/unit",
        "decision_interface": "legacy_v1",
        "partial_orders": [[0, "BUY", 12, 50], [2, "sell", 4, 44]],
        "submitted_order_ledger": {"submitted_order_count": 2, "remaining_order_slots": 3},
        "own_side": "sell",
        "own_qty_units": 2,
    }
    control = render_state(request(state), "C")
    state = control
    req = request({**state, "partial_orders": [[0, "BUY", 12, 50], [2, "sell", 4, 44]],
                   "partial_order_fields": "slot,side,qty units,limit $/unit",
                   "submitted_order_ledger": {"submitted_order_count": 2, "remaining_order_slots": 3},
                   "own_side": "sell", "own_qty_units": 2})
    control = render_order_state(req, CONTROL)
    explicit = render_order_state(req, EXPLICIT)
    assert list(explicit) == [key for key in control if key != "partial_order_fields"]
    assert "partial_order_fields" not in explicit
    assert explicit["partial_orders"] == [
        {"slot": 0, "side": "BUY", "quantity_units": 12, "limit_price_dollars_per_unit": 50},
        {"slot": 2, "side": "sell", "quantity_units": 4, "limit_price_dollars_per_unit": 44},
    ]
    assert list(inverse_explicit_state(explicit).items()) == list(control.items())
    assert req.observation["partial_orders"][0] == [0, "BUY", 12, 50]


def test_explicit_render_rejects_unknown_legend_instead_of_guessing():
    state = {"round": 14, "own_p": 43, "own_q_units": 400, "own_d_units": 90,
             "history": [], "partial_order_fields": "side,slot,qty units,limit $/unit", "partial_orders": []}
    with pytest.raises(ValueError, match="legend"):
        render_order_state(request(state), EXPLICIT)


def test_explicit_prompt_appends_one_fixed_semantics_paragraph():
    req = SimpleNamespace(observation={"round": 14, "phase": "orders", "decision_field": "side"}, candidates=[])
    control = order_prompt(req, CONTROL)
    explicit = order_prompt(req, EXPLICIT)
    assert explicit.startswith(control + "\n\n")
    assert explicit.count("\n\n") == control.count("\n\n") + 1
    for phrase in ("predicted earlier", "not executed trades", "unfinished order", "STOP", "SELL order-submission capacity"):
        assert phrase.lower() in explicit.lower()




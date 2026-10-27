"""Pure renderer and fixed explanation for the paired current-order pilot."""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

CONTROL = "C_joint_control"
EXPLICIT = "C_joint_explicit"
ARMS = (CONTROL, EXPLICIT)
PARTIAL_ORDER_FIELDS = "slot,side,qty units,limit $/unit"
EXPLANATION = (
    "Each object in partial_orders is one complete order already predicted earlier in this same forecast. "
    "These are accumulated submitted order intentions, not executed trades or newly observed actions by the participant. "
    "quantity_units is the quantity of that individual order, and limit_price_dollars_per_unit is its per-unit limit price. "
    "Adding an order leaves every earlier order in the portfolio. The submitted_order_ledger summarizes only these complete orders; "
    "it does not describe filled purchases or sales. When present, own_side and own_qty_units describe the current order being constructed; "
    "that unfinished order is not yet included in partial_orders or the ledger. Selecting STOP predicts that the participant adds no further order; "
    "it keeps the earlier orders in the predicted portfolio. remaining_sell_submission_capacity_units is remaining gross SELL order-submission capacity, not inventory after trade execution."
)


def _arm(arm: str) -> None:
    if arm not in ARMS:
        raise ValueError("unknown order-explanation arm")


def _validate_legend(state: Mapping[str, Any]) -> None:
    if state.get("partial_order_fields") != PARTIAL_ORDER_FIELDS:
        raise ValueError("partial-order legend does not match the frozen positional legend")


def _convert_orders(rows: Any) -> list[dict[str, Any]]:
    if not isinstance(rows, list):
        raise ValueError("partial_orders must be a list")
    converted = []
    for row in rows:
        if not isinstance(row, (list, tuple)) or len(row) != 4:
            raise ValueError("partial order must have the frozen slot,side,qty,price shape")
        slot, side, qty, price = row
        converted.append({"slot": slot, "side": side, "quantity_units": qty,
                          "limit_price_dollars_per_unit": price})
    return converted


def _restore_orders(rows: Any) -> list[list[Any]]:
    if not isinstance(rows, list):
        raise ValueError("explicit partial_orders must be a list")
    restored = []
    fields = ("slot", "side", "quantity_units", "limit_price_dollars_per_unit")
    for row in rows:
        if not isinstance(row, Mapping) or list(row) != list(fields):
            raise ValueError("explicit partial order does not have the frozen named shape")
        restored.append([row[name] for name in fields])
    return restored


def render_order_state(request: Any, arm: str) -> dict[str, Any]:
    """Render the unchanged C state or its exact named-partial-order treatment."""
    _arm(arm)
    from capclaims.injection_context import render_state

    c_state = render_state(request, "C")
    if arm == CONTROL:
        return c_state
    if "partial_orders" not in c_state:
        raise ValueError("C_joint state has no partial_orders field")
    _validate_legend(c_state)
    if "partial_orders" not in c_state or "partial_order_fields" not in c_state:
        raise ValueError("C_joint state lacks partial-order fields")
    result: dict[str, Any] = {}
    for key, value in c_state.items():
        if key == "partial_orders":
            result[key] = _convert_orders(value)
        elif key == "partial_order_fields":
            continue
        else:
            result[key] = deepcopy(value)
    return result


def inverse_explicit_state(state: Mapping[str, Any]) -> dict[str, Any]:
    """Restore explicit treatment state to its exact original C key order/content."""
    if not isinstance(state, Mapping):
        raise TypeError("explicit state must be a mapping")
    if "partial_orders" not in state or "partial_order_fields" in state:
        raise ValueError("explicit state has an invalid partial-order representation")
    restored: dict[str, Any] = {}
    for key, value in state.items():
        if key == "partial_orders":
            restored[key] = _restore_orders(value)
        else:
            restored[key] = deepcopy(value)
    if "decision_interface" not in restored:
        raise ValueError("explicit state lacks the frozen decision_interface insertion anchor")
    entries = list(restored.items())
    anchor = next(i for i, pair in enumerate(entries) if pair[0] == "decision_interface")
    entries.insert(anchor, ("partial_order_fields", PARTIAL_ORDER_FIELDS))
    restored = dict(entries)
    return restored


def order_prompt(request: Any, arm: str) -> str:
    """Return the frozen C_joint prompt, with one fixed paragraph in treatment."""
    _arm(arm)
    from capclaims.continuation_study import JOINT_APPEND
    from capclaims.backends.jev_instructions import build_instruction

    original = build_instruction(request, "portfolio_third_v1") + "\n\n" + JOINT_APPEND
    if arm == CONTROL:
        return original
    return original + "\n\n" + EXPLANATION

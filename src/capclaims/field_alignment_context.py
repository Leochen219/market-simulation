"""Exact, reversible field-name treatment for the C_joint state."""
from __future__ import annotations

from collections import OrderedDict
from typing import Any, Mapping

CONTROL = "C_joint_control"
ALIGNED = "C_joint_aligned"
RENAMES = OrderedDict((
    ("own_p", "retail_price_dollars_per_unit"),
    ("own_q_units", "initial_inventory_units"),
    ("own_d_units", "realized_demand_units"),
    ("round", "game_round"),
))
_INVERSE = {value: key for key, value in RENAMES.items()}


def _validate_arm(arm: str) -> None:
    if arm not in {CONTROL, ALIGNED}:
        raise ValueError("unknown field-alignment arm")


def render_field_state(request: Any, arm: str) -> dict[str, Any]:
    """Return the original C rendering or four-key aligned rendering, without mutation."""
    _validate_arm(arm)
    from capclaims.injection_context import render_state

    original = render_state(request, "C")
    missing = set(RENAMES) - set(original)
    collisions = set(RENAMES.values()) & set(original)
    if missing or collisions:
        raise ValueError("C_joint state lacks source fields or already contains aligned aliases")
    if arm == CONTROL:
        return original
    return {RENAMES.get(key, key): value for key, value in original.items()}


def inverse_aligned_state(state: Mapping[str, Any]) -> dict[str, Any]:
    """Restore an aligned state to the exact original C_joint key order and content."""
    if not isinstance(state, Mapping):
        raise TypeError("aligned state must be a mapping")
    missing = set(RENAMES.values()) - set(state)
    collisions = set(RENAMES) & set(state)
    if missing or collisions:
        raise ValueError("aligned state lacks treatment fields or contains control aliases")
    restored: dict[str, Any] = {}
    for key, value in state.items():
        original_key = _INVERSE.get(key, key)
        if original_key in restored:
            raise ValueError("aligned state contains a rename collision")
        restored[original_key] = value
    return restored

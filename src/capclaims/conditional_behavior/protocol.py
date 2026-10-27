"""Pure protocol validation and deterministic raw-distribution sampling."""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

from . import VERSION


def deterministic_draw(*, seed: int, version: str, arm: str, person: str,
                       target: int, sample: int, stage: str) -> float:
    """Return a stable [0,1) variate derived only from frozen semantic coordinates."""
    if (type(seed) is not int or not all(isinstance(v, str) and v for v in
            (version, arm, person, stage)) or type(target) is not int or
            type(sample) is not int or sample < 0):
        raise ValueError("draw coordinates are invalid")
    payload = json.dumps([version, seed, arm, person, target, sample, stage],
                         ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
    digest = hashlib.sha256(payload).digest()
    return (int.from_bytes(digest[:8], "big") >> 11) / (1 << 53)


def sample_categorical(probabilities: Mapping[str, Any], draw: float,
                      candidate_order: Sequence[str] | None = None) -> str:
    """Sample normalized raw provider probabilities, without truncation or smoothing."""
    if not isinstance(probabilities, Mapping) or not probabilities:
        raise ValueError("candidate probabilities must be a nonempty mapping")
    if isinstance(draw, bool) or not isinstance(draw, (int, float)) or not math.isfinite(draw) or not 0 <= draw < 1:
        raise ValueError("draw must be finite and in [0, 1)")
    values: list[tuple[str, float]] = []
    for key, value in probabilities.items():
        if not isinstance(key, str) or not key:
            raise ValueError("candidate IDs must be nonempty strings")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("probabilities must be numeric")
        number = float(value)
        if not math.isfinite(number) or number < 0:
            raise ValueError("probabilities must be finite and nonnegative")
        values.append((key, number))
    order = tuple(candidate_order) if candidate_order is not None else tuple(key for key, _ in values)
    from capclaims.backends.jev import normalize_jev_probabilities
    normalized, _ = normalize_jev_probabilities(dict(values), order)
    threshold = draw
    cumulative = 0.0
    last_positive = None
    for key in order:
        value = normalized[key]
        if value <= 0:
            continue
        last_positive = key
        cumulative += value
        if threshold < cumulative:
            return key
    # Floating summation can leave the last ULP below one.
    assert last_positive is not None
    return last_positive


def validate_orders(orders: Sequence[Mapping[str, Any]], *, initial_q: int,
                    max_orders: int = 5) -> tuple[dict[str, Any], ...]:
    if type(initial_q) is not int or not 0 <= initial_q <= 400:
        raise ValueError("initial Q must be an integer from 0 through 400")
    if not isinstance(orders, (tuple, list)) or len(orders) > max_orders:
        raise ValueError("portfolio must contain no more than five orders")
    result = []
    gross_sell = 0
    for order in orders:
        if not isinstance(order, Mapping) or order.get("side") not in {"buy", "sell"}:
            raise ValueError("each order must have side buy or sell")
        qty, price = order.get("qty"), order.get("price", order.get("limit_dollars"))
        if type(qty) is not int or not 1 <= qty <= 999:
            raise ValueError("order quantity must be an integer from 1 through 999")
        if type(price) is not int or not 1 <= price <= 999:
            raise ValueError("order price must be an integer from 1 through 999")
        if order["side"] == "sell":
            gross_sell += qty
        result.append({"side": order["side"], "qty": qty, "price": price})
    if gross_sell > initial_q:
        raise ValueError("gross SELL submissions exceed initial Q")
    return tuple(result)


def replay_choices(state: Mapping[str, Any], actions: Sequence[Mapping[str, Any]], *,
                   interface: str) -> dict[str, Any]:
    """Replay complete semantic action choices and enforce J1/J2 stop/count rules.

    J1 consumes successive order or stop actions. J2 must begin with one count
    action, then exactly that many order actions; count zero is a complete empty
    portfolio. No action after a terminal state is accepted.
    """
    if interface not in {"J1", "J2"} or not isinstance(state, Mapping):
        raise ValueError("invalid replay interface or state")
    if not isinstance(actions, (tuple, list)):
        raise ValueError("actions must be a sequence")
    q = state.get("Q", state.get("own_q_units", state.get("own_q")))
    if type(q) is not int or not 0 <= q <= 400:
        raise ValueError("state Q must be an integer from 0 through 400")
    if interface == "J2":
        if not actions or actions[0].get("kind") != "count":
            raise ValueError("J2 must choose order count first")
        count = actions[0].get("count")
        if type(count) is not int or not 0 <= count <= 5:
            raise ValueError("J2 count must be from zero through five")
        if len(actions) != count + 1:
            raise ValueError("J2 order count must match the number of supplied orders")
        raw_orders = []
        for action in actions[1:]:
            if action.get("kind") != "order":
                raise ValueError("J2 requires an order after its count choice")
            raw_orders.append(action)
    else:
        raw_orders = []
        terminal = False
        for action in actions:
            if terminal:
                raise ValueError("J1 contains actions after STOP")
            if action.get("kind") == "stop":
                terminal = True
                continue
            if action.get("kind") != "order":
                raise ValueError("J1 expects an order or STOP choice")
            raw_orders.append(action)
            if len(raw_orders) == 5:
                terminal = True
        if not terminal:
            raise ValueError("J1 trace must stop explicitly or reach five orders")
    orders = validate_orders(raw_orders, initial_q=q)
    return {"status": "complete", "interface": interface, "orders": list(orders),
            "terminal_reason": "count_first" if interface == "J2" else "stop_or_five_orders"}


def validated_history_row(row: Mapping[str, Any], *, allow_profit: bool) -> dict[str, Any]:
    """Validate one prior episode and emit only approved observed fields."""
    if not isinstance(row, Mapping):
        raise ValueError("history episode must be a mapping")
    if any(key in row for key in ("fills", "executed_orders", "mcp_cents", "market_clearing_price")):
        raise ValueError("history may not use hidden matching or market outcome fields")
    round_no = row.get("round")
    p, q, d = row.get("p"), row.get("Q", row.get("q")), row.get("D", row.get("d"))
    if any(type(value) is not int for value in (round_no, p, q, d)):
        raise ValueError("history round, p, Q, and D must be integers")
    if not (1 <= p <= 999 and 0 <= q <= 400 and 0 <= d <= 200):
        raise ValueError("history state is outside the economic_v2 domain")
    orders = validate_orders(row.get("orders", []), initial_q=q)
    result: dict[str, Any] = {"round": round_no, "p": p, "Q": q, "D": d,
                              "orders": list(orders)}
    if allow_profit:
        profit = row.get("profit_cents")
        if type(profit) is not int:
            raise ValueError("profit feedback requires verified integer final_profit_cents")
        result["profit_cents"] = profit
    return result


def build_observation(*, state: Mapping[str, Any], history: Sequence[Mapping[str, Any]],
                      arm: str) -> dict[str, Any]:
    """Build a clean third-person input; no current-round outcome is exposed."""
    if arm not in {"A", "B", "C", "D", "E"}:
        raise ValueError("unknown arm")
    if any(key in state for key in ("profit_cents", "fills", "executed_orders", "mcp_cents")):
        raise ValueError("current target state may not include realized outcome fields")
    allow_profit = arm in {"C", "D", "E"}
    checked_history = [validated_history_row(row, allow_profit=allow_profit) for row in history]
    current = {key: state[key] for key in ("round", "p", "w", "Q", "D") if key in state}
    if set(current) != {"round", "p", "w", "Q", "D"}:
        raise ValueError("current state must include round,p,w,Q,D")
    result = {"horizon": 20, "round": current["round"] + 5, "market_size": 35,
        "own_p": current["p"], "wholesale_price_dollars": state.get("w"),
        "own_q_units": current["Q"], "own_d_units": current["D"],
        "phase": "orders", "decision_interface": "economic_v2",
        "own_pretrade_profit_dollars": f"{(current['p'] * min(current['D'], current['Q']) - state['w'] * current['Q']):.2f}",
        "history_window_n": len(checked_history),
        "history_feedback": "final_post_trade_profit_included" if allow_profit else "not_provided",
        "history": checked_history if arm != "A" else [],
        "history_fields": "round,p,Q,D,orders[slot,side,quantity_units,limit_price_dollars_per_unit],profit_dollars_optional",
        "partial_orders": [],
        "submitted_order_ledger": {"submitted_order_count": 0, "remaining_order_slots": 5,
            "submitted_buy_quantity_units": 0, "submitted_sell_quantity_units": 0,
            "remaining_sell_submission_capacity_units": current["Q"]},
        "public_instructions": {"trading": "Submit up to five mixed BUY/SELL limit orders; each quantity and integer-dollar price is 1..999.",
            "sell_cap": "Gross submitted SELL quantities cannot exceed initial Q; BUY quantities never increase this cap.",
            "stop": "J1 STOP ends additional submissions and preserves all earlier submitted orders."}}
    if arm == "E":
        result["protocol_interface"] = "J2"
    else:
        result["protocol_interface"] = "J1"
    for row in result["history"]:
        row["round"] += 5
        row["orders"] = [{"slot": index, "side": order["side"].upper(),
            "quantity_units": order["qty"], "limit_price_dollars_per_unit": order["price"]}
            for index, order in enumerate(row["orders"])]
        if "profit_cents" in row:
            cents = row.pop("profit_cents")
            row["profit_dollars"] = f"{'-' if cents < 0 else ''}{abs(cents) // 100}.{abs(cents) % 100:02d}"
    if "profit_cents" in result:
        raise AssertionError("current profit leaked into observation")
    return result


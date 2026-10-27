"""Pure, arm-specific state rendering for the five-arm history study.

The original DecisionRequest remains untouched: this module changes only the
effective SDK state. Arm identity and donor identifiers are study metadata and
are deliberately never inserted into that state.
"""
from __future__ import annotations

from copy import deepcopy
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping

ARMS = ("A", "B", "C", "D", "E")


def _integer_money(value: Any, name: str) -> int:
    """Accept an integer-valued dollar string without floating point rounding."""
    if type(value) is int:
        return value
    if isinstance(value, str):
        try:
            parsed = Decimal(value)
        except InvalidOperation as exc:
            raise ValueError(f"{name} must be an exact integer-dollar value") from exc
        if parsed.is_finite() and parsed == parsed.to_integral_value():
            return int(parsed)
    raise ValueError(f"{name} must be an exact integer-dollar value")


def _compact_history(history: Any) -> list[dict[str, Any]]:
    if not isinstance(history, (tuple, list)):
        raise ValueError("history must be a list")
    result = []
    for item in history:
        if isinstance(item, Mapping):
            round_no = item.get("round")
            p, q, d = item.get("p"), item.get("Q", item.get("q")), item.get("D", item.get("d"))
            orders = item.get("orders", item.get("submitted_orders"))
        elif isinstance(item, (tuple, list)) and len(item) == 8:
            # Legacy economic_v2 wire row: r,p,Q,D,MCP,profit,net,orders.
            # MCP/profit/net may be null placeholders; they are deliberately
            # discarded because those outcomes were not provided to the model.
            round_no, p, q, d, _mcp, _profit, _net, orders = item
        else:
            raise ValueError("history episodes must use the frozen economic_v2 row shape")
        if isinstance(p, str):
            p = _integer_money(p, "historical retail price")
        if any(type(x) is not int for x in (round_no, p, q, d)) or not isinstance(orders, (tuple, list)):
            raise ValueError("history episode is missing numeric state or submitted orders")
        named_orders = []
        for slot, order in enumerate(orders):
            if isinstance(order, Mapping):
                side = order.get("side", order.get("type"))
                qty = order.get("qty", order.get("amount"))
                price = order.get("limit_dollars", order.get("price"))
                slot = order.get("slot", slot)
            elif isinstance(order, (tuple, list)) and len(order) == 4:
                slot, side, qty, price = order
            else:
                raise ValueError("historical submitted order has an unknown wire shape")
            if isinstance(price, str):
                price = _integer_money(price, "historical order price")
            side = {"sell": "SELL", "buy": "BUY", "SELL": "SELL", "BUY": "BUY"}.get(side)
            if side is None or type(qty) is not int or type(price) is not int or type(slot) is not int:
                raise ValueError("historical submitted order is invalid")
            named_orders.append({"slot": slot, "side": side, "quantity_units": qty,
                                 "limit_price_dollars_per_unit": price})
        result.append({"game_round": round_no, "retail_price_dollars_per_unit": p,
            "initial_inventory_units": q, "realized_demand_units": d,
            "submitted_orders": named_orders,
            "outcomes": "not_provided_in_input"})
    return result


def semantic_history(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return the canonical semantic facts regardless of compact/named shape."""
    history = state.get("own_history", state.get("history", []))
    canonical = []
    for item in history:
        if "game_round" in item:
            orders = [{"slot": o["slot"], "side": o["side"].lower(), "qty": o["quantity_units"],
                       "limit_dollars": o["limit_price_dollars_per_unit"]}
                      for o in item["submitted_orders"]]
            canonical.append({"round": item["game_round"], "p": item["retail_price_dollars_per_unit"],
                "Q": item["initial_inventory_units"], "D": item["realized_demand_units"], "orders": orders})
        elif isinstance(item, Mapping):
            p = item["p"]
            if isinstance(p, str):
                p = _integer_money(p, "historical retail price")
            orders = []
            for slot, order in enumerate(item.get("orders", [])):
                side = order.get("side", order.get("type"))
                qty = order.get("qty", order.get("amount"))
                price = order.get("limit_dollars", order.get("price"))
                if isinstance(price, str):
                    price = _integer_money(price, "historical order price")
                orders.append({"slot": slot, "side": str(side).lower(), "qty": qty, "limit_dollars": price})
            canonical.append({"round": item["round"], "p": p,
                "Q": item.get("Q", item.get("q")), "D": item.get("D", item.get("d")), "orders": orders})
        elif isinstance(item, (tuple, list)) and len(item) == 8:
            round_no, p, q, d, _mcp, _profit, _net, source_orders = item
            orders = []
            for slot, order in enumerate(source_orders):
                order_slot, side, qty, price = order
                if isinstance(price, str):
                    price = _integer_money(price, "historical order price")
                orders.append({"slot": order_slot if type(order_slot) is int else slot,
                    "side": str(side).lower(), "qty": qty, "limit_dollars": price})
            if isinstance(p, str):
                p = _integer_money(p, "historical retail price")
            canonical.append({"round": round_no, "p": p, "Q": q, "D": d, "orders": orders})
        else:
            raise ValueError("history episode has an unknown wire shape")
    return canonical


def _ledger(state: Mapping[str, Any]) -> dict[str, int]:
    partial = state.get("partial_orders", [])
    if not isinstance(partial, (tuple, list)):
        raise ValueError("partial_orders must be a list")
    buys = sells = 0
    for order in partial:
        if isinstance(order, Mapping):
            side = order.get("side", order.get("type"))
            qty = order.get("qty", order.get("amount"))
        elif isinstance(order, (tuple, list)) and len(order) == 4:
            _slot, side, qty, _price = order
        else:
            raise ValueError("accepted partial order must use the frozen slot,side,qty,price shape")
        if side not in {"buy", "sell", "BUY", "SELL"} or type(qty) is not int or qty < 1:
            raise ValueError("accepted partial order is invalid")
        if str(side).lower() == "buy":
            buys += qty
        else:
            sells += qty
    initial_q = state.get("own_q_units", state.get("q", state.get("Q")))
    if type(initial_q) is not int or initial_q < 0 or sells > initial_q:
        raise ValueError("current inventory is invalid for submitted sell totals")
    slots = 5 - len(partial)
    if slots < 0:
        raise ValueError("more than five submitted orders")
    return {"submitted_order_count": len(partial), "remaining_order_slots": slots,
        "submitted_buy_quantity_units": buys, "submitted_sell_quantity_units": sells,
        "remaining_sell_submission_capacity_units": initial_q - sells}


def render_state(request: Any, arm: str, *, donor_history: Any = None) -> dict[str, Any]:
    """Render one of the five preregistered effective states, without mutating request."""
    if arm not in ARMS:
        raise ValueError("unknown history injection arm")
    original = request.observation
    if not isinstance(original, Mapping):
        raise ValueError("request observation must be a mapping")
    state = deepcopy(dict(original))
    if arm == "A":
        return state
    if arm in {"B", "C"}:
        history_key = "own_history" if "own_history" in original else "history"
        source_history = original.get(history_key, [])
    elif arm == "D":
        if donor_history is None:
            raise ValueError("donor history is required for arm D")
        source_history = donor_history
        # Preview records use formal-round labels 6..8. The existing market
        # request expresses the same experiences on the game clock 11..13.
        rounds = [row.get("round") if isinstance(row, Mapping) else row[0] for row in source_history]
        if rounds and all(type(r) is int and 6 <= r <= 8 for r in rounds):
            source_history = [({**dict(row), "round": row["round"] + 5}
                if isinstance(row, Mapping) else [row[0] + 5, *row[1:]]) for row in source_history]
    else:
        source_history = []
    history_key = "own_history" if "own_history" in original else "history"
    state[history_key] = _compact_history(source_history)
    state.pop("own_history_fields", None)
    if "history" in state and history_key != "history":
        state.pop("history", None)
    if arm in {"C", "D", "E"}:
        state["submitted_order_ledger"] = _ledger(original)
    return state

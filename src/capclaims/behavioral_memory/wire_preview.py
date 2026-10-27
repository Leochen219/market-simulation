"""Offline SDK-exact body previews for first and longest legal J2 requests."""
from __future__ import annotations

import base64
import hashlib
import json
from functools import lru_cache
from collections.abc import Mapping
from typing import Any

from capclaims.paper.codec import prefix_mask
from capclaims.paper.interface import build_candidates
from .design import build_prompt, build_schedule, stage_instructions
from .memory import compile_behavioral_memory
from .transport import serialize_body


def _order_state(base: Mapping[str, Any], orders: list[dict[str, Any]]) -> dict[str, Any]:
    named = [{"slot": i, "side": row["side"].upper(), "quantity_units": row["qty"],
              "limit_price_dollars_per_unit": row["price"]} for i, row in enumerate(orders)]
    buy = sum(row["qty"] for row in orders if row["side"] == "buy")
    sell = sum(row["qty"] for row in orders if row["side"] == "sell")
    return {"partial_orders": named, "submitted_order_ledger": {
        "submitted_order_count": len(orders), "remaining_order_slots": 5 - len(orders),
        "submitted_buy_quantity_units": buy, "submitted_sell_quantity_units": sell,
        "remaining_sell_submission_capacity_units": base["own_q_units"] - sell}}


def _instruction(stage: str, slot: int | None = None, side: str | None = None,
                 prefix: str = "") -> str:
    return stage_instructions(stage, slot=slot, side=side, prefix=prefix)


def _length(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8"))


def _wire_length(state: dict[str, Any], instructions: str, criteria: dict[str, str]) -> int:
    return _length({"state": state, "model": "jev-1.14.0", "questions": {
        "decision": {"type": "choice", "instructions": instructions, "criteria": criteria}}})


@lru_cache(maxsize=None)
def _sell_partition(count: int, total: int) -> tuple[int, ...] | None:
    """Positive quantities maximizing decimal byte length for a fixed SELL sum."""
    if count == 0:
        return () if total == 0 else None
    if total < count:
        return None
    if count == 1:
        return (total,)
    best = None
    score = -1
    for first in range(1, total - count + 2):
        tail = _sell_partition(count - 1, total - first)
        if tail is not None:
            candidate = (first,) + tail
            length = sum(len(str(q)) for q in candidate)
            if length > score:
                best, score = candidate, length
    return best


@lru_cache(maxsize=None)
def _numeric_max(field: str, side: str, capacity: int) -> tuple[dict[str, Any], str, dict[str, str]]:
    """Exhaust all legal digit prefixes; digit labels do not depend on prior orders."""
    bound = capacity if field == "quantity" and side == "sell" else 999
    best = None
    best_length = -1
    prefixes = [""] + [str(n) for n in range(10)] + [f"{n:02d}" for n in range(100)]
    for prefix in prefixes:
        digits = prefix_mask(prefix, 1, bound)
        if not digits:
            continue
        state = {"phase": "orders", "decision_field": field, "slot": 4, "own_side": side,
            "numeric_prefix": prefix, "choice_minimum": 1, "choice_maximum": bound,
            "decimal_place": ("h", "t", "u")[len(prefix)], "selected_order_count": 5,
            "remaining_planned_order_slots": 1, "decision_interface": "economic_v2"}
        if field == "price":
            state["own_qty_units"] = capacity if side == "sell" else 999
        criteria = {c.candidate_id: c.label for c in build_candidates(digits, state)}
        instructions = _instruction(field, slot=4, side=side, prefix=prefix)
        size = _wire_length(state, instructions, criteria)
        if size > best_length:
            best, best_length = (state, instructions, criteria), size
    assert best is not None
    return best


@lru_cache(maxsize=None)
def maximum_legal_request(initial_q: int) -> tuple[dict[str, Any], str, dict[str, str]]:
    """Exact longest legal subsequent request, independent of constant history bytes.

    Four previous orders dominate any shorter prefix: extra named-order bytes
    exceed possible changes in the few ledger integers. BUY=999 and price=999
    maximize integer widths. For each possible SELL total, dynamic programming
    maximizes widths of its positive quantity partition; all SELL counts 0..4
    are considered. For the current order all legal numeric prefixes are tried.
    Order permutation cannot change byte length. Model-visible base history is
    an additive constant, so this search can be cached by initial Q.
    """
    if type(initial_q) is not int or not 0 <= initial_q <= 400:
        raise ValueError("invalid initial Q for payload bound")
    base = {"own_q_units": initial_q, "decision_interface": "economic_v2"}
    best_by_sell = {}
    for sell_total in range(initial_q + 1):
        for sell_count in range(5):
            partition = _sell_partition(sell_count, sell_total)
            if partition is None:
                continue
            prior = [{"side": "sell", "qty": q, "price": 999} for q in partition]
            prior += [{"side": "buy", "qty": 999, "price": 999} for _ in range(4 - sell_count)]
            prefix_state = _order_state(base, prior)
            if sell_total not in best_by_sell or _length(prefix_state) > _length(best_by_sell[sell_total]):
                best_by_sell[sell_total] = prefix_state
    best, best_length = None, -1
    for sell_total, prior in best_by_sell.items():
        remaining = initial_q - sell_total
        side_state = {**base, **prior, "phase": "orders", "decision_field": "side", "slot": 4,
                      "selected_order_count": 5, "remaining_planned_order_slots": 1}
        sides = {"BUY": "BUY: submit a buy limit order"}
        if remaining:
            sides["SELL"] = "SELL: submit a sell limit order"
        options = [(side_state, _instruction("side", slot=4), sides)]
        for side in ("buy", "sell") if remaining else ("buy",):
            for field in ("quantity", "price"):
                numeric, instructions, criteria = _numeric_max(field, side, remaining if side == "sell" else 999)
                options.append(({**base, **prior, **numeric}, instructions, criteria))
        for state, instructions, criteria in options:
            size = _wire_length(state, instructions, criteria)
            if size > best_length:
                best, best_length = (state, instructions, criteria), size
    assert best is not None
    return best


async def build_wire_previews(cases: list[dict[str, Any]], *, seed: int = 20261026) -> dict[str, Any]:
    """Capture exact SDK body bytes offline, never using credentials or a live endpoint."""
    by_person = {case["person_id"]: case for case in cases}
    schedule = build_schedule(list(by_person), seed=seed)
    previews = {}
    for trajectory in schedule:
        case = by_person[trajectory["person_id"]]
        memory, _ = compile_behavioral_memory(case["history"], case["current_state"], 12)
        prompt = build_prompt(arm=trajectory["arm"], current_state=case["current_state"],
            history=memory["history"], memory=memory if trajectory["arm"] != "M0" else None)
        base = __import__("json").loads(prompt)["observation"]
        count_state = {**base, "phase": "orders", "decision_field": "order_count",
            **_order_state(base, []), "slots_remaining": 5}
        criteria = {f"N{n}": f"Submit exactly {n} orders" for n in range(6)}
        first = await serialize_body(state=count_state, instructions=_instruction("count"), criteria=criteria)
        maximum, max_instructions, max_criteria = maximum_legal_request(base["own_q_units"])
        max_state = {**base, **maximum}
        longest = await serialize_body(state=max_state, instructions=max_instructions, criteria=max_criteria)
        if len(first) != _wire_length(count_state, _instruction("count"), criteria) or len(longest) != _wire_length(
                max_state, max_instructions, max_criteria):
            raise ValueError("SDK wire encoding differs from exhaustive payload-size calculation")
        key = trajectory["request_prefix"]
        previews[key] = {kind: {"body_base64": base64.b64encode(body).decode("ascii"),
                "body_sha256": hashlib.sha256(body).hexdigest(), "body_bytes": len(body)}
            for kind, body in (("first_count", first), ("maximum_legal_prefix", longest))}
    return previews

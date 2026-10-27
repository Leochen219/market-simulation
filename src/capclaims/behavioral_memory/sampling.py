"""Sequential J2 sampling with arm-independent common random numbers."""
from __future__ import annotations

import hashlib
import inspect
import json
from collections.abc import Mapping
from typing import Any

from capclaims.backends.jev import normalize_jev_probabilities
from capclaims.paper.codec import prefix_mask
from capclaims.paper.interface import build_candidates
from capclaims.conditional_behavior.protocol import replay_choices, validate_orders

from .design import keyed_uniform, stage_instructions


def _draw(probabilities: Mapping[str, Any], u: float, candidate_order: list[str]) -> str:
    normalized, _ = normalize_jev_probabilities(dict(probabilities), candidate_order)
    cumulative = 0.0
    last = None
    for candidate in candidate_order:
        probability = normalized[candidate]
        if probability > 0:
            last = candidate
            cumulative += probability
            if u < cumulative:
                return candidate
    if last is None:
        raise ValueError("provider returned no positive candidate probability")
    return last


def _choice(response: Any, candidates: list[str]) -> Mapping[str, Any]:
    if not isinstance(response, Mapping) or response.get("model") != "jev-1.14.0":
        raise ValueError("provider response model is not the pinned Jev model")
    answer = response.get("answers", {}).get("decision")
    probs = answer.get("probabilities") if isinstance(answer, Mapping) else None
    if not isinstance(probs, Mapping) or set(probs) != set(candidates):
        raise ValueError("provider probabilities do not exactly match frozen candidates")
    return answer


async def sample_trajectory(client: Any, row: Mapping[str, Any], *, seed: int = 20261026,
                            on_step: Any = None, on_request: Any = None) -> dict[str, Any]:
    """Sample an entire J2 portfolio, preserving raw probabilities and realized prefixes."""
    person, target, repeat, arm = row["person_id"], row["target_round"], row["repeat"], row["arm"]
    if arm not in {"M0", "M1", "M2"} or target != 13 or repeat not in (0, 1):
        raise ValueError("frozen behavioral-memory trajectory coordinates are invalid")
    prompt_data = json.loads(row["prompt"])
    state0 = prompt_data["observation"]
    orders: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []

    def order_state() -> dict[str, Any]:
        buys = sum(o["qty"] for o in orders if o["side"] == "buy")
        sells = sum(o["qty"] for o in orders if o["side"] == "sell")
        named = [{"slot": i, "side": o["side"].upper(), "quantity_units": o["qty"],
                  "limit_price_dollars_per_unit": o["price"]} for i, o in enumerate(orders)]
        return {"partial_orders": named, "submitted_order_ledger": {
            "submitted_order_count": len(orders), "remaining_order_slots": 5 - len(orders),
            "submitted_buy_quantity_units": buys, "submitted_sell_quantity_units": sells,
            "remaining_sell_submission_capacity_units": state0["own_q_units"] - sells}}

    async def choose(coordinate: str, field: str, candidates: Mapping[str, str], *, slot: int | None = None,
                     side: str | None = None, prefix: str = "") -> str:
        ordered = list(candidates)
        u = keyed_uniform(seed=seed, person_id=person, target_round=target, repeat=repeat,
                          decision_coordinate=coordinate)
        state = {**state0, "phase": "orders", "decision_field": field, **order_state()}
        if field == "order_count":
            state["slots_remaining"] = 5
        if slot is not None:
            state["slot"] = slot
        if side is not None:
            state["own_side"] = side
        if field in {"quantity", "price"}:
            state["numeric_prefix"] = prefix
        if actions:
            state["selected_order_count"] = actions[0]["count"]
            state["remaining_planned_order_slots"] = actions[0]["count"] - len(orders)
        if field in {"quantity", "price"}:
            state["choice_minimum"] = 1
            state["choice_maximum"] = (999 if field == "price" or side == "buy" else
                state0["own_q_units"] - sum(o["qty"] for o in orders if o["side"] == "sell"))
            state["decimal_place"] = ("h", "t", "u")[len(prefix)]
        if field == "price":
            state["own_qty_units"] = qty
        rid = hashlib.sha256(f"{row['request_prefix']}\0{coordinate}".encode()).hexdigest()
        stage = "count" if field == "order_count" else field
        instructions = stage_instructions(stage, slot=slot, side=side, prefix=prefix)
        request_evidence = {"coordinate": coordinate, "request_id": rid, "state": state,
            "instructions": instructions, "criteria": dict(candidates), "candidate_order": ordered,
            "draw": u}
        if on_request is not None:
            saved_request = on_request(request_evidence)
            if inspect.isawaitable(saved_request):
                await saved_request
        response = await client.choose(request_id=rid, state=state, instructions=instructions,
                                       criteria=dict(candidates))
        answer = _choice(response, ordered)
        selected = _draw(answer["probabilities"], u, ordered)
        trace_item = {**request_evidence, "response": dict(response),
            "raw_probabilities": dict(answer["probabilities"]),
            "provider_choice": answer.get("choice"), "selected": selected}
        trace.append(trace_item)
        if on_step is not None:
            saved = on_step(trace_item)
            if inspect.isawaitable(saved):
                await saved
        return selected

    count_candidates = {f"N{n}": f"Submit exactly {n} orders" for n in range(6)}
    count_key = await choose("count", "order_count", count_candidates)
    count = int(count_key[1:])
    actions.append({"kind": "count", "count": count})
    for slot in range(count):
        sell_remaining = state0["own_q_units"] - sum(o["qty"] for o in orders if o["side"] == "sell")
        sides = {"BUY": "BUY: submit a buy limit order"}
        if sell_remaining >= 1:
            sides["SELL"] = "SELL: submit a sell limit order"
        side = (await choose(f"order:{slot}:side", "side", sides, slot=slot)).lower()
        for field, bound in (("quantity", 999 if side == "buy" else sell_remaining), ("price", 999)):
            prefix = ""
            for place, label in enumerate(("hundreds", "tens", "units")):
                minimum = 1
                maximum = bound
                digits = prefix_mask(prefix, minimum, maximum)
                numeric_state = {**state0, "phase": "orders", "decision_field": field, "slot": slot,
                    "own_side": side, "numeric_prefix": prefix, "choice_minimum": 1,
                    "choice_maximum": maximum, "decimal_place": ("h", "t", "u")[place],
                    **order_state(), "selected_order_count": count,
                    "remaining_planned_order_slots": count - len(orders)}
                if field == "price":
                    numeric_state["own_qty_units"] = qty
                candidates = {candidate.candidate_id: candidate.label
                    for candidate in build_candidates(digits, numeric_state)}
                digit_key = await choose(f"order:{slot}:{field}:{place}", field, candidates,
                    slot=slot, side=side, prefix=prefix)
                if not digit_key.startswith("digit") or not digit_key[5:].isdigit():
                    raise ValueError(f"numeric candidate id is malformed: {digit_key!r}; {list(candidates)}")
                prefix += digit_key[5:]
            value = int(prefix)
            if field == "quantity":
                qty = value
            else:
                price = value
        orders.append({"side": side, "qty": qty, "price": price})
        actions.append({"kind": "order", "side": side, "qty": qty, "price": price})
    checked = validate_orders(orders, initial_q=state0["own_q_units"])
    replay = replay_choices({"Q": state0["own_q_units"]}, actions, interface="J2")
    if list(checked) != replay["orders"] or len(trace) != 1 + 7 * count:
        raise ValueError("sampled trace fails exact portfolio replay")
    return {"status": "COMPLETE", "person_id": person, "target_round": target,
        "repeat": repeat, "arm": arm, "orders": list(checked), "actions": actions,
        "trace": trace, "physical_requests": len(trace)}

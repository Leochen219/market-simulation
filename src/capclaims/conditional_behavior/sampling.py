"""Sequential conditional sampling over frozen J1/J2 decision protocols."""
from __future__ import annotations

import hashlib
from typing import Any, Mapping

from capclaims.paper.interface import build_candidates
from capclaims.paper.codec import prefix_mask

from . import VERSION
from .prompts import stage_instruction
from .protocol import deterministic_draw, replay_choices, sample_categorical


def _order_state(base: Mapping[str, Any], orders: list[dict[str, Any]]) -> dict[str, Any]:
    named = [{"slot": index, "side": order["side"].upper(),
              "quantity_units": order["qty"],
              "limit_price_dollars_per_unit": order["price"]}
             for index, order in enumerate(orders)]
    buy = sum(order["qty"] for order in orders if order["side"] == "buy")
    sell = sum(order["qty"] for order in orders if order["side"] == "sell")
    ledger = {"submitted_order_count": len(orders), "remaining_order_slots": 5 - len(orders),
        "submitted_buy_quantity_units": buy, "submitted_sell_quantity_units": sell,
        "remaining_sell_submission_capacity_units": base["own_q_units"] - sell}
    return {"partial_orders": named, "submitted_order_ledger": ledger}


def _answer(response: Mapping[str, Any]) -> Mapping[str, Any]:
    if response.get("model") != "jev-1.14.0":
        raise ValueError("provider returned an unexpected model")
    answers = response.get("answers")
    decision = answers.get("decision") if isinstance(answers, Mapping) else None
    if not isinstance(decision, Mapping) or decision.get("type") != "choice":
        raise ValueError("provider response has an invalid choice shape")
    probabilities = decision.get("probabilities")
    if not isinstance(probabilities, Mapping):
        raise ValueError("provider choice has no probability mapping")
    return decision


async def sample_portfolio(client: Any, row: Mapping[str, Any], *, seed: int,
                           category: str = "human", oracle_order_count: int | None = None) -> dict[str, Any]:
    """Draw one full portfolio along its realized prefixes, with stable semantic seeds."""
    observation = row["observation"]
    interface = row["interface"]
    person, target, sample, arm = row["person_id"], row["target_round"], row["sample"], row["arm"]
    if interface != observation["protocol_interface"]:
        raise ValueError("frozen row interface differs from the state protocol")
    if category not in {"human", "synthetic"}:
        raise ValueError("unknown request category")
    if oracle_order_count is not None and (category != "synthetic" or type(oracle_order_count) is not int
            or not 0 <= oracle_order_count <= 5):
        raise ValueError("oracle order count is permitted only as an explicit synthetic diagnostic")
    orders: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    request_count = 0
    synthetic_history_note = None
    if category == "synthetic":
        history = observation.get("history", [])
        profit_n = sum(isinstance(item, Mapping) and "profit_dollars" in item for item in history)
        synthetic_history_note = (f"Synthetic diagnostic payload: this condition contains {len(history)} historical "
            f"example(s); final post-trade profit is present in {profit_n} of those examples. "
            "Use only the facts actually present in this payload.")

    async def choose(stage: str, stage_state: dict[str, Any], candidates: Mapping[str, str],
                     draw_stage: str, *, slot: int | None = None, side: str | None = None,
                     prefix: str = "") -> str:
        nonlocal request_count
        coordinate = f"{draw_stage}"
        draw = deterministic_draw(seed=seed, version=VERSION, arm=arm, person=person,
            target=target, sample=sample, stage=coordinate)
        request_id = hashlib.sha256(f"{row['case_id']}\0{coordinate}".encode()).hexdigest()
        instructions = stage_instruction(stage, interface=interface, arm=arm,
            slot=slot, side=side, prefix=prefix,
            synthetic_history_note=synthetic_history_note)
        response = await client.choose(request_id=request_id, state=stage_state,
            instructions=instructions, criteria=dict(candidates))
        answer = _answer(response)
        raw = answer["probabilities"]
        if set(raw) != set(candidates) or len(raw) != len(candidates):
            raise ValueError("provider probabilities must exactly match the candidate set")
        selected = sample_categorical(raw, draw, tuple(candidates))
        trace.append({"stage": coordinate, "request_id": request_id,
            "study_version": VERSION, "random_draw": draw,
            "state": stage_state, "instructions": instructions,
            "criteria": dict(candidates), "raw_response": dict(response),
            "candidate_order": list(candidates), "raw_probabilities": dict(raw),
            "provider_choice_ignored_for_draw": answer.get("choice"),
            "selected": selected, "model": response["model"],
            "usage": response.get("usage", {})})
        request_count += 1
        return selected

    base_state = dict(observation)
    if interface == "J2":
        criteria = {f"N{number}": f"Submit exactly {number} orders" for number in range(6)}
        state = {**base_state, "decision_field": "order_count", "phase": "orders",
            **_order_state(base_state, []), "slots_remaining": 5}
        if oracle_order_count is None:
            chosen = await choose("count", state, criteria, "count")
            count = int(chosen[1:])
        else:
            count = oracle_order_count
            trace.append({"stage": "count", "study_version": VERSION,
                "origin": "synthetic_oracle", "selected_count": count})
        actions.append({"kind": "count", "count": count})
    else:
        count = 5

    for slot in range(count):
        partial_state = _order_state(base_state, orders)
        partial = partial_state["partial_orders"]
        if interface == "J1":
            remaining_sell = base_state["own_q_units"] - sum(
                order["qty"] for order in orders if order["side"] == "sell")
            criteria = {"BUY": "BUY: submit a buy limit order",
                        "STOP": "STOP: submit no further orders and retain earlier orders"}
            if remaining_sell > 0:
                criteria["SELL"] = "SELL: submit a sell limit order"
            state = {**base_state, "phase": "orders", "decision_field": "side", "slot": slot,
                **partial_state}
            selected_side = await choose("side", state, criteria, f"order:{slot}:side", slot=slot)
            if selected_side == "STOP":
                actions.append({"kind": "stop"})
                break
            side = selected_side.lower()
        else:
            side = None
        if interface == "J2":
            remaining_sell = base_state["own_q_units"] - sum(
                order["qty"] for order in orders if order["side"] == "sell")
            side_criteria = {"BUY": "BUY: submit a buy limit order"}
            if remaining_sell > 0:
                side_criteria["SELL"] = "SELL: submit a sell limit order"
            state = {**base_state, "phase": "orders", "decision_field": "side", "slot": slot,
                **partial_state,
                "selected_order_count": count, "remaining_planned_order_slots": count - len(orders),
                }
            side = (await choose("side", state, side_criteria, f"order:{slot}:side", slot=slot)).lower()
        qty_prefix = ""
        qty_digits = []
        qty_max = 999 if side == "buy" else base_state["own_q_units"] - sum(
            order["qty"] for order in orders if order["side"] == "sell")
        if qty_max < 1:
            raise ValueError("no legal sell quantity remains, but a sell side was selected")
        for place in "htu":
            digits = prefix_mask(qty_prefix, 1, qty_max)
            stage_state = {**base_state, "phase": "orders", "decision_field": "quantity",
                "slot": slot, "own_side": side, "numeric_prefix": qty_prefix,
                "choice_minimum": 1, "choice_maximum": qty_max,
                "decimal_place": place, **partial_state}
            if interface == "J2":
                stage_state.update({"selected_order_count": count,
                    "remaining_planned_order_slots": count - len(orders)})
            candidates = build_candidates(digits, stage_state)
            criteria = {candidate.candidate_id: candidate.label for candidate in candidates}
            selected = await choose("quantity", stage_state, criteria,
                f"order:{slot}:quantity:{place}", slot=slot, side=side, prefix=qty_prefix)
            if not selected.startswith("digit") or not selected[5:].isdigit():
                raise ValueError(f"numeric candidate ID is malformed: {selected!r}")
            digit = int(selected[5:])
            qty_digits.append(digit)
            qty_prefix += str(digit)
        quantity = int(qty_prefix)
        price_prefix = ""
        price_digits = []
        for place in "htu":
            digits = prefix_mask(price_prefix, 1, 999)
            stage_state = {**base_state, "phase": "orders", "decision_field": "price",
                "slot": slot, "own_side": side, "own_qty_units": quantity,
                "numeric_prefix": price_prefix, "choice_minimum": 1,
                "choice_maximum": 999, "decimal_place": place,
                **partial_state}
            if interface == "J2":
                stage_state.update({"selected_order_count": count,
                    "remaining_planned_order_slots": count - len(orders)})
            candidates = build_candidates(digits, stage_state)
            criteria = {candidate.candidate_id: candidate.label for candidate in candidates}
            selected = await choose("price", stage_state, criteria,
                f"order:{slot}:price:{place}", slot=slot, side=side, prefix=price_prefix)
            if not selected.startswith("digit") or not selected[5:].isdigit():
                raise ValueError(f"numeric candidate ID is malformed: {selected!r}")
            digit = int(selected[5:])
            price_digits.append(digit)
            price_prefix += str(digit)
        order = {"side": side, "qty": quantity, "price": int(price_prefix)}
        orders.append(order)
        actions.append({"kind": "order", **order})
    if interface == "J1" and (not actions or actions[-1].get("kind") != "stop") and len(orders) < 5:
        # Reaching 5 is terminal; otherwise side sequence should have selected STOP.
        raise ValueError("J1 sampling ended before STOP or five submitted orders")
    replay = replay_choices({"Q": base_state["own_q_units"]}, actions, interface=interface)
    if replay["orders"] != orders:
        raise ValueError("sampled choices differ from strict protocol replay")
    return {"orders": orders, "provider_requests": request_count,
        "trace": trace, "raw_wires": [], "category": category}

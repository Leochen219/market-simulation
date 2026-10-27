"""Pure, replayable transitions for forecasting one human's complete portfolio."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import math
from typing import Any, Mapping, Sequence

from capclaims.domain.types import Candidate
from capclaims.engine.protocol import DecisionReply, DecisionRequest, validate_reply
from capclaims.paper.codec import decode_prefix, prefix_mask
from capclaims.paper.interface import build_candidates, numeric_observation
from capclaims.paper.observations import add_order_choice_context, build_paper_observation
from capclaims.paper.orders import PaperOrder, validate_portfolio
from capclaims.randomness import canonical_json

_BASE_FIELDS = {"agent_id", "round", "treatment", "wholesale_price_dollars", "p", "Q", "D", "history"}
_HISTORY_FIELDS = {"round", "p", "Q", "D", "orders"}
_PLACE = ("h", "t", "u")
MAX_ORDERS = 5
MARKET_N = 35


def _sha(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _strict_int(value: Any, low: int, high: int, name: str) -> int:
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer from {low} through {high}")
    return value


def _normalize_historical_order(row: Mapping[str, Any], slot: int) -> dict[str, Any]:
    side = row.get("side", row.get("type"))
    if side == "BUY":
        side = "buy"
    elif side == "SELL":
        side = "sell"
    qty = row.get("qty", row.get("amount"))
    price = row.get("limit_dollars", row.get("price"))
    return {"slot": slot, "side": side, "qty": qty, "limit_dollars": price}


def _validate_base(base: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(base, Mapping) or set(base) != _BASE_FIELDS:
        raise ValueError("portfolio base has an unknown or missing field")
    agent = base["agent_id"]
    if not isinstance(agent, str) or not agent:
        raise ValueError("portfolio base requires an opaque nonempty agent_id")
    round_no = _strict_int(base["round"], 9, 10, "round")
    if base["treatment"] != "GA":
        raise ValueError("portfolio prediction supports the GA treatment only")
    wholesale = _strict_int(base["wholesale_price_dollars"], 10, 10, "wholesale price")
    p = _strict_int(base["p"], 51, 100, "retail value")
    q = _strict_int(base["Q"], 0, 400, "initial procurement")
    demand = _strict_int(base["D"], 0, 200, "own demand")
    history = base["history"]
    if not isinstance(history, (tuple, list)) or len(history) > 3:
        raise ValueError("portfolio history must contain at most three prior examples")
    normalized_history = []
    previous_round = 0
    for row in history:
        if not isinstance(row, Mapping) or set(row) != _HISTORY_FIELDS:
            raise ValueError("historical example must contain only round,p,Q,D,orders")
        hist_round = _strict_int(row["round"], 1, 8, "history round")
        if hist_round <= previous_round or hist_round >= round_no:
            raise ValueError("history rounds must be strictly ordered before the held-out target")
        previous_round = hist_round
        hp = _strict_int(row["p"], 51, 100, "historical retail value")
        hq = _strict_int(row["Q"], 0, 400, "historical procurement")
        hd = _strict_int(row["D"], 0, 200, "historical demand")
        if not isinstance(row["orders"], (tuple, list)) or len(row["orders"]) > MAX_ORDERS:
            raise ValueError("historical portfolio must contain at most five orders")
        if any(not isinstance(order, Mapping) for order in row["orders"]):
            raise ValueError("historical orders must be mappings")
        normalized_orders = [_normalize_historical_order(order, i) for i, order in enumerate(row["orders"])]
        hist_orders = tuple(PaperOrder(f"history{hist_round}:slot{i}", agent,
            order["side"], order["qty"], order["limit_dollars"])
            for i, order in enumerate(normalized_orders))
        validate_portfolio(hist_orders, agent, hq)
        normalized_history.append({"round": hist_round, "p": hp, "Q": hq, "D": hd,
            "orders": [asdict(order) for order in hist_orders]})
    return {"agent_id": agent, "round": round_no, "treatment": "GA",
        "wholesale_price_dollars": wholesale, "p": p, "Q": q, "D": demand,
        "history": normalized_history}


def _new_trace_state() -> dict[str, Any]:
    return {"orders": [], "pending_side": None, "pending_qty": None,
        "numeric_field": None, "digits": [], "terminal": False, "failure_reason": None}


def _request_for(base: Mapping[str, Any], state: Mapping[str, Any]) -> DecisionRequest | None:
    if state["terminal"] or state["failure_reason"]:
        return None
    orders = state["orders"]
    slot = len(orders)
    if slot >= MAX_ORDERS:
        return None
    pending_side = state["pending_side"]
    partial = [asdict(order) for order in orders]
    # Workbook rounds 1..15 are formal rounds after five warmups in the paper's
    # 20-round game; preserve the actual game clock on every model-visible state.
    model_round = base["round"] + 5
    model_history = [{**row, "round": row["round"] + 5} for row in base["history"]]
    common = dict(agent_id=base["agent_id"], round_index=model_round - 1,
        treatment="GA", n_agents=MARKET_N, wholesale_price=base["wholesale_price_dollars"],
        p=base["p"], history=model_history, partial_orders=partial, phase="orders",
        q=base["Q"], d=base["D"], slot=slot, horizon=20, history_window=3,
        equality_crossing="ge", self_match_policy="unknown", decision_interface="economic_v2")
    if pending_side is None:
        sell_remaining = base["Q"] - sum(order.qty for order in orders if order.side == "sell")
        legal = ("buy", "STOP", "sell") if sell_remaining > 0 else ("buy", "STOP")
        observation = build_paper_observation(**common, decision_field="side")
        _sanitize_forecast_observation(observation)
        observation["legal_actions"] = list(legal)
        candidates = build_candidates(legal, observation)
        return DecisionRequest(base["agent_id"], model_round - 1,
            f"paper:slot{slot}:side", observation, tuple(candidates))
    field = state["numeric_field"]
    if field not in {"quantity", "price"}:
        raise AssertionError("active side requires a quantity or price field")
    side = pending_side
    own_qty = state["pending_qty"] if field == "price" else None
    if field == "quantity":
        maximum = base["Q"] - sum(order.qty for order in orders if order.side == "sell") if side == "sell" else 999
    else:
        maximum = 999
    prefix = "".join(str(digit) for digit in state["digits"])
    place = _PLACE[len(prefix)]
    observation = build_paper_observation(**common, decision_field=field, own_side=side, own_qty=own_qty,
        prefix=prefix, minimum=1, maximum=maximum)
    observation = add_order_choice_context(observation, decision_field=field, own_side=side, own_qty=own_qty)
    observation = numeric_observation(observation, prefix, 1, maximum, place)
    _sanitize_forecast_observation(observation)
    mask = prefix_mask(prefix, 1, maximum)
    candidates = build_candidates(mask, observation)
    return DecisionRequest(base["agent_id"], model_round - 1,
        f"paper:slot{slot}:{'qty' if field == 'quantity' else 'price'}:{place}",
        observation, tuple(candidates))


def _sanitize_forecast_observation(observation: dict[str, Any]) -> None:
    """Keep the human-visible accounting projection while neutralizing optimizer-only claims."""
    instructions = observation.get("public_instructions")
    if isinstance(instructions, dict):
        instructions["clearing"] = (
            "Orders are blind and clear by price priority. A match may be partial or zero; matched trades "
            "use the uniform midpoint of the last matched bid and ask. Exact tie allocation and self-match "
            "handling are not established."
        )
        instructions["profit"] = (
            "Accounting identity: profit equals sales revenue minus procurement cost and net market "
            "purchase spending; surplus inventory expires. This is not an instruction to optimize profit."
        )
    observation.pop("self_match_policy", None)
    if any("maximize" in str(value).lower() for value in (instructions or {}).values()):
        raise ValueError("portfolio forecast observation contains optimizer wording")


def _current(base: Mapping[str, Any], state: Mapping[str, Any]) -> dict[str, Any]:
    if state["terminal"]:
        return {"kind": "complete"}
    if state["failure_reason"]:
        return {"kind": "incomplete", "reason": state["failure_reason"]}
    request = _request_for(base, state)
    if request is None:
        return {"kind": "complete"}
    candidates = request.candidates
    if len(candidates) == 1:
        return {"kind": "forced", "request": request, "candidate": candidates[0]}
    return {"kind": "request", "request": request}


def _binding(request: DecisionRequest) -> dict[str, Any]:
    return {"request_id": request.request_id, "stage": request.stage,
        "observation_hash": request.observation_hash,
        "candidate_set_hash": request.candidate_set_hash,
        "candidates": [{"candidate_id": c.candidate_id, "value": c.value, "label": c.label}
            for c in request.candidates]}


def _apply_choice(base: Mapping[str, Any], state: dict[str, Any], request: DecisionRequest,
                  value: Any) -> None:
    if request.stage.endswith(":side"):
        if value == "STOP":
            state["terminal"] = True
            return
        if value not in {"buy", "sell"}:
            raise ValueError("invalid side choice")
        if value == "sell" and base["Q"] - sum(o.qty for o in state["orders"] if o.side == "sell") <= 0:
            raise ValueError("sell side is unavailable after gross sell capacity is exhausted")
        state["pending_side"] = value
        state["numeric_field"] = "quantity"
        state["digits"] = []
        return
    if type(value) is not int or value not in {candidate.value for candidate in request.candidates}:
        raise ValueError("numeric choice is outside the current legal prefix mask")
    state["digits"].append(value)
    if len(state["digits"]) < 3:
        return
    decoded = decode_prefix(tuple(state["digits"]))
    field = state["numeric_field"]
    if field == "quantity":
        maximum = base["Q"] - sum(o.qty for o in state["orders"] if o.side == "sell") if state["pending_side"] == "sell" else 999
        if not 1 <= decoded <= maximum:
            raise ValueError("completed order quantity violates the current gross sell capacity")
        state["pending_qty"] = decoded
        state["numeric_field"] = "price"
        state["digits"] = []
        return
    if field != "price" or state["pending_qty"] is None:
        raise AssertionError("price digits require a pending quantity")
    slot = len(state["orders"])
    order = PaperOrder(f"r{base['round']:03d}:{base['agent_id']}:slot{slot}",
        base["agent_id"], state["pending_side"], state["pending_qty"], decoded)
    validate_portfolio(tuple(state["orders"] + [order]), base["agent_id"], base["Q"])
    state["orders"].append(order)
    state["pending_side"] = None
    state["pending_qty"] = None
    state["numeric_field"] = None
    state["digits"] = []
    if len(state["orders"]) == MAX_ORDERS:
        state["terminal"] = True


def _decode_trace(base: Mapping[str, Any], trace: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    checked = _validate_base(base)
    state = _new_trace_state()
    for event in trace:
        current = _current(checked, state)
        if current["kind"] in {"complete", "incomplete"}:
            raise ValueError("trace contains events after its terminal or failed state")
        request = current["request"]
        if event.get("binding") != _binding(request):
            raise ValueError("trace request binding differs from reconstructed own state")
        kind = current["kind"]
        if kind == "forced":
            candidate = current["candidate"]
            if (event.get("kind") != "forced" or event.get("origin") != "constraint"
                    or event.get("candidate_id") != candidate.candidate_id
                    or event.get("value") != candidate.value):
                raise ValueError("trace omitted or changed a singleton constraint")
            _apply_choice(checked, state, request, candidate.value)
            continue
        if event.get("kind") == "failure":
            reason = event.get("reason")
            if reason not in {"provider_failure", "invalid_reply", "unresolved_tie"}:
                raise ValueError("trace has an unknown failure reason")
            state["failure_reason"] = reason
            continue
        if event.get("kind") != "choice" or event.get("origin") != "model":
            raise ValueError("trace requires a model choice for the current request")
        probabilities, raw_scores = event.get("probabilities"), event.get("raw_scores")
        if not isinstance(probabilities, Mapping) or not isinstance(raw_scores, Mapping):
            raise ValueError("trace choice is missing its raw probability evidence")
        reply = DecisionReply.from_request(request, probabilities, raw_scores,
            backend=str(event.get("backend", "trace")), model_revision=str(event.get("model_revision", "trace")))
        validate_reply(request, reply)
        best = max(reply.probabilities.values())
        winners = [candidate for candidate in request.candidates if reply.probabilities[candidate.candidate_id] == best]
        if len(winners) != 1:
            raise ValueError("model-choice trace must not resolve a top-probability tie")
        winner = winners[0]
        if event.get("candidate_id") != winner.candidate_id or event.get("value") != winner.value:
            raise ValueError("trace choice does not match the unique maximum-probability candidate")
        _apply_choice(checked, state, request, winner.value)
    return state, _current(checked, state)


def next_trace_action(base: Mapping[str, Any], trace: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Describe whether a model request, deterministic constraint, or terminal state is next."""
    state, current = _decode_trace(base, trace)
    if current["kind"] in {"request", "forced"}:
        result = {"kind": current["kind"], "binding": _binding(current["request"])}
        if current["kind"] == "forced":
            result["value"] = current["candidate"].value
        return result
    return {"kind": current["kind"], "reason": current.get("reason"),
        "completed_order_n": len(state["orders"])}


def build_order_request(base: Mapping[str, Any], trace: Sequence[Mapping[str, Any]]) -> DecisionRequest | None:
    _, current = _decode_trace(base, trace)
    return current["request"] if current["kind"] == "request" else None


def advance_trace(base: Mapping[str, Any], trace: Sequence[Mapping[str, Any]],
                  reply: DecisionReply | None) -> tuple[dict[str, Any], ...]:
    checked = _validate_base(base)
    state, current = _decode_trace(checked, trace)
    if current["kind"] in {"complete", "incomplete"}:
        raise ValueError("cannot advance a complete or failed portfolio trace")
    request = current["request"]
    binding = _binding(request)
    new_event: dict[str, Any]
    if current["kind"] == "forced":
        if reply is not None:
            raise ValueError("singleton constraints do not issue model calls")
        candidate = current["candidate"]
        new_event = {"kind": "forced", "origin": "constraint", "binding": binding,
            "candidate_id": candidate.candidate_id, "value": candidate.value}
    elif reply is None:
        new_event = {"kind": "failure", "reason": "provider_failure", "origin": "provider", "binding": binding}
    else:
        try:
            validate_reply(request, reply)
            best = max(reply.probabilities.values())
            winners = [candidate for candidate in request.candidates
                if reply.probabilities[candidate.candidate_id] == best]
            if len(winners) != 1:
                new_event = {"kind": "failure", "reason": "unresolved_tie", "origin": "model", "binding": binding}
            else:
                candidate = winners[0]
                new_event = {"kind": "choice", "origin": "model", "binding": binding,
                    "candidate_id": candidate.candidate_id, "value": candidate.value,
                    "probabilities": dict(reply.probabilities), "raw_scores": dict(reply.raw_scores),
                    "backend": reply.backend, "model_revision": reply.model_revision}
        except Exception:
            new_event = {"kind": "failure", "reason": "invalid_reply", "origin": "model", "binding": binding}
    return tuple([*trace, new_event])


def replay_trace(base: Mapping[str, Any], events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    state, current = _decode_trace(base, events)
    complete = current["kind"] == "complete"
    orders = tuple(state["orders"]) if complete else ()
    return {"status": "complete" if complete else "incomplete",
        "reason": state["failure_reason"] or (None if complete else "truncated_trace"),
        "orders": orders, "completed_order_n": len(state["orders"]),
        "terminal_reason": "stop_or_five_orders" if complete else None}


def final_portfolio(base: Mapping[str, Any], trace: Sequence[Mapping[str, Any]]) -> tuple[PaperOrder, ...]:
    result = replay_trace(base, trace)
    if result["status"] != "complete":
        raise ValueError(f"portfolio prediction is incomplete: {result['reason']}")
    return result["orders"]

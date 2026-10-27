"""Frozen historical empirical distributions and donor matching."""
from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from typing import Any

from .protocol import validate_orders


def _canonical_orders(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    source = row.get("orders", [])
    normalized = []
    for order in source:
        if "quantity_units" in order:
            normalized.append({"side": order["side"].lower(), "qty": order["quantity_units"],
                "price": order["limit_price_dollars_per_unit"]})
        else:
            normalized.append({"side": order.get("side", order.get("type", "")).lower(),
                "qty": order.get("qty", order.get("amount")),
                "price": order.get("price", order.get("limit_dollars"))})
    q = row.get("Q", row.get("initial_inventory_units", row.get("own_q_units")))
    return list(validate_orders(normalized, initial_q=int(q)))


def _tie(seed: int, *parts: Any) -> str:
    import json
    return hashlib.sha256(json.dumps([seed, *parts], sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def select_donor(target_history: Sequence[Mapping[str, Any]], donors: Sequence[Mapping[str, Any]], *,
                 seed: int, target_id: str) -> dict[str, Any]:
    if (not isinstance(target_history, Sequence) or len(target_history) != 8
            or [row.get("round") for row in target_history] != list(range(6, 14)) or not donors):
        raise ValueError("matching requires target and donor histories for rounds 6 through 13")
    ranked = []
    for donor in donors:
        history = donor.get("history", [])
        if len(history) != 8 or [row.get("round") for row in history] != list(range(6, 14)):
            continue
        distance = math.fsum(abs(left[axis] - right[axis]) / scale
            for left, right in zip(target_history, history)
            for axis, scale in (("p", 49), ("Q", 400), ("D", 200))) / 8
        tie = _tie(seed, target_id, donor["person_id"])
        ranked.append((distance, tie, donor))
    if not ranked:
        raise ValueError("donor pool has no candidate history")
    distance, tie, donor = min(ranked, key=lambda item: (item[0], item[1]))
    return {"person_id": donor["person_id"], "distance": distance, "tie_sha256": tie,
            "history": [dict(row) for row in donor["history"]]}


def _legal_rows(rows: Sequence[Mapping[str, Any]], q: int) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        try:
            orders = _canonical_orders(row)
        except (ValueError, TypeError, KeyError):
            continue
        if sum(order["qty"] for order in orders if order["side"] == "sell") <= q:
            result.append({**dict(row), "orders": orders})
    return result


def baseline_distributions(*, target: Mapping[str, Any], target_id: str,
                           public_history_pool: Sequence[Mapping[str, Any]],
                           personal_history: Sequence[Mapping[str, Any]], seed: int) -> dict[str, dict[str, Any]]:
    """Return equal-weight legal records for population, state, and personal baselines."""
    q = target["Q"]
    pool = _legal_rows(public_history_pool, q)
    state_ranked = []
    for row in pool:
        distance = abs(row["p"] - target["p"]) / 49 + abs(row["Q"] - q) / 400 + abs(row["D"] - target["D"]) / 200
        tie = _tie(seed, target_id, row.get("person_id"), row.get("round"))
        state_ranked.append((distance, -int(row.get("round", 0)), tie, row))
    nearest = [row for _, _, _, row in sorted(state_ranked, key=lambda item: item[:3])[:5]]
    personal = _legal_rows(personal_history, q)
    empty = [{"round": 0, "orders": []}]
    return {
        "population": {"records": pool or empty, "source_n": len(pool), "fallback_empty": not pool},
        "state": {"records": nearest or empty, "source_n": len(nearest), "fallback_empty": not nearest},
        "personal": {"records": personal or empty, "source_n": len(personal), "fallback_empty": not personal},
    }


def expected_fair_score(records: Sequence[Mapping[str, Any]], truth: Any,
                        point_score: Any, distance: Any, *, empty_fallback: bool = False) -> float:
    """Exact fair energy/CRPS expectation; EMPTY support is an ordinary point loss."""
    if not records:
        raise ValueError("baseline records are required")
    values = [point_score(record["orders"]) for record in records]
    if empty_fallback:
        return distance(values[0], truth)
    if len(values) == 1:
        return distance(values[0], truth)
    return math.fsum(distance(value, truth) for value in values) / len(values) - math.fsum(
        distance(values[i], values[j]) for i in range(len(values))
        for j in range(len(values))) / (2 * len(values) * len(values))


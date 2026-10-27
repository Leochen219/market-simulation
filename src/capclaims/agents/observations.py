"""Construct phase-specific, private observations from caller-owned history."""

from collections.abc import Mapping, Sequence
from typing import Any

from capclaims.config import RunConfig

HISTORY_FIELDS = frozenset({"p", "D", "Q", "A", "profit_cents", "m_cents", "fillqty", "side", "qty", "price"})


def build_observation(
    config: RunConfig, round_index: int, agent_id: str, p: int,
    history: Sequence[Mapping[str, Any]], *, phase: str,
    q: int | None = None, d: int | None = None, side: str | None = None, qty: int | None = None,
) -> dict[str, Any]:
    if phase not in {"purchase", "trade_side", "trade_qty", "trade_price"}:
        raise ValueError("unknown observation phase")
    if isinstance(round_index, bool) or not isinstance(round_index, int) or round_index < 0:
        raise ValueError("round_index must be a nonnegative integer")
    if not isinstance(agent_id, str) or not agent_id:
        raise ValueError("agent_id must be a nonempty string")
    if isinstance(p, bool) or not isinstance(p, int) or not 51 <= p <= 100:
        raise ValueError("p must be an integer from 51 to 100")
    own_history = [{k: item[k] for k in HISTORY_FIELDS if k in item} for item in history[-5:]]
    obs: dict[str, Any] = {
        "n_agents": config.n_agents,
        "w": config.wholesale_price,
        "market_open": config.market_open,
        "round_index": round_index,
        "horizon": config.rounds,
        "own_p": p,
        "mechanism_version": config.mechanism_version,
        "rules": {
            "purchase_grid": list(range(0, 401, 25)),
            "demand_range": [0, 200],
            "valuation_range": [51, 100],
            "trade_price_grid": list(range(0, 101, 10)),
            "monetary_units": "dollars_per_capacity_unit; history profit_cents and m_cents are cents",
        },
        "history": own_history,
    }
    if phase != "purchase":
        if q is None or d is None or isinstance(q, bool) or isinstance(d, bool) or not isinstance(q, int) or not isinstance(d, int) or not 0 <= q <= 400 or not 0 <= d <= 200:
            raise ValueError("trade observations require valid own q and d")
        obs["own_q"] = q
        obs["own_d"] = d
    if phase in {"trade_qty", "trade_price"}:
        if side not in {"buy", "sell"}:
            raise ValueError("quantity and price observations require own buy/sell side")
        obs["own_side"] = side
    if phase == "trade_price":
        if isinstance(qty, bool) or not isinstance(qty, int) or not 1 <= qty <= 400:
            raise ValueError("price observation requires a valid own quantity")
        obs["own_qty"] = qty
    return obs

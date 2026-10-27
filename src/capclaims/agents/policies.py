"""Reproducible baseline purchase and trading policies."""

from collections.abc import Sequence

from capclaims.agents.candidates import PRICE_GRID, PURCHASE_GRID, QUANTITY_GRID
from capclaims.config import RunConfig
from capclaims.domain.types import TradeAction
from capclaims.randomness import rng_for_policy


def optimal_quantity(p: int, w: int, grid: Sequence[int] = PURCHASE_GRID) -> int:
    if isinstance(p, bool) or not isinstance(p, int) or not 51 <= p <= 100:
        raise ValueError("p must be an integer from 51 to 100")
    if isinstance(w, bool) or not isinstance(w, int) or w not in (10, 50):
        raise ValueError("w must be 10 or 50")
    legal = tuple(grid)
    if not legal or any(isinstance(q, bool) or not isinstance(q, int) or not 0 <= q <= 400 for q in legal):
        raise ValueError("grid must contain legal quantities from 0 to 400")
    def numerator(qty: int) -> int:
        return p * sum(min(d, qty) for d in range(201)) - 201 * w * qty
    return max(legal, key=lambda qty: (numerator(qty), -qty))


def _random_trade(rng, q: int) -> TradeAction:
    side = rng.choice(("buy", "sell", "hold") if q else ("buy", "hold"))
    if side == "hold":
        return TradeAction()
    legal_qty = tuple(n for n in QUANTITY_GRID if n <= (400 if side == "buy" else q))
    if not legal_qty:
        return TradeAction()
    qty = rng.choice(legal_qty)
    return TradeAction(side, qty, rng.choice(PRICE_GRID) * 100)


def _rule_trade(p: int, q: int, d: int) -> TradeAction:
    if q > d:
        legal_qty = [n for n in QUANTITY_GRID if n <= q - d]
        return TradeAction("sell", max(legal_qty), 0) if legal_qty else TradeAction()
    if d > q:
        legal_qty = [n for n in QUANTITY_GRID if n <= d - q]
        legal_prices = [price for price in PRICE_GRID if price <= p]
        return TradeAction("buy", max(legal_qty), max(legal_prices) * 100) if legal_qty else TradeAction()
    return TradeAction()


def _ensure_baseline(config: RunConfig) -> None:
    if config.policy not in {"random", "grid_rule", "noisy_rule"}:
        raise ValueError(f"{config.policy} is a backend policy and has no local baseline helper")


def purchase_policy(config: RunConfig, round_index: int, agent_id: str, p: int) -> int:
    _ensure_baseline(config)
    if isinstance(round_index, bool) or not isinstance(round_index, int) or round_index < 0:
        raise ValueError("round_index must be a nonnegative integer")
    if not isinstance(agent_id, str) or not agent_id:
        raise ValueError("agent_id must be a nonempty string")
    if config.policy == "grid_rule":
        return optimal_quantity(p, config.wholesale_price)
    rng = rng_for_policy(config.master_seed, config.seed_index, config.policy, round_index, agent_id, "purchase", "choice")
    if config.policy == "random":
        return rng.choice(PURCHASE_GRID)
    if rng.random() < 0.2:
        return rng.choice(PURCHASE_GRID)
    return optimal_quantity(p, config.wholesale_price)


def trade_policy(config: RunConfig, round_index: int, agent_id: str, p: int, q: int, d: int) -> TradeAction:
    _ensure_baseline(config)
    if isinstance(round_index, bool) or not isinstance(round_index, int) or round_index < 0:
        raise ValueError("round_index must be a nonnegative integer")
    if not isinstance(agent_id, str) or not agent_id:
        raise ValueError("agent_id must be a nonempty string")
    if isinstance(p, bool) or not isinstance(p, int) or not 51 <= p <= 100:
        raise ValueError("p must be an integer from 51 to 100")
    if isinstance(q, bool) or not isinstance(q, int) or not 0 <= q <= 400:
        raise ValueError("q must be an integer from 0 to 400")
    if isinstance(d, bool) or not isinstance(d, int) or not 0 <= d <= 200:
        raise ValueError("d must be an integer from 0 to 200")
    if config.policy == "grid_rule":
        return _rule_trade(p, q, d)
    rng = rng_for_policy(config.master_seed, config.seed_index, config.policy, round_index, agent_id, "trade", "action")
    if config.policy == "random" or rng.random() < 0.2:
        return _random_trade(rng, q)
    return _rule_trade(p, q, d)

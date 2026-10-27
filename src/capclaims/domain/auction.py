"""Deterministic single-order uniform-price double auction."""

from collections import defaultdict
from collections.abc import Sequence

from capclaims.domain.types import AuctionResult, Order, Trade
from capclaims.randomness import auction_rank


def _valid_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")


def clear_auction(
    orders: Sequence[Order], master_seed: int = 0, seed_index: int = 0, round_index: int = 0,
) -> AuctionResult:
    by_id: dict[str, Order] = {}
    for order in orders:
        if not isinstance(order.agent_id, str) or not order.agent_id:
            raise ValueError("agent_id must be a nonempty string")
        if order.agent_id in by_id:
            raise ValueError(f"duplicate order for agent {order.agent_id}")
        if order.side not in {"buy", "sell"}:
            raise ValueError("order side must be buy or sell")
        _valid_integer("qty", order.qty)
        _valid_integer("limit_cents", order.limit_cents)
        if not 1 <= order.qty <= 400:
            raise ValueError("qty must be between 1 and 400")
        if not 0 <= order.limit_cents <= 10_000 or order.limit_cents % 2:
            raise ValueError("limit_cents must be an even integer from 0 to 10000")
        by_id[order.agent_id] = order

    ranks = {agent_id: auction_rank(master_seed, seed_index, round_index, agent_id) for agent_id in by_id}
    buys = sorted((o for o in orders if o.side == "buy"), key=lambda o: (-o.limit_cents, ranks[o.agent_id]))
    sells = sorted((o for o in orders if o.side == "sell"), key=lambda o: (o.limit_cents, ranks[o.agent_id]))
    remaining_buy = [o.qty for o in buys]
    remaining_sell = [o.qty for o in sells]
    fills: list[tuple[str, str, int, int, int]] = []
    bi = si = 0
    while bi < len(buys) and si < len(sells) and buys[bi].limit_cents >= sells[si].limit_cents:
        qty = min(remaining_buy[bi], remaining_sell[si])
        fills.append((buys[bi].agent_id, sells[si].agent_id, qty, buys[bi].limit_cents, sells[si].limit_cents))
        remaining_buy[bi] -= qty
        remaining_sell[si] -= qty
        if remaining_buy[bi] == 0:
            bi += 1
        if remaining_sell[si] == 0:
            si += 1
    if not fills:
        return AuctionResult(None, (), {agent_id: 0 for agent_id in by_id}, ranks)

    last_bid, last_ask = fills[-1][3], fills[-1][4]
    price = (last_bid + last_ask) // 2
    allocations: dict[str, int] = defaultdict(int)
    trades = []
    for buyer, seller, qty, _, _ in fills:
        trades.append(Trade(buyer, seller, qty, price))
        allocations[buyer] += qty
        allocations[seller] -= qty
    return AuctionResult(price, tuple(trades), {agent_id: allocations[agent_id] for agent_id in by_id}, ranks)

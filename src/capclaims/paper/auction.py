"""Multi-order, order-ID keyed uniform-price auction."""
from dataclasses import dataclass
from collections import defaultdict
from capclaims.randomness import hash_int
from capclaims.paper.orders import PaperOrder, validate_portfolio


@dataclass(frozen=True)
class PaperTrade:
    buyer_id: str
    seller_id: str
    qty: int
    price_cents: int
    buyer_order_id: str
    seller_order_id: str


@dataclass(frozen=True)
class PaperAuctionResult:
    price_cents: int | None
    trades: tuple[PaperTrade, ...]
    net_allocations: dict[str, int]
    tie_ranks: dict[str, int]
    self_volume: int = 0


def clear_paper_auction(orders: tuple[PaperOrder, ...] | list[PaperOrder], master_seed: int = 0,
                        seed_index: int = 0, round_index: int = 0, equality_crossing: str = "ge",
                        self_match_policy: str = "allow") -> PaperAuctionResult:
    if equality_crossing not in {"ge", "gt"}:
        raise ValueError("equality_crossing must be ge or gt")
    if self_match_policy != "allow":
        raise ValueError("self-match exclusion is not implemented; refusing unsupported sensitivity")
    ids = [o.order_id for o in orders]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate order_id")
    owners = sorted({o.owner for o in orders})
    for owner in owners:
        portfolio = tuple(o for o in orders if o.owner == owner)
        validate_portfolio(portfolio, owner, 400)
    ranks = {o.order_id: hash_int(master_seed, seed_index, round_index, o.order_id) for o in orders}
    buys = sorted((o for o in orders if o.side == "buy"), key=lambda o: (-o.limit_cents, ranks[o.order_id]))
    sells = sorted((o for o in orders if o.side == "sell"), key=lambda o: (o.limit_cents, ranks[o.order_id]))
    rb, rs = [o.qty for o in buys], [o.qty for o in sells]
    bi = si = self_volume = 0
    crossing = lambda bid, ask: bid >= ask if equality_crossing == "ge" else bid > ask
    fills: list[tuple[PaperOrder, PaperOrder, int]] = []
    while bi < len(buys) and si < len(sells) and crossing(buys[bi].limit_cents, sells[si].limit_cents):
        qty = min(rb[bi], rs[si])
        fills.append((buys[bi], sells[si], qty))
        if buys[bi].owner == sells[si].owner:
            self_volume += qty
        rb[bi] -= qty; rs[si] -= qty
        if rb[bi] == 0: bi += 1
        if rs[si] == 0: si += 1
    if not fills:
        return PaperAuctionResult(None, (), {owner: 0 for owner in owners}, ranks)
    last_buy, last_sell, _ = fills[-1]
    price = (last_buy.limit_cents + last_sell.limit_cents) // 2
    trades = tuple(PaperTrade(b.owner, s.owner, q, price, b.order_id, s.order_id) for b, s, q in fills)
    allocations: dict[str, int] = defaultdict(int)
    for trade in trades:
        allocations[trade.buyer_id] += trade.qty
        allocations[trade.seller_id] -= trade.qty
    return PaperAuctionResult(price, trades, {o: allocations[o] for o in owners}, ranks, self_volume)

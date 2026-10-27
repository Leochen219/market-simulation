"""Round settlement, claim fulfillment, profit calculation, and treasury sweep."""

from capclaims.domain.ledger import Ledger
from capclaims.domain.types import AuctionResult, FrozenRoundState, JournalEntry, SettlementResult


def _int_map(name: str, values: dict[str, int], low: int, high: int) -> None:
    for agent_id, value in values.items():
        if not isinstance(agent_id, str) or not agent_id:
            raise ValueError(f"{name} agent ids must be nonempty strings")
        if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
            raise ValueError(f"{name} values must be integers in [{low}, {high}]")


def settle_round(
    state: FrozenRoundState, auction: AuctionResult, wholesale_price: int,
    initial_cash_cents: int = 10_000_000,
) -> SettlementResult:
    p, d, q = dict(state.valuations), dict(state.demands), dict(state.initial_quantities)
    agents = set(p)
    if not agents or set(d) != agents or set(q) != agents:
        raise ValueError("state mappings must contain identical nonempty agent keys")
    _int_map("valuations", p, 51, 100)
    _int_map("demands", d, 0, 200)
    _int_map("initial_quantities", q, 0, 400)
    if isinstance(wholesale_price, bool) or wholesale_price not in (10, 50):
        raise ValueError("wholesale_price must be 10 or 50")
    if isinstance(initial_cash_cents, bool) or not isinstance(initial_cash_cents, int) or initial_cash_cents < 0:
        raise ValueError("initial_cash_cents must be a nonnegative integer")
    if set(auction.net_allocations) != agents:
        raise ValueError("auction net allocations must cover every state agent")
    if any(isinstance(a, bool) or not isinstance(a, int) for a in auction.net_allocations.values()):
        raise ValueError("auction net allocations must be integers")
    if sum(auction.net_allocations.values()) != 0:
        raise ValueError("auction net allocations must sum to zero")
    filled_net = {agent_id: 0 for agent_id in agents}
    for trade in auction.trades:
        if trade.buyer_id not in agents or trade.seller_id not in agents or trade.buyer_id == trade.seller_id:
            raise ValueError("auction contains an invalid counterparty")
        if isinstance(trade.qty, bool) or not isinstance(trade.qty, int) or not 1 <= trade.qty <= 400:
            raise ValueError("auction trade qty must be an integer from 1 to 400")
        if isinstance(trade.price_cents, bool) or not isinstance(trade.price_cents, int) or not 0 <= trade.price_cents <= 10_000:
            raise ValueError("auction trade price is invalid")
        filled_net[trade.buyer_id] += trade.qty
        filled_net[trade.seller_id] -= trade.qty
    if filled_net != auction.net_allocations:
        raise ValueError("auction net allocations do not match fills")
    if auction.trades:
        if auction.price_cents is None or any(t.price_cents != auction.price_cents for t in auction.trades):
            raise ValueError("trades must share a non-null uniform price")
    elif auction.price_cents is not None:
        raise ValueError("empty auction must have a null price")
    if any(q[i] + auction.net_allocations[i] < 0 for i in agents):
        raise ValueError("seller fills exceed initial holdings")

    ledger = Ledger()
    for i in sorted(agents):
        ledger.post(JournalEntry("treasury", i, "CASH_CENTS", initial_cash_cents, "round cash grant"))
        ledger.post(JournalEntry("supplier", i, "CAPACITY", q[i], "capacity issuance"))
        ledger.post(JournalEntry(i, "supplier", "CASH_CENTS", wholesale_price * q[i] * 100, "capacity procurement"))
    for trade in auction.trades:
        ledger.post(JournalEntry(trade.buyer_id, trade.seller_id, "CASH_CENTS", trade.price_cents * trade.qty, "market trade"))
        ledger.post(JournalEntry(trade.seller_id, trade.buyer_id, "CAPACITY", trade.qty, "market trade"))

    fulfilled = {i: min(d[i], q[i] + auction.net_allocations[i]) for i in agents}
    leftovers = {i: q[i] + auction.net_allocations[i] - fulfilled[i] for i in agents}
    shortages = {i: d[i] - fulfilled[i] for i in agents}
    profits: dict[str, int] = {}
    post_sweep: dict[str, int] = {}
    for i in sorted(agents):
        ledger.post(JournalEntry(i, "customer", "CAPACITY", fulfilled[i], "claim redemption"))
        ledger.post(JournalEntry("customer", i, "CASH_CENTS", p[i] * fulfilled[i] * 100, "customer payment"))
        ledger.post(JournalEntry(i, "burn", "CAPACITY", leftovers[i], "unused claim burn"))
        trade_value = (auction.price_cents or 0) * auction.net_allocations[i]
        profits[i] = -wholesale_price * q[i] * 100 - trade_value + p[i] * fulfilled[i] * 100
        cash = ledger.balance("CASH_CENTS", i)
        if cash < 0:
            raise ValueError("agent cash overdraft")
        ledger.post(JournalEntry(i, "treasury", "CASH_CENTS", cash, "round-end treasury sweep"))
        post_sweep[i] = ledger.balance("CASH_CENTS", i)
    if not ledger.validate_asset_conservation():
        raise ArithmeticError("ledger asset conservation failed")
    issued = sum(q.values())
    redeemed = sum(fulfilled.values())
    burned = sum(leftovers.values())
    if issued != redeemed + burned:
        raise ArithmeticError("capacity issuance does not equal redemption plus burn")
    return SettlementResult(profits, fulfilled, leftovers, shortages, ledger.entries, post_sweep, issued, redeemed, burned)

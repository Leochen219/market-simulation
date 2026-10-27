"""Paper-profile net-flow settlement with internal, nonbinding treasury accounting."""
from dataclasses import dataclass
from capclaims.domain.ledger import Ledger
from capclaims.domain.types import JournalEntry
from capclaims.paper.auction import PaperAuctionResult

CASH_GRANT_CENTS = 501_000_500


@dataclass(frozen=True)
class PaperSettlementResult:
    profits_cents: dict[str, int]
    fulfilled: dict[str, int]
    leftovers: dict[str, int]
    shortages: dict[str, int]
    entries: tuple[JournalEntry, ...]
    post_sweep_balances: dict[str, int]
    issued_units: int
    redeemed_units: int
    burned_units: int
    cash_grant_cents: int = CASH_GRANT_CENTS


def settle_paper_round(state: dict[str, tuple[int, int, int]], auction: PaperAuctionResult,
                       wholesale_price: int) -> PaperSettlementResult:
    if not state or wholesale_price not in (10, 50):
        raise ValueError("state must be nonempty and wholesale price must be 10 or 50")
    if set(state) != set(auction.net_allocations) or sum(auction.net_allocations.values()) != 0:
        raise ValueError("auction allocations must cover state and conserve claims")
    p, d, q = {}, {}, {}
    for agent, values in state.items():
        if len(values) != 3:
            raise ValueError("state rows must be (p, D, Q)")
        p[agent], d[agent], q[agent] = values
        if not 51 <= p[agent] <= 100 or not 0 <= d[agent] <= 200 or not 0 <= q[agent] <= 400:
            raise ValueError("paper state values outside legal domains")
    net = {a: auction.net_allocations[a] for a in state}
    flow_net = {a: 0 for a in state}
    for trade in auction.trades:
        if (trade.buyer_id not in state or trade.seller_id not in state or
                isinstance(trade.qty, bool) or not isinstance(trade.qty, int) or trade.qty < 1 or
                isinstance(trade.price_cents, bool) or not isinstance(trade.price_cents, int) or trade.price_cents < 0):
            raise ValueError("invalid auction trade")
        flow_net[trade.buyer_id] += trade.qty
        flow_net[trade.seller_id] -= trade.qty
    if flow_net != net:
        raise ValueError("auction net allocations do not match its fills")
    if any(q[a] + net[a] < 0 for a in state):
        raise ValueError("market fills exceed initial holdings")
    ledger = Ledger()
    for agent in sorted(state):
        ledger.post(JournalEntry("treasury", agent, "CASH_CENTS", CASH_GRANT_CENTS, "paper bookkeeping grant"))
        ledger.post(JournalEntry("supplier", agent, "CAPACITY", q[agent], "capacity issuance"))
        ledger.post(JournalEntry(agent, "supplier", "CASH_CENTS", wholesale_price * q[agent] * 100, "capacity procurement"))
    for trade in auction.trades:
        if trade.buyer_id not in state or trade.seller_id not in state or trade.qty < 1:
            raise ValueError("invalid auction trade")
        if trade.buyer_id != trade.seller_id:
            ledger.post(JournalEntry(trade.buyer_id, trade.seller_id, "CASH_CENTS", trade.price_cents * trade.qty, "market trade"))
            ledger.post(JournalEntry(trade.seller_id, trade.buyer_id, "CAPACITY", trade.qty, "market trade"))
    fulfilled = {a: min(d[a], q[a] + net[a]) for a in state}
    leftovers = {a: q[a] + net[a] - fulfilled[a] for a in state}
    shortages = {a: d[a] - fulfilled[a] for a in state}
    profits, balances = {}, {}
    for agent in sorted(state):
        ledger.post(JournalEntry(agent, "customer", "CAPACITY", fulfilled[agent], "claim redemption"))
        ledger.post(JournalEntry("customer", agent, "CASH_CENTS", p[agent] * fulfilled[agent] * 100, "customer payment"))
        ledger.post(JournalEntry(agent, "burn", "CAPACITY", leftovers[agent], "unused claim burn"))
        trade_cashflow = sum(
            (trade.price_cents * trade.qty if trade.seller_id == agent else 0) -
            (trade.price_cents * trade.qty if trade.buyer_id == agent else 0)
            for trade in auction.trades if trade.buyer_id != trade.seller_id
        )
        profits[agent] = -wholesale_price * q[agent] * 100 + trade_cashflow + p[agent] * fulfilled[agent] * 100
        cash_before_sweep = ledger.balance("CASH_CENTS", agent)
        expected_cash = CASH_GRANT_CENTS + profits[agent]
        if cash_before_sweep != expected_cash:
            raise ArithmeticError("ledger cash flow does not match economic profit")
        balance = cash_before_sweep
        if balance < 0:
            raise ValueError("paper treasury grant failed nonbinding cash bound")
        ledger.post(JournalEntry(agent, "treasury", "CASH_CENTS", balance, "round-end treasury sweep"))
        balances[agent] = ledger.balance("CASH_CENTS", agent)
    if not ledger.validate_asset_conservation():
        raise ArithmeticError("paper ledger conservation failed")
    issued, redeemed, burned = sum(q.values()), sum(fulfilled.values()), sum(leftovers.values())
    if issued != redeemed + burned:
        raise ArithmeticError("capacity issuance does not equal redemption plus burn")
    return PaperSettlementResult(profits, fulfilled, leftovers, shortages, ledger.entries, balances,
                                 issued, redeemed, burned)

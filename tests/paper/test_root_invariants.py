"""Independent full-domain auction/accounting checks over seeded portfolios."""
import random

from capclaims.paper.auction import PaperAuctionResult, clear_paper_auction
from capclaims.paper.codec import prefix_mask
from capclaims.paper.orders import PaperOrder, validate_portfolio
from capclaims.paper.settlement import settle_paper_round


def test_all_sell_caps_and_positive_quantity_domains_are_reachable():
    for cap in range(1, 401):
        leaves = set()
        for h in prefix_mask("", 1, cap):
            for t in prefix_mask(str(h), 1, cap):
                for u in prefix_mask(f"{h}{t}", 1, cap):
                    leaves.add(100 * h + 10 * t + u)
        assert leaves == set(range(1, cap + 1))


def test_seeded_full_portfolios_conserve_cash_claims_and_respect_filled_limits():
    for seed in range(200):
        rng = random.Random(seed)
        state, orders = {}, []
        for i in range(10):
            owner = f"a{i}"
            q = rng.randrange(401)
            state[owner] = (rng.randrange(51, 101), rng.randrange(201), q)
            remaining = q
            portfolio = []
            for slot in range(rng.randrange(6)):
                side = rng.choice(("buy", "sell")) if remaining else "buy"
                qty = rng.randrange(1, remaining + 1) if side == "sell" else rng.randrange(1, 1000)
                remaining -= qty if side == "sell" else 0
                portfolio.append(PaperOrder(f"{owner}:{slot}", owner, side, qty, rng.randrange(1, 1000)))
            validate_portfolio(portfolio, owner, q)
            orders.extend(portfolio)
        result = clear_paper_auction(orders, master_seed=seed)
        assert result == clear_paper_auction(list(reversed(orders)), master_seed=seed)
        by_id = {order.order_id: order for order in orders}
        for trade in result.trades:
            assert by_id[trade.seller_order_id].limit_cents <= trade.price_cents
            assert trade.price_cents <= by_id[trade.buyer_order_id].limit_cents
        full = PaperAuctionResult(result.price_cents, result.trades,
            {owner: result.net_allocations.get(owner, 0) for owner in state}, result.tie_ranks,
            result.self_volume)
        w = (10, 50)[seed % 2]
        settled = settle_paper_round(state, full, w)
        assert settled.issued_units == settled.redeemed_units + settled.burned_units
        assert all(value == 0 for value in settled.post_sweep_balances.values())
        revenue = sum(state[a][0] * settled.fulfilled[a] * 100 for a in state)
        supplier_revenue = sum(row[2] for row in state.values()) * w * 100
        assert sum(settled.profits_cents.values()) + supplier_revenue == revenue
        balances = {}
        for entry in settled.entries:
            for owner, sign in ((entry.source, -1), (entry.destination, 1)):
                key = (entry.asset, owner)
                balances[key] = balances.get(key, 0) + sign * entry.amount
                if owner in state:
                    assert balances[key] >= 0

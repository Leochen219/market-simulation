"""Independent domain checks across heterogeneous legal seeded markets."""
import random
from collections import defaultdict

import pytest

from capclaims.domain.auction import clear_auction
from capclaims.domain.settlement import settle_round
from capclaims.domain.types import AuctionResult, FrozenRoundState, Order


@pytest.mark.parametrize("w", [10, 50])
def test_seeded_markets_preserve_balances_and_profit_identity(w):
    for seed in range(100):
        rng = random.Random(seed)
        ids = [f"agent_{i:02d}" for i in range(10)]
        p = {i: rng.randint(51, 100) for i in ids}
        d = {i: rng.randint(0, 200) for i in ids}
        q = {i: rng.randrange(17) * 25 for i in ids}
        orders = []
        for i in ids:
            side = rng.choice(["buy", "sell", "hold"])
            cap = q[i] if side == "sell" else 400
            if side != "hold" and cap:
                orders.append(Order(i, side, rng.randint(1, cap), rng.randrange(11) * 1000))
        cleared = clear_auction(orders, master_seed=seed)
        # The engine includes zero allocations for HOLD participants.
        auction = AuctionResult(cleared.price_cents, cleared.trades,
                                {i: cleared.net_allocations.get(i, 0) for i in ids}, cleared.tie_ranks)
        result = settle_round(FrozenRoundState(p, d, q), auction, w)
        balances = defaultdict(int)
        pre_sweep_cash = {}
        for entry in result.entries:
            assert entry.amount > 0
            if entry.reason == "round-end treasury sweep":
                pre_sweep_cash[entry.source] = balances[(entry.asset, entry.source)]
            balances[(entry.asset, entry.source)] -= entry.amount
            balances[(entry.asset, entry.destination)] += entry.amount
            for i in ids:
                assert balances[(entry.asset, i)] >= 0
        for i in ids:
            assert result.profits_cents[i] == pre_sweep_cash[i] - 10_000_000
            assert balances[("CASH_CENTS", i)] == 0
            assert balances[("CAPACITY", i)] == 0
            assert q[i] + auction.net_allocations[i] == result.fulfilled[i] + result.leftovers[i]
        for asset in ["CASH_CENTS", "CAPACITY"]:
            assert sum(v for (name, _), v in balances.items() if name == asset) == 0
        assert result.issued_units == result.redeemed_units + result.burned_units
        if auction.trades:
            assert all(t.price_cents == auction.price_cents for t in auction.trades)
        assert clear_auction(list(reversed(orders)), master_seed=seed) == cleared

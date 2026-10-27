import pytest

from capclaims.domain.auction import clear_auction
from capclaims.domain.settlement import settle_round
from capclaims.domain.types import FrozenRoundState, Order


def test_settlement_claim_conservation_profit_and_cash_sweep():
    state = FrozenRoundState(
        valuations={"seller": 100, "buyer": 90},
        demands={"seller": 0, "buyer": 3},
        initial_quantities={"seller": 4, "buyer": 0},
    )
    auction = clear_auction([Order("buyer", "buy", 2, 8000), Order("seller", "sell", 2, 2000)])
    result = settle_round(state, auction, wholesale_price=10)
    assert result.profits_cents["seller"] == -4_000 + 2 * auction.price_cents
    assert result.profits_cents["buyer"] == 18_000 - 2 * auction.price_cents
    assert result.post_sweep_balances == {"seller": 0, "buyer": 0}
    assert result.issued_units == result.redeemed_units + result.burned_units
    assert sum(auction.net_allocations.values()) == 0
    assert result.entries


def test_settlement_rejects_sell_fills_beyond_holdings():
    state = FrozenRoundState({"a": 90, "b": 90}, {"a": 0, "b": 0}, {"a": 1, "b": 0})
    auction = clear_auction([Order("a", "sell", 2, 0), Order("b", "buy", 2, 100)])
    with pytest.raises(ValueError, match="holdings"):
        settle_round(state, auction, 10)


def test_settlement_rejects_inconsistent_keys_and_bad_state():
    state = FrozenRoundState({"a": 101}, {"a": 0}, {"a": 0})
    with pytest.raises(ValueError):
        settle_round(state, clear_auction([]), 10)

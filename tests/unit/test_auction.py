from capclaims.domain.auction import clear_auction
from capclaims.domain.types import Order


def test_uniform_price_multimatch_uses_last_accepted_midpoint():
    result = clear_auction(
        [Order("b1", "buy", 2, 100), Order("b2", "buy", 1, 80),
         Order("s1", "sell", 1, 20), Order("s2", "sell", 2, 60)],
        master_seed=1,
    )
    assert sum(t.qty for t in result.trades) == 3
    assert result.price_cents == 70
    assert result.net_allocations["b1"] == 2
    assert result.net_allocations["b2"] == 1


def test_partial_fill_ties_are_submission_order_invariant():
    orders = [Order("b1", "buy", 2, 100), Order("b2", "buy", 2, 100),
              Order("s1", "sell", 3, 20)]
    forward = clear_auction(orders, 4, 0, 0)
    reverse = clear_auction(list(reversed(orders)), 4, 0, 0)
    assert forward == reverse
    assert sum(t.qty for t in forward.trades) == 3
    first = min(("b1", "b2"), key=lambda agent: forward.tie_ranks[agent])
    second = "b2" if first == "b1" else "b1"
    assert forward.net_allocations[first] == 2
    assert forward.net_allocations[second] == 1
    assert sum(forward.net_allocations.values()) == 0


def test_no_trade_has_null_price_and_zero_net():
    result = clear_auction([Order("b", "buy", 1, 20), Order("s", "sell", 1, 40)])
    assert result.price_cents is None
    assert result.trades == ()
    assert set(result.net_allocations.values()) == {0}


def test_auction_rejects_odd_cent_limits_and_duplicate_orders():
    import pytest

    with pytest.raises(ValueError, match="even"):
        clear_auction([Order("b", "buy", 1, 101)])
    with pytest.raises(ValueError, match="duplicate"):
        clear_auction([Order("a", "buy", 1, 100), Order("a", "sell", 1, 20)])

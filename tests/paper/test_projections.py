import pytest
from capclaims.paper.projections import hypothetical_profit_cents


def test_hypothetical_calculator_uses_supplied_demand_and_signed_net_trade():
    assert hypothetical_profit_cents(p=80,wholesale_price=10,q=100,demand=120,
                                    net_quantity=20,market_price_cents=3250) == 795000
    assert hypothetical_profit_cents(p=80,wholesale_price=10,q=100,demand=50,
                                    net_quantity=-20,market_price_cents=3250) == 365000
    assert hypothetical_profit_cents(p=80,wholesale_price=10,q=100,demand=0) == -100000
    with pytest.raises(ValueError):
        hypothetical_profit_cents(p=80,wholesale_price=10,q=1,demand=50,net_quantity=-2)

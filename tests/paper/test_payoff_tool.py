import pytest

from capclaims.paper.payoff_tool import calculate_payoff

BASE = dict(p=70, wholesale_price=10, q=60, demand=20,
            net_quantity=-20, market_price_cents=1500)
EXPECTED_KEYS = {'procurement_cost_cents', 'trade_cashflow_cents',
                 'inventory_after_trade', 'fulfilled_units',
                 'retail_revenue_cents', 'final_profit_cents'}


def calc(**over):
    return calculate_payoff(**{**BASE, **over})


def test_sell_scenario():
    r = calc()
    assert r == {'procurement_cost_cents': 60000, 'trade_cashflow_cents': 30000,
                 'inventory_after_trade': 40, 'fulfilled_units': 20,
                 'retail_revenue_cents': 140000, 'final_profit_cents': 110000}


def test_stop_scenario():
    assert calc(net_quantity=0, market_price_cents=0)['final_profit_cents'] == 80000


def test_zero_demand():
    r = calc(p=90, q=10, demand=0, net_quantity=-10, market_price_cents=100)
    assert r['fulfilled_units'] == 0
    assert r['final_profit_cents'] == -9000


def test_partial_shortfall_two_independent_calls():
    sell = calc(q=28, net_quantity=-10)
    stop = calc(q=28, net_quantity=0, market_price_cents=0)
    assert sell['inventory_after_trade'] == 18
    assert sell['fulfilled_units'] == 18
    assert sell['final_profit_cents'] == 113000
    assert stop['final_profit_cents'] == 112000


def test_no_recommendation_fields():
    keys = set(calc())
    assert keys == EXPECTED_KEYS
    for bad in ('recommend', 'optimal', 'best', 'delta', 'advice', 'compare'):
        assert not any(bad in k for k in keys)


@pytest.mark.parametrize('over', [
    {'p': True}, {'net_quantity': False}, {'market_price_cents': True},  # bools
    {'q': 60.0}, {'demand': '20'}, {'market_price_cents': None},         # non-ints
    {'net_quantity': -61},                                               # oversell
    {'net_quantity': 1000}, {'q': 0, 'net_quantity': 1000},              # A too large
    {'market_price_cents': 99901}, {'market_price_cents': -1},           # price
    {'p': 50}, {'p': 101}, {'wholesale_price': 20}, {'q': 401},          # domain
    {'q': -1}, {'demand': 201}, {'demand': -1},
])
def test_rejects_invalid(over):
    with pytest.raises(ValueError):
        calc(**over)

"""Pure one-scenario payoff calculator (accounting only, per paper Eq. 1).

Evaluates exactly one user-specified scenario. Gives no advice, comparison,
delta, optimal action, future-state lookup or enumeration.
"""
from capclaims.paper.projections import hypothetical_profit_cents

MAX_ABS_NET_QUANTITY = 999
MAX_MARKET_PRICE_CENTS = 99900


def calculate_payoff(*, p: int, wholesale_price: int, q: int, demand: int,
                     net_quantity: int, market_price_cents: int) -> dict[str, int]:
    # Existing helper validates types/domains and computes profit.
    profit = hypothetical_profit_cents(
        p=p, wholesale_price=wholesale_price, q=q, demand=demand,
        net_quantity=net_quantity, market_price_cents=market_price_cents)
    if abs(net_quantity) > MAX_ABS_NET_QUANTITY:
        raise ValueError('net_quantity outside tool bounds')
    if not 0 <= market_price_cents <= MAX_MARKET_PRICE_CENTS:
        raise ValueError('market_price_cents outside tool bounds')
    inventory_after_trade = q + net_quantity
    fulfilled_units = min(demand, inventory_after_trade)
    return {
        'procurement_cost_cents': 100 * wholesale_price * q,
        'trade_cashflow_cents': -market_price_cents * net_quantity,
        'inventory_after_trade': inventory_after_trade,
        'fulfilled_units': fulfilled_units,
        'retail_revenue_cents': 100 * p * fulfilled_units,
        'final_profit_cents': profit,
    }

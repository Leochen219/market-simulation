"""Hypothetical payoff calculator; callers supply Q/D, never a future-state lookup."""


def hypothetical_profit_cents(*, p: int, wholesale_price: int, q: int, demand: int,
                              net_quantity: int = 0, market_price_cents: int = 0) -> int:
    """Paper Eq.(1), given hypothetical signed net buy quantity and uniform price.

    No random generator, coordinator state, or actual unrevealed demand is read.
    ``net_quantity > 0`` means a net purchase. This calculator is an API utility;
    Laya receives its known-demand pretrade projection, not interactive tool calls.
    """
    values = (p, wholesale_price, q, demand, net_quantity, market_price_cents)
    if any(isinstance(v, bool) or not isinstance(v, int) for v in values):
        raise ValueError("calculator inputs must be integers")
    if (not 51 <= p <= 100 or wholesale_price not in (10, 50) or not 0 <= q <= 400
            or not 0 <= demand <= 200 or market_price_cents < 0 or q + net_quantity < 0):
        raise ValueError("calculator inputs outside economic domains")
    return -100 * wholesale_price * q - market_price_cents * net_quantity + 100 * p * min(demand, q + net_quantity)

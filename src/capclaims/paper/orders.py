"""Order portfolio records and paper action validation."""
from dataclasses import dataclass


@dataclass(frozen=True)
class PaperOrder:
    order_id: str
    owner: str
    side: str
    qty: int
    limit_dollars: int

    @property
    def limit_cents(self) -> int:
        return self.limit_dollars * 100


def validate_portfolio(orders: tuple[PaperOrder, ...] | list[PaperOrder], owner: str, initial_q: int,
                       maximum_orders: int = 5) -> None:
    if len(orders) > maximum_orders:
        raise ValueError("portfolio exceeds five total orders")
    if isinstance(initial_q, bool) or not isinstance(initial_q, int) or not 0 <= initial_q <= 400:
        raise ValueError("initial procurement must be an integer from 0 to 400")
    ids: set[str] = set()
    gross_sell = 0
    for order in orders:
        if order.owner != owner or not order.order_id or order.order_id in ids:
            raise ValueError("portfolio owner or unique order_id is invalid")
        ids.add(order.order_id)
        if order.side not in {"buy", "sell"}:
            raise ValueError("order side must be buy or sell")
        if isinstance(order.qty, bool) or not isinstance(order.qty, int) or not 1 <= order.qty <= 999:
            raise ValueError("order quantity must be an integer from 1 to 999")
        if isinstance(order.limit_dollars, bool) or not isinstance(order.limit_dollars, int) or not 1 <= order.limit_dollars <= 999:
            raise ValueError("order limit must be an integer dollar from 1 to 999")
        if order.side == "sell":
            gross_sell += order.qty
    if gross_sell > initial_q:
        raise ValueError("gross sell submissions exceed initial procurement")

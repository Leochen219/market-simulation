"""Stable legal candidate sets for purchase and one-order trade actions."""

from capclaims.domain.types import Candidate

PURCHASE_GRID = tuple(range(0, 401, 25))
QUANTITY_GRID = (5, 10, 25, 50, 75, 100, 150, 200, 300, 400)
PRICE_GRID = tuple(range(0, 101, 10))


def purchase_candidates() -> tuple[Candidate, ...]:
    return tuple(Candidate(f"q_{q:03d}", q, str(q)) for q in PURCHASE_GRID)


def side_candidates(q: int) -> tuple[Candidate, ...]:
    if isinstance(q, bool) or not isinstance(q, int) or not 0 <= q <= 400:
        raise ValueError("q must be an integer from 0 to 400")
    sides = ("buy", "hold") if q == 0 else ("buy", "sell", "hold")
    return tuple(Candidate(side, side, side.upper()) for side in sides)


def quantity_candidates(side: str, q: int) -> tuple[Candidate, ...]:
    if side not in {"buy", "sell"}:
        raise ValueError("quantity candidates require buy or sell")
    if isinstance(q, bool) or not isinstance(q, int) or not 0 <= q <= 400:
        raise ValueError("q must be an integer from 0 to 400")
    cap = 400 if side == "buy" else q
    return tuple(Candidate(f"qty_{n:03d}", n, str(n)) for n in QUANTITY_GRID if n <= cap)


def price_candidates() -> tuple[Candidate, ...]:
    return tuple(Candidate(f"p_{p:03d}", p, f"${p}") for p in PRICE_GRID)

"""Predeclared proper scores and point summaries for conditional portfolios."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


def _finite_vector(values: Sequence[float]) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if not result or any(not math.isfinite(value) for value in result):
        raise ValueError("vectors must be nonempty and finite")
    return result


def _distance(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("vectors must have the same dimension")
    return math.sqrt(math.fsum((a - b) ** 2 for a, b in zip(left, right)))


def feature_vector(orders: Sequence[Mapping[str, Any]]) -> tuple[float, ...]:
    """Return frozen z=(has sides, counts/5, quantities/200, weighted means/100)."""
    if len(orders) > 5:
        raise ValueError("a predicted or observed portfolio cannot exceed five orders")
    by_side: dict[str, list[Mapping[str, Any]]] = {"buy": [], "sell": []}
    for order in orders:
        side, qty, price = order.get("side"), order.get("qty"), order.get("price", order.get("limit_dollars"))
        if side not in by_side or type(qty) is not int or not 1 <= qty <= 999 or type(price) is not int or not 1 <= price <= 999:
            raise ValueError("portfolio order is malformed")
        by_side[side].append(order)
    buy, sell = by_side["buy"], by_side["sell"]
    bqty, sqty = sum(row["qty"] for row in buy), sum(row["qty"] for row in sell)
    bmean = sum(row["qty"] * row.get("price", row.get("limit_dollars")) for row in buy) / bqty if bqty else 0
    smean = sum(row["qty"] * row.get("price", row.get("limit_dollars")) for row in sell) / sqty if sqty else 0
    return (float(bool(buy)), float(bool(sell)), len(buy) / 5, len(sell) / 5,
            bqty / 200, sqty / 200, bmean / 100, smean / 100)


def fair_energy_score(samples: Sequence[Sequence[float]], truth: Sequence[float]) -> float:
    if len(samples) < 2:
        raise ValueError("fair score requires at least two independent forecast samples")
    checked = [_finite_vector(row) for row in samples]
    target = _finite_vector(truth)
    if any(len(row) != len(target) for row in checked):
        raise ValueError("forecast and observation dimensions differ")
    n = len(checked)
    first = math.fsum(_distance(row, target) for row in checked) / n
    second = math.fsum(_distance(checked[i], checked[j]) for i in range(n) for j in range(n) if i != j) / (2 * n * (n - 1))
    value = first - second
    return 0.0 if abs(value) < 1e-14 else value


def fair_crps(samples: Sequence[float], truth: float) -> float:
    if len(samples) < 2:
        raise ValueError("fair score requires at least two independent forecast samples")
    values = _finite_vector(samples)
    y = float(truth)
    if not math.isfinite(y):
        raise ValueError("truth must be finite")
    n = len(values)
    first = math.fsum(abs(x - y) for x in values) / n
    second = math.fsum(abs(values[i] - values[j]) for i in range(n) for j in range(n) if i != j) / (2 * n * (n - 1))
    return max(0.0, first - second)


def fair_brier(samples: Sequence[int | bool], truth: int | bool) -> float:
    if len(samples) < 2 or any(value not in (0, 1, False, True) for value in samples) or truth not in (0, 1, False, True):
        raise ValueError("Brier inputs must be binary and contain at least two samples")
    p = sum(bool(value) for value in samples) / len(samples)
    n = len(samples)
    return (p - int(bool(truth))) ** 2 - p * (1 - p) / (n - 1)


def _curves(orders: Sequence[Mapping[str, Any]]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    buys, sells = [], []
    for price in range(1, 1000):
        buys.append(sum(int(row["qty"]) for row in orders if row["side"] == "buy" and row.get("price", row.get("limit_dollars")) >= price))
        sells.append(sum(int(row["qty"]) for row in orders if row["side"] == "sell" and row.get("price", row.get("limit_dollars")) <= price))
    return tuple(buys), tuple(sells)


def curve_distance(left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]], *,
                   low: int = 1, high: int = 999) -> float:
    if type(low) is not int or type(high) is not int or not 1 <= low <= high <= 999:
        raise ValueError("price window must be inside 1..999")
    a, b = _curves(left), _curves(right)
    return math.fsum(abs(a[s][p - 1] - b[s][p - 1]) for s in (0, 1) for p in range(low, high + 1))


def medoid_index(samples: Sequence[Sequence[Mapping[str, Any]]],
                 sample_ids: Sequence[int] | None = None) -> int:
    """Choose a sample minimizing original L1 loss, breaking ties by frozen sample ID."""
    if not samples:
        raise ValueError("medoid needs at least one portfolio")
    ids = tuple(range(len(samples))) if sample_ids is None else tuple(sample_ids)
    if len(ids) != len(samples) or any(type(item) is not int for item in ids) or len(set(ids)) != len(ids):
        raise ValueError("medoid sample IDs must be unique integers aligned with portfolios")
    losses = [math.fsum(curve_distance(candidate, other) for j, other in enumerate(samples) if i != j)
              for i, candidate in enumerate(samples)]
    return min(range(len(losses)), key=lambda index: (losses[index], ids[index]))


def incomplete_energy_interval(q: int) -> tuple[float, float]:
    if type(q) is not int or not 0 <= q <= 400:
        raise ValueError("Q must be an integer from 0 through 400")
    maximum = math.sqrt(4 + (4995 / 200) ** 2 + (q / 200) ** 2 + 2 * (999 / 100) ** 2)
    return 0.0, maximum


def incomplete_scalar_interval(*, upper: float) -> tuple[float, float]:
    if not math.isfinite(upper) or upper < 0:
        raise ValueError("upper bound must be finite and nonnegative")
    return 0.0, upper


def expected_energy_score(distribution: Mapping[tuple[float, ...], float], truth: Sequence[float]) -> float:
    """Exact weighted fair-score expectation when every ensemble is drawn iid from P."""
    target = _finite_vector(truth)
    rows = [(tuple(point), float(weight)) for point, weight in distribution.items() if weight > 0]
    if not rows or any(weight < 0 or not math.isfinite(weight) for _, weight in rows):
        raise ValueError("distribution weights must be finite with positive total mass")
    total = math.fsum(weight for _, weight in rows)
    p = [(point, weight / total) for point, weight in rows]
    if any(len(point) != len(target) or any(not math.isfinite(float(v)) for v in point) for point, _ in p):
        raise ValueError("distribution point dimension or values are invalid")
    first = math.fsum(weight * _distance(point, target) for point, weight in p)
    pair = math.fsum(w1 * w2 * _distance(x, y) for x, w1 in p for y, w2 in p)
    return first - 0.5 * pair


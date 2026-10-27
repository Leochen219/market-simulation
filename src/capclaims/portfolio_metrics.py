"""Pure curve-based evaluation for complete and missing paper portfolios."""
from __future__ import annotations

from collections import Counter
from statistics import median
from typing import Any, Mapping, Sequence

from capclaims.paper.orders import PaperOrder

MAX_PRICE = 999
MAX_ORDERS = 5
MAX_ORDER_QTY = 999


def _validate_orders(orders: Sequence[PaperOrder]) -> tuple[PaperOrder, ...]:
    result = tuple(orders)
    if len(result) > MAX_ORDERS:
        raise ValueError("a portfolio can contain no more than five orders")
    for order in result:
        if not isinstance(order, PaperOrder) or order.side not in {"buy", "sell"}:
            raise ValueError("orders must be PaperOrder values with buy or sell sides")
        if type(order.qty) is not int or not 1 <= order.qty <= MAX_ORDER_QTY:
            raise ValueError("order quantity must be from 1 through 999")
        if type(order.limit_dollars) is not int or not 1 <= order.limit_dollars <= MAX_PRICE:
            raise ValueError("limit price must be from 1 through 999")
    return result


def quote_curves(orders: Sequence[PaperOrder], maximum_price: int = MAX_PRICE) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if type(maximum_price) is not int or not 1 <= maximum_price <= MAX_PRICE:
        raise ValueError("maximum_price must be from 1 through 999")
    portfolio = _validate_orders(orders)
    buys, sells = [], []
    for price in range(1, maximum_price + 1):
        buys.append(sum(order.qty for order in portfolio if order.side == "buy" and order.limit_dollars >= price))
        sells.append(sum(order.qty for order in portfolio if order.side == "sell" and order.limit_dollars <= price))
    return tuple(buys), tuple(sells)


def _curve_error(curves_a: tuple[Sequence[float], Sequence[float]],
                 curves_b: tuple[Sequence[float], Sequence[float]], maximum_price: int) -> float:
    if len(curves_a[0]) != maximum_price or len(curves_a[1]) != maximum_price or \
            len(curves_b[0]) != maximum_price or len(curves_b[1]) != maximum_price:
        raise ValueError("curve lengths must match maximum_price")
    numerator = sum(abs(a - b) for side_a, side_b in zip(curves_a, curves_b)
        for a, b in zip(side_a, side_b))
    return numerator / (2 * maximum_price)


def curve_error(pred: Sequence[PaperOrder], human: Sequence[PaperOrder], maximum_price: int = MAX_PRICE) -> float:
    return _curve_error(quote_curves(pred, maximum_price), quote_curves(human, maximum_price), maximum_price)


def failure_upper_bound(human: Sequence[PaperOrder], Q: int, maximum_price: int = MAX_PRICE) -> float:
    if type(Q) is not int or not 0 <= Q <= 400:
        raise ValueError("Q must be an integer from 0 through 400")
    if type(maximum_price) is not int or not 1 <= maximum_price <= MAX_PRICE:
        raise ValueError("maximum_price must be from 1 through 999")
    buy_curve, sell_curve = quote_curves(human, maximum_price)
    upper = sum(max(value, MAX_ORDERS * MAX_ORDER_QTY - value)
        + max(sell, Q - sell) for value, sell in zip(buy_curve, sell_curve))
    return upper / (2 * maximum_price)


def _merged_quotes(orders: Sequence[PaperOrder]) -> Counter[tuple[str, int, int]]:
    return Counter((order.side, order.limit_dollars, order.qty) for order in _validate_orders(orders))


def _merged_same_price(orders: Sequence[PaperOrder]) -> Counter[tuple[str, int, int]]:
    merged: Counter[tuple[str, int, int]] = Counter()
    for order in _validate_orders(orders):
        merged[(order.side, order.limit_dollars, 0)] += order.qty
    return merged


def side_use_class(orders: Sequence[PaperOrder]) -> str:
    sides = {order.side for order in _validate_orders(orders)}
    return "empty" if not sides else "mixed" if len(sides) == 2 else f"{next(iter(sides))}_only"


def portfolio_comparison(pred: Sequence[PaperOrder], human: Sequence[PaperOrder]) -> dict[str, Any]:
    pred, human = _validate_orders(pred), _validate_orders(human)
    return {
        "primary_curve_error": curve_error(pred, human),
        "retail_window_1_100_curve_error": curve_error(pred, human, maximum_price=100),
        "predicted_gross_buy": sum(o.qty for o in pred if o.side == "buy"),
        "human_gross_buy": sum(o.qty for o in human if o.side == "buy"),
        "predicted_gross_sell": sum(o.qty for o in pred if o.side == "sell"),
        "human_gross_sell": sum(o.qty for o in human if o.side == "sell"),
        "predicted_side_class": side_use_class(pred), "human_side_class": side_use_class(human),
        "side_class_agreement": side_use_class(pred) == side_use_class(human),
        "order_count_error": abs(len(pred) - len(human)),
        "empty_agreement": (not pred) == (not human),
        "exact_order_multiset_match": _merged_quotes(pred) == _merged_quotes(human),
        "merged_same_side_price_match": _merged_same_price(pred) == _merged_same_price(human),
    }


def score_targets(predicted: Mapping[str, Sequence[PaperOrder] | None],
                  human: Mapping[str, Sequence[PaperOrder]],
                  Q_by_target: Mapping[str, int]) -> dict[str, Any]:
    if set(human) != set(Q_by_target) or set(human) != set(predicted):
        raise ValueError("prediction, human target, and Q mappings must cover the same planned targets")
    all_errors, completed_errors, per_target = [], [], {}
    complete_n = 0
    for target in sorted(human):
        human_portfolio = _validate_orders(human[target])
        q = Q_by_target[target]
        if type(q) is not int or not 0 <= q <= 400:
            raise ValueError("target Q must be an integer from 0 through 400")
        if any(order.owner != human_portfolio[0].owner for order in human_portfolio):
            raise ValueError("human target portfolio must have one owner")
        human_sells = sum(order.qty for order in human_portfolio if order.side == "sell")
        if human_sells > q:
            raise ValueError("human target exceeds current gross sell capacity")
        result = predicted[target]
        if result is None:
            error = failure_upper_bound(human_portfolio, q)
            secondary_error = failure_upper_bound(human_portfolio, q, maximum_price=100)
            per_target[target] = {"status": "incomplete", "scored_error": error,
                "retail_window_1_100_scored_error": secondary_error,
                "score_kind": "conservative_upper_bound"}
        else:
            result = _validate_orders(result)
            if result and any(order.owner != result[0].owner for order in result):
                raise ValueError("predicted portfolio must have one owner")
            if sum(order.qty for order in result if order.side == "sell") > q:
                raise ValueError("predicted portfolio exceeds current gross sell capacity")
            comparison = portfolio_comparison(result, human_portfolio)
            error = comparison["primary_curve_error"]
            per_target[target] = {"status": "complete", **comparison}
            completed_errors.append(error)
            complete_n += 1
        all_errors.append(error)
    return {"target_n": len(human), "completed_n": complete_n,
        "coverage": complete_n / len(human) if human else None,
        "all_target_mean_curve_error": sum(all_errors) / len(all_errors) if all_errors else None,
        "completed_case_mean_curve_error": sum(completed_errors) / len(completed_errors) if completed_errors else None,
        "all_target_mean_retail_window_1_100_curve_error": sum(
            row.get("retail_window_1_100_scored_error", row.get("retail_window_1_100_curve_error", 0.0))
            for row in per_target.values()) / len(per_target) if per_target else None,
        "targets": per_target}


def pointwise_median_curves(history: Sequence[Mapping[str, Any]]) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Training-only statistical reference; curves need not correspond to an executable portfolio."""
    if not history:
        raise ValueError("training history is required for personal median curves")
    if any(type(row.get("round")) is not int or row["round"] > 8 for row in history):
        raise ValueError("median-curve baseline may use training rounds 1 through 8 only")
    curves = [quote_curves(row["orders"]) for row in history]
    buys = tuple(float(median([curve[0][i] for curve in curves])) for i in range(MAX_PRICE))
    sells = tuple(float(median([curve[1][i] for curve in curves])) for i in range(MAX_PRICE))
    return buys, sells


def training_baselines(history: Sequence[Mapping[str, Any]], targets: Mapping[str, Sequence[PaperOrder]],
                       Q_by_target: Mapping[str, int]) -> dict[str, Any]:
    if not history or any(type(row.get("round")) is not int or row["round"] > 8 for row in history):
        raise ValueError("training baselines require prior examples from rounds 1 through 8 only")
    if set(targets) != set(Q_by_target):
        raise ValueError("baseline target portfolios and current Q values must align")
    last = max(history, key=lambda row: row["round"])
    if last["round"] != 8:
        raise ValueError("last-training-portfolio baseline requires the round-8 example")
    last_portfolio = _validate_orders(last["orders"])
    median_curves = pointwise_median_curves(history)
    empty_curves = quote_curves(())
    target_ids = sorted(targets)
    result = {"training_last_round": last["round"], "targets": {}}
    for target in target_ids:
        truth = _validate_orders(targets[target])
        current_q = Q_by_target[target]
        truth_curves = quote_curves(truth)
        last_sell = sum(order.qty for order in last_portfolio if order.side == "sell")
        result["targets"][target] = {
            "last_training_portfolio_curve_error": curve_error(last_portfolio, truth),
            "last_training_portfolio_feasible_under_current_Q": last_sell <= current_q,
            "last_training_portfolio_unclipped_gross_sell": last_sell,
            "current_Q": current_q,
            "pointwise_personal_median_curve_error": _curve_error(median_curves, truth_curves, MAX_PRICE),
            "empty_portfolio_curve_error": _curve_error(empty_curves, truth_curves, MAX_PRICE),
        }
    return result

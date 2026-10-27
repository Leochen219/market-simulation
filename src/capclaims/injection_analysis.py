"""Offline analysis helpers for the preregistered five-arm injection pilot."""
from __future__ import annotations

from collections import defaultdict
from statistics import mean, median
from typing import Any, Mapping, Sequence

from capclaims.paper.orders import PaperOrder
from capclaims.portfolio_metrics import failure_upper_bound, portfolio_comparison, quote_curves, MAX_ORDERS, MAX_ORDER_QTY


def collapse_repeats(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    """Average session losses within each person/arm before any arm summary."""
    buckets: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        person, arm, loss = row.get("person_id"), row.get("arm"), row.get("loss")
        if not isinstance(person, str) or arm not in {"A", "B", "C", "D", "E"}:
            raise ValueError("analysis row requires a valid person and arm")
        if isinstance(loss, bool) or not isinstance(loss, (int, float)):
            raise ValueError("session loss must be numeric")
        buckets[person][arm].append(float(loss))
    return {person: {arm: mean(values) for arm, values in arms.items()}
            for person, arms in buckets.items()}


def paired_arm_summary(person_arm_losses: Mapping[str, Mapping[str, float]], left: str, right: str) -> dict[str, Any]:
    paired = [(person, person_arm_losses[person][left] - person_arm_losses[person][right])
              for person in sorted(person_arm_losses)
              if left in person_arm_losses[person] and right in person_arm_losses[person]]
    values = [value for _, value in paired]
    return {"left_arm": left, "right_arm": right, "person_n": len(values),
            "mean_left_minus_right_loss": mean(values) if values else None,
            "median_left_minus_right_loss": median(values) if values else None,
            "paired_differences": [{"person_id": person, "left_minus_right": value} for person, value in paired]}


def _side_curve_error(predicted: Sequence[PaperOrder], truth: Sequence[PaperOrder], side: str,
                      maximum_price: int) -> float:
    pred_curves = quote_curves([row for row in predicted if row.side == side], maximum_price)
    truth_curves = quote_curves([row for row in truth if row.side == side], maximum_price)
    pred, actual = pred_curves[0 if side == "buy" else 1], truth_curves[0 if side == "buy" else 1]
    return sum(abs(x - y) for x, y in zip(pred, actual)) / maximum_price


def score_portfolio(predicted: Sequence[PaperOrder] | None, truth: Sequence[PaperOrder], q: int) -> dict[str, Any]:
    """Score only full portfolios; missing/tied traces receive the fixed full-domain bound."""
    if predicted is None:
        human_buy, human_sell = quote_curves(truth)
        buy_upper = sum(max(value, MAX_ORDERS * MAX_ORDER_QTY - value) for value in human_buy) / 999
        sell_upper = sum(max(value, q - value) for value in human_sell) / 999
        buy_100, sell_100 = quote_curves(truth, maximum_price=100)
        buy_upper_100 = sum(max(value, MAX_ORDERS * MAX_ORDER_QTY - value) for value in buy_100) / 100
        sell_upper_100 = sum(max(value, q - value) for value in sell_100) / 100
        return {"status": "unresolved", "score_kind": "conservative_upper_bound",
                "loss": failure_upper_bound(truth, q),
                "retail_window_1_100_loss": failure_upper_bound(truth, q, maximum_price=100),
                "buy_curve_error": buy_upper, "sell_curve_error": sell_upper,
                "buy_window_1_100_error": buy_upper_100, "sell_window_1_100_error": sell_upper_100,
                "sell_curve_error_as_share_of_Q": sell_upper / q if q > 0 else None,
                "sell_window_1_100_error_as_share_of_Q": sell_upper_100 / q if q > 0 else None}
    comparison = portfolio_comparison(predicted, truth)
    unique_order_keys = {(row.side, row.qty, row.limit_dollars) for row in predicted}
    return {"status": "complete", "score_kind": "full_portfolio", "loss": comparison["primary_curve_error"],
            "retail_window_1_100_loss": comparison["retail_window_1_100_curve_error"],
            "buy_curve_error": _side_curve_error(predicted, truth, "buy", 999),
            "sell_curve_error": _side_curve_error(predicted, truth, "sell", 999),
            "buy_window_1_100_error": _side_curve_error(predicted, truth, "buy", 100),
            "sell_window_1_100_error": _side_curve_error(predicted, truth, "sell", 100),
            "sell_curve_error_as_share_of_Q": (_side_curve_error(predicted, truth, "sell", 999) / q
                if q > 0 else None),
            "sell_window_1_100_error_as_share_of_Q": (_side_curve_error(predicted, truth, "sell", 100) / q
                if q > 0 else None),
            "duplicate_order_n": len(predicted) - len(unique_order_keys),
            "five_order_portfolio": len(predicted) == MAX_ORDERS,
            "comparison": comparison}

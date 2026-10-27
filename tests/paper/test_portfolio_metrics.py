import pytest

from capclaims.paper.orders import PaperOrder
from capclaims.portfolio_metrics import (
    curve_error, failure_upper_bound, pointwise_median_curves,
    portfolio_comparison, quote_curves, score_targets, training_baselines,
)


def order(side, qty, price, slot=0):
    return PaperOrder(f"slot{slot}", "opaque", side, qty, price)


def test_quote_curve_and_error_are_permutation_invariant_and_keep_sides_separate():
    a = (order("buy", 10, 50, 0), order("sell", 4, 30, 1), order("buy", 7, 20, 2))
    b = tuple(reversed(a))
    assert quote_curves(a) == quote_curves(b)
    assert curve_error(a, b) == 0
    buys, sells = quote_curves(a)
    assert buys[0] == 17 and buys[19] == 17 and buys[20] == 10 and buys[49] == 10 and buys[50] == 0
    assert sells[28] == 0 and sells[29] == 4


def test_quote_curves_are_invariant_to_splitting_same_side_same_price_order():
    whole = (order("buy", 10, 50, 0), order("sell", 5, 30, 1))
    split = (order("buy", 4, 50, 0), order("sell", 5, 30, 1), order("buy", 6, 50, 2))
    assert quote_curves(whole) == quote_curves(split)
    assert curve_error(whole, split) == 0
    comparison = portfolio_comparison(whole, split)
    assert not comparison["exact_order_multiset_match"]
    assert comparison["merged_same_side_price_match"]


def test_secondary_retail_window_is_scored_separately():
    pred = (order("buy", 10, 999),)
    human = (order("buy", 10, 100),)
    full = curve_error(pred, human, 999)
    retail = curve_error(pred, human, 100)
    assert full > 0
    assert retail == 0


def test_secondary_failure_bound_uses_window_denominator_and_is_safe():
    human = (order("buy", 999, 999), order("sell", 400, 1, 1))
    upper = failure_upper_bound(human, 400, maximum_price=100)
    assert upper >= curve_error((), human, maximum_price=100)
    assert upper <= (5 * 999 + 400)


def test_missing_portfolio_upper_bound_covers_extreme_legal_curve_errors():
    human = (order("sell", 300, 1), order("buy", 400, 999))
    q = 300
    upper = failure_upper_bound(human, q)
    legal_predictions = [(), (order("buy", 999, 999),),
        tuple(order("buy", 999, 999, slot) for slot in range(5)),
        tuple(order("sell", 60, 10, slot) for slot in range(5))]
    for prediction in legal_predictions:
        assert curve_error(prediction, human) <= upper
    result = score_targets({"t": None}, {"t": human}, {"t": q})
    assert result["target_n"] == 1 and result["completed_n"] == 0
    assert result["coverage"] == 0
    assert result["all_target_mean_curve_error"] == pytest.approx(upper)
    assert result["completed_case_mean_curve_error"] is None
    assert result["targets"]["t"]["score_kind"] == "conservative_upper_bound"


def test_completed_scoring_and_secondary_multiset_metrics():
    human = (order("buy", 10, 50, 0), order("sell", 5, 30, 1))
    pred = (order("sell", 5, 30, 4), order("buy", 10, 50, 2))
    score = score_targets({"a": pred}, {"a": human}, {"a": 10})
    assert score["coverage"] == 1
    metrics = score["targets"]["a"]
    assert metrics["primary_curve_error"] == 0
    assert metrics["exact_order_multiset_match"]
    assert metrics["merged_same_side_price_match"]
    assert metrics["side_class_agreement"]


def test_training_baselines_use_only_rounds_through_eight_and_do_not_clip_sell_totals():
    history = [
        {"round": 7, "orders": [order("buy", 20, 50, 0)]},
        {"round": 8, "orders": [order("sell", 50, 20, 0)]},
    ]
    truth = {"target": (order("sell", 10, 30, 0),)}
    baselines = training_baselines(history, truth, {"target": 10})["targets"]["target"]
    assert baselines["last_training_portfolio_unclipped_gross_sell"] == 50
    assert baselines["last_training_portfolio_feasible_under_current_Q"] is False
    with pytest.raises(ValueError, match="rounds 1 through 8"):
        pointwise_median_curves(history + [{"round": 9, "orders": []}])
    with pytest.raises(ValueError, match="rounds 1 through 8"):
        training_baselines(history + [{"round": 9, "orders": []}], truth, {"target": 10})


def test_curve_bounds_and_score_mapping_alignment_fail_closed():
    with pytest.raises(ValueError, match="maximum_price"):
        quote_curves((), maximum_price=1000)
    with pytest.raises(ValueError, match="Q"):
        failure_upper_bound((), -1)
    with pytest.raises(ValueError, match="same planned targets"):
        score_targets({"a": ()}, {"b": ()}, {"b": 0})

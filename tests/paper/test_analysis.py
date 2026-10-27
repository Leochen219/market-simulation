from statistics import NormalDist
from capclaims.paper.analysis import (summarize_events, continuous_newsvendor_deviation,
                                      coopetitor_normal_approx_feature)


def test_summary_excludes_warmup_and_reports_market_metrics_by_formal_round():
    events = [
        {"round_index": 0, "warmup": True, "procurement": {"a": 5}, "orders": [], "markets": [],
         "net_allocations": {"a": 0}, "settlement": {"leftovers": {"a": 5}, "shortages": {"a": 0},
         "profits_cents": {"a": -5000}, "fulfilled": {"a": 0}}},
        {"round_index": 1, "warmup": False, "procurement": {"a": 10}, "orders": [
            {"order_id": "b", "owner": "a", "side": "buy", "qty": 4, "limit_dollars": 5}],
         "markets": [{"members": ["a"], "price_cents": 400, "self_volume": 0,
                      "trades": [{"buyer_id": "a", "seller_id": "x", "qty": 2, "price_cents": 400}]}],
         "net_allocations": {"a": 2}, "settlement": {"leftovers": {"a": 0}, "shortages": {"a": 2},
         "profits_cents": {"a": 1000}, "fulfilled": {"a": 3}}},
    ]
    result = summarize_events(events, wholesale_price=10)
    assert result["formal_rounds"] == 1
    assert result["rounds"][0]["procurement_mean"] == 10
    assert result["rounds"][0]["excess_units"] == 0
    assert result["rounds"][0]["shortage_units"] == 2
    assert result["rounds"][0]["buy_fill_rate"] == 0.5
    assert result["rounds"][0]["mcp_mean_cents"] == 400
    assert result["supplier_revenue_cents"] == 10000
    assert result["rounds"][0]["supply_chain_profit_cents"] == 11000


def test_no_trade_market_price_is_missing_and_newsvendor_references_are_separate():
    event = {"round_index": 5, "warmup": False, "procurement": {"a": 137}, "orders": [],
        "markets": [{"members": ["a"], "price_cents": None, "self_volume": 0, "trades": []}],
        "net_allocations": {"a": 0}, "exogenous": {"a": {"p": 80, "D": 100}},
        "settlement": {"leftovers": {"a": 37}, "shortages": {"a": 0},
        "profits_cents": {"a": 7000}, "fulfilled": {"a": 100}}}
    result = summarize_events([event], wholesale_price=10)
    assert result["mcp_mean_cents"] is None
    feature = result["agent_features"]["a"]
    assert feature["nv_exact_discrete_quantity"] is not None
    assert feature["nv_continuous_reference_quantity"] is not None
    assert feature["q_over_200_rate"] == 0


def test_appendix_e_normal_approximation_uses_own_p_and_local_population_size():
    p, w, n = 80, 10, 3
    mu = 100 * n
    sigma = 200 * (n / 12) ** 0.5
    x_star = mu + sigma * NormalDist().inv_cdf((p - w) / p)
    assert abs(coopetitor_normal_approx_feature(x_star / n, p, w, n)) < 1e-10
    assert continuous_newsvendor_deviation(100 * (p - w) / p, p, w) >= 0

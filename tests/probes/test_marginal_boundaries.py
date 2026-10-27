"""Independent acceptance checks for the predeclared economic boundaries."""
from tools.probes.jev_marginal_value import scenarios, external_oracle


def oracle(Q, D, p=70, m=15):
    return external_oracle(dict(Q=Q, D=D, p=p, m=m, w=10, x=10))


def test_inventory_and_demand_crossings_are_partial_sacrifice_not_just_full_excess():
    assert oracle(27, 20)["optimal_actions"] == ["STOP"]
    assert oracle(28, 20)["optimal_actions"] == ["SELL"]
    assert oracle(40, 32)["optimal_actions"] == ["SELL"]
    assert oracle(40, 33)["optimal_actions"] == ["STOP"]


def test_equal_trade_and_retail_price_is_tie_only_when_capacity_is_needed():
    assert set(oracle(20, 20, m=70)["optimal_actions"]) == {"STOP", "SELL"}
    assert oracle(40, 20, m=70)["optimal_actions"] == ["SELL"]
    assert oracle(40, 20, m=1)["optimal_actions"] == ["SELL"]


def test_deduplication_reuses_shared_points_without_illegal_inventory():
    rows = scenarios()
    assert len(rows) == 76
    assert len({(r["Q"], r["D"], r["p"], r["m"], r["w"], r["x"]) for r in rows}) == 76
    assert sum(len(r["series"]) for r in rows) == 81
    assert all(r["Q"] >= r["x"] == 10 and 0 <= r["D"] <= 200 and 1 <= r["m"] <= 999 for r in rows)

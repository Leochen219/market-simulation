from capclaims.agents.candidates import (
    price_candidates,
    purchase_candidates,
    quantity_candidates,
    side_candidates,
)
from capclaims.agents.observations import build_observation
from capclaims.agents.policies import optimal_quantity, purchase_policy, trade_policy
from capclaims.config import RunConfig
import pytest


def test_candidate_grids_and_order_legality():
    assert len(purchase_candidates()) == 17
    assert [c.value for c in side_candidates(0)] == ["buy", "hold"]
    assert all(c.value != "sell" for c in side_candidates(0))
    assert [c.value for c in quantity_candidates("buy", 0)] == [5, 10, 25, 50, 75, 100, 150, 200, 300, 400]
    assert all(c.value <= 25 for c in quantity_candidates("sell", 25))
    assert len(price_candidates()) == 11


def test_optimal_quantity_and_policy_are_reproducible():
    assert optimal_quantity(100, 10) == 175
    config = RunConfig(policy="grid_rule")
    q = purchase_policy(config, 0, "a", 80)
    assert q in range(0, 401, 25)
    assert purchase_policy(config, 0, "a", 80) == q
    action = trade_policy(config, 0, "a", 80, q=100, d=160)
    assert action.side in {"buy", "sell", "hold"}
    if action.side == "buy":
        assert action.qty <= 60 and action.limit_cents is not None


def test_policy_helpers_reject_invalid_valuation():
    config = RunConfig(policy="grid_rule")
    with pytest.raises(ValueError):
        purchase_policy(config, 0, "a", 101)
    with pytest.raises(ValueError):
        trade_policy(config, 0, "a", 0, q=10, d=20)


def test_purchase_observation_excludes_private_realizations_and_filters_history():
    obs = build_observation(
        RunConfig(), 2, "a", 75,
        [{"p": 70, "D": 3, "Q": 5, "A": 1, "profit_cents": 10,
          "other_agent": "secret", "world_hash": "secret"}],
        phase="purchase", q=400, d=200,
    )
    assert obs["own_p"] == 75
    assert obs["rules"]["valuation_range"] == [51, 100]
    assert obs["rules"]["trade_price_grid"] == list(range(0, 101, 10))
    assert "profit_cents" in obs["rules"]["monetary_units"]
    assert "D" not in obs and "own_d" not in obs and "own_q" not in obs
    assert "world_hash" not in obs
    assert "other_agent" not in obs["history"][-1]

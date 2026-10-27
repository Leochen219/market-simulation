import pytest

from capclaims.config import RunConfig
from capclaims.randomness import (
    auction_rank,
    canonical_json,
    decision_rank,
    exogenous_value,
    rng_for_policy,
)


def test_config_defaults_and_bounds():
    config = RunConfig()
    assert (config.n_agents, config.rounds, config.wholesale_price) == (10, 15, 10)
    assert RunConfig(wholesale_price=50).wholesale_price == 50
    with pytest.raises(ValueError):
        RunConfig(n_agents=True)
    with pytest.raises(ValueError):
        RunConfig(wholesale_price=11)
    with pytest.raises(ValueError):
        RunConfig(wholesale_price=50, initial_cash_cents=6_000_000 - 1)


def test_config_yaml_is_strict(tmp_path):
    good = tmp_path / "run.yaml"
    good.write_text("master_seed: 7\nmarket_open: false\n", encoding="utf-8")
    assert RunConfig.load(good).master_seed == 7
    assert RunConfig.load(good).market_open is False
    bad = tmp_path / "bad.yaml"
    bad.write_text("master_seed: 7\nunknown: true\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown"):
        RunConfig.load(bad)


def test_randomness_is_keyed_and_treatment_independent():
    a = exogenous_value(9, 2, 3, "agent-1", "p")
    assert 51 <= a <= 100
    assert exogenous_value(9, 2, 3, "agent-1", "p") == a
    assert exogenous_value(9, 2, 3, "agent-1", "D") in range(201)
    x = rng_for_policy(9, 2, "random", 3, "agent-1", "purchase", "choice")
    y = rng_for_policy(9, 2, "random", 3, "agent-1", "purchase", "choice")
    assert [x.random() for _ in range(4)] == [y.random() for _ in range(4)]
    assert canonical_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'
    assert auction_rank(9, 2, 3, "agent-1") == auction_rank(9, 2, 3, "agent-1")
    assert decision_rank(9, 2, 3, "agent-1", "trade", "p_010") != decision_rank(
        9, 2, 3, "agent-1", "trade", "p_020"
    )

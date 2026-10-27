import itertools
import math

import pytest

from capclaims.conditional_behavior.protocol import (
    deterministic_draw,
    replay_choices,
    sample_categorical,
    validate_orders,
)
from capclaims.conditional_behavior.scoring import (
    expected_energy_score,
    fair_brier,
    fair_crps,
    fair_energy_score,
    feature_vector,
    incomplete_energy_interval,
    medoid_index,
)
from capclaims.conditional_behavior.baselines import expected_fair_score


def test_raw_categorical_sampler_keeps_every_positive_branch_and_renormalizes():
    probabilities = {"first": 0.999, "rare": 0.001, "zero": 0.0}
    assert sample_categorical(probabilities, 0.9999) == "rare"
    assert sample_categorical(probabilities, 0.0) == "first"
    with pytest.raises(ValueError):
        sample_categorical({"a": 0.0}, 0.25)
    with pytest.raises(ValueError):
        sample_categorical({"a": 10.0, "b": 20.0}, 0.25)


def test_seed_coordinates_are_stable_and_separate_by_study_arm_context_and_stage():
    coords = dict(seed=20261026, version="jev_conditional_behavior_v1", arm="E",
                 person="opaque", target=14, sample=2, stage="price:order:0:side:buy")
    assert deterministic_draw(**coords) == deterministic_draw(**coords)
    draws = {deterministic_draw(**{**coords, key: value}) for key, value in (
        ("arm", "D"), ("target", 15), ("sample", 3), ("stage", "price:order:1"))}
    assert len(draws | {deterministic_draw(**coords)}) == 5


def test_protocol_replay_enforces_five_orders_price_quantity_and_gross_sell_capacity():
    state = {"p": 70, "w": 50, "Q": 10, "D": 20, "round": 14, "history": []}
    actions = [{"kind": "order", "side": "sell", "qty": 10, "price": 999},
               {"kind": "stop"}]
    result = replay_choices(state, actions, interface="J1")
    assert result["orders"] == [{"side": "sell", "qty": 10, "price": 999}]
    assert result["status"] == "complete"
    for bad in (
        [{"side": "sell", "qty": 11, "price": 1}],
        [{"side": "buy", "qty": 0, "price": 1}],
        [{"side": "buy", "qty": 1, "price": 1000}],
        [{"side": "buy", "qty": 1, "price": 1}] * 6,
    ):
        with pytest.raises(ValueError):
            validate_orders(bad, initial_q=10)


def test_fair_scores_match_finite_distribution_expectations_and_degenerate_losses():
    forecast = [(0.0,), (2.0,)]
    truth = (1.0,)
    assert fair_crps([0.0, 2.0], 1.0) == 0.0  # K=2 bracketing case
    assert fair_energy_score(forecast, truth) == 0.0
    assert fair_crps([2.0, 2.0, 2.0, 2.0], 1.0) == 1.0
    assert fair_energy_score([(2.0,)] * 4, (1.0,)) == 1.0
    assert fair_brier([0, 1, 1, 0], 1) == pytest.approx(1 / 6)
    # For a true two-point distribution, expected fair CRPS equals its exact
    # expected absolute-error score under independent forecast draws.
    outcomes = [0.0, 2.0]
    empirical = sum(fair_crps([a, b], y) for a, b, y in itertools.product(outcomes, repeat=3)) / 8
    exact = 0.5 * sum(abs(x - y) for x in outcomes for y in outcomes) / 4
    assert empirical == exact
    empirical_p = {(0.0,): 0.5, (2.0,): 0.5}
    assert expected_energy_score(empirical_p, (1.0,)) == pytest.approx(0.5)
    records = [{"orders": [{"side": "buy", "qty": 1, "price": value}]} for value in (0, 2)]
    assert expected_fair_score(records, 1.0,
        lambda orders: orders[0]["price"], lambda x, y: abs(x - y)) == pytest.approx(0.5)


def test_feature_summary_and_failure_bound_preserve_extremes_and_order_invariance():
    empty = feature_vector([])
    extreme = feature_vector([{"side": "buy", "qty": 999, "price": 999},
                              {"side": "sell", "qty": 10, "price": 1}])
    assert empty == (0.0,) * 8
    assert extreme[2:6] == (0.2, 0.2, 4.995, 0.05)
    assert extreme[6:] == (9.99, 0.01)
    bound = incomplete_energy_interval(10)
    assert bound[0] == 0.0 and math.isfinite(bound[1]) and bound[1] > 0
    samples = [[{"side": "buy", "qty": 1, "price": 100}],
               [{"side": "buy", "qty": 2, "price": 200}],
               [{"side": "sell", "qty": 3, "price": 300}]]
    assert samples[medoid_index(samples)] == list(reversed(samples))[medoid_index(list(reversed(samples)))]


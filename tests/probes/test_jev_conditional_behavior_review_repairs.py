import math
import hashlib
import json
import sqlite3

from capclaims.conditional_behavior.analysis import (
    METRIC_REGISTRY, _curve_l2, _curves, _difference_summary, _expected_energy_curve,
    _metric_interval, _weighted_wasserstein_1d, _side_price_w1, _portfolio_exactness,
    _pooled_raw_w1,
)
from capclaims.conditional_behavior.baselines import expected_fair_score
from capclaims.conditional_behavior.scoring import feature_vector
from capclaims.conditional_behavior.runner import _provider_attempt_summary
from capclaims.conditional_behavior.cli import _save_private_score
from capclaims.conditional_behavior.cli import _safe_score
from capclaims.conditional_behavior.scoring import medoid_index


def test_step_curve_distance_matches_full_discrete_grid():
    left = [{"side": "buy", "qty": 7, "price": 40},
            {"side": "sell", "qty": 3, "price": 25}]
    right = [{"side": "buy", "qty": 2, "price": 38},
             {"side": "sell", "qty": 5, "price": 29}]
    a, b = _curves(left), _curves(right)
    expected = math.sqrt(sum((a[s][p] - b[s][p]) ** 2 for s in range(2) for p in range(999)))
    assert _curve_l2(left, right) == expected


def test_medoid_tie_uses_frozen_sample_number_after_input_shuffle():
    same = [{"side": "buy", "qty": 4, "price": 19}]
    assert medoid_index([same, same, same], sample_ids=[3, 1, 2]) == 1


def test_paired_summary_keeps_all_people_and_bootstraps_bounds():
    bounds = {f"p{i:02}": [-1.0, -0.5] if i == 0 else [-2.0, -1.0] for i in range(17)}
    result = _difference_summary(bounds, seed=5)
    assert len(result["participant_differences"]) == 17
    assert result["mean_difference_interval"] == [(-1 - 32) / 17, (-0.5 - 16) / 17]
    assert len(result["leave_one_participant_out_intervals"]) == 17
    assert result["participant_reweighting_95_interval"][1] < 0


def test_fixed_metric_registry_has_a_bound_for_an_all_failed_context():
    assert len(METRIC_REGISTRY) == len(set(METRIC_REGISTRY))
    assert all(_metric_interval(metric, 100)[0] <= _metric_interval(metric, 100)[1]
               for metric in METRIC_REGISTRY)


def test_unfinished_attempt_count_comes_from_durable_request_evidence(tmp_path):
    request_id = "r-1"
    folder = tmp_path / hashlib.sha256(request_id.encode()).hexdigest()
    folder.mkdir()
    (folder / "attempt.json").write_text(json.dumps({"request_id": request_id}))
    assert _provider_attempt_summary(tmp_path, None)["ambiguous_or_unfinished_attempts"] == 1
    (folder / "response.json").write_bytes(b"{}")
    (folder / "timing.json").write_bytes(b"{}")
    assert _provider_attempt_summary(tmp_path, None)["ambiguous_or_unfinished_attempts"] == 0
    (folder / "failure.json").write_bytes(b"{}")
    assert _provider_attempt_summary(tmp_path, None)["ambiguous_or_unfinished_attempts"] == 1


def test_unfinished_attempt_uses_global_ledger_state(tmp_path):
    request_id = "r-unknown"
    evidence = tmp_path / "wire"
    folder = evidence / hashlib.sha256(request_id.encode()).hexdigest()
    folder.mkdir(parents=True)
    (folder / "attempt.json").write_text(json.dumps({"request_id": request_id}))
    ledger = tmp_path / "ledger.sqlite"
    with sqlite3.connect(ledger) as db:
        db.execute("CREATE TABLE attempts (request_id TEXT, state TEXT)")
        db.execute("INSERT INTO attempts VALUES (?, ?)", (request_id, "unknown"))
    summary = _provider_attempt_summary(evidence, ledger)
    assert summary["ledger_states"] == {"unknown": 1}
    assert summary["ambiguous_or_unfinished_attempts"] == 1


def test_two_point_baseline_full_curve_matches_direct_exact_expectation():
    records = [{"orders": []}, {"orders": [{"side": "buy", "qty": 2, "price": 3}]}]
    truth = [{"side": "sell", "qty": 1, "price": 2}]
    flat = lambda orders: tuple(_curves(orders)[0]) + tuple(_curves(orders)[1])
    distance = lambda left, right: math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))
    direct = expected_fair_score(records, flat(truth), flat, distance)
    assert _expected_energy_curve(records, truth) == direct


def test_pooled_wasserstein_uses_marginals_not_paired_absolute_errors():
    # Pairwise MAE is 100 for the natural pairing, but both weighted marginals
    # are the same {0,100} distribution, so exact W1 is zero.
    assert _weighted_wasserstein_1d([0, 100], [0.5, 0.5], [100, 0], [0.5, 0.5]) == 0


def test_pooled_raw_order_total_w1_uses_context_and_draw_weights():
    contexts = {("p1", 14, "A"): {"K": 4, "stratum": "D>Q"},
                ("p2", 14, "A"): {"K": 4, "stratum": "D<Q"}}
    labels = {"p1:14": {"Q": 0, "orders": []},
              "p2:14": {"Q": 0, "orders": [{"side": "buy", "qty": 100, "price": 10}]}}
    samples = {("p1", 14, "A"): [[{"side": "buy", "qty": 100, "price": 10}]] * 4,
               ("p2", 14, "A"): [[]] * 4}
    result = _pooled_raw_w1(contexts, labels, samples)
    assert result["by_side"]["buy"]["w1"] == 0
    assert result["by_side"]["buy"]["human"]["mean"] == 50
    assert result["by_side"]["buy"]["model"]["mean"] == 50
    assert result["by_side"]["buy"]["human"]["population_sd"] == 50


def test_side_price_w1_normalizes_quantity_and_skips_absent_direction():
    one = [{"side": "buy", "qty": 1, "price": 25}]
    doubled = [{"side": "buy", "qty": 20, "price": 25}]
    assert _side_price_w1(one, doubled, "buy") == 0
    shifted = [{"side": "buy", "qty": 20, "price": 27}]
    assert _side_price_w1(one, shifted, "buy") == 2
    assert _side_price_w1([], doubled, "buy") is None


def test_stratified_raw_w1_requires_five_unique_participants():
    contexts = {(f"p{i}", 14, "A"): {"K": 4, "stratum": "D>Q"} for i in range(4)}
    labels = {f"p{i}:14": {"Q": 0, "orders": []} for i in range(4)}
    samples = {(f"p{i}", 14, "A"): [[]] * 4 for i in range(4)}
    result = _pooled_raw_w1(contexts, labels, samples, stratum="D>Q")
    assert result["status"] == "INSUFFICIENT_UNIQUE_PEOPLE"
    assert result["unique_people_n"] == 4


def test_exactness_distinguishes_order_repeats_from_repeated_forecasts():
    repeated_orders = [{"side": "buy", "qty": 3, "price": 10}] * 2
    truth = [{"side": "buy", "qty": 3, "price": 10}]
    details = _portfolio_exactness([repeated_orders, repeated_orders, truth, []], truth)
    assert details["mean_within_portfolio_repeated_order_count"] == 0.5
    assert details["within_portfolio_repeated_any_rate"] == 0.5
    assert details["mean_random_draw_exact_match_human"] == 0.25
    assert details["duplicate_forecast_sample_n"] == 1


def test_private_score_artifact_is_content_addressed_and_repeatable(tmp_path):
    score = {"preview_sha256": "a" * 64, "target_rows": [{"person_id": "private"}]}
    first = _save_private_score(tmp_path, score)
    second = _save_private_score(tmp_path, score)
    assert first == second
    assert len(first["sha256"]) == 64


def test_public_report_recursively_removes_identifiers_and_per_person_arrays():
    sentinel = "RAW-PARTICIPANT-IDENTIFIER-SENTINEL"
    private = {"status": "DESCRIPTIVE_CONDITIONAL_DISTRIBUTION", "preview_sha256": "a" * 64,
        "execution_mode": "fake", "planned_targets": 34, "completed_sample_rows": 2,
        "target_rows": [{"person_id": sentinel}], "participants_by_arm": {sentinel: {}},
        "primary_E_vs_state": {"participant_differences": {sentinel: [-1, 0]},
            "leave_one_participant_out_intervals": [[-1, 0]] * 17,
            "aggregate_difference_interval": [-1, 0]},
        "paired_energy_contrasts": {"C_minus_D": {"participant_differences": {sentinel: [0, 1]},
            "leave_one_participant_out_intervals": [[0, 1]] * 17, "mean_difference_interval": [0, 1]}}}
    public = _safe_score(private)
    text = json.dumps(public)
    assert sentinel not in text
    assert "participant_differences" not in text
    assert "leave_one_participant_out_intervals" not in text
    assert public["primary_E_vs_state"]["aggregate_difference_interval"] == [-1, 0]

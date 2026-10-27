from capclaims.behavioral_memory.analysis import chinese_report, public_report, score_trajectories
from capclaims.behavioral_memory.design import build_schedule


def fixture():
    people = [f"participant-private-{i}" for i in range(18)]
    schedule = build_schedule(people)
    labels = {person: {"orders": [], "profit_cents": i} for i, person in enumerate(people)}
    results = [{**row, "status": "COMPLETE", "orders": [], "actions": [{"kind": "count", "count": 0}],
                "trace": []} for row in schedule]
    return schedule, labels, results


def test_scoring_uses_fixed_two_sample_denominator_and_scrubs_public_people():
    schedule, labels, results = fixture()
    score = score_trajectories(schedule, labels, results)
    assert score["status"] == "COMPLETE"
    assert score["planned_trajectories"] == score["completed_trajectories"] == 108
    assert score["contrasts"]["M1-M0"]["fair_energy_score"] == 0
    assert len(score["contrasts"]["M1-M0"]["per_person_deltas"]) == 18
    public = public_report(score)
    assert "per_person" not in public
    assert "participant-private" not in str(public)
    assert "描述性" in chinese_report(score)


def test_any_missing_trajectory_returns_incomplete_without_reduced_k_scores():
    schedule, labels, results = fixture()
    score = score_trajectories(schedule, labels, results[:-1])
    assert score["status"] == "INCOMPLETE"
    assert score["planned_trajectories"] == 108
    assert score["completed_trajectories"] == 107
    assert "per_arm" not in score


def test_price_errors_keep_absence_missing_and_report_duplicate_order_rates():
    schedule = build_schedule([f"p{i}" for i in range(18)])
    labels = {f"p{i}": {"orders": [], "profit_cents": 0} for i in range(18)}
    results = []
    for row in schedule:
        orders = ([{"side": "buy", "qty": 1, "price": 99}] * 2
                   if row["person_id"] == "p0" and row["arm"] == "M1" and row["repeat"] == 0 else [])
        results.append({**row, "status": "COMPLETE", "orders": orders,
                        "trace": [], "actions": [{"kind": "count", "count": len(orders)}]})
    score = score_trajectories(schedule, labels, results)
    person = score["per_person"]["p0"]["M1"]
    assert person["mean_duplicate_order_count"] == 0.5
    assert person["any_duplicate_order_rate"] == 0.5
    assert person["abs_error_buy_price"] is None
    assert person["paired_price_error_coverage_buy_price"] == 0


def test_a_failed_result_keeps_the_fixed_108_denominator_and_status_counts():
    schedule, labels, results = fixture()
    results[-1] = {**results[-1], "status": "FAILED", "error_type": "ProviderTimeout"}
    score = score_trajectories(schedule, labels, results)
    assert score["status"] == "INCOMPLETE"
    assert score["planned_trajectories"] == 108
    assert score["completed_trajectories"] == 107
    assert score["failed_trajectory_n"] == 1
    assert score["result_status_counts"] == {"COMPLETE": 107, "FAILED": 1, "INCOMPLETE": 0, "ABSENT": 0}
    assert "per_arm" not in score


def test_provider_choice_diagnostics_separate_missing_and_allow_probability_ties():
    schedule, labels, results = fixture()
    first = next(r for r in results if r["person_id"] == "participant-private-0"
                 and r["arm"] == "M1" and r["repeat"] == 0)
    second = next(r for r in results if r["person_id"] == "participant-private-0"
                  and r["arm"] == "M1" and r["repeat"] == 1)
    first["trace"] = [{"raw_probabilities": {"a": 0.5, "b": 0.5},
                        "provider_choice": None, "selected": "a"}]
    second["trace"] = [{"raw_probabilities": {"a": 0.5, "b": 0.5000000000005},
                         "provider_choice": "a", "selected": "b"}]
    score = score_trajectories(schedule, labels, results)
    metrics = score["per_person"]["participant-private-0"]["M1"]
    assert metrics["provider_choice_present_n"] == 1
    assert metrics["provider_choice_missing_n"] == 1
    assert metrics["provider_choice_missing_rate"] == 0.5
    assert metrics["provider_choice_non_map_rate"] == 0
    assert metrics["sample_differs_from_provider_choice_rate"] == 1

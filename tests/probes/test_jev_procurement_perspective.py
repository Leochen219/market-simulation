import pytest

from capclaims.procurement_perspective import (
    ARMS, StudyBudget, build_choice_prefixes, decode_prediction,
    development_winner, filtered_training, select_participants,
)
from capclaims.personalization import Episode
from capclaims.backends.jev_instructions import build_instruction
from capclaims.domain.types import Candidate
from capclaims.engine.protocol import DecisionRequest
from capclaims.backends.jev import JevBackend
from pathlib import Path
import asyncio
import uuid
from capclaims.procurement_perspective import (
    HISTORY_FOR_ARM, MAX_PREDICTIONS, _sdk_mock, build_synthetic_study, execute_study,
    _metrics, maximum_http_overlap,
)


def test_seeded_participant_selection_uses_only_roster_identifiers():
    roster = [f"person-{i:02}" for i in range(35)]
    assert select_participants(roster, seed=20261024) == select_participants(roster, seed=20261024)
    assert len(select_participants(roster, seed=20261024)) == 3
    assert set(select_participants(roster, seed=20261024)) <= set(roster)


def test_prefixes_reach_all_legal_quantities_and_cap_at_47_nodes():
    nodes = build_choice_prefixes(0, 400)
    assert len(nodes) == 47
    assert nodes[0] == ""
    from capclaims.paper.codec import legal_prefix_choices
    assert legal_prefix_choices("", 400) == (0, 1, 2, 3, 4)
    assert legal_prefix_choices("4", 400) == (0,)
    assert len(build_choice_prefixes(0, 400)) < 60


def test_greedy_decoder_never_uses_truth_to_change_predicted_prefix():
    assert decode_prediction([0, 0, 1]) == 1
    assert decode_prediction([4, 0, 0]) == 400
    with pytest.raises(ValueError, match="legal"):
        decode_prediction([4, 0, 1])
    with pytest.raises(ValueError, match="tie"):
        decode_prediction([None, 0, 0])


def test_training_history_filters_rounds_before_target_conditioning():
    eps = [Episode("p", "GA 10", "GA", n, "procurement", {"p": 70, "w": 10}, {"Q": n})
            for n in range(1, 16)]
    train = filtered_training(eps)
    assert [e.round for e in train] == list(range(1, 9))


def test_development_winner_uses_mae_then_baseline_first():
    scores = {ARMS[0]: [1.0, 2.0], ARMS[1]: [1.5, 1.5]}
    assert development_winner(scores) == ARMS[0]
    tied = {ARMS[0]: [2.0], ARMS[1]: [2.0]}
    assert development_winner(tied) == ARMS[0]


def test_shared_budget_enforces_one_global_request_ceiling():
    budget = StudyBudget(max_requests=2, max_estimated_bytes=100)
    budget.reserve(1, 40)
    budget.reserve(1, 60)
    with pytest.raises(ValueError, match="shared study budget"):
        budget.reserve(1, 1)


def test_two_prompt_profiles_differ_only_in_perspective_and_reject_target_labels():
    request = DecisionRequest("opaque", 8, "paper:purchase:h", {
        "own_p": 70, "wholesale_price_dollars": 10, "round": 9,
        "numeric_prefix": "", "choice_minimum": 0, "choice_maximum": 400,
        "decision_field": "purchase", "phase": "procurement"},
        tuple(Candidate(f"d{d}", d, str(d)) for d in range(5)))
    first = build_instruction(request, "procurement_first_v1")
    third = build_instruction(request, "procurement_third_v1")
    first_prefix = "You are this participant, choosing your own procurement quantity. "
    third_prefix = "You are an observer, predicting this participant's procurement quantity. "
    assert first.removeprefix(first_prefix) == third.removeprefix(third_prefix)
    with pytest.raises(ValueError, match="realized outcome"):
        build_instruction(DecisionRequest(request.agent_id, request.round_index, request.stage,
            {**request.observation, "D": 40}, request.candidates), "procurement_first_v1")


def test_study_objective_is_separate_from_optimizer_and_conditioner():
    class Client: pass
    params = dict(api_key="dummy", rps=40, tps=50_000, max_requests=480,
        max_estimated_input_tokens=3_200_000, client=Client(), load_environment=False)
    with pytest.raises(ValueError, match="procurement prediction profiles"):
        JevBackend(**params, instruction_profile="procurement_first_v1", objective="optimizer")
    with pytest.raises(ValueError, match="study-owned history"):
        JevBackend(**params, instruction_profile="procurement_first_v1",
            objective="procurement_prediction_v1", participant_conditioner=object())


def test_synthetic_plan_freezes_four_arms_without_future_history_or_roster_as_market_size():
    plan = build_synthetic_study()
    assert len(plan["_sessions"]) == MAX_PREDICTIONS == 126
    assert plan["expected_prediction_calls"] == 378
    assert plan["unique_current_input_state_n"] == 21
    assert "market_size" not in plan
    for arm, history in HISTORY_FOR_ARM.items():
        sessions = [session for session in plan["_sessions"] if session["arm"] == arm]
        assert len(sessions) == (42 if arm in {"first_own", "third_own"} else 21)
        for session in sessions:
            episodes = session["state"]["participant_history"]["episodes"]
            assert len(episodes) <= 3
            assert all(set(episode["action"]) == {"Q"} for episode in episodes)
            if history == "none":
                assert episodes == []
            if history == "shuffled":
                assert session["donor_index"] != session["participant"]
                assert len(episodes) <= 3
    rules = plan["_sessions"][0]["state"]["public_instructions"]
    assert "five blind mixed buy/sell" in rules["trading"]
    assert "uniform midpoint" in rules["clearing"]
    assert "unknown at procurement" in rules["profit"]
    assert plan["roster_n"] == 35 and "market_size" not in plan


def test_full_pipeline_uses_only_injected_mock_sdk(tmp_path):
    plan = build_synthetic_study()
    out = Path(__file__).resolve().parents[2] / "artifacts" / "participant-data" / f"synthetic-worker-{uuid.uuid4().hex}"
    fake = _sdk_mock()
    result = asyncio.run(execute_study(plan, out, fake_client=fake))
    assert result["status"] == "COMPLETE"
    assert result["prediction_n"] == MAX_PREDICTIONS
    assert result["budget"]["requests"] <= 480
    assert fake.closed
    assert (out / "frozen-plan.json").exists()
    assert (out / "private-predictions.json").exists()
    assert (out / "safe-aggregate.json").exists()


def test_output_path_cannot_escape_ignored_participant_artifacts():
    plan = build_synthetic_study()
    outside = Path(__file__).resolve().parents[2] / "outside-procurement-study"
    with pytest.raises(ValueError, match="participant-data"):
        asyncio.run(execute_study(plan, outside, fake_client=_sdk_mock()))


def test_visible_history_range_and_shuffled_donor_diagnostics_are_labeled_correctly():
    base = {"sequence": 0, "participant": "0", "round": 9, "history_condition": "own",
        "state": {"participant_history": {"episodes": [
            {"action": {"Q": 80}}, {"action": {"Q": 90}}, {"action": {"Q": 100}}]}},
        "donor_training_quantities": [80, 90]}
    prediction = [{"sequence": 0, "predicted_quantity": 95, "status": "complete"}]
    own = _metrics([base], prediction, {"0:9": 95}, {"0": [10, 20]})["copy_diagnostics"]
    assert own["range_diagnostic_n"] == 1
    assert own["within_visible_history_range_n"] == 1
    assert "within_own_training_history_range_n" not in own
    assert own["donor_diagnostic_n"] == 0
    shuffled = {**base, "history_condition": "shuffled"}
    donor = _metrics([shuffled], prediction, {"0:9": 95}, {"0": [10, 20]})["copy_diagnostics"]
    assert donor["donor_diagnostic_n"] == 1
    assert donor["copied_shuffled_donor_last_q_n"] == 0


def test_http_overlap_uses_half_open_start_finish_intervals():
    timings = [
        {"started_at": 1.0, "finished_at": 3.0},
        {"started_at": 2.0, "finished_at": 4.0},
        {"started_at": 3.0, "finished_at": 5.0},
    ]
    assert maximum_http_overlap(timings) == 2
    assert maximum_http_overlap([]) == 0
    with pytest.raises(ValueError, match="timestamps"):
        maximum_http_overlap([{"started_at": 2.0, "finished_at": 1.0}])

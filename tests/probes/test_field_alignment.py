import asyncio
import pytest

from capclaims.continuation_study import _serialize as old_serialize
from capclaims.injection_study import _maximal_legal_requests
from capclaims.field_alignment_context import inverse_aligned_state, render_field_state
from capclaims.field_alignment_analysis import summarize_repeated_pairs
from capclaims.field_alignment_study import serialize_alignment_wire, _session_schedule


def _requests():
    return _maximal_legal_requests([])


def test_aligned_renderer_renames_only_four_keys_and_roundtrips_without_mutation():
    req = _requests()[-1]
    before = req.observation.copy()
    control = render_field_state(req, "C_joint_control")
    aligned = render_field_state(req, "C_joint_aligned")
    mapping = {"own_p": "retail_price_dollars_per_unit", "own_q_units": "initial_inventory_units",
        "own_d_units": "realized_demand_units", "round": "game_round"}
    assert list(aligned) == [mapping.get(key, key) for key in control]
    assert inverse_aligned_state(aligned) == control
    assert [(k, type(v), v) for k, v in aligned.items()] == [
        (mapping.get(k, k), type(v), v) for k, v in control.items()]
    assert req.observation == before
    for key in ("partial_orders", "own_side", "own_qty_units", "own_history"):
        assert aligned[key] == control[key]
    assert not set(mapping).intersection(aligned)


def test_aligned_inverse_rejects_missing_and_alias_collision():
    state = render_field_state(_requests()[-1], "C_joint_aligned")
    with pytest.raises(ValueError):
        inverse_aligned_state({k: v for k, v in state.items() if k != "game_round"})
    with pytest.raises(ValueError):
        inverse_aligned_state({**state, "round": 14})


def test_ten_initial_cases_are_distinct_paired_wire_inputs():
    sessions = _session_schedule(["P1", "P2", "P3", "P4", "P5", "P6"])
    assert len(sessions) == 24
    assert len({(row["person_id"], row["arm"]) for row in sessions[:10]}) == 10
    assert sessions == _session_schedule(["P1", "P2", "P3", "P4", "P5", "P6"])


@pytest.mark.asyncio
async def test_control_wire_matches_original_c_joint_serializer_for_every_prefix():
    from capclaims.backends.jev import MODEL
    requests = _requests()
    assert len(requests) == 11
    for request in requests:
        old = await old_serialize(request, True, "C_joint")
        new = await serialize_alignment_wire(request, "C_joint_control")
        assert (new.method, new.url, new.body, new.body_sha256) == (
            old.method, old.url, old.body, old.body_sha256)
        import json
        wire_state = json.loads(new.body)["state"]
        assert wire_state == render_field_state(request, "C_joint_control")
        assert MODEL in json.loads(new.body).get("model", MODEL)


def test_repeat_analysis_collapses_before_paired_person_summary_and_counts_casewise_excess():
    rows = []
    for i, person in enumerate("abcdef"):
        for arm in ("C_joint_control", "C_joint_aligned"):
            offset = 1 if arm.endswith("aligned") else 0
            buy = 10 if person == "a" else i
            for repeat, error in ((0, i+offset), (1, i+offset+2)):
                pred_buy = buy + (10 if person == "a" and repeat == 0 else -10 if person == "a" else 0)
                rows.append({"person_id": person, "arm": arm, "repeat": repeat,
                    "primary_curve_error": error, "duplicate_order_n": i,
                    "predicted_gross_buy": pred_buy, "human_gross_buy": buy,
                    "positive_excess_buy": max(0, pred_buy-buy), "signed_buy_bias": pred_buy-buy,
                    "absolute_buy_error": abs(pred_buy-buy)})
    summary = summarize_repeated_pairs(rows)
    assert summary["n_people"] == 6
    assert summary["per_arm"]["C_joint_control"]["primary_curve_error"] == pytest.approx(3.5)
    assert summary["paired_aligned_minus_control"]["primary_curve_error"]["mean_aligned_minus_control"] == 1
    assert summary["repeat_consistency"]["mean_absolute_primary_error_difference"] == 2
    assert summary["private_person_scores"][0]["positive_excess_buy"] == 5
    assert "leave_one_person_out_mean_min" in summary["paired_aligned_minus_control"]["primary_curve_error"]
    assert "leave_one_person_out_mean" not in summary["paired_aligned_minus_control"]["primary_curve_error"]


def test_actual_portfolio_boolean_metrics_are_reported_as_rates():
    from capclaims.paper.orders import PaperOrder
    from capclaims.portfolio_metrics import portfolio_comparison
    truth = [PaperOrder("human", "P", "buy", 3, 50)]
    exact = portfolio_comparison(truth, truth)
    empty_miss = portfolio_comparison([], truth)
    rows = []
    for person in "abcdef":
        for arm, score in (("C_joint_control", exact), ("C_joint_aligned", empty_miss)):
            for repeat in (0, 1):
                rows.append({"person_id": person, "arm": arm, "repeat": repeat, **score})
    result = summarize_repeated_pairs(rows)
    control, aligned = result["per_arm"]["C_joint_control"], result["per_arm"]["C_joint_aligned"]
    assert control["empty_agreement"] == 1.0
    assert control["exact_order_multiset_match"] == 1.0
    assert control["merged_same_side_price_match"] == 1.0
    assert control["side_class_agreement"] == 1.0
    assert aligned["empty_agreement"] == 0.0
    assert aligned["exact_order_multiset_match"] == 0.0
    assert aligned["side_class_agreement"] == 0.0
    assert result["paired_aligned_minus_control"]["side_class_agreement"]["mean_aligned_minus_control"] == -1.0

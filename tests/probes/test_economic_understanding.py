import pytest

from tools.probes.economic_understanding_probe import (
    economic_cases, numeric_targets, action_profit_dollars, evaluate_readiness,
)


def test_frozen_cases_have_unique_profit_maximizers_and_include_premium_selling():
    cases = economic_cases()
    assert len(cases) >= 10
    for case in cases:
        profits = {a["key"]: action_profit_dollars(case, a) for a in case["actions"]}
        ranked = sorted(profits, key=profits.get, reverse=True)
        assert profits[ranked[0]] > profits[ranked[1]]
        assert ranked[0] == case["expected"]
    premium = next(c for c in cases if c["id"] == "sell_needed_premium")
    assert premium["expected"] == "sell"
    assert premium["Q"] == premium["D"]


def test_numeric_set_covers_repeated_digits_neighbors_and_paper_endpoints():
    targets = numeric_targets()
    qs = {t["target"] for t in targets if t["field"] == "purchase"}
    ps = {t["target"] for t in targets if t["field"] == "price"}
    assert {0,137,332,333,334,400} <= qs
    assert {1,333,999} <= ps
    assert set("0123456789") <= set("".join(f'{t["target"]:03d}' for t in targets))


def test_gate_fails_closed_for_any_wrong_critical_answer_or_missing_evidence():
    assert not evaluate_readiness([], [])
    assert evaluate_readiness([{"correct": True}], [{"correct": True}])
    assert not evaluate_readiness([{"correct": False}], [{"correct": True}])
    assert not evaluate_readiness([{"correct": True}], [{"correct": False}])


def test_numeric_probe_preserves_versioned_production_mask_and_procurement_privacy():
    from tools.probes.economic_understanding_probe import numeric_requests
    for interface in ("legacy_v1", "economic_v2"):
        for request, _ in numeric_requests(interface):
            obs = request.observation
            if interface == "legacy_v1":
                assert set(obs["legal_prefix_mask"]) == {c.value for c in request.candidates}
            else:
                assert "legal_prefix_mask" not in obs
            if obs["phase"] == "procurement":
                assert not {"own_d", "own_d_units", "own_q", "own_q_units"} & obs.keys()

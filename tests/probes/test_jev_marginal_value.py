from __future__ import annotations

from dataclasses import asdict
import pytest

from capclaims.domain.types import Candidate
from capclaims.engine.protocol import DecisionReply, DecisionRequest
from tools.probes.jev_marginal_value import (
    build_plan,
    external_oracle,
    main,
    scenarios,
    score,
    summarize,
)


def test_grid_counts_and_exact_oracle_boundaries():
    rows = scenarios()
    assert len(rows) == 76
    assert sum(len(row["series"]) for row in rows) == 81
    assert len({(r["Q"], r["D"], r["p"], r["m"], r["w"], r["x"]) for r in rows}) == 76

    def optimum(Q, D, p=70, m=15):
        return external_oracle({"Q": Q, "D": D, "p": p, "m": m, "w": 10, "x": 10})

    assert optimum(27, 20)["optimal_actions"] == ["STOP"]
    assert optimum(28, 20)["optimal_actions"] == ["SELL"]
    assert optimum(40, 32)["optimal_actions"] == ["SELL"]
    assert optimum(40, 33)["optimal_actions"] == ["STOP"]
    assert optimum(20, 20, m=70)["optimal_actions"] == ["STOP", "SELL"]
    assert optimum(40, 20, m=1)["optimal_actions"] == ["SELL"]


def test_plan_freezes_456_calls_in_10_request_waves_without_oracle_leakage():
    plan = build_plan()
    assert plan["count"] == 456
    assert sum(len(wave) for wave in plan["waves"]) == 456
    assert max(map(len, plan["waves"])) == 10
    assert plan["estimated_input_tokens"] <= plan["token_cap"] == 1_500_000
    assert plan["request_cap"] == 456
    assert len(plan["orderedwirehash"]) == 64

    first = plan["waves"][0][0]
    assert list(first["wire"]["state"]) == [
        "retail_value_per_unit_dollars",
        "procured_capacity_units",
        "customer_demand_units",
        "unit_procurement_cost_dollars",
        "procurement_already_paid",
    ]
    assert first["wire"]["criteria"]["c0"] == "STOP: no trade; the already-paid procurement cost is not refunded"
    assert "pays you" in first["wire"]["criteria"]["c1"]
    forbidden = ("delta_profit", "optimal_actions", "excess", "expected", "series", "case_id", "repetition")
    wire_text = str(first["wire"]).lower()
    assert not any(word in wire_text for word in forbidden)

    by_economic_state = {}
    for wave in plan["waves"]:
        for row in wave:
            state = tuple(row[key] for key in ("Q", "D", "p", "m", "w", "x"))
            by_economic_state.setdefault(state, []).append(row)
    assert set(map(len, by_economic_state.values())) == {6}
    for repeated in by_economic_state.values():
        assert len({str(r["wire"]) for r in repeated if r["order"] == "original"}) == 1
        assert len({str(r["wire"]) for r in repeated if r["order"] == "reversed"}) == 1
        assert repeated[0]["wire_sha256"] != next(
            r["wire_sha256"] for r in repeated if r["order"] != repeated[0]["order"]
        )


def test_score_uses_candidate_semantics_and_separates_probability_ties():
    request = DecisionRequest(
        agent_id="probe_000", round_index=0, stage="jev:one_step_trade",
        observation={"retail_value_per_unit_dollars": 70, "procured_capacity_units": 40,
                     "customer_demand_units": 20, "unit_procurement_cost_dollars": 10,
                     "procurement_already_paid": True},
        candidates=(Candidate("c1", "SELL", "sell"), Candidate("c0", "STOP", "stop")),
    )
    external = external_oracle({"Q": 40, "D": 20, "p": 70, "m": 15, "w": 10, "x": 10})
    plan_row = {
        "request": asdict(request), "expected": external["optimal_actions"],
        "delta_profit": external["delta_profit"], "economic_tie": False,
        "condition": "surplus_m", "case_id": "m15", "series": ["surplus_m"],
        "repetition": 0, "order": "reversed", "wire_sha256": "a" * 64,
        "estimated_input_tokens": 999,
    }
    reply = DecisionReply.from_request(request, {"c0": 0.2, "c1": 0.8})
    row = score(plan_row, reply)
    assert row["selected"] == "SELL"
    assert row["correct"] is True
    assert row["correct_margin"] == pytest.approx(0.6)
    assert row["sell_probability"] == 0.8

    tied = DecisionReply.from_request(request, {"c0": 0.5, "c1": 0.5})
    row = score(plan_row, tied)
    assert row["selected"] == "TOP_TIE"
    assert row["correct"] is False
    assert row["top_candidate_ids"] == ["c1", "c0"]


def test_summaries_report_economic_ties_price_heuristics_and_repeat_variation():
    plan = build_plan()
    rows = []
    for wave in plan["waves"]:
        for item in wave:
            request_data = item["request"]
            request = DecisionRequest(**{
                **request_data,
                "candidates": tuple(Candidate(**c) for c in request_data["candidates"]),
            })
            action = "SELL" if item["expected"] == ["SELL"] else "STOP"
            probabilities = {c.candidate_id: (0.8 if c.value == action else 0.2)
                             for c in request.candidates}
            rows.append(score(item, DecisionReply.from_request(request, probabilities)))
    summary = summarize(rows, plan)
    assert summary["unique_accuracy"]["n"] == 450
    assert summary["economic_ties"]["n"] == 6
    assert summary["price_heuristics"]["strict_m_gt_p"]["agreement_rate"] is not None
    assert summary["price_heuristics"]["weak_m_ge_p"]["agreement_rate"] is not None
    strict = summary["price_heuristics"]["strict_m_gt_p"]
    weak = summary["price_heuristics"]["weak_m_ge_p"]
    assert strict["n"] == weak["n"] == 450
    assert weak["agreements"] == strict["agreements"] + 6
    assert strict["discriminating_n"] == weak["discriminating_n"] + 6
    assert summary["repeat_variation"]["scenarios_with_action_variation"] == 0
    assert len(summary["series"]) == 6


def test_preview_never_constructs_backend_or_reads_execute_path(tmp_path, monkeypatch):
    import tools.probes.jev_marginal_value as probe

    def forbidden_backend(*args, **kwargs):
        raise AssertionError("offline preview must not construct a provider backend")

    monkeypatch.setattr(probe, "JevBackend", forbidden_backend)
    output = tmp_path / "preview"
    assert main(["--output-dir", str(output)]) == 0
    assert (output / "preview.json").is_file()
    assert not (output / "frozen-plan.json").exists()

from __future__ import annotations

from dataclasses import asdict
import json

from capclaims.backends.jev_instructions import build_instruction
from capclaims.domain.types import Candidate
from capclaims.engine.protocol import DecisionReply, DecisionRequest
from tools.probes import jev_input_ablation as probe


def _request_for(condition: str, case_id: str, order: str = "original"):
    plan = probe.build_plan()
    return next(row for wave in plan["waves"] for row in wave
                if row["condition"] == condition and row["case_id"] == case_id
                and row["order"] == order and row["repetition"] == 0)


def test_fixture_matrix_has_six_cases_and_balanced_unique_optima():
    fixtures = probe.fixtures()
    assert len(fixtures) == 6
    assert [f.cohort for f in fixtures] == ["seen"] * 3 + ["new"] * 3
    assert len(probe.conditions()) == 11
    winners = {}
    for fixture in fixtures:
        profits = {side: probe._profit(fixture, side, qty, price)
                   for side, qty, price in (("STOP", 0, 0), *fixture.actions)}
        max_profit = max(profits.values())
        assert list(profits.values()).count(max_profit) == 1
        winners[fixture.case_id] = next(side for side, value in profits.items() if value == max_profit)
    assert winners == {
        "sell_surplus": "SELL", "expiry_zero_demand": "SELL", "sell_needed_premium": "SELL",
        "new_surplus": "SELL", "new_stop_needed_capacity": "STOP", "new_buy_shortage": "BUY",
    }


def test_wire_isolation_and_no_current_payoff_or_oracle_leakage():
    base = _request_for("market_0_history_none", "sell_surplus")
    market = _request_for("market_1_history_none", "sell_surplus")
    history = _request_for("market_0_history_333", "sell_surplus")
    state = base["wire"]["state"]
    assert set(state) == {"retail_value_per_unit_dollars", "procured_capacity_units",
                          "customer_demand_units", "unit_procurement_cost_dollars",
                          "procurement_already_paid"}
    assert "profit" not in json.dumps(state).lower()
    assert "oracle" not in json.dumps(base["wire"]).lower()
    assert "case_id" not in json.dumps(base["wire"])
    assert "condition" not in json.dumps(base["wire"])
    assert "repetition" not in json.dumps(base["wire"])
    assert market["wire"]["state"]["public_instructions"] == probe._market_rules()
    assert history["wire"]["state"]["own_history"] == probe._history_observation(333)[0]
    assert "own_pretrade_profit" not in json.dumps(history["wire"]["state"])
    assert "pretrade_profit" not in json.dumps(history["wire"]["state"])
    assert all("expected_profits_dollars" not in json.dumps(row["wire"])
               for wave in probe.build_plan()["waves"] for row in wave)


def test_factorial_changes_only_the_named_context_and_history_factors():
    b = _request_for("market_0_history_none", "sell_surplus")["wire"]["state"]
    m = _request_for("market_1_history_none", "sell_surplus")["wire"]["state"]
    h10 = _request_for("market_0_history_10", "sell_surplus")["wire"]["state"]
    h333 = _request_for("market_0_history_333", "sell_surplus")["wire"]["state"]
    assert m.keys() == b.keys() | {"public_instructions"}
    assert h10.keys() == b.keys() | {"own_history", "own_history_fields"}
    assert {k: v for k, v in h10.items() if k != "own_history"} == {
        **b, "own_history_fields": h10["own_history_fields"]}
    assert h10["own_history"] == probe._history_observation(10)[0]
    assert h333["own_history"] == probe._history_observation(333)[0]
    assert h333["own_history_fields"] == h10["own_history_fields"]
    rows10 = [list(row) for row in h10["own_history"]]
    rows333 = [list(row) for row in h333["own_history"]]
    rows10 = [row[:4] + [row[4]] + row[5:] for row in rows10]
    assert len(rows10) == len(rows333)
    assert all(row[0:4] == other[0:4] and row[5:] == other[5:]
               for row, other in zip(rows10, rows333, strict=True))


def test_common_instruction_and_label_salience_arms():
    requests = [
        _request_for("market_0_history_none", "sell_surplus"),
        _request_for("clean_labels_bare", "sell_surplus"),
        _request_for("clean_labels_sell_cash", "sell_surplus"),
        _request_for("clean_labels_stop_cost", "sell_surplus"),
    ]
    assert len({row["wire"]["instruction"] for row in requests}) == 1
    assert all(build_instruction(None, "ablation_v1") == row["wire"]["instruction"] for row in requests)
    criteria = [{v["candidate_id"]: v["label"] for v in row["wire"]["criteria"].values()}
                if False else row["wire"]["criteria"] for row in requests]
    assert "pays you" not in criteria[1]["c1"]
    assert "pays you" in criteria[2]["c1"]
    assert "not refunded" not in criteria[1]["c0"]
    assert "not refunded" in criteria[3]["c0"]
    instruction = requests[0]["wire"]["instruction"]
    assert "paid and will not be refunded" in instruction
    assert "Customer fulfillment is capped at the stated demand" in instruction
    assert "background only" in instruction


def test_plan_has_396_unique_requests_interleaved_and_three_identical_wire_repeats():
    plan = probe.build_plan()
    assert plan["count"] == 396
    assert len(plan["waves"]) == 40
    assert len(plan["waves"][-1]) == 6
    assert plan["estimated_input_tokens"] <= plan["token_cap"] == 1_500_000
    rows = [row for wave in plan["waves"] for row in wave]
    assert len({row["request"]["agent_id"] for row in rows}) == 396
    assert len({row["wire_sha256"] for row in rows}) < len(rows)
    assert probe.build_plan()["orderedwirehash"] == plan["orderedwirehash"]
    assert probe.build_plan()["matrixhash"] == plan["matrixhash"]
    key = ("market_0_history_none", "sell_surplus", "original")
    repeats = [r for r in rows if (r["condition"], r["case_id"], r["order"]) == key]
    assert len(repeats) == 3
    assert len({r["wire_sha256"] for r in repeats}) == 1
    assert len({r["request"]["agent_id"] for r in repeats}) == 3
    assert len({(r["condition"], r["case_id"], r["order"], r["repetition"]) for r in rows}) == 396


def test_order_reversal_keeps_candidate_ids_bound_to_the_same_actions():
    original = _request_for("market_0_history_none", "expiry_zero_demand", "original")["wire"]["criteria"]
    reversed_ = _request_for("market_0_history_none", "expiry_zero_demand", "reversed")["wire"]["criteria"]
    assert original.keys() == reversed_.keys() == {"c0", "c1", "c2"}
    assert original == reversed_
    assert list(original) == ["c0", "c1", "c2"]
    assert list(reversed_) == ["c2", "c1", "c0"]
    original_row = _request_for("market_0_history_none", "expiry_zero_demand", "original")
    reversed_row = _request_for("market_0_history_none", "expiry_zero_demand", "reversed")
    assert original_row["wire_sha256"] != reversed_row["wire_sha256"]
    new_buy = _request_for("market_0_history_none", "new_buy_shortage")
    criteria = new_buy["wire"]["criteria"]
    assert set(criteria) == {"c0", "c1", "c2"}


def test_summarize_keeps_strict_ties_top_sets_and_probability_vectors():
    source_plan_row = _request_for("market_0_history_none", "sell_surplus")
    request_data = source_plan_row["request"]
    request = DecisionRequest(**{**request_data,
        "candidates": tuple(Candidate(**c) for c in request_data["candidates"])})
    reply = DecisionReply.from_request(request, {"c0": 0.5, "c1": 0.5}, backend="jev")
    row = probe._row_from_reply({"condition": "x", "case_id": "y", "repetition": 0,
        "order": "original", "cohort": "seen", "expected": "SELL", "family": "surplus",
        "request": asdict(request), "wire_sha256": source_plan_row["wire_sha256"],
        "estimated_input_tokens": source_plan_row["estimated_input_tokens"]}, reply)
    result = probe.summarize([row])
    assert result["n_rows"] == 1
    assert result["groups"][0]["top_tie_count"] == 1
    assert result["groups"][0]["correct_strict"] == 0
    assert result["groups"][0]["probability_vectors"] == [{"c0": 0.5, "c1": 0.5}]
    assert result["groups"][0]["top_candidate_ids"] == [["c0", "c1"]]
    assert row["top_margin"] == 0.0
    assert row["selected"] == "TOP_TIE"


def test_preview_cli_never_constructs_backend_or_loads_env(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("preview must not construct the live backend")

    monkeypatch.setattr(probe, "JevBackend", forbidden)
    output = tmp_path / "preview"
    assert probe.main.__module__ == "tools.probes.jev_input_ablation"
    monkeypatch.setattr("sys.argv", ["jev_input_ablation", "--output-dir", str(output)])
    assert probe.main() == 0
    preview = json.loads((output / "preview.json").read_text(encoding="utf-8"))
    assert preview["preview"] is True
    assert preview["count"] == 396
    assert not (output / "frozen-plan.json").exists()


def test_execute_stops_after_first_failed_wave_and_keeps_partial_status(tmp_path, monkeypatch):
    class FailingBackend:
        calls = 0

        def __init__(self, **kwargs):
            assert kwargs["instruction_profile"] == "ablation_v1"
            assert kwargs["max_requests"] == 396
            assert kwargs["max_estimated_input_tokens"] == 1_500_000

        async def decide_many(self, requests):
            self.calls += 1
            raise RuntimeError("provider body must not be persisted")

        @property
        def budget_usage(self):
            return {"requests_started": 0}

        @property
        def window_timings(self):
            return ()

        @property
        def transport_timings(self):
            return ()

        async def aclose(self):
            pass

    monkeypatch.setattr(probe, "JevBackend", FailingBackend)
    output = tmp_path / "failed"
    monkeypatch.setattr("sys.argv", ["jev_input_ablation", "--output-dir", str(output), "--execute"])
    assert probe.main() == 1
    failures = json.loads((output / "failures.json").read_text(encoding="utf-8"))
    raw = json.loads((output / "raw-evidence.json").read_text(encoding="utf-8"))
    result = json.loads((output / "result.json").read_text(encoding="utf-8"))
    assert result["status"] == "PARTIAL"
    assert result["stopped_after_failure"] is True
    assert result["evidence_rows"] == 0
    assert len(failures) == 396
    assert sum(row["status"] == "RuntimeError" for row in failures) == 10
    assert sum(row["status"] == "not_dispatched_after_prior_wave_failure" for row in failures) == 386
    assert raw == []
    assert "provider body" not in (output / "failures.json").read_text(encoding="utf-8")
    assert (output / "frozen-plan.json").exists()

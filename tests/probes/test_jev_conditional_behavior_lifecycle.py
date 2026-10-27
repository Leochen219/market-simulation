import hashlib
import json

import pytest

from capclaims.conditional_behavior.runner import execute_preview_sync, replay_run
from capclaims.conditional_behavior.analysis import score_run
from capclaims.conditional_behavior.study import build_preview, verify_preview


def _plan():
    inputs = []
    targets = {}
    for person_index in range(17):
        person = f"t{person_index:02d}"
        for target_round in (14, 15):
            key = f"{person}:{target_round}"
            targets[key] = {"person_id": person, "target_round": target_round,
                "Q": 10, "D": 20, "orders": [], "profit_cents": 0}
            for arm in "ABCDE":
                for sample in range(4):
                    inputs.append({"case_id": f"{key}:{arm}:{sample}", "person_id": person,
                        "target_round": target_round, "arm": arm, "sample": sample,
                        "interface": "J1" if arm != "E" else "J2",
                        "observation": {"round": target_round, "own_p": 50,
                            "wholesale_price_dollars": 10, "own_q_units": 10,
                            "own_d_units": 20, "market_size": 35, "history": [],
                            "decision_interface": "economic_v2",
                            "protocol_interface": "J1" if arm != "E" else "J2",
                            "own_pretrade_profit_dollars": "400.00",
                            "submitted_order_ledger": {"submitted_order_count": 0,
                                "remaining_order_slots": 5, "submitted_buy_quantity_units": 0,
                                "submitted_sell_quantity_units": 0,
                                "remaining_sell_submission_capacity_units": 10}}})
    return {"schema": "jev_conditional_behavior_v1", "seed": 20261026,
        "inputs": inputs, "targets": targets, "donor_matches": {}}


def test_preview_runner_replay_and_tamper_gates(tmp_path):
    preview = tmp_path / "preview"
    manifest = build_preview(_plan(), preview, source_sha256="a" * 64,
        exclusion_manifest_sha256={"procurement": "b" * 64, "portfolio": "c" * 64, "injection": "d" * 64},
        source_policy_verified=True)
    assert manifest["target_n"] == 34 and manifest["sample_rows"] == 680
    private_manifest = json.loads((preview / "manifest.json").read_text(encoding="utf-8"))
    assert private_manifest["request_previews_n"] > 0
    assert private_manifest["request_previews_max_body_bytes"] <= 12000
    assert "request-previews.jsonl" in {path.name for path in preview.iterdir()}
    assert private_manifest["scoring_targets_sha256"] == hashlib.sha256((preview / "scoring-targets.json").read_bytes()).hexdigest()
    assert "profit_cents" not in (preview / "inputs.json").read_text(encoding="utf-8")
    verify_preview(preview, manifest["preview_sha256"])
    with pytest.raises(ValueError, match="cost-bound attestation"):
        execute_preview_sync(preview, tmp_path / "live", manifest["preview_sha256"], fake=False)
    run = tmp_path / "run"
    result = execute_preview_sync(preview, run, manifest["preview_sha256"], fake=True)
    assert result["status"] == "COMPLETE" and result["samples"] == 680
    state_rows = [json.loads(line) for line in (run / "case-states.jsonl").read_text().splitlines()]
    latest = {row["case_id"]: row for row in state_rows}
    assert len(latest) == 680 and {row["status"] for row in latest.values()} == {"completed"}
    assert replay_run(preview, run)["status"] == "REPLAY_VALID"
    report = score_run(preview, run)
    assert report["planned_targets"] == 34
    assert report["completed_sample_rows"] == 680
    assert set(report["per_arm_participant_first"]) == set("ABCDE")
    assert set(report["per_arm_participant_first_bounds"]) == set("ABCDE")
    assert report["weighted_order_totals_w1_by_arm"]["A"]["overall"]["status"] == "SCORED"
    assert report["weighted_order_totals_w1_by_arm"]["A"]["overall"]["by_side"]["buy"]["w1"] == 0
    assert report["weighted_price_w1_by_arm"]["A"]["overall"]["by_side"]["buy"]["status"] == "NO_ELIGIBLE_DIRECTION"
    empty_context = next(row for row in report["target_rows"] if row["arm"] == "A")
    assert empty_context["medoid_exact_match_human"] is True
    assert empty_context["mean_random_draw_exact_match_human"] == 1
    assert empty_context["mean_within_portfolio_repeated_order_count"] == 0

    failed_run = tmp_path / "failed-run"
    failed_wire = tmp_path / "failed-wire"
    def fail_first(observation, row):
        if row["sample"] == 0:
            request_id = hashlib.sha256(f"{row['case_id']}\\0order:0:side".encode()).hexdigest()
            folder = failed_wire / hashlib.sha256(request_id.encode()).hexdigest()
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "attempt.json").write_text(json.dumps({"request_id": request_id}))
            raise RuntimeError("injected technical failure")
        return {"orders": [], "provider_requests": 0, "trace": [{"kind": "local_fake"}]}
    failure = execute_preview_sync(preview, failed_run, manifest["preview_sha256"],
        fake=True, sampler=fail_first, transport_evidence_dir=failed_wire)
    assert failure["status"] == "INCOMPLETE"
    failed_replay = replay_run(preview, failed_run)
    assert failed_replay["status"] == "REPLAY_INCOMPLETE"
    failed_report = score_run(preview, failed_run)
    assert len(failed_report["target_rows"]) == 34 * 5
    assert set(failed_report["per_arm_participant_first_bounds"]) == set("ABCDE")
    assert all("fair_energy_score" in failed_report["per_arm_participant_first_bounds"][arm]
               for arm in "ABCDE")
    assert failed_report["primary_E_vs_state"]["aggregate_difference_interval"] is not None
    assert not failed_report["primary_E_vs_state"]["meets_10pct_and_interval_gate"]
    fail_terminal = json.loads((failed_run / "terminal.json").read_text())
    assert fail_terminal["physical_attempted_requests"] >= 1
    assert fail_terminal["completed_trace_requests"] == 0
    assert fail_terminal["case_state_counts"]["failed"] > 0
    assert fail_terminal["case_state_counts"]["completed"] > 0
    assert fail_terminal["case_state_counts"]["not_started"] > 0
    assert failed_replay["case_state_counts"] == fail_terminal["case_state_counts"]
    assert failed_replay["physical_attempted_requests"] == fail_terminal["physical_attempted_requests"]

    all_failed_run = tmp_path / "all-failed-run"
    def fail_all(observation, row):
        raise ValueError("deterministic test failure")
    execute_preview_sync(preview, all_failed_run, manifest["preview_sha256"],
        fake=True, sampler=fail_all)
    all_failed_score = score_run(preview, all_failed_run)
    assert all(row["observed_K"] == 0 for row in all_failed_score["target_rows"])
    assert all("fair_energy_score" in all_failed_score["per_arm_participant_first_bounds"][arm]
               for arm in "ABCDE")
    incomplete_w1 = all_failed_score["weighted_order_totals_w1_by_arm"]["A"]["overall"]
    assert incomplete_w1["status"] == "INCOMPLETE"
    assert incomplete_w1["by_side"]["buy"]["w1"] is None
    assert incomplete_w1["by_side"]["buy"]["w1_interval"] == [0.0, 4995.0]
    with (run / "events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"case_id": "tamper"}) + "\n")
    with pytest.raises(ValueError):
        replay_run(preview, run)
    with (preview / "scoring-targets.json").open("a", encoding="utf-8") as stream:
        stream.write(" ")
    with pytest.raises(ValueError):
        verify_preview(preview, manifest["preview_sha256"])


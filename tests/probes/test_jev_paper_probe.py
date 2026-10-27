from tools.probes.jev_paper_probe import build_plan, overlap_summary


def test_preview_freezes_only_selected_fixtures_and_keeps_answers_out_of_wire_payload():
    plan = build_plan("development", "explicit_v1", include_history=False)
    assert plan["numeric_count"] == 174
    assert plan["economic_count"] == 48
    assert plan["request_count"] == 232  # ten production overlap calls
    assert len(plan["requests"]) == plan["request_count"]
    for item in plan["requests"]:
        wire = item["wire"]
        assert set(wire) == {"model", "state", "questions"}
        assert "expected" not in wire["state"]
        assert wire["questions"]["decision"]["instructions"]
        assert item["estimated_input_tokens"] > 0


def test_confirmation_is_separate_and_preview_is_repeatable():
    first = build_plan("confirmation", "matched_v2", include_history=False)
    second = build_plan("confirmation", "matched_v2", include_history=False)
    assert first == second
    assert first["numeric_count"] == 36
    assert first["economic_count"] == 12
    assert first["request_count"] == 58


def test_smoke_is_only_ten_development_economic_calls():
    plan = build_plan("smoke", "explicit_v1")
    assert plan["request_count"] == plan["economic_count"] == 10
    assert plan["numeric_count"] == 0
    assert all("diagnostic_task" in item["wire"]["state"] for item in plan["requests"])


def test_default_preview_does_not_construct_api_client(tmp_path, monkeypatch):
    from tools.probes import jev_paper_probe as probe
    def forbidden(**kwargs):
        raise AssertionError("preview constructed API client")
    monkeypatch.setattr(probe, "JevBackend", forbidden)
    assert probe.main(["--output-dir", str(tmp_path / "preview")]) == 0
    import json
    result = json.loads((tmp_path / "preview" / "result.json").read_text())
    assert result["status"] == "PREVIEW" and result["real_api_calls"] == 0


def test_overlap_requires_observed_request_windows_not_task_creation():
    assert overlap_summary([{"started_at": 1, "finished_at": 2},
                            {"started_at": 2, "finished_at": 3}])["peak_in_flight"] == 1
    assert overlap_summary([{"started_at": 1, "finished_at": 3},
                            {"started_at": 2, "finished_at": 4}])["peak_in_flight"] == 2


def test_whole_readiness_requires_ten_actual_overlapping_calls():
    from tools.probes.jev_paper_probe import qualify_readiness
    correct = [{"correct": True}]
    assert not qualify_readiness(correct, correct, {"request_count": 10, "peak_in_flight": 2})
    assert qualify_readiness(correct, correct, {"request_count": 10, "peak_in_flight": 10})

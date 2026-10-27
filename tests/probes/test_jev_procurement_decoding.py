import asyncio
import hashlib
import json
from pathlib import Path
import uuid

import pytest

from capclaims.procurement_decoding import (
    SINGLETON_PREFIXES, _assert_unique_prediction_sequences, _conditional_distribution, _frozen_payload,
    _write_json, build_diagnostic_plan,
    execute_diagnostic, make_synthetic_plan, maximum_overlap, summarize_tree,
)
import capclaims.procurement_decoding as decoding
from capclaims.paper.codec import legal_prefix_choices


def _conditional_rows():
    rows = {}
    for prefix in ("",) + tuple(p for p in __import__(
            "capclaims.procurement_perspective", fromlist=["build_choice_prefixes"]
        ).build_choice_prefixes() if p):
        digits = [str(value) for value in legal_prefix_choices(prefix, 400)]
        rows[prefix] = {digit: float(digit == digits[0]) for digit in digits}
    rows[""] = {"0": .6, "1": .4, "2": 0., "3": 0., "4": 0.}
    rows["0"] = {str(d): (.6 if d == 0 else .4 if d == 1 else 0.) for d in legal_prefix_choices("0", 400)}
    rows["00"] = {str(d): (.55 if d == 0 else .45 if d == 1 else 0.)
        for d in legal_prefix_choices("00", 400)}
    rows["1"] = {str(d): (1. if d == 0 else 0.) for d in legal_prefix_choices("1", 400)}
    rows["10"] = {str(d): (.8 if d == 0 else .2 if d == 1 else 0.)
        for d in legal_prefix_choices("10", 400)}
    return rows


def test_induced_distribution_covers_all_401_quantities_and_separates_greedy_from_mode():
    summary = summarize_tree(_conditional_rows())
    assert len(summary["quantity_probabilities"]) == 401
    assert 400 in summary["quantity_probabilities"]
    assert sum(summary["quantity_probabilities"].values()) == pytest.approx(1.0)
    assert summary["mass"] == pytest.approx(1.0)
    assert summary["greedy_quantity"] == 0
    assert summary["mode_quantity"] == 100
    assert summary["median_lower"] == 10
    assert summary["mass_below_100"] == pytest.approx(.6)


def test_top_ties_are_unresolved_and_mode_returns_full_tied_set():
    rows = _conditional_rows()
    rows[""] = {"0": .5, "1": .5, "2": 0., "3": 0., "4": 0.}
    rows["0"] = {str(d): (1. if d == 0 else 0.) for d in legal_prefix_choices("0", 400)}
    rows["00"] = {str(d): (1. if d == 0 else 0.) for d in legal_prefix_choices("00", 400)}
    rows["1"] = {str(d): (1. if d == 0 else 0.) for d in legal_prefix_choices("1", 400)}
    rows["10"] = {str(d): (1. if d == 0 else 0.) for d in legal_prefix_choices("10", 400)}
    result = summarize_tree(rows)
    assert result["greedy_quantity"] is None
    assert result["mode_set"] == [0, 100]
    assert result["mode_quantity"] is None
    assert result["median_lower"] == 0


def test_conditional_tree_rejects_non_normalized_rows():
    rows = _conditional_rows()
    rows["00"]["0"] = .9
    with pytest.raises(ValueError, match="do not sum"):
        summarize_tree(rows)


def test_near_normalized_conditional_rows_are_normalized_before_tree_product():
    rows = _conditional_rows()
    rows["00"]["0"] += 4e-7
    result = summarize_tree(rows)
    assert sum(result["quantity_probabilities"].values()) == pytest.approx(1.0, abs=1e-12)
    assert result["mass"] == pytest.approx(1.0, abs=1e-12)
    rows["00"]["0"] += 2e-6
    with pytest.raises(ValueError, match="do not sum"):
        summarize_tree(rows)


def test_duplicate_parent_prediction_sequences_are_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        _assert_unique_prediction_sequences([{"sequence": 7}, {"sequence": 7}])


def test_missing_source_attestation_fails_parent_preflight(tmp_path):
    import json
    from capclaims.backends.jev import MODEL
    from capclaims.procurement_decoding import EXPECTED_SOURCE_SHA
    parent = tmp_path / "parent"; parent.mkdir()
    (parent / "frozen-plan.json").write_text(json.dumps({
        "schema": "jev_procurement_perspective_v1", "model": MODEL,
        "source_kind": "human", "source_sha256": EXPECTED_SOURCE_SHA,
        "source_attestations": {"codebook_verified": True,
            "intended_use_authorized": False, "provider_transmission_permitted": True}}), encoding="utf-8")
    (parent / "result.json").write_text(json.dumps({}), encoding="utf-8")
    (parent / "private-predictions.json").write_text("[]", encoding="utf-8")
    with pytest.raises(PermissionError, match="authorized human-source"):
        build_diagnostic_plan(parent)


def test_synthetic_plan_has_two_full_trees_and_singletons_without_human_source():
    plan = make_synthetic_plan()
    assert plan["diagnostic_request_count"] == 90
    assert plan["diagnostic_singleton_count"] == 4
    assert len(plan["prefix_wires"]) == 94
    singleton_rows = [row for row in plan["prefix_wires"] if row["prefix"] in SINGLETON_PREFIXES]
    assert len(singleton_rows) == 4
    query_hashes = {row["wire_sha256"] for row in plan["prefix_wires"]
        if row["prefix"] not in SINGLETON_PREFIXES}
    assert len([row for row in plan["prefix_wires"] if row["prefix"] not in SINGLETON_PREFIXES]) == 90
    assert plan["_allowed_hashes"] == query_hashes
    assert all(len(legal_prefix_choices(row["prefix"], 400)) == 1 for row in singleton_rows)
    assert plan["source_attestations"] == {}
    assert plan["protocol_sha256"] == hashlib.sha256(plan["protocol_text"].encode()).hexdigest()
    assert plan["source_code_sha256"][plan["protocol_path"]] == plan["protocol_sha256"]


def test_protocol_mutation_fails_before_output_or_client_use(monkeypatch):
    plan = make_synthetic_plan()
    changed_text = plan["protocol_text"] + "tampered"
    monkeypatch.setattr(decoding, "_protocol_snapshot",
        lambda: (changed_text, hashlib.sha256(changed_text.encode()).hexdigest()))
    class UntouchedClient:
        called = False
    client = UntouchedClient()
    output = Path(__file__).resolve().parents[2] / "artifacts" / "participant-data" / f"protocol-mutated-{uuid.uuid4().hex}"
    with pytest.raises(ValueError, match="protocol changed"):
        asyncio.run(execute_diagnostic(plan, output, fake_client=client))
    assert not output.exists()
    assert client.called is False


def test_real_parent_preflight_reuses_exact_frozen_wires_and_caps():
    parent = Path("artifacts/participant-data/procurement-live-24")
    if not (parent / "frozen-plan.json").exists():
        pytest.skip("reviewed local pilot artifact is not present in this checkout")
    # A historical experiment must refuse newer shared backend code. Its
    # positive real-data preflight belongs to its recorded Git checkpoint.
    archived = json.loads((parent / "frozen-plan.json").read_text(encoding="utf-8"))
    project = Path(__file__).resolve().parents[2]
    changed = any(hashlib.sha256((project / name).read_bytes()).hexdigest() != digest
                  for name, digest in archived["source_code_sha256"].items())
    if changed:
        with pytest.raises(ValueError, match="frozen parent source-code hash changed"):
            build_diagnostic_plan(parent)
        return
    plan = build_diagnostic_plan(parent)
    assert [row["participant"] for row in plan["_sessions"]] == ["0", "1"]
    assert plan["diagnostic_request_count"] == 90
    assert plan["combined_request_ceiling"] == 467
    assert plan["combined_estimated_bytes"] <= 3_200_000
    assert len(plan["_prefix_hashes"]) == 94
    assert len(plan["_allowed_hashes"]) == 90


def test_mutated_preview_blocks_real_execute_before_output_or_credentials(monkeypatch):
    plan = make_synthetic_plan()
    # Use a live-shaped synthetic plan to test this gate without private data.
    plan["schema"] = "jev_procurement_decoding_diagnostic_v1"
    monkeypatch.setattr("capclaims.backends.jev.load_local_env",
                        lambda: pytest.fail("preview mismatch reached credential loading"))
    root = Path(__file__).resolve().parents[2] / "artifacts" / "participant-data"
    preview_dir = root / f"mutated-preview-{uuid.uuid4().hex}"; preview_dir.mkdir()
    preview = preview_dir / "frozen-diagnostic-plan.json"
    _write_json(preview, _frozen_payload(plan))
    data = json.loads(preview.read_text(encoding="utf-8")); data["parent_plan_sha256"] = "mutated"
    _write_json(preview, data)
    output = root / f"must-not-start-{uuid.uuid4().hex}"
    with pytest.raises(ValueError, match="does not exactly match"):
        asyncio.run(execute_diagnostic(plan, output, preview_plan_path=preview))
    assert not output.exists()


def test_real_execute_requires_explicit_preview_before_any_setup(monkeypatch):
    plan = make_synthetic_plan()
    plan["schema"] = "jev_procurement_decoding_diagnostic_v1"
    monkeypatch.setattr("capclaims.backends.jev.load_local_env",
                        lambda: pytest.fail("missing preview reached credential loading"))
    output = Path(__file__).resolve().parents[2] / "artifacts" / "participant-data" / f"missing-preview-{uuid.uuid4().hex}"
    with pytest.raises(ValueError, match="requires --preview-plan"):
        asyncio.run(execute_diagnostic(plan, output))
    assert not output.exists()


def test_http_overlap_calculation_uses_half_open_intervals():
    rows = [{"started_at": 1., "finished_at": 3.}, {"started_at": 2., "finished_at": 4.},
        {"started_at": 3., "finished_at": 5.}]
    assert maximum_overlap(rows) == 2
    assert maximum_overlap([]) == 0


def test_end_to_end_synthetic_diagnostic_uses_only_injected_fake_sdk():
    plan = make_synthetic_plan()
    output = Path(__file__).resolve().parents[2] / "artifacts" / "participant-data" / f"decode-synthetic-{uuid.uuid4().hex}"
    from capclaims.procurement_perspective import _sdk_mock
    fake = _sdk_mock()
    result = asyncio.run(execute_diagnostic(plan, output, fake_client=fake))
    assert result["status"] == "COMPLETE"
    assert result["actual_requests"] == 90
    assert result["responses"] == 90
    assert result["singleton_constraints"] == 4
    assert result["max_adapter_in_flight"] <= 10
    assert result["analysis_complete"] is True
    assert fake.closed
    assert (output / "private-diagnostic.json").exists()
    assert (output / "transport-timings.json").exists()
    import json
    private = json.loads((output / "private-diagnostic.json").read_text(encoding="utf-8"))
    assert all("raw_probability_total" in row for row in private["prefix_results"])
    assert all("raw_probability_total" in row["usage"] for row in private["prefix_results"])
    for alias in ("0", "1"):
        summary = private["analysis"]["inputs"][alias]
        assert len(summary["quantity_probabilities"]) == 401
        assert "greedy_tie_prefix" in summary


@pytest.mark.parametrize("failed_stage", ["analysis", "overlap"])
def test_analysis_exceptions_still_write_sanitized_result_and_partial_files(monkeypatch, failed_stage):
    plan = make_synthetic_plan()
    output = Path(__file__).resolve().parents[2] / "artifacts" / "participant-data" / f"decode-failed-{failed_stage}-{uuid.uuid4().hex}"
    from capclaims.procurement_perspective import _sdk_mock
    if failed_stage == "analysis":
        def fail_analysis(*args, **kwargs):
            raise RuntimeError("synthetic private detail")
        monkeypatch.setattr(decoding, "_analyze", fail_analysis)
        expected_type = "RuntimeError"
    else:
        def fail_overlap(*args, **kwargs):
            raise ValueError("synthetic private detail")
        monkeypatch.setattr(decoding, "maximum_overlap", fail_overlap)
        expected_type = "ValueError"
    result = asyncio.run(execute_diagnostic(plan, output, fake_client=_sdk_mock()))
    assert result["status"] == "ANALYSIS_FAILED"
    assert result["analysis_error_type"] == expected_type
    assert (output / "private-diagnostic.json").exists()
    assert (output / "partial-prefix-results.json").exists()
    assert (output / "result.json").exists()
    persisted = __import__("json").loads((output / "result.json").read_text(encoding="utf-8"))
    assert persisted["status"] == "ANALYSIS_FAILED"
    assert "synthetic private detail" not in (output / "result.json").read_text(encoding="utf-8")

"""Synthetic lifecycle invariants for the paired current-order pilot."""
import asyncio
import hashlib
import json
import os
from pathlib import Path

import pytest

from capclaims import order_explanation_study as study


def _synthetic_source(temp):
    root = temp / "repo"
    private = root / "artifacts" / "participant-data"
    private.mkdir(parents=True)
    source = {"source_kind": "human", "source_sha256": "source-hash"}
    policy = {"codebook_verified": True, "intended_use_authorized": True,
              "provider_transmission_permitted": True, **source}
    policy_path = root / study.SOURCE_POLICY_RELATIVE
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    policy_bytes = (json.dumps(policy, sort_keys=True, separators=(",", ":")) + "\n").encode()
    policy_path.write_bytes(policy_bytes)
    bases = {}
    targets = {}
    for index in range(6):
        person = f"S{index + 1}"
        bases[person] = {"agent_id": f"opaque-{index + 1}", "round": 9, "treatment": "GA",
            "wholesale_price_dollars": 10, "p": 51 + index, "Q": 250 + index, "D": 80 + index,
            "history": [{"round": round_no, "p": 55 + index, "Q": 200 + index, "D": 60 + index,
                "orders": [{"type": "BUY", "amount": 2 + index, "price": 70 + index}]}
                for round_no in (6, 7, 8)]}
        targets[person] = {"orders": [{"side": "buy", "qty": 2 + index, "limit_dollars": 70 + index}]}
    parent = private / "synthetic-parent"
    parent.mkdir()
    inputs_bytes = (json.dumps({"bases": bases}, sort_keys=True, separators=(",", ":")) + "\n").encode()
    labels_bytes = (json.dumps(targets, sort_keys=True, separators=(",", ":")) + "\n").encode()
    (parent / "inputs.json").write_bytes(inputs_bytes)
    (parent / "scoring-targets.json").write_bytes(labels_bytes)
    (parent / "manifest.json").write_text("{}\n", encoding="utf-8")
    parent_manifest = {"source": source, "source_policy_verified": True,
        "source_policy_sha256": hashlib.sha256(policy_bytes).hexdigest(),
        "scoring_targets_sha256": hashlib.sha256(labels_bytes).hexdigest()}
    return root, private, parent, parent_manifest


def test_synthetic_lifecycle_label_blind_invalidates_http500_and_rejects_portfolio_tamper(tmp_path, monkeypatch):
    root, private, parent, parent_manifest = _synthetic_source(tmp_path)
    from capclaims import field_alignment_study
    monkeypatch.setattr(study, "ROOT", root)
    monkeypatch.setattr(study, "_code_hashes", lambda: {"synthetic-code": "stable"})
    monkeypatch.setattr(field_alignment_study, "verify_preview", lambda *_args: parent_manifest)
    preview, run, failed = private / "preview", private / "run", private / "failed"
    manifest = study.build_preview(parent, preview)
    assert study.verify_preview(preview, manifest["preview_sha256"])["preflight_max_body_bytes"] <= study.MAX_BODY

    import httpx2
    arrived = set()
    barrier = asyncio.Event()
    overlap = {"current": 0, "peak": 0}
    lock = asyncio.Lock()

    async def successful_provider(request):
        body = await request.aread()
        signature = hashlib.sha256(body).hexdigest()
        async with lock:
            arrived.add(signature)
            overlap["current"] += 1
            overlap["peak"] = max(overlap["peak"], overlap["current"])
            if len(arrived) == study.MAX_IN_FLIGHT:
                barrier.set()
        try:
            if len(arrived) <= study.MAX_IN_FLIGHT:
                await barrier.wait()
            body_json = json.loads(body.decode("utf-8"))
            criteria = body_json["questions"]["decision"]["criteria"]
            ids = list(criteria)
            selected = next((key for key, label in criteria.items() if label.startswith("STOP:")), ids[-1])
            probs = {key: float(key == selected) for key in ids}
            return httpx2.Response(200, json={"model": "jev-1.14.0", "usage": {"input_tokens": 3, "output_tokens": 1},
                "answers": {"decision": {"type": "choice", "choice": selected,
                    "confidence": 1.0, "probabilities": probs}}})
        finally:
            async with lock:
                overlap["current"] -= 1

    labels = preview / "scoring-targets.json"
    hidden_labels = preview / "scoring-targets.hidden"
    os.replace(labels, hidden_labels)
    try:
        # The actual preview verifier, execute and strict replay run while labels do not exist.
        study.verify_preview(preview, manifest["preview_sha256"])
        executed = asyncio.run(study.execute_preview(preview, run, manifest["preview_sha256"], fake=True,
            fake_transport=httpx2.MockTransport(successful_provider)))
        assert executed["status"] == "COMPLETE" and executed["attempted_sessions"] == 24
        assert executed["max_overlap"] == study.MAX_IN_FLIGHT
        assert overlap["peak"] == study.MAX_IN_FLIGHT and len(arrived) >= study.MAX_IN_FLIGHT
        replay = study.replay_run(preview, run)
        assert replay["status"] == "REPLAY_VALID" and len(replay["portfolios"]) == 24
    finally:
        os.replace(hidden_labels, labels)

    events_path, terminal_path = run / "events.json", run / "terminal.json"
    original_events, original_terminal = events_path.read_bytes(), terminal_path.read_bytes()
    rows = json.loads(original_events.decode("utf-8"))
    rows[0]["orders"] = [] if rows[0]["orders"] else [{"side": "buy", "qty": 1, "limit_dollars": 1}]
    tampered = (json.dumps(rows, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n").encode()
    events_path.write_bytes(tampered)
    terminal = json.loads(original_terminal.decode("utf-8"))
    terminal["event_sha256"] = hashlib.sha256(tampered).hexdigest()
    terminal_path.write_text(json.dumps(terminal, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="saved portfolio differs"):
        study.replay_run(preview, run)
    events_path.write_bytes(original_events)
    terminal_path.write_bytes(original_terminal)

    score = study.score_run(preview, run)
    assert set(score["per_arm"]) == {study.CONTROL, study.EXPLICIT}
    assert score["paired_treatment_minus_control"]["primary_curve_error"]["mean_treatment_minus_control"] is not None
    private_scores = json.loads((run / "private-scores.json").read_text(encoding="utf-8"))
    assert len(private_scores["rows"]) == 24
    assert len(private_scores["private_person_scores"]) == 12
    assert len(private_scores["private_repeat_rows"]) == 12

    failures = []

    async def failing_provider(_request):
        failures.append(1)
        return httpx2.Response(500, json={"error": "synthetic"})

    with pytest.raises(RuntimeError, match="study wave invalidated"):
        asyncio.run(study.execute_preview(preview, failed, manifest["preview_sha256"], fake=True,
            fake_transport=httpx2.MockTransport(failing_provider)))
    assert (failed / "invalid.json").is_file() and not (failed / "terminal.json").exists()
    attempts = [line for line in (failed / "attempts.jsonl").read_text(encoding="utf-8").splitlines() if line]
    assert len(attempts) == len(failures) and 1 <= len(failures) <= study.MAX_IN_FLIGHT

    # A changed source-permission artifact is rejected before environment/key loading.
    policy_path = root / study.SOURCE_POLICY_RELATIVE
    policy_path.write_text("{}\n", encoding="utf-8")
    from capclaims.backends import jev
    monkeypatch.setattr(jev, "load_local_env", lambda: pytest.fail("environment loaded before source-policy check"))
    with pytest.raises(ValueError, match="source-policy hash"):
        asyncio.run(study.execute_preview(preview, private / "blocked", manifest["preview_sha256"], fake=False))
    assert not (private / "blocked").exists()

import asyncio
import json
import sys

import pytest

import capclaims.behavioral_memory.runner as runner_module
from capclaims.behavioral_memory.lifecycle import freeze_preview
from capclaims.behavioral_memory.runner import execute_preview, replay_preview
from capclaims.behavioral_memory.lifecycle import _canonical, _sha
from capclaims.behavioral_memory import transport as behavioral_transport

if sys.platform == "win32":
    pytestmark = pytest.mark.asyncio


def cases():
    return [{"person_id": f"dev-{n}",
        "history": [{"round": r, "p": 100, "Q": 10, "D": 8, "orders": [], "profit_cents": 0}
                    for r in range(6, 13)],
        "current_state": {"round": 13, "p": 100, "w": 10, "Q": 10, "D": 8},
        "label": {"orders": [], "profit_cents": 0}} for n in range(18)]


@pytest.mark.asyncio
async def test_mock_runner_emits_108_replayable_full_trajectories(tmp_path):
    preview_dir, run_dir = tmp_path / "preview", tmp_path / "run"
    frozen = freeze_preview(cases(), {"source_sha256": "a" * 64,
        "source_policy_verified": True}, preview_dir)
    run = await execute_preview(preview_dir, run_dir, frozen["preview_sha256"], fake=True)
    assert run["status"] == "COMPLETE"
    assert run["completed_trajectories"] == 108
    replay = await replay_preview(preview_dir, run_dir, frozen["preview_sha256"])
    assert replay["status"] == "REPLAY_VALID"


@pytest.mark.asyncio
async def test_replay_rejects_changed_hash_chained_event(tmp_path):
    preview_dir, run_dir = tmp_path / "preview", tmp_path / "run"
    frozen = freeze_preview(cases(), {"source_sha256": "a" * 64,
        "source_policy_verified": True}, preview_dir)
    await execute_preview(preview_dir, run_dir, frozen["preview_sha256"], fake=True)
    events = run_dir / "events.jsonl"
    data = events.read_text(encoding="utf-8")
    events.write_text(data.replace('"mode":"MOCK"', '"mode":"LIVE"', 1), encoding="utf-8")
    with pytest.raises(ValueError):
        await replay_preview(preview_dir, run_dir, frozen["preview_sha256"])


def _rewrite_events(run_dir, mutate):
    path = run_dir / "events.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    previous = "0" * 64
    rewritten = []
    for sequence, line in enumerate(lines):
        event = json.loads(line)
        event["sequence"] = sequence
        event["previous_sha256"] = previous
        mutate(event)
        event["sha256"] = _sha(_canonical({key: event[key] for key in
            ("sequence", "previous_sha256", "type", "data")}))
        previous = event["sha256"]
        rewritten.append(_canonical(event).decode("utf-8"))
    path.write_text("\n".join(rewritten) + "\n", encoding="utf-8")
    summary_path = run_dir / "run.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["final_event_sha256"] = previous
    summary_path.write_bytes(_canonical(summary) + b"\n")


@pytest.mark.asyncio
async def test_incomplete_replay_rejects_mutated_successful_step_before_failure(tmp_path, monkeypatch):
    preview_dir, run_dir = tmp_path / "preview", tmp_path / "run"
    frozen = freeze_preview(cases(), {"source_sha256": "a" * 64,
        "source_policy_verified": True}, preview_dir)

    class FailAfterCount:
        async def choose(self, *, request_id, state, instructions, criteria):
            if state["decision_field"] != "order_count":
                raise RuntimeError("simulated failure on second request")
            probabilities = {candidate: float(candidate == "N1") for candidate in criteria}
            return {"model": "jev-1.14.0", "usage": {"input_tokens": 1},
                "answers": {"decision": {"type": "choice", "choice": "N1",
                    "probabilities": probabilities}}}

    monkeypatch.setattr(runner_module, "ProbabilityMock", FailAfterCount)
    await execute_preview(preview_dir, run_dir, frozen["preview_sha256"], fake=True)
    assert (await replay_preview(preview_dir, run_dir, frozen["preview_sha256"]))["status"] == "INCOMPLETE"

    def mutate(event):
        if event["type"] == "TRAJECTORY_STEP" and event["data"]["person_id"] == "dev-0":
            event["data"]["step"]["selected"] = "N0"

    _rewrite_events(run_dir, mutate)
    with pytest.raises(ValueError, match="successful prefix|step|replay"):
        await replay_preview(preview_dir, run_dir, frozen["preview_sha256"])


@pytest.mark.asyncio
async def test_live_replay_distinguishes_reserved_attempts_from_queued_peers(tmp_path, monkeypatch):
    import httpx2

    preview_dir, run_dir = tmp_path / "preview", tmp_path / "run"
    frozen = freeze_preview(cases(), {"source_sha256": "a" * 64,
        "source_policy_verified": True}, preview_dir)
    original_verify = runner_module.verify_preview

    def verify_live_fixture(path, digest):
        manifest = original_verify(path, digest)
        manifest["wire_preview_mode"] = "SDK_EXACT"
        manifest["metadata"] = {**manifest["metadata"], "source_paths": {"fixture": "mock-only"}}
        return manifest

    monkeypatch.setattr(runner_module, "verify_preview", verify_live_fixture)
    ledger_path = tmp_path / "ledger.sqlite"
    monkeypatch.setattr(behavioral_transport, "DEFAULT_LEDGER", ledger_path)
    ledger = behavioral_transport.CostLedger(ledger_path, preview_hash=frozen["preview_sha256"])
    client_ref = {}

    async def provider(request):
        client_ref["client"].halted = True
        return httpx2.Response(503, json={"error": "mock first-attempt failure"})

    async with behavioral_transport.JevWireClient(api_key="mock-only", ledger=ledger,
            evidence_dir=run_dir.parent / f"{run_dir.name}.transport-evidence",
            transport=httpx2.MockTransport(provider)) as client:
        client_ref["client"] = client
        run = await execute_preview(preview_dir, run_dir, frozen["preview_sha256"],
                                    fake=False, client=client,
                                    preflight=lambda manifest: True)
    assert run["status"] == "INCOMPLETE"
    assert 0 < ledger.summary()["physical_requests"] < 10
    replay = await replay_preview(preview_dir, run_dir, frozen["preview_sha256"])
    assert replay["status"] == "INCOMPLETE"
    statuses = replay["failed_terminal_request_status_counts"]
    assert statuses["PHYSICAL_ATTEMPT"] == ledger.summary()["physical_requests"]
    assert statuses["NOT_RESERVED/NOT_DISPATCHED"] == 10 - ledger.summary()["physical_requests"]

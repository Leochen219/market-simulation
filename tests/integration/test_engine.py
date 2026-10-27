import asyncio
import json
from pathlib import Path

import pytest

from capclaims.backends.rules import RuleBackend
from capclaims.config import RunConfig
from capclaims.engine.session import run_session
from capclaims.engine.replay import replay


def run(config, output, backend=None):
    return asyncio.run(run_session(config, backend or RuleBackend(config), output))


def test_ten_agent_rule_session_is_reproducible_and_replayable(tmp_path):
    config = RunConfig(policy="grid_rule", rounds=15, market_open=True, master_seed=123)
    first = run(config, tmp_path / "first")
    second = run(config, tmp_path / "second")
    assert first.status == second.status == "COMPLETE"
    assert first.rounds == 15
    assert first.trades > 0
    assert (tmp_path / "first" / "economic.jsonl").read_bytes() == (tmp_path / "second" / "economic.jsonl").read_bytes()
    assert replay(tmp_path / "first").rounds == 15
    assert replay(tmp_path / "first").final_hash == first.final_hash


def test_replay_rejects_tampered_economic_record(tmp_path):
    config = RunConfig(policy="grid_rule", rounds=1, market_open=False, master_seed=4)
    output = tmp_path / "tamper"
    run(config, output)
    lines = (output / "economic.jsonl").read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[0])
    row["event"]["round_index"] = 900
    lines[0] = json.dumps(row, sort_keys=True, separators=(",", ":"))
    (output / "economic.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash|replay|round"):
        replay(output)


def test_replay_rejects_valid_prefix_when_manifest_round_count_is_shortened(tmp_path):
    config = RunConfig(policy="grid_rule", rounds=2, market_open=False, master_seed=15)
    output = tmp_path / "prefix"
    run(config, output)
    lines = (output / "economic.jsonl").read_text(encoding="utf-8").splitlines()
    first = json.loads(lines[0])
    (output / "economic.jsonl").write_text(lines[0] + "\n", encoding="utf-8")
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["rounds"] = 1
    manifest["economic_final_hash"] = first["hash"]
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="round|replay"):
        replay(output)


def test_backend_failure_marks_run_incomplete_without_round_commit(tmp_path):
    class BrokenBackend:
        async def decide_many(self, requests):
            raise RuntimeError("synthetic phase failure")

        async def aclose(self):
            pass

    output = tmp_path / "failed"
    result = run(RunConfig(policy="laya", rounds=2), output, BrokenBackend())
    assert result.status == "INCOMPLETE"
    assert result.rounds == 0
    assert json.loads((output / "manifest.json").read_text())["status"] == "INCOMPLETE"
    assert (output / "economic.jsonl").read_text(encoding="utf-8") == ""


def test_wave_timeout_cancels_backend_and_does_not_commit(tmp_path):
    class SlowBackend:
        def __init__(self):
            self.cancelled = False

        async def decide_many(self, requests):
            try:
                await asyncio.sleep(1)
            except asyncio.CancelledError:
                self.cancelled = True
                raise

        async def aclose(self):
            pass

    backend = SlowBackend()
    output = tmp_path / "timeout"
    result = asyncio.run(run_session(
        RunConfig(policy="laya", rounds=1, market_open=False), backend, output,
        wave_timeout_seconds=.01,
    ))
    assert result.status == "INCOMPLETE"
    assert backend.cancelled
    assert (output / "economic.jsonl").read_text(encoding="utf-8") == ""


def test_external_cancellation_marks_incomplete_closes_backend_and_reraises(tmp_path):
    class WaitingBackend:
        def __init__(self):
            self.cancelled = False
            self.closed = False

        async def decide_many(self, requests):
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled = True
                raise

        async def aclose(self):
            self.closed = True

    async def exercise(backend, output):
        task = asyncio.create_task(run_session(RunConfig(policy="laya", rounds=2), backend, output))
        await asyncio.sleep(.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    backend = WaitingBackend()
    output = tmp_path / "cancelled"
    asyncio.run(exercise(backend, output))
    assert backend.cancelled and backend.closed
    assert json.loads((output / "manifest.json").read_text(encoding="utf-8"))["status"] == "INCOMPLETE"
    assert (output / "economic.jsonl").read_text(encoding="utf-8") == ""


def test_batch_forward_evidence_is_counted_once_per_call_group(tmp_path):
    from capclaims.engine.protocol import DecisionReply

    class GroupedBackend:
        async def decide_many(self, requests):
            replies = []
            for request in requests:
                candidate = request.candidates[0]
                scores = {item.candidate_id: float(item == candidate) for item in request.candidates}
                replies.append(DecisionReply.from_request(
                    request, scores, scores, "laya-test", "model-rev",
                    timing={"forward_batch_sizes": [10], "group_id": "purchase-wave-group-1"},
                ))
            return replies

        async def aclose(self):
            pass

    output = tmp_path / "group-count"
    result = run(RunConfig(policy="laya", rounds=1, market_open=False), output, GroupedBackend())
    assert result.status == "COMPLETE"
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["observed_batch_sizes"] == [10]

"""Reject corrupt phase envelopes before settlement and recompute rehashed ledgers."""
import asyncio
import json
from dataclasses import replace

import pytest

from capclaims.backends.rules import RuleBackend
from capclaims.config import RunConfig
from capclaims.engine.protocol import DecisionReply
from capclaims.engine.replay import replay
from capclaims.engine.session import _event_row, run_session
from capclaims.randomness import canonical_json


@pytest.mark.parametrize("corruption", ["stage", "observation", "candidates", "nan", "duplicate", "unknown"])
def test_corrupt_wave_never_reaches_settlement(tmp_path, corruption):
    class CorruptBackend:
        closed = False

        async def decide_many(self, requests):
            replies = []
            for request in requests:
                scores = {c.candidate_id: float(n == 0) for n, c in enumerate(request.candidates)}
                replies.append(DecisionReply.from_request(request, scores))
            if corruption == "stage":
                replies[0] = replace(replies[0], stage="trade_price")
            elif corruption == "observation":
                replies[0] = replace(replies[0], observation_hash="bad")
            elif corruption == "candidates":
                replies[0] = replace(replies[0], probabilities={"unrequested": 1.0})
            elif corruption == "nan":
                scores = dict(replies[0].probabilities)
                scores[next(iter(scores))] = float("nan")
                replies[0] = replace(replies[0], probabilities=scores)
            elif corruption == "duplicate":
                replies[1] = replies[0]
            else:
                replies[0] = replace(replies[0], request_id="other-session:unknown")
            return replies

        async def aclose(self):
            self.closed = True

    backend = CorruptBackend()
    result = asyncio.run(run_session(RunConfig(policy="laya", rounds=1), backend, tmp_path / "run"))
    assert result.status == "INCOMPLETE" and result.rounds == 0
    assert backend.closed
    assert (tmp_path / "run/economic.jsonl").read_bytes() == b""
    with pytest.raises(ValueError, match="COMPLETE"):
        replay(tmp_path / "run")


def test_replay_recomputes_ledger_even_if_hash_chain_was_rebuilt(tmp_path):
    config = RunConfig(rounds=1, market_open=False)
    output = tmp_path / "run"
    result = asyncio.run(run_session(config, RuleBackend(config), output))
    assert result.status == "COMPLETE"
    path = output / "economic.jsonl"
    row = json.loads(path.read_text(encoding="utf-8"))
    row["event"]["settlement"]["profits_cents"]["agent_00"] += 1
    # A valid new checksum must not substitute for economic recomputation.
    row = _event_row(row["seq"], row["prev_hash"], row["event"])
    path.write_text(canonical_json(row) + "\n", encoding="utf-8")
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["economic_final_hash"] = row["hash"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="recomputed market or ledger differs"):
        replay(output)

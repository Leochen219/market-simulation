import asyncio
import faulthandler

from capclaims.paper.config import PaperConfig
from capclaims.paper.session import run_paper_session
from capclaims.paper.replay import replay_paper
from capclaims.engine.protocol import DecisionReply


class FirstChoiceBackend:
    async def decide_many(self, requests):
        # Reverse delivery proves reply correlation uses request identity, not position.
        replies = []
        for request in reversed(requests):
            scores = {c.candidate_id: float(i == 0) for i, c in enumerate(request.candidates)}
            replies.append(DecisionReply.from_request(request, scores, scores))
        return replies

    async def aclose(self):
        pass


class StopBackend(FirstChoiceBackend):
    async def decide_many(self, requests):
        replies = []
        for request in reversed(requests):
            selected = next((c for c in request.candidates if c.value == "STOP"), request.candidates[0])
            scores = {c.candidate_id: float(c == selected) for c in request.candidates}
            replies.append(DecisionReply.from_request(request, scores, scores))
        return replies


class MultiOrderBackend(FirstChoiceBackend):
    async def decide_many(self, requests):
        replies = []
        for request in reversed(requests):
            if request.stage.endswith(":side"):
                slot = int(request.stage.split(":slot", 1)[1].split(":", 1)[0])
                value = "buy" if slot < 2 else "STOP"
                selected = next(c for c in request.candidates if c.value == value)
            else:
                selected = request.candidates[0]
            scores = {candidate.candidate_id: float(candidate == selected)
                      for candidate in request.candidates}
            replies.append(DecisionReply.from_request(request, scores, scores))
        return replies


def test_single_round_nv_run_replays_without_inference_and_filters_warmup(tmp_path):
    faulthandler.dump_traceback_later(3)
    config = PaperConfig(treatment="NV", n_agents=2, warmup_rounds=1, analyzed_rounds=1)
    path = tmp_path / "run"
    result = asyncio.run(run_paper_session(config, FirstChoiceBackend(), path))
    replay = replay_paper(path)
    assert result.rounds == replay.rounds == 2
    assert replay.formal_rounds == 1
    assert replay.final_hash == result.final_hash
    assert all(not event["orders"] for event in replay.events)
    faulthandler.cancel_dump_traceback_later()


def test_ga_multiorder_run_replays_canonical_agent_major_orders(tmp_path):
    config = PaperConfig(treatment="GA", n_agents=3, warmup_rounds=0, analyzed_rounds=1,
                         policy="laya")
    path = tmp_path / "run"
    result = asyncio.run(run_paper_session(config, MultiOrderBackend(), path))
    replay = replay_paper(path)
    submitted = result.events[0]["orders"]
    counts = {agent: sum(order["owner"] == agent for order in submitted)
              for agent in ("agent_00", "agent_01", "agent_02")}
    assert min(counts.values()) == max(counts.values()) == 2
    assert replay.rounds == result.rounds == 1
    assert replay.final_hash == result.final_hash
    assert replay.events[0]["orders"] == submitted


def test_paper_replay_rejects_rehashed_economic_tamper(tmp_path):
    import json
    config = PaperConfig(treatment="NV", n_agents=2, warmup_rounds=0, analyzed_rounds=1)
    path = tmp_path / "run"
    asyncio.run(run_paper_session(config, FirstChoiceBackend(), path))
    event_path = path / "economic.jsonl"
    event = json.loads(event_path.read_text().splitlines()[0])
    event["orders"] = [{"order_id": "forged", "owner": "agent_00", "side": "sell", "qty": 999, "limit_dollars": 999}]
    from capclaims.engine.protocol import digest_json
    event["event_hash"] = digest_json({k: v for k, v in event.items() if k != "event_hash"})
    event_path.write_text(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")
    try:
        replay_paper(path)
    except ValueError as exc:
        assert "replay" in str(exc).lower() or "gross sell" in str(exc).lower()
    else:
        raise AssertionError("replay accepted forged, rehashed economics")


def test_paper_replay_rejects_missing_order_side_audits_even_when_no_orders(tmp_path):
    import json
    config = PaperConfig(treatment="GA", n_agents=2, warmup_rounds=0, analyzed_rounds=1,
                         policy="newsvendor")
    path = tmp_path / "run"
    asyncio.run(run_paper_session(config, StopBackend(), path))
    audit_path = path / "audit.jsonl"
    rows = [json.loads(line) for line in audit_path.read_text().splitlines()]
    rows = [row for row in rows if not row["stage"].endswith(":side")]
    audit_path.write_text("".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows))
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["decision_audit_count"] = len(rows)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True))
    try:
        replay_paper(path)
    except ValueError as exc:
        assert "missing" in str(exc).lower() or "audit" in str(exc).lower()
    else:
        raise AssertionError("replay accepted missing side decisions as implicit STOP")


def test_paper_replay_rejects_foreign_order_owner_even_when_market_ignores_it(tmp_path):
    import json
    from capclaims.engine.protocol import digest_json
    config = PaperConfig(treatment="GA", n_agents=2, warmup_rounds=0, analyzed_rounds=1,
                         policy="newsvendor")
    path = tmp_path / "run"
    asyncio.run(run_paper_session(config, StopBackend(), path))
    event_path = path / "economic.jsonl"
    event = json.loads(event_path.read_text().splitlines()[0])
    event["orders"].append({"order_id": "r000:outsider:slot0", "owner": "outsider",
                            "side": "buy", "qty": 1, "limit_dollars": 1})
    event["event_hash"] = digest_json({k: v for k, v in event.items() if k != "event_hash"})
    event_path.write_text(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")
    try:
        replay_paper(path)
    except ValueError as exc:
        assert "owner" in str(exc).lower() or "agent" in str(exc).lower()
    else:
        raise AssertionError("replay accepted order from an unknown owner")

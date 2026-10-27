import asyncio

from capclaims.config import RunConfig
from capclaims.engine.protocol import DecisionReply
from capclaims.engine.session import run_session


def test_coordinator_handles_parallel_out_of_order_replies(tmp_path):
    class DelayedBackend:
        def __init__(self):
            self.active = 0
            self.max_active = 0

        async def decide_many(self, requests):
            async def respond(index, request):
                self.active += 1
                self.max_active = max(self.max_active, self.active)
                try:
                    await asyncio.sleep((len(requests) - index) * .003)
                    ids = [candidate.candidate_id for candidate in request.candidates]
                    scores = {candidate_id: float(index == 0 and position == 0)
                              for position, candidate_id in enumerate(ids)}
                    if not any(scores.values()):
                        scores[ids[0]] = 1.0
                    return DecisionReply.from_request(request, scores, scores, "test", "rev-1")
                finally:
                    self.active -= 1

            replies = await asyncio.gather(*(respond(i, request) for i, request in enumerate(requests)))
            return list(reversed(replies))

        async def aclose(self):
            pass

    config = RunConfig(policy="laya", rounds=1, market_open=False, master_seed=8)
    backend = DelayedBackend()
    result = asyncio.run(run_session(config, backend, tmp_path / "parallel"))
    second_backend = DelayedBackend()
    second_result = asyncio.run(run_session(config, second_backend, tmp_path / "parallel-repeat"))
    assert result.status == "COMPLETE"
    assert backend.max_active == 10
    assert result.rounds == 1
    assert second_result.status == "COMPLETE"
    assert (tmp_path / "parallel" / "economic.jsonl").read_bytes() == (tmp_path / "parallel-repeat" / "economic.jsonl").read_bytes()


def test_phase_observations_keep_each_agents_state_private(tmp_path):
    class AuditingBackend:
        def __init__(self):
            self.requests = []

        async def decide_many(self, requests):
            self.requests.extend(requests)
            replies = []
            for request in requests:
                candidate = request.candidates[0]
                scores = {item.candidate_id: float(item == candidate) for item in request.candidates}
                replies.append(DecisionReply.from_request(request, scores, scores, "test", "rev-1"))
            return replies

        async def aclose(self):
            pass

    backend = AuditingBackend()
    config = RunConfig(policy="laya", rounds=1, market_open=True, master_seed=19)
    result = asyncio.run(run_session(config, backend, tmp_path / "private"))
    assert result.status == "COMPLETE"
    assert {request.stage for request in backend.requests} == {"purchase", "trade_side", "trade_qty", "trade_price"}
    allowed_history = {"p", "D", "Q", "A", "profit_cents", "m_cents", "fillqty", "side", "qty", "price"}
    for request in backend.requests:
        observation = request.observation
        assert "world_hash" not in observation
        assert not ({"other_agents", "all_p", "all_d", "all_q", "orders"} & set(observation))
        assert all(set(record) <= allowed_history for record in observation["history"])
        if request.stage == "purchase":
            assert "own_q" not in observation and "own_d" not in observation
        else:
            assert observation["own_q"] in range(401) and observation["own_d"] in range(201)
        if request.stage in {"trade_qty", "trade_price"}:
            assert observation["own_side"] in {"buy", "sell"}
        if request.stage == "trade_price":
            assert observation["own_qty"] in range(1, 401)

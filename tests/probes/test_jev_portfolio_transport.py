import asyncio
import json
import time

import httpx2
import pytest
from typesafe_sdk import AsyncTypeSafeClient
from typesafe_sdk._core.retry import RetryPolicy

from capclaims.backends.jev import MODEL, JevBackend, _CURRENT_REQUEST_ID
from capclaims.portfolio_prediction import build_order_request
from capclaims.portfolio_transport import (DEFAULT_URL, GuardTransport, SharedBudget,
    serialize_sdk_wire)


def _base(person):
    return {"agent_id": person, "round": 9, "treatment": "GA", "wholesale_price_dollars": 10,
        "p": 70, "Q": 50, "D": 40, "history": []}


def test_sdk_serialization_guard_is_one_shot_and_charges_before_http(tmp_path):
    async def run():
        request = build_order_request(_base("opaque-one"), ())
        wire = await serialize_sdk_wire(request, "portfolio_first_v1")
        max_active = 0
        active = 0

        async def responder(outgoing):
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            await asyncio.sleep(0.01)
            body = json.loads((await outgoing.aread()).decode())
            criteria = body["questions"]["decision"]["criteria"]
            key = next(iter(criteria))
            active -= 1
            return httpx2.Response(200, json={"model": MODEL,
                "usage": {"input_tokens": 1, "output_tokens": 1},
                "answers": {"decision": {"type": "choice", "choice": key,
                    "confidence": 1, "probabilities": {item: float(item == key) for item in criteria}}}})

        budget = SharedBudget(tmp_path / "attempts.jsonl", max_requests=10, max_bytes=75_000,
            rps=10000, bytes_per_second=100_000_000)
        guard = GuardTransport(httpx2.MockTransport(responder), DEFAULT_URL, budget)
        guard.register(wire)
        http = httpx2.AsyncClient(transport=guard, follow_redirects=False)
        client = AsyncTypeSafeClient(api_key="dummy", model=MODEL,
            retry=RetryPolicy(max_retries=0), http_client=http,
            base_url="https://api.typesafe.ai", compression=False)
        backend = JevBackend(api_key="dummy", rps=10000, tps=100_000_000,
            max_requests=10, max_estimated_input_tokens=1_000_000,
            instruction_profile="portfolio_first_v1", objective="portfolio_prediction_v1",
            client=client, load_environment=False)
        try:
            (reply,) = await backend.decide_many([request])
            assert reply.model_revision == MODEL
            assert budget.usage["requests_charged"] == 1
            assert len((tmp_path / "attempts.jsonl").read_text().splitlines()) == 1
            context = _CURRENT_REQUEST_ID.set(request.request_id)
            try:
                with pytest.raises(RuntimeError, match="consumed"):
                    await http.post(DEFAULT_URL, content=wire.body)
            finally:
                _CURRENT_REQUEST_ID.reset(context)
            assert budget.usage["requests_charged"] == 1
        finally:
            backend._closed = True
            await client.aclose()

    asyncio.run(run())


def test_missing_request_context_fails_before_budget_charge(tmp_path):
    async def run():
        budget = SharedBudget(tmp_path / "attempts.jsonl")
        guard = GuardTransport(httpx2.MockTransport(lambda request: None), DEFAULT_URL, budget)
        http = httpx2.AsyncClient(transport=guard, follow_redirects=False)
        try:
            with pytest.raises(RuntimeError, match="missing, unknown"):
                await http.post(DEFAULT_URL, content=b"{}")
            assert budget.usage["requests_charged"] == 0
        finally:
            await http.aclose()
    asyncio.run(run())


def test_ten_sdk_requests_overlap_with_independent_one_shot_contexts(tmp_path):
    async def run():
        requests = [build_order_request(_base(f"opaque-{i}"), ()) for i in range(10)]
        wires = await asyncio.gather(*(serialize_sdk_wire(request, "portfolio_first_v1") for request in requests))
        budget = SharedBudget(tmp_path / "attempts.jsonl", max_requests=10, max_bytes=75_000,
            rps=10_000, bytes_per_second=100_000_000)
        active = peak = 0
        all_entered = asyncio.Event()

        async def responder(outgoing):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            if active == 10:
                all_entered.set()
            await asyncio.wait_for(all_entered.wait(), timeout=5)
            await asyncio.sleep(0.01)
            body = json.loads((await outgoing.aread()).decode())
            criteria = body["questions"]["decision"]["criteria"]
            key = next(iter(criteria))
            active -= 1
            return httpx2.Response(200, json={"model": MODEL, "usage": {},
                "answers": {"decision": {"type": "choice", "choice": key, "confidence": 1,
                    "probabilities": {item: float(item == key) for item in criteria}}}})

        guard = GuardTransport(httpx2.MockTransport(responder), DEFAULT_URL, budget)
        for wire in wires:
            guard.register(wire)
        http = httpx2.AsyncClient(transport=guard, follow_redirects=False)
        client = AsyncTypeSafeClient(api_key="dummy", model=MODEL,
            retry=RetryPolicy(max_retries=0), http_client=http,
            base_url="https://api.typesafe.ai", compression=False)
        backend = JevBackend(api_key="dummy", rps=10000, tps=100_000_000,
            max_requests=10, max_estimated_input_tokens=1_000_000,
            instruction_profile="portfolio_first_v1", objective="portfolio_prediction_v1",
            max_in_flight=10, client=client, load_environment=False)
        try:
            replies = await backend.decide_many(requests)
            assert len(replies) == 10
            assert peak == 10
            assert budget.usage["requests_charged"] == 10
        finally:
            backend._closed = True
            await client.aclose()
    asyncio.run(run())

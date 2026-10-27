"""Small explicit-dependency primitives for paired portfolio sessions."""
from __future__ import annotations

import asyncio
import hashlib
from typing import Any, Callable, Mapping, Sequence

from capclaims.backends.jev import MODEL, JevBackend, estimate_input_tokens
from capclaims.order_presentation import ARMS, order_prompt, render_order_state
from capclaims.portfolio_transport import ExpectedWire, _url_tuple


async def serialize_explanation_wire(request: Any, arm: str) -> ExpectedWire:
    """Capture the exact pinned SDK body for one rendered arm and request."""
    if arm not in ARMS:
        raise ValueError("unknown order-explanation arm")
    import httpx2
    from typesafe_sdk import AsyncTypeSafeClient, Choice
    from typesafe_sdk._core.retry import RetryPolicy

    captured: dict[str, Any] = {}
    ids = [choice.candidate_id for choice in request.candidates]

    async def capture(outgoing: Any) -> Any:
        captured.update(method=outgoing.method, url=str(outgoing.url), body=await outgoing.aread())
        probabilities = {key: float(key == ids[0]) for key in ids}
        return httpx2.Response(200, json={"model": MODEL, "usage": {"input_tokens": 1, "output_tokens": 1},
            "answers": {"decision": {"type": "choice", "choice": ids[0], "confidence": 1,
                "probabilities": probabilities}}})

    http = httpx2.AsyncClient(transport=httpx2.MockTransport(capture), follow_redirects=False)
    client = AsyncTypeSafeClient(api_key="dry-run-dummy", model=MODEL, retry=RetryPolicy(max_retries=0),
        http_client=http, base_url="https://api.typesafe.ai", compression=False)
    try:
        await client.system_one(state=render_order_state(request, arm),
            questions={"decision": Choice(instructions=order_prompt(request, arm),
                criteria={choice.candidate_id: choice.label for choice in request.candidates})}, model=MODEL)
    finally:
        await client.aclose()
    body = captured["body"]
    return ExpectedWire(request.request_id, captured["method"], _url_tuple(captured["url"]), body,
        hashlib.sha256(body).hexdigest())


class PairedPortfolioEngine:
    """Run a trace with injected rendering/serialization and caller-owned I/O."""

    def __init__(self, *, render: Callable[[Any, str], Mapping[str, Any]] = render_order_state,
                 prompt: Callable[[Any, str], str] = order_prompt,
                 serializer: Callable[[Any, str], Any] = serialize_explanation_wire) -> None:
        self.render = render
        self.prompt = prompt
        self.serializer = serializer

    async def run_session(self, base: Mapping[str, Any], arm: str, backend: Any,
                          next_action: Callable[[Any, Sequence[Any]], Mapping[str, Any]],
                          build_request: Callable[[Any, Sequence[Any]], Any],
                          advance: Callable[[Any, Sequence[Any], Any, Any], Sequence[Any]],
                          replay: Callable[[Any, Sequence[Any]], Mapping[str, Any]],
                          register_wire: Callable[[Any], Any], recorder: Any) -> dict[str, Any]:
        """Execute one complete session; all transport and trace policy stays injected."""
        if arm not in ARMS:
            raise ValueError("unknown order-explanation arm")
        trace: Sequence[Any] = ()
        events: list[Any] = []
        while True:
            action = next_action(base, trace)
            if action["kind"] == "complete":
                result = replay(base, trace)
                if result.get("status") != "complete":
                    raise RuntimeError("provider trace did not complete")
                return {"trace": events, "orders": [order.__dict__ for order in result["orders"]]}
            if action["kind"] == "forced":
                trace = advance(base, trace, None, None)
                events.append(trace[-1])
                continue
            if action["kind"] != "request":
                raise RuntimeError("provider trace became incomplete")
            request = build_request(base, trace)
            wire = await self.serializer(request, arm)
            register_wire(wire)
            (reply,) = await backend.decide_many([request])
            record = recorder.rows.get(request.request_id)
            if not record or record.get("status") != 200:
                raise RuntimeError("request-ID-bound HTTP response is missing or unsuccessful")
            import json
            answer = json.loads(record["response_body_utf8"])["answers"]["decision"]
            trace = advance(base, trace, answer, reply)
            events.append(trace[-1])


class ExplanationBackend(JevBackend):
    """Jev backend bound to injected arm-specific prompt and state renderers."""

    def __init__(self, *args: Any, arm: str, render: Callable = render_order_state,
                 prompt: Callable = order_prompt, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if arm not in ARMS:
            raise ValueError("unknown order-explanation arm")
        self.arm, self.render_request, self.prompt_request = arm, render, prompt

    async def decide_many(self, requests: Any) -> list[Any]:
        requests = tuple(requests)
        rendered = [(request, self.prompt_request(request, self.arm), self.render_request(request, self.arm))
                    for request in requests]
        estimates = [estimate_input_tokens(request, prompt, state)
                     for request, prompt, state in rendered]
        await self._reserve(requests, estimates)
        return list(await asyncio.gather(*(self._one(request, estimate, prompt, state)
            for (request, prompt, state), estimate in zip(rendered, estimates))))

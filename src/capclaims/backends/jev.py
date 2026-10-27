"""Rate-bounded asynchronous Jev adapter for private paper-market decisions."""
from __future__ import annotations

import asyncio
import contextvars
import copy
import hashlib
import math
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from capclaims.engine.protocol import DecisionReply, DecisionRequest
from capclaims.backends.jev_instructions import build_instruction, validate_participant_prediction_request
from capclaims.randomness import canonical_json

MODEL = "jev-1.14.0"
_CURRENT_REQUEST_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar("jev_request_id", default=None)


class RecordingTransport:
    """Instrument HTTP dispatch and response-header timing without retaining payloads."""

    def __init__(self, inner: Any = None) -> None:
        import httpx2
        self._inner = inner if inner is not None else httpx2.AsyncHTTPTransport(retries=0)
        self._timings: list[dict[str, Any]] = []

    @property
    def timings(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(row) for row in self._timings)

    async def handle_async_request(self, request: Any) -> Any:
        started = time.monotonic()
        request_id = _CURRENT_REQUEST_ID.get()
        try:
            response = await self._inner.handle_async_request(request)
        except BaseException as exc:
            finished = time.monotonic()
            self._timings.append({"request_id": request_id, "started_at": started,
                "finished_at": finished, "duration_seconds": finished - started,
                "status": "cancelled" if isinstance(exc, asyncio.CancelledError) else "transport_error"})
            raise
        finished = time.monotonic()
        self._timings.append({"request_id": request_id, "started_at": started,
            "finished_at": finished, "duration_seconds": finished - started,
            "status": "response_headers_received", "status_code": response.status_code})
        return response

    async def aclose(self) -> None:
        close = getattr(self._inner, "aclose", None)
        if close is not None:
            await close()


def load_local_env(path: str | Path | None = None) -> dict[str, str]:
    """Load simple KEY=VALUE entries without overriding inherited environment."""
    env_path = Path(path) if path is not None else Path.cwd() / ".env"
    loaded: dict[str, str] = {}
    if not env_path.is_file():
        return loaded
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip("\"'")
        if key and key not in os.environ:
            os.environ[key] = value
            loaded[key] = value
        elif key:
            loaded[key] = os.environ[key]
    return loaded


def estimate_input_tokens(request: DecisionRequest, instruction_text: str | None = None,
                          effective_state: Mapping[str, Any] | None = None) -> int:
    """Conservative preflight estimate: UTF-8 bytes of state, instruction and labels."""
    instruction_text = instruction_text or build_instruction(request)
    wire_equivalent = canonical_json({"state": request.observation if effective_state is None else effective_state, "instruction": instruction_text,
        "criteria": {c.candidate_id: c.label for c in request.candidates}})
    # This is a deterministic UTF-8 byte estimate plus fixed allowance, not provider token usage.
    return max(1, len(wire_equivalent.encode("utf-8")) + 256)


def normalize_jev_probabilities(raw: Mapping[str, float], candidate_ids: Sequence[str]) -> tuple[dict[str, float], dict[str, Any]]:
    """Normalize only provider outputs demonstrably rounded to two decimals.

    Jev's observed response precision is two decimals. For a k-option response,
    independent rounding can move its total by at most k*0.005. Larger or
    otherwise unexplained errors remain contract failures.
    """
    if len(candidate_ids) != len(set(candidate_ids)) or set(raw) != set(candidate_ids):
        raise ValueError("Jev candidate probability keys do not match the requested candidate set")
    values: dict[str, float] = {}
    for key in candidate_ids:
        value = raw[key]
        if isinstance(value, bool):
            raise ValueError("Jev probabilities must be finite nonnegative numbers")
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("Jev probabilities must be finite nonnegative numbers") from exc
        if not math.isfinite(number) or number < 0 or number > 1:
            raise ValueError("Jev probabilities must be finite numbers between zero and one")
        values[key] = number
    total = math.fsum(values.values())
    if total <= 0:
        raise ValueError("Jev probability sum must be positive")
    two_decimal = all(abs(value - round(value, 2)) <= 1e-12 for value in values.values())
    normalized = False
    if abs(total - 1.0) > 1e-6:
        bound = len(values) * 0.005 + 1e-12
        if not two_decimal or abs(total - 1.0) > bound:
            raise ValueError(f"Jev probability sum {total} exceeds verified two-decimal rounding bound")
        values = {key: value / total for key, value in values.items()}
        normalized = True
    metadata = {"rounding_normalized": normalized,
        "source_precision_decimals": 2 if two_decimal else None,
        "raw_probability_total": total}
    return values, metadata


def _limit(name: str, explicit: float | int | None, *, integer: bool = False) -> float | int:
    raw: Any = explicit if explicit is not None else os.getenv(name)
    if raw is None or raw == "":
        raise ValueError(f"Jev requires {name.removeprefix('JEV_')} configured as an account limit or local test bound")
    try:
        if integer and (isinstance(raw, bool) or (isinstance(raw, float) and not raw.is_integer())):
            raise ValueError
        value = int(raw) if integer else float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive number") from exc
    if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive number")
    return value


def _prediction_state(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Copy an observation and replace only the known optimizer objective with neutral accounting text."""
    state = copy.deepcopy(dict(observation))
    instructions = state.get("public_instructions")
    if isinstance(instructions, dict):
        profit = instructions.get("profit")
        prefix = "Maximize per-round expected profit:"
        if isinstance(profit, str) and profit.startswith(prefix):
            instructions["profit"] = "Per-round profit accounting: " + profit[len(prefix):].lstrip()
    return state


class JevBackend:
    """One isolated request per agent, bounded by shared rate and run budgets."""

    def __init__(self, *, api_key: str | None = None, rps: float | None = None,
                 tps: float | None = None, max_requests: int | None = None,
                 max_estimated_input_tokens: int | None = None,
                 instruction_profile: str | None = None, request_timeout: float = 30,
                 max_in_flight: int = 10, client: Any = None,
                 transport_recorder: RecordingTransport | None = None,
                 objective: str = "optimizer", participant_conditioner: Any = None,
                 participant_targets: Mapping[str, Any] | None = None,
                 load_environment: bool = True) -> None:
        if load_environment:
            load_local_env()
        self.api_key = api_key or os.getenv("TYPESAFE_API_KEY")
        if not self.api_key:
            raise ValueError("Jev API key missing; set TYPESAFE_API_KEY in the process environment or project .env")
        self.rps = float(_limit("JEV_RPS", rps if rps is not None else os.getenv("JEV_RPS", "40")))
        self.tps = float(_limit("JEV_TPS", tps if tps is not None else os.getenv("JEV_TPS", "50000")))
        self.max_requests = int(_limit("JEV_MAX_REQUESTS", max_requests if max_requests is not None else os.getenv("JEV_MAX_REQUESTS", "10"), integer=True))
        self.max_estimated_input_tokens = int(_limit(
            "JEV_MAX_ESTIMATED_INPUT_TOKENS", max_estimated_input_tokens if max_estimated_input_tokens is not None else os.getenv("JEV_MAX_ESTIMATED_INPUT_TOKENS", "100000"), integer=True))
        if not math.isfinite(request_timeout) or request_timeout <= 0:
            raise ValueError("request_timeout must be positive")
        if (isinstance(max_in_flight, bool) or not isinstance(max_in_flight, int)
                or not 1 <= max_in_flight <= 10):
            raise ValueError("max_in_flight must be between 1 and 10")
        self.instruction_profile = instruction_profile or os.getenv("JEV_INSTRUCTION_PROFILE", "explicit_v1")
        if self.instruction_profile not in {"explicit_v1", "matched_v2", "ablation_v1", "calculator_v1", "calculator_v2", "calculator_v3", "participant_prediction_v1", "procurement_first_v1", "procurement_third_v1", "portfolio_first_v1", "portfolio_third_v1"}:
            raise ValueError("unknown JEV_INSTRUCTION_PROFILE")
        if objective not in {"optimizer", "participant_prediction_v1", "procurement_prediction_v1", "portfolio_prediction_v1"}:
            raise ValueError("unknown Jev objective")
        if objective == "procurement_prediction_v1" and participant_conditioner is not None:
            raise ValueError("procurement prediction uses study-owned history, not ParticipantConditioner")
        if participant_conditioner is not None and (objective != "participant_prediction_v1"
                or self.instruction_profile != "participant_prediction_v1"):
            raise ValueError("participant conditioning requires participant_prediction_v1 objective and profile")
        if objective == "participant_prediction_v1" and self.instruction_profile != "participant_prediction_v1":
            raise ValueError("participant_prediction_v1 objective cannot use optimizer/calculator instructions")
        if self.instruction_profile == "participant_prediction_v1" and objective != "participant_prediction_v1":
            raise ValueError("participant_prediction_v1 instructions require the participant prediction objective")
        procurement_profile = self.instruction_profile in {"procurement_first_v1", "procurement_third_v1"}
        if procurement_profile != (objective == "procurement_prediction_v1"):
            raise ValueError("procurement prediction profiles require procurement_prediction_v1 objective")
        portfolio_profile = self.instruction_profile in {"portfolio_first_v1", "portfolio_third_v1"}
        if portfolio_profile != (objective == "portfolio_prediction_v1"):
            raise ValueError("portfolio prediction profiles require portfolio_prediction_v1 objective")
        if objective == "portfolio_prediction_v1" and participant_conditioner is not None:
            raise ValueError("portfolio prediction uses study-owned history, not ParticipantConditioner")
        self.objective = objective
        self.participant_conditioner = participant_conditioner
        self.participant_targets = dict(participant_targets or {})
        self.request_timeout = request_timeout
        self._semaphore = asyncio.Semaphore(max_in_flight)
        self._budget_lock = asyncio.Lock()
        self._pace_lock = asyncio.Lock()
        self._next_request = 0.0
        self._next_token = 0.0
        self._reserved_requests = 0
        self._reserved_tokens = 0
        self._completed_requests = 0
        self._started_requests = 0
        self._actual_input_tokens = 0
        self._actual_output_tokens = 0
        self._timings: list[dict[str, Any]] = []
        self._in_flight = 0
        self._max_observed_in_flight = 0
        self._closed = False
        self._transport_recorder = transport_recorder
        if client is None:
            from typesafe_sdk import AsyncTypeSafeClient
            from typesafe_sdk._core.retry import RetryPolicy
            self._transport_recorder = transport_recorder or RecordingTransport()
            client = AsyncTypeSafeClient(api_key=self.api_key, model=MODEL,
                                         retry=RetryPolicy(max_retries=0), timeout=request_timeout,
                                         transport=self._transport_recorder)
        self._client = client

    @property
    def revision(self) -> str:
        return MODEL

    @property
    def budget_usage(self) -> dict[str, int | float]:
        return {"requests_reserved": self._reserved_requests,
            "estimated_input_tokens_reserved": self._reserved_tokens,
            "requests_completed": self._completed_requests,
            "requests_started": self._started_requests,
            "actual_input_tokens_reported": self._actual_input_tokens,
            "actual_output_tokens_reported": self._actual_output_tokens,
            "max_requests": self.max_requests,
            "max_estimated_input_tokens": self.max_estimated_input_tokens,
            "rate_limit_source": "client_configured_margin_from_user_reported_org_limit"}

    @property
    def window_timings(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(row) for row in self._timings)

    @property
    def transport_timings(self) -> tuple[dict[str, Any], ...]:
        """Actual HTTP dispatch windows; empty for clients without an instrumented transport."""
        if self._transport_recorder is None:
            return ()
        return self._transport_recorder.timings

    async def _reserve(self, requests: Sequence[DecisionRequest], estimates: Sequence[int]) -> None:
        async with self._budget_lock:
            new_requests = self._reserved_requests + len(requests)
            new_tokens = self._reserved_tokens + sum(estimates)
            if new_requests > self.max_requests:
                raise ValueError("Jev request budget would be exceeded; no calls were sent")
            if new_tokens > self.max_estimated_input_tokens:
                raise ValueError("Jev estimated input-token budget would be exceeded; no calls were sent")
            self._reserved_requests, self._reserved_tokens = new_requests, new_tokens

    async def _pace(self, estimated_tokens: int) -> None:
        async with self._pace_lock:
            now = time.monotonic()
            start = max(now, self._next_request, self._next_token)
            self._next_request = start + 1.0 / self.rps
            self._next_token = start + estimated_tokens / self.tps
        delay = start - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)

    async def _one(self, request: DecisionRequest, estimate: int, prompt: str,
                   effective_state: Mapping[str, Any], conditioner_metadata: Mapping[str, Any] | None = None) -> DecisionReply:
        from typesafe_sdk import Choice
        criteria = {candidate.candidate_id: candidate.label for candidate in request.candidates}
        async with self._semaphore:
            await self._pace(estimate)
            started = time.monotonic()
            self._in_flight += 1
            self._started_requests += 1
            self._max_observed_in_flight = max(self._max_observed_in_flight, self._in_flight)
            in_flight_at_start = self._in_flight
            call_status = "response_received"
            context_token = _CURRENT_REQUEST_ID.set(request.request_id)
            try:
                response = await asyncio.wait_for(self._client.system_one(
                    state=dict(effective_state),
                    questions={"decision": Choice(instructions=prompt, criteria=criteria)},
                    model=MODEL,
                ), timeout=self.request_timeout)
            except asyncio.TimeoutError as exc:
                call_status = "timeout"
                raise TimeoutError(f"Jev request timed out for {request.request_id}") from exc
            except Exception as exc:
                call_status = "provider_error"
                # SDK exceptions can contain raw provider response text. Keep evidence metadata safe.
                raise RuntimeError(f"Jev provider call failed for {request.request_id} ({type(exc).__name__})") from None
            except BaseException:
                call_status = "cancelled"
                raise
            finally:
                _CURRENT_REQUEST_ID.reset(context_token)
                finished = time.monotonic()
                self._in_flight -= 1
                timing_window = {"request_id": request.request_id, "started_at": started,
                    "finished_at": finished, "duration_seconds": finished - started,
                    "in_flight_at_start": in_flight_at_start,
                    "max_observed_in_flight": self._max_observed_in_flight,
                    "status": call_status,
                    "rate_limit_source": "client_configured_margin_from_user_reported_org_limit"}
                self._timings.append(timing_window)
        model = _field(response, "model")
        usage_obj = _field(response, "usage", {})
        input_tokens = _optional_field(usage_obj, "input_tokens")
        output_tokens = _optional_field(usage_obj, "output_tokens")
        timing_window["response_model"] = model
        timing_window["actual_input_tokens"] = input_tokens
        timing_window["actual_output_tokens"] = output_tokens
        # Capture provider usage immediately, even if model/probability validation below fails.
        if isinstance(input_tokens, int) and not isinstance(input_tokens, bool):
            self._actual_input_tokens += input_tokens
        if isinstance(output_tokens, int) and not isinstance(output_tokens, bool):
            self._actual_output_tokens += output_tokens
        if model != MODEL:
            raise ValueError(f"Jev returned unexpected model identity for {request.request_id}")
        answer = _field(response, "choices", {}).get("decision")
        raw_probabilities = _field(answer, "probabilities")
        if not isinstance(raw_probabilities, Mapping) or set(raw_probabilities) != set(criteria):
            raise ValueError(f"Jev probability candidate coverage mismatch for {request.request_id}")
        probabilities, probability_metadata = normalize_jev_probabilities(raw_probabilities, tuple(criteria))
        usage = {"estimated_input_tokens": estimate,
                 "actual_input_tokens": input_tokens,
                 "actual_output_tokens": output_tokens,
                 **probability_metadata}
        self._completed_requests += 1
        timing = {"started_at": started, "finished_at": finished,
            "duration_seconds": finished - started, "in_flight_at_start": in_flight_at_start,
            "max_observed_in_flight": self._max_observed_in_flight,
            "rate_limit_source": "client_configured_margin_from_user_reported_org_limit",
            "instruction_profile": self.instruction_profile, "instruction_text": prompt,
            "instruction_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest()}
        if conditioner_metadata is not None:
            timing["participant_history_sha256"] = conditioner_metadata["context_sha256"]
            timing["participant_history_episode_count"] = conditioner_metadata["episode_count"]
            timing["effective_state_sha256"] = hashlib.sha256(
                canonical_json(effective_state).encode("utf-8")).hexdigest()
        return DecisionReply.from_request(request, probabilities, dict(raw_probabilities), backend="jev",
            model_revision=model, usage=usage, timing=timing)

    async def decide_many(self, requests: Sequence[DecisionRequest]) -> list[DecisionReply]:
        if self._closed:
            raise RuntimeError("Jev backend is closed")
        requests = tuple(requests)
        if not requests:
            return []
        if len({request.request_id for request in requests}) != len(requests):
            raise ValueError("duplicate Jev request IDs")
        prepared: list[tuple[DecisionRequest, str, Mapping[str, Any], Mapping[str, Any] | None]] = []
        for request in requests:
            if self.objective != "participant_prediction_v1":
                prompt = build_instruction(request, self.instruction_profile)
                prepared.append((request, prompt, copy.deepcopy(dict(request.observation)), None))
                continue
            expected_phase = validate_participant_prediction_request(request)
            effective_state = _prediction_state(request.observation)
            metadata = None
            if self.participant_conditioner is not None:
                target = self.participant_targets.get(request.request_id)
                if target is None or target.request_id != request.request_id or target.agent_id != request.agent_id:
                    raise ValueError(f"participant conditioning target missing or mismatched for {request.request_id}")
                observation_round = request.observation.get("round", request.round_index + 1)
                if type(observation_round) is not int or target.target_round != observation_round:
                    raise ValueError("participant conditioning target round does not match request observation")
                if target.phase != expected_phase:
                    raise ValueError("participant conditioning target phase does not match request")
                nested_context = request.observation.get("context")
                if ("participant_history" in request.observation
                        or isinstance(nested_context, Mapping) and "participant_history" in nested_context):
                    raise ValueError("request uses reserved participant history/context namespace")
                if expected_phase == "procurement" and any(request.observation.get(key) is not None
                        for key in ("D", "d", "demand", "own_d", "own_d_units", "customer_demand_units",
                                    "realized_demand", "customer_demand", "demand_revealed", "profit",
                                    "profit_cents", "market_clearing_price", "fills", "executed_orders")):
                    raise ValueError("procurement observation contains realized demand/outcome fields")
                metadata = self.participant_conditioner.prepare(target, provider=True)
                context = {key: value for key, value in metadata.items()
                           if key not in {"context_sha256", "episode_count"}}
                effective_state["participant_history"] = context
            prompt = build_instruction(request, self.instruction_profile)
            prepared.append((request, prompt, effective_state, metadata))
        # All targets, contexts, and instructions are validated before estimating or reserving any batch budget.
        estimates = [estimate_input_tokens(request, prompt, state)
                     for request, prompt, state, _ in prepared]
        await self._reserve(requests, estimates)
        tasks = [asyncio.create_task(self._one(request, estimate, prompt, state, metadata))
                 for (request, prompt, state, metadata), estimate in zip(prepared, estimates)]
        try:
            return list(await asyncio.gather(*tasks))
        except BaseException:
            for task in tasks:
                if not task.done(): task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        close = getattr(self._client, "aclose", None) or getattr(self._client, "close", None)
        if close is not None:
            result = close()
            if hasattr(result, "__await__"):
                await result


def _field(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping): return obj.get(name, default)
    return getattr(obj, name, default)


def _optional_field(obj: Any, name: str) -> Any:
    return _field(obj, name)

"""Real Jev 1.14 Choice score and request-overlap probe."""
from __future__ import annotations
import argparse
import asyncio
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.probes.common.fixture import load_worst_case_fixture, sha256_json, write_json
from tools.probes.common.fixture_validation import validate_fixture
from tools.probes.common.result import base_result, package_version, sanitize
from tools.probes.common.score_contract import normalize_scores

MODEL_ID = "jev-1.14.0"
MAX_IN_FLIGHT = 10


class ConfiguredRateLimiter:
    """Pace logical requests using user-supplied RPS and estimated input TPS ceilings."""
    def __init__(self, requests_per_second: float, tokens_per_second: float) -> None:
        if not math.isfinite(requests_per_second) or not math.isfinite(tokens_per_second) or requests_per_second <= 0 or tokens_per_second <= 0:
            raise ValueError("configured RPS and TPS must be positive")
        self.rps = requests_per_second
        self.tps = tokens_per_second
        self.next_request_at = 0.0
        self.next_tokens_at = 0.0
        self.lock = asyncio.Lock()

    async def acquire(self, estimated_tokens: int) -> float:
        loop = asyncio.get_running_loop()
        async with self.lock:
            now = loop.time()
            dispatch_at = max(now, self.next_request_at, self.next_tokens_at)
            self.next_request_at = dispatch_at + 1.0 / self.rps
            self.next_tokens_at = dispatch_at + max(1, estimated_tokens) / self.tps
        delay = max(0.0, dispatch_at - loop.time())
        if delay:
            await asyncio.sleep(delay)
        return dispatch_at


def _as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "dict"):
        return value.dict()
    if hasattr(value, "__dict__"):
        return vars(value)
    raise TypeError(f"unsupported Jev response payload type: {type(value).__name__}")


def _choice_payload(response: Any) -> tuple[str, dict[str, float], Any]:
    choices = getattr(response, "choices", None)
    if choices is None and isinstance(response, dict):
        choices = response.get("choices")
    decision = choices.get("decision") if isinstance(choices, dict) else None
    if decision is None:
        raise RuntimeError(f"Jev response is missing choices['decision']: {_as_mapping(response)}")
    selected = getattr(decision, "choice", None)
    probabilities = getattr(decision, "probabilities", None)
    if isinstance(decision, dict):
        selected = decision.get("choice", selected)
        probabilities = decision.get("probabilities", probabilities)
    if selected is None or not isinstance(probabilities, dict):
        raise RuntimeError(f"Jev Choice response lacks selected choice or full probabilities: {_as_mapping(decision)}")
    return str(selected), {str(key): float(value) for key, value in probabilities.items()}, _as_mapping(response)


def _estimated_tokens(state: str, wave: dict[str, Any]) -> int:
    criteria = " ".join(candidate["label"] for candidate in wave["candidates"])
    # Conservative local token estimate for rate pacing only; actual usage is recorded separately.
    return max(1, (len(state) + len(wave["question"]) + len(criteria) + 3) // 4)


def _overlap_count(windows: list[dict[str, Any]]) -> int:
    return sum(
        1 for i, left in enumerate(windows)
        for right in windows[i + 1:]
        if max(left["started_at"], right["started_at"]) < min(left["finished_at"], right["finished_at"])
    )


async def _run_real_smoke(api_key: str, rps: float, tps: float, deadline_seconds: float) -> tuple[dict[str, Any], dict[str, Any]]:
    from typesafe_sdk import AsyncTypeSafeClient, Choice, RetryPolicy

    fixture = load_worst_case_fixture()
    wave = fixture["waves"]["purchase"]
    criteria = {candidate["candidate_id"]: candidate["label"] for candidate in wave["candidates"]}
    states = [
        json.dumps({"observation": fixture["observation"], "history": fixture["history"], "probe_row": index},
                   sort_keys=True, ensure_ascii=False)
        for index in range(2)
    ]
    limiter = ConfiguredRateLimiter(rps, tps)
    semaphore = asyncio.Semaphore(MAX_IN_FLIGHT)
    windows: list[dict[str, Any]] = []
    raw_outputs: list[dict[str, Any]] = []

    async with AsyncTypeSafeClient(api_key=api_key, retry=RetryPolicy(max_retries=0), model=MODEL_ID) as client:
        async def request(index: int) -> dict[str, Any]:
            loop = asyncio.get_running_loop()
            deadline_at = loop.time() + deadline_seconds
            estimated_tokens = _estimated_tokens(states[index], wave)
            await limiter.acquire(estimated_tokens)
            async with semaphore:
                remaining = deadline_at - loop.time()
                if remaining <= 0:
                    raise TimeoutError("Jev request deadline expired while waiting for limiter or semaphore")
                start = time.perf_counter()
                try:
                    response = await asyncio.wait_for(
                        client.system_one(
                            state=states[index],
                            questions={"decision": Choice(instructions=wave["question"], criteria=criteria)},
                        ),
                        timeout=remaining,
                    )
                finally:
                    finish = time.perf_counter()
                    windows.append({"request_index": index, "started_at": start, "finished_at": finish,
                                    "duration_seconds": finish - start})
            selected, scores, raw = _choice_payload(response)
            probabilities = normalize_scores(scores, list(criteria), "probabilities")
            raw_outputs.append({"request_index": index, "selected": selected, "probabilities": scores,
                                "normalized_probabilities": probabilities, "response": raw})
            return {"selected": selected, "scores": probabilities}
        responses = await asyncio.gather(*(request(index) for index in range(2)))
    return ({"windows": windows, "overlap_pairs": _overlap_count(windows), "responses": responses,
             "rps": rps, "tps": tps, "max_in_flight": MAX_IN_FLIGHT},
            {"request_responses": raw_outputs})


def run_probe(rps: float | None, tps: float | None, output_dir: Path, deadline_seconds: float = 30.0) -> dict[str, Any]:
    fixture = load_worst_case_fixture()
    validate_fixture(fixture)
    result = base_result("jev", sha256_json(fixture))
    result["model_revision"] = MODEL_ID
    raw: dict[str, Any] = {"fixture_hash": result["fixture_hash"], "model": MODEL_ID,
                           "rps_limit": rps, "tps_limit": tps, "sdk_retry_max_retries": 0}
    api_key = os.environ.get("TYPESAFE_API_KEY")
    if not api_key:
        result["status"] = "BLOCKED"
        result["error"] = "TYPESAFE_API_KEY is not set; real Jev access and overlap cannot be claimed."
    elif rps is None or tps is None:
        result["status"] = "BLOCKED"
        result["error"] = "Explicit account RPS and TPS limits are required before real Jev requests."
    else:
        try:
            result["supports_batch"] = False
            result["tested_candidate_count"] = len(criteria)
            result["supports_scores"] = False
            result["status"] = "FAIL"
            smoke, raw_data = asyncio.run(_run_real_smoke(api_key, rps, tps, deadline_seconds))
            raw.update(smoke)
            raw.update(raw_data)
            result["sdk_revision"] = package_version("typesafe-sdk") or package_version("typesafe_sdk")
            result["supports_scores"] = all(len(item["scores"]) == result["tested_candidate_count"] for item in smoke["responses"])
            result["score_semantics"] = "probabilities"
            result["cost_sample"] = {"requests": 2, "usage_by_request": [item["response"].get("usage") for item in raw.get("request_responses", [])], "provider_idempotency_support": "not verified"}
            result["parallel_overlap_pairs"] = smoke["overlap_pairs"]
            result["real_requests_overlapped"] = smoke["overlap_pairs"] >= 1
            result["argmax_agreement"] = sum(item["selected"] == max(item["scores"], key=item["scores"].get) for item in smoke["responses"]) / len(smoke["responses"])
            result["status"] = "PASS" if result["supports_scores"] and result["real_requests_overlapped"] else "FAIL"
            if not result["real_requests_overlapped"]:
                result["error"] = "Two real requests completed without an in-flight overlap; account limits did not satisfy M0 concurrency."
        except Exception as exc:
            result["status"] = "FAIL"
            result["error"] = f"{type(exc).__name__}: {exc}"
            raw["error"] = result["error"]
    write_json(output_dir / "jev-result.json", sanitize(result))
    write_json(output_dir / "jev-raw.json", sanitize(raw))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rps", type=float, help="explicit account request-per-second ceiling")
    parser.add_argument("--tps", type=float, help="explicit account token-per-second ceiling")
    parser.add_argument("--deadline-seconds", type=float, default=30.0)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    result = run_probe(args.rps, args.tps, args.output_dir, args.deadline_seconds)
    print(json.dumps(sanitize(result), indent=2, sort_keys=True))
    return 0 if result.get("status") == "PASS" else (2 if result.get("status") == "BLOCKED" else 1)

if __name__ == "__main__":
    raise SystemExit(main())

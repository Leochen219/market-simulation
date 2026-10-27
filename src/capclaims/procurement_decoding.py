"""Private all-prefix decoding diagnostic for two frozen procurement inputs."""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from capclaims.backends.jev import MODEL, JevBackend, RecordingTransport
from capclaims.domain.types import Candidate
from capclaims.engine.protocol import DecisionRequest, validate_reply
from capclaims.procurement_perspective import (
    EXPECTED_SOURCE_SHA, MAX_ESTIMATED_BYTES, MAX_IN_FLIGHT, MAX_REQUESTS,
    MAX_RPS, MAX_TPS_BYTES, SharedJevClient, StudyBudget, _sdk_mock,
    _sha, _wire, build_choice_prefixes, make_request,
)
from capclaims.paper.codec import legal_prefix_choices
from capclaims.randomness import canonical_json

DIAGNOSTIC_MAX_CALLS = 90
SINGLETON_PREFIXES = frozenset({"4", "40"})
PARENT_REQUEST_COUNT = 377
PARENT_ESTIMATED_BYTES = 1_483_040
SOURCE_CODE_PATHS = (
    "src/capclaims/procurement_decoding.py", "tools/probes/jev_procurement_decoding.py",
    "src/capclaims/procurement_perspective.py", "src/capclaims/backends/jev.py",
    "src/capclaims/backends/jev_instructions.py", "src/capclaims/paper/codec.py",
    "src/capclaims/paper/interface.py", "src/capclaims/engine/protocol.py",
    "src/capclaims/randomness.py",
    "docs/superpowers/plans/2026-10-24-procurement-decoding-diagnostic.md",
)

PROTOCOL_PATH = "docs/superpowers/plans/2026-10-24-procurement-decoding-diagnostic.md"


def _protocol_snapshot() -> tuple[str, str]:
    path = Path(__file__).resolve().parents[2] / PROTOCOL_PATH
    content = path.read_bytes()
    return content.decode("utf-8"), hashlib.sha256(content).hexdigest()


def _validate_protocol_snapshot(plan: Mapping[str, Any]) -> None:
    text, digest = _protocol_snapshot()
    if digest != plan.get("protocol_sha256") or text != plan.get("protocol_text"):
        raise ValueError("frozen decoding diagnostic protocol changed since plan creation")


def _validate_code_snapshot(plan: Mapping[str, Any]) -> None:
    if _code_hashes() != plan.get("source_code_sha256"):
        raise ValueError("frozen diagnostic source or protocol code changed since plan creation")


def maximum_overlap(timings: Sequence[Mapping[str, Any]]) -> int:
    events = []
    for row in timings:
        start, finish = row.get("started_at"), row.get("finished_at")
        if (not isinstance(start, (int, float)) or not isinstance(finish, (int, float))
                or finish < start):
            raise ValueError("transport timing rows must contain valid intervals")
        events.extend(((float(start), 1), (float(finish), -1)))
    active = maximum = 0
    for _, delta in sorted(events, key=lambda item: (item[0], item[1])):
        active += delta
        maximum = max(maximum, active)
    return maximum


def _code_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in SOURCE_CODE_PATHS}


def _wire_table(parent: Mapping[str, Any], sessions: Sequence[Mapping[str, Any]]) -> tuple[dict, dict, list]:
    selected = {(int(row["sequence"]), row["arm"], prefix)
        for row in sessions for prefix in build_choice_prefixes()}
    stored = {(int(row["sequence"]), row["arm"], row["prefix"]): row
        for row in parent["wire_prefixes"] if (int(row["sequence"]), row["arm"], row["prefix"]) in selected}
    if set(stored) != selected:
        raise ValueError("parent plan is missing one or more frozen prefix wires")
    prefix_hashes, allowed, rows = {}, set(), []
    for key in sorted(selected):
        frozen = stored[key]
        digest = _sha(frozen["wire"])
        if digest != frozen["wire_sha256"]:
            raise ValueError("parent frozen prefix wire digest mismatch")
        prefix_hashes[key] = digest
        allowed.add(digest)
        rows.append(frozen)
    return prefix_hashes, allowed, rows


def _assert_unique_prediction_sequences(rows: Sequence[Mapping[str, Any]]) -> None:
    sequences = [row.get("sequence") for row in rows]
    if any(type(sequence) is not int for sequence in sequences) or len(set(sequences)) != len(sequences):
        raise ValueError("parent private predictions contain duplicate or invalid sequences")


def build_diagnostic_plan(parent_run: str | Path) -> dict[str, Any]:
    parent_dir = Path(parent_run).resolve()
    plan_path, result_path, predictions_path = (parent_dir / "frozen-plan.json",
        parent_dir / "result.json", parent_dir / "private-predictions.json")
    parent_bytes = plan_path.read_bytes()
    parent = json.loads(parent_bytes)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    old_predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    if parent.get("schema") != "jev_procurement_perspective_v1":
        raise ValueError("parent run is not the frozen procurement perspective plan")
    if parent.get("model") != MODEL:
        raise ValueError("parent model differs from the frozen decoding diagnostic model")
    attest = parent.get("source_attestations", {})
    if (parent.get("source_kind") != "human" or parent.get("source_sha256") != EXPECTED_SOURCE_SHA
            or not all(attest.get(k) is True for k in ("codebook_verified", "intended_use_authorized", "provider_transmission_permitted"))):
        raise PermissionError("verified and authorized human-source parent is required")
    if result.get("status") != "COMPLETE" or result.get("prediction_n") != 126 or result.get("failure_n") != 0:
        raise ValueError("diagnostic requires a complete 126-prediction parent run")
    parent_budget = result.get("budget", {})
    if parent_budget.get("requests") != PARENT_REQUEST_COUNT or parent_budget.get("estimated_bytes") != PARENT_ESTIMATED_BYTES:
        raise ValueError("completed parent budget differs from the reviewed pilot budget")
    expected_hashes = parent.get("source_code_sha256", {})
    root = Path(__file__).resolve().parents[2]
    for relpath, expected in expected_hashes.items():
        if hashlib.sha256((root / relpath).read_bytes()).hexdigest() != expected:
            raise ValueError(f"frozen parent source-code hash changed: {relpath}")
    _assert_unique_prediction_sequences(old_predictions)
    sessions = [row for row in parent["sessions"] if row["arm"] == "first_own"
        and row["phase"] == "development" and row["round"] == 9
        and row["repetition"] == 1 and row["participant"] in {"0", "1"}]
    if sorted(row["participant"] for row in sessions) != ["0", "1"] or len(sessions) != 2:
        raise ValueError("parent run does not contain the two frozen alias inputs")
    selected_predictions = {row["sequence"]: row for row in old_predictions
        if row.get("sequence") in {s["sequence"] for s in sessions}}
    if len(selected_predictions) != 2:
        raise ValueError("saved greedy outputs are missing for selected inputs")
    prefix_hashes, _all_prefix_hashes, frozen_rows = _wire_table(parent, sessions)
    allowed = {digest for (sequence, arm, prefix), digest in prefix_hashes.items()
        if prefix not in SINGLETON_PREFIXES}
    code_hashes = _code_hashes()
    protocol_text, protocol_sha256 = _protocol_snapshot()
    for session in sessions:
        for prefix in build_choice_prefixes():
            request = make_request(session, prefix)
            regenerated = _wire(request, "first_own")
            frozen = next(row["wire"] for row in frozen_rows
                if row["sequence"] == session["sequence"] and row["prefix"] == prefix)
            if regenerated != frozen:
                raise ValueError("current request/prompt construction differs from the original frozen wire")
    singleton_count = 2 * len(SINGLETON_PREFIXES)
    request_count = 2 * (len(build_choice_prefixes()) - len(SINGLETON_PREFIXES))
    estimated_bytes = sum(row["estimated_bytes"] for row in frozen_rows
        if row["prefix"] not in SINGLETON_PREFIXES)
    if request_count != DIAGNOSTIC_MAX_CALLS or singleton_count != 4:
        raise AssertionError("unexpected diagnostic prefix count")
    if PARENT_REQUEST_COUNT + request_count > MAX_REQUESTS:
        raise ValueError("combined pilot and diagnostic request budget exceeds 480")
    if PARENT_ESTIMATED_BYTES + estimated_bytes > MAX_ESTIMATED_BYTES:
        raise ValueError("combined pilot and diagnostic estimated bytes exceed 3.2 million")
    truth = parent["truth"]
    selected_truth = {f"{s['participant']}:{s['round']}": truth[f"{s['participant']}:{s['round']}"] for s in sessions}
    return {
        "schema": "jev_procurement_decoding_diagnostic_v1",
        "parent_plan_sha256": hashlib.sha256(parent_bytes).hexdigest(),
        "parent_source_sha256": parent["source_sha256"], "source_attestations": attest,
        "parent_request_count": PARENT_REQUEST_COUNT, "parent_estimated_bytes": PARENT_ESTIMATED_BYTES,
        "diagnostic_request_count": request_count, "diagnostic_singleton_count": singleton_count,
        "combined_request_ceiling": PARENT_REQUEST_COUNT + request_count,
        "diagnostic_estimated_bytes": estimated_bytes,
        "combined_estimated_bytes": PARENT_ESTIMATED_BYTES + estimated_bytes,
        "max_requests": MAX_REQUESTS, "max_estimated_bytes": MAX_ESTIMATED_BYTES,
        "rps": MAX_RPS, "tps_bytes": MAX_TPS_BYTES, "max_in_flight": MAX_IN_FLIGHT,
        "model": parent["model"], "instruction_profile": "procurement_first_v1",
        "objective": "procurement_prediction_v1", "prefix_count_per_input": len(build_choice_prefixes()),
        "source_code_sha256": code_hashes,
        "protocol_path": PROTOCOL_PATH, "protocol_sha256": protocol_sha256,
        "protocol_text": protocol_text,
        "selected_sessions": sessions,
        "selected_saved_greedy": {row["participant"]: selected_predictions[row["sequence"]]["predicted_quantity"]
            for row in sessions},
        "prefix_hashes": [{"sequence": seq, "arm": arm, "prefix": prefix, "sha256": digest}
            for (seq, arm, prefix), digest in sorted(prefix_hashes.items())],
        "allowed_wire_hashes": sorted(allowed),
        "prefix_wires": frozen_rows,
        "truth": selected_truth,
        "_sessions": sessions, "_prefix_hashes": prefix_hashes,
        "_allowed_hashes": allowed, "_truth": selected_truth,
        "_saved_greedy": {row["participant"]: selected_predictions[row["sequence"]]["predicted_quantity"]
            for row in sessions},
    }


def make_synthetic_plan() -> dict[str, Any]:
    """Create a mock-only pair without loading or consulting any human source artifact."""
    sessions, truth, wires, prefix_hashes, allowed = [], {}, [], {}, set()
    for alias, q in (("0", 117), ("1", 263)):
        seq = int(alias)
        session = {"sequence": seq, "arm": "first_own", "participant": alias, "round": 9,
            "repetition": 1, "phase": "development", "history_condition": "own",
            "agent_id": f"synthetic-diagnostic-{alias}",
            "state": {"treatment": "GA", "market_size_note": "synthetic fixture market count unknown",
                "round": 9, "wholesale_price_dollars": 10, "own_p": 70,
                "public_instructions": {"treatment": "GA synthetic fixture"},
                "participant_history": {"schema": "participant_history_v1", "source_kind": "synthetic",
                    "treatment": "GA", "phase": "procurement", "episodes": [
                        {"history_ordinal": 1, "observed": {"p": 70, "w": 10}, "action": {"Q": 80}}]}}}
        sessions.append(session); truth[f"{alias}:9"] = q
        for prefix in build_choice_prefixes():
            request = make_request(session, prefix)
            wire = _wire(request, "first_own")
            digest = _sha(wire)
            prefix_hashes[(seq, "first_own", prefix)] = digest; allowed.add(digest)
            wires.append({"sequence": seq, "arm": "first_own", "prefix": prefix,
                "wire": wire, "wire_sha256": digest,
                "estimated_bytes": len(canonical_json(wire).encode()) + 256})
    calls = sum(row["prefix"] not in SINGLETON_PREFIXES for row in wires)
    byte_count = sum(row["estimated_bytes"] for row in wires if row["prefix"] not in SINGLETON_PREFIXES)
    query_hashes = {prefix_hashes[(seq, arm, prefix)] for seq, arm, prefix in prefix_hashes
        if prefix not in SINGLETON_PREFIXES}
    protocol_text, protocol_sha256 = _protocol_snapshot()
    return {"schema": "jev_procurement_decoding_synthetic_v1", "parent_plan_sha256": "synthetic",
        "parent_source_sha256": "synthetic", "source_attestations": {},
        "parent_request_count": 0, "parent_estimated_bytes": 0, "diagnostic_request_count": calls,
        "diagnostic_singleton_count": 4, "combined_request_ceiling": calls,
        "diagnostic_estimated_bytes": byte_count, "combined_estimated_bytes": byte_count,
        "max_requests": calls, "max_estimated_bytes": MAX_ESTIMATED_BYTES,
        "rps": MAX_RPS, "tps_bytes": MAX_TPS_BYTES, "max_in_flight": MAX_IN_FLIGHT,
        "model": MODEL, "instruction_profile": "procurement_first_v1", "objective": "procurement_prediction_v1",
        "prefix_count_per_input": 47, "source_code_sha256": _code_hashes(),
        "protocol_path": PROTOCOL_PATH, "protocol_sha256": protocol_sha256,
        "protocol_text": protocol_text,
        "selected_sessions": sessions, "selected_saved_greedy": {"0": None, "1": None},
        "prefix_hashes": [{"sequence": seq, "arm": arm, "prefix": prefix, "sha256": digest}
            for (seq, arm, prefix), digest in sorted(prefix_hashes.items())],
        "allowed_wire_hashes": sorted(query_hashes), "prefix_wires": wires, "truth": truth,
        "_sessions": sessions, "_prefix_hashes": prefix_hashes, "_allowed_hashes": query_hashes,
        "_truth": truth, "_saved_greedy": {"0": None, "1": None}}


def _conditional_distribution(rows: Mapping[str, Mapping[str, float]]) -> dict[str, Any]:
    probabilities: dict[int, float] = {}
    for q in range(401):
        digits = f"{q:03d}"
        value = 1.0
        for depth in range(3):
            prefix = digits[:depth]
            row = rows[prefix]
            value *= row[digits[depth]]
        probabilities[q] = value
    total = math.fsum(probabilities.values())
    if not math.isfinite(total) or abs(total - 1.0) > 1e-12:
        raise ValueError(f"induced quantity distribution is not normalized (mass={total})")
    modes = [q for q, p in probabilities.items() if abs(p - max(probabilities.values())) <= 1e-12]
    cumulative = 0.0
    median = 400
    for q in range(401):
        cumulative += probabilities[q]
        if cumulative >= 0.5:
            median = q
            break
    return {"probabilities": probabilities, "mass": total, "mode_set": modes,
        "median_lower": median, "expected_quantity": math.fsum(q * p for q, p in probabilities.items()),
        "mass_below_100": math.fsum(probabilities[q] for q in range(100))}


def summarize_tree(rows: Mapping[str, Mapping[str, float]]) -> dict[str, Any]:
    all_prefixes = set(build_choice_prefixes())
    if set(rows) != all_prefixes:
        raise ValueError("a full fresh tree requires all 47 legal prefixes")
    normalized: dict[str, dict[str, float]] = {}
    for prefix, values in rows.items():
        legal = [str(value) for value in legal_prefix_choices(prefix, 400)]
        if set(values) != set(legal):
            raise ValueError(f"conditional candidate set mismatch for prefix {prefix!r}")
        if any(not math.isfinite(float(p)) or float(p) < 0 for p in values.values()):
            raise ValueError("conditional probabilities must be finite and nonnegative")
        total = math.fsum(float(p) for p in values.values())
        if abs(total - 1) > 1e-6:
            raise ValueError(f"conditional probabilities do not sum to one at prefix {prefix!r}")
        normalized[prefix] = {digit: float(values[digit]) / total for digit in legal}
    distribution = _conditional_distribution(normalized)
    greedy_digits = []
    prefix = ""
    while len(greedy_digits) < 3:
        candidates = normalized[prefix]
        best = max(candidates.values())
        tied = [digit for digit, probability in candidates.items() if abs(probability - best) <= 1e-12]
        if len(tied) != 1:
            greedy = None
            break
        digit = tied[0]
        greedy_digits.append(int(digit)); prefix += digit
    else:
        greedy = int("".join(str(d) for d in greedy_digits))
    return {"greedy_quantity": greedy, "greedy_tie_prefix": prefix if greedy is None else None,
        "mode_set": distribution["mode_set"], "mode_quantity": distribution["mode_set"][0]
            if len(distribution["mode_set"]) == 1 else None,
        "median_lower": distribution["median_lower"], "expected_quantity": distribution["expected_quantity"],
        "mass_below_100": distribution["mass_below_100"], "mass": distribution["mass"],
        "quantity_probabilities": distribution["probabilities"]}


def _conditional_row(reply: Any, request: DecisionRequest) -> tuple[dict[str, float], float]:
    validate_reply(request, reply)
    result = {str(candidate.value): float(reply.probabilities[candidate.candidate_id]) for candidate in request.candidates}
    raw_total = math.fsum(result.values())
    if any(not math.isfinite(p) or p < 0 for p in result.values()) or not math.isfinite(raw_total) or abs(raw_total - 1) > 1e-6:
        raise ValueError("backend returned invalid conditional probabilities")
    return {digit: probability / raw_total for digit, probability in result.items()}, raw_total


async def _query_prefix(session: Mapping[str, Any], prefix: str, backend: JevBackend,
                        prefix_hashes: Mapping[tuple[int, str, str], str]) -> dict[str, Any]:
    request = make_request(session, prefix)
    wire = _wire(request, "first_own")
    digest = _sha(wire)
    if prefix_hashes.get((session["sequence"], "first_own", prefix)) != digest:
        raise ValueError("diagnostic actual request differs from the original frozen prefix wire")
    (reply,) = await backend.decide_many([request])
    probabilities, raw_total = _conditional_row(reply, request)
    reply_row = asdict(reply)
    usage = reply_row.get("usage")
    if not isinstance(usage, dict):
        usage = {}
    usage["raw_probability_total"] = raw_total
    return {"sequence": session["sequence"], "prefix": prefix,
        "probabilities": probabilities, "raw_probability_total": raw_total,
        "usage": usage, "reply": reply_row, "wire_sha256": digest,
        "status": "model_response"}


def _singleton_row(session: Mapping[str, Any], prefix: str,
                   prefix_hashes: Mapping[tuple[int, str, str], str]) -> dict[str, Any]:
    request = make_request(session, prefix)
    wire = _wire(request, "first_own")
    digest = _sha(wire)
    if prefix_hashes.get((session["sequence"], "first_own", prefix)) != digest:
        raise ValueError("singleton diagnostic wire differs from frozen parent wire")
    values = [str(candidate.value) for candidate in request.candidates]
    if len(values) != 1:
        raise ValueError("singleton-prefix table disagrees with legal candidates")
    return {"sequence": session["sequence"], "prefix": prefix,
        "probabilities": {values[0]: 1.0}, "raw_probability_total": 1.0,
        "usage": {"raw_probability_total": 1.0}, "reply": None, "wire_sha256": digest,
        "status": "deterministic_singleton"}


def _analyze(plan: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    trees = {}
    for session in plan["_sessions"]:
        seq = session["sequence"]
        tree_rows = {row["prefix"]: row["probabilities"] for row in rows if row["sequence"] == seq}
        trees[str(session["participant"])] = summarize_tree(tree_rows)
    # Human labels are read and joined only after both complete trees have been summarized.
    outputs = {}
    for session in plan["_sessions"]:
        alias = str(session["participant"])
        truth = plan["_truth"][f"{alias}:{session['round']}"]
        tree = trees[alias]
        decoders = {"fresh_greedy": tree["greedy_quantity"], "mode": tree["mode_quantity"],
            "median_lower": tree["median_lower"]}
        outputs[alias] = {"saved_greedy": plan["_saved_greedy"].get(alias),
            "fresh_greedy": tree["greedy_quantity"], "truth": truth,
            "mass_below_100": tree["mass_below_100"], "expected_quantity": tree["expected_quantity"],
            "mode_set": tree["mode_set"], "median_lower": tree["median_lower"],
            "greedy_tie_prefix": tree["greedy_tie_prefix"],
            "quantity_probabilities": tree["quantity_probabilities"],
            "decoder_abs_error": {name: abs(value - truth) if value is not None else None
                for name, value in decoders.items()},
            "decoder_signed_error": {name: value - truth if value is not None else None
                for name, value in decoders.items()}}
    return {"input_n": len(outputs), "tree_n": len(trees), "all_401_quantities_normalized": True,
        "no_significance_or_population_claim": True, "inputs": outputs,
        "paired_decoder_abs_error_differences": {
            "fresh_greedy_minus_mode": _paired_difference(outputs, "fresh_greedy", "mode"),
            "fresh_greedy_minus_median": _paired_difference(outputs, "fresh_greedy", "median_lower")}}


def _paired_difference(rows: Mapping[str, Mapping[str, Any]], left: str, right: str) -> list[float | None]:
    return [None if row["decoder_abs_error"][left] is None or row["decoder_abs_error"][right] is None else
        row["decoder_abs_error"][left] - row["decoder_abs_error"][right] for row in rows.values()]


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _public_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in plan.items() if not key.startswith("_")}


def _frozen_payload(plan: Mapping[str, Any]) -> dict[str, Any]:
    frozen = _public_plan(plan)
    frozen["diagnostic_code_sha256"] = _code_hashes()
    return frozen


def _verify_preview_plan(plan: Mapping[str, Any], preview_plan_path: str | Path) -> str:
    preview_path = Path(preview_plan_path).resolve()
    artifact_root = (Path(__file__).resolve().parents[2] / "artifacts" / "participant-data").resolve()
    if (preview_path.name != "frozen-diagnostic-plan.json" or not preview_path.is_file()
            or artifact_root not in preview_path.parents):
        raise ValueError("execution requires frozen-diagnostic-plan.json from an ignored preview directory")
    actual = json.loads(preview_path.read_text(encoding="utf-8"))
    if canonical_json(actual) != canonical_json(_frozen_payload(plan)):
        raise ValueError("execution plan does not exactly match the frozen preview")
    return hashlib.sha256(preview_path.read_bytes()).hexdigest()


async def execute_diagnostic(plan: dict[str, Any], output: str | Path, *, fake_client: Any = None,
                             preview_plan_path: str | Path | None = None) -> dict[str, Any]:
    _validate_protocol_snapshot(plan)
    _validate_code_snapshot(plan)
    if fake_client is not None and plan.get("schema") != "jev_procurement_decoding_synthetic_v1":
        raise ValueError("mock SDK execution accepts synthetic diagnostic plans only")
    if fake_client is None and plan.get("schema") != "jev_procurement_decoding_diagnostic_v1":
        raise PermissionError("live provider execution requires an authorized frozen parent plan")
    preview_digest = None
    if fake_client is None:
        if preview_plan_path is None:
            raise ValueError("live execution requires --preview-plan from a reviewed preview")
        preview_digest = _verify_preview_plan(plan, preview_plan_path)
    outdir = Path(output).resolve()
    artifact_root = (Path(__file__).resolve().parents[2] / "artifacts" / "participant-data").resolve()
    if outdir == artifact_root or artifact_root not in outdir.parents:
        raise ValueError("diagnostic outputs must be a new directory under ignored participant-data")
    outdir.mkdir(parents=True, exist_ok=False)
    # Freeze the inputs, code/source hashes, wire allowlist and byte forecast before credentials.
    _validate_protocol_snapshot(plan)
    _validate_code_snapshot(plan)
    frozen = _frozen_payload(plan)
    _write_json(outdir / "frozen-diagnostic-plan.json", frozen)
    budget = StudyBudget(max_requests=DIAGNOSTIC_MAX_CALLS,
        max_estimated_bytes=int(plan["diagnostic_estimated_bytes"]),
        rps=MAX_RPS, tps_bytes=MAX_TPS_BYTES, max_in_flight=MAX_IN_FLIGHT)
    sdk = fake_client
    transport = None
    if sdk is None:
        if not all(plan["source_attestations"].get(k) is True for k in
                ("codebook_verified", "intended_use_authorized", "provider_transmission_permitted")):
            raise PermissionError("source authorization gates failed before credentials")
        from capclaims.backends.jev import load_local_env
        load_local_env()
        api_key = os.environ.get("TYPESAFE_API_KEY")
        if not api_key:
            raise ValueError("TYPESAFE_API_KEY unavailable after diagnostic plan freeze")
        from typesafe_sdk import AsyncTypeSafeClient
        from typesafe_sdk._core.retry import RetryPolicy
        transport = RecordingTransport()
        sdk = AsyncTypeSafeClient(api_key=api_key, model=MODEL,
            retry=RetryPolicy(max_retries=0), transport=transport)
    else:
        api_key = "synthetic-no-key"
    shared = SharedJevClient(sdk, budget, set(plan["_allowed_hashes"]))
    backend = JevBackend(api_key=api_key, rps=MAX_RPS, tps=MAX_TPS_BYTES,
        max_requests=DIAGNOSTIC_MAX_CALLS,
        max_estimated_input_tokens=int(plan["diagnostic_estimated_bytes"]),
        max_in_flight=MAX_IN_FLIGHT, instruction_profile="procurement_first_v1",
        objective="procurement_prediction_v1", client=shared, load_environment=False)
    planned = []
    complete_rows: list[dict[str, Any]] = []
    failures = []
    singleton_rows = []
    analysis = None
    analysis_error_type = None
    for session in plan["_sessions"]:
        for prefix in build_choice_prefixes():
            if prefix in SINGLETON_PREFIXES:
                singleton_rows.append(_singleton_row(session, prefix, plan["_prefix_hashes"]))
            else:
                planned.append((session, prefix))
    try:
        for start in range(0, len(planned), MAX_IN_FLIGHT):
            wave = planned[start:start + MAX_IN_FLIGHT]
            responses = await asyncio.gather(*(_query_prefix(session, prefix, backend, plan["_prefix_hashes"])
                for session, prefix in wave), return_exceptions=True)
            for (session, prefix), response in zip(wave, responses):
                if isinstance(response, BaseException):
                    failures.append({"sequence": session["sequence"], "prefix": prefix,
                        "error_type": type(response).__name__})
                else:
                    complete_rows.append(response)
            _write_json(outdir / "partial-prefix-results.json", complete_rows + singleton_rows)
            if failures:
                break
        if not failures:
            try:
                analysis = _analyze(plan, complete_rows + singleton_rows)
            except Exception as exc:
                analysis_error_type = type(exc).__name__
    finally:
        backend._closed = True
        try:
            await shared.close_inner()
        except Exception as exc:
            analysis_error_type = analysis_error_type or type(exc).__name__
    private = {"prefix_results": complete_rows + singleton_rows, "truth": plan["_truth"],
        "analysis": analysis, "analysis_error_type": analysis_error_type, "failures": failures}
    _write_json(outdir / "private-diagnostic.json", private)
    timings = list(transport.timings) if transport is not None else []
    _write_json(outdir / "transport-timings.json", timings)
    try:
        http_overlap = maximum_overlap(timings)
        if transport is not None and not failures and len(timings) != budget.requests:
            analysis_error_type = analysis_error_type or "HTTPTimingCountMismatch"
        if transport is not None and not failures and http_overlap > MAX_IN_FLIGHT:
            analysis_error_type = analysis_error_type or "HTTPConcurrencyLimitExceeded"
    except Exception as exc:
        http_overlap = None
        analysis_error_type = analysis_error_type or type(exc).__name__
    if analysis_error_type:
        private["analysis_error_type"] = analysis_error_type
        _write_json(outdir / "private-diagnostic.json", private)
    complete = not failures and len(complete_rows) == DIAGNOSTIC_MAX_CALLS
    result = {"status": "ANALYSIS_FAILED" if analysis_error_type else
            "COMPLETE" if complete else "PARTIAL",
        "actual_requests": budget.requests, "responses": len(complete_rows),
        "singleton_constraints": len(singleton_rows), "failed_requests": len(failures),
        "unattempted_requests": max(0, DIAGNOSTIC_MAX_CALLS - len(complete_rows) - len(failures)),
        "estimated_bytes": budget.estimated_bytes, "max_adapter_in_flight": budget.max_observed_in_flight,
        "actual_http_max_overlap": http_overlap, "analysis_complete": analysis is not None,
        "analysis_error_type": analysis_error_type, "preview_plan_sha256": preview_digest}
    _write_json(outdir / "result.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Enumerate the frozen procurement elicitation tree")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--synthetic", action="store_true", help="run synthetic fixture through an injected fake SDK")
    modes.add_argument("--preview", action="store_true", help="freeze a diagnostic plan without provider setup")
    modes.add_argument("--execute", action="store_true", help="execute the authorized diagnostic")
    parser.add_argument("--parent-run", type=Path, default=Path("artifacts/participant-data/procurement-live-24"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preview-plan", type=Path,
        help="the frozen-diagnostic-plan.json produced by a reviewed --preview run; required for --execute")
    args = parser.parse_args(argv)
    if args.execute and args.preview_plan is None:
        parser.error("--execute requires --preview-plan from a reviewed --preview run")
    if (args.synthetic or args.preview) and args.preview_plan is not None:
        parser.error("--preview-plan is accepted only with --execute")
    if args.synthetic:
        plan = make_synthetic_plan()
        result = asyncio.run(execute_diagnostic(plan, args.output_dir, fake_client=_sdk_mock()))
    else:
        plan = build_diagnostic_plan(args.parent_run)
        if args.preview:
            outdir = args.output_dir.resolve()
            root = (Path(__file__).resolve().parents[2] / "artifacts" / "participant-data").resolve()
            if root not in outdir.parents:
                raise ValueError("diagnostic outputs must be under ignored participant-data")
            outdir.mkdir(parents=True, exist_ok=False)
            _validate_protocol_snapshot(plan)
            _validate_code_snapshot(plan)
            public = _frozen_payload(plan)
            _write_json(outdir / "frozen-diagnostic-plan.json", public)
            result = {"status": "FROZEN", "diagnostic_request_count": plan["diagnostic_request_count"],
                "combined_request_ceiling": plan["combined_request_ceiling"],
                "combined_estimated_bytes": plan["combined_estimated_bytes"]}
        else:
            result = asyncio.run(execute_diagnostic(plan, args.output_dir,
                preview_plan_path=args.preview_plan))
    print(json.dumps(result, indent=2))
    return 0 if result["status"] in {"COMPLETE", "FROZEN"} else 2

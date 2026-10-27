"""Frozen private-input portfolio prediction study and offline scoring."""
from __future__ import annotations

import asyncio
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import time
from typing import Any, Mapping, Sequence

from capclaims.backends.jev import MODEL, JevBackend, load_local_env
from capclaims.backends.jev_instructions import build_instruction
from capclaims.engine.protocol import DecisionReply
from capclaims.personalization import Episode, SourceManifest
from capclaims.personalization_xlsx import load_paper_episodes
from capclaims.paper.orders import PaperOrder
from capclaims.portfolio_metrics import portfolio_comparison, score_targets, training_baselines
from capclaims.portfolio_prediction import advance_trace, build_order_request, final_portfolio, next_trace_action, replay_trace
from capclaims.portfolio_transport import (DEFAULT_URL, MAX_BODY_BYTES, MAX_IN_FLIGHT,
    MAX_REQUESTS, MAX_RESERVED_BYTES, MAX_RPS, MAX_BYTES_PER_SECOND, ExpectedWire,
    GuardTransport, SharedBudget, serialize_sdk_wire, _url_tuple)
from capclaims.randomness import canonical_json

SEED = 20261024
EXPECTED_SOURCE_SHA = "f9978ebebe862e4dab51a1885a5009b7d570da1a9e49e01d2f3becfe1af385a9"
ARMS = ("portfolio_first_v1", "portfolio_third_v1")
FORMAL_TARGET_ROUNDS = (9, 10)
MAX_PORTFOLIOS = 12
PRIVATE_ROOT_PARTS = ("artifacts", "participant-data")
CODE_PATHS = (
    "src/capclaims/portfolio_prediction.py", "src/capclaims/portfolio_metrics.py",
    "src/capclaims/portfolio_transport.py", "src/capclaims/portfolio_study.py",
    "src/capclaims/backends/jev.py", "src/capclaims/backends/jev_instructions.py",
    "src/capclaims/engine/protocol.py", "src/capclaims/paper/codec.py",
    "src/capclaims/paper/interface.py", "src/capclaims/paper/observations.py",
    "src/capclaims/paper/orders.py", "src/capclaims/randomness.py",
    "src/capclaims/personalization_xlsx.py", "tools/probes/jev_portfolio_prediction.py",
    "docs/research/portfolio-prediction-design.md",
)
DESIGN_SPEC_PATH = "docs/research/portfolio-prediction-design.md"


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha(value: Any) -> str:
    return _sha_bytes(canonical_json(value).encode("utf-8"))


def _max_http_overlap(rows: Sequence[Mapping[str, Any]]) -> int:
    events = []
    for row in rows:
        if isinstance(row.get("started_at"), (int, float)) and isinstance(row.get("finished_at"), (int, float)):
            events.extend(((float(row["started_at"]), 1), (float(row["finished_at"]), -1)))
    active = peak = 0
    for _, delta in sorted(events, key=lambda item: (item[0], item[1])):
        active += delta
        peak = max(peak, active)
    return peak


def _session_base(base: Mapping[str, Any], arm: str) -> dict[str, Any]:
    """Apply only the local opaque request namespace; model-visible state is identical across views."""
    return {**base, "agent_id": base["agent_id"] + "__" + arm.split("_")[1]}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _private_output_path(path: str | Path) -> Path:
    root = _project_root()
    target = Path(path).resolve()
    private_root = (root / Path(*PRIVATE_ROOT_PARTS)).resolve()
    if target != private_root and private_root not in target.parents:
        raise ValueError("portfolio study artifacts must be stored under ignored artifacts/participant-data")
    return target


def select_participants(roster: Sequence[str], excluded_hashes: set[str], *, seed: int = SEED) -> tuple[str, ...]:
    if len(roster) != 35 or len(set(roster)) != 35 or any(not isinstance(x, str) or not x for x in roster):
        raise ValueError("GA10 roster must contain 35 unique local participant identifiers")
    eligible = [person for person in roster if hashlib.sha256(person.encode()).hexdigest() not in excluded_hashes]
    if len(eligible) < 3:
        raise ValueError("fewer than three eligible participants remain after excluding the prior pilot")
    return tuple(sorted(eligible, key=lambda person: hashlib.sha256(f"{seed}:{person}".encode()).digest())[:3])


def prior_pilot_exclusions(parent_plan: Mapping[str, Any], roster: Sequence[str]) -> set[str]:
    excluded = set(parent_plan.get("selected_participants", []))
    roster_hashes = {hashlib.sha256(person.encode()).hexdigest() for person in roster}
    if (len(excluded) != 3 or parent_plan.get("source_sha256") != EXPECTED_SOURCE_SHA
            or len(excluded & roster_hashes) != 3 or excluded - roster_hashes):
        raise ValueError("prior procurement exclusions must be exactly three known GA10 roster participants")
    return excluded


def _order_rows(action: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = action.get("orders")
    if not isinstance(rows, (list, tuple)) or len(rows) > 5:
        raise ValueError("source order portfolio is invalid")
    result = []
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {"type", "amount", "price"}:
            raise ValueError("source order must have exactly type, amount, and price")
        side = row["type"]
        if side not in {"BUY", "SELL"}:
            raise ValueError("source order side is invalid")
        if type(row["amount"]) is not int or type(row["price"]) is not int:
            raise ValueError("source order quantity and price must be integers")
        result.append({"type": side, "amount": row["amount"], "price": row["price"]})
    return result


def _episode_map(episodes: Sequence[Episode]) -> dict[tuple[str, int], Episode]:
    rows = [episode for episode in episodes if episode.scope == "GA 10" and episode.treatment == "GA"
        and episode.phase == "orders"]
    indexed = {(episode.participant_id, episode.round): episode for episode in rows}
    if len(indexed) != len(rows):
        raise ValueError("duplicate GA10 orders episode")
    return indexed


def _code_hashes() -> dict[str, str]:
    root = _project_root()
    return {path: _sha_bytes((root / path).read_bytes()) for path in CODE_PATHS}


def _history_row(person: str, formal_round: int, episode: Episode) -> dict[str, Any]:
    observed, action = episode.observed, episode.action
    if not {"p", "w", "Q", "D"}.issubset(observed):
        raise ValueError("training history is missing known pre-trading facts")
    return {"round": formal_round, "p": observed["p"], "Q": observed["Q"], "D": observed["D"],
        "orders": _order_rows(action)}


def _target_portfolio(alias: str, episode: Episode, formal_round: int) -> list[dict[str, Any]]:
    return [{"order_id": f"{alias}:r{formal_round}:slot{slot}", "owner": alias,
        "side": row["type"].lower(), "qty": row["amount"], "limit_dollars": row["price"]}
        for slot, row in enumerate(_order_rows(episode.action))]


async def _preflight_max_body(base: Mapping[str, Any], profile: str) -> int:
    from dataclasses import replace
    from capclaims.domain.types import Candidate
    request = build_order_request(base, ())
    if request is None:
        raise ValueError("portfolio side request unexpectedly missing")
    # Conservative superset of live state: the exact frozen history plus five
    # maximum-width pending rows, without padding or truncating live calls.
    observation = json.loads(canonical_json(request.observation))
    maximal_order = {"slot": 4, "side": "sell", "qty": 999, "limit_dollars": 999}
    observation["partial_orders"] = [dict(maximal_order, slot=i) for i in range(5)]
    # Keep the exact frozen histories, while including every fixed-width numeric
    # optional field and the longest legal final-digit criteria labels.
    observation.update({"numeric_prefix": "99", "choice_minimum": 1, "choice_maximum": 999,
        "decimal_place": "u", "decimal_place_name": "ones", "numeric_value_name": "limit price",
        "numeric_units": "dollars per unit", "decision_field": "price", "own_side": "sell",
        "own_qty_units": 999, "slot": 4, "partial_order_fields": "slot,side,qty units,limit $/unit"})
    candidates = tuple(Candidate(f"digit{i}", i, f"{i} {('900..999' if i == 9 else '1..999')} dollars per unit")
        for i in range(10))
    maximal_request = replace(request, stage="paper:slot4:price:u", observation=observation,
        candidates=candidates)
    wire = await serialize_sdk_wire(maximal_request, profile, observation)
    return len(wire.body)


def _save_preview(output: Path, manifest: dict[str, Any], inputs: dict[str, Any], targets: dict[str, Any]) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    input_bytes = canonical_json(inputs).encode("utf-8")
    target_bytes = canonical_json(targets).encode("utf-8")
    manifest["inputs_sha256"] = _sha_bytes(input_bytes)
    manifest["scoring_targets_sha256"] = _sha_bytes(target_bytes)
    _write_json(output / "inputs.json", inputs)
    _write_json(output / "scoring-targets.json", targets)
    _write_json(output / "frozen-plan.json", manifest)
    return manifest


def build_synthetic_preview(output_dir: str | Path) -> dict[str, Any]:
    """Create a wholly synthetic preview; it never reads a human source or env file."""
    people = [f"synthetic-person-{i:02d}" for i in range(3)]
    inputs, targets, sessions, baseline_training = {}, {}, [], {}
    for person_i, person in enumerate(people):
        alias = f"synthetic-{person_i}"
        history = [{"round": r, "p": 60 + r, "Q": 20 + r, "D": 30 + r,
            "orders": [{"type": "BUY", "amount": 1, "price": 50}]} for r in (6, 7, 8)]
        baseline_training[alias] = [{"round": r, "orders": [{"type": "BUY", "amount": 1, "price": 50}]}
            for r in range(1, 9)]
        for round_no in FORMAL_TARGET_ROUNDS:
            key = f"{alias}:r{round_no}"
            base = {"agent_id": alias, "round": round_no, "treatment": "GA",
                "wholesale_price_dollars": 10, "p": 70, "Q": 50, "D": 40, "history": history}
            inputs[key] = base
            targets[key] = {"orders": []}
            for arm in ARMS:
                sessions.append({"target_id": key, "arm": arm})
    maximum = max(asyncio.run(_preflight_max_body(base, arm))
        for base in inputs.values() for arm in ARMS)
    if maximum > MAX_BODY_BYTES:
        raise ValueError("synthetic maximal request envelope exceeds 7500 bytes")
    manifest = _manifest_base(source_kind="synthetic", source_sha256="0" * 64,
        source_attestations={"codebook_verified": True, "intended_use_authorized": True,
            "provider_transmission_permitted": True}, sessions=sessions, max_body=maximum,
        parent_plan_sha256=None, input_ids=sorted(inputs))
    output = _private_output_path(output_dir)
    return _save_preview(output, manifest, {"bases": inputs, "baseline_training": baseline_training}, targets)


def _manifest_base(*, source_kind: str, source_sha256: str, source_attestations: Mapping[str, bool],
                   sessions: Sequence[Mapping[str, Any]], parent_plan_sha256: str | None,
                   input_ids: Sequence[str], selected_participant_hashes: Sequence[str] = (),
                   max_body: int = MAX_BODY_BYTES) -> dict[str, Any]:
    from capclaims.backends.jev_instructions import build_instruction
    # Prompt snapshots are fixed here and body preflight runs before any SDK/API credential setup.
    prompts = {arm: build_instruction(_prompt_probe_request(), arm) for arm in ARMS}
    return {"schema": "jev_portfolio_prediction_v1", "model": MODEL,
        "sdk_version": "typesafe-sdk==0.7.5", "base_url": DEFAULT_URL,
        "source_kind": source_kind, "source_sha256": source_sha256,
        "source_attestations": dict(source_attestations), "parent_plan_sha256": parent_plan_sha256,
        "selected_participant_hashes": list(selected_participant_hashes), "seed": SEED,
        "formal_target_rounds": list(FORMAL_TARGET_ROUNDS), "training_round_max": 8,
        "game_round_offset": 5, "history_latest_n": 3, "market_size": 35,
        "arms": list(ARMS), "prompt_text": prompts,
        "prompt_sha256": {arm: _sha_bytes(text.encode()) for arm, text in prompts.items()},
        "protocol_sha256": _protocol_hash(), "code_sha256": _code_hashes(),
        "design_spec_path": DESIGN_SPEC_PATH,
        "design_spec_text": (_project_root() / DESIGN_SPEC_PATH).read_text(encoding="utf-8"),
        "design_spec_sha256": _sha_bytes((_project_root() / DESIGN_SPEC_PATH).read_bytes()),
        "input_ids": list(input_ids), "sessions": [dict(row) for row in sessions],
        "limits": {"portfolios": MAX_PORTFOLIOS, "requests": MAX_REQUESTS,
            "bytes_per_request": MAX_BODY_BYTES, "reserved_bytes": MAX_RESERVED_BYTES,
            "rps": MAX_RPS, "bytes_per_second": MAX_BYTES_PER_SECOND,
            "max_in_flight": MAX_IN_FLIGHT, "sdk_retries": 0, "compression": False},
        "preflight_max_body_bytes": max_body}


def _prompt_probe_request():
    from capclaims.domain.types import Candidate
    from capclaims.engine.protocol import DecisionRequest
    return DecisionRequest("opaque", 13, "paper:slot0:side",
        {"phase": "orders", "decision_field": "side"},
        (Candidate("buy", "buy", "Buy"), Candidate("stop", "STOP", "Stop"),
         Candidate("sell", "sell", "Sell")))


def _protocol_hash() -> str:
    protocol = {"base_fields": ["agent_id", "round", "treatment", "wholesale_price_dollars", "p", "Q", "D", "history"],
        "trace": "slot-side-qty-3digits-price-3digits;singleton-forced;unresolved-tie-fails-whole-portfolio",
        "orders": "maximum5;qty-price1..999;gross-sells<=initial-Q;buy-does-not-replenish;mixed-and-duplicates-preserved",
        "scoring": "separate-buy-sell-cumulative-curves;999-prices;incomplete-upper-bound",
        "wire": "SDK0.7.5;POST;https://api.typesafe.ai/v1/systemone;compression=false;retry=0"}
    return _sha(protocol)


async def build_preview(source: SourceManifest, episodes: Sequence[Episode], parent_run: str | Path,
                        output_dir: str | Path, *, source_policy_sha256: str | None = None) -> dict[str, Any]:
    if source.source_kind != "human" or source.source_sha256 != EXPECTED_SOURCE_SHA:
        raise ValueError("portfolio preview requires the verified paper workbook checksum")
    if not (source.codebook_verified and source.intended_use_authorized and source.provider_transmission_permitted):
        raise PermissionError("source and provider attestations must be true before human contexts are prepared")
    parent_plan_path = Path(parent_run).resolve()
    parent_plan = json.loads(parent_plan_path.read_text(encoding="utf-8"))
    if parent_plan.get("source_sha256") != EXPECTED_SOURCE_SHA:
        raise ValueError("prior procurement pilot plan does not bind the verified source")
    roster = sorted({e.participant_id for e in episodes if e.scope == "GA 10" and e.treatment == "GA"})
    old_selected = prior_pilot_exclusions(parent_plan, roster)
    selected = select_participants(roster, old_selected)
    episodes_by_person_round = _episode_map(episodes)
    inputs, targets, sessions, baseline_training = {}, {}, [], {}
    for index, person in enumerate(selected):
        alias = f"person-{index}-{hashlib.sha256(person.encode()).hexdigest()[:10]}"
        history_episodes = [episodes_by_person_round.get((person, round_no)) for round_no in range(1, 9)]
        if any(row is None for row in history_episodes):
            raise ValueError("selected GA participant lacks one or more formal training order rounds 1..8")
        baseline_training[alias] = [{"round": row.round,
            "orders": _order_rows(row.action)} for row in history_episodes]
        for target_round in FORMAL_TARGET_ROUNDS:
            target_episode = episodes_by_person_round.get((person, target_round))
            if target_episode is None:
                raise ValueError("selected GA participant lacks a held-out target round")
            observed = target_episode.observed
            history = [_history_row(person, row.round, row) for row in history_episodes[-3:]]
            key = f"{alias}:r{target_round}"
            inputs[key] = {"agent_id": alias, "round": target_round, "treatment": "GA",
                "wholesale_price_dollars": observed["w"], "p": observed["p"], "Q": observed["Q"],
                "D": observed["D"], "history": history}
            targets[key] = {"orders": _target_portfolio(alias, target_episode, target_round)}
            from capclaims.paper.orders import validate_portfolio
            human_orders = tuple(PaperOrder(**row) for row in targets[key]["orders"])
            validate_portfolio(human_orders, alias, observed["Q"])
            for arm in ARMS:
                sessions.append({"target_id": key, "arm": arm})
    maximum = 0
    for target_id, base in inputs.items():
        for arm in ARMS:
            maximum = max(maximum, await _preflight_max_body(base, arm))
    if maximum > MAX_BODY_BYTES:
        raise ValueError("maximal portfolio request envelope exceeds 7500 bytes")
    manifest = _manifest_base(source_kind=source.source_kind, source_sha256=source.source_sha256,
        source_attestations={"codebook_verified": source.codebook_verified,
            "intended_use_authorized": source.intended_use_authorized,
            "provider_transmission_permitted": source.provider_transmission_permitted},
        sessions=sessions, parent_plan_sha256=_sha_bytes(parent_plan_path.read_bytes()),
        input_ids=sorted(inputs), selected_participant_hashes=[hashlib.sha256(p.encode()).hexdigest() for p in selected],
        max_body=maximum)
    manifest["source_policy_sha256"] = source_policy_sha256
    output = _private_output_path(output_dir)
    return _save_preview(output, manifest, {"bases": inputs, "baseline_training": baseline_training}, targets)


def verify_preview(preview_dir: str | Path, reviewed_preview_sha256: str | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    path = Path(preview_dir).resolve()
    manifest_bytes = (path / "frozen-plan.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    if reviewed_preview_sha256 is not None and _sha_bytes(manifest_bytes) != reviewed_preview_sha256:
        raise ValueError("reviewed frozen preview hash does not match")
    design_bytes = (_project_root() / DESIGN_SPEC_PATH).read_bytes()
    if (manifest.get("protocol_sha256") != _protocol_hash() or manifest.get("code_sha256") != _code_hashes()
            or manifest.get("design_spec_path") != DESIGN_SPEC_PATH
            or manifest.get("design_spec_sha256") != _sha_bytes(design_bytes)
            or manifest.get("design_spec_text") != design_bytes.decode("utf-8")):
        raise ValueError("portfolio protocol or code changed after preview")
    inputs_bytes = (path / "inputs.json").read_bytes()
    if _sha_bytes(inputs_bytes) != manifest.get("inputs_sha256"):
        raise ValueError("frozen portfolio inputs were changed")
    if manifest.get("preflight_max_body_bytes", MAX_BODY_BYTES + 1) > MAX_BODY_BYTES:
        raise ValueError("frozen preview exceeds the request body bound")
    inputs = json.loads(inputs_bytes)
    return manifest, inputs


class _HashTranscript:
    def __init__(self, path: Path):
        self.path = path
        self.previous = "0" * 64
        self.events: list[dict[str, Any]] = []

    def append(self, event: Mapping[str, Any]) -> str:
        body = {"previous_sha256": self.previous, "event": dict(event)}
        digest = _sha(body)
        row = {**body, "sha256": digest}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(canonical_json(row) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.events.append(row)
        self.previous = digest
        return digest


def _failure_stage(guard: GuardTransport, request_id: str,
                   *, reply_received: bool = False) -> str:
    ledger_path = guard.budget.ledger_path
    rows = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines() if line]
    if not any(row.get("request_id") == request_id for row in rows):
        return "pre_send"
    request_path = ledger_path.with_name("transport-requests.jsonl")
    request_ids = {json.loads(line).get("request_id") for line in request_path.read_text(encoding="utf-8").splitlines()} if request_path.exists() else set()
    timing_ids = {row.get("request_id") for row in guard.http_timings}
    if request_id not in request_ids or request_id not in timing_ids:
        return "charged_no_dispatch"
    body_path = ledger_path.with_name("transport-bodies.jsonl")
    response_rows = [json.loads(line) for line in body_path.read_text(encoding="utf-8").splitlines()
        if line] if body_path.exists() else []
    response = next((row for row in response_rows if row.get("request_id") == request_id), None)
    timing = next((row for row in guard.http_timings if row.get("request_id") == request_id), {})
    if response is None or not isinstance(response.get("status"), int) or response["status"] >= 300:
        return "charged_transport"
    return "transition_failure" if reply_received else "reply_validation"


def _failure_event(base: Mapping[str, Any], trace: tuple[dict[str, Any], ...],
                   request: Any, session: Mapping[str, Any], profile: str,
                   wire: ExpectedWire | None, failure_stage: str, error: BaseException,
                   transcript: _HashTranscript, reply: Any = None) -> tuple[dict[str, Any], ...]:
    proposed = advance_trace(base, trace, None)
    row = {"target_id": session["target_id"], "arm": profile, "kind": "failure",
        "failure_stage": failure_stage, "error_type": type(error).__name__,
        "expected_wire_sha256": wire.body_sha256 if wire else None,
        "expected_body_bytes": len(wire.body) if wire else None,
        "request": asdict(request), "event": proposed[-1]}
    if reply is not None:
        row["reply"] = asdict(reply)
    transcript.append(row)
    return proposed


async def _run_session(session: Mapping[str, Any], base: Mapping[str, Any], backend: JevBackend,
                       profile: str, guard: GuardTransport, transcript: _HashTranscript) -> dict[str, Any]:
    trace: tuple[dict[str, Any], ...] = ()
    calls = 0
    while True:
        action = next_trace_action(base, trace)
        if action["kind"] == "complete":
            result = replay_trace(base, trace)
            return {**dict(session), "status": "complete", "orders": [asdict(o) for o in result["orders"]],
                "call_count": calls, "trace_head": transcript.previous}
        if action["kind"] == "incomplete":
            return {**dict(session), "status": "incomplete", "reason": action.get("reason"),
                "call_count": calls, "trace_head": transcript.previous}
        if action["kind"] == "forced":
            proposed = advance_trace(base, trace, None)
            transcript.append({"target_id": session["target_id"], "arm": profile,
                "kind": "forced", "event": proposed[-1]})
            trace = proposed
            continue
        request = build_order_request(base, trace)
        if request is None:
            raise AssertionError("portfolio model request unexpectedly absent")
        wire = None
        try:
            wire = await serialize_sdk_wire(request, profile)
            guard.register(wire)
        except Exception as exc:
            calls += 1
            try:
                trace = _failure_event(base, trace, request, session, profile, wire,
                    "pre_send", exc, transcript)
            except Exception:
                raise
            return {**dict(session), "status": "incomplete", "reason": "provider_or_validation_failure",
                "failure_stage": "pre_send", "call_count": calls, "trace_head": transcript.previous}
        calls += 1
        reply = None
        try:
            (reply,) = await backend.decide_many([request])
        except Exception as exc:
            failure_stage = _failure_stage(guard, request.request_id)
            trace = _failure_event(base, trace, request, session, profile, wire,
                failure_stage, exc, transcript)
            return {**dict(session), "status": "incomplete", "reason": "provider_or_validation_failure",
                "failure_stage": failure_stage, "call_count": calls, "trace_head": transcript.previous}
        try:
            proposed = advance_trace(base, trace, reply)
        except Exception as exc:
            trace = _failure_event(base, trace, request, session, profile, wire,
                "transition_failure", exc, transcript, reply=reply)
            return {**dict(session), "status": "incomplete", "reason": "provider_or_validation_failure",
                "failure_stage": "transition_failure", "call_count": calls, "trace_head": transcript.previous}
        try:
            transcript.append({"target_id": session["target_id"], "arm": profile,
                "kind": "model", "request": asdict(request), "expected_wire_sha256": wire.body_sha256,
                "reply": asdict(reply), "event": proposed[-1]})
        except Exception as exc:
            trace = _failure_event(base, trace, request, session, profile, wire,
                "recording_failure", exc, transcript, reply=reply)
            return {**dict(session), "status": "incomplete", "reason": "provider_or_validation_failure",
                "failure_stage": "recording_failure", "call_count": calls, "trace_head": transcript.previous}
        trace = proposed


async def execute_preview(preview_dir: str | Path, run_dir: str | Path, *,
                          reviewed_preview_sha256: str | None = None, fake: bool = False,
                          fake_transport: Any = None) -> dict[str, Any]:
    manifest, inputs = verify_preview(preview_dir, reviewed_preview_sha256)
    if not fake and not reviewed_preview_sha256:
        raise ValueError("real execution requires an explicit reviewed frozen-preview SHA256")
    output = _private_output_path(run_dir)
    output.mkdir(parents=True, exist_ok=False)
    (output / "attempts.jsonl").touch(exist_ok=False)
    if not all(manifest.get("source_attestations", {}).values()):
        return {"status": "BLOCKED", "reason": "source attestations are false", "attempted_n": 0}
    if not fake and manifest.get("source_kind") != "human":
        return {"status": "BLOCKED", "reason": "live execution requires authorized human source", "attempted_n": 0}
    if not fake and (manifest.get("source_sha256") != EXPECTED_SOURCE_SHA
            or not manifest.get("source_policy_sha256")):
        return {"status": "BLOCKED", "reason": "human source or source-policy checksum is not frozen", "attempted_n": 0}
    # Full validation and output freeze precede environment access or SDK construction.
    execution_mode = "mock" if fake else "provider"
    _write_json(output / "run-manifest.json", {"preview_sha256": _sha_bytes(
        (Path(preview_dir) / "frozen-plan.json").read_bytes()), "execution_mode": execution_mode,
        "status": "RUNNING"})
    budget = SharedBudget(output / "attempts.jsonl",
        rps=10_000 if fake else MAX_RPS,
        bytes_per_second=100_000_000 if fake else MAX_BYTES_PER_SECOND)
    if fake:
        transport_inner = fake_transport or _mock_http_transport()
    else:
        load_local_env()
        if not os.getenv("TYPESAFE_API_KEY"):
            raise ValueError("TYPESAFE_API_KEY is unavailable after all preview/source gates")
        transport_inner = __import__("httpx2").AsyncHTTPTransport(retries=0)
    guard = GuardTransport(transport_inner, DEFAULT_URL, budget, max_body_bytes=MAX_BODY_BYTES)
    httpx2 = __import__("httpx2")
    from typesafe_sdk import AsyncTypeSafeClient
    from typesafe_sdk._core.retry import RetryPolicy
    api_key = "offline-dummy-key" if fake else os.environ["TYPESAFE_API_KEY"]
    http = httpx2.AsyncClient(transport=guard, follow_redirects=False)
    client = AsyncTypeSafeClient(api_key=api_key, model=MODEL, retry=RetryPolicy(max_retries=0),
        timeout=30, http_client=http, base_url="https://api.typesafe.ai", compression=False)
    backends = {arm: JevBackend(api_key=api_key, rps=1000, tps=10_000_000,
        max_requests=MAX_REQUESTS, max_estimated_input_tokens=10_000_000,
        instruction_profile=arm, objective="portfolio_prediction_v1", request_timeout=30,
        max_in_flight=MAX_IN_FLIGHT, client=client, load_environment=False) for arm in ARMS}
    transcript = _HashTranscript(output / "trace.jsonl")
    statuses: list[dict[str, Any]] = []
    sessions = list(manifest["sessions"])
    # Keep target truth absent from this process: only base input data is loaded above.
    try:
        for wave_start in range(0, len(sessions), MAX_IN_FLIGHT):
            wave = sessions[wave_start:wave_start + MAX_IN_FLIGHT]
            results = await asyncio.gather(*(
                _run_session(row, _session_base(inputs["bases"][row["target_id"]], row["arm"]),
                    backends[row["arm"]],
                    row["arm"], guard, transcript) for row in wave), return_exceptions=True)
            for row, result in zip(wave, results):
                if isinstance(result, BaseException):
                    prior = [item for item in transcript.events if item["event"].get("target_id") == row["target_id"]
                        and item["event"].get("arm") == row["arm"]]
                    statuses.append({**row, "status": "error", "error_type": type(result).__name__,
                        "trace_head": prior[-1]["sha256"] if prior else None,
                        "call_count": sum(item["event"].get("kind") in {"model", "failure"} for item in prior)})
                else:
                    statuses.append(result)
            _write_json(output / "partial-statuses.json", statuses)
            if any(row["status"] == "error" or row.get("reason") in {
                    "provider_or_validation_failure", "provider_failure", "invalid_reply"}
                    for row in statuses[-len(wave):]):
                break
    finally:
        for backend in backends.values():
            backend._closed = True
        await client.aclose()
    attempted_n = len(statuses)
    attempted_keys = {(row["target_id"], row["arm"]) for row in statuses}
    for row in sessions:
        if (row["target_id"], row["arm"]) not in attempted_keys:
            statuses.append({**row, "status": "unattempted"})
    bodies_path = output / "transport-bodies.jsonl"
    body_payload = bodies_path.read_bytes() if bodies_path.exists() else b""
    requests_path = output / "transport-requests.jsonl"
    request_payload = requests_path.read_bytes() if requests_path.exists() else b""
    timings_path = output / "transport-timings.json"
    _write_json(timings_path, guard.http_timings)
    timings_payload = timings_path.read_bytes()
    request_rows = [json.loads(line) for line in request_payload.decode("utf-8").splitlines() if line]
    actual_body_bytes = sum(row["body_bytes"] for row in request_rows)
    terminal = {"schema": "jev_portfolio_terminal_v1", "execution_mode": execution_mode,
        "preview_sha256": _sha_bytes(
            (Path(preview_dir) / "frozen-plan.json").read_bytes()),
        "statuses": statuses,
        "trace_chain_head": transcript.previous, "trace_event_n": len(transcript.events),
        "transport_bodies_sha256": _sha_bytes(body_payload),
        "transport_body_line_n": len(body_payload.splitlines()),
        "transport_requests_sha256": _sha_bytes(request_payload),
        "transport_request_line_n": len(request_payload.splitlines()),
        "transport_timings_sha256": _sha_bytes(timings_payload),
        "transport_timing_n": len(guard.http_timings),
        "attempts_sha256": _sha_bytes((output / "attempts.jsonl").read_bytes()),
        "actual_request_body_bytes": actual_body_bytes,
        "ledger_line_n": budget.usage["requests_charged"], "budget": budget.usage}
    overall_status = "COMPLETE" if len(statuses) == len(sessions) and all(
        row["status"] == "complete" for row in statuses) else "PARTIAL"
    _write_json(output / "run-manifest.json", {"preview_sha256": terminal["preview_sha256"],
        "execution_mode": execution_mode, "status": overall_status})
    terminal["run_manifest_sha256"] = _sha_bytes((output / "run-manifest.json").read_bytes())
    _write_json(output / "terminal.json", terminal)
    _write_json(output / "result.json", {"execution_mode": execution_mode,
        "status": overall_status,
        "planned_portfolios": len(sessions), "attempted_portfolios": attempted_n,
        "unattempted_portfolios": len(sessions) - attempted_n,
        "incomplete_or_failed_portfolios": sum(row["status"] != "complete" for row in statuses),
        "actual_http_request_n": len(guard.http_timings),
        "actual_http_max_overlap": _max_http_overlap(guard.http_timings),
        "actual_request_body_bytes": actual_body_bytes,
        "failure_types": sorted({row.get("error_type", row.get("reason", ""))
            for row in statuses if row["status"] != "complete"}),
        "terminal_sha256": _sha(terminal), "budget": budget.usage})
    return json.loads((output / "result.json").read_text(encoding="utf-8"))


def _mock_http_transport():
    import httpx2
    async def handler(request: Any):
        body = json.loads((await request.aread()).decode("utf-8"))
        criteria = body["questions"]["decision"]["criteria"]
        chosen = next(iter(criteria))
        return httpx2.Response(200, json={"model": MODEL,
            "usage": {"input_tokens": 1, "output_tokens": 1},
            "answers": {"decision": {"type": "choice", "choice": chosen,
                "confidence": 1.0, "probabilities": {key: float(key == chosen) for key in criteria}}}})
    return httpx2.MockTransport(handler)


def verify_trace(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir)
    terminal = json.loads((root / "terminal.json").read_text(encoding="utf-8"))
    run_manifest_bytes = (root / "run-manifest.json").read_bytes()
    run_manifest = json.loads(run_manifest_bytes)
    rows = [json.loads(line) for line in (root / "trace.jsonl").read_text(encoding="utf-8").splitlines() if line]
    previous = "0" * 64
    for row in rows:
        if row.get("previous_sha256") != previous or row.get("sha256") != _sha(
                {"previous_sha256": previous, "event": row.get("event")}):
            raise ValueError("portfolio trace chain is invalid or tampered")
        previous = row["sha256"]
    ledger_n = len([line for line in (root / "attempts.jsonl").read_text(encoding="utf-8").splitlines() if line])
    bodies_path = root / "transport-bodies.jsonl"
    body_payload = bodies_path.read_bytes() if bodies_path.exists() else b""
    body_lines = len(body_payload.splitlines())
    requests_path = root / "transport-requests.jsonl"
    request_payload = requests_path.read_bytes() if requests_path.exists() else b""
    request_lines = len(request_payload.splitlines())
    timings_payload = (root / "transport-timings.json").read_bytes()
    timings = json.loads(timings_payload)
    attempts_payload = (root / "attempts.jsonl").read_bytes()
    ledger_rows = [json.loads(line) for line in attempts_payload.splitlines() if line]
    request_rows = [json.loads(line) for line in request_payload.splitlines() if line]
    body_rows = [json.loads(line) for line in body_payload.splitlines() if line]
    ids = [row.get("request_id") for row in ledger_rows]
    request_ids = [row.get("request_id") for row in request_rows]
    timing_ids = [row.get("request_id") for row in timings]
    body_ids = [row.get("request_id") for row in body_rows]
    if (terminal.get("schema") != "jev_portfolio_terminal_v1"
            or terminal.get("execution_mode") not in {"mock", "provider"}
            or run_manifest.get("execution_mode") != terminal.get("execution_mode")
            or run_manifest.get("preview_sha256") != terminal.get("preview_sha256")
            or run_manifest.get("run_manifest_sha256") is not None
            or terminal.get("run_manifest_sha256") != _sha_bytes(run_manifest_bytes)
            or terminal.get("trace_chain_head") != previous or terminal.get("trace_event_n") != len(rows)
            or terminal.get("ledger_line_n") != ledger_n or len(terminal.get("statuses", [])) != MAX_PORTFOLIOS
            or terminal.get("transport_bodies_sha256") != _sha_bytes(body_payload)
            or terminal.get("transport_body_line_n") != body_lines
            or terminal.get("transport_requests_sha256") != _sha_bytes(request_payload)
            or terminal.get("transport_request_line_n") != request_lines
            or terminal.get("transport_timings_sha256") != _sha_bytes(timings_payload)
            or terminal.get("transport_timing_n") != len(timings)
            or terminal.get("attempts_sha256") != _sha_bytes(attempts_payload)
            or request_lines > ledger_n or len(timings) > request_lines
            or len(set(ids)) != ledger_n or not set(request_ids).issubset(set(ids))
            or not set(timing_ids).issubset(set(request_ids)) or len(set(body_ids)) != body_lines
            or not set(body_ids).issubset(set(timing_ids))):
        raise ValueError("portfolio terminal manifest does not bind complete evidence")
    ledger_by_id = {row["request_id"]: row for row in ledger_rows}
    request_by_id = {row["request_id"]: row for row in request_rows}
    timing_by_id = {row["request_id"]: row for row in timings}
    body_by_id = {row["request_id"]: row for row in body_rows}
    for expected_seq, (request_id, ledger) in enumerate(ledger_by_id.items(), 1):
        if (ledger.get("seq") != expected_seq or not isinstance(request_id, str) or not request_id
                or ledger.get("method") != "POST" or _url_tuple(ledger.get("url", "")) != _url_tuple(DEFAULT_URL)
                or ledger.get("expected_sha256") != ledger.get("actual_sha256")
                or type(ledger.get("body_bytes")) is not int or ledger["body_bytes"] > MAX_BODY_BYTES):
            raise ValueError("attempt ledger, sent request body, and HTTP timing do not agree")
        body = request_by_id.get(request_id)
        timing = timing_by_id.get(request_id)
        if body is not None and (ledger.get("expected_sha256") != body.get("body_sha256")
                or ledger.get("actual_sha256") != body.get("body_sha256")
                or ledger.get("body_bytes") != body.get("body_bytes")
                or hashlib.sha256(body.get("body_utf8", "").encode("utf-8")).hexdigest() != body.get("body_sha256")):
            raise ValueError("attempt ledger differs from the preserved sent request body")
        if timing is not None and request_id not in request_by_id:
            raise ValueError("HTTP timing exists without a persisted request body")
        if request_id in body_by_id:
            response = body_by_id[request_id]
            if (body is None or timing is None or response.get("request_body_sha256") != body.get("body_sha256")
                    or response.get("request_body_utf8") != body.get("body_utf8")
                    or response.get("status") != timing.get("status_code", timing.get("status"))):
                raise ValueError("response record does not match its sent request and HTTP status")
            response_text = response.get("response_body_utf8")
            if response_text is not None and hashlib.sha256(response_text.encode("utf-8")).hexdigest() != response.get("response_body_sha256"):
                raise ValueError("preserved HTTP response body differs from its recorded digest")
            if 200 <= response.get("status", 0) < 300 and response_text is None:
                raise ValueError("successful HTTP response is missing its private response body")
    return {"trace_event_n": len(rows), "trace_chain_head": previous, "ledger_line_n": ledger_n,
        "status_n": len(terminal["statuses"]), "transport_body_line_n": body_lines,
        "transport_request_line_n": request_lines, "attempts_sha256": terminal["attempts_sha256"],
        "ledger_by_id": ledger_by_id, "request_by_id": request_by_id,
        "timing_by_id": timing_by_id, "body_by_id": body_by_id,
        "trace_rows": rows}


def replay_run(preview_dir: str | Path, run_dir: str | Path) -> dict[str, Any]:
    manifest, inputs = verify_preview(preview_dir)
    audit = verify_trace(run_dir)
    terminal = json.loads((Path(run_dir) / "terminal.json").read_text(encoding="utf-8"))
    if terminal.get("preview_sha256") != _sha_bytes((Path(preview_dir) / "frozen-plan.json").read_bytes()):
        raise ValueError("terminal run record does not bind the frozen preview")
    expected_statuses = [(s["target_id"], s["arm"]) for s in manifest["sessions"]]
    actual_statuses = [(s.get("target_id"), s.get("arm")) for s in terminal["statuses"]]
    if actual_statuses != expected_statuses:
        raise ValueError("terminal run record omits, duplicates, or reorders planned session statuses")
    expected_run_status = ("COMPLETE" if all(row.get("status") == "complete" for row in terminal["statuses"])
        else "PARTIAL")
    run_manifest = json.loads((Path(run_dir) / "run-manifest.json").read_text(encoding="utf-8"))
    if run_manifest.get("status") != expected_run_status:
        raise ValueError("run manifest status differs from terminal session statuses")
    # Every session's ordered trace events and every model request reconstruct
    # against immutable base facts, then join to its one-shot HTTP evidence.
    chain_rows = [json.loads(line) for line in (Path(run_dir) / "trace.jsonl").read_text(encoding="utf-8").splitlines() if line]
    rows = [row["event"] for row in chain_rows]
    sessions_by_key = {(row["target_id"], row["arm"]): row for row in terminal["statuses"]}
    last_head_by_session = {}
    for chain_row in chain_rows:
        event = chain_row["event"]
        last_head_by_session[(event["target_id"], event["arm"])] = chain_row["sha256"]
    linked_request_ids: set[str] = set()
    result = {}
    for session in manifest["sessions"]:
        key = (session["target_id"], session["arm"])
        session_events = [row for row in rows if row.get("target_id") == key[0] and row.get("arm") == key[1]]
        base = _session_base(inputs["bases"][key[0]], key[1])
        trace_rows: list[dict[str, Any]] = []
        cursor = 0
        model_call_n = 0
        while cursor < len(session_events):
            row = session_events[cursor]
            action = next_trace_action(base, trace_rows)
            if action["kind"] == "forced":
                if row.get("kind") != "forced":
                    raise ValueError("replay trace omitted a forced singleton constraint")
                advanced = advance_trace(base, trace_rows, None)
                if advanced[-1] != row["event"]:
                    raise ValueError("recorded forced singleton differs from reconstructed constraint")
                trace_rows.append(advanced[-1])
            elif action["kind"] == "request":
                request = build_order_request(base, trace_rows)
                if request is None or canonical_json(row.get("request")) != canonical_json(asdict(request)):
                    raise ValueError("recorded request differs from the independently reconstructed state")
                request_id = request.request_id
                model_call_n += 1
                ledger = audit["ledger_by_id"].get(request_id)
                sent = audit["request_by_id"].get(request_id)
                timing = audit["timing_by_id"].get(request_id)
                expected_wire = asyncio.run(serialize_sdk_wire(request, key[1]))
                response_record = audit["body_by_id"].get(request_id)
                if row.get("kind") == "model":
                    if ledger is None or sent is None or timing is None:
                        raise ValueError("model choice has no unique charged HTTP attempt")
                    linked_request_ids.add(request_id)
                    _compare_wire_evidence(request_id, expected_wire, ledger, sent)
                    reply = DecisionReply(**row["reply"])
                    actual_reply, error_type = asyncio.run(_replay_backend(request, key[1],
                        response_record, timing))
                    if error_type is not None or actual_reply is None or not _same_reply(actual_reply, reply):
                        raise ValueError("installed Jev backend does not accept the recorded successful reply")
                    advanced = advance_trace(base, trace_rows, reply)
                    if advanced[-1] != row["event"]:
                        raise ValueError("recorded model choice differs from reply validation")
                elif row.get("kind") == "failure":
                    failure_stage = row.get("failure_stage")
                    if failure_stage not in {"pre_send", "charged_no_dispatch", "charged_transport",
                            "reply_validation", "transition_failure", "recording_failure"}:
                        raise ValueError("failure event lacks a recognized transport failure stage")
                    if (row.get("expected_wire_sha256") not in (None, expected_wire.body_sha256)
                            or row.get("expected_body_bytes") not in (None, len(expected_wire.body))):
                        raise ValueError("pre-send failure differs from reconstructed SDK wire")
                    if failure_stage == "pre_send":
                        if any(value is not None for value in (ledger, sent, timing, response_record)):
                            raise ValueError("zero-charge pre-send failure has network evidence")
                    else:
                        if ledger is None:
                            raise ValueError("charged failure lacks its persistent attempt ledger row")
                        if request_id in linked_request_ids:
                            raise ValueError("one charged attempt was linked to multiple trace decisions")
                        linked_request_ids.add(request_id)
                        if (ledger.get("expected_sha256") != expected_wire.body_sha256
                                or ledger.get("actual_sha256") != expected_wire.body_sha256
                                or ledger.get("body_bytes") != len(expected_wire.body)):
                            raise ValueError("charged failure ledger differs from reconstructed SDK wire")
                    if failure_stage == "charged_no_dispatch":
                        if timing is not None or response_record is not None:
                            raise ValueError("charged no-dispatch failure has HTTP dispatch evidence")
                        if sent is not None:
                            _compare_wire_evidence(request_id, expected_wire, ledger, sent)
                    elif failure_stage == "charged_transport":
                        if sent is None or timing is None:
                            raise ValueError("charged transport failure lacks request or timing evidence")
                        _compare_wire_evidence(request_id, expected_wire, ledger, sent)
                        _validate_transport_failure(request, response_record, timing)
                    elif failure_stage == "reply_validation":
                        if sent is None or timing is None or response_record is None:
                            raise ValueError("reply-validation failure lacks its response evidence")
                        _compare_wire_evidence(request_id, expected_wire, ledger, sent)
                        error_type = asyncio.run(_replay_backend_failure(request, key[1], response_record, timing))
                        if error_type != row.get("error_type"):
                            raise ValueError("recorded reply-validation failure differs from SDK/backend acceptance")
                    elif failure_stage in {"transition_failure", "recording_failure"}:
                        if sent is None or timing is None or response_record is None or row.get("reply") is None:
                            raise ValueError("post-reply failure lacks its successful response and reply record")
                        _compare_wire_evidence(request_id, expected_wire, ledger, sent)
                        expected_reply, error_type = asyncio.run(_replay_backend(request, key[1],
                            response_record, timing))
                        recorded_reply = DecisionReply(**row["reply"])
                        if error_type is not None or expected_reply is None or not _same_reply(expected_reply, recorded_reply):
                            raise ValueError("post-reply failure transcript does not match accepted backend reply")
                    else:  # pre-send failures must have no attempt row at all.
                        if ledger is not None:
                            raise ValueError("pre-send failure unexpectedly charged an attempt")
                    advanced = advance_trace(base, trace_rows, None)
                    if advanced[-1] != row["event"]:
                        raise ValueError("recorded failure differs from request state")
                else:
                    raise ValueError("unknown event kind in portfolio trace")
                trace_rows.append(row["event"])
            elif action["kind"] in {"complete", "incomplete"}:
                raise ValueError("trace contains events after a terminal state")
            cursor += 1
        trace = tuple(trace_rows)
        replayed = replay_trace(base, trace)
        status = sessions_by_key[key]
        last_head = last_head_by_session.get(key)
        if status.get("trace_head") != last_head:
            raise ValueError("terminal session trace head differs from reconstructed trace")
        if status.get("status") in {"complete", "incomplete"}:
            if status.get("call_count") != model_call_n or replayed["status"] != status["status"]:
                raise ValueError("terminal session status or call count differs from reconstructed trace")
            expected_reason = ("provider_or_validation_failure" if replayed.get("reason") == "provider_failure"
                else replayed.get("reason"))
            if status.get("status") == "incomplete" and status.get("reason") != expected_reason:
                raise ValueError("terminal incomplete reason differs from reconstructed trace")
            if status.get("failure_stage") != next((row.get("failure_stage") for row in session_events
                    if row.get("kind") == "failure"), None):
                raise ValueError("terminal failure stage differs from reconstructed trace")
            if status.get("status") == "complete" and canonical_json(status.get("orders")) != canonical_json(
                    [asdict(order) for order in replayed["orders"]]):
                raise ValueError("terminal portfolio differs from reconstructed complete trace")
        elif status.get("status") == "unattempted":
            if session_events or model_call_n or last_head is not None:
                raise ValueError("unattempted session has trace or HTTP events")
        elif status.get("status") == "error":
            if replayed["status"] != "incomplete" or status.get("call_count") != model_call_n:
                raise ValueError("terminal error session unexpectedly reconstructs as complete")
        else:
            raise ValueError("terminal session has an unknown status")
        result[f"{key[0]}:{key[1]}"] = replayed
    if linked_request_ids != set(audit["ledger_by_id"]):
        raise ValueError("charged HTTP attempts do not exactly match reconstructed model decisions")
    return {"audit": audit, "sessions": result}


def _compare_wire_evidence(request_id: str, expected: ExpectedWire, ledger: Mapping[str, Any],
                           sent: Mapping[str, Any]) -> None:
    if (ledger.get("request_id") != request_id or ledger.get("method") != expected.method
            or _url_tuple(ledger.get("url", "")) != expected.url
            or expected.url != _url_tuple(DEFAULT_URL)
            or ledger.get("expected_sha256") != expected.body_sha256
            or ledger.get("actual_sha256") != expected.body_sha256
            or ledger.get("body_bytes") != len(expected.body)
            or sent.get("body_sha256") != expected.body_sha256
            or sent.get("body_bytes") != len(expected.body)
            or sent.get("body_utf8", "").encode("utf-8") != expected.body):
        raise ValueError("recorded request or HTTP body differs from the frozen SDK wire")


def _same_reply(left: DecisionReply, right: DecisionReply) -> bool:
    fields = ("request_id", "agent_id", "round_index", "stage", "observation_hash",
        "candidate_set_hash", "raw_scores", "probabilities", "backend", "model_revision", "usage")
    return all(canonical_json(getattr(left, field)) == canonical_json(getattr(right, field)) for field in fields)


async def _replay_backend(request: Any, profile: str, response_record: Mapping[str, Any] | None,
                          timing: Mapping[str, Any] | None) -> tuple[DecisionReply | None, str | None]:
    """Re-run the installed SDK/backend parser against a recorded response body offline."""
    import httpx2
    from typesafe_sdk import AsyncTypeSafeClient
    from typesafe_sdk._core.retry import RetryPolicy
    if (response_record is None or timing is None
            or not 200 <= int(response_record.get("status", 0)) < 300
            or not isinstance(response_record.get("response_body_utf8"), str)
            or timing.get("status") != response_record.get("status")):
        raise ValueError("backend replay requires a preserved successful HTTP response")
    status = response_record["status"]
    response_bytes = response_record["response_body_utf8"].encode("utf-8")
    async def responder(outgoing: Any) -> httpx2.Response:
        await outgoing.aread()
        return httpx2.Response(status, content=response_bytes)
    http = httpx2.AsyncClient(transport=httpx2.MockTransport(responder), follow_redirects=False)
    client = AsyncTypeSafeClient(api_key="offline-replay-dummy", model=MODEL,
        retry=RetryPolicy(max_retries=0), http_client=http,
        base_url="https://api.typesafe.ai", compression=False)
    backend = JevBackend(api_key="offline-replay-dummy", rps=10000, tps=100_000_000,
        max_requests=1, max_estimated_input_tokens=1_000_000,
        instruction_profile=profile, objective="portfolio_prediction_v1",
        request_timeout=30, max_in_flight=1, client=client, load_environment=False)
    try:
        try:
            (reply,) = await backend.decide_many([request])
            return reply, None
        except Exception as exc:
            return None, type(exc).__name__
    finally:
        backend._closed = True
        await client.aclose()


async def _replay_backend_failure(request: Any, profile: str,
                                  response_record: Mapping[str, Any] | None,
                                  timing: Mapping[str, Any] | None) -> str:
    reply, error_type = await _replay_backend(request, profile, response_record, timing)
    if reply is not None or error_type is None:
        raise ValueError("SDK/backend accepts the recorded response, so it cannot be a reply-validation failure")
    return error_type


def _validate_transport_failure(request: Any, response_record: Mapping[str, Any] | None,
                                timing: Mapping[str, Any]) -> None:
    if response_record is None:
        if isinstance(timing.get("status"), int):
            raise ValueError("HTTP response exists but has no preserved response record")
        return  # Charged network timeout/connection failure; no response was received.
    status = response_record.get("status")
    if not isinstance(status, int):
        raise ValueError("failure response lacks an HTTP status")
    if timing.get("status") != status:
        raise ValueError("failed HTTP response status differs from transport timing evidence")
    if 200 <= status < 300:
        raise ValueError("successful HTTP response must use the reply-validation or post-reply failure stage")


def score_run(preview_dir: str | Path, run_dir: str | Path) -> dict[str, Any]:
    manifest, inputs = verify_preview(preview_dir)
    replayed = replay_run(preview_dir, run_dir)
    target_path = Path(preview_dir) / "scoring-targets.json"
    target_bytes = target_path.read_bytes()
    if _sha_bytes(target_bytes) != manifest["scoring_targets_sha256"]:
        raise ValueError("scoring target file checksum differs from the frozen manifest")
    targets = json.loads(target_bytes)
    statuses = json.loads((Path(run_dir) / "terminal.json").read_text(encoding="utf-8"))["statuses"]
    terminal = json.loads((Path(run_dir) / "terminal.json").read_text(encoding="utf-8"))
    execution_mode = terminal["execution_mode"]
    per_arm = {}
    for arm in ARMS:
        predicted, human, q_values = {}, {}, {}
        for session in manifest["sessions"]:
            if session["arm"] != arm:
                continue
            target_id = session["target_id"]
            key = f"{target_id}:{arm}"
            result = replayed["sessions"][key]
            predicted[target_id] = result["orders"] if result["status"] == "complete" else None
            human[target_id] = tuple(PaperOrder(**row) for row in targets[target_id]["orders"])
            q_values[target_id] = inputs["bases"][target_id]["Q"]
        per_arm[arm] = score_targets(predicted, human, q_values)
    target_differences = {}
    for target in targets:
        first_row, third_row = (per_arm[arm]["targets"][target] for arm in ARMS)
        first_error = first_row.get("scored_error", first_row.get("primary_curve_error"))
        third_error = third_row.get("scored_error", third_row.get("primary_curve_error"))
        target_differences[target] = first_error - third_error
    by_person: dict[str, dict[str, list[float]]] = {}
    classes: dict[str, dict[str, int]] = {}
    for arm in ARMS:
        for target, row in per_arm[arm]["targets"].items():
            person = target.rsplit(":r", 1)[0]
            by_person.setdefault(arm, {}).setdefault(person, []).append(row.get("scored_error",
                row.get("primary_curve_error")))
            if row.get("status") == "complete":
                classes.setdefault(arm, {}).setdefault(row["predicted_side_class"], 0)
                classes[arm][row["predicted_side_class"]] += 1
    private_training = {}
    baseline_training = inputs.get("baseline_training", {})
    baseline_history = {}
    for person, rows in baseline_training.items():
        baseline_history[person] = [{"round": row["round"], "orders": tuple(
            PaperOrder(f"{person}:training:{row['round']}:slot{slot}", person,
                order["type"].lower(), order["amount"], order["price"])
            for slot, order in enumerate(row["orders"]))} for row in rows]
    human_for_baseline = {key: tuple(PaperOrder(**row) for row in targets[key]["orders"]) for key in targets}
    q_for_baseline = {key: inputs["bases"][key]["Q"] for key in targets}
    # Each person's baselines use only that person's eight formal training rounds.
    training_scores = {"per_person": {}}
    for person, history in baseline_history.items():
        own_targets = {key: value for key, value in human_for_baseline.items()
            if inputs["bases"][key]["agent_id"] == person}
        own_q = {key: q_for_baseline[key] for key in own_targets}
        training_scores["per_person"][person] = training_baselines(history, own_targets, own_q)
    statuses = json.loads((Path(run_dir) / "terminal.json").read_text(encoding="utf-8"))["statuses"]
    per_person_metrics = {arm: {person: {"target_n": len(values), "mean_curve_error": sum(values) / len(values)}
        for person, values in people.items()} for arm, people in by_person.items()}
    private_result = {"status": "SCORED_OFFLINE", "source_kind": manifest["source_kind"],
        "execution_mode": execution_mode,
        "target_n": len(targets), "arms": per_arm, "per_person": per_person_metrics,
        "predicted_side_class_counts": classes, "training_only_baselines": training_scores,
        "paired_primary_curve_error_difference_first_minus_third_by_target": target_differences,
        "query_calls_by_arm": {arm: sum(row.get("call_count", 0) for row in statuses if row.get("arm") == arm)
            for arm in ARMS}, "replay_audit": replayed["audit"]}
    _write_json(Path(run_dir) / "private-scores.json", private_result)
    safe = {"status": "SCORED_OFFLINE", "execution_mode": execution_mode,
        "scope": "three new GA participants; six held-out portfolios per view",
        "portfolio_n": len(targets), "source_kind": manifest["source_kind"],
        "arms": {arm: {key: value for key, value in per_arm[arm].items()
            if key != "targets"} for arm in ARMS},
        "paired_curve_error_difference_first_minus_third_mean": sum(target_differences.values()) / len(target_differences)
            if target_differences else None,
        "query_calls_by_arm": private_result["query_calls_by_arm"],
        "training_baselines_reported": bool(training_scores),
        "limitations": ["Conditional on actual pre-trading p, Q, and D; no procurement prediction.",
            "No calculator tool; incomplete histories omit round-9 feedback in round 10.",
            "Three selected people do not support population-equivalence claims."]}
    _write_json(Path(run_dir) / "safe-aggregate.json", safe)
    return safe


def main(argv: Sequence[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Frozen private Jev portfolio prediction study")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--synthetic", action="store_true", help="run only an injected mock SDK fixture")
    mode.add_argument("--preview", action="store_true", help="freeze authorized human inputs without provider setup")
    mode.add_argument("--execute", action="store_true", help="execute a reviewed frozen preview")
    mode.add_argument("--replay", action="store_true", help="audit a completed private trace offline")
    mode.add_argument("--score", action="store_true", help="score human targets offline after inference")
    parser.add_argument("--source", type=Path, default=Path("C:/Users/hongy/Downloads/mnsc.2023.03771.zip"))
    parser.add_argument("--policy", type=Path, default=Path("artifacts/participant-data/quote-perspective-24/source-policy-authorized.json"))
    parser.add_argument("--parent-run", type=Path, default=Path("artifacts/participant-data/procurement-perspective-24/frozen-plan.json"))
    parser.add_argument("--preview-dir", type=Path, default=Path("artifacts/participant-data/portfolio-prediction-24"))
    parser.add_argument("--run-dir", type=Path, default=Path("artifacts/participant-data/portfolio-prediction-run-24"))
    parser.add_argument("--reviewed-preview-sha256")
    parser.add_argument("--mock", action="store_true", help="use an injected mock transport; never reads credentials")
    args = parser.parse_args(argv)
    if args.synthetic:
        preview = args.preview_dir
        build_synthetic_preview(preview)
        result = asyncio.run(execute_preview(preview, args.run_dir, fake=True))
        print(json.dumps({k: v for k, v in result.items() if k != "aggregate"}, indent=2))
        return 0 if result["status"] == "COMPLETE" else 2
    if args.replay:
        audit = replay_run(args.preview_dir, args.run_dir)
        statuses = [row["status"] for row in json.loads(
            (args.run_dir / "terminal.json").read_text(encoding="utf-8"))["statuses"]]
        summary = {"status": "REPLAY_VALID", "audit": {key: value for key, value in audit["audit"].items()
            if key in {"trace_event_n", "ledger_line_n", "status_n", "transport_body_line_n",
                "transport_request_line_n"}},
            "complete_n": statuses.count("complete"), "incomplete_n": statuses.count("incomplete"),
            "error_n": statuses.count("error"), "unattempted_n": statuses.count("unattempted")}
        print(json.dumps(summary, indent=2))
        return 0
    if args.score:
        print(json.dumps(score_run(args.preview_dir, args.run_dir), indent=2))
        return 0
    if args.execute:
        result = asyncio.run(execute_preview(args.preview_dir, args.run_dir,
            reviewed_preview_sha256=args.reviewed_preview_sha256, fake=args.mock))
        print(json.dumps(result, indent=2))
        return 0 if result["status"] == "COMPLETE" else 2
    policy = json.loads(args.policy.read_text(encoding="utf-8"))
    source, episodes = load_paper_episodes(args.source,
        intended_use_authorized=policy.get("intended_use_authorized") is True,
        provider_transmission_permitted=policy.get("provider_transmission_permitted") is True,
        codebook_verified=policy.get("codebook_verified") is True)
    if (policy.get("source_kind") != "human" or policy.get("source_sha256") != source.source_sha256
            or source.source_sha256 != EXPECTED_SOURCE_SHA):
        raise ValueError("source policy and verified workbook checksum do not match")
    if not all((source.codebook_verified, source.intended_use_authorized, source.provider_transmission_permitted)):
        print(json.dumps({"status": "BLOCKED", "reason": "source attestations are false; no contexts or SDK initialized"}))
        return 3
    if args.preview:
        policy_sha = _sha_bytes(args.policy.resolve().read_bytes())
        plan = asyncio.run(build_preview(source, episodes, args.parent_run, args.preview_dir,
            source_policy_sha256=policy_sha))
        path = _private_output_path(args.preview_dir)
        print(json.dumps({"status": "FROZEN", "preview_sha256": _sha_bytes((path / "frozen-plan.json").read_bytes()),
            "portfolio_n": len(plan["sessions"]), "max_body_bytes": plan["preflight_max_body_bytes"]}, indent=2))
        return 0
    raise AssertionError("unreachable CLI mode")

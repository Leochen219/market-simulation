"""Offline-first Jev comparison of human first-person and third-person procurement prediction."""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import threading
import time
from typing import Any, Mapping, Sequence

from capclaims.backends.jev import MODEL, JevBackend, RecordingTransport
from capclaims.backends.jev_instructions import build_instruction
from capclaims.domain.types import Candidate
from capclaims.engine.protocol import DecisionRequest, validate_reply
from capclaims.personalization import Episode, ParticipantConditioner, SourceManifest, Target
from capclaims.personalization_xlsx import load_paper_episodes
from capclaims.paper.codec import decode_prefix, legal_prefix_choices
from capclaims.paper.interface import build_candidates
from capclaims.randomness import canonical_json

SEED = 20261024
ARMS = ("first_own", "third_own", "first_none", "first_shuffled")
MAIN_ARMS = ARMS[:2]
PROFILE_FOR_ARM = {"first_own": "procurement_first_v1", "third_own": "procurement_third_v1",
    "first_none": "procurement_first_v1", "first_shuffled": "procurement_first_v1"}
HISTORY_FOR_ARM = {"first_own": "own", "third_own": "own", "first_none": "none", "first_shuffled": "shuffled"}
REPS_FOR_ARM = {"first_own": (1, 2), "third_own": (1, 2), "first_none": (1,), "first_shuffled": (1,)}
BASELINE = "first_own"
TRAINING_MAX_ROUND = 8
DEV_ROUNDS = (9, 10, 11)
CONFIRM_ROUNDS = (12, 13, 14, 15)
REPETITIONS = (1, 2)
MAX_PREDICTIONS = 126
MAX_REQUESTS = 480
MAX_ESTIMATED_BYTES = 3_200_000
MAX_RPS = 40
MAX_TPS_BYTES = 50_000
MAX_IN_FLIGHT = 10
EXPECTED_SOURCE_SHA = "f9978ebebe862e4dab51a1885a5009b7d570da1a9e49e01d2f3becfe1af385a9"
TRAINING_FAILURE_PENALTY = 401


def select_participants(roster: Sequence[str], *, seed: int = SEED) -> tuple[str, ...]:
    """Select three roster members using IDs only; no actions or outcomes are accepted here."""
    if len(roster) != 35 or len(set(roster)) != 35 or any(not isinstance(x, str) or not x for x in roster):
        raise ValueError("GA10 roster must contain 35 unique local participant identifiers")
    return tuple(sorted(roster, key=lambda person: hashlib.sha256(f"{seed}:{person}".encode()).digest())[:3])


def build_choice_prefixes(minimum: int = 0, maximum: int = 400) -> tuple[str, ...]:
    """Enumerate all reachable own prefixes through three digits for the frozen quantity range."""
    if (minimum, maximum) != (0, 400):
        raise ValueError("the participant procurement study freezes the full Q range 0..400")
    prefixes = [""]
    for h in legal_prefix_choices("", 400):
        first = str(h)
        prefixes.append(first)
        for t in legal_prefix_choices(first, 400):
            second = first + str(t)
            prefixes.append(second)
    return tuple(prefixes)


def decode_prediction(digits: Sequence[int | None]) -> int:
    if any(d is None for d in digits):
        raise ValueError("unresolved top-probability tie has no decoded quantity")
    if len(digits) != 3 or any(type(d) is not int or not 0 <= d <= 9 for d in digits):
        raise ValueError("prediction requires exactly three legal digit choices")
    quantity = decode_prefix(tuple(digits))
    if quantity > 400:
        raise ValueError("decoded quantity is outside legal 0..400 range")
    return quantity


def filtered_training(episodes: Sequence[Episode]) -> tuple[Episode, ...]:
    """Freeze the study corpus to procurement rounds 1..8 before target construction."""
    rows = tuple(e for e in episodes if e.phase == "procurement" and e.scope == "GA 10"
        and e.treatment == "GA" and e.round <= TRAINING_MAX_ROUND)
    if any(e.round > TRAINING_MAX_ROUND for e in rows):
        raise AssertionError("training filter admitted a held-out row")
    return rows


def development_winner(scores: Mapping[str, Sequence[float | None]]) -> str:
    if set(scores) != set(MAIN_ARMS):
        raise ValueError("only the two frozen main perspective arms are eligible for selection")
    ranked = []
    for i, arm in enumerate(MAIN_ARMS):
        values = scores[arm]
        if not values:
            raise ValueError("development scores cannot be empty")
        # Failure gets a fixed 401 penalty: worse than every possible legal quantity error.
        loss = sum(TRAINING_FAILURE_PENALTY if value is None else value for value in values) / len(values)
        ranked.append((loss, i, arm))
    return min(ranked)[-1]


class StudyBudget:
    """One global thread-safe request/estimated-byte budget and paced concurrency gate."""
    def __init__(self, *, max_requests: int = MAX_REQUESTS, max_estimated_bytes: int = MAX_ESTIMATED_BYTES,
                 rps: float = MAX_RPS, tps_bytes: int = MAX_TPS_BYTES, max_in_flight: int = MAX_IN_FLIGHT):
        if max_requests <= 0 or max_estimated_bytes <= 0 or rps <= 0 or tps_bytes <= 0 or max_in_flight != 10:
            raise ValueError("invalid procurement study shared budget")
        self.max_requests, self.max_estimated_bytes = max_requests, max_estimated_bytes
        self.rps, self.tps_bytes, self.max_in_flight = rps, tps_bytes, max_in_flight
        self.requests = self.estimated_bytes = 0
        self._lock = threading.Lock()
        self._next_request = self._next_bytes = 0.0
        self._semaphore: asyncio.Semaphore | None = None
        self.in_flight = self.max_observed_in_flight = 0
        self.dispatch_starts: list[float] = []

    def reserve(self, count: int, estimated_bytes: int) -> None:
        if type(count) is not int or type(estimated_bytes) is not int or count < 0 or estimated_bytes < 0:
            raise ValueError("budget reservation must use nonnegative integer counts")
        with self._lock:
            if self.requests + count > self.max_requests or self.estimated_bytes + estimated_bytes > self.max_estimated_bytes:
                raise ValueError("shared study budget would be exceeded")
            self.requests += count
            self.estimated_bytes += estimated_bytes

    async def enter(self, estimated_bytes: int) -> None:
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_in_flight)
        await self._semaphore.acquire()
        with self._lock:
            self.in_flight += 1
            self.max_observed_in_flight = max(self.max_observed_in_flight, self.in_flight)
            now = time.monotonic()
            start = max(now, self._next_request, self._next_bytes)
            self._next_request = start + 1 / self.rps
            self._next_bytes = start + estimated_bytes / self.tps_bytes
            self.dispatch_starts.append(start)
        try:
            await asyncio.sleep(max(0.0, start - time.monotonic()))
        except BaseException:
            self.exit()
            raise

    def exit(self) -> None:
        with self._lock:
            self.in_flight -= 1
        if self._semaphore is not None:
            self._semaphore.release()


class SharedJevClient:
    """Validates actual condition-independent wire hashes and charges failed calls globally."""
    def __init__(self, inner: Any, budget: StudyBudget, allowed_wire_hashes: set[str]):
        self.inner, self.budget = inner, budget
        self.allowed_wire_hashes = frozenset(allowed_wire_hashes)
        self.closed = False

    @staticmethod
    def wire(state: Mapping[str, Any], questions: Mapping[str, Any]) -> tuple[dict[str, Any], int]:
        choice = questions["decision"]
        prompt = choice.instructions if hasattr(choice, "instructions") else choice["instructions"]
        criteria = choice.criteria if hasattr(choice, "criteria") else choice["criteria"]
        value = {"state": dict(state), "instruction": prompt, "criteria": dict(criteria)}
        return value, len(canonical_json(value).encode()) + 256

    async def system_one(self, *, state: Mapping[str, Any], questions: Mapping[str, Any], **kwargs: Any) -> Any:
        if self.closed:
            raise RuntimeError("shared Jev client is closed")
        wire, byte_count = self.wire(state, questions)
        digest = _sha(wire)
        if digest not in self.allowed_wire_hashes:
            raise ValueError("actual Jev wire is outside frozen prefix hash allowlist")
        self.budget.reserve(1, byte_count)  # Failed transport attempts remain charged.
        await self.budget.enter(byte_count)
        try:
            return await self.inner.system_one(state=state, questions=questions, **kwargs)
        finally:
            self.budget.exit()

    async def aclose(self) -> None:
        return None  # All condition backends share this owner.

    async def close_inner(self) -> None:
        if self.closed:
            return
        self.closed = True
        close = getattr(self.inner, "aclose", None) or getattr(self.inner, "close", None)
        if close is not None:
            result = close()
            if hasattr(result, "__await__"):
                await result


def _sha(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def maximum_http_overlap(timings: Sequence[Mapping[str, Any]]) -> int:
    """Maximum simultaneous HTTP request intervals from payload-free transport timings."""
    events = []
    for row in timings:
        start, finish = row.get("started_at"), row.get("finished_at")
        if (not isinstance(start, (int, float)) or not isinstance(finish, (int, float))
                or finish < start):
            raise ValueError("HTTP timing rows require valid start and finish timestamps")
        events.extend(((float(start), 1), (float(finish), -1)))
    active = maximum = 0
    # At equal times, end intervals before starting new ones (half-open intervals).
    for _, delta in sorted(events, key=lambda event: (event[0], event[1])):
        active += delta
        maximum = max(maximum, active)
    return maximum


def _session_state(episode: Episode, person_index: int, arm: str,
                   source: SourceManifest, history_person: str | None,
                   conditioner: ParticipantConditioner | None) -> dict[str, Any]:
    if history_person is None:
        history = {"schema": "participant_history_v1", "source_kind": source.source_kind,
            "treatment": "GA", "phase": "procurement", "episodes": []}
    else:
        opaque_agent = f"study-agent-{person_index}-{arm}-r{episode.round}"
        target = Target(request_id=f"opaque-target-{person_index}-{arm}-{episode.round}",
            agent_id=opaque_agent, participant_id=history_person, scope="GA 10", treatment="GA",
            target_round=episode.round, phase="procurement", logical_decision_key=f"round-{episode.round}")
        if conditioner is None:
            raise AssertionError("history source requires a conditioner")
        history = conditioner.prepare(target, provider=True)
    return {"treatment": "GA", "market_size_note": "historical market participant count not reconstructed",
        "round": episode.round,
        "wholesale_price_dollars": episode.observed["w"], "own_p": episode.observed["p"],
        "public_instructions": {
            "treatment": "GA treatment; historical market composition and participant count are not reconstructed from this workbook.",
            "procurement": "Choose an integer Q from 0 through 400 before your own demand is revealed. Each procured unit costs the wholesale price.",
            "demand": "Own demand is an independent uniform integer from 0 through 200 and is revealed only after procurement.",
            "trading": "After own demand is revealed, agents submit up to five blind mixed buy/sell limit orders; order quantities and integer-dollar limit prices range 1..999.",
            "sell_cap": "Gross submitted sell quantities cannot exceed initial procurement Q; buys do not increase this cap.",
            "clearing": "Fills may be partial or zero. Matched trades use the uniform midpoint of the last matched bid and ask. You receive p for every fulfilled unit. The original market participant count is unknown here.",
            "capacity": "Capacity not used to fulfill your own demand expires after the round.",
            "profit": "Accounting: -w*Q - m*A + p*min(D,Q+A), with market price m and net market allocation A determined after the blind auction; both are unknown at procurement."},
        "participant_history": {k: v for k, v in history.items() if k not in {"context_sha256", "episode_count"}}}


def make_request(session: Mapping[str, Any], prefix: str) -> DecisionRequest:
    place = "htu"[len(prefix)]
    values = legal_prefix_choices(prefix, 400)
    observation = {"decision_interface": "economic_v2", "decision_field": "purchase", "phase": "procurement",
        "numeric_prefix": prefix, "choice_minimum": 0, "choice_maximum": 400,
        "decimal_place_name": {"h": "hundreds", "t": "tens", "u": "ones"}[place],
        "numeric_value_name": "procurement quantity", "numeric_units": "units"}
    candidates = build_candidates(values, observation)
    return DecisionRequest(session["agent_id"], session["round"] - 1,
        f"paper:purchase:{place}", {**session["state"], **observation}, tuple(candidates))


def _wire(request: DecisionRequest, arm: str) -> dict[str, Any]:
    return {"state": dict(request.observation), "instruction": build_instruction(request, PROFILE_FOR_ARM[arm]),
        "criteria": {candidate.candidate_id: candidate.label for candidate in request.candidates}}


def _prefix_wire_rows(session: Mapping[str, Any], arm: str) -> dict[str, dict[str, Any]]:
    nodes = {}
    for prefix in build_choice_prefixes():
        request = make_request(session, prefix)
        wire = _wire(request, arm)
        nodes[prefix] = {"wire": wire, "wire_sha256": _sha(wire),
            "estimated_bytes": len(canonical_json(wire).encode()) + 256}
    return nodes


def _prepare_plan(source: SourceManifest, episodes: Sequence[Episode], *, seed: int) -> dict[str, Any]:
    corpus = filtered_training(episodes)
    roster = sorted({e.participant_id for e in episodes if e.phase == "procurement" and e.scope == "GA 10" and e.treatment == "GA"})
    chosen = select_participants(roster, seed=seed)
    all_rows = [e for e in episodes if e.phase == "procurement" and e.scope == "GA 10" and e.treatment == "GA"]
    by_target = {(e.participant_id, e.round): e for e in all_rows}
    if len(all_rows) != 35 * 15 or any((p, r) not in by_target for p in roster for r in range(1, 16)):
        raise ValueError("GA10 procurement source must contain one row per roster member and formal round 1..15")
    selected_indices = {person: i for i, person in enumerate(chosen)}
    # Fixed three-person derangement; donor assignment uses selected IDs/order only, never actions.
    donor_for = {person: chosen[(index + 1) % len(chosen)] for index, person in enumerate(chosen)}
    conditioners = {(arm, history_person): ParticipantConditioner(source,
        [e for e in corpus if e.participant_id == history_person])
        for arm in ARMS for history_person in chosen}
    sessions = []
    prefix_hashes: dict[tuple[int, str, str], str] = {}
    allowed_hashes: set[str] = set()
    truth: dict[str, int] = {}
    training: dict[str, list[int]] = {}
    seq = 0
    targets = [(p, r) for p in chosen for r in (*DEV_ROUNDS, *CONFIRM_ROUNDS)]
    for person, round_no in targets:
        episode = by_target[(person, round_no)]
        opaque_person = str(selected_indices[person])
        truth[f"{opaque_person}:{round_no}"] = episode.action["Q"]
    for arm in ARMS:
        for person, round_no in targets:
            episode = by_target[(person, round_no)]
            person_index = selected_indices[person]
            history_mode = HISTORY_FOR_ARM[arm]
            history_person = person if history_mode == "own" else donor_for[person] if history_mode == "shuffled" else None
            conditioner = conditioners[(arm, history_person)] if history_person is not None else None
            state = _session_state(episode, person_index, arm, source, history_person, conditioner)
            donor_index = selected_indices[history_person] if history_mode == "shuffled" else None
            train_values = sorted((e.round, e.action["Q"]) for e in corpus if e.participant_id == person)
            training.setdefault(str(person_index), [q for _, q in train_values])
            donor_values = sorted((e.round, e.action["Q"]) for e in corpus if e.participant_id == history_person) if history_person else []
            for repetition in REPS_FOR_ARM[arm]:
                session = {"sequence": seq, "arm": arm, "participant": str(person_index),
                    "round": round_no, "repetition": repetition, "phase": "development" if round_no in DEV_ROUNDS else "confirmation",
                    "history_condition": history_mode, "donor_index": donor_index,
                    "donor_training_quantities": [q for _, q in donor_values],
                    "agent_id": f"study-agent-{person_index}-{arm}-r{round_no}", "state": state}
                sessions.append(session)
                prefix_rows = _prefix_wire_rows(session, arm)
                for prefix, row in prefix_rows.items():
                    prefix_hashes[(seq, arm, prefix)] = row["wire_sha256"]
                    allowed_hashes.add(row["wire_sha256"])
                # Maximum legitimate size over all three-call paths, computed over Q=0..400.
                path_costs = []
                for q in range(401):
                    digits = f"{q:03d}"
                    path_costs.append(sum(prefix_rows[digits[:depth]]["estimated_bytes"] for depth in range(3)))
                session["worst_path_estimated_bytes"] = max(path_costs)
                session["_prefix_rows"] = prefix_rows
                seq += 1
    # Wire-equivalent facts and labels stay separate; truth is not included in session state or frozen hash.
    planned_requests = sum(3 for _ in sessions)
    if len(sessions) != MAX_PREDICTIONS or planned_requests != 378:
        raise ValueError("unexpected frozen procurement target/call count")
    estimated_bytes = sum(s["worst_path_estimated_bytes"] for s in sessions)
    if estimated_bytes > MAX_ESTIMATED_BYTES:
        raise ValueError("frozen procurement study exceeds 3.2m estimated input bytes")
    prompt_samples = {}
    for arm in ARMS:
        request = make_request(sessions[0], "")
        prompt = build_instruction(request, PROFILE_FOR_ARM[arm])
        prompt_samples[arm] = {"text": prompt, "sha256": hashlib.sha256(prompt.encode()).hexdigest()}
    base_input_hashes = {_sha({key: value for key, value in row["state"].items() if key != "participant_history"})
        for row in sessions}
    full_input_hashes_by_arm = {arm: sorted({_sha(row["state"]) for row in sessions if row["arm"] == arm})
        for arm in ARMS}
    project = Path(__file__).resolve().parents[2]
    code_paths = ("src/capclaims/procurement_perspective.py", "src/capclaims/backends/jev.py",
        "src/capclaims/backends/jev_instructions.py", "src/capclaims/personalization.py",
        "src/capclaims/personalization_xlsx.py", "src/capclaims/paper/codec.py",
        "src/capclaims/paper/interface.py", "src/capclaims/engine/protocol.py", "src/capclaims/randomness.py")
    code_hashes = {path: hashlib.sha256((project / path).read_bytes()).hexdigest() for path in code_paths}
    return {"schema": "jev_procurement_perspective_v1", "source_sha256": source.source_sha256,
        "source_kind": source.source_kind, "source_provenance": source.provenance,
        "source_attestations": {"codebook_verified": source.codebook_verified,
            "intended_use_authorized": source.intended_use_authorized,
            "provider_transmission_permitted": source.provider_transmission_permitted},
        "model": MODEL, "seed": seed, "selected_participants": [hashlib.sha256(p.encode()).hexdigest() for p in chosen],
        "roster_n": 35, "training_round_max": 8, "development_rounds": list(DEV_ROUNDS),
        "confirmation_rounds": list(CONFIRM_ROUNDS),
        "arms": list(ARMS), "objective": "procurement_prediction_v1",
        "history_conditions": HISTORY_FOR_ARM,
        "repetitions_by_arm": {arm: list(reps) for arm, reps in REPS_FOR_ARM.items()},
        "history_control_donor_derangement": {str(selected_indices[person]): str(selected_indices[donor_for[person]])
            for person in chosen},
        "unique_current_input_state_n": len(base_input_hashes),
        "full_input_state_n_by_arm": {arm: len(values) for arm, values in full_input_hashes_by_arm.items()},
        "quantity_range": [0, 400], "max_predictions": MAX_PREDICTIONS,
        "expected_prediction_calls": planned_requests, "max_requests": MAX_REQUESTS,
        "planned_worst_path_estimated_bytes": estimated_bytes,
        "max_estimated_bytes": MAX_ESTIMATED_BYTES, "rps": MAX_RPS,
        "tps_bytes": MAX_TPS_BYTES, "max_in_flight": MAX_IN_FLIGHT,
        "source_code_sha256": code_hashes, "prompt_manifest": prompt_samples,
        "wire_hash_allowlist": sorted(allowed_hashes), "wire_prefix_hashes": [
            {"sequence": seq, "arm": arm, "prefix": prefix, "wire_sha256": digest}
            for (seq, arm, prefix), digest in sorted(prefix_hashes.items())],
        "wire_prefixes": [{"sequence": row["sequence"], "arm": row["arm"], "prefix": prefix,
            **wire_row} for row in sessions for prefix, wire_row in row["_prefix_rows"].items()],
        "sessions": [{k: v for k, v in row.items() if not k.startswith("_")} for row in sessions],
        "truth": truth, "training_quantities": training,
        "_sessions": sessions, "_prefix_hashes": prefix_hashes,
        "_allowed_hashes": allowed_hashes, "_truth": truth, "_training": training,
        "_chosen": tuple(chosen)}


def build_study(source: SourceManifest, episodes: Sequence[Episode], *, seed: int = SEED) -> dict[str, Any]:
    if source.source_kind != "human" or source.source_sha256 != EXPECTED_SOURCE_SHA:
        raise ValueError("live procurement comparison requires the verified GA10 human-source checksum")
    if not (source.codebook_verified and source.intended_use_authorized and source.provider_transmission_permitted):
        raise PermissionError("human source use and provider transmission must be authorized before context preparation")
    return _prepare_plan(source, episodes, seed=seed)


def _synthetic_source() -> tuple[SourceManifest, list[Episode]]:
    source = SourceManifest("synthetic", "generated mock-only fixtures", "0" * 64, True, True, True)
    rows = []
    for person in range(35):
        pid = f"mock-participant-{person:02d}"
        for round_no in range(1, 16):
            p = 55 + (person + 3 * round_no) % 46
            q = (17 * person + 29 * round_no) % 401
            rows.append(Episode(pid, "GA 10", "GA", round_no, "procurement",
                {"p": p, "w": 10}, {"Q": q}))
    return source, rows


def build_synthetic_study(*, seed: int = SEED) -> dict[str, Any]:
    source, episodes = _synthetic_source()
    return _prepare_plan(source, episodes, seed=seed)


async def run_session(session: Mapping[str, Any], backend: JevBackend,
                      prefix_hashes: Mapping[tuple[int, str, str], str],
                      transcript: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    digits: list[int] = []
    calls: list[dict[str, Any]] = []
    prefix = ""
    for _ in range(3):
        request = make_request(session, prefix)
        wire = _wire(request, session["arm"])
        digest = _sha(wire)
        if prefix_hashes.get((session["sequence"], session["arm"], prefix)) != digest:
            raise ValueError("runtime procurement wire differs from frozen prefix")
        (reply,) = await backend.decide_many([request])
        validate_reply(request, reply)
        top = max(reply.probabilities.values())
        winners = [candidate for candidate in request.candidates
            if abs(reply.probabilities[candidate.candidate_id] - top) <= 1e-12]
        row = {"sequence": session["sequence"], "arm": session["arm"],
            "prefix": prefix, "request": asdict(request), "wire": wire, "reply": asdict(reply)}
        calls.append(row)
        if transcript is not None:
            transcript.append(row)
        if len(winners) != 1:
            return {"sequence": session["sequence"], "arm": session["arm"],
                "participant": session["participant"], "round": session["round"],
                "repetition": session["repetition"], "predicted_quantity": None,
                "status": "unresolved_digit_tie", "digits": digits, "call_count": len(calls)}
        digit = winners[0].value
        if type(digit) is not int or digit not in legal_prefix_choices(prefix, 400):
            raise ValueError("model selected a digit outside the legal prefix candidates")
        digits.append(digit)
        prefix += str(digit)
    quantity = decode_prediction(digits)
    return {"sequence": session["sequence"], "arm": session["arm"],
        "participant": session["participant"], "round": session["round"],
        "repetition": session["repetition"], "predicted_quantity": quantity,
        "status": "complete", "digits": digits, "call_count": len(calls)}


def _metrics(sessions: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]],
             truth: Mapping[str, int], training: Mapping[str, Sequence[int]]) -> dict[str, Any]:
    pred_by_seq = {p["sequence"]: p for p in predictions}
    complete = []; all_errors = []; per_person: dict[str, list[float]] = {
        person: [] for person in sorted({s["participant"] for s in sessions})}
    grouped: dict[tuple[str, int], list[int | None]] = {}
    failures = ties = 0
    copy_last_q = within_visible_history_range = copy_donor_last_q = 0
    copy_diag_n = range_diag_n = donor_diag_n = 0
    for session in sessions:
        key = f"{session['participant']}:{session['round']}"
        q = truth[key]
        prediction = pred_by_seq.get(session["sequence"], {}).get("predicted_quantity")
        if prediction is None:
            failures += 1
            if pred_by_seq.get(session["sequence"], {}).get("status") == "unresolved_digit_tie":
                ties += 1
            error = None
        else:
            error = prediction - q
            complete.append((prediction, q, session["participant"], session["round"]))
            all_errors.append(error)
            per_person.setdefault(session["participant"], []).append(abs(error))
            own_train = training[session["participant"]]
            copy_diag_n += 1
            copy_last_q += int(prediction == own_train[-1])
            visible_history = [episode["action"]["Q"] for episode in session["state"]["participant_history"]["episodes"]]
            if visible_history:
                range_diag_n += 1
                within_visible_history_range += int(min(visible_history) <= prediction <= max(visible_history))
            donor_train = session.get("donor_training_quantities", [])
            if donor_train and session.get("history_condition") == "shuffled":
                donor_diag_n += 1
                copy_donor_last_q += int(prediction == donor_train[-1])
        grouped.setdefault((session["participant"], session["round"]), []).append(prediction)
    target_repeat = []
    for values in grouped.values():
        if len(values) == 2 and values[0] is not None and values[1] is not None:
            target_repeat.append(int(values[0] == values[1]))
    n = len(complete)
    target_errors: dict[tuple[str, int], list[int]] = {}
    for prediction, q, person, round_no in complete:
        target_errors.setdefault((person, round_no), []).append(prediction - q)
    per_target_abs = [sum(abs(error) for error in values) / len(values) for values in target_errors.values()]
    per_target_signed = [sum(values) / len(values) for values in target_errors.values()]
    metrics = {"prediction_sessions": len(sessions), "completed_n": n,
        "failed_or_unresolved_n": failures, "unresolved_digit_ties_n": ties,
        "mae": sum(per_target_abs) / len(per_target_abs) if per_target_abs else None,
        "completed_targets_n": len(per_target_abs),
        "exact_n": sum(e == 0 for e in all_errors), "exact_rate_completed": sum(e == 0 for e in all_errors) / n if n else None,
        "within_10_n": sum(abs(e) <= 10 for e in all_errors),
        "within_10_rate_completed": sum(abs(e) <= 10 for e in all_errors) / n if n else None,
        "signed_error_mean": sum(per_target_signed) / len(per_target_signed) if per_target_signed else None,
        "per_person_mae": {p: (sum(v) / len(v) if v else None) for p, v in per_person.items()},
        "repeated_choice_same_n": sum(target_repeat), "repeated_choice_pair_n": len(target_repeat),
        "repeated_choice_consistency": sum(target_repeat) / len(target_repeat) if target_repeat else None,
        "copy_diagnostics": {"completed_n": copy_diag_n, "copied_own_last_training_q_n": copy_last_q,
            "copied_own_last_training_q_rate": copy_last_q / copy_diag_n if copy_diag_n else None,
            "range_diagnostic_n": range_diag_n,
            "within_visible_history_range_n": within_visible_history_range,
            "within_visible_history_range_rate": within_visible_history_range / range_diag_n if range_diag_n else None,
            "donor_diagnostic_n": donor_diag_n, "copied_shuffled_donor_last_q_n": copy_donor_last_q}}
    # Individual baselines use only rounds 1..8, never held-out choices.
    selected_people = sorted({session["participant"] for session in sessions})
    pooled = [q for person in selected_people for q in training[person]]
    median = lambda values: (sorted(values)[(len(values) - 1) // 2] + sorted(values)[len(values) // 2]) / 2
    baseline_values = {
        "last_training_q": {person: training[person][-1] for person in selected_people},
        "participant_training_median": {person: median(training[person]) for person in selected_people},
        "pooled_training_median": {person: median(pooled) for person in selected_people} if pooled else {}}
    baseline_metrics = {}
    unique_targets = {(s["participant"], s["round"]) for s in sessions}
    for name, values in baseline_values.items():
        errors = [values[person] - truth[f"{person}:{round_no}"] for person, round_no in unique_targets
            if person in values]
        baseline_metrics[name] = {"unique_target_n": len(errors), "mae": sum(abs(e) for e in errors) / len(errors) if errors else None,
            "exact_n": sum(e == 0 for e in errors), "within_10_n": sum(abs(e) <= 10 for e in errors),
            "signed_error_mean": sum(errors) / len(errors) if errors else None}
    return {**metrics, "training_only_baselines": baseline_metrics}


def _sdk_mock():
    from types import SimpleNamespace
    class FakeClient:
        max_observed_in_flight = 0
        in_flight = 0
        closed = False
        async def system_one(self, *, state, questions, model):
            self.in_flight += 1
            self.max_observed_in_flight = max(self.max_observed_in_flight, self.in_flight)
            try:
                await asyncio.sleep(0.01)
                criteria = questions["decision"].criteria
                # Deliberately choose first available legal digit, using only the wire.
                first = next(iter(criteria))
                return SimpleNamespace(model=model, choices={"decision": SimpleNamespace(
                    probabilities={key: 1.0 if key == first else 0.0 for key in criteria})}, usage={})
            finally:
                self.in_flight -= 1
        async def aclose(self):
            self.closed = True
    return FakeClient()


def _new_backends(shared: SharedJevClient, api_key: str) -> dict[str, JevBackend]:
    return {arm: JevBackend(api_key=api_key, rps=MAX_RPS, tps=MAX_TPS_BYTES,
        max_requests=MAX_REQUESTS, max_estimated_input_tokens=MAX_ESTIMATED_BYTES,
        max_in_flight=MAX_IN_FLIGHT,
        objective="procurement_prediction_v1", client=shared, load_environment=False,
        instruction_profile=PROFILE_FOR_ARM[arm]) for arm in ARMS}


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_frozen_plan(plan: Mapping[str, Any], path: Path) -> None:
    public = {key: value for key, value in plan.items() if not key.startswith("_")}
    _write_json(path, public)


async def execute_study(plan: dict[str, Any], output: Path, *, fake_client: Any = None) -> dict[str, Any]:
    if fake_client is None:
        if plan["source_kind"] != "human" or not all(plan["source_attestations"].values()):
            raise PermissionError("human-source gates must pass before provider initialization")
    elif plan["source_kind"] != "synthetic":
        raise ValueError("synthetic/mock SDK execution requires a generated synthetic study plan")
    artifact_root = (Path(__file__).resolve().parents[2] / "artifacts" / "participant-data").resolve()
    output = output.resolve()
    if output != artifact_root and artifact_root not in output.parents:
        raise ValueError("study outputs must be stored under the ignored participant-data artifact directory")
    output.mkdir(parents=True, exist_ok=False)
    _write_frozen_plan(plan, output / "frozen-plan.json")
    budget = StudyBudget()
    sdk = fake_client
    transport_recorder = None
    if sdk is None:
        from capclaims.backends.jev import load_local_env
        load_local_env()
        api_key = os.environ.get("TYPESAFE_API_KEY")
        if not api_key:
            raise ValueError("TYPESAFE_API_KEY unavailable after source and wire preflight")
        from typesafe_sdk import AsyncTypeSafeClient
        from typesafe_sdk._core.retry import RetryPolicy
        transport_recorder = RecordingTransport()
        sdk = AsyncTypeSafeClient(api_key=api_key, model=MODEL, retry=RetryPolicy(max_retries=0),
            transport=transport_recorder)
    else:
        api_key = "synthetic-no-key"
    shared = SharedJevClient(sdk, budget, set(plan["_allowed_hashes"]))
    backends = None
    predictions: list[dict[str, Any]] = []
    transcript: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    sessions = list(plan["_sessions"])
    try:
        backends = _new_backends(shared, api_key)
        phases = ("development", "confirmation")
        for phase in phases:
            phase_rows = [row for row in sessions if row["phase"] == phase]
            random.Random(SEED + (0 if phase == "development" else 1)).shuffle(phase_rows)
            for start in range(0, len(phase_rows), MAX_IN_FLIGHT):
                wave = phase_rows[start:start + MAX_IN_FLIGHT]
                results = await asyncio.gather(*(run_session(row, backends[row["arm"]], plan["_prefix_hashes"], transcript)
                    for row in wave), return_exceptions=True)
                for row, result in zip(wave, results):
                    if isinstance(result, BaseException):
                        predictions.append({"sequence": row["sequence"], "arm": row["arm"],
                            "participant": row["participant"], "round": row["round"],
                            "repetition": row["repetition"], "predicted_quantity": None,
                            "status": "provider_or_validation_failure"})
                        failures.append({"sequence": row["sequence"], "error_type": type(result).__name__})
                    else:
                        predictions.append(result)
                _write_json(output / "partial-predictions.json", predictions)
                _write_json(output / "call-transcript.json", transcript)
                if failures:
                    break
            if failures:
                break
            if phase == "development":
                truth, training = plan["_truth"], plan["_training"]
                dev = {arm: [row for row in sessions if row["arm"] == arm and row["phase"] == phase] for arm in ARMS}
                by_sequence = {p["sequence"]: p for p in predictions}
                selection_scores = {}
                for arm in MAIN_ARMS:
                    by_target: dict[str, list[float]] = {}
                    for row in dev[arm]:
                        key = f"{row['participant']}:{row['round']}"
                        pred = by_sequence[row["sequence"]]["predicted_quantity"]
                        q = truth[key]
                        by_target.setdefault(key, []).append(TRAINING_FAILURE_PENALTY if pred is None else abs(pred - q))
                    selection_scores[arm] = [sum(values) / len(values) for values in by_target.values()]
                winner = development_winner(selection_scores)
                _write_json(output / "development-winner.json", {"winner": winner,
                    "selection_loss_with_401_failure_penalty": selection_scores})
        if not failures:
            truth, training = plan["_truth"], plan["_training"]
            dev = {arm: [row for row in plan["_sessions"] if row["arm"] == arm and row["phase"] == "development"] for arm in ARMS}
            confirm = {arm: [row for row in plan["_sessions"] if row["arm"] == arm and row["phase"] == "confirmation"] for arm in ARMS}
            by_sequence = {p["sequence"]: p for p in predictions}
            selection_scores = {}
            for arm in MAIN_ARMS:
                by_target: dict[str, list[float]] = {}
                for row in dev[arm]:
                    key = f"{row['participant']}:{row['round']}"
                    pred = by_sequence[row["sequence"]]["predicted_quantity"]
                    q = truth[key]
                    by_target.setdefault(key, []).append(TRAINING_FAILURE_PENALTY if pred is None else abs(pred - q))
                selection_scores[arm] = [sum(values) / len(values) for values in by_target.values()]
            winner = development_winner(selection_scores)
            result_metrics = {"development": {arm: _metrics(dev[arm], predictions, truth, training) for arm in ARMS},
                "confirmation": {arm: _metrics(confirm[arm], predictions, truth, training) for arm in ARMS},
                "development_selection_loss_with_401_failure_penalty": selection_scores,
                "development_winner": winner,
                "confirmation_uses_both_frozen_prompts": True}
            _write_json(output / "safe-aggregate.json", result_metrics)
        else:
            truth, training = plan["_truth"], plan["_training"]
            by_sequence = {p["sequence"]: p for p in predictions}
            attempted_sessions = [row for row in sessions if row["sequence"] in by_sequence]
            result_metrics = {"partial": True, "development": {}, "confirmation": {},
                "development_winner": None}
            for phase in ("development", "confirmation"):
                for arm in ARMS:
                    chosen = [row for row in attempted_sessions if row["phase"] == phase and row["arm"] == arm]
                    result_metrics[phase][arm] = _metrics(chosen, predictions, truth, training)
    except Exception as exc:
        failures.append({"phase": "study", "error_type": type(exc).__name__})
        result_metrics = None
    finally:
        for backend in (backends or {}).values():
            backend._closed = True
        await shared.close_inner()
    _write_json(output / "private-predictions.json", predictions)
    _write_json(output / "call-transcript.json", transcript)
    transport_timings = list(transport_recorder.timings) if transport_recorder is not None else []
    _write_json(output / "transport-timings.json", transport_timings)
    _write_json(output / "failures.json", failures)
    result = {"status": "COMPLETE" if not failures and len(predictions) == MAX_PREDICTIONS else "PARTIAL",
        "prediction_n": len(predictions), "failure_n": len(failures),
        "planned_n": MAX_PREDICTIONS, "unattempted_n": max(0, MAX_PREDICTIONS - len(predictions)),
        "unresolved_or_failed_prediction_n": sum(p.get("predicted_quantity") is None for p in predictions),
        "aggregate": result_metrics,
        "actual_http_max_overlap": maximum_http_overlap(transport_timings),
        "budget": {"requests": budget.requests, "estimated_bytes": budget.estimated_bytes,
            "max_in_flight_observed": budget.max_observed_in_flight,
            "request_cap": MAX_REQUESTS, "estimated_byte_cap": MAX_ESTIMATED_BYTES}}
    _write_json(output / "result.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Frozen GA10 procurement prediction study")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--synthetic", action="store_true", help="run only the generated mock SDK fixture")
    mode.add_argument("--preview", action="store_true", help="freeze a plan without provider setup")
    mode.add_argument("--execute", action="store_true", help="execute the authorized bounded human study")
    parser.add_argument("--source", type=Path, default=Path("C:/Users/hongy/Downloads/mnsc.2023.03771.zip"))
    parser.add_argument("--policy", type=Path, default=Path("artifacts/participant-data/quote-perspective-24/source-policy-authorized.json"))
    parser.add_argument("--output-dir", type=Path,
        default=Path("artifacts/participant-data/procurement-perspective-24"))
    args = parser.parse_args(argv)
    if args.synthetic:
        plan = build_synthetic_study()
        result = asyncio.run(execute_study(plan, args.output_dir, fake_client=_sdk_mock()))
        print(json.dumps({k: v for k, v in result.items() if k != "aggregate"}, indent=2))
        return 0 if result["status"] == "COMPLETE" else 2
    policy_path = args.policy.resolve()
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    manifest, episodes = load_paper_episodes(args.source,
        intended_use_authorized=policy.get("intended_use_authorized") is True,
        provider_transmission_permitted=policy.get("provider_transmission_permitted") is True,
        codebook_verified=policy.get("codebook_verified") is True)
    if (policy.get("source_kind") != "human" or policy.get("source_sha256") != manifest.source_sha256
            or manifest.source_sha256 != EXPECTED_SOURCE_SHA):
        raise ValueError("source policy and verified human-source checksum do not match")
    if args.preview:
        if not all((manifest.codebook_verified, manifest.intended_use_authorized,
                    manifest.provider_transmission_permitted)):
            print(json.dumps({"status": "BLOCKED", "reason": "source attestations are false"}))
            return 3
        plan = build_study(manifest, episodes)
        output = args.output_dir.resolve()
        artifact_root = (Path(__file__).resolve().parents[2] / "artifacts" / "participant-data").resolve()
        if output != artifact_root and artifact_root not in output.parents:
            raise ValueError("study outputs must be stored under participant-data")
        output.mkdir(parents=True, exist_ok=False)
        _write_frozen_plan(plan, output / "frozen-plan.json")
        print(json.dumps({"status": "FROZEN", "prediction_n": len(plan["_sessions"]),
            "request_cap": MAX_REQUESTS, "estimated_bytes": plan["planned_worst_path_estimated_bytes"]}))
        return 0
    if not all((manifest.codebook_verified, manifest.intended_use_authorized,
                manifest.provider_transmission_permitted)):
        print(json.dumps({"status": "BLOCKED", "reason": "source attestations are false; no provider initialized"}))
        return 3
    plan = build_study(manifest, episodes)
    result = asyncio.run(execute_study(plan, args.output_dir))
    print(json.dumps({k: v for k, v in result.items() if k != "aggregate"}, indent=2))
    return 0 if result["status"] == "COMPLETE" else 2

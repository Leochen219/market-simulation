"""Frozen selection, matching, scheduling and offline replay for history injection.

This module accepts already-loaded, authorized Episode objects. It never opens
the source workbook itself and keeps forecast labels separate from inference
inputs.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import asyncio
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from capclaims.injection_context import ARMS
from capclaims.injection_context import render_state
from capclaims.backends.jev import JevBackend, estimate_input_tokens, load_local_env, MODEL
from capclaims.backends.jev_instructions import build_instruction
from capclaims.personalization import Episode, SourceManifest
from capclaims.portfolio_prediction import (advance_trace, build_order_request, next_trace_action,
    replay_trace)
from capclaims.portfolio_transport import serialize_sdk_wire, DEFAULT_URL, _url_tuple
from capclaims.portfolio_transport import GuardTransport, SharedBudget
from capclaims.engine.protocol import DecisionReply
from capclaims.portfolio_study import _replay_backend
from capclaims.randomness import canonical_json

SEED = 20261025
SOURCE_SHA = "f9978ebebe862e4dab51a1885a5009b7d570da1a9e49e01d2f3becfe1af385a9"
MAX_SESSIONS = 40
MAX_MODEL_CALLS_PER_PORTFOLIO = 35  # 5 sides + 15 quantity digits + 15 price digits.
MAX_REQUESTS = 1400
MAX_RESERVED_BYTES = 10_500_000
MAX_BODY_BYTES = 7500
MAX_WALL_SECONDS = 600
MAX_RPS = 40
MAX_BYTES_PER_SECOND = 50_000
MAX_IN_FLIGHT = 10
HISTORY_ROUNDS = (6, 7, 8)
BASELINE_ROUNDS = tuple(range(1, 9))
TARGET_ROUND = 9
_ROOT = Path(__file__).resolve().parents[2]


def create_global_budget(ledger_path: str | Path, *, fake: bool = False) -> SharedBudget:
    """Construct the one shared all-arm budget; fake acceleration is explicit."""
    if MAX_REQUESTS != MAX_SESSIONS * MAX_MODEL_CALLS_PER_PORTFOLIO:
        raise AssertionError("global request cap must cover exactly the complete legal 40-session envelope")
    return SharedBudget(ledger_path, max_requests=MAX_REQUESTS, max_bytes=MAX_RESERVED_BYTES,
        rps=10_000 if fake else MAX_RPS,
        bytes_per_second=100_000_000 if fake else MAX_BYTES_PER_SECOND,
        max_in_flight=MAX_IN_FLIGHT)


def create_guarded_transport(inner: Any, budget: SharedBudget) -> GuardTransport:
    """Bind all arm requests to one immutable endpoint and global reservation ledger."""
    if budget.max_requests != MAX_REQUESTS or budget.max_bytes != MAX_RESERVED_BYTES:
        raise ValueError("history injection requires the full frozen global budget object")
    return GuardTransport(inner, DEFAULT_URL, budget, max_body_bytes=MAX_BODY_BYTES)


class InMemoryWireRecorder:
    """Capture forwarded raw responses for session binding without rereading growing JSONL files."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.rows: dict[str, dict[str, Any]] = {}

    async def handle_async_request(self, request: Any) -> Any:
        from capclaims.backends.jev import _CURRENT_REQUEST_ID
        request_id = _CURRENT_REQUEST_ID.get()
        request_body = await request.aread()
        response = await self.inner.handle_async_request(request)
        response_body = await response.aread()
        if request_id:
            self.rows[request_id] = {"request_id": request_id, "method": request.method,
                "url": str(request.url), "request_body_utf8": request_body.decode("utf-8"),
                "request_body_sha256": _sha_bytes(request_body),
                "response_body_utf8": response_body.decode("utf-8", errors="replace"),
                "response_body_sha256": _sha_bytes(response_body), "status": response.status_code}
        return response

    async def aclose(self) -> None:
        close = getattr(self.inner, "aclose", None)
        if close is not None:
            await close()


class InjectionBackend(JevBackend):
    """Session-local Jev view; the request envelope and candidate set stay original."""

    def __init__(self, *args: Any, arm: str, donor_history: Any = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if self.instruction_profile != "portfolio_third_v1" or self.objective != "portfolio_prediction_v1":
            raise ValueError("all history-injection arms require portfolio_third_v1 unchanged")
        self.arm = arm
        self.donor_history = donor_history

    async def decide_many(self, requests: Sequence[Any]) -> list[Any]:
        if self._closed:
            raise RuntimeError("Jev backend is closed")
        requests = tuple(requests)
        if not requests:
            return []
        if len({request.request_id for request in requests}) != len(requests):
            raise ValueError("duplicate Jev request IDs")
        rendered = [(request, build_instruction(request, "portfolio_third_v1"),
                     render_state(request, self.arm, donor_history=self.donor_history))
                    for request in requests]
        estimates = [estimate_input_tokens(request, prompt, state) for request, prompt, state in rendered]
        await self._reserve(requests, estimates)
        return list(await __import__("asyncio").gather(*(
            self._one(request, estimate, prompt, state)
            for (request, prompt, state), estimate in zip(rendered, estimates))))


def _one_hot_reply(request: Any, selected: Any) -> DecisionReply:
    candidate = next(c for c in request.candidates if c.value == selected)
    probabilities = {c.candidate_id: float(c.candidate_id == candidate.candidate_id) for c in request.candidates}
    return DecisionReply.from_request(request, probabilities)


def _maximal_legal_request(history: Sequence[Mapping[str, Any]]) -> Any:
    """Build a valid mixed-side slot-4 price-prefix state for closed byte preflight."""
    base = {"agent_id": "preflight-opaque", "round": 9, "treatment": "GA",
        "wholesale_price_dollars": 10, "p": 100, "Q": 400, "D": 200,
        "history": list(history)}
    trace: tuple[dict[str, Any], ...] = ()
    # Four accepted mixed-side orders consume 398 sell units; a final one-unit
    # pending sell is legal and leaves sell capacity to validate the ledger.
    steps = iter(("sell", 1, 9, 9, 9, 9, 9, "buy", 9, 9, 9, 9, 9, 9,
                  "sell", 1, 9, 9, 9, 9, 9, "buy", 9, 9, 9, 9, 9, 9,
                  "sell", 1, 9, 9))
    while True:
        action = next_trace_action(base, trace)
        if action["kind"] == "forced":
            trace = advance_trace(base, trace, None)
            continue
        request = build_order_request(base, trace)
        if request is None:
            raise AssertionError("closed envelope preflight unexpectedly reached terminal state")
        if request.stage.endswith("slot4:price:u"):
            return request
        selected = next(steps)
        trace = advance_trace(base, trace, _one_hot_reply(request, selected))


def _maximal_legal_requests(history: Sequence[Mapping[str, Any]]) -> tuple[Any, ...]:
    """Enumerate widest legal slot-four side, quantity, and price digit states."""
    base = {"agent_id": "preflight-opaque", "round": 9, "treatment": "GA",
        "wholesale_price_dollars": 10, "p": 100, "Q": 400, "D": 200,
        "history": list(history)}
    trace: tuple[dict[str, Any], ...] = ()
    prefix_choices = ("sell", 1, 9, 9, 9, 9, 9, "buy", 9, 9, 9, 9, 9, 9,
        "sell", 1, 9, 9, 9, 9, 9, "buy", 9, 9, 9, 9, 9, 9)
    for value in prefix_choices:
        request = build_order_request(base, trace)
        if request is None:
            action = next_trace_action(base, trace)
            if action["kind"] != "forced":
                raise AssertionError("maximal legal prefix unexpectedly terminated")
            trace = advance_trace(base, trace, None)
            request = build_order_request(base, trace)
        if request is None:
            raise AssertionError("maximal legal prefix request is absent")
        trace = advance_trace(base, trace, _one_hot_reply(request, value))
    side = build_order_request(base, trace)
    if side is None or side.stage != "paper:slot4:side":
        raise AssertionError("preflight did not reach slot-four side choice")
    requests = [side]
    for pending_side, quantity_digits in (("buy", [9, 9, 9]), ("sell", [2])):
        branch = advance_trace(base, trace, _one_hot_reply(side, pending_side))
        while True:
            action = next_trace_action(base, branch)
            if action["kind"] == "forced":
                branch = advance_trace(base, branch, None)
                continue
            request = build_order_request(base, branch)
            if request is None or not request.stage.startswith("paper:slot4:"):
                raise AssertionError("widest slot-four branch terminated unexpectedly")
            requests.append(request)
            if request.stage.endswith(":price:u"):
                break
            if ":qty:" in request.stage:
                selected = quantity_digits.pop(0)
            elif ":price:" in request.stage:
                selected = 9
            else:
                raise AssertionError("unexpected slot-four preflight phase")
            branch = advance_trace(base, branch, _one_hot_reply(request, selected))
    return tuple(requests)


async def preflight_max_body(sessions: Sequence[Mapping[str, Any]]) -> int:
    """Serialize all widest valid slot-four states for every session and arm."""
    maximum = 0
    for session in sessions:
        requests = _maximal_legal_requests(session["base"]["history"])
        for arm in ARMS:
            donor = session["donor_history"] if arm == "D" else None
            for request in requests:
                state = render_state(request, arm, donor_history=donor)
                wire = await serialize_sdk_wire(request, "portfolio_third_v1", state)
                maximum = max(maximum, len(wire.body))
    if maximum > MAX_BODY_BYTES:
        raise ValueError("valid widest request variant exceeds the frozen 7500-byte body cap")
    return maximum


def _sha(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _max_overlap(windows: Sequence[tuple[float, float]]) -> int:
    events = []
    for start, finish in windows:
        events.extend(((start, 1), (finish, -1)))
    active = peak = 0
    for _time, delta in sorted(events, key=lambda item: (item[0], item[1])):
        active += delta
        peak = max(peak, active)
    return peak


def choose_targets(roster: Sequence[str], excluded_ids: set[str], *, seed: int = SEED) -> tuple[str, ...]:
    if len(roster) != 35 or len(set(roster)) != 35 or any(not isinstance(x, str) or not x for x in roster):
        raise ValueError("GA10 roster must contain 35 unique participant identifiers")
    if len(excluded_ids) != 6 or not excluded_ids <= set(roster):
        raise ValueError("exactly six known prior-study participants must be excluded")
    eligible = [person for person in roster if person not in excluded_ids]
    return tuple(sorted(eligible, key=lambda p: hashlib.sha256(f"{seed}:{p}".encode()).digest())[:6])


def _history_key(record: Mapping[str, Any]) -> tuple[int, ...]:
    rows = record.get("history")
    if not isinstance(rows, (tuple, list)) or len(rows) != 3:
        raise ValueError("each participant needs exactly three frozen history episodes")
    rounds = tuple(row.get("round") for row in rows)
    if rounds != HISTORY_ROUNDS:
        raise ValueError("history must be exactly formal rounds 6, 7, and 8")
    return rounds


def match_donors(targets: Sequence[Mapping[str, Any]], donors: Sequence[Mapping[str, Any]], *,
                 seed: int = SEED) -> tuple[dict[str, Any], ...]:
    """Greedy one-to-one match using only historic p and D; preserve donor records."""
    if len(targets) != 6 or len(donors) < 6:
        raise ValueError("donor assignment requires six targets and at least six eligible donors")
    ids = [x.get("participant_id") for x in targets + donors]
    if any(not isinstance(x, str) or not x for x in ids) or len(ids) != len(set(ids)):
        raise ValueError("target and donor identifiers must be unique and disjoint")
    for row in targets + donors:
        _history_key(row)
    target_order = sorted(targets, key=lambda x: hashlib.sha256(f"{seed}:{x['participant_id']}".encode()).digest())
    available = {x["participant_id"]: x for x in donors}
    pairs = []
    for target in target_order:
        choices = []
        for donor_id, donor in available.items():
            distance = sum(abs(a["p"] - b["p"]) / 49 + abs(a["D"] - b["D"]) / 200
                for a, b in zip(target["history"], donor["history"])) / 3
            tie = hashlib.sha256(f"{seed}:{target['participant_id']}:{donor_id}".encode()).digest()
            choices.append((distance, tie, donor_id))
        distance, _, donor_id = min(choices)
        pairs.append({"target_id": target["participant_id"], "donor_id": donor_id,
                      "distance": distance})
        del available[donor_id]
    return tuple(pairs)


def make_schedule(people: Sequence[str], *, seed: int = SEED) -> tuple[dict[str, Any], ...]:
    if len(people) != 6 or len(set(people)) != 6:
        raise ValueError("schedule needs six distinct selected participants")
    rows = [{"person_id": person, "arm": arm, "repeat": repeat}
            for index, person in enumerate(people) for arm in ARMS
            for repeat in range(2 if index < 2 else 1)]
    random.Random(seed).shuffle(rows)
    if len(rows) != MAX_SESSIONS:
        raise AssertionError("frozen schedule must contain 40 sessions")
    for i, row in enumerate(rows):
        row["session_id"] = f"session-{i + 1:02d}"
    return tuple(rows)


def resolve_prior_exclusions(procurement_plan: Mapping[str, Any], portfolio_plan: Mapping[str, Any],
                             roster: Sequence[str], *,
                             manifest_sha256: Mapping[str, str]) -> tuple[set[str], dict[str, str]]:
    """Verify both prior study manifests against the current roster/source and return their exact six IDs."""
    roster_hash = {hashlib.sha256(person.encode()).hexdigest(): person for person in roster}
    if (len(roster) != 35 or len(set(roster)) != 35
            or procurement_plan.get("source_sha256") != SOURCE_SHA
            or portfolio_plan.get("source_sha256") != SOURCE_SHA):
        raise ValueError("both exclusion manifests must bind the current 35-person GA10 source roster")
    procurement_ids = procurement_plan.get("selected_participants")
    # The frozen procurement pilot stores SHA256 participant IDs. Resolve
    # those only against the exact, source-bound roster supplied by the
    # caller; never treat an unrecognized token as a raw participant ID.
    if not isinstance(procurement_ids, list) or len(procurement_ids) != 3:
        raise ValueError("procurement exclusion manifest must identify exactly three current roster members")
    if len(set(procurement_ids)) != 3:
        raise ValueError("procurement exclusion manifest contains duplicate participants")
    if set(procurement_ids) <= set(roster):
        procurement_resolved = set(procurement_ids)
    elif set(procurement_ids) <= set(roster_hash):
        procurement_resolved = {roster_hash[value] for value in procurement_ids}
    else:
        raise ValueError("procurement exclusion IDs must resolve uniquely against the current roster")
    portfolio_hashes = portfolio_plan.get("selected_participant_hashes")
    if (not isinstance(portfolio_hashes, list) or len(portfolio_hashes) != 3
            or len(set(portfolio_hashes)) != 3 or not set(portfolio_hashes) <= set(roster_hash)):
        raise ValueError("portfolio exclusion manifest must hash exactly three current roster members")
    portfolio_ids = {roster_hash[value] for value in portfolio_hashes}
    if procurement_resolved & portfolio_ids:
        raise ValueError("the two frozen prior pilots must have disjoint participants")
    excluded = procurement_resolved | portfolio_ids
    if len(excluded) != 6:
        raise ValueError("prior-pilot exclusion union must contain exactly six participants")
    if (set(manifest_sha256) != {"procurement", "portfolio"}
            or any(not isinstance(value, str) or len(value) != 64
                or any(ch not in "0123456789abcdef" for ch in value)
                for value in manifest_sha256.values())):
        raise ValueError("exact-byte hashes for both exclusion manifest files are required")
    return excluded, dict(manifest_sha256)


def _episode_order(action: Mapping[str, Any], initial_q: int, *, owner: str) -> list[dict[str, Any]]:
    orders = action.get("orders")
    if not isinstance(orders, (list, tuple)) or len(orders) > 5:
        raise ValueError("portfolio target must contain at most five source orders")
    result = []
    for index, item in enumerate(orders):
        if not isinstance(item, Mapping) or set(item) != {"type", "amount", "price"}:
            raise ValueError("source order must contain type, amount, and price")
        result.append({"type": item["type"], "amount": item["amount"], "price": item["price"]})
    from capclaims.paper.orders import PaperOrder, validate_portfolio
    legal = [PaperOrder(order_id=f"{owner}-order-{index}", owner=owner,
        side=order["type"].lower(), qty=order["amount"], limit_dollars=order["price"])
        for index, order in enumerate(result)]
    validate_portfolio(legal, owner, initial_q)
    return result


def _matching_record(person: str, rounds: Mapping[int, Episode]) -> dict[str, Any]:
    """Return only the allowed matching variables: history round, p, and D."""
    history = []
    for round_no in HISTORY_ROUNDS:
        episode = rounds.get(round_no)
        if episode is None:
            raise ValueError(f"participant {person} lacks history round {round_no}")
        history.append({"round": round_no, "p": episode.observed["p"], "D": episode.observed["D"]})
    if rounds.get(TARGET_ROUND) is None:
        raise ValueError(f"participant {person} lacks target round 9")
    return {"participant_id": person, "history": history}


def _full_history(person: str, rounds: Mapping[int, Episode]) -> list[dict[str, Any]]:
    history = []
    for round_no in HISTORY_ROUNDS:
        episode = rounds.get(round_no)
        if episode is None:
            raise ValueError(f"participant {person} lacks history round {round_no}")
        history.append({"round": round_no, "p": episode.observed["p"], "Q": episode.observed["Q"],
                        "D": episode.observed["D"], "orders": _episode_order(episode.action,
                            episode.observed["Q"], owner=f"{person}-{round_no}")})
    if rounds.get(TARGET_ROUND) is None:
        raise ValueError(f"participant {person} lacks target round 9")
    return history


def _baseline_history(person: str, rounds: Mapping[int, Episode]) -> list[dict[str, Any]]:
    history = []
    for round_no in BASELINE_ROUNDS:
        episode = rounds.get(round_no)
        if episode is None:
            raise ValueError(f"participant {person} lacks offline baseline round {round_no}")
        history.append({"round": round_no, "orders": _episode_order(episode.action,
            episode.observed["Q"], owner=f"{person}-{round_no}")})
    return history


def _private_dir(path: str | Path) -> Path:
    target = Path(path).resolve()
    root = (_ROOT / "artifacts" / "participant-data").resolve()
    if target != root and root not in target.parents:
        raise ValueError("study outputs must be stored under ignored artifacts/participant-data")
    return target


def build_preview(source: SourceManifest, episodes: Sequence[Episode], roster: Sequence[str],
                  excluded_ids: set[str], output_dir: str | Path, *,
                  source_policy_sha256: str, exclusion_manifest_sha256: Mapping[str, str],
                  seed: int = SEED) -> dict[str, Any]:
    """Build an immutable private input/label split from already-authorized episodes."""
    if (source.source_kind != "human" or source.source_sha256 != SOURCE_SHA
            or not source.codebook_verified or not source.intended_use_authorized
            or not source.provider_transmission_permitted):
        raise ValueError("source manifest does not authorize this verified human-data study")
    if (not isinstance(source_policy_sha256, str) or len(source_policy_sha256) != 64
            or any(ch not in "0123456789abcdef" for ch in source_policy_sha256)):
        raise ValueError("source policy SHA256 is required before preview freeze")
    if (set(exclusion_manifest_sha256) != {"procurement", "portfolio"}
            or any(not isinstance(value, str) or len(value) != 64
                   or any(ch not in "0123456789abcdef" for ch in value)
                   for value in exclusion_manifest_sha256.values())):
        raise ValueError("hashes for both frozen prior-pilot exclusion manifests are required")
    people = choose_targets(roster, excluded_ids, seed=seed)
    eligible = [p for p in roster if p not in excluded_ids]
    indexed: dict[str, dict[int, Episode]] = {p: {} for p in eligible}
    for episode in episodes:
        if episode.scope == "GA 10" and episode.treatment == "GA" and episode.phase == "orders" \
                and episode.participant_id in indexed and episode.round in (*BASELINE_ROUNDS, TARGET_ROUND):
            if episode.round in indexed[episode.participant_id]:
                raise ValueError("duplicate participant-round input")
            indexed[episode.participant_id][episode.round] = episode
    matching_records = {p: _matching_record(p, indexed[p]) for p in eligible}
    # Only the pre-trading target state is read before matching. Current target
    # orders stay untouched until target selection and donor assignment finish.
    target_states = {p: {key: indexed[p][TARGET_ROUND].observed[key]
        for key in ("p", "w", "Q", "D")} for p in eligible}
    targets = [matching_records[p] for p in people]
    donor_records = [matching_records[p] for p in eligible if p not in set(people)]
    if len(donor_records) != 23:
        raise ValueError("after six exclusions and six targets, the unused donor pool must contain 23 people")
    matches = match_donors(targets, donor_records, seed=seed)
    schedule = make_schedule(people, seed=seed)
    donor_ids = {row["donor_id"] for row in matches}
    full_history_by_person = {person: _full_history(person, indexed[person])
        for person in set(people) | donor_ids}
    baseline_records = {person: _baseline_history(person, indexed[person])
        for person in set(people) | donor_ids}
    donor_by_target = {row["target_id"]: full_history_by_person[row["donor_id"]] for row in matches}
    sessions = []
    for scheduled in schedule:
        target = target_states[scheduled["person_id"]]
        opaque_id = hashlib.sha256(f"{seed}:{scheduled['session_id']}:{scheduled['person_id']}".encode()).hexdigest()[:24]
        sessions.append({**scheduled, "base": {"agent_id": f"inj-{opaque_id}",
            "round": 9, "treatment": "GA", "wholesale_price_dollars": target["w"],
            "p": target["p"], "Q": target["Q"], "D": target["D"],
            "history": full_history_by_person[scheduled["person_id"]]},
            "donor_history": donor_by_target[scheduled["person_id"]]})
    target_labels = {}
    for person in people:
        episode = indexed[person][TARGET_ROUND]
        target_labels[person] = {**target_states[person], "orders": _episode_order(episode.action,
            episode.observed["Q"], owner=f"{person}-{TARGET_ROUND}")}
    target_inputs = {p: {"history": full_history_by_person[p]} for p in people}
    maximum_body = asyncio.run(preflight_max_body(sessions))
    prompt_text = build_instruction(_maximal_legal_request(sessions[0]["base"]["history"]), "portfolio_third_v1")
    frozen = {"study": "jev_history_injection_v1", "seed": seed, "source": source.__dict__,
        "source_policy_sha256": source_policy_sha256,
        "exclusion_manifest_sha256": dict(exclusion_manifest_sha256),
        "model": MODEL, "sdk_version": _sdk_version(),
        "instruction_profile": "portfolio_third_v1", "prompt_text": prompt_text,
        "prompt_sha256": _sha_bytes(prompt_text.encode("utf-8")),
        "selected_people": list(people), "exclusions": sorted(excluded_ids), "donor_matches": list(matches),
        "sessions": sessions, "request_cap": MAX_REQUESTS, "reserved_bytes_cap": MAX_RESERVED_BYTES,
        "body_bytes_cap": MAX_BODY_BYTES, "preflight_max_body_bytes": maximum_body,
        "preflight_widest_variant_n": 11,
        "wall_seconds_cap": MAX_WALL_SECONDS,
        "rps_cap": MAX_RPS, "reserved_bytes_per_second_cap": MAX_BYTES_PER_SECOND,
        "in_flight_cap": MAX_IN_FLIGHT, "retry_count": 0,
        "code_sha256": _code_hashes()}
    frozen["input_sha256"] = _sha(target_inputs)
    frozen["labels_sha256"] = _sha(target_labels)
    frozen["inputs_file_sha256"] = _sha(sessions)
    frozen["labels_file_sha256"] = _sha(target_labels)
    frozen["baselines_file_sha256"] = _sha(baseline_records)
    frozen["preview_sha256"] = _sha(frozen)
    target = _private_dir(output_dir)
    if target.exists():
        raise FileExistsError("preview output already exists; overwrite/resume is forbidden")
    target.mkdir(parents=True, exist_ok=False)
    _write(target / "manifest.json", frozen)
    _write(target / "inputs.json", {"sessions": sessions})
    _write(target / "labels.json", target_labels)
    _write(target / "baselines.json", baseline_records)
    return frozen


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as f:
        f.write(canonical_json(value))
        f.flush()
        os.fsync(f.fileno())


def _write_derived_idempotent(path: Path, value: Any) -> None:
    """Permit repeat scoring only when the already-derived private result is identical."""
    payload = canonical_json(value).encode("utf-8")
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError("existing derived score differs; raw run evidence is immutable")
        return
    _write(path, value)


def _code_hashes() -> dict[str, str]:
    frozen_paths = ("src/capclaims/injection_context.py", "src/capclaims/injection_study.py",
        "src/capclaims/injection_analysis.py", "src/capclaims/portfolio_prediction.py",
        "src/capclaims/portfolio_metrics.py", "src/capclaims/portfolio_transport.py",
        "src/capclaims/portfolio_study.py", "src/capclaims/backends/jev.py",
        "src/capclaims/backends/jev_instructions.py", "src/capclaims/engine/protocol.py",
        "src/capclaims/paper/codec.py", "src/capclaims/paper/interface.py",
        "src/capclaims/paper/observations.py", "src/capclaims/paper/orders.py",
        "src/capclaims/randomness.py", "src/capclaims/personalization_xlsx.py",
        "tools/probes/jev_portfolio_prediction.py", "tools/probes/jev_history_injection.py",
        "docs/superpowers/specs/2026-10-25-history-injection-design.md")
    return {name: hashlib.sha256((_ROOT / name).read_bytes()).hexdigest() for name in frozen_paths}


def _sdk_version() -> str:
    from importlib.metadata import version
    return version("typesafe-sdk")


def verify_preview(preview_dir: str | Path, reviewed_sha256: str) -> dict[str, Any]:
    root = Path(preview_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    expected = manifest.pop("preview_sha256", None)
    actual = _sha(manifest)
    manifest["preview_sha256"] = expected
    if not expected or expected != actual or reviewed_sha256 != expected:
        raise ValueError("preview digest is invalid or has not been independently reviewed")
    if (manifest.get("code_sha256") != _code_hashes() or manifest.get("sdk_version") != "0.7.5"
            or _sdk_version() != "0.7.5"):
        raise ValueError("study code or pinned SDK changed after preview freeze")
    if (manifest.get("model") != MODEL or manifest.get("instruction_profile") != "portfolio_third_v1"
            or manifest.get("prompt_text") != build_instruction(
                _maximal_legal_request(manifest["sessions"][0]["base"]["history"]), "portfolio_third_v1")
            or manifest.get("prompt_sha256") != _sha_bytes(manifest["prompt_text"].encode("utf-8"))):
        raise ValueError("frozen model, prompt profile, or prompt text changed")
    inputs = json.loads((root / "inputs.json").read_text(encoding="utf-8"))
    if (inputs.get("sessions") != manifest.get("sessions")
            or _sha(inputs.get("sessions")) != manifest.get("inputs_file_sha256")):
        raise ValueError("frozen inference input file changed after preview")
    if (len(manifest.get("sessions", [])) != 40 or manifest.get("request_cap") != MAX_REQUESTS
            or manifest.get("reserved_bytes_cap") != MAX_RESERVED_BYTES
            or manifest.get("body_bytes_cap") != MAX_BODY_BYTES
            or manifest.get("wall_seconds_cap") != MAX_WALL_SECONDS
            or manifest.get("preflight_widest_variant_n") != 11
            or manifest.get("preflight_max_body_bytes", MAX_BODY_BYTES + 1) > MAX_BODY_BYTES):
        raise ValueError("preview does not satisfy the frozen study envelope")
    if (len(manifest.get("selected_people", [])) != 6 or len(set(manifest["selected_people"])) != 6
            or len(manifest.get("donor_matches", [])) != 6
            or len({row.get("donor_id") for row in manifest["donor_matches"]}) != 6
            or len(manifest.get("exclusions", [])) != 6):
        raise ValueError("preview selection or one-to-one donor assignment is malformed")
    return manifest


async def execute_preview(preview_dir: str | Path, run_dir: str | Path, *,
                          reviewed_preview_sha256: str, fake: bool = False,
                          fake_transport: Any = None) -> dict[str, Any]:
    """Execute one all-or-none 40-session run after an explicit preview review.

    The receipt is durable before reading credential environment or constructing
    an SDK client. Fake mode is a test-only explicit pacing override.
    """
    manifest = verify_preview(preview_dir, reviewed_preview_sha256)
    if not all(manifest.get("source", {}).get(key) is True for key in
               ("codebook_verified", "intended_use_authorized", "provider_transmission_permitted")):
        raise ValueError("source policy attestations block provider execution")
    if manifest["source"].get("source_kind") != "human" or manifest.get("source_policy_sha256") is None:
        raise ValueError("live execution requires an explicitly frozen authorized source policy")
    target = _private_dir(run_dir)
    if target.exists():
        raise FileExistsError("run directory already exists; resume and overwrite are forbidden")
    target.mkdir(parents=True, exist_ok=False)
    started_monotonic = time.monotonic()
    started_utc = datetime.now(timezone.utc).isoformat()
    receipt = {"study": manifest["study"], "preview_sha256": manifest["preview_sha256"],
        "execution_mode": "mock" if fake else "provider", "status": "RUNNING",
        "request_cap": MAX_REQUESTS, "reserved_bytes_cap": MAX_RESERVED_BYTES,
        "wall_seconds_cap": MAX_WALL_SECONDS, "retry_count": 0,
        "started_utc": started_utc, "started_monotonic": started_monotonic}
    _write(target / "receipt.json", receipt)
    attempts = target / "attempts.jsonl"
    with attempts.open("x", encoding="utf-8", newline="\n") as f:
        f.flush()
        os.fsync(f.fileno())

    # Credential and client access begins only after the fsynced receipt.
    if fake:
        api_key = "offline-dummy-key"
        inner = fake_transport if fake_transport is not None else _mock_transport()
    else:
        load_local_env()
        api_key = os.getenv("TYPESAFE_API_KEY")
        if not api_key:
            raise ValueError("TYPESAFE_API_KEY is unavailable after the durable pre-provider receipt")
        import httpx2
        inner = httpx2.AsyncHTTPTransport(retries=0)

    import httpx2
    from typesafe_sdk import AsyncTypeSafeClient
    from typesafe_sdk._core.retry import RetryPolicy
    budget = create_global_budget(attempts, fake=fake)
    memory_wire = InMemoryWireRecorder(inner)
    guard = create_guarded_transport(memory_wire, budget)
    http = httpx2.AsyncClient(transport=guard, follow_redirects=False)
    client = AsyncTypeSafeClient(api_key=api_key, model=MODEL,
        retry=RetryPolicy(max_retries=0), timeout=30, http_client=http,
        base_url="https://api.typesafe.ai", compression=False)
    backend_by_session = {row["session_id"]: InjectionBackend(api_key=api_key,
        rps=MAX_RPS, tps=100_000, max_requests=MAX_REQUESTS,
        max_estimated_input_tokens=MAX_REQUESTS * MAX_BODY_BYTES,
        instruction_profile="portfolio_third_v1", objective="portfolio_prediction_v1",
        request_timeout=30, max_in_flight=MAX_IN_FLIGHT, client=client,
        arm=row["arm"], donor_history=row.get("donor_history"), load_environment=False)
        for row in manifest["sessions"]}
    inputs = json.loads((Path(preview_dir) / "inputs.json").read_text(encoding="utf-8"))
    sessions_by_id = {row["session_id"]: row for row in inputs["sessions"]}
    session_results: dict[str, dict[str, Any]] = {}
    call_records: dict[str, dict[str, Any]] = {}
    started = started_monotonic

    async def run_session(session: Mapping[str, Any]) -> dict[str, Any]:
        trace: tuple[dict[str, Any], ...] = ()
        call_count = 0
        base = session["base"]
        backend = backend_by_session[session["session_id"]]
        while True:
            action = next_trace_action(base, trace)
            if action["kind"] == "complete":
                return {"session_id": session["session_id"], "status": "complete", "trace": list(trace),
                        "wire_calls": [row for row in call_records.values()
                            if row.get("_session_id") == session["session_id"]]}
            if action["kind"] == "incomplete":
                if action.get("reason") == "unresolved_tie":
                    return {"session_id": session["session_id"], "status": "verified_tie", "trace": list(trace),
                            "wire_calls": [row for row in call_records.values()
                                if row.get("_session_id") == session["session_id"]]}
                raise RuntimeError("technical failure invalidates the full study")
            if action["kind"] == "forced":
                trace = advance_trace(base, trace, None)
                continue
            request = build_order_request(base, trace)
            if request is None:
                raise AssertionError("expected Jev request could not be reconstructed")
            call_count += 1
            if call_count > MAX_MODEL_CALLS_PER_PORTFOLIO:
                raise RuntimeError("portfolio exceeded the proven 35-call full-action upper bound")
            effective_state = render_state(request, session["arm"],
                donor_history=session.get("donor_history"))
            wire = await serialize_sdk_wire(request, "portfolio_third_v1", effective_state)
            guard.register(wire)
            (reply,) = await backend.decide_many([request])
            proposed = advance_trace(base, trace, reply)
            # Bind the parsed reply to the exact body forwarded below the guard.
            body = memory_wire.rows.get(request.request_id)
            timing = next((row for row in reversed(guard.http_timings)
                if row.get("request_id") == request.request_id), None)
            if body is None or timing is None:
                raise RuntimeError("successful provider reply lacks fsynced raw transport evidence")
            call_records[request.request_id] = {"_session_id": session["session_id"],
                "request_id": request.request_id, "method": body["method"], "url": body["url"],
                "request_body_utf8": body["request_body_utf8"],
                "request_body_sha256": body["request_body_sha256"],
                "response_body_utf8": body["response_body_utf8"],
                "response_body_sha256": body["response_body_sha256"],
                "status": body["status"], "timing": timing,
                "reply": {"request_id": reply.request_id, "agent_id": reply.agent_id,
                    "round_index": reply.round_index, "stage": reply.stage,
                    "observation_hash": reply.observation_hash,
                    "candidate_set_hash": reply.candidate_set_hash,
                    "raw_scores": dict(reply.raw_scores), "probabilities": dict(reply.probabilities),
                    "backend": reply.backend, "model_revision": reply.model_revision,
                    "usage": dict(reply.usage), "timing": dict(reply.timing)}}
            trace = proposed

    overall_error = None
    try:
        all_sessions = list(manifest["sessions"])
        for start in range(0, len(all_sessions), MAX_IN_FLIGHT):
            if time.monotonic() - started >= MAX_WALL_SECONDS:
                overall_error = "wall_clock_limit"
                break
            wave = all_sessions[start:start + MAX_IN_FLIGHT]
            results = await asyncio.wait_for(asyncio.gather(*(run_session(row) for row in wave),
                return_exceptions=True), timeout=max(0.1, MAX_WALL_SECONDS - (time.monotonic() - started)))
            failed = False
            for row, result in zip(wave, results):
                if isinstance(result, BaseException):
                    session_results[row["session_id"]] = {"session_id": row["session_id"],
                        "status": "invalid", "error_type": type(result).__name__}
                    overall_error = type(result).__name__
                    failed = True
                else:
                    session_results[row["session_id"]] = result
            if failed:
                break
    except asyncio.TimeoutError:
        overall_error = "wall_clock_limit"
    finally:
        for backend in backend_by_session.values():
            backend._closed = True
        await client.aclose()

    results = []
    for session in manifest["sessions"]:
        row = session_results.get(session["session_id"])
        if row is None:
            row = {"session_id": session["session_id"], "status": "unattempted"}
        else:
            row = {key: value for key, value in row.items() if key != "wire_calls"}
            row["wire_calls"] = [dict(call, _session_id=None) for call in call_records.values()
                if call.get("_session_id") == session["session_id"]]
        results.append(row)
    output_events = [{"session_id": row["session_id"], "trace": row.get("trace", []),
        "wire_calls": row.get("wire_calls", [])} for row in results
        if row.get("status") in {"complete", "verified_tie"}]
    _write(target / "events.json", output_events)
    chain_rows = []
    previous = "0" * 64
    for row in output_events:
        digest = _sha({"previous_sha256": previous, "event": row})
        chain_rows.append({"previous_sha256": previous, "event": row, "sha256": digest})
        previous = digest
    _write(target / "events-chain.json", chain_rows)
    transport_paths = {name: target / name for name in
        ("transport-requests.jsonl", "transport-bodies.jsonl")}
    for path in transport_paths.values():
        if not path.exists():
            with path.open("x", encoding="utf-8", newline="\n") as f:
                f.flush()
                os.fsync(f.fileno())
    _write(target / "transport-timings.json", guard.http_timings)
    terminal = {"schema": "jev_history_injection_terminal_v1",
        "execution_mode": "mock" if fake else "provider",
        "preview_sha256": manifest["preview_sha256"],
        "session_statuses": [{"session_id": row["session_id"], "status": row["status"]}
                              for row in results],
        "events_sha256": _sha_bytes((target / "events.json").read_bytes()),
        "events_chain_sha256": _sha_bytes((target / "events-chain.json").read_bytes()),
        "events_chain_head": previous,
        "attempts_sha256": _sha_bytes(attempts.read_bytes()),
        "requests_sha256": _sha_bytes(transport_paths["transport-requests.jsonl"].read_bytes()),
        "bodies_sha256": _sha_bytes(transport_paths["transport-bodies.jsonl"].read_bytes()),
        "timings_sha256": _sha_bytes((target / "transport-timings.json").read_bytes()),
        "request_n": budget.usage["requests_charged"],
        "reserved_bytes": budget.usage["reserved_bytes"],
        "attempt_line_n": len([row for row in attempts.read_text(encoding="utf-8").splitlines() if row]),
        "request_line_n": len([row for row in transport_paths["transport-requests.jsonl"].read_text(encoding="utf-8").splitlines() if row]),
        "body_line_n": len([row for row in transport_paths["transport-bodies.jsonl"].read_text(encoding="utf-8").splitlines() if row]),
        "timing_n": len(guard.http_timings),
        "receipt_sha256": _sha_bytes((target / "receipt.json").read_bytes()),
        "sdk_version": _sdk_version(),
        "source_policy_sha256": manifest["source_policy_sha256"],
        "status": "INVALID" if (overall_error or len(results) != MAX_SESSIONS
            or any(row["status"] not in {"complete", "verified_tie"} for row in results)) else "COMPLETE"}
    _write(target / "terminal.json", terminal)
    result = {"status": terminal["status"],
        "reason": overall_error, "planned_sessions": MAX_SESSIONS,
        "attempted_sessions": sum(row["status"] in {"complete", "verified_tie", "invalid"} for row in results),
        "unattempted_sessions": sum(row["status"] == "unattempted" for row in results),
        "requests_charged": budget.usage["requests_charged"],
        "reserved_bytes": budget.usage["reserved_bytes"],
        "sessions": [{"session_id": row["session_id"], "status": row["status"]} for row in results]}
    _write(target / "result.json", result)
    return result


def _mock_transport() -> Any:
    import httpx2
    async def choose_first(request: Any) -> Any:
        body = json.loads((await request.aread()).decode("utf-8"))
        criteria = body["questions"]["decision"]["criteria"]
        stop_id = next((key for key in criteria if "stop" in key.lower()), None)
        chosen = stop_id if stop_id is not None else next(iter(criteria))
        return httpx2.Response(200, json={"model": MODEL,
            "usage": {"input_tokens": 1, "output_tokens": 1},
            "answers": {"decision": {"type": "choice", "choice": chosen,
                "confidence": 1.0, "probabilities": {key: float(key == chosen) for key in criteria}}}})
    return httpx2.MockTransport(choose_first)


def replay_run(preview_dir: str | Path, run_dir: str | Path) -> dict[str, Any]:
    """Reconstruct all effective SDK wires, replies, choices and legal transitions.

    events.json is a private session list. Each session has trace (the normal
    transition events) and wire_calls, a request-id keyed list preserving the
    actual body, response body, HTTP status/timing, and parsed DecisionReply.
    """
    preview_file = json.loads((Path(preview_dir) / "manifest.json").read_text(encoding="utf-8"))
    preview = verify_preview(preview_dir, preview_file.get("preview_sha256", ""))
    run_path = Path(run_dir) / "events.json"
    events = json.loads(run_path.read_text(encoding="utf-8"))
    if len(events) != len(preview["sessions"]):
        raise ValueError("run is missing one or more planned session traces")
    run_root = Path(run_dir)
    terminal = json.loads((run_root / "terminal.json").read_text(encoding="utf-8"))
    expected_files = {"events_sha256": "events.json", "events_chain_sha256": "events-chain.json",
        "attempts_sha256": "attempts.jsonl", "requests_sha256": "transport-requests.jsonl",
        "bodies_sha256": "transport-bodies.jsonl", "timings_sha256": "transport-timings.json"}
    if (terminal.get("schema") != "jev_history_injection_terminal_v1"
            or terminal.get("status") != "COMPLETE"
            or terminal.get("preview_sha256") != preview.get("preview_sha256")
            or terminal.get("sdk_version") != "0.7.5" or _sdk_version() != "0.7.5"):
        raise ValueError("run terminal is missing, invalid, incomplete, or bound to another preview")
    for digest_field, filename in expected_files.items():
        if _sha_bytes((run_root / filename).read_bytes()) != terminal.get(digest_field):
            raise ValueError(f"terminal digest mismatch for {filename}")
    receipt_raw = (run_root / "receipt.json").read_bytes()
    receipt = json.loads(receipt_raw)
    try:
        receipt_utc = datetime.fromisoformat(receipt["started_utc"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("pre-provider receipt lacks a valid UTC start time") from exc
    receipt_monotonic = receipt.get("started_monotonic")
    if (receipt.get("preview_sha256") != preview.get("preview_sha256")
            or receipt.get("retry_count") != 0 or receipt_utc.tzinfo is None
            or isinstance(receipt_monotonic, bool) or not isinstance(receipt_monotonic, (int, float))
            or not math.isfinite(receipt_monotonic)
            or terminal.get("receipt_sha256") != _sha_bytes(receipt_raw)):
        raise ValueError("run receipt was not durably bound to the preview and execution envelope")
    chain = json.loads((run_root / "events-chain.json").read_text(encoding="utf-8"))
    previous = "0" * 64
    if len(chain) != len(events):
        raise ValueError("event hash chain length differs from session events")
    for row, event in zip(chain, events):
        expected = _sha({"previous_sha256": previous, "event": row.get("event")})
        if row.get("previous_sha256") != previous or row.get("sha256") != expected or row.get("event") != event:
            raise ValueError("session event hash chain is invalid or tampered")
        previous = expected
    if previous != terminal.get("events_chain_head"):
        raise ValueError("terminal event chain head differs from recomputed chain")
    terminal_statuses = terminal.get("session_statuses", [])
    if (len(terminal_statuses) != MAX_SESSIONS
            or [row.get("session_id") for row in terminal_statuses]
                != [row["session_id"] for row in preview["sessions"]]
            or any(row.get("status") not in {"complete", "verified_tie"} for row in terminal_statuses)):
        raise ValueError("terminal does not bind exactly 40 complete or verified-tie session statuses")
    terminal_status_by_id = {row["session_id"]: row["status"] for row in terminal_statuses}
    attempts = [json.loads(line) for line in (run_root / "attempts.jsonl").read_text(encoding="utf-8").splitlines() if line]
    requests = [json.loads(line) for line in (run_root / "transport-requests.jsonl").read_text(encoding="utf-8").splitlines() if line]
    bodies = [json.loads(line) for line in (run_root / "transport-bodies.jsonl").read_text(encoding="utf-8").splitlines() if line]
    timings = json.loads((run_root / "transport-timings.json").read_text(encoding="utf-8"))
    if (len(attempts) != terminal.get("request_n") or len(attempts) > MAX_REQUESTS
            or len(attempts) != terminal.get("attempt_line_n")
            or len(requests) != terminal.get("request_line_n")
            or len(bodies) != terminal.get("body_line_n")
            or len(timings) != terminal.get("timing_n")
            or terminal.get("reserved_bytes") != len(attempts) * MAX_BODY_BYTES
            or terminal.get("reserved_bytes", MAX_RESERVED_BYTES + 1) > MAX_RESERVED_BYTES):
        raise ValueError("global request or reserved-byte budget evidence is invalid")
    attempt_by_id = {row.get("request_id"): row for row in attempts}
    request_by_id = {row.get("request_id"): row for row in requests}
    body_by_id = {row.get("request_id"): row for row in bodies}
    timing_by_id = {row.get("request_id"): row for row in timings}
    if (None in attempt_by_id or len(attempt_by_id) != len(attempts)
            or set(attempt_by_id) != set(request_by_id) or set(attempt_by_id) != set(body_by_id)
            or set(attempt_by_id) != set(timing_by_id)):
        raise ValueError("charged requests, raw bodies, and HTTP timing evidence do not fully join")
    if ([row.get("seq") for row in attempts] != list(range(1, len(attempts) + 1))
            or any(row.get("reserved_bytes") != MAX_BODY_BYTES or row.get("method") != "POST"
                or _url_tuple(row.get("url", "")) != _url_tuple(DEFAULT_URL)
                or row.get("expected_sha256") != row.get("actual_sha256") for row in attempts)):
        raise ValueError("attempt ledger violates sequence, endpoint, or reserved-byte invariants")
    timing_windows = []
    for request_id, attempt in attempt_by_id.items():
        request_row, body_row = request_by_id[request_id], body_by_id[request_id]
        raw_request = request_row.get("body_utf8", "").encode("utf-8")
        raw_response = body_row.get("response_body_utf8")
        if (attempt.get("actual_sha256") != request_row.get("body_sha256")
                or attempt.get("body_bytes") != len(raw_request) or len(raw_request) > MAX_BODY_BYTES
                or _sha_bytes(raw_request) != request_row.get("body_sha256")
                or body_row.get("request_body_utf8") != request_row.get("body_utf8")
                or body_row.get("request_body_sha256") != request_row.get("body_sha256")
                or not isinstance(raw_response, str)
                or _sha_bytes(raw_response.encode("utf-8")) != body_row.get("response_body_sha256")
                or body_row.get("status") != timing_by_id[request_id].get("status")
                or body_row.get("status") != 200):
            raise ValueError("raw HTTP request or response differs from durable transport evidence")
        timing = timing_by_id[request_id]
        start, finish = timing.get("started_at"), timing.get("finished_at")
        if (isinstance(start, bool) or isinstance(finish, bool)
                or not isinstance(start, (int, float)) or not isinstance(finish, (int, float))
                or not math.isfinite(start) or not math.isfinite(finish) or start < receipt_monotonic
                or finish < start or finish - receipt_monotonic > MAX_WALL_SECONDS):
            raise ValueError("HTTP timing lies outside the durable 600-second run window")
        timing_windows.append((float(start), float(finish)))
    if _max_overlap(timing_windows) > MAX_IN_FLIGHT:
        raise ValueError("recorded provider overlap exceeded the global in-flight limit")
    statuses = []
    linked_global_ids: set[str] = set()
    for session, event in zip(preview["sessions"], events):
        if session["session_id"] != event.get("session_id"):
            raise ValueError("trace schedule differs from frozen preview")
        calls = event.get("wire_calls")
        if not isinstance(calls, list) or len({row.get("request_id") for row in calls}) != len(calls):
            raise ValueError("session wire evidence is missing or contains duplicate request IDs")
        call_map = {row["request_id"]: row for row in calls}
        trace: list[dict[str, Any]] = []
        linked = set()
        for recorded in event.get("trace", []):
            action = next_trace_action(session["base"], trace)
            if action["kind"] == "forced":
                proposed = advance_trace(session["base"], trace, None)
                if proposed[-1] != recorded:
                    raise ValueError("recorded forced step differs from reconstructed legal action")
                trace.append(recorded)
                continue
            if action["kind"] != "request":
                raise ValueError("trace contains an event after a terminal state")
            request = build_order_request(session["base"], trace)
            if request is None:
                raise ValueError("expected model request could not be reconstructed")
            call = call_map.get(request.request_id)
            if call is None or request.request_id in linked or request.request_id in linked_global_ids:
                raise ValueError("model request has no unique preserved wire evidence")
            linked.add(request.request_id)
            linked_global_ids.add(request.request_id)
            attempt = attempt_by_id.get(request.request_id)
            sent = request_by_id.get(request.request_id)
            raw_body = body_by_id.get(request.request_id)
            http_timing = timing_by_id.get(request.request_id)
            if (attempt is None or sent is None or raw_body is None or http_timing is None
                    or call.get("method") != attempt.get("method")
                    or call.get("url") != attempt.get("url")
                    or call.get("request_body_utf8") != sent.get("body_utf8")
                    or call.get("request_body_sha256") != sent.get("body_sha256")
                    or call.get("request_body_utf8") != raw_body.get("request_body_utf8")
                    or call.get("response_body_utf8") != raw_body.get("response_body_utf8")
                    or call.get("response_body_sha256") != raw_body.get("response_body_sha256")
                    or call.get("status") != raw_body.get("status")
                    or call.get("timing") != http_timing):
                raise ValueError("session wire call does not join exactly to independent transport evidence")
            expected_state = render_state(request, session["arm"], donor_history=session.get("donor_history"))
            expected_wire = asyncio.run(serialize_sdk_wire(request, "portfolio_third_v1", expected_state))
            body = call.get("request_body_utf8")
            if (call.get("method") != "POST" or _url_tuple(call.get("url", "")) != _url_tuple(DEFAULT_URL)
                    or call.get("request_body_sha256") != expected_wire.body_sha256
                    or not isinstance(body, str) or body.encode("utf-8") != expected_wire.body):
                raise ValueError("actual SDK body differs from reconstructed arm-specific effective state")
            if call.get("response_body_sha256") != _sha_bytes(str(call.get("response_body_utf8")).encode("utf-8")):
                raise ValueError("preserved SDK response body hash is invalid")
            response_record = {"status": call.get("status"),
                "response_body_utf8": call.get("response_body_utf8")}
            timing = {"status": call.get("status")}
            if not isinstance(call.get("reply"), Mapping):
                raise ValueError("wire evidence lacks the parsed backend reply")
            reply = DecisionReply(**call["reply"])
            expected_prompt = build_instruction(request, "portfolio_third_v1")
            expected_estimate = estimate_input_tokens(request, expected_prompt, expected_state)
            if reply.usage.get("estimated_input_tokens") != expected_estimate:
                raise ValueError("recorded backend estimate differs from reconstructed effective input")
            parsed_reply, parser_error = asyncio.run(_replay_backend(request, "portfolio_third_v1",
                response_record, timing))
            if (parser_error is not None or parsed_reply is None
                    or not _same_reply_except_state_estimate(parsed_reply, reply)):
                raise ValueError("recorded reply differs from the pinned SDK/backend parser")
            proposed = advance_trace(session["base"], trace, reply)
            if proposed[-1] != recorded:
                raise ValueError("recorded model decision differs from reconstructed probability choice")
            trace.append(recorded)
        if linked != set(call_map):
            raise ValueError("wire evidence contains an unlinked provider call")
        status = replay_trace(session["base"], trace)
        if status.get("status") == "incomplete":
            if status.get("reason") != "unresolved_tie":
                raise ValueError("technical failure or truncated trace invalidates the all-or-none pilot")
            status_name = "verified_tie"
        else:
            status_name = "complete"
        if terminal_status_by_id.get(session["session_id"]) != status_name:
            raise ValueError("terminal per-session status differs from independently replayed trace")
        statuses.append({"session_id": session["session_id"], "status": status_name,
                         "orders": [asdict(order) for order in status["orders"]]})
    if linked_global_ids != set(attempt_by_id):
        raise ValueError("charged attempts are not linked one-to-one to reconstructed model decisions")
    return {"valid": True, "session_n": len(statuses), "statuses": statuses}


def _same_reply_except_state_estimate(parsed: DecisionReply, recorded: DecisionReply) -> bool:
    """Compare parser-derived reply while validating the independently rendered estimate separately."""
    fields = ("request_id", "agent_id", "round_index", "stage", "observation_hash",
        "candidate_set_hash", "raw_scores", "probabilities", "backend", "model_revision")
    if any(canonical_json(getattr(parsed, name)) != canonical_json(getattr(recorded, name)) for name in fields):
        return False
    left_usage, right_usage = dict(parsed.usage), dict(recorded.usage)
    left_usage.pop("estimated_input_tokens", None)
    right_usage.pop("estimated_input_tokens", None)
    return canonical_json(left_usage) == canonical_json(right_usage)


def score_run(preview_dir: str | Path, run_dir: str | Path) -> dict[str, Any]:
    """Score audited full portfolios and verified ties, collapsing repeats first."""
    from capclaims.injection_analysis import collapse_repeats, paired_arm_summary, score_portfolio
    from capclaims.paper.orders import PaperOrder
    preview = json.loads((Path(preview_dir) / "manifest.json").read_text(encoding="utf-8"))
    replayed = replay_run(preview_dir, run_dir)
    labels_path = Path(preview_dir) / "labels.json"
    labels_raw = labels_path.read_bytes()
    preview_manifest = json.loads((Path(preview_dir) / "manifest.json").read_text(encoding="utf-8"))
    if _sha_bytes(labels_raw) != preview_manifest.get("labels_file_sha256"):
        raise ValueError("held-out target labels changed after preview freeze")
    labels = json.loads(labels_raw.decode("utf-8"))
    baselines_path = Path(preview_dir) / "baselines.json"
    baselines_raw = baselines_path.read_bytes()
    if _sha_bytes(baselines_raw) != preview_manifest.get("baselines_file_sha256"):
        raise ValueError("offline baseline inputs changed after preview freeze")
    baseline_records = json.loads(baselines_raw.decode("utf-8"))
    events = json.loads((Path(run_dir) / "events.json").read_text(encoding="utf-8"))
    sessions = {row["session_id"]: row for row in preview["sessions"]}
    event_by_id = {row["session_id"]: row for row in events}
    replay_by_id = {row["session_id"]: row for row in replayed["statuses"]}
    cases = []
    for sid, session in sessions.items():
        person, arm = session["person_id"], session["arm"]
        truth_row = labels[person]
        truth = tuple(PaperOrder(f"truth-{person}-{i}", person, row["type"].lower(), row["amount"], row["price"])
                      for i, row in enumerate(truth_row["orders"]))
        result = replay_by_id[sid]
        predicted = tuple(PaperOrder(**row) for row in result["orders"]) if result["status"] == "complete" else None
        scored = score_portfolio(predicted, truth, session["base"]["Q"])
        comparison = scored.get("comparison", {})
        predicted_orders = predicted
        baseline_orders = [tuple(sorted((order["type"].lower(), order["amount"], order["price"])
            for order in history_row["orders"])) for history_row in baseline_records[person]]
        predicted_signature = tuple(sorted((order.side, order.qty, order.limit_dollars)
            for order in predicted_orders)) if predicted is not None else None
        cases.append({"person_id": person, "arm": arm, "repeat": session["repeat"],
                      "status": result["status"], "loss": scored["loss"],
                      "retail_window_1_100_loss": scored["retail_window_1_100_loss"],
                      "buy_curve_error": scored["buy_curve_error"],
                      "sell_curve_error": scored["sell_curve_error"],
                      "sell_curve_error_as_share_of_Q": scored["sell_curve_error_as_share_of_Q"],
                      "buy_window_1_100_error": scored["buy_window_1_100_error"],
                      "sell_window_1_100_error": scored["sell_window_1_100_error"],
                      "sell_window_1_100_error_as_share_of_Q": scored["sell_window_1_100_error_as_share_of_Q"],
                      "gross_buy": sum(order.qty for order in predicted_orders if order.side == "buy") if predicted is not None else None,
                      "gross_sell": sum(order.qty for order in predicted_orders if order.side == "sell") if predicted is not None else None,
                      "gross_sell_as_share_of_Q": (sum(order.qty for order in predicted_orders if order.side == "sell") /
                          session["base"]["Q"] if predicted is not None and session["base"]["Q"] > 0 else None),
                      "order_count": len(predicted_orders) if predicted is not None else None,
                      "duplicate_order_n": scored.get("duplicate_order_n"),
                      "duplicate_order_rate": (scored["duplicate_order_n"] / max(1, len(predicted_orders))
                          if predicted is not None else None),
                      "five_order_portfolio": scored.get("five_order_portfolio"),
                      "complete_case": result["status"] == "complete",
                      "copies_personal_last_round8": (predicted_signature == baseline_orders[-1]
                          if predicted is not None else None),
                      "copies_any_personal_training_portfolio": (predicted_signature in baseline_orders
                          if predicted is not None else None),
                      "portfolio_signature": predicted_signature,
                      "exact_order_multiset_match": comparison.get("exact_order_multiset_match"),
                      "merged_same_side_price_match": comparison.get("merged_same_side_price_match"),
                      "side_class_agreement": comparison.get("side_class_agreement"),
                      "empty_agreement": comparison.get("empty_agreement")})
    from capclaims.portfolio_metrics import training_baselines, portfolio_comparison
    baseline_person = {}
    donor_by_target = {row["target_id"]: row["donor_id"] for row in preview_manifest["donor_matches"]}
    for person in preview_manifest["selected_people"]:
        target = labels[person]
        truth = tuple(PaperOrder(f"truth-{person}-{i}", person, row["type"].lower(), row["amount"], row["price"])
                      for i, row in enumerate(target["orders"]))
        q = target["Q"]
        own_history = [{"round": row["round"], "orders": tuple(PaperOrder(
            f"{person}:baseline:{row['round']}:{i}", person, order["type"].lower(), order["amount"], order["price"])
            for i, order in enumerate(row["orders"]))} for row in baseline_records[person]]
        own = training_baselines(own_history, {person: truth}, {person: q})["targets"][person]
        donor_id = donor_by_target[person]
        donor_row = next(row for row in baseline_records[donor_id] if row["round"] == 8)
        donor_orders = tuple(PaperOrder(f"{donor_id}:round8:{i}", donor_id, order["type"].lower(),
            order["amount"], order["price"]) for i, order in enumerate(donor_row["orders"]))
        donor_comp = portfolio_comparison(donor_orders, truth)
        donor_sell = sum(order.qty for order in donor_orders if order.side == "sell")
        baseline_person[person] = {**own,
            "donor_last_round8_curve_error": donor_comp["primary_curve_error"],
            "donor_last_round8_feasible_under_target_Q": donor_sell <= q,
            "donor_last_round8_unclipped_gross_sell": donor_sell}
    collapsed = collapse_repeats(cases)
    contrasts = {"C_minus_D": paired_arm_summary(collapsed, "C", "D"),
                 "C_minus_E": paired_arm_summary(collapsed, "C", "E"),
                 "A_minus_B": paired_arm_summary(collapsed, "A", "B"),
                 "B_minus_C": paired_arm_summary(collapsed, "B", "C"),
                 "D_minus_E": paired_arm_summary(collapsed, "D", "E")}
    _write_derived_idempotent(Path(run_dir) / "private-scores.json", {"cases": cases,
        "person_collapsed_losses": collapsed, "paired_contrasts": contrasts,
        "training_only_baselines_by_person": baseline_person})
    safe_contrasts = {}
    for name, row in contrasts.items():
        deltas = [item["left_minus_right"] for item in row["paired_differences"]]
        safe_contrasts[name] = {key: row[key] for key in
            ("left_arm", "right_arm", "person_n", "mean_left_minus_right_loss", "median_left_minus_right_loss")}
        safe_contrasts[name]["sign_counts"] = {"left_loss_higher": sum(x > 0 for x in deltas),
            "left_loss_lower": sum(x < 0 for x in deltas), "tie": sum(x == 0 for x in deltas)}
    metric_fields = ("loss", "retail_window_1_100_loss", "buy_curve_error", "sell_curve_error",
        "sell_curve_error_as_share_of_Q",
        "buy_window_1_100_error", "sell_window_1_100_error", "gross_buy", "gross_sell",
        "sell_window_1_100_error_as_share_of_Q",
        "gross_sell_as_share_of_Q", "copies_personal_last_round8", "copies_any_personal_training_portfolio",
        "order_count", "duplicate_order_n", "duplicate_order_rate", "five_order_portfolio", "complete_case",
        "exact_order_multiset_match",
        "merged_same_side_price_match", "side_class_agreement", "empty_agreement")
    metric_summary = {}
    for metric in metric_fields:
        metric_rows = [{"person_id": row["person_id"], "arm": row["arm"], "loss": float(row[metric])}
                       for row in cases if row[metric] is not None]
        collapsed_metric = collapse_repeats(metric_rows)
        metric_summary[metric] = {arm: (sum(person[arm] for person in collapsed_metric.values()
            if arm in person) / sum(arm in person for person in collapsed_metric.values())
            if any(arm in person for person in collapsed_metric.values()) else None) for arm in ARMS}
    baseline_fields = ("last_training_portfolio_curve_error", "pointwise_personal_median_curve_error",
                       "empty_portfolio_curve_error", "donor_last_round8_curve_error")
    baseline_summary = {field: sum(row[field] for row in baseline_person.values()) / len(baseline_person)
                        for field in baseline_fields}
    baseline_summary["donor_last_round8_feasible_n"] = sum(
        row["donor_last_round8_feasible_under_target_Q"] for row in baseline_person.values())
    baseline_summary["own_last_round8_feasible_n"] = sum(
        row["last_training_portfolio_feasible_under_current_Q"] for row in baseline_person.values())
    baseline_summary["person_n"] = len(baseline_person)
    complete_collapsed = collapse_repeats([row for row in cases if row["complete_case"]])
    repeat_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in cases:
        repeat_groups.setdefault((row["person_id"], row["arm"]), []).append(row)
    repeated = [rows for rows in repeat_groups.values() if len(rows) > 1]
    complete_repeated = [rows for rows in repeated if all(row["complete_case"] for row in rows)]
    repeat_consistency = {"repeated_person_arm_n": len(repeated),
        "complete_repeated_person_arm_n": len(complete_repeated),
        "exact_portfolio_match_n": sum(len({row["portfolio_signature"] for row in rows}) == 1
            for rows in complete_repeated),
        "mean_loss_range": (sum(max(row["loss"] for row in rows) - min(row["loss"] for row in rows)
            for rows in repeated) / len(repeated) if repeated else None)}
    safe = {"status": "SCORED_OFFLINE", "session_n": len(cases), "person_n": len(collapsed),
        "complete_portfolio_n": sum(row["status"] == "complete" for row in cases),
        "verified_tie_n": sum(row["status"] == "verified_tie" for row in cases),
        "person_collapsed_mean_loss_by_arm": {arm: sum(person[arm] for person in collapsed.values()
            if arm in person) / sum(arm in person for person in collapsed.values()) for arm in ARMS},
        "person_collapsed_metric_means_by_arm": metric_summary,
        "complete_case_person_collapsed_mean_loss_by_arm": {arm: sum(person[arm] for person in complete_collapsed.values()
            if arm in person) / sum(arm in person for person in complete_collapsed.values())
            if any(arm in person for person in complete_collapsed.values()) else None for arm in ARMS},
        "repeat_consistency": repeat_consistency,
        "coprimary_person_paired_contrasts": safe_contrasts,
        "offline_baselines_person_mean": baseline_summary,
        "limitations": ["Six people in one shared market do not support population-equivalence claims.",
                        "Scores measure submitted order portfolios conditional on actual p, Q, and D."]}
    return safe

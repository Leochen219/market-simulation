"""Predeclared, preview-first Jev input ablation for one-step trade choices.

Preview is offline and does not load the project .env. With --execute, this
module freezes the complete request plan before dispatching the first request.
Oracle profit arithmetic remains in metadata and is never serialized to a
model-visible observation, instruction, or candidate label.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Iterable, Mapping

from capclaims.backends.jev import JevBackend, estimate_input_tokens
from capclaims.backends.jev_instructions import build_instruction
from capclaims.domain.types import Candidate
from capclaims.engine.protocol import DecisionReply, DecisionRequest, validate_reply
from capclaims.paper.observations import build_paper_observation
from capclaims.randomness import canonical_json

MODEL = "jev-1.14.0"
SEED = 20261024
REQUEST_CAP = 396
TOKEN_CAP = 1_500_000
COMMON_INSTRUCTION = build_instruction(None, "ablation_v1")


@dataclass(frozen=True)
class Fixture:
    case_id: str
    cohort: str
    family: str
    p: int
    Q: int
    D: int
    w: int
    actions: tuple[tuple[str, int, int], ...]  # side, units, quoted dollars/unit


@dataclass(frozen=True)
class Condition:
    condition: str
    market_context: bool
    history_mcp: int | None
    label_arm: str


def fixtures() -> list[Fixture]:
    """Three observed STOP errors and three balanced fresh-number controls."""
    return [
        Fixture("sell_surplus", "seen", "surplus", 70, 60, 20, 10, (("SELL", 20, 15),)),
        Fixture("expiry_zero_demand", "seen", "zero_demand", 90, 10, 0, 10,
                (("BUY", 1, 1), ("SELL", 10, 1))),
        Fixture("sell_needed_premium", "seen", "needed_capacity", 60, 20, 20, 10,
                (("SELL", 10, 90), ("BUY", 10, 90))),
        Fixture("new_surplus", "new", "surplus", 85, 70, 25, 10, (("SELL", 30, 12),)),
        Fixture("new_stop_needed_capacity", "new", "needed_capacity", 78, 22, 22, 10,
                (("SELL", 8, 19),)),
        Fixture("new_buy_shortage", "new", "shortage", 83, 9, 24, 10,
                (("BUY", 15, 18), ("SELL", 5, 18))),
    ]


def conditions() -> list[Condition]:
    """Eight context/history factorial cells plus three clean label controls."""
    rows = [Condition(f"market_{int(market)}_history_{'none' if mcp is None else mcp}",
                      market, mcp, "both_explicit")
            for market in (False, True) for mcp in (None, 10, 333, 777)]
    rows.extend(Condition(f"clean_labels_{arm}", False, None, arm)
                for arm in ("bare", "sell_cash", "stop_cost"))
    return rows


def _market_rules() -> dict[str, str]:
    """Return the exact existing public_instructions source, with no edits."""
    source = build_paper_observation(agent_id="source", round_index=1, treatment="GA", n_agents=10,
        wholesale_price=10, p=70, history=(), partial_orders=(), phase="orders", q=100, d=100,
        decision_interface="economic_v2")
    return dict(source["public_instructions"])


def _history(mcp: int) -> list[dict[str, Any]]:
    return [dict(round=r, p=70, Q=100, D=100, orders=[], profit_cents=600000,
                 net_quantity=0, mcp_cents=mcp * 100) for r in (1, 2)]


def _history_observation(mcp: int) -> tuple[list[Any], str]:
    source = build_paper_observation(agent_id="source", round_index=1, treatment="GA", n_agents=10,
        wholesale_price=10, p=70, history=_history(mcp), partial_orders=(), phase="orders", q=100, d=100,
        decision_interface="economic_v2")
    return source["own_history"], source["own_history_fields"]


def _profit(f: Fixture, side: str, qty: int, price: int) -> int:
    delta = qty if side == "BUY" else -qty if side == "SELL" else 0
    cash = -price * qty if side == "BUY" else price * qty if side == "SELL" else 0
    return -f.w * f.Q + cash + f.p * min(f.D, f.Q + delta)


def _action_labels(f: Fixture, arm: str) -> tuple[list[Candidate], dict[str, int]]:
    specs = [("STOP", 0, 0), *f.actions]
    labels: list[Candidate] = []
    profits: dict[str, int] = {}
    for index, (side, qty, price) in enumerate(specs):
        candidate_id = f"c{index}"
        if side == "STOP":
            text = "STOP: no trade"
            if arm in {"both_explicit", "stop_cost"}:
                text += "; the already-paid procurement cost is not refunded"
        else:
            text = f"{side} {qty} units at ${price} per unit"
            if side == "SELL" and arm in {"both_explicit", "sell_cash"}:
                text += f"; the other party pays you ${price} for each unit transferred"
        labels.append(Candidate(candidate_id, side, text))
        profits[candidate_id] = _profit(f, side, qty, price)
    return labels, profits


def _request(f: Fixture, c: Condition, order: str, repetition: int, sequence: int
             ) -> tuple[DecisionRequest, dict[str, Any]]:
    observation: dict[str, Any] = {
        "retail_value_per_unit_dollars": f.p,
        "procured_capacity_units": f.Q,
        "customer_demand_units": f.D,
        "unit_procurement_cost_dollars": f.w,
        "procurement_already_paid": True,
    }
    if c.market_context:
        observation["public_instructions"] = _market_rules()
    if c.history_mcp is not None:
        observation["own_history"], observation["own_history_fields"] = _history_observation(c.history_mcp)
    candidates, profits = _action_labels(f, c.label_arm)
    if order == "reversed":
        candidates.reverse()
    request = DecisionRequest(f"ablation_{sequence:03d}", 0, "jev:one_step_trade",
                              observation, tuple(candidates))
    best = max(profits.values())
    winners = [key for key, value in profits.items() if value == best]
    if len(winners) != 1:
        raise ValueError(f"fixture {f.case_id} does not have a unique best action")
    action_for_id = {candidate.candidate_id: str(candidate.value) for candidate in candidates}
    expected = action_for_id[winners[0]]
    metadata = {
        "condition": c.condition, "case_id": f.case_id, "repetition": repetition,
        "order": order, "cohort": f.cohort, "expected": expected, "family": f.family,
        "expected_profits_dollars": profits,
    }
    return request, metadata


def _wire_payload(request: DecisionRequest) -> dict[str, Any]:
    return {"state": dict(request.observation), "instruction": COMMON_INSTRUCTION,
            "criteria": {candidate.candidate_id: candidate.label for candidate in request.candidates}}


def _sha(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _wire_sha(payload: Mapping[str, Any]) -> str:
    """Hash criteria as an ordered pair list so the order manipulation is auditable."""
    ordered = dict(payload)
    ordered["criteria"] = list(payload["criteria"].items())
    return _sha(ordered)


def build_plan() -> dict[str, Any]:
    """Freeze deterministic, interleaved requests and hashes before any call."""
    raw: list[tuple[Fixture, Condition, str, int]] = []
    for c in conditions():
        for f in fixtures():
            for order in ("original", "reversed"):
                for repetition in range(3):
                    raw.append((f, c, order, repetition))
    random.Random(SEED).shuffle(raw)
    pairs = [_request(f, c, order, repetition, index)
             for index, (f, c, order, repetition) in enumerate(raw)]
    rows = []
    total_tokens = 0
    wire_hashes = []
    for request, metadata in pairs:
        estimate = estimate_input_tokens(request, COMMON_INSTRUCTION)
        total_tokens += estimate
        payload = _wire_payload(request)
        wire_hash = _wire_sha(payload)
        wire_hashes.append(wire_hash)
        rows.append({**metadata, "request": asdict(request), "wire": payload,
                     "wire_sha256": wire_hash, "estimated_input_tokens": estimate})
    if len(rows) != REQUEST_CAP:
        raise ValueError(f"request count {len(rows)} differs from cap {REQUEST_CAP}")
    if total_tokens > TOKEN_CAP:
        raise ValueError(f"estimated input tokens {total_tokens} exceed cap {TOKEN_CAP}")
    definition = {"fixtures": [asdict(f) for f in fixtures()],
                  "conditions": [asdict(c) for c in conditions()], "seed": SEED,
                  "instruction": COMMON_INSTRUCTION}
    return {"model": MODEL, "seed": SEED, "fixtures": definition["fixtures"],
            "conditions": definition["conditions"], "matrixhash": _sha(definition),
            "orderedwirehash": _sha(wire_hashes), "count": len(rows),
            "estimated_input_tokens": total_tokens, "request_cap": REQUEST_CAP,
            "token_cap": TOKEN_CAP, "max_in_flight": 10,
            "waves": [rows[i:i + 10] for i in range(0, len(rows), 10)]}


def build_pairs() -> list[dict[str, Any]]:
    """Compatibility helper returning the frozen, scheduled plan rows."""
    return [row for wave in build_plan()["waves"] for row in wave]


def summarize(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Descriptive condition/case summaries; failed and tied choices stay visible."""
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["condition"]), str(row["case_id"])), []).append(row)
    summaries = []
    for (condition, case_id), group in sorted(grouped.items()):
        summaries.append({
            "condition": condition, "case_id": case_id, "cohort": group[0].get("cohort"),
            "family": group[0].get("family"), "expected": group[0].get("expected"),
            "n": len(group), "correct_strict": sum(row.get("correct") is True for row in group),
            "strict_rate": sum(row.get("correct") is True for row in group) / len(group),
            "mean_sell_probability": sum(float(row.get("sell_probability", 0.0)) for row in group) / len(group),
            "top_tie_count": sum(not bool(row.get("strict_ranking")) for row in group),
            "probability_vectors": [row.get("reply", {}).get("probabilities") for row in group],
            "top_candidate_ids": [row.get("top_candidate_ids", []) for row in group],
        })
    return {"n_rows": sum(len(v) for v in grouped.values()), "groups": summaries}


def _row_from_reply(plan_row: Mapping[str, Any], reply: DecisionReply) -> dict[str, Any]:
    request = DecisionRequest(**{**plan_row["request"],
        "candidates": tuple(Candidate(**candidate) for candidate in plan_row["request"]["candidates"])})
    validate_reply(request, reply)
    probs = dict(reply.probabilities)
    ordered = sorted(probs.values(), reverse=True)
    strict = len(ordered) == 1 or ordered[0] > ordered[1]
    top = max(probs.values())
    top_ids = [c.candidate_id for c in request.candidates if probs[c.candidate_id] == top]
    selected = (next(c.value for c in request.candidates if c.candidate_id == top_ids[0])
                if strict else "TOP_TIE")
    correct = strict and selected == plan_row["expected"]
    strengths = sorted(probs.values(), reverse=True)
    margin = strengths[0] - strengths[1] if len(strengths) > 1 else strengths[0]
    sell_id = next((c.candidate_id for c in request.candidates if c.value == "SELL"), None)
    return ({k: plan_row[k] for k in ("condition", "case_id", "repetition", "order", "cohort", "expected", "family",
                                      "wire_sha256", "estimated_input_tokens")}
    | {"request": asdict(request), "reply": asdict(reply), "selected": selected,
       "correct": correct, "strict_ranking": strict, "top_candidate_ids": top_ids,
       "top_margin": margin,
       "sell_probability": probs.get(sell_id, 0.0) if sell_id is not None else 0.0})


async def _execute(plan: Mapping[str, Any], output: Path) -> dict[str, Any]:
    # Persist the complete wire manifest and source/instruction hashes before dispatch.
    root = Path(__file__).resolve().parents[2]
    source_files = [
        Path(__file__).resolve(),
        root / "src/capclaims/backends/jev.py",
        root / "src/capclaims/backends/jev_instructions.py",
        root / "src/capclaims/engine/protocol.py",
        root / "src/capclaims/paper/observations.py",
    ]
    source_hashes = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in source_files}
    frozen = {k: v for k, v in plan.items() if k != "waves"}
    frozen["source_sha256"] = source_hashes
    frozen["instruction_sha256"] = hashlib.sha256(COMMON_INSTRUCTION.encode()).hexdigest()
    (output / "frozen-plan.json").write_text(json.dumps({**frozen, "waves": plan["waves"]}, indent=2) + "\n", encoding="utf-8")
    backend = JevBackend(instruction_profile="ablation_v1", max_requests=REQUEST_CAP,
                         max_estimated_input_tokens=TOKEN_CAP, max_in_flight=10)
    evidence: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    stopped_after_failure = False
    try:
        for wave_index, wave in enumerate(plan["waves"]):
            requests = []
            for row in wave:
                data = dict(row["request"])
                data["candidates"] = tuple(Candidate(**candidate) for candidate in data["candidates"])
                requests.append(DecisionRequest(**data))
            try:
                replies = await backend.decide_many(requests)
                by_id = {reply.request_id: reply for reply in replies}
                if len(by_id) != len(requests):
                    raise ValueError("reply identity coverage mismatch")
                for row, request in zip(wave, requests, strict=True):
                    reply = by_id.get(request.request_id)
                    if reply is None:
                        failures.append({"request_id": request.request_id, "status": "missing_reply"})
                    else:
                        evidence.append(_row_from_reply(row, reply))
            except Exception as exc:
                # Do not persist exception bodies, which may contain provider text.
                failures.extend({"request_id": r.request_id, "status": type(exc).__name__} for r in requests)
            (output / "raw-evidence.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
            (output / "failures.json").write_text(json.dumps(failures, indent=2) + "\n", encoding="utf-8")
            if failures:
                stopped_after_failure = True
                for later in plan["waves"][wave_index + 1:]:
                    for row in later:
                        req = row["request"]
                        request_id = f"r{req['round_index']:03d}:{req['stage']}:{req['agent_id']}"
                        failures.append({"request_id": request_id,
                                         "status": "not_dispatched_after_prior_wave_failure"})
                (output / "failures.json").write_text(json.dumps(failures, indent=2) + "\n", encoding="utf-8")
                break
        result = {"status": "COMPLETE" if not failures and len(evidence) == REQUEST_CAP else "PARTIAL",
                  "model": MODEL, "plan_hash": frozen["orderedwirehash"], "evidence_rows": len(evidence),
                  "failed_rows": len(failures), "stopped_after_failure": stopped_after_failure,
                  "budget_usage": backend.budget_usage,
                  "window_timings": list(backend.window_timings),
                  "transport_timings": list(backend.transport_timings), "summary": summarize(evidence)}
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return result
    finally:
        await backend.aclose()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true", help="send the frozen plan to Jev; preview is the default")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    plan = build_plan()
    if args.execute:
        result = asyncio.run(_execute(plan, args.output_dir))
    else:
        preview = {k: v for k, v in plan.items() if k != "waves"}
        preview["preview"] = True
        preview["wave_count"] = len(plan["waves"])
        preview["wire_preview"] = plan["waves"][0]
        (args.output_dir / "preview.json").write_text(json.dumps(preview, indent=2) + "\n", encoding="utf-8")
        result = preview
    print(json.dumps(result, indent=2))
    return 0 if not args.execute or result.get("status") == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())

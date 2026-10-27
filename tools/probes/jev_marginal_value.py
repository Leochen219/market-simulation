"""Offline-first Jev probe for the marginal value of inventory in a one-step sale.

The external oracle and experiment metadata are kept outside the model-visible
wire. Preview builds the same frozen plan used by execution without creating a
backend, loading local environment files, or making a network request.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Iterable, Mapping

from capclaims.backends.jev import JevBackend, estimate_input_tokens
from capclaims.backends.jev_instructions import build_instruction
from capclaims.domain.types import Candidate
from capclaims.engine.protocol import DecisionReply, DecisionRequest, validate_reply
from capclaims.randomness import canonical_json

MODEL = "jev-1.14.0"
SEED = 20261024
REQUEST_CAP = 456
TOKEN_CAP = 1_500_000
MAX_IN_FLIGHT = 10
COMMON_INSTRUCTION = build_instruction(None, "ablation_v1")
SELL_QUANTITY = 10
PROCUREMENT_COST = 10


def _series_definitions() -> list[dict[str, Any]]:
    """Return six predeclared one-variable grids in their interpretation order."""
    return [
        {"name": "inventory_Q", "axis": "Q", "fixed": {"p": 70, "m": 15, "D": 20},
         "values": [10, 15, 19, 20, 21, 22, 25, 27, 28, 29, 30, 31, 40, 60]},
        {"name": "demand_D", "axis": "D", "fixed": {"p": 70, "m": 15, "Q": 40},
         "values": [0, 10, 20, 29, 30, 31, 32, 33, 35, 39, 40, 50, 100]},
        {"name": "surplus_sell_price_m", "axis": "m", "fixed": {"p": 70, "Q": 40, "D": 20},
         "values": [1, 5, 9, 10, 11, 14, 15, 20, 35, 50, 69, 70, 71, 85, 99, 100, 140, 333, 999]},
        {"name": "needed_sell_price_m", "axis": "m", "fixed": {"p": 70, "Q": 20, "D": 20},
         "values": [1, 5, 9, 10, 11, 14, 15, 20, 35, 50, 69, 70, 71, 85, 99, 100, 140, 333, 999]},
        {"name": "surplus_retail_value_p", "axis": "p", "fixed": {"m": 70, "Q": 40, "D": 20},
         "values": [51, 60, 69, 70, 71, 80, 90, 100]},
        {"name": "needed_retail_value_p", "axis": "p", "fixed": {"m": 70, "Q": 20, "D": 20},
         "values": [51, 60, 69, 70, 71, 80, 90, 100]},
    ]


def scenarios() -> list[dict[str, Any]]:
    """Build deduplicated economic states while retaining series memberships."""
    indexed: dict[tuple[int, int, int, int, int, int], dict[str, Any]] = {}
    for series in _series_definitions():
        for value in series["values"]:
            factors = {**series["fixed"], series["axis"]: value}
            state = (factors["Q"], factors["D"], factors["p"], factors["m"],
                     PROCUREMENT_COST, SELL_QUANTITY)
            row = indexed.get(state)
            if row is None:
                Q, D, p, m, w, x = state
                row = {"case_id": f"s{len(indexed):03d}", "Q": Q, "D": D,
                       "p": p, "m": m, "w": w, "x": x, "series": []}
                indexed[state] = row
            row["series"].append({"name": series["name"], "axis": series["axis"], "value": value})
    return list(indexed.values())


def external_oracle(scenario: Mapping[str, int]) -> dict[str, Any]:
    """Compute the exact outside-wire change in profit and optimal action set."""
    Q, D, p, m, w, x = (scenario[key] for key in ("Q", "D", "p", "m", "w", "x"))
    if any(isinstance(v, bool) or not isinstance(v, int) for v in (Q, D, p, m, w, x)):
        raise ValueError("oracle inputs must be integers")
    if x <= 0 or Q < x or D < 0 or min(p, m, w) < 0:
        raise ValueError("oracle inputs outside experiment domain")
    delta = m * x - p * (min(D, Q) - min(D, Q - x))
    actions = ["SELL"] if delta > 0 else ["STOP"] if delta < 0 else ["STOP", "SELL"]
    return {"delta_profit": delta, "optimal_actions": actions}


def _sha(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _request(scenario: Mapping[str, Any], order: str, repetition: int, sequence: int
             ) -> tuple[DecisionRequest, dict[str, Any]]:
    # These are the only five state fields. Do not add oracle or grid metadata.
    observation = {
        "retail_value_per_unit_dollars": scenario["p"],
        "procured_capacity_units": scenario["Q"],
        "customer_demand_units": scenario["D"],
        "unit_procurement_cost_dollars": scenario["w"],
        "procurement_already_paid": True,
    }
    candidates = [
        Candidate("c0", "STOP", "STOP: no trade; the already-paid procurement cost is not refunded"),
        Candidate("c1", "SELL", f"SELL {scenario['x']} units at ${scenario['m']} per unit; "
                  f"the other party pays you ${scenario['m']} for each unit transferred"),
    ]
    if order == "reversed":
        candidates.reverse()
    elif order != "original":
        raise ValueError(f"unknown candidate order: {order}")
    request = DecisionRequest(f"probe_{sequence:03d}", 0, "jev:one_step_trade",
                              observation, tuple(candidates))
    oracle = external_oracle(scenario)
    meta = {
        "case_id": scenario["case_id"], "series": list(scenario["series"]),
        "Q": scenario["Q"], "D": scenario["D"], "p": scenario["p"],
        "m": scenario["m"], "w": scenario["w"], "x": scenario["x"],
        "expected": oracle["optimal_actions"], "delta_profit": oracle["delta_profit"],
        "economic_tie": len(oracle["optimal_actions"]) > 1,
        "repetition": repetition, "order": order,
    }
    return request, meta


def _wire_payload(request: DecisionRequest) -> dict[str, Any]:
    return {"state": dict(request.observation), "instruction": COMMON_INSTRUCTION,
            "criteria": {candidate.candidate_id: candidate.label for candidate in request.candidates}}


def _wire_sha(payload: Mapping[str, Any]) -> str:
    # Preserve candidate insertion order: reverse-order requests must hash differently.
    ordered = dict(payload)
    ordered["criteria"] = list(payload["criteria"].items())
    return _sha(ordered)


def build_plan() -> dict[str, Any]:
    """Freeze 456 deterministic calls, externally scored but wire-clean."""
    all_scenarios = scenarios()
    schedule = [(scenario, order, repetition)
                for scenario in all_scenarios
                for order in ("original", "reversed")
                for repetition in range(3)]
    random.Random(SEED).shuffle(schedule)
    rows: list[dict[str, Any]] = []
    total_tokens = 0
    wire_hashes: list[str] = []
    for sequence, (scenario, order, repetition) in enumerate(schedule):
        request, metadata = _request(scenario, order, repetition, sequence)
        wire = _wire_payload(request)
        estimate = estimate_input_tokens(request, COMMON_INSTRUCTION)
        total_tokens += estimate
        wire_sha = _wire_sha(wire)
        wire_hashes.append(wire_sha)
        rows.append({**metadata, "request": asdict(request), "wire": wire,
                     "wire_sha256": wire_sha, "estimated_input_tokens": estimate})
    if len(rows) != REQUEST_CAP:
        raise ValueError(f"request count {len(rows)} differs from required cap {REQUEST_CAP}")
    if total_tokens > TOKEN_CAP:
        raise ValueError(f"estimated input tokens {total_tokens} exceed hard cap {TOKEN_CAP}")
    definition = {"scenarios": all_scenarios, "series": _series_definitions(),
                  "instruction": COMMON_INSTRUCTION, "seed": SEED,
                  "candidate_ids": {"STOP": "c0", "SELL": "c1"}}
    return {
        "model": MODEL, "seed": SEED, "count": len(rows),
        "scenario_count": len(all_scenarios), "series_memberships": sum(len(r["series"]) for r in all_scenarios),
        "matrixhash": _sha(definition), "orderedwirehash": _sha(wire_hashes),
        "estimated_input_tokens": total_tokens, "request_cap": REQUEST_CAP,
        "token_cap": TOKEN_CAP, "max_in_flight": MAX_IN_FLIGHT,
        "instruction": COMMON_INSTRUCTION, "series": definition["series"],
        "scenarios": all_scenarios,
        "waves": [rows[i:i + MAX_IN_FLIGHT] for i in range(0, len(rows), MAX_IN_FLIGHT)],
    }


def score(plan_row: Mapping[str, Any], reply: DecisionReply) -> dict[str, Any]:
    """Validate and score one response; strict choice ties are not broken by order."""
    data = dict(plan_row["request"])
    data["candidates"] = tuple(Candidate(**candidate) for candidate in data["candidates"])
    request = DecisionRequest(**data)
    validate_reply(request, reply)
    probs = {key: float(value) for key, value in reply.probabilities.items()}
    top = max(probs.values())
    top_ids = [c.candidate_id for c in request.candidates if probs[c.candidate_id] == top]
    strict = len(top_ids) == 1
    selected = next(c.value for c in request.candidates if c.candidate_id == top_ids[0]) if strict else "TOP_TIE"
    expected = list(plan_row["expected"])
    correct_margin = None
    correct: bool | None
    if bool(plan_row.get("economic_tie", len(expected) > 1)):
        correct = None
    else:
        correct_action = expected[0]
        expected_id = next(c.candidate_id for c in request.candidates if c.value == correct_action)
        other_ids = [c.candidate_id for c in request.candidates if c.candidate_id != expected_id]
        correct_margin = probs[expected_id] - max(probs[candidate_id] for candidate_id in other_ids)
        correct = bool(strict and selected == correct_action)
    sell_id = next((c.candidate_id for c in request.candidates if c.value == "SELL"), None)
    return {
        **{key: plan_row[key] for key in (
            "case_id", "series", "Q", "D", "p", "m", "w", "x", "expected",
            "delta_profit", "economic_tie", "repetition", "order", "wire_sha256",
            "estimated_input_tokens") if key in plan_row},
        "request": asdict(request), "reply": asdict(reply), "selected": selected,
        "correct": correct, "correct_margin": correct_margin,
        "strict_ranking": strict, "top_candidate_ids": top_ids,
        "top_margin": (top - sorted(probs.values(), reverse=True)[1]) if len(probs) > 1 else top,
        "sell_probability": probs[sell_id] if sell_id is not None else 0.0,
    }


def summarize(rows: Iterable[Mapping[str, Any]], plan: Mapping[str, Any]) -> dict[str, Any]:
    """Summarize economic accuracy, predeclared price rules, series and repeats."""
    rows = list(rows)
    unique = [row for row in rows if not row.get("economic_tie")]
    ties = [row for row in rows if row.get("economic_tie")]
    scenarios_by_id = {scenario["case_id"]: scenario for scenario in plan["scenarios"]}
    correct_count = sum(row.get("correct") is True for row in unique)
    heuristic_results: dict[str, dict[str, Any]] = {}
    for name, predicate in (("strict_m_gt_p", lambda row: row["m"] > row["p"]),
                            ("weak_m_ge_p", lambda row: row["m"] >= row["p"])):
        eligible = unique
        discriminating = [row for row in unique
                          if row["expected"] != ["SELL" if predicate(row) else "STOP"]]
        heuristic_action = lambda row: "SELL" if predicate(row) else "STOP"
        agreements = [row for row in eligible if row["selected"] == heuristic_action(row)]
        informative_agreements = [row for row in discriminating
                                  if row["selected"] == heuristic_action(row)]
        heuristic_results[name] = {
            "n": len(eligible), "agreements": len(agreements),
            "agreement_rate": len(agreements) / len(eligible) if eligible else None,
            "discriminating_n": len(discriminating),
            "discriminating_agreements": len(informative_agreements),
            "discriminating_agreement_rate": (len(informative_agreements) / len(discriminating)
                                              if discriminating else None),
            "discriminating_oracle_accuracy": (sum(row["correct"] is True for row in discriminating)
                                                / len(discriminating) if discriminating else None),
        }

    by_case: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_case.setdefault(str(row["case_id"]), []).append(row)
    scenario_points = []
    for case_id, scenario in scenarios_by_id.items():
        group = by_case.get(case_id, [])
        if not group:
            continue
        scenario_points.append({
            "case_id": case_id, "Q": scenario["Q"], "D": scenario["D"],
            "p": scenario["p"], "m": scenario["m"],
            "series": list(scenario["series"]), "expected": group[0]["expected"],
            "delta_profit": group[0]["delta_profit"], "economic_tie": group[0]["economic_tie"],
            "n": len(group), "accuracy": (sum(r["correct"] is True for r in group if r["correct"] is not None)
                                           / sum(r["correct"] is not None for r in group)
                                           if any(r["correct"] is not None for r in group) else None),
            "mean_sell_probability": sum(float(r["sell_probability"]) for r in group) / len(group),
            "mean_correct_margin": (sum(float(r["correct_margin"]) for r in group if r["correct_margin"] is not None)
                                    / sum(r["correct_margin"] is not None for r in group)
                                    if any(r["correct_margin"] is not None for r in group) else None),
            "top_ties": sum(not r["strict_ranking"] for r in group),
            "repeat_vectors": [r["reply"]["probabilities"] for r in group],
            "repeat_top_sets": [r["top_candidate_ids"] for r in group],
        })
    per_series = []
    for definition in plan["series"]:
        name = definition["name"]
        points = []
        for point in scenario_points:
            membership = next((item for item in point["series"] if item["name"] == name), None)
            if membership:
                points.append({"value": membership["value"], "case_id": point["case_id"],
                               "expected": point["expected"], "delta_profit": point["delta_profit"],
                               "mean_sell_probability": point["mean_sell_probability"],
                               "accuracy": point["accuracy"], "economic_tie": point["economic_tie"]})
        points.sort(key=lambda point: point["value"])
        per_series.append({"name": name, "axis": definition["axis"], "points": points})
    variation_groups: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        variation_groups.setdefault((str(row["case_id"]), str(row["order"])), []).append(row)
    variable_cases = sum(len({r["selected"] for r in group}) > 1 for group in variation_groups.values())
    return {
        "n_rows": len(rows),
        "unique_accuracy": {"n": len(unique), "correct": correct_count,
                            "rate": correct_count / len(unique) if unique else None},
        "economic_ties": {"n": len(ties), "probability_top_ties": sum(not r["strict_ranking"] for r in ties),
                          "actions": sorted({action for r in ties for action in r["expected"]})},
        "price_heuristics": heuristic_results,
        "scenario_points": scenario_points,
        "series": per_series,
        "repeat_variation": {"groups": len(variation_groups), "scenarios_with_action_variation": variable_cases,
                             "probability_vectors": sum(len({tuple(sorted(r["reply"]["probabilities"].items()))
                                                              for r in group}) > 1
                                                        for group in variation_groups.values())},
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    files = [Path(__file__).resolve(), root / "src/capclaims/backends/jev.py",
             root / "src/capclaims/backends/jev_instructions.py",
             root / "src/capclaims/engine/protocol.py", root / "src/capclaims/randomness.py",
             root / "src/capclaims/domain/types.py"]
    return {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in files}


async def _execute(plan: Mapping[str, Any], output: Path) -> dict[str, Any]:
    # Freeze all code and wire facts before constructing the backend/client.
    frozen = {key: value for key, value in plan.items() if key != "waves"}
    frozen["source_sha256"] = _source_hashes()
    frozen["instruction_sha256"] = hashlib.sha256(COMMON_INSTRUCTION.encode("utf-8")).hexdigest()
    _write_json(output / "frozen-plan.json", {**frozen, "waves": plan["waves"]})

    evidence: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    backend = None
    stopped_after_failure = False
    try:
        backend = JevBackend(instruction_profile="ablation_v1", max_requests=REQUEST_CAP,
                             max_estimated_input_tokens=TOKEN_CAP, max_in_flight=MAX_IN_FLIGHT)
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
                        evidence.append(score(row, reply))
            except Exception as exc:
                # Error bodies may contain provider data; persist only the type.
                failures.extend({"request_id": request.request_id, "status": type(exc).__name__}
                                for request in requests)
            _write_json(output / "raw-evidence.json", evidence)
            if failures:
                stopped_after_failure = True
                for later in plan["waves"][wave_index + 1:]:
                    for row in later:
                        request = row["request"]
                        request_id = f"r{request['round_index']:03d}:{request['stage']}:{request['agent_id']}"
                        failures.append({"request_id": request_id,
                                         "status": "not_dispatched_after_prior_wave_failure"})
                break
        status = "COMPLETE" if not failures and len(evidence) == REQUEST_CAP else "PARTIAL"
    except Exception as exc:
        status = "BLOCKED" if not evidence else "PARTIAL"
        failures.append({"status": type(exc).__name__, "phase": "backend_initialization_or_run"})
    finally:
        if backend is not None:
            await backend.aclose()
    result = {"status": status, "model": MODEL, "plan_hash": plan["orderedwirehash"],
              "evidence_rows": len(evidence), "failed_rows": len(failures),
              "stopped_after_failure": stopped_after_failure,
              "budget_usage": backend.budget_usage if backend is not None else None,
              "window_timings": list(backend.window_timings) if backend is not None else [],
              "transport_timings": list(backend.transport_timings) if backend is not None else [],
              "summary": summarize(evidence, plan)}
    _write_json(output / "failures.json", failures)
    _write_json(output / "result.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execute", action="store_true", help="send the frozen plan to Jev; preview is offline")
    args = parser.parse_args(argv)
    output = args.output_dir
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    plan = build_plan()
    if args.execute:
        result = asyncio.run(_execute(plan, output))
    else:
        preview = {key: value for key, value in plan.items() if key != "waves"}
        preview.update({"preview": True, "wave_count": len(plan["waves"]),
                        "wire_preview": plan["waves"][0],
                        "output_note": "No backend or environment file was loaded; zero calls sent."})
        _write_json(output / "preview.json", preview)
        result = preview
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if not args.execute or result.get("status") == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Barrier-synchronous coordinator and versioned economic journal for paper runs."""
from __future__ import annotations
import asyncio
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import shutil
import uuid
from typing import Any, Sequence

from capclaims.domain.types import Candidate
from capclaims.engine.protocol import DecisionRequest, DecisionReply, chosen_candidate, validate_reply, digest_json
from capclaims.randomness import canonical_json, exogenous_value
from capclaims.paper import PROFILE, SCHEMA_VERSION
from capclaims.paper.auction import PaperAuctionResult, PaperTrade, clear_paper_auction
from capclaims.paper.codec import decode_prefix, prefix_mask
from capclaims.paper.config import PaperConfig
from capclaims.paper.groups import generate_g3_schedule
from capclaims.paper.observations import add_order_choice_context, build_paper_observation
from capclaims.paper.interface import build_candidates, numeric_observation
from capclaims.paper.orders import PaperOrder, validate_portfolio
from capclaims.paper.settlement import settle_paper_round


@dataclass(frozen=True)
class PaperRunResult:
    status: str
    rounds: int
    formal_rounds: int
    final_hash: str
    events: tuple[dict[str, Any], ...]


def _candidate_set(values: Sequence[int | str], observation: dict) -> tuple[Candidate, ...]:
    return build_candidates(values, observation)


async def _wave(backend: Any, round_index: int, specs: Sequence[tuple[str, str, dict, Sequence[int | str]]],
                seed: int, seed_index: int, audits: list[dict], timeout: float) -> dict[str, int | str]:
    if not specs:
        return {}
    requests = [DecisionRequest(agent, round_index, stage, obs, _candidate_set(values, obs))
                for agent, stage, obs, values in specs]
    if len({r.request_id for r in requests}) != len(requests):
        raise ValueError("paper decision wave contains duplicate request IDs")
    replies = await asyncio.wait_for(backend.decide_many(requests), timeout=timeout)
    if len(replies) != len(requests):
        raise ValueError("incomplete paper decision wave")
    by_id = {reply.request_id: reply for reply in replies}
    if len(by_id) != len(replies) or set(by_id) != {r.request_id for r in requests}:
        raise ValueError("paper wave replies have missing or duplicate identities")
    decisions: dict[str, int | str] = {}
    staged_audits = []
    for request in requests:
        reply = by_id[request.request_id]
        validate_reply(request, reply)
        selected = chosen_candidate(request, reply, seed, seed_index)
        decisions[request.agent_id] = selected.value
        staged_audits.append({
            "request_id": request.request_id, "agent_id": request.agent_id,
            "round_index": round_index, "stage": request.stage,
            "observation": dict(request.observation),
            "candidates": [asdict(c) for c in request.candidates],
            "observation_hash": request.observation_hash,
            "candidate_set_hash": request.candidate_set_hash,
            "raw_scores": dict(reply.raw_scores), "probabilities": dict(reply.probabilities),
            "backend": reply.backend, "model_revision": reply.model_revision,
            "usage": dict(reply.usage), "telemetry": dict(reply.timing),
            "selected": selected.value,
        })
    audits.extend(staged_audits)
    return decisions


def _digits(value: int, minimum: int, maximum: int) -> tuple[int, int, int]:
    if not minimum <= value <= maximum:
        raise ValueError("selected decimal value outside legal bounds")
    return value // 100, value // 10 % 10, value % 10


async def _choose_number(backend: Any, agents: Sequence[str], *, round_index: int, stage_prefix: str,
                         observations: dict[str, dict], minimum: dict[str, int], maximum: dict[str, int],
                         seed: int, seed_index: int, audits: list[dict], timeout: float) -> dict[str, int]:
    prefixes = {agent: "" for agent in agents}
    for place, suffix in enumerate(("h", "t", "u")):
        specs = []
        for agent in agents:
            lo, hi, prefix = minimum[agent], maximum[agent], prefixes[agent]
            values = prefix_mask(prefix, lo, hi)
            obs = numeric_observation(observations[agent], prefix, lo, hi, suffix)
            if obs.get("decision_interface", "legacy_v1") == "legacy_v1":
                obs["legal_prefix_mask"] = list(values)
            specs.append((agent, f"{stage_prefix}:{suffix}", obs, values))
        chosen = await _wave(backend, round_index, specs, seed, seed_index, audits, timeout)
        for agent, digit in chosen.items():
            prefixes[agent] += str(digit)
    values = {agent: decode_prefix(tuple(int(c) for c in prefixes[agent])) for agent in agents}
    for agent in agents:
        if not minimum[agent] <= values[agent] <= maximum[agent]:
            raise ValueError("paper decimal prefix produced out-of-domain value")
    return values


def _history_record(event: dict) -> dict:
    return {"round": event["round_index"] + 1, "own_p": None, "own_q": None,
            "own_d": None, "orders": None, "profit_cents": None, "net_quantity": None,
            "mcp_cents": None}


def _write_run_files(output: Path, manifest: dict, events: list[dict], audits: list[dict]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for filename, rows in (("economic.jsonl", events), ("audit.jsonl", audits)):
        target = output / filename
        temp = output / f"{filename}.tmp"
        temp.write_text("".join(canonical_json(x) + "\n" for x in rows), encoding="utf-8")
        os.replace(temp, target)
    temp = output / "manifest.json.tmp"
    temp.write_text(json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8")
    os.replace(temp, output / "manifest.json")


async def run_paper_session(config: PaperConfig, backend: Any, output_dir: str | Path,
                            wave_timeout_seconds: float = 120.0) -> PaperRunResult:
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"paper output directory is not empty: {output}")
    agents = tuple(f"agent_{i:02d}" for i in range(config.n_agents))
    schedule = (generate_g3_schedule(agents, config.total_rounds, config.master_seed)
                if config.treatment == "G3" else ())
    history: dict[str, list[dict]] = {agent: [] for agent in agents}
    audits: list[dict] = []
    events: list[dict] = []
    previous_hash = "0" * 64
    base_manifest = {
        "profile": PROFILE, "schema_version": SCHEMA_VERSION, "status": "INCOMPLETE",
        "acceptance_level": "adapted_ten_agent_pilot" if config.n_agents == 10 else "custom_ai_pilot",
        "config": asdict(config), "total_rounds": config.total_rounds,
        "formal_rounds": config.analyzed_rounds,
        "deviations": ["AI decision makers", f"population n={config.n_agents}",
            "decimal-prefix interface changes joint choice geometry", "declared own-history window",
            "seeded order-slot tie rank", "self-match allowed as unverified interpretation",
            "nonbinding internal bookkeeping grant assumes no participant credit constraint"],
        "assumptions": {"equality_crossing": config.equality_crossing,
            "self_match_policy": config.self_match_policy,
            "g3_repeat_limit_includes_warmup": config.treatment == "G3",
            "cash_grant_cents_per_agent_round": 501000500},
        "schedule": [[list(group) for group in groups] for groups in schedule],
        "committed_rounds": 0, "decision_audit_count": 0,
    }
    output.mkdir(parents=True, exist_ok=True)
    _write_run_files(output, base_manifest, events, audits)
    try:
        for r in range(config.total_rounds):
            p = {a: exogenous_value(config.master_seed, config.seed_index, r, a, "p") for a in agents}
            d = {a: exogenous_value(config.master_seed, config.seed_index, r, a, "D") for a in agents}
            procurement_obs = {
                a: build_paper_observation(agent_id=a, round_index=r, treatment=config.treatment,
                    n_agents=config.n_agents, wholesale_price=config.wholesale_price, p=p[a],
                    history=history[a][-config.history_window:] if config.history_window else (),
                    partial_orders=(), phase="procurement", horizon=config.total_rounds,
                    history_window=config.history_window, decision_field="purchase",
                    equality_crossing=config.equality_crossing, self_match_policy=config.self_match_policy,
                    decision_interface=config.decision_interface) for a in agents}
            q = await _choose_number(backend, agents, round_index=r, stage_prefix="paper:purchase",
                observations=procurement_obs, minimum={a: 0 for a in agents}, maximum={a: 400 for a in agents},
                seed=config.master_seed, seed_index=config.seed_index, audits=audits,
                timeout=wave_timeout_seconds)
            all_orders: list[PaperOrder] = []
            partial = {a: [] for a in agents}
            sell_remaining = {a: q[a] for a in agents}
            active = [] if config.treatment == "NV" else list(agents)
            for slot in range(config.maximum_orders):
                if not active:
                    break
                observations = {a: build_paper_observation(agent_id=a, round_index=r,
                    treatment=config.treatment, n_agents=config.n_agents,
                    wholesale_price=config.wholesale_price, p=p[a],
                    history=history[a][-config.history_window:] if config.history_window else (),
                    partial_orders=[asdict(o) for o in partial[a]], phase="orders", q=q[a], d=d[a], slot=slot,
                    horizon=config.total_rounds, history_window=config.history_window, decision_field="side",
                    equality_crossing=config.equality_crossing, self_match_policy=config.self_match_policy,
                    decision_interface=config.decision_interface)
                    for a in active}
                side_values = {a: ("buy", "STOP", "sell") if sell_remaining[a] > 0 else ("buy", "STOP")
                               for a in active}
                side_specs = [(a, f"paper:slot{slot}:side",
                    dict(observations[a], slot=slot, legal_actions=list(side_values[a])), side_values[a])
                    for a in active]
                sides = await _wave(backend, r, side_specs, config.master_seed, config.seed_index,
                                    audits, wave_timeout_seconds)
                constructing = [a for a in active if sides[a] != "STOP"]
                active = constructing
                if not active:
                    break
                qty_max = {a: sell_remaining[a] if sides[a] == "sell" else 999 for a in active}
                qty_observations = {a: add_order_choice_context(observations[a],
                    decision_field="quantity", own_side=sides[a]) for a in active}
                qty = await _choose_number(backend, active, round_index=r,
                    stage_prefix=f"paper:slot{slot}:qty", observations=qty_observations,
                    minimum={a: 1 for a in active}, maximum=qty_max, seed=config.master_seed,
                    seed_index=config.seed_index, audits=audits, timeout=wave_timeout_seconds)
                price_observations = {a: add_order_choice_context(observations[a],
                    decision_field="price", own_side=sides[a], own_qty=qty[a]) for a in active}
                price = await _choose_number(backend, active, round_index=r,
                    stage_prefix=f"paper:slot{slot}:price", observations=price_observations,
                    minimum={a: 1 for a in active}, maximum={a: 999 for a in active}, seed=config.master_seed,
                    seed_index=config.seed_index, audits=audits, timeout=wave_timeout_seconds)
                for agent in active:
                    order = PaperOrder(f"r{r:03d}:{agent}:slot{slot}", agent,
                                       str(sides[agent]), qty[agent], price[agent])
                    partial[agent].append(order)
                    if sides[agent] == "sell":
                        sell_remaining[agent] -= qty[agent]
            for agent in agents:
                validate_portfolio(tuple(partial[agent]), agent, q[agent], config.maximum_orders)
                all_orders.extend(partial[agent])

            groups = (() if config.treatment == "NV" else ((agents,) if config.treatment == "GA" else schedule[r]))
            market_records = []
            trades: list[PaperTrade] = []
            allocations = {a: 0 for a in agents}
            ranks = {}
            for market_index, members in enumerate(groups):
                member_set = set(members)
                market_orders = [o for o in all_orders if o.owner in member_set]
                for owner in members:
                    validate_portfolio(tuple(o for o in market_orders if o.owner == owner), owner, q[owner])
                result = clear_paper_auction(market_orders, config.master_seed, config.seed_index,
                    r, config.equality_crossing, config.self_match_policy)
                trades.extend(result.trades)
                allocations.update(result.net_allocations)
                ranks.update(result.tie_ranks)
                market_records.append({"market_index": market_index, "members": list(members),
                    "price_cents": result.price_cents, "trades": [asdict(t) for t in result.trades],
                    "tie_ranks": result.tie_ranks, "self_volume": result.self_volume})
            combined = PaperAuctionResult(None, tuple(trades), allocations, ranks,
                                          sum(item["self_volume"] for item in market_records))
            settled = settle_paper_round({a: (p[a], d[a], q[a]) for a in agents}, combined,
                                         config.wholesale_price)
            event = {
                "round_index": r, "warmup": r < config.warmup_rounds,
                "exogenous": {a: {"p": p[a], "D": d[a]} for a in agents},
                "procurement": q, "orders": [asdict(o) for o in all_orders],
                "groups": [list(group) for group in groups], "markets": market_records,
                "net_allocations": allocations,
                "settlement": {"profits_cents": settled.profits_cents, "fulfilled": settled.fulfilled,
                    "leftovers": settled.leftovers, "shortages": settled.shortages,
                    "entries": [asdict(e) for e in settled.entries],
                    "issued_units": settled.issued_units, "redeemed_units": settled.redeemed_units,
                    "burned_units": settled.burned_units, "cash_grant_cents": settled.cash_grant_cents},
                "previous_hash": previous_hash,
            }
            event["event_hash"] = digest_json(event)
            previous_hash = event["event_hash"]
            events.append(event)
            order_summary = {a: [asdict(o) for o in all_orders if o.owner == a] for a in agents}
            mcp_by_agent = {a: next((m["price_cents"] for m in market_records if a in m["members"]), None) for a in agents}
            for a in agents:
                history[a].append({"round": r + 1, "p": p[a], "Q": q[a], "D": d[a],
                    "orders": order_summary[a], "profit_cents": settled.profits_cents[a],
                    "net_quantity": allocations[a], "mcp_cents": mcp_by_agent[a]})
            snapshot = dict(base_manifest, status="RUNNING", committed_rounds=len(events),
                            decision_audit_count=len(audits),
                            model_backends=sorted({x["backend"] for x in audits}))
            _write_run_files(output, snapshot, events, audits)
    except BaseException as exc:
        incomplete = dict(base_manifest, status="INCOMPLETE", committed_rounds=len(events),
                          decision_audit_count=len(audits),
                          error=f"{type(exc).__name__}: {exc}")
        _write_run_files(output, incomplete, events, audits)
        raise
    finally:
        close = getattr(backend, "aclose", None)
        if close:
            await close()
    manifest = dict(base_manifest, status="COMPLETE", committed_rounds=len(events),
                    decision_audit_count=len(audits),
                    model_backends=sorted({x["backend"] for x in audits}))
    _write_run_files(output, manifest, events, audits)
    return PaperRunResult("COMPLETE", len(events), config.analyzed_rounds, previous_hash, tuple(events))

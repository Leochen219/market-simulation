"""Async phase barriers, single-writer settlement, and immutable run output."""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from capclaims.agents.candidates import (
    price_candidates, purchase_candidates, quantity_candidates, side_candidates,
)
from capclaims.agents.observations import build_observation
from capclaims.config import RunConfig
from capclaims.domain.auction import clear_auction
from capclaims.domain.settlement import settle_round
from capclaims.domain.types import FrozenRoundState, Order
from capclaims.engine.protocol import DecisionReply, DecisionRequest, chosen_candidate, validate_reply
from capclaims.randomness import canonical_json, exogenous_value

WAVE_TIMEOUT_SECONDS = 60.0


@dataclass(frozen=True)
class RunResult:
    status: str
    rounds: int
    trades: int
    volume: int
    profit_cents: int
    final_hash: str | None
    observed_batch_sizes: tuple[int, ...] = ()


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, sort_keys=True, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _manifest(config: RunConfig, backend: Any, status: str, **extra: Any) -> dict[str, Any]:
    cls = type(backend)
    data = {
        "format_version": 1,
        "status": status,
        "config": asdict(config),
        "backend": cls.__name__,
        "policy": config.policy,
        "model_revision": getattr(backend, "revision", f"policy:{config.policy}" if config.policy not in {"jev", "laya"} else None),
        "python": __import__("sys").version.split()[0],
        "package_version": "0.1.0",
    }
    data.update(extra)
    return data


def _event_row(seq: int, previous_hash: str, event: Mapping[str, Any]) -> dict[str, Any]:
    core = {"seq": seq, "prev_hash": previous_hash, "event": event}
    digest = hashlib.sha256(canonical_json(core).encode("utf-8")).hexdigest()
    return {**core, "hash": digest}


def _request(config: RunConfig, agent_id: str, round_index: int, stage: str,
             p: int, history: Sequence[Mapping[str, Any]], candidates: tuple,
             *, q: int | None = None, d: int | None = None, side: str | None = None,
             qty: int | None = None) -> DecisionRequest:
    observation = build_observation(
        config, round_index, agent_id, p, history, phase=stage,
        q=q, d=d, side=side, qty=qty,
    )
    return DecisionRequest(agent_id, round_index, stage, observation, candidates)


async def _decision_wave(backend: Any, requests: Sequence[DecisionRequest], timeout: float) -> dict[str, DecisionReply]:
    if not requests:
        return {}
    raw = await asyncio.wait_for(backend.decide_many(tuple(requests)), timeout=timeout)
    replies = list(raw)
    if len(replies) != len(requests):
        raise ValueError(f"backend returned {len(replies)} replies for {len(requests)} requests")
    expected = {request.request_id: request for request in requests}
    result: dict[str, DecisionReply] = {}
    for reply in replies:
        if not isinstance(reply, DecisionReply) or reply.request_id not in expected:
            raise ValueError("backend returned an unknown or malformed reply")
        if reply.request_id in result:
            raise ValueError(f"backend returned duplicate reply for {reply.request_id}")
        validate_reply(expected[reply.request_id], reply)
        result[reply.request_id] = reply
    if set(result) != set(expected):
        raise ValueError("backend omitted one or more required replies")
    return result


def _decision_event(request: DecisionRequest, reply: DecisionReply, chosen_id: str) -> dict[str, Any]:
    return {
        "request_id": request.request_id,
        "agent_id": request.agent_id,
        "round_index": request.round_index,
        "stage": request.stage,
        "observation": dict(request.observation),
        "observation_hash": request.observation_hash,
        "candidates": [asdict(candidate) for candidate in request.candidates],
        "candidate_set_hash": request.candidate_set_hash,
        "probabilities": dict(reply.probabilities),
        "chosen_id": chosen_id,
    }


async def run_session(
    config: RunConfig, backend: Any, output_dir: str | Path,
    *, wave_timeout_seconds: float = WAVE_TIMEOUT_SECONDS,
) -> RunResult:
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"run output already exists: {output}")
    if wave_timeout_seconds <= 0:
        raise ValueError("wave_timeout_seconds must be positive")
    output.mkdir(parents=True, exist_ok=False)
    economic_path, telemetry_path = output / "economic.jsonl", output / "telemetry.jsonl"
    economic_path.touch(exist_ok=False)
    telemetry_path.touch(exist_ok=False)
    manifest_path = output / "manifest.json"
    manifest = _manifest(config, backend, "RUNNING")
    _write_json(manifest_path, manifest)

    history: dict[str, list[dict[str, Any]]] = {f"agent_{index:02d}": [] for index in range(config.n_agents)}
    previous_hash = "0" * 64
    trades_count = volume = profit_total = completed_rounds = seq = 0
    batch_sizes: list[int] = []
    seen_batch_groups: set[str] = set()
    try:
        for round_index in range(config.rounds):
            agent_ids = sorted(history)
            p = {agent: exogenous_value(config.master_seed, config.seed_index, round_index, agent, "p") for agent in agent_ids}
            d = {agent: exogenous_value(config.master_seed, config.seed_index, round_index, agent, "D") for agent in agent_ids}
            events: list[dict[str, Any]] = []
            q: dict[str, int] = {}
            actions: dict[str, dict[str, Any]] = {agent: {} for agent in agent_ids}

            async def run_wave(stage: str, requests: list[DecisionRequest]) -> dict[str, Any]:
                replies = await _decision_wave(backend, requests, wave_timeout_seconds)
                selected: dict[str, Any] = {}
                for request in sorted(requests, key=lambda item: item.agent_id):
                    reply = replies[request.request_id]
                    candidate = chosen_candidate(request, reply, config.master_seed, config.seed_index)
                    selected[request.agent_id] = candidate
                    events.append(_decision_event(request, reply, candidate.candidate_id))
                    telemetry = {
                        "timestamp": _utc_now(), "type": "decision_reply",
                        "request_id": request.request_id, "agent_id": request.agent_id,
                        "round_index": request.round_index, "stage": stage,
                        "backend": reply.backend, "model_revision": reply.model_revision,
                        "raw_scores": dict(reply.raw_scores), "usage": dict(reply.usage),
                        "timing": dict(reply.timing),
                    }
                    with telemetry_path.open("a", encoding="utf-8", newline="\n") as handle:
                        handle.write(canonical_json(telemetry) + "\n")
                    forward_sizes = reply.timing.get("forward_batch_sizes", [])
                    if forward_sizes:
                        group_id = reply.timing.get("group_id")
                        if not isinstance(group_id, str) or not group_id:
                            raise ValueError("batched inference telemetry requires a call group id")
                        if group_id not in seen_batch_groups:
                            batch_sizes.extend(forward_sizes)
                            seen_batch_groups.add(group_id)
                return selected

            requests = [_request(config, agent, round_index, "purchase", p[agent], history[agent], purchase_candidates()) for agent in agent_ids]
            selected = await run_wave("purchase", requests)
            for agent, candidate in selected.items():
                q[agent] = int(candidate.value)

            if config.market_open:
                requests = [_request(config, agent, round_index, "trade_side", p[agent], history[agent],
                                     side_candidates(q[agent]), q=q[agent], d=d[agent]) for agent in agent_ids]
                sides = await run_wave("trade_side", requests)
                for agent, candidate in sides.items():
                    actions[agent]["side"] = str(candidate.value)
                active = [agent for agent in agent_ids if actions[agent]["side"] != "hold"]
                qty_requests = [_request(config, agent, round_index, "trade_qty", p[agent], history[agent],
                                          quantity_candidates(actions[agent]["side"], q[agent]), q=q[agent], d=d[agent],
                                          side=actions[agent]["side"]) for agent in active]
                quantities = await run_wave("trade_qty", qty_requests)
                for agent, candidate in quantities.items():
                    actions[agent]["qty"] = int(candidate.value)
                price_requests = [_request(config, agent, round_index, "trade_price", p[agent], history[agent],
                                           price_candidates(), q=q[agent], d=d[agent], side=actions[agent]["side"],
                                           qty=actions[agent]["qty"]) for agent in active]
                prices = await run_wave("trade_price", price_requests)
                for agent, candidate in prices.items():
                    actions[agent]["price"] = int(candidate.value)

            orders = []
            for agent in agent_ids:
                if actions[agent].get("side") in {"buy", "sell"}:
                    orders.append(Order(agent, actions[agent]["side"], actions[agent]["qty"], actions[agent]["price"] * 100))
            auction = clear_auction(orders, config.master_seed, config.seed_index, round_index)
            auction = type(auction)(
                auction.price_cents, auction.trades,
                {agent: auction.net_allocations.get(agent, 0) for agent in agent_ids},
                auction.tie_ranks,
            )
            settlement = settle_round(FrozenRoundState(p, d, q), auction, config.wholesale_price, config.initial_cash_cents)
            event = {
                "round_index": round_index,
                "p": p, "d": d, "q": q,
                "actions": actions,
                "decisions": sorted(events, key=lambda item: (item["stage"], item["agent_id"])),
                "orders": [asdict(order) for order in orders],
                "auction": asdict(auction),
                "settlement": asdict(settlement),
            }
            row = _event_row(seq, previous_hash, event)
            with economic_path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(canonical_json(row) + "\n")
            previous_hash = row["hash"]
            seq += 1
            completed_rounds += 1
            trades_count += len(auction.trades)
            volume += sum(trade.qty for trade in auction.trades)
            profit_total += sum(settlement.profits_cents.values())
            for agent in agent_ids:
                own_net = auction.net_allocations[agent]
                own_fills = sum(t.qty for t in auction.trades if t.buyer_id == agent or t.seller_id == agent)
                history[agent].append({
                    "p": p[agent], "D": d[agent], "Q": q[agent], "A": own_net,
                    "profit_cents": settlement.profits_cents[agent], "m_cents": auction.price_cents,
                    "fillqty": own_fills, "side": actions[agent].get("side", "hold"),
                    "qty": actions[agent].get("qty", 0), "price": actions[agent].get("price"),
                })

        manifest = _manifest(config, backend, "COMPLETE", rounds=completed_rounds,
                             economic_final_hash=previous_hash, observed_batch_sizes=batch_sizes)
        _write_json(manifest_path, manifest)
        return RunResult("COMPLETE", completed_rounds, trades_count, volume, profit_total, previous_hash, tuple(batch_sizes))
    except asyncio.CancelledError:
        with telemetry_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(canonical_json({"timestamp": _utc_now(), "type": "run_cancelled"}) + "\n")
        manifest = _manifest(config, backend, "INCOMPLETE", rounds=completed_rounds,
                             economic_final_hash=previous_hash, error="CancelledError: run cancelled")
        _write_json(manifest_path, manifest)
        raise
    except Exception as exc:
        with telemetry_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(canonical_json({"timestamp": _utc_now(), "type": "run_error", "error": f"{type(exc).__name__}: {exc}"}) + "\n")
        manifest = _manifest(config, backend, "INCOMPLETE", rounds=completed_rounds,
                             economic_final_hash=previous_hash, error=f"{type(exc).__name__}: {exc}")
        _write_json(manifest_path, manifest)
        return RunResult("INCOMPLETE", completed_rounds, trades_count, volume, profit_total,
                         previous_hash if completed_rounds else None, tuple(batch_sizes))
    finally:
        await backend.aclose()

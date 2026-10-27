"""Validate the economic hash chain and recompute every committed round offline."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from capclaims.agents.candidates import price_candidates, purchase_candidates, quantity_candidates, side_candidates
from capclaims.agents.observations import build_observation
from capclaims.config import RunConfig
from capclaims.domain.auction import clear_auction
from capclaims.domain.settlement import settle_round
from capclaims.domain.types import FrozenRoundState, Order
from capclaims.engine.protocol import chosen_candidate, digest_json
from capclaims.randomness import canonical_json, exogenous_value


@dataclass(frozen=True)
class ReplayResult:
    rounds: int
    trades: int
    volume: int
    profit_cents: int
    final_hash: str


def _selected_value(
    decision: dict[str, Any], agent_id: str, round_index: int, stage: str,
    observation: dict[str, Any], candidates: tuple, config: RunConfig,
) -> Any:
    if decision.get("agent_id") != agent_id or decision.get("round_index") != round_index or decision.get("stage") != stage:
        raise ValueError(f"replay decision identity mismatch for {agent_id}/{stage}")
    if decision.get("request_id") != f"r{round_index:03d}:{stage}:{agent_id}":
        raise ValueError("replay request id mismatch")
    if decision.get("observation") != observation or decision.get("observation_hash") != digest_json(observation):
        raise ValueError("replay observation mismatch")
    expected_candidates = [asdict(item) for item in candidates]
    if decision.get("candidates") != expected_candidates:
        raise ValueError("replay legal candidate list mismatch")
    if decision.get("candidate_set_hash") != digest_json(expected_candidates):
        raise ValueError("replay candidate set hash mismatch")
    probabilities = decision.get("probabilities")
    ids = [candidate.candidate_id for candidate in candidates]
    if not isinstance(probabilities, dict) or set(probabilities) != set(ids):
        raise ValueError("replay decision scores omit or add candidates")
    try:
        numeric = [float(probabilities[key]) for key in ids]
    except (TypeError, ValueError) as exc:
        raise ValueError("replay probability is not numeric") from exc
    if any(not math.isfinite(value) or value < 0 for value in numeric) or abs(math.fsum(numeric) - 1.0) > 1e-6:
        raise ValueError("replay probability distribution is invalid")
    top = max(numeric)
    tied = [candidate for candidate in candidates if float(probabilities[candidate.candidate_id]) == top]
    from capclaims.randomness import decision_rank
    selected = min(tied, key=lambda candidate: decision_rank(
        config.master_seed, config.seed_index, round_index, agent_id, stage, candidate.candidate_id))
    if decision.get("chosen_id") != selected.candidate_id:
        raise ValueError("replay selected candidate does not match deterministic argmax")
    return selected.value


def replay(run_dir: str | Path) -> ReplayResult:
    root = Path(run_dir)
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read run manifest: {exc}") from exc
    if manifest.get("status") != "COMPLETE":
        raise ValueError("only COMPLETE runs can be replayed")
    try:
        config = RunConfig(**manifest["config"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("manifest contains invalid run configuration") from exc
    previous_hash, rounds, trade_count, volume, profit_total = "0" * 64, 0, 0, 0, 0
    history: dict[str, list[dict[str, Any]]] = {f"agent_{index:02d}": [] for index in range(config.n_agents)}
    try:
        lines = (root / "economic.jsonl").read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"cannot read economic log: {exc}") from exc
    for expected_seq, line in enumerate(lines):
        try:
            row = json.loads(line)
            event = row["event"]
            core = {"seq": row["seq"], "prev_hash": row["prev_hash"], "event": event}
            expected_hash = hashlib.sha256(canonical_json(core).encode("utf-8")).hexdigest()
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError(f"malformed economic event {expected_seq}") from exc
        if row["seq"] != expected_seq or row["prev_hash"] != previous_hash or row.get("hash") != expected_hash:
            raise ValueError(f"economic hash chain failed at event {expected_seq}")
        round_index = rounds
        if event.get("round_index") != round_index:
            raise ValueError("replay round sequence mismatch")
        agents = sorted(history)
        p = {agent: exogenous_value(config.master_seed, config.seed_index, round_index, agent, "p") for agent in agents}
        d = {agent: exogenous_value(config.master_seed, config.seed_index, round_index, agent, "D") for agent in agents}
        if event.get("p") != p or event.get("d") != d:
            raise ValueError("replay exogenous variables do not match seed")
        decisions = event.get("decisions")
        if not isinstance(decisions, list):
            raise ValueError("replay decisions missing")
        decision_map = {(decision.get("stage"), decision.get("agent_id")): decision for decision in decisions}
        if len(decision_map) != len(decisions):
            raise ValueError("duplicate replay decisions")
        used: set[tuple[str, str]] = set()
        q: dict[str, int] = {}
        actions: dict[str, dict[str, Any]] = {agent: {} for agent in agents}

        def choose(agent: str, stage: str, candidates: tuple, **extra: Any) -> Any:
            observation = build_observation(config, round_index, agent, p[agent], history[agent], phase=stage, **extra)
            key = (stage, agent)
            if key not in decision_map:
                raise ValueError(f"missing replay decision {stage}/{agent}")
            decision = decision_map[key]
            used.add(key)
            value = _selected_value(decision, agent, round_index, stage, observation, candidates, config)
            return value

        for agent in agents:
            q[agent] = int(choose(agent, "purchase", purchase_candidates()))
        if config.market_open:
            for agent in agents:
                actions[agent]["side"] = str(choose(agent, "trade_side", side_candidates(q[agent]), q=q[agent], d=d[agent]))
            for agent in agents:
                if actions[agent]["side"] != "hold":
                    actions[agent]["qty"] = int(choose(
                        agent, "trade_qty", quantity_candidates(actions[agent]["side"], q[agent]),
                        q=q[agent], d=d[agent], side=actions[agent]["side"],
                    ))
            for agent in agents:
                if actions[agent]["side"] != "hold":
                    actions[agent]["price"] = int(choose(
                        agent, "trade_price", price_candidates(), q=q[agent], d=d[agent],
                        side=actions[agent]["side"], qty=actions[agent]["qty"],
                    ))
        if used != set(decision_map):
            raise ValueError("economic log contains extra replay decisions")
        canonical_decisions = sorted(decisions, key=lambda item: (item["stage"], item["agent_id"]))
        if decisions != canonical_decisions:
            raise ValueError("replay decisions are not in canonical order")
        orders = [Order(agent, actions[agent]["side"], actions[agent]["qty"], actions[agent]["price"] * 100)
                  for agent in agents if actions[agent].get("side") in {"buy", "sell"}]
        auction = clear_auction(orders, config.master_seed, config.seed_index, round_index)
        auction = type(auction)(
            auction.price_cents, auction.trades,
            {agent: auction.net_allocations.get(agent, 0) for agent in agents},
            auction.tie_ranks,
        )
        settlement = settle_round(FrozenRoundState(p, d, q), auction, config.wholesale_price, config.initial_cash_cents)
        rebuilt = {
            "round_index": round_index, "p": p, "d": d, "q": q, "actions": actions,
            "decisions": canonical_decisions,
            "orders": [asdict(order) for order in orders], "auction": asdict(auction),
            "settlement": asdict(settlement),
        }
        if canonical_json(rebuilt) != canonical_json(event):
            raise ValueError(f"recomputed market or ledger differs at round {round_index}")
        previous_hash = row["hash"]
        rounds += 1
        trade_count += len(auction.trades)
        volume += sum(trade.qty for trade in auction.trades)
        profit_total += sum(settlement.profits_cents.values())
        for agent in agents:
            own_fills = sum(t.qty for t in auction.trades if t.buyer_id == agent or t.seller_id == agent)
            history[agent].append({
                "p": p[agent], "D": d[agent], "Q": q[agent], "A": auction.net_allocations[agent],
                "profit_cents": settlement.profits_cents[agent], "m_cents": auction.price_cents,
                "fillqty": own_fills, "side": actions[agent].get("side", "hold"),
                "qty": actions[agent].get("qty", 0), "price": actions[agent].get("price"),
            })
    if (not rounds or rounds != config.rounds or rounds != manifest.get("rounds")
            or previous_hash != manifest.get("economic_final_hash")):
        raise ValueError("manifest final hash or round count does not match configured replay")
    return ReplayResult(rounds, trade_count, volume, profit_total, previous_hash)

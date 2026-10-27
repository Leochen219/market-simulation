"""Strict inference-free verification of schema-2 paper economic journals."""
from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from capclaims.domain.types import Candidate
from capclaims.engine.protocol import DecisionRequest, DecisionReply, validate_reply, chosen_candidate, digest_json
from capclaims.randomness import canonical_json, exogenous_value
from capclaims.paper import PROFILE, SCHEMA_VERSION
from capclaims.paper.auction import PaperAuctionResult, clear_paper_auction
from capclaims.paper.config import PaperConfig
from capclaims.paper.groups import generate_g3_schedule
from capclaims.paper.codec import prefix_mask
from capclaims.paper.observations import add_order_choice_context, build_paper_observation
from capclaims.paper.interface import build_candidates, numeric_observation
from capclaims.paper.orders import PaperOrder, validate_portfolio
from capclaims.paper.settlement import settle_paper_round


@dataclass(frozen=True)
class PaperReplayResult:
    rounds: int
    formal_rounds: int
    trades: int
    volume: int
    profit_cents: int
    final_hash: str
    events: tuple[dict[str, Any], ...]


def _lines(path: Path) -> list[dict]:
    try:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid replay file {path.name}: {exc}") from exc


def _verified_decisions(audits: list[dict], config: PaperConfig, total_rounds: int,
                        events: list[dict]) -> dict[tuple[int, str, str], int | str]:
    decisions = {}
    agents = {f"agent_{i:02d}" for i in range(config.n_agents)}
    for item in audits:
        stage = item.get("stage")
        agent = item.get("agent_id")
        r = item.get("round_index")
        if agent not in agents or isinstance(r, bool) or not isinstance(r, int) or not 0 <= r < total_rounds:
            raise ValueError("replay audit contains an invalid agent or round")
        if not isinstance(stage, str) or not stage.startswith("paper:"):
            raise ValueError("replay audit contains a non-paper stage")
        if stage.startswith("paper:purchase:") and "own_d" in item.get("observation", {}):
            raise ValueError("replay procurement observation reveals demand")
        candidate_rows = item.get("candidates")
        if not isinstance(candidate_rows, list) or not candidate_rows:
            raise ValueError("replay audit lacks candidates")
        candidates = tuple(Candidate(**row) for row in candidate_rows)
        request = DecisionRequest(agent, r, stage, item["observation"], candidates)
        reply = DecisionReply(item["request_id"], agent, r, stage, item["observation_hash"],
            item["candidate_set_hash"], item["raw_scores"], item["probabilities"],
            item["backend"], item["model_revision"], item.get("usage", {}), item.get("telemetry", {}))
        validate_reply(request, reply)
        selected = chosen_candidate(request, reply, config.master_seed, config.seed_index).value
        if selected != item.get("selected"):
            raise ValueError("replay selected choice differs from audited scores")
        key = (r, agent, stage)
        if key in decisions:
            raise ValueError("duplicate replay audit stage")
        decisions[key] = selected
        observation = dict(item["observation"])
        if (observation.get("horizon") != config.total_rounds or
                observation.get("n_agents") != config.n_agents or
                observation.get("market_size") != (3 if config.treatment == "G3" else config.n_agents) or
                observation.get("wholesale_price_dollars") != config.wholesale_price or
                observation.get("own_p") != exogenous_value(config.master_seed, config.seed_index, r, agent, "p")):
            raise ValueError("replay observation public parameters or private p mismatch")
        if "other_agents" in observation or "group_members" in observation:
            raise ValueError("replay observation exposes other agents' private data")
        _validate_logged_observation_and_candidates(item, selected, config, events, decisions)
    return decisions


def _validate_audit_completeness(decisions: dict, config: PaperConfig, total_rounds: int) -> None:
    agents = tuple(f"agent_{i:02d}" for i in range(config.n_agents))
    expected = set()
    for r in range(total_rounds):
        for agent in agents:
            for suffix in ("h", "t", "u"):
                expected.add((r, agent, f"paper:purchase:{suffix}"))
            if config.treatment == "NV":
                continue
            for slot in range(config.maximum_orders):
                side_key = (r, agent, f"paper:slot{slot}:side")
                expected.add(side_key)
                side = decisions.get(side_key)
                if side is None:
                    raise ValueError("replay audit is missing an order side decision")
                if side == "STOP":
                    break
                if side not in {"buy", "sell"}:
                    raise ValueError("replay audit has an invalid order side")
                for kind in ("qty", "price"):
                    for suffix in ("h", "t", "u"):
                        expected.add((r, agent, f"paper:slot{slot}:{kind}:{suffix}"))
    if set(decisions) != expected:
        raise ValueError("replay audit has missing or unexpected paper decision stages")


def _orders_from_decisions(r: int, agents: tuple[str, ...], decisions: dict) -> list[PaperOrder]:
    orders = []
    # § SPECIFIED: Session economics serialize submitted portfolios agent-major,
    # preserving each owner's slot order. Replay must reproduce that canonical
    # ordering because order sequence is part of the hashed economic event.
    for agent in agents:
        for slot in range(5):
            side = decisions.get((r, agent, f"paper:slot{slot}:side"))
            if side is None or side == "STOP":
                break
            qty = _number_choice_for_round(r, agent, f"paper:slot{slot}:qty", decisions)
            price = _number_choice_for_round(r, agent, f"paper:slot{slot}:price", decisions)
            orders.append(PaperOrder(f"r{r:03d}:{agent}:slot{slot}", agent, str(side), qty, price))
    return orders


def _number_choice_for_round(r: int, agent: str, prefix: str, decisions: dict) -> int:
    return sum(int(decisions[(r, agent, f"{prefix}:{suffix}")]) * place
               for suffix, place in (("h", 100), ("t", 10), ("u", 1)))


def _validate_logged_observation_and_candidates(item: dict, selected: int | str,
                                                config: PaperConfig, events: list[dict],
                                                decisions: dict[tuple[int, str, str], int | str]) -> None:
    """Rebuild the exact paper observation and candidate domain for this logged step."""
    stage, agent, r = item["stage"], item["agent_id"], item["round_index"]
    event = events[r]
    p, q = event["exogenous"][agent]["p"], event["procurement"][agent]
    d = event["exogenous"][agent]["D"]
    history = []
    for previous in events[:r]:
        price = next((m["price_cents"] for m in previous["markets"] if agent in m["members"]), None)
        own_orders = [o for o in previous["orders"] if o["owner"] == agent]
        history.append({"round": previous["round_index"] + 1, "p": previous["exogenous"][agent]["p"],
            "Q": previous["procurement"][agent], "D": previous["exogenous"][agent]["D"],
            "orders": own_orders, "profit_cents": previous["settlement"]["profits_cents"][agent],
            "net_quantity": previous["net_allocations"][agent], "mcp_cents": price})
    history = history[-config.history_window:] if config.history_window else []
    base = dict(agent_id=agent, round_index=r, treatment=config.treatment, n_agents=config.n_agents,
        wholesale_price=config.wholesale_price, p=p, history=history, horizon=config.total_rounds,
        history_window=config.history_window, equality_crossing=config.equality_crossing,
        self_match_policy=config.self_match_policy, decision_interface=config.decision_interface)
    legal: tuple[int | str, ...]
    if stage.startswith("paper:purchase:"):
        suffix = stage.rsplit(":", 1)[1]
        obs = build_paper_observation(**base, partial_orders=(), phase="procurement",
            decision_field="purchase")
        lo, hi, prefix = 0, 400, ""
        if suffix in {"t", "u"}:
            prefix = _prefix_before_digit(item, decisions, "paper:purchase", suffix)
        place = {"h": "h", "t": "t", "u": "u"}.get(suffix)
        if place is None:
            raise ValueError("replay has invalid procurement digit stage")
        legal = prefix_mask(prefix, lo, hi)
        obs = numeric_observation(obs, prefix, lo, hi, suffix)
        if config.decision_interface == "legacy_v1":
            obs["legal_prefix_mask"] = list(legal)
    else:
        if ":slot" not in stage:
            raise ValueError("replay has invalid order stage")
        tail = stage.split(":slot", 1)[1]
        slot_text, field = tail.split(":", 1)
        slot = int(slot_text)
        prior_orders = [o for o in event["orders"] if o["owner"] == agent and int(o["order_id"].rsplit("slot", 1)[1]) < slot]
        partial = [dict(o) for o in sorted(prior_orders, key=lambda x: x["order_id"])]
        side = None
        qty = None
        if field == "side":
            sell_remaining = q - sum(o["qty"] for o in partial if o["side"] == "sell")
            legal = ("buy", "STOP", "sell") if sell_remaining > 0 else ("buy", "STOP")
            obs = build_paper_observation(**base, partial_orders=partial, phase="orders", q=q, d=d,
                slot=slot, decision_field="side")
            obs["legal_actions"] = list(legal)
        elif field.startswith(("qty:", "price:")):
            kind, suffix = field.split(":", 1)
            side = _choice_for(item, decisions, f"paper:slot{slot}:side")
            if side not in {"buy", "sell"}:
                raise ValueError("replay numeric order stage has no legal side")
            if kind == "qty":
                cap = q - sum(o["qty"] for o in partial if o["side"] == "sell") if side == "sell" else 999
                lo, hi = 1, cap
                obs = build_paper_observation(**base, partial_orders=partial, phase="orders", q=q, d=d,
                    slot=slot, decision_field="quantity", own_side=side)
                obs = add_order_choice_context(obs, decision_field="quantity", own_side=side)
            else:
                qty = _number_choice(item, decisions, f"paper:slot{slot}:qty")
                lo, hi = 1, 999
                obs = build_paper_observation(**base, partial_orders=partial, phase="orders", q=q, d=d,
                    slot=slot, decision_field="price", own_side=side, own_qty=qty)
                obs = add_order_choice_context(obs, decision_field="price", own_side=side, own_qty=qty)
            prefix = _prefix_before_digit(item, decisions, f"paper:slot{slot}:{kind}", suffix)
            if suffix not in {"h", "t", "u"}:
                raise ValueError("replay has invalid numeric digit stage")
            legal = prefix_mask(prefix, lo, hi)
            obs = numeric_observation(obs, prefix, lo, hi, suffix)
            if config.decision_interface == "legacy_v1":
                obs["legal_prefix_mask"] = list(legal)
        else:
            raise ValueError("replay has unknown paper order decision stage")
    expected_candidates = [vars(candidate) for candidate in build_candidates(legal, obs)]
    if item["observation"] != obs or item["candidates"] != expected_candidates:
        raise ValueError("replay decision observation or legal candidates differ from reconstructed state")
    if selected not in legal:
        raise ValueError("replay selected a choice outside reconstructed legal candidates")


def _choice_for(item: dict, decisions: dict, suffix: str) -> str:
    r, agent = item["round_index"], item["agent_id"]
    try:
        return decisions[(r, agent, suffix)]
    except KeyError as exc:
        raise ValueError("replay lacks prior choice") from exc


def _number_choice(item: dict, decisions: dict, prefix: str) -> int:
    digits = [_choice_for(item, decisions, f"{prefix}:{part}") for part in ("h", "t", "u")]
    if any(isinstance(x, bool) or not isinstance(x, int) for x in digits):
        raise ValueError("replay prior numeric choice is invalid")
    return digits[0] * 100 + digits[1] * 10 + digits[2]


def _prefix_before_digit(item: dict, decisions: dict, prefix: str, suffix: str) -> str:
    parts = {"h": (), "t": ("h",), "u": ("h", "t")}.get(suffix)
    if parts is None:
        return ""
    return "".join(str(_choice_for(item, decisions, f"{prefix}:{part}")) for part in parts)


def replay_paper(run_dir: str | Path) -> PaperReplayResult:
    root = Path(run_dir)
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid paper manifest: {exc}") from exc
    if manifest.get("profile") != PROFILE or manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported paper replay profile or schema")
    if manifest.get("status") != "COMPLETE":
        raise ValueError("paper replay requires a complete run")
    try:
        config = PaperConfig(**manifest["config"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid paper replay configuration: {exc}") from exc
    events, audits = _lines(root / "economic.jsonl"), _lines(root / "audit.jsonl")
    if len(events) != config.total_rounds or manifest.get("total_rounds") != config.total_rounds:
        raise ValueError("paper replay round count does not match configuration")
    decisions = _verified_decisions(audits, config, config.total_rounds, events)
    _validate_audit_completeness(decisions, config, config.total_rounds)
    agents = tuple(f"agent_{i:02d}" for i in range(config.n_agents))
    schedule = (generate_g3_schedule(agents, config.total_rounds, config.master_seed)
                if config.treatment == "G3" else ())
    if manifest.get("schedule") != [[list(group) for group in row] for row in schedule]:
        raise ValueError("paper replay schedule differs from deterministic schedule")
    previous = "0" * 64
    total_trades = total_volume = total_profit = 0
    for r, event in enumerate(events):
        if event.get("round_index") != r or event.get("warmup") != (r < config.warmup_rounds):
            raise ValueError("paper replay round chronology or warmup tag mismatch")
        expected_hash = digest_json({k: v for k, v in event.items() if k != "event_hash"})
        if event.get("previous_hash") != previous or event.get("event_hash") != expected_hash:
            raise ValueError("paper replay hash chain mismatch")
        previous = expected_hash
        p, d = {}, {}
        for agent in agents:
            p[agent] = exogenous_value(config.master_seed, config.seed_index, r, agent, "p")
            d[agent] = exogenous_value(config.master_seed, config.seed_index, r, agent, "D")
        if event.get("exogenous") != {a: {"p": p[a], "D": d[a]} for a in agents}:
            raise ValueError("paper replay exogenous inputs mismatch")
        q = {}
        for agent in agents:
            digits = [decisions.get((r, agent, f"paper:purchase:{suffix}")) for suffix in ("h", "t", "u")]
            if any(isinstance(x, bool) or not isinstance(x, int) or not 0 <= x <= 9 for x in digits):
                raise ValueError("paper replay lacks procurement digit choices")
            q[agent] = digits[0] * 100 + digits[1] * 10 + digits[2]
            if q[agent] > 400 or q[agent] != event.get("procurement", {}).get(agent):
                raise ValueError("paper replay procurement differs from audited choices")
        orders = [PaperOrder(**row) for row in event.get("orders", [])]
        for order in orders:
            if order.owner not in agents:
                raise ValueError("paper replay contains an order with an unknown owner")
        canonical_orders = _orders_from_decisions(r, agents, decisions) if config.treatment != "NV" else []
        if [vars(o) for o in orders] != [vars(o) for o in canonical_orders]:
            raise ValueError("paper replay orders differ from canonical audited order slots")
        if config.treatment == "NV" and orders:
            raise ValueError("NV replay cannot contain orders")
        for agent in agents:
            portfolio = tuple(o for o in orders if o.owner == agent)
            validate_portfolio(portfolio, agent, q[agent], config.maximum_orders)
            side_values = []
            for slot in range(config.maximum_orders):
                side = decisions.get((r, agent, f"paper:slot{slot}:side"))
                if side is None or side == "STOP":
                    break
                if side not in {"buy", "sell"}:
                    raise ValueError("paper replay has invalid side choice")
                side_values.append(side)
                for kind, upper in (("qty", q[agent] if side == "sell" else 999), ("price", 999)):
                    prefix = [decisions.get((r, agent, f"paper:slot{slot}:{kind}:{s}")) for s in ("h", "t", "u")]
                    if any(isinstance(x, bool) or not isinstance(x, int) or not 0 <= x <= 9 for x in prefix):
                        raise ValueError("paper replay lacks order numeric-prefix choices")
                    number = prefix[0] * 100 + prefix[1] * 10 + prefix[2]
                    if not 1 <= number <= upper:
                        raise ValueError("paper replay order digit choice is outside legal bounds")
            expected = [(o.side, o.qty, o.limit_dollars) for o in sorted(portfolio, key=lambda x: x.order_id)]
            actual = [(side, decisions[(r, agent, f"paper:slot{slot}:qty:h")] * 100 +
                       decisions[(r, agent, f"paper:slot{slot}:qty:t")] * 10 + decisions[(r, agent, f"paper:slot{slot}:qty:u")],
                       decisions[(r, agent, f"paper:slot{slot}:price:h")] * 100 +
                       decisions[(r, agent, f"paper:slot{slot}:price:t")] * 10 + decisions[(r, agent, f"paper:slot{slot}:price:u")])
                      for slot, side in enumerate(side_values)]
            if expected != actual:
                raise ValueError("paper replay orders differ from audited portfolio decisions")
        groups = (() if config.treatment == "NV" else ((agents,) if config.treatment == "GA" else schedule[r]))
        if event.get("groups") != [list(g) for g in groups]:
            raise ValueError("paper replay market groups mismatch")
        market_rows = event.get("markets", [])
        if len(market_rows) != len(groups):
            raise ValueError("paper replay market count mismatch")
        trades, allocations, ranks = [], {a: 0 for a in agents}, {}
        for index, members in enumerate(groups):
            member_set = set(members)
            market_orders = [o for o in orders if o.owner in member_set]
            result = clear_paper_auction(market_orders, config.master_seed, config.seed_index, r,
                                         config.equality_crossing, config.self_match_policy)
            expected_market = {"market_index": index, "members": list(members),
                "price_cents": result.price_cents, "trades": [vars(t) for t in result.trades],
                "tie_ranks": result.tie_ranks, "self_volume": result.self_volume}
            if market_rows[index] != expected_market:
                raise ValueError("paper replay auction result mismatch")
            trades.extend(result.trades); allocations.update(result.net_allocations); ranks.update(result.tie_ranks)
        if event.get("net_allocations") != allocations:
            raise ValueError("paper replay net allocations mismatch")
        combined = PaperAuctionResult(None, tuple(trades), allocations, ranks,
                                      sum(row["self_volume"] for row in market_rows))
        settled = settle_paper_round({a: (p[a], d[a], q[a]) for a in agents}, combined, config.wholesale_price)
        settlement = {"profits_cents": settled.profits_cents, "fulfilled": settled.fulfilled,
            "leftovers": settled.leftovers, "shortages": settled.shortages,
            "entries": [vars(e) for e in settled.entries], "issued_units": settled.issued_units,
            "redeemed_units": settled.redeemed_units, "burned_units": settled.burned_units,
            "cash_grant_cents": settled.cash_grant_cents}
        if event.get("settlement") != settlement:
            raise ValueError("paper replay settlement mismatch")
        total_trades += len(trades)
        total_volume += sum(t.qty for t in trades)
        if not event["warmup"]:
            total_profit += sum(settled.profits_cents.values())
    if manifest.get("decision_audit_count") != len(audits):
        raise ValueError("paper replay audit count mismatch")
    return PaperReplayResult(len(events), config.analyzed_rounds, total_trades, total_volume,
                             total_profit, previous, tuple(events))

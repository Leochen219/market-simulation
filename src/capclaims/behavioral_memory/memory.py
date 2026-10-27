"""Pure, replayable factual-memory compiler. No target labels or hidden outcomes."""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from decimal import Decimal, ROUND_HALF_UP
from fractions import Fraction
import statistics
from typing import Any

from . import VERSION

FORMULA_VERSION = "episode-relations-1"
CONDITIONAL_FORMULA_VERSION = "conditional-groups-1"


def _money(value: Decimal | int | float) -> str:
    return str(Decimal(value).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def _profit_dollars(cents: int) -> str:
    sign = "-" if cents < 0 else ""
    return f"{sign}{abs(cents) // 100}.{abs(cents) % 100:02d}"


def _number(value: Any, field: str, low: int, high: int) -> int:
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{field} must be an integer from {low} through {high}")
    return value


def _orders(raw: Any, initial_q: int) -> list[dict[str, Any]]:
    if not isinstance(raw, (list, tuple)) or len(raw) > 5:
        raise ValueError("orders must be a list with no more than five submissions")
    clean, sells = [], 0
    for order in raw:
        if not isinstance(order, Mapping) or order.get("side") not in {"buy", "sell"}:
            raise ValueError("each order must have side buy or sell")
        qty = _number(order.get("qty"), "order qty", 1, 999)
        price = _number(order.get("price"), "order price", 1, 999)
        sells += qty if order["side"] == "sell" else 0
        clean.append({"side": order["side"], "qty": qty, "price": price})
    if sells > initial_q:
        raise ValueError("gross SELL submissions exceed initial Q")
    return clean


def _episode(row: Mapping[str, Any], cutoff: int, current_round: int) -> dict[str, Any]:
    if not isinstance(row, Mapping):
        raise ValueError("history episode must be a mapping")
    forbidden = {"w", "historypretradeprofit", "pretrade_profit", "fills", "executed_orders",
                 "mcp_cents", "market_clearing_price", "target", "label", "labels",
                 "target_orders", "target_label", "profit", "final_profit_cents",
                 "person_id", "participant_id"}
    if forbidden.intersection(row):
        raise ValueError("history contains a forbidden field")
    round_no = _number(row.get("round"), "history round", 1, 20)
    if round_no >= current_round or round_no > cutoff:
        raise ValueError("future history is not allowed")
    p = _number(row.get("p"), "history p", 1, 999)
    if "Q" in row and "q" in row and row["Q"] != row["q"]:
        raise ValueError("history Q and q aliases conflict")
    if "D" in row and "d" in row and row["D"] != row["d"]:
        raise ValueError("history D and d aliases conflict")
    q = _number(row.get("Q", row.get("q")), "history Q", 0, 400)
    d = _number(row.get("D", row.get("d")), "history D", 0, 200)
    profit = row.get("profit_cents")
    if type(profit) is not int:
        raise ValueError("verified integer final profit_cents is required")
    orders = _orders(row.get("orders"), q)
    counts = Counter((o["side"], o["qty"], o["price"]) for o in orders)
    duplicates = sum(n - 1 for n in counts.values())
    side_totals = {s: sum(o["qty"] for o in orders if o["side"] == s) for s in ("buy", "sell")}
    side_counts = {s: sum(o["side"] == s for o in orders) for s in ("buy", "sell")}
    side_prices: dict[str, str | None] = {}
    side_means: dict[str, Decimal | None] = {}
    side_quote_ranges: dict[str, dict[str, int] | None] = {}
    for side in ("buy", "sell"):
        denom = side_totals[side]
        side_orders = [o for o in orders if o["side"] == side]
        weighted = (Decimal(sum(o["qty"] * o["price"] for o in side_orders)) / denom
                    if denom else None)
        side_means[side] = weighted
        side_prices[side] = _money(weighted) if weighted is not None else None
        side_quote_ranges[side] = ({"min": min(o["price"] for o in side_orders),
            "max": max(o["price"] for o in side_orders)} if side_orders else None)
    diff, gap, surplus = d - q, max(d - q, 0), max(q - d, 0)
    side_set = {o["side"] for o in orders}
    relation = "shortage" if diff > 0 else "equal" if diff == 0 else "surplus"
    raw = {"round": round_no, "p": p, "Q": q, "D": d, "orders": deepcopy(orders),
           "profit_dollars": _profit_dollars(profit)}
    derived = {
        "formula_version": FORMULA_VERSION, "buy_quantity": side_totals["buy"],
        "sell_quantity": side_totals["sell"], "buy_order_count": side_counts["buy"],
        "sell_order_count": side_counts["sell"], "duplicate_submissions": duplicates,
        "order_count": len(orders), "buy_weighted_price": side_prices["buy"],
        "sell_weighted_price": side_prices["sell"],
        "buy_price_range": side_quote_ranges["buy"], "sell_price_range": side_quote_ranges["sell"],
        "buy_weighted_price_minus_p": _money(side_means["buy"] - p) if side_means["buy"] is not None else None,
        "sell_weighted_price_minus_p": _money(side_means["sell"] - p) if side_means["sell"] is not None else None,
        "D_minus_Q": diff, "gap": gap, "surplus": surplus,
        "buy_minus_gap": side_totals["buy"] - gap,
        "sell_minus_surplus": side_totals["sell"] - surplus,
        "demand_relation": relation,
        "buy_price_missing_reason": None if side_totals["buy"] else "no_buy_submissions",
        "sell_price_missing_reason": None if side_totals["sell"] else "no_sell_submissions",
        "order_mix": "empty" if not orders else "mixed" if len(side_set) == 2 else next(iter(side_set)),
    }
    return {"source_ref": f"round:{round_no}", "raw": raw, "derived": derived,
            "raw_orders": deepcopy(orders)}


def _conditional(episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = {name: [] for name in ("shortage", "equal", "surplus")}
    for ep in episodes:
        groups[ep["derived"]["demand_relation"]].append(ep)
    summaries = []
    for name, rows in groups.items():
        refs = [e["source_ref"] for e in rows]
        def nums(key: str) -> list[int]:
            return [e["derived"][key] for e in rows]
        def stat(values: list[int]) -> dict[str, Any]:
            return {"n": len(values), "median": statistics.median(values) if values else None,
                    "min": min(values) if values else None, "max": max(values) if values else None,
                    "source_refs": refs}
        mixes = Counter(e["derived"]["order_mix"] for e in rows)
        sides = {}
        for side in ("buy", "sell"):
            vals = [Decimal(sum(o["qty"] * o["price"] for o in e["raw"]["orders"] if o["side"] == side)) /
                    sum(o["qty"] for o in e["raw"]["orders"] if o["side"] == side)
                    for e in rows
                    if e["derived"][f"{side}_weighted_price"] is not None]
            quote_prices = [o["price"] for e in rows for o in e["raw"]["orders"] if o["side"] == side]
            sides[side] = {"valid_n": len(vals),
                "median": _money(Decimal(str(statistics.median(vals)))) if vals else None,
                "min": _money(min(vals)) if vals else None,
                "max": _money(max(vals)) if vals else None,
                "quote_price_min": min(quote_prices) if quote_prices else None,
                "quote_price_max": max(quote_prices) if quote_prices else None,
                "source_refs": [e["source_ref"] for e in rows
                                if e["derived"][f"{side}_weighted_price"] is not None]}
        summaries.append({"demand_relation": name, "formula_version": CONDITIONAL_FORMULA_VERSION,
            "n": len(rows), "any_buy_count": sum(e["derived"]["buy_order_count"] > 0 for e in rows),
            "any_sell_count": sum(e["derived"]["sell_order_count"] > 0 for e in rows),
            "empty_count": mixes["empty"], "mixed_count": mixes["mixed"],
            "count_source_refs": {"any_buy": [e["source_ref"] for e in rows if e["derived"]["buy_order_count"] > 0],
                "any_sell": [e["source_ref"] for e in rows if e["derived"]["sell_order_count"] > 0],
                "empty": [e["source_ref"] for e in rows if e["derived"]["order_mix"] == "empty"],
                "mixed": [e["source_ref"] for e in rows if e["derived"]["order_mix"] == "mixed"]},
            "quantity": {"buy": stat(nums("buy_quantity")), "sell": stat(nums("sell_quantity"))},
            "order_count": stat(nums("order_count")), "prices": sides, "source_refs": refs})
    return summaries


def compile_behavioral_memory(history: Sequence[Mapping[str, Any]], current_state: Mapping[str, Any],
                              cutoff: int) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return model-visible factual memory and a local audit, both derived from same records."""
    if not isinstance(history, (tuple, list)) or not isinstance(current_state, Mapping):
        raise ValueError("history and current_state have invalid types")
    cutoff = _number(cutoff, "cutoff", 1, 20)
    cur_round = _number(current_state.get("round"), "current round", 2, 21)
    if cutoff >= cur_round:
        raise ValueError("cutoff must be earlier than the current formal round")
    if any(key in current_state for key in ("profit_cents", "profit_dollars", "fills", "executed_orders",
            "mcp_cents", "market_clearing_price", "label", "labels", "target", "target_orders",
            "target_label", "orders", "profit", "final_profit_cents", "person_id", "participant_id")):
        raise ValueError("current state contains a forbidden target or outcome field")
    if "Q" in current_state and "q" in current_state and current_state["Q"] != current_state["q"]:
        raise ValueError("current Q and q aliases conflict")
    if "D" in current_state and "d" in current_state and current_state["D"] != current_state["d"]:
        raise ValueError("current D and d aliases conflict")
    p = _number(current_state.get("p"), "current p", 1, 999)
    q = _number(current_state.get("Q", current_state.get("q")), "current Q", 0, 400)
    d = _number(current_state.get("D", current_state.get("d")), "current D", 0, 200)
    seen: set[int] = set()
    episodes = []
    for row in history:
        episode = _episode(row, cutoff, cur_round)
        r = episode["raw"]["round"]
        if r in seen:
            raise ValueError("duplicate history round")
        seen.add(r)
        episodes.append(episode)
    episodes.sort(key=lambda e: e["raw"]["round"])
    for ep in episodes:
        ep["derived"]["relevance"] = {
            "numerator_components": {"abs_p_minus_ph": abs(p - ep["raw"]["p"]),
                "abs_Q_minus_Qh": abs(q - ep["raw"]["Q"]),
                "abs_D_minus_Dh": abs(d - ep["raw"]["D"])},
            "fraction": {"numerator": abs(p - ep["raw"]["p"]) * 400 +
                        abs(q - ep["raw"]["Q"]) * 49 + abs(d - ep["raw"]["D"]) * 98,
                        "denominator": 19600},
            "distance": (abs(p - ep["raw"]["p"]) / 49 + abs(q - ep["raw"]["Q"]) / 400 +
                         abs(d - ep["raw"]["D"]) / 200),
        }
    closest = sorted(episodes, key=lambda e: (
        e["derived"]["relevance"]["fraction"]["numerator"], -e["raw"]["round"]))[:2]
    ranges = {axis: ([e["raw"][field] for e in episodes] or [])
              for axis, field in (("p", "p"), ("Q", "Q"), ("D", "D"))}
    outside = {axis: (bool(vals) and not min(vals) <= value <= max(vals))
               for axis, vals, value in (("p", ranges["p"], p), ("Q", ranges["Q"], q), ("D", ranges["D"], d))}
    memory = {"version": VERSION, "cutoff_round": cutoff,
        "history": [deepcopy(e["raw"]) for e in episodes],
        "wire_history": [{"round": e["raw"]["round"] + 5, "formal_round": e["raw"]["round"],
            "p": e["raw"]["p"], "Q": e["raw"]["Q"], "D": e["raw"]["D"],
            "orders": [{"slot": i, "side": order["side"].upper(),
                        "quantity_units": order["qty"],
                        "limit_price_dollars_per_unit": order["price"]}
                       for i, order in enumerate(e["raw"]["orders"])],
            "profit_dollars": e["raw"]["profit_dollars"]} for e in episodes],
        "episodes": episodes, "conditional_summaries": _conditional(episodes),
        "relevance": {"formula": "abs(p-ph)/49+abs(Q-Qh)/400+abs(D-Dh)/200",
            "closest_source_refs": [e["source_ref"] for e in closest],
            "per_axis_outside_observed_range": outside,
            "per_axis_observed_range": {axis: ([min(vals), max(vals)] if vals else None)
                                         for axis, vals in ranges.items()},
            "per_axis_source_refs": {axis: [e["source_ref"] for e in episodes]
                                     for axis in ranges}}
    }
    exact_arithmetic = {}
    field_names = {
        "buy_quantity": ("sum(qty where side=BUY)", "units"),
        "sell_quantity": ("sum(qty where side=SELL)", "units"),
        "buy_order_count": ("count(side=BUY submissions)", "orders"),
        "sell_order_count": ("count(side=SELL submissions)", "orders"),
        "duplicate_submissions": ("sum(max(multiplicity(exact side,qty,price)-1,0))", "orders"),
        "order_count": ("count(all submitted orders)", "orders"),
        "buy_price_range": ("[min(price),max(price)] for BUY submissions", "dollars/unit"),
        "sell_price_range": ("[min(price),max(price)] for SELL submissions", "dollars/unit"),
        "buy_weighted_price": ("sum(qty*price)/sum(qty) for BUY", "dollars/unit"),
        "sell_weighted_price": ("sum(qty*price)/sum(qty) for SELL", "dollars/unit"),
        "buy_weighted_price_minus_p": ("BUY weighted price - same episode p", "dollars/unit"),
        "sell_weighted_price_minus_p": ("SELL weighted price - same episode p", "dollars/unit"),
        "D_minus_Q": ("D-Q", "units"), "gap": ("max(D-Q,0)", "units"),
        "surplus": ("max(Q-D,0)", "units"), "buy_minus_gap": ("BUY quantity-gap", "units"),
        "sell_minus_surplus": ("SELL quantity-surplus", "units"),
    }
    for ep in episodes:
        exact_arithmetic[ep["source_ref"]] = {}
        for side in ("buy", "sell"):
            subset = [o for o in ep["raw"]["orders"] if o["side"] == side]
            numerator = sum(o["qty"] * o["price"] for o in subset)
            denominator = sum(o["qty"] for o in subset)
            exact_arithmetic[ep["source_ref"]][side] = {
                "weighted_price": {"numerator": numerator, "denominator": denominator,
                    "missing_reason": "no_submissions" if denominator == 0 else None},
                "weighted_price_minus_p": {"numerator": numerator - ep["raw"]["p"] * denominator,
                    "denominator": denominator, "missing_reason": "no_submissions" if denominator == 0 else None}}
    conditional_price_exact = _conditional_price_exact(episodes)
    audit = {"version": VERSION, "formula_versions": [FORMULA_VERSION, CONDITIONAL_FORMULA_VERSION],
        "episode_count": len(episodes), "source_refs": [e["source_ref"] for e in episodes],
        "episode_derivations": {e["source_ref"]: deepcopy(e["derived"]) for e in episodes},
        "exact_arithmetic": exact_arithmetic,
        "field_provenance": {field: {"formula": formula, "units": unit,
            "source_refs": [e["source_ref"] for e in episodes],
            "missing_reasons": {e["source_ref"]: e["derived"][f"{field.split('_')[0]}_price_missing_reason"]
                for e in episodes if field.startswith(("buy_", "sell_")) and "price" in field}}
            for field, (formula, unit) in field_names.items()},
        "conditional_price_exact": conditional_price_exact,
        "verified_profit_cents": {e["source_ref"]: next(row["profit_cents"] for row in history
            if row["round"] == e["raw"]["round"]) for e in episodes},
        "closest_refs": memory["relevance"]["closest_source_refs"],
        "conditional_summaries": deepcopy(memory["conditional_summaries"]),
        "markdown_zh": _audit_markdown(memory, field_names, exact_arithmetic, conditional_price_exact)}
    return memory, audit


def _conditional_price_exact(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    groups = {name: [] for name in ("shortage", "equal", "surplus")}
    for episode in episodes:
        groups[episode["derived"]["demand_relation"]].append(episode)
    result = {}
    for name, rows in groups.items():
        result[name] = {}
        for side in ("buy", "sell"):
            values = []
            for episode in rows:
                orders = [o for o in episode["raw"]["orders"] if o["side"] == side]
                quantity = sum(o["qty"] for o in orders)
                if quantity:
                    numerator = sum(o["qty"] * o["price"] for o in orders)
                    values.append((Fraction(numerator, quantity), episode["source_ref"]))
            values.sort(key=lambda item: (item[0], item[1]))
            refs = [ref for _, ref in values]
            if values:
                n = len(values)
                if n % 2:
                    median = values[n // 2][0]
                    median_refs = [values[n // 2][1]]
                else:
                    median = (values[n // 2 - 1][0] + values[n // 2][0]) / 2
                    median_refs = [values[n // 2 - 1][1], values[n // 2][1]]
                fraction = lambda v: {"numerator": v.numerator, "denominator": v.denominator}
                result[name][side] = {"valid_n": n, "source_refs": refs,
                    "median": fraction(median), "median_source_refs": median_refs,
                    "min": fraction(values[0][0]), "max": fraction(values[-1][0]),
                    "missing_reason": None}
            else:
                result[name][side] = {"valid_n": 0, "source_refs": [], "median": None,
                    "median_source_refs": [], "min": None, "max": None,
                    "missing_reason": "no side submissions in conditional group"}
    return result


def _audit_markdown(memory: Mapping[str, Any], field_names: Mapping[str, tuple[str, str]],
                    exact_arithmetic: Mapping[str, Any], conditional_price_exact: Mapping[str, Any]) -> str:
    lines = ["# 行为记忆审计", f"版本：{memory['version']}",
             f"历史记录数：{len(memory['episodes'])}",
             "最近参考：" + ("、".join(memory["relevance"]["closest_source_refs"]) or "无")]
    lines.append("字段公式与单位：")
    lines.extend(f"- {field} = {formula}；单位：{unit}。"
                 for field, (formula, unit) in field_names.items())
    for ep in memory["episodes"]:
        d = ep["derived"]
        lines.append(f"- {ep['source_ref']}：BUY量 {d['buy_quantity']}，SELL量 {d['sell_quantity']}，"
                     f"重复提交 {d['duplicate_submissions']}，D-Q {d['D_minus_Q']}（{d['demand_relation']}）；"
                     f"精确相关距离 {d['relevance']['fraction']['numerator']}/"
                     f"{d['relevance']['fraction']['denominator']}。")
        for side in ("buy", "sell"):
            exact = exact_arithmetic[ep["source_ref"]][side]
            lines.append(f"  - {side.upper()}加权价精确分数 "
                f"{exact['weighted_price']['numerator']}/{exact['weighted_price']['denominator']}；"
                f"相对p差值 {exact['weighted_price_minus_p']['numerator']}/"
                f"{exact['weighted_price_minus_p']['denominator']}。")
    for group in memory["conditional_summaries"]:
        lines.append(f"- 条件组 {group['demand_relation']}：n={group['n']}，"
                     f"有BUY {group['any_buy_count']}，有SELL {group['any_sell_count']}，"
                     f"空组合 {group['empty_count']}，混合方向 {group['mixed_count']}。")
        for side in ("buy", "sell"):
            stats = conditional_price_exact[group["demand_relation"]][side]
            median = stats["median"]
            value = (f"{median['numerator']}/{median['denominator']}" if median else "无")
            lines.append(f"  - {side.upper()}加权价中位数精确分数 {value}，有效n={stats['valid_n']}，"
                         f"来源：{ '、'.join(stats['source_refs']) or '无'}。")
    return "\n".join(lines)

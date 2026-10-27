"""Descriptive outcome analysis; warmup rounds never enter formal summaries."""
from __future__ import annotations
from collections import defaultdict
import math
import statistics
from statistics import NormalDist
from typing import Any, Sequence

from capclaims.paper.baselines import exact_discrete_newsvendor_q


def _mean(values: Sequence[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _sd(values: Sequence[float]) -> float | None:
    return statistics.pstdev(values) if values else None


def continuous_newsvendor_deviation(q: float, p: float, w: int) -> float:
    """Appendix E.1/E.3: continuous-uniform NV reference deviation, in dollars."""
    if not 51 <= p <= 100 or w not in (10, 50) or q < 0:
        raise ValueError("invalid continuous newsvendor inputs")
    profit = p * (q - q * q / 400) - w * q if q <= 200 else 100 * p - w * q
    optimum = 100 * (p - w) ** 2 / p
    return optimum - profit


def coopetitor_normal_approx_feature(q: float, p: float, w: int, market_size: int) -> float:
    """Appendix E.8–E.12 normal-approximation coopetitor deviation (dollars)."""
    if not 51 <= p <= 100 or w not in (10, 50) or market_size < 1:
        raise ValueError("invalid Appendix E coopetitor inputs")
    mu = 100 * market_size
    sigma = 200 * math.sqrt(market_size / 12)
    normal = NormalDist()

    def pi(x: float) -> float:
        t = (x - mu) / sigma
        tail = 1 - normal.cdf(t)
        return p * mu - w * x - p * sigma * (normal.pdf(t) - t * tail)

    x_star = mu + sigma * normal.inv_cdf((p - w) / p)
    return (pi(x_star) - pi(market_size * q)) / market_size


def summarize_events(events: Sequence[dict[str, Any]], wholesale_price: int) -> dict[str, Any]:
    """Return per-round outcomes and formal-round aggregates in declared units."""
    formal = [event for event in events if not event["warmup"]]
    rounds = []
    agent_records: dict[str, list[dict]] = defaultdict(list)
    market_sizes: dict[str, list[int]] = defaultdict(list)
    for event in formal:
        q = event["procurement"]
        values = list(q.values())
        buy_submitted = sum(o["qty"] for o in event["orders"] if o["side"] == "buy")
        sell_submitted = sum(o["qty"] for o in event["orders"] if o["side"] == "sell")
        buy_filled = sum(t["qty"] for market in event["markets"] for t in market["trades"])
        sell_filled = sum(t["qty"] for market in event["markets"] for t in market["trades"])
        market_prices = [m["price_cents"] for m in event["markets"] if m["price_cents"] is not None]
        profits = event["settlement"]["profits_cents"]
        round_row = {
            "round": event["round_index"] + 1,
            "procurement_mean": _mean(values), "procurement_sd": _sd(values),
            "procurement_cov": (_sd(values) / _mean(values) if _mean(values) else None),
            "excess_units": sum(event["settlement"]["leftovers"].values()),
            "shortage_units": sum(event["settlement"]["shortages"].values()),
            "buy_submitted_units": buy_submitted, "sell_submitted_units": sell_submitted,
            "buy_filled_units": buy_filled, "sell_filled_units": sell_filled,
            "buy_fill_rate": buy_filled / buy_submitted if buy_submitted else None,
            "sell_fill_rate": sell_filled / sell_submitted if sell_submitted else None,
            "self_volume_units": sum(m["self_volume"] for m in event["markets"]),
            "mcp_mean_cents": _mean(market_prices),
            "mcp_sd_cents": _sd(market_prices), "nonempty_market_count": len(market_prices),
            "supplier_revenue_cents": wholesale_price * sum(values) * 100,
            "retailer_profit_cents": sum(profits.values()),
            # Supplier production cost is set to zero; supplier receipts are added back
            # to retailer profit after the internal wholesale transfer cancels.
            "supply_chain_profit_cents": sum(profits.values()) + wholesale_price * sum(values) * 100,
        }
        rounds.append(round_row)
        if "exogenous" in event:
            for agent, q_value in q.items():
                p = event["exogenous"][agent]["p"]
                n_group = next((len(m["members"]) for m in event["markets"] if agent in m["members"]), None)
                agent_records[agent].append({"q": q_value, "p": p,
                    "profit_cents": profits[agent], "net_quantity": event["net_allocations"][agent],
                    "market_size": n_group})
    aggregate_names = ("procurement_mean", "procurement_sd", "procurement_cov", "excess_units",
        "shortage_units", "buy_fill_rate", "sell_fill_rate", "self_volume_units", "mcp_mean_cents",
        "mcp_sd_cents", "supplier_revenue_cents", "retailer_profit_cents", "supply_chain_profit_cents")
    aggregate = {name: _mean([row[name] for row in rounds if row[name] is not None]) for name in aggregate_names}
    prices = [m["price_cents"] for event in formal for m in event["markets"] if m["price_cents"] is not None]
    aggregate["mcp_mean_cents"] = _mean(prices)
    aggregate["mcp_sd_cents"] = _sd(prices)
    agent_features = {}
    for agent, records in sorted(agent_records.items()):
        q_values = [row["q"] for row in records]
        p_values = [row["p"] for row in records]
        exact = _mean([exact_discrete_newsvendor_q(p, wholesale_price) for p in p_values])
        continuous = _mean([max(0.0, 200.0 * (1 - wholesale_price / p)) for p in p_values])
        continuous_deviation = _mean([continuous_newsvendor_deviation(row["q"], row["p"], wholesale_price) for row in records])
        coopetitor_values = [coopetitor_normal_approx_feature(row["q"], row["p"], wholesale_price,
                                                               row["market_size"])
                             for row in records if row["market_size"] is not None]
        coopetitor = _mean(coopetitor_values)
        agent_features[agent] = {
            "formal_round_count": len(records), "mean_procurement": _mean(q_values),
            "mean_profit_cents": _mean([row["profit_cents"] for row in records]),
            "mean_net_market_quantity": _mean([row["net_quantity"] for row in records]),
            "nv_exact_discrete_quantity": exact,
            "nv_continuous_reference_quantity": continuous,
            "nv_continuous_deviation_dollars": continuous_deviation,
            "coopetitor_normal_approx_feature": coopetitor,
            "coopetitor_approximation": ("Appendix E.8-E.12 continuous normal approximation; p is own p; N is local market size"
                if coopetitor_values else "Not applicable in no-market NV treatment"),
            "q_over_200_rate": sum(q > 200 for q in q_values) / len(q_values) if q_values else None,
        }
    return {"formal_rounds": len(formal), "rounds": rounds, "aggregate": aggregate, **aggregate,
            "mcp_mean_cents": _mean(prices), "mcp_sd_cents": _sd(prices),
            "agent_features": agent_features}

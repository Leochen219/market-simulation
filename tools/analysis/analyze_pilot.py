"""Deterministic, inference-free behavioral analysis of the bounded Laya pilot.

Run from the repository root with its configured Python interpreter. The script
replays both saved runs before reading their economic and decision records.
All money fields are reported in integer cents and dollar equivalents.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from capclaims.paper.replay import replay_paper

ROOT = Path(__file__).resolve().parents[2]
GA = ROOT / "artifacts/runs/paper-ga-laya-bounded-24"
NV = ROOT / "artifacts/runs/paper-nv-w10-final-24"
EXPECTED_HASH = "f09e5fc828d4a124a53a5abc7f077b4e104fe582e7abf1f9a38af53fa131ae41"
AGENTS = tuple(f"agent_{i:02d}" for i in range(10))


def read_lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def mean(xs: list[float]) -> float | None:
    return statistics.mean(xs) if xs else None


def pct(part: int | float, whole: int | float) -> float | None:
    return 100 * part / whole if whole else None


def discrete_nv(p: int, w: int) -> int:
    """Ex-ante Q optimum for D uniformly distributed on integer 0..200."""
    vals = [(p * sum(min(d, q) for d in range(201)) - w * q * 201, q)
            for q in range(401)]
    best = max(v for v, _ in vals)
    return min(q for v, q in vals if v == best)


def audit_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    stages: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        stages[row["stage"]].append(row)
    by_field: dict[str, list[dict]] = defaultdict(list)
    for stage, entries in stages.items():
        if stage.startswith("paper:purchase:"):
            field = "purchase"
        elif stage.endswith(":side"):
            field = "side"
        elif ":qty:" in stage:
            field = "quantity"
        elif ":price:" in stage:
            field = "price"
        else:
            continue
        by_field[field].extend(entries)
    result = {"audit_rows": len(rows), "fields": {}}
    for field, entries in by_field.items():
        margins, candidate_counts, ties = [], [], 0
        digit_freq: dict[str, Counter] = defaultdict(Counter)
        chosen_scores = []
        for row in entries:
            candidates = row["candidates"]
            scores = [float(row["raw_scores"][c["candidate_id"]]) for c in candidates]
            candidate_counts.append(len(candidates))
            top = max(scores)
            winners = [i for i, score in enumerate(scores) if score == top]
            ties += len(winners) > 1
            ordered = sorted(scores, reverse=True)
            margins.append(top - ordered[1] if len(ordered) > 1 else 0.0)
            chosen_scores.append(float(row["raw_scores"][next(c["candidate_id"] for c in candidates if c["value"] == row["selected"])]))
            digit_freq[row["observation"]["decimal_place"] if field != "side" else "side"][str(row["selected"]) ] += 1
        result["fields"][field] = {
            "decisions": len(entries), "candidate_count_mean": mean(candidate_counts),
            "candidate_count_min": min(candidate_counts), "candidate_count_max": max(candidate_counts),
            "score_margin_mean": mean(margins), "score_margin_median": statistics.median(margins),
            "exact_top_score_ties": int(ties), "exact_top_score_tie_share_pct": pct(ties, len(entries)),
            "selected_score_mean": mean(chosen_scores),
            "selected_values_by_digit_stage": {stage: dict(sorted(counts.items())) for stage, counts in digit_freq.items()},
        }
    return result


def percentile(values: list[int], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    low, high = math.floor(pos), math.ceil(pos)
    return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)


def run_analysis() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ga_replay = replay_paper(GA)
    assert ga_replay.final_hash == EXPECTED_HASH, ga_replay.final_hash
    nv_replay = replay_paper(NV)
    assert ga_replay.rounds == nv_replay.rounds == 20 and ga_replay.formal_rounds == nv_replay.formal_rounds == 15
    ga_events, nv_events = list(ga_replay.events), list(nv_replay.events)
    assert [(e["round_index"], e["exogenous"]) for e in ga_events] == [(e["round_index"], e["exogenous"]) for e in nv_events]
    audits = read_lines(GA / "audit.jsonl")
    formal = [e for e in ga_events if not e["warmup"]]
    warmup = [e for e in ga_events if e["warmup"]]
    per_agent: dict[str, dict[str, Any]] = {}
    rows_csv: list[dict[str, Any]] = []
    all_orders, all_trades = [], []
    round_metrics = []
    count_checks = {"market_cash_zero": True, "capacity_balances": True, "profit_identity": True,
                    "nonself_cash_balances": True}
    matched_profits = defaultdict(list)
    for e in ga_events:
        r = e["round_index"]
        nve = nv_events[r]
        warm = e["warmup"]
        order_by_id = {o["order_id"]: o for o in e["orders"]}
        alloc = e["net_allocations"]
        trades = [t for m in e["markets"] for t in m["trades"]]
        all_trades.extend((e, t) for t in trades)
        all_orders.extend((e, o) for o in e["orders"])
        buy_cash = sum(t["qty"] * t["price_cents"] for t in trades)
        sell_cash = buy_cash
        nonself_trades = [t for t in trades if t["buyer_id"] != t["seller_id"]]
        nonself_cash = sum(t["qty"] * t["price_cents"] for t in nonself_trades)
        total_q = sum(e["procurement"].values())
        total_d = sum(e["exogenous"][a]["D"] for a in AGENTS)
        mcp = e["markets"][0]["price_cents"] if e["markets"] else None
        for a in AGENTS:
            p, d, q = e["exogenous"][a]["p"], e["exogenous"][a]["D"], e["procurement"][a]
            net = alloc[a]
            settlement = e["settlement"]
            profit = settlement["profits_cents"][a]
            optimal = discrete_nv(p, 10)
            own_orders = [o for o in e["orders"] if o["owner"] == a]
            bought = sum(t["qty"] for t in trades if t["buyer_id"] == a and t["buyer_id"] != t["seller_id"])
            sold = sum(t["qty"] for t in trades if t["seller_id"] == a and t["buyer_id"] != t["seller_id"])
            self_buy = sum(t["qty"] for t in trades if t["buyer_id"] == a == t["seller_id"])
            own_gross_buy = sum(t["qty"] for t in trades if t["buyer_id"] == a)
            own_gross_sell = sum(t["qty"] for t in trades if t["seller_id"] == a)
            unsold = settlement["leftovers"][a]
            shortage = settlement["shortages"][a]
            actual_revenue = p * 100 * settlement["fulfilled"][a]
            if profit != actual_revenue - 1000 * q - net * (mcp or 0):
                count_checks["profit_identity"] = False
            if q + net != settlement["fulfilled"][a] + unsold:
                count_checks["capacity_balances"] = False
            nvp = nve["settlement"]["profits_cents"][a]
            matched_profits[a].append(profit - nvp)
            if a not in per_agent:
                per_agent[a] = {k: [] for k in ("p", "D", "Q", "nv_optimal_Q", "profit_cents", "nv_profit_cents", "trade_buy_qty", "trade_sell_qty", "net_allocation", "shortage_before", "leftover_before", "shortage_after", "leftover_after")}
            vals = {"p":p,"D":d,"Q":q,"nv_optimal_Q":optimal,"profit_cents":profit,"nv_profit_cents":nvp,
                    "trade_buy_qty":bought,"trade_sell_qty":sold,"net_allocation":net,
                    "shortage_before":max(0,d-q),"leftover_before":max(0,q-d),
                    "shortage_after":shortage,"leftover_after":unsold}
            if not warm:
                for k,v in vals.items(): per_agent[a][k].append(v)
            if not warm:
                rows_csv.append({"round":r+1,"agent":a,"p_dollars":p,"demand":d,"procurement":q,"discrete_nv_optimal_q":optimal,
                    "q_above_nv_optimal":q>optimal,"profit_cents":profit,"nv_profit_cents_same_exogenous":nvp,"profit_delta_vs_nv_cents":profit-nvp,
                    "net_allocation":net,"gross_buy_fills":own_gross_buy,"gross_sell_fills":own_gross_sell,"shortage_before":max(0,d-q),
                    "leftover_before":max(0,q-d),"shortage_after":shortage,"leftover_after":unsold})
        assert buy_cash == sell_cash
        total_profit = sum(e["settlement"]["profits_cents"].values())
        pre_short = sum(max(0,e["exogenous"][a]["D"]-e["procurement"][a]) for a in AGENTS)
        pre_left = sum(max(0,e["procurement"][a]-e["exogenous"][a]["D"]) for a in AGENTS)
        post_short = sum(e["settlement"]["shortages"].values())
        post_left = sum(e["settlement"]["leftovers"].values())
        central = sorted(AGENTS,key=lambda a:(-e["exogenous"][a]["p"],a))
        remaining = total_q; planner_revenue = 0
        for a in central:
            sold_plan = min(e["exogenous"][a]["D"],remaining)
            planner_revenue += sold_plan * e["exogenous"][a]["p"] * 100
            remaining -= sold_plan
        sales_revenue = sum(e["settlement"]["profits_cents"][a] + 1000*e["procurement"][a] + e["net_allocations"][a]*(mcp or 0) for a in AGENTS)
        round_metrics.append({"round":r+1,"period":"warmup" if warm else "formal","total_Q":total_q,"total_D":total_d,"MCP_cents":mcp,
            "orders":len(e["orders"]),"trade_fills":len(trades),"trade_volume_including_self":sum(t["qty"] for t in trades),
            "self_trade_fills":sum(t["buyer_id"]==t["seller_id"] for t in trades),"self_trade_volume":sum(t["qty"] for t in trades if t["buyer_id"]==t["seller_id"]),
            "gross_fill_notional_including_self_cents":buy_cash,
            "actual_nonself_buyer_cash_cents":nonself_cash,"actual_nonself_seller_cash_cents":nonself_cash,
            "net_capacity_reallocation_units":sum(abs(e["net_allocations"][a]) for a in AGENTS)/2,
            "shortage_before":pre_short,"leftover_before":pre_left,"shortage_after":post_short,"leftover_after":post_left,
            "sales_revenue_cents":sales_revenue,"planner_revenue_same_Q_cents":planner_revenue,
            "actual_revenue_efficiency_pct":pct(sales_revenue,planner_revenue),
            "profit_cents":total_profit,"NV_profit_cents":sum(nve["settlement"]["profits_cents"].values())})
    assert all(count_checks.values()), count_checks
    formal_rows = [x for x in round_metrics if x["period"] == "formal"]
    formal_orders = [(e,o) for e,o in all_orders if not e["warmup"]]
    formal_trades = [(e,t) for e,t in all_trades if not e["warmup"]]
    last_matched_self_rounds = [e["round_index"]+1 for e in formal
        if (lambda ts: bool(ts) and ts[-1]["buyer_id"] == ts[-1]["seller_id"])([t for m in e["markets"] for t in m["trades"]])]
    assert last_matched_self_rounds == [7, 9, 20], last_matched_self_rounds
    agents_formal = 150
    qvals = [e["procurement"][a] for e in formal for a in AGENTS]
    dvals = [e["exogenous"][a]["D"] for e in formal for a in AGENTS]
    nvopts = [discrete_nv(e["exogenous"][a]["p"],10) for e in formal for a in AGENTS]
    q_above = sum(e["procurement"][a] > discrete_nv(e["exogenous"][a]["p"],10) for e in formal for a in AGENTS)
    mcp_p_buyer, intr_loss_cents, expensive_exec_count, expensive_exec_qty = 0,0,0,0
    gross_buys = Counter(); gross_sells = Counter(); net_market_cash = Counter()
    for e,t in formal_trades:
        buyer, seller, qty, price = t["buyer_id"], t["seller_id"], t["qty"], t["price_cents"]
        gross_buys[buyer] += qty; gross_sells[seller] += qty
        net_market_cash[buyer] -= qty*price; net_market_cash[seller] += qty*price
        if buyer != seller and price > 100*e["exogenous"][buyer]["p"]:
            expensive_exec_count += 1; expensive_exec_qty += qty
            intr_loss_cents += qty*(price-100*e["exogenous"][buyer]["p"])
    formal_net_cash = sum(net_market_cash.values())
    count_checks["market_cash_zero"] = formal_net_cash == 0
    formal_actual_nonself_buyer_cash = sum(t["qty"]*t["price_cents"] for _,t in formal_trades if t["buyer_id"]!=t["seller_id"])
    formal_actual_nonself_seller_cash = sum(t["qty"]*t["price_cents"] for _,t in formal_trades if t["buyer_id"]!=t["seller_id"])
    count_checks["nonself_cash_balances"] = formal_actual_nonself_buyer_cash == formal_actual_nonself_seller_cash
    assert all(count_checks.values()), count_checks
    self_trades = [(e,t) for e,t in formal_trades if t["buyer_id"]==t["seller_id"]]
    markets_with_price = [e["markets"][0]["price_cents"] for e in formal if e["markets"][0]["price_cents"] is not None]
    price_counts = Counter(o["limit_dollars"] for _,o in formal_orders)
    qty_counts = Counter(o["qty"] for _,o in formal_orders)
    repeated = lambda n: len(set(str(n))) == 1 and len(str(n)) >= 2
    repeated_prices = [o for _,o in formal_orders if repeated(o["limit_dollars"])]
    repeated_qtys = [o for _,o in formal_orders if repeated(o["qty"])]
    stagewise = audit_metrics(audits)
    baseline_no_trade = []
    roundtrip_both_sides = 0
    agent_round_trade_summary: dict[tuple[int, str], dict[str, int]] = defaultdict(lambda: {"gross_buy": 0, "gross_sell": 0, "net": 0})
    for e,t in formal_trades:
        agent_round_trade_summary[(e["round_index"],t["buyer_id"])]["gross_buy"] += t["qty"]
        agent_round_trade_summary[(e["round_index"],t["seller_id"])]["gross_sell"] += t["qty"]
    for e in formal:
        for a in AGENTS:
            agent_round_trade_summary[(e["round_index"],a)]["net"] = e["net_allocations"][a]
    roundtrip_both_sides = sum(x["gross_buy"]>0 and x["gross_sell"]>0 for x in agent_round_trade_summary.values())
    buyer_net_positive_leftover_buy_volume = sum(x["gross_buy"] for (r,a),x in agent_round_trade_summary.items()
        if x["net"]>0 and formal[r-5]["settlement"]["leftovers"][a]>0)
    seller_net_negative_shortage_sell_volume = sum(x["gross_sell"] for (r,a),x in agent_round_trade_summary.items()
        if x["net"]<0 and formal[r-5]["settlement"]["shortages"][a]>0)
    regret_cents = []
    regret_rows = []
    for e in formal:
        for a in AGENTS:
            p,d,q=e["exogenous"][a]["p"],e["exogenous"][a]["D"],e["procurement"][a]
            no_trade_profit=100*p*min(d,q)-1000*q
            actual=e["settlement"]["profits_cents"][a]
            regret=max(0,no_trade_profit-actual)
            regret_cents.append(regret)
            regret_rows.append({"round":e["round_index"]+1,"agent":a,"no_trade_profit_cents":no_trade_profit,
                "actual_profit_cents":actual,"foregone_profit_cents":regret})
            baseline_no_trade.append({"p":p,"D":d,"Q":q,"profit_cents":no_trade_profit,
                "shortage":max(0,d-q),"leftover":max(0,q-d)})
    no_trade_revenue=sum(100*x["p"]*min(x["D"],x["Q"]) for x in baseline_no_trade)
    actual_revenue=sum(r["sales_revenue_cents"] for r in formal_rows)
    planner_revenue=sum(r["planner_revenue_same_Q_cents"] for r in formal_rows)
    def view_period(events: list[dict]) -> dict[str, Any]:
        n=len(events)*10
        ps=[e["exogenous"][a]["p"] for e in events for a in AGENTS]
        ds=[e["exogenous"][a]["D"] for e in events for a in AGENTS]
        qs=[e["procurement"][a] for e in events for a in AGENTS]
        profits=[e["settlement"]["profits_cents"][a] for e in events for a in AGENTS]
        shorts=[e["settlement"]["shortages"][a] for e in events for a in AGENTS]
        leftovers=[e["settlement"]["leftovers"][a] for e in events for a in AGENTS]
        preS=sum(max(0,e["exogenous"][a]["D"]-e["procurement"][a]) for e in events for a in AGENTS)
        preL=sum(max(0,e["procurement"][a]-e["exogenous"][a]["D"]) for e in events for a in AGENTS)
        postS=sum(shorts); postL=sum(leftovers)
        round_cv_pop=[]; round_cv_sample=[]
        for e in events:
            round_q=[e["procurement"][a] for a in AGENTS]
            round_mean=statistics.mean(round_q)
            round_cv_pop.append(100*statistics.pstdev(round_q)/round_mean if round_mean else 0.0)
            round_cv_sample.append(100*statistics.stdev(round_q)/round_mean if round_mean else 0.0)
        return {"rounds":len(events),"agent_rounds":n,"p_dollars":{"mean":mean(ps),"min":min(ps),"max":max(ps)},
            "demand":{"mean":mean(ds),"min":min(ds),"max":max(ds)},"Q":{"mean":mean(qs),"median":statistics.median(qs),"sd_population":statistics.pstdev(qs),"sd_sample":statistics.stdev(qs),
            "pooled_agent_round_cv_population_pct":100*statistics.pstdev(qs)/statistics.mean(qs),"pooled_agent_round_cv_sample_pct":100*statistics.stdev(qs)/statistics.mean(qs),
            "mean_per_round_cross_agent_cv_population_pct":mean(round_cv_pop),"mean_per_round_cross_agent_cv_sample_pct":mean(round_cv_sample),"min":min(qs),"max":max(qs),
            "p50":percentile(qs,.5),"p90":percentile(qs,.9)},"mean_discrete_nv_optimal_Q":mean([discrete_nv(p,10) for p in ps]),
            "Q_above_discrete_nv_optimal_count":sum(q>discrete_nv(p,10) for p,q in zip(ps,qs)),"Q_above_discrete_nv_optimal_pct":pct(sum(q>discrete_nv(p,10) for p,q in zip(ps,qs)),n),
            "profits_cents":{"mean":mean(profits),"total":sum(profits),"negative_count":sum(x<0 for x in profits),"zero_count":sum(x==0 for x in profits),"positive_count":sum(x>0 for x in profits),"p10":percentile(profits,.1),"median":statistics.median(profits),"p90":percentile(profits,.9)},
            "pre_market_shortage_units":preS,"pre_market_leftover_units":preL,"post_market_shortage_units":postS,"post_market_leftover_units":postL,
            "served_before_market_units":sum(min(e["exogenous"][a]["D"],e["procurement"][a]) for e in events for a in AGENTS),
            "served_after_market_units":sum(e["settlement"]["fulfilled"][a] for e in events for a in AGENTS),
            "realized_demand_units":sum(ds),
            "shortage_reduction_units":preS-postS,"shortage_reduction_pct":pct(preS-postS,preS),"leftover_reduction_units":preL-postL,"leftover_reduction_pct":pct(preL-postL,preL)}
    ga_formal_summary=view_period(formal); ga_warm_summary=view_period(warmup)
    total_profit_by_agent={a:sum(e["settlement"]["profits_cents"][a] for e in formal) for a in AGENTS}
    total_abs=sum(abs(v) for v in total_profit_by_agent.values())
    loss_agents=[a for a,v in total_profit_by_agent.items() if v<0]
    positive_profit_total=sum(max(0,v) for v in total_profit_by_agent.values())
    side_orders={}
    for side in ("buy","sell"):
        these=[o for _,o in formal_orders if o["side"]==side]
        pc=Counter(o["limit_dollars"] for o in these); qc=Counter(o["qty"] for o in these)
        side_orders[side]={"count":len(these),"mean_limit_dollars":mean([o["limit_dollars"] for o in these]),
            "median_limit_dollars":statistics.median([o["limit_dollars"] for o in these]),"mean_quantity":mean([o["qty"] for o in these]),
            "median_quantity":statistics.median([o["qty"] for o in these]),"limit_price_top10":pc.most_common(10),
            "quantity_top10":qc.most_common(10),"limit_price_111_count":pc[111],"limit_price_999_count":pc[999],
            "quantity_111_count":qc[111],"quantity_999_count":qc[999]}
    qsum=sum(qvals)
    summary={
        "provenance":{"pilot_run":"artifacts/runs/paper-ga-laya-bounded-24","nv_run":"artifacts/runs/paper-nv-w10-final-24",
            "replay":"PASS","pilot_final_hash":ga_replay.final_hash,"rounds":20,"warmup_rounds":5,"formal_rounds":15,
            "N":10,"policy":"Laya","treatment":"GA","wholesale_price_dollars":10,"seed_index":0,
            "declared_pilot_deviations":json.loads((GA/"manifest.json").read_text(encoding="utf-8"))["deviations"],
            "no_new_inference_or_experiments":True,"units":{"p_and_prices":"dollars/unit unless marked cents","money":"integer cents; dollar equivalent = cents/100","quantity":"units"}},
        "formal":ga_formal_summary,"warmup":ga_warm_summary,
        "round_trends":round_metrics,
        "per_agent_formal":{a:{k:{"mean":mean(v),"total":sum(v),"min":min(v),"max":max(v)} for k,v in data.items()} for a,data in per_agent.items()},
        "profit_concentration":{"scope":"Retailer profit only; supplier revenue is excluded.","per_agent_total_profit_cents":total_profit_by_agent,"negative_profit_agents":loss_agents,
            "negative_agent_rounds":sum(1 for e in formal for a in AGENTS if e["settlement"]["profits_cents"][a]<0),
            "worst_actual_profit_agent_rounds":[{"round":e["round_index"]+1,"agent":a,"profit_cents":e["settlement"]["profits_cents"][a],
                "p_dollars":e["exogenous"][a]["p"],"D":e["exogenous"][a]["D"],"Q":e["procurement"][a],"net_allocation":e["net_allocations"][a],
                "MCP_cents":e["markets"][0]["price_cents"]} for e,a in sorted(((e,a) for e in formal for a in AGENTS),key=lambda ea:ea[0]["settlement"]["profits_cents"][ea[1]])[:10]],
            "top_agent_share_of_positive_profit_pct":pct(max(total_profit_by_agent.values()),positive_profit_total),
            "top_three_share_of_positive_profit_pct":pct(sum(sorted(total_profit_by_agent.values(),reverse=True)[:3]),positive_profit_total),
            "profit_hhi_on_positive_agent_profit_shares":sum((max(v,0)/positive_profit_total)**2 for v in total_profit_by_agent.values())},
        "orders":{"formal_order_count":len(formal_orders),"formal_order_sides":dict(Counter(o["side"] for _,o in formal_orders)),
            "by_side":side_orders,
            "formal_order_prices_dollars":{"mean":mean([o["limit_dollars"] for _,o in formal_orders]),"median":statistics.median([o["limit_dollars"] for _,o in formal_orders]),
                "min":min(price_counts),"max":max(price_counts),"frequency_top20":price_counts.most_common(20),"at_111_count":price_counts[111],"at_999_count":price_counts[999]},
            "formal_order_quantities":{"mean":mean([o["qty"] for _,o in formal_orders]),"median":statistics.median([o["qty"] for _,o in formal_orders]),
                "min":min(qty_counts),"max":max(qty_counts),"frequency_top20":qty_counts.most_common(20),"at_111_count":qty_counts[111],"at_999_count":qty_counts[999]},
            "repeated_digit_price_orders":len(repeated_prices),"repeated_digit_price_share_pct":pct(len(repeated_prices),len(formal_orders)),
            "repeated_digit_qty_orders":len(repeated_qtys),"repeated_digit_qty_share_pct":pct(len(repeated_qtys),len(formal_orders)),
            "repeated_digits_by_price":{"111":price_counts[111],"222":price_counts[222],"333":price_counts[333],"444":price_counts[444],"555":price_counts[555],"666":price_counts[666],"777":price_counts[777],"888":price_counts[888],"999":price_counts[999]},
            "repeated_digits_by_quantity":{"111":qty_counts[111],"222":qty_counts[222],"333":qty_counts[333],"444":qty_counts[444],"555":qty_counts[555],"666":qty_counts[666],"777":qty_counts[777],"888":qty_counts[888],"999":qty_counts[999]},
            "procurement_Q_frequency_top20":Counter(qvals).most_common(20),
            "orders_per_agent_round_mean":len(formal_orders)/agents_formal},
        "execution":{"formal_trade_fills_including_self":len(formal_trades),"formal_volume_including_self":sum(t["qty"] for _,t in formal_trades),
            "self_trade_fills":len(self_trades),"self_trade_volume":sum(t["qty"] for _,t in self_trades),
            "self_trade_volume_share_pct":pct(sum(t["qty"] for _,t in self_trades),sum(t["qty"] for _,t in formal_trades)),
            "MCP_dollars":{"mean":mean(markets_with_price)/100 if markets_with_price else None,"median":statistics.median(markets_with_price)/100 if markets_with_price else None,
                "sd_population":statistics.pstdev(markets_with_price)/100 if markets_with_price else None,"sd_sample":statistics.stdev(markets_with_price)/100 if len(markets_with_price)>1 else None,
                "min":min(markets_with_price)/100 if markets_with_price else None,"max":max(markets_with_price)/100 if markets_with_price else None,"priced_rounds":len(markets_with_price)},
            "gross_nonself_buy_fill_price_above_own_p":{"fill_count":expensive_exec_count,"volume":expensive_exec_qty,
                "share_of_nonself_volume_pct":pct(expensive_exec_qty,sum(t["qty"] for _,t in formal_trades if t["buyer_id"]!=t["seller_id"])),
                "share_of_all_volume_including_self_pct":pct(expensive_exec_qty,sum(t["qty"] for _,t in formal_trades)),
                "gross_price_spread_exposure_cents":intr_loss_cents,"gross_price_spread_exposure_dollars":intr_loss_cents/100,
                "caveat":"Gross executed buy volume with MCP above the buyer's own p, summed fill-by-fill before offsetting same-round sales or inventory outcomes; this is exposure, not realized net loss or buyer-error rate."},
            "gross_fill_notional_including_self_cents":sum(t["qty"]*t["price_cents"] for _,t in formal_trades),
            "gross_fill_notional_caveat":"Includes self-match notional for descriptive fill totals; self-match cash is skipped in the actual ledger.",
            "actual_nonself_buyer_cash_cents":formal_actual_nonself_buyer_cash,
            "actual_nonself_seller_cash_cents":formal_actual_nonself_seller_cash,
            "net_capacity_reallocation_units_formal":sum(r["net_capacity_reallocation_units"] for r in formal_rows),
            "net_capacity_reallocation_units_all_rounds":sum(r["net_capacity_reallocation_units"] for r in round_metrics),
            "last_matched_self_rounds":last_matched_self_rounds,
            "last_matched_self_price_note":"These rounds end with a self-match as the last recorded matched pair, which can be price-setting under the clearing rule; descriptive flag only, not a causal estimate of self-match price impact.",
            "net_transfer_by_agent_cents":{a:net_market_cash[a] for a in AGENTS},"gross_buy_fill_units_by_agent":{a:gross_buys[a] for a in AGENTS},"gross_sell_fill_units_by_agent":{a:gross_sells[a] for a in AGENTS}},
        "no_trade_counterfactual_at_realized_Q":{"scope":"Retailer profit.","sum_profit_cents":sum(x["profit_cents"] for x in baseline_no_trade),
            "actual_market_minus_all_no_orders_profit_cents":sum(total_profit_by_agent.values())-sum(x["profit_cents"] for x in baseline_no_trade),
            "negative_agent_rounds":sum(x["profit_cents"]<0 for x in baseline_no_trade),"shortage_units":sum(x["shortage"] for x in baseline_no_trade),"leftover_units":sum(x["leftover"] for x in baseline_no_trade),
            "definition":"Joint all-no-orders conditional counterfactual: each agent retains its actual Q and sells only min(D,Q). This mechanically removes market trade at fixed procurement; it is not a market-treatment causal estimate because procurement behavior is held fixed."},
        "fixed_total_Q_allocation_efficiency":{"planner_revenue_cents":sum(r["planner_revenue_same_Q_cents"] for r in formal_rows),
            "actual_sales_revenue_cents":sum(r["sales_revenue_cents"] for r in formal_rows),
            "revenue_efficiency_pct":pct(sum(r["sales_revenue_cents"] for r in formal_rows),sum(r["planner_revenue_same_Q_cents"] for r in formal_rows)),
            "avoidable_revenue_gap_cents":sum(r["planner_revenue_same_Q_cents"]-r["sales_revenue_cents"] for r in formal_rows),
            "no_trade_sales_revenue_cents":no_trade_revenue,"realized_market_sales_revenue_delta_vs_no_trade_cents":actual_revenue-no_trade_revenue,
            "maximum_allocation_gain_vs_no_trade_cents":planner_revenue-no_trade_revenue,
            "scope":"Ex-post benchmark reallocates each round's fixed aggregate Q to highest-p agents up to realized D; excludes wholesale procurement cost."},
        "matched_exogenous_NV":{"comparison":"same p and D for every agent-round; GA trading pilot versus separately saved NV policy run; policy/decision interface differs, so not a treatment effect. Profit fields are retailer profit, not chain profit.",
            "ga_profit_cents":sum(e["settlement"]["profits_cents"][a] for e in formal for a in AGENTS),"nv_profit_cents":sum(e["settlement"]["profits_cents"][a] for e in nv_events if not e["warmup"] for a in AGENTS),
            "ga_minus_nv_profit_cents":sum(e["settlement"]["profits_cents"][a]-nv_events[e["round_index"]]["settlement"]["profits_cents"][a] for e in formal for a in AGENTS),
            "ga_minus_nv_profit_dollars":sum(e["settlement"]["profits_cents"][a]-nv_events[e["round_index"]]["settlement"]["profits_cents"][a] for e in formal for a in AGENTS)/100,
            "ga_minus_nv_profit_by_agent_cents":{a:sum(matched_profits[a]) for a in AGENTS}},
        "accounting_scope":{"retailer_profit_cents":sum(total_profit_by_agent.values()),
            "supplier_wholesale_revenue_cents":sum(e["procurement"][a]*1000 for e in formal for a in AGENTS),
            "combined_chain_profit_cents":sum(r["sales_revenue_cents"] for r in formal_rows),
            "identity":"Retailer profit plus supplier wholesale receipts equals consumer sales revenue because the model has zero production cost and market cash transfers net to zero; do not label retailer profit alone as chain profit."},
        "audit_model_score_diagnostics":{"all_rounds":stagewise,"formal_rounds":audit_metrics([x for x in audits if x["round_index"]>=5]),
            "warmup_rounds":audit_metrics([x for x in audits if x["round_index"]<5])},
        "unilateral_no_order_regret":{"definition":"Per agent-round max(0, same-Q realized-D no-order profit minus realized market profit). Submitting no orders guarantees that agent net allocation A is zero, so this is an exact ex-post unilateral own-payoff alternative with Q,D held fixed. Other orders re-clear. Summing unilateral regrets is not the joint all-no-orders counterfactual or a causal market effect.",
            "regret_total_cents":sum(regret_cents),"regret_total_dollars":sum(regret_cents)/100,
            "agent_rounds_with_positive_regret":sum(x>0 for x in regret_cents),"agent_rounds_with_loss_over_1000_dollars":sum(x>100000 for x in regret_cents),
            "median_regret_cents":statistics.median(regret_cents),"p90_regret_cents":percentile(regret_cents,.9),
            "largest_examples":sorted(regret_rows,key=lambda x:x["foregone_profit_cents"],reverse=True)[:10]},
        "realized_net_position_cohorts":{"agent_rounds_both_bought_and_sold_gross":roundtrip_both_sides,
            "gross_buy_volume_of_buyer_agent_rounds_ending_net_long_with_leftovers":buyer_net_positive_leftover_buy_volume,
            "gross_sell_volume_of_seller_agent_rounds_ending_net_short_with_shortages":seller_net_negative_shortage_sell_volume,
            "interpretation":"These are agent-round cohorts based on final net allocation and ending balance; the statistic does not attribute a particular fill to a leftover unit or unmet sale."},
        "verification":{"assertions":count_checks,"GA_hash_asserted":True,"matched_exogenous_inputs_asserted":True,"market_cash_net_zero_cents":formal_net_cash}
    }
    return summary, rows_csv


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,default=ROOT/"docs/analysis/pilot-behavior-data.json")
    parser.add_argument("--csv",type=Path,default=None,help="optional formal agent-round CSV path")
    args=parser.parse_args()
    data, rows=run_analysis()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(data,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    if args.csv:
        args.csv.parent.mkdir(parents=True,exist_ok=True)
        with args.csv.open("w",newline="",encoding="utf-8") as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]))
            writer.writeheader();writer.writerows(rows)
    print(json.dumps({"output":str(args.output),"formal":data["formal"],"execution":data["execution"],
        "matched_exogenous_NV":data["matched_exogenous_NV"],"hash":data["provenance"]["pilot_final_hash"]},indent=2))


if __name__=="__main__":
    main()

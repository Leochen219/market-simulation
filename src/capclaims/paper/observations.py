"""Private, compact model-visible paper observations."""


def add_order_choice_context(observation: dict, *, decision_field: str,
                             own_side: str, own_qty: int | None = None) -> dict:
    """Apply numeric order context using the interface's canonical field names."""
    result = dict(observation, decision_field=decision_field, own_side=own_side)
    if own_qty is not None:
        key = "own_qty_units" if result.get("decision_interface") == "economic_v2" else "own_qty"
        result[key] = own_qty
    return result


def _compact_order(order: dict) -> dict:
    order_id = order.get("order_id", "")
    try:
        slot = int(order_id.rsplit("slot", 1)[1])
    except (IndexError, ValueError):
        slot = int(order.get("slot", 0))
    return {"slot": slot, "side": order["side"], "qty": order["qty"],
            "limit_dollars": order["limit_dollars"]}

def build_paper_observation(*, agent_id: str, round_index: int, treatment: str, n_agents: int,
                            wholesale_price: int, p: int, history: tuple | list,
                            partial_orders: tuple | list, phase: str, q: int | None = None,
                            d: int | None = None, slot: int | None = None,
                            prefix: str = "", minimum: int = 0, maximum: int = 999,
                            horizon: int = 20, history_window: int = 2, decision_field: str | None = None,
                            own_side: str | None = None, own_qty: int | None = None,
                            equality_crossing: str = "ge", self_match_policy: str = "allow",
                            decision_interface: str = "legacy_v1") -> dict:
    from capclaims.paper.interface import validate_interface
    validate_interface(decision_interface)
    if phase not in {"procurement", "orders"}:
        raise ValueError("phase must be procurement or orders")
    if len(history) > history_window:
        raise ValueError("paper observation exceeds declared own history window")
    observation = {
        "horizon": horizon, "round": round_index + 1, "n_agents": n_agents,
        "market_size": 3 if treatment == "G3" else n_agents,
        "wholesale_price_dollars": wholesale_price,
        "equality_crossing": equality_crossing,
        "self_match_policy": self_match_policy,
        "maximum_total_orders": 5,
        "public_instructions": {
            "demand": "After procurement, own demand is revealed; each agent demand is independent uniform integer 0..200.",
            "sale_value": "Own sale value is shown in dollars per unit; unused capacity expires after this round.",
            "procurement": "Choose an integer quantity 0..400 before demand is revealed; procurement costs wholesale price times quantity.",
            "trading": "Submit up to five total blind buy/sell limit orders; quantity and integer-dollar limit are each 1..999.",
            "sell_cap": "Gross submitted sell quantities cannot exceed initial procurement; buys never increase this cap.",
            "clearing": ("Orders clear by price priority with partial fills and seeded order-slot tie priority. "
                         "All fills in each market use the midpoint of the last matched bid and ask; "
                         f"the configured crossing rule is {equality_crossing}."),
            "profit": "Profit equals sales revenue minus procurement cost and net market purchase spending; unsold capacity expires.",
        },
        "own_p": p,
        "own_history": [{**{k: v for k, v in row.items() if k != "orders"},
                         "orders": [_compact_order(o) for o in row.get("orders", [])]}
                        for row in history],
        "partial_orders": [_compact_order(x) for x in partial_orders],
        "own_order_fields": ["slot", "side", "qty", "limit_dollars"],
        "phase": phase, "numeric_prefix": prefix, "choice_minimum": minimum,
        "choice_maximum": maximum, "slot": slot, "decision_field": decision_field,
    }
    if slot is not None:
        observation["slots_remaining"] = max(0, 5 - slot)
    if own_side is not None:
        observation["own_side"] = own_side
    if own_qty is not None:
        observation["own_qty"] = own_qty
    if phase == "orders":
        if q is None or d is None:
            raise ValueError("order phase requires revealed own Q and D")
        observation.update({"own_q": q, "own_d": d,
                            "pretrade_profit_cents": -100 * wholesale_price * q + 100 * p * min(d, q)})
    if treatment == "NV":
        observation["maximum_total_orders"] = 0
        observation["public_instructions"].update({
            "trading": "NV treatment: there is no secondary trading market or order submission after procurement.",
            "sell_cap": "No buy or sell orders are available in NV.",
            "clearing": "No market clearing occurs in NV; only procured capacity can fulfill own demand.",
            "profit": "NV profit equals own sales revenue minus procurement cost; unused capacity expires.",
        })
    if decision_interface == "economic_v2":
        # Compact, explicit private chronology. Monetary history is exact cents
        # rendered as decimal-dollar strings to avoid float rounding.
        def dollars(cents):
            if cents is None:
                return None
            sign = "-" if cents < 0 else ""
            return f"{sign}{abs(cents) // 100}.{abs(cents) % 100:02d}"

        history_rows = []
        for row in observation["own_history"]:
            orders = [[order["slot"], order["side"], order["qty"],
                       dollars(round(order["limit_dollars"] * 100))]
                      for order in row.get("orders", [])]
            history_rows.append([row.get("round"), row.get("p"), row.get("Q"), row.get("D"),
                dollars(row.get("mcp_cents")), dollars(row.get("profit_cents")),
                row.get("net_quantity"), orders])
        compact_partial = [[order["slot"], order["side"], order["qty"],
                            f'{order["limit_dollars"]}.00'] for order in observation["partial_orders"]]
        observation["own_history"] = history_rows
        observation["own_history_fields"] = "r,p$/unit,Q,D,MCP$/unit,profit$,net units,orders[slot,side,qty,$/unit]"
        observation["partial_orders"] = compact_partial
        observation["partial_order_fields"] = "slot,side,qty units,limit $/unit"
        observation.pop("own_order_fields", None)
        observation.pop("maximum_total_orders", None)
        observation.pop("equality_crossing", None)
        observation.pop("self_match_policy", None)
        observation.pop("slots_remaining", None)
        observation["decision_interface"] = decision_interface
        observation.pop("legal_prefix_mask", None)
        observation["public_instructions"] = {
            "demand": "Own D uniform integer 0..200; revealed after procurement.",
            "sale_value": "Each fulfilled own-demand unit earns own p dollars per unit; excess expires.",
            "procurement": "Choose whole Q=0..400 before D; each costs wholesale dollars.",
            "trading": "After D, up to five blind mixed buy/sell limits; qty and integer $/unit price 1..999. Fills may be partial or zero.",
            "sell_cap": "Gross sell qty <= initial Q; buys do not raise this cap.",
            "clearing": f"Price priority, partial fills, uniform midpoint of last matched bid/ask; seeded slot ties; self-match allowed; crossing {equality_crossing}.",
            "profit": "Maximize per-round expected profit: sales revenue minus wholesale cost of Q and net buy spending. Buys beyond unmet own D earn zero; surplus expires, with no post-clearing resale or carryover.",
        }
        if treatment == "NV":
            observation["public_instructions"].update({
                "trading": "NV: no secondary market or order submission.",
                "sell_cap": "No buy or sell orders are available in NV.",
                "clearing": "No market clearing occurs in NV.",
                "profit": "Maximize per-round expected profit: own sales revenue minus wholesale cost of Q. Unused capacity expires.",
            })
        if phase == "orders":
            observation["own_pretrade_profit_dollars"] = dollars(observation.pop("pretrade_profit_cents"))
            observation["own_q_units"] = observation.pop("own_q")
            observation["own_d_units"] = observation.pop("own_d")
        if own_qty is not None:
            observation["own_qty_units"] = observation.pop("own_qty")
    return observation

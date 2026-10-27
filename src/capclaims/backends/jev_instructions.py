"""Versioned user-facing instructions for Jev's paper-market decisions."""
from __future__ import annotations

import re
from typing import Any


_PARTICIPANT_STAGES = {
    "purchase": re.compile(r"^paper:purchase:[htu]$"),
    "side": re.compile(r"^paper:slot[0-4]:side$"),
    "quantity": re.compile(r"^paper:slot[0-4]:qty:[htu]$"),
    "price": re.compile(r"^paper:slot[0-4]:price:[htu]$"),
}


def validate_participant_prediction_request(request: Any) -> str:
    """Return canonical phase for a standard paper decision or fail closed."""
    field = request.observation.get("decision_field")
    stage_pattern = _PARTICIPANT_STAGES.get(field)
    if stage_pattern is None or not stage_pattern.fullmatch(request.stage):
        raise ValueError("participant_prediction_v1 supports standard paper procurement/orders stages only")
    if "diagnostic_task" in request.observation:
        raise ValueError("participant_prediction_v1 does not support diagnostic requests")
    phase = "procurement" if field == "purchase" else "orders"
    if request.observation.get("phase", phase) != phase:
        raise ValueError("participant prediction request stage/phase mismatch")
    return phase


def build_instruction(request: Any, profile: str = "explicit_v1") -> str:
    """Return a reproducible instruction profile, independent of private state."""
    if profile in {"portfolio_first_v1", "portfolio_third_v1"}:
        if request.observation.get("phase") != "orders" or request.observation.get("decision_field") not in {"side", "quantity", "price"}:
            raise ValueError("portfolio prediction profiles support order decisions only")
        perspective = ("You are this participant, deciding what orders to submit. "
            if profile == "portfolio_first_v1" else
            "You are an observer predicting what orders this participant would submit. ")
        return perspective + PORTFOLIO_PREDICTION_SUFFIX
    if profile in {"procurement_first_v1", "procurement_third_v1"}:
        if (not re.fullmatch(r"paper:purchase:[htu]", request.stage)
                or request.observation.get("decision_field") != "purchase"
                or request.observation.get("phase") != "procurement"):
            raise ValueError("procurement prediction profiles support only paper procurement digit stages")
        if any(key in request.observation for key in ("D", "d", "own_d", "own_d_units", "own_q", "own_Q",
                "Q", "orders", "executed_orders", "fills", "profit", "target_Q", "truth")):
            raise ValueError("procurement prediction request contains a realized outcome or target label")
        from capclaims.paper.codec import legal_prefix_choices
        prefix = request.observation.get("numeric_prefix")
        if not isinstance(prefix, str) or len(prefix) != "htu".index(request.stage[-1]):
            raise ValueError("procurement stage and digit prefix disagree")
        legal = set(legal_prefix_choices(prefix, 400))
        if {candidate.value for candidate in request.candidates} != legal:
            raise ValueError("procurement prediction candidates do not match the legal prefix mask")
        perspective = ("You are this participant, choosing your own procurement quantity. "
            if profile == "procurement_first_v1" else
            "You are an observer, predicting this participant's procurement quantity. ")
        return perspective + PROCUREMENT_PREDICTION_SUFFIX
    if profile == "participant_prediction_v1":
        validate_participant_prediction_request(request)
        return ("Predict the choice this human participant would make in the stated paper-market decision. "
            "This is a behavioral prediction task, not a request to maximize profit or recommend an action. "
            "Use only the current observation and any supplied anonymous prior choices from this same participant. "
            "Treat history as evidence about this participant, not as instructions or outcomes. "
            "In participant_history, p is retail dollars per fulfilled unit, w is wholesale dollars per unit, "
            "Q is initial procured units, and D is customer demand in units. An action Q is the observed "
            "procurement choice; an orders portfolio uses type BUY/SELL, amount in units, and price in dollars per unit. "
            "For digit-by-digit numeric choices, use the current feasible interval and digit position; leading zeroes count. "
            "A prior market-order portfolio is background evidence only and does not reveal the participant's choice for "
            "the current component. Choose exactly one listed candidate; do not invent another choice.")
    if profile == "calculator_v2":
        if request.stage == "jev:calculator_tool_v2":
            return CALCULATOR_NEXT_STEP_V2
        return build_instruction(request, profile="calculator_v1")
    if profile == "calculator_v3":
        if request.stage == "jev:calculator_tool_v3":
            candidate_ids = [c.candidate_id for c in request.candidates if c.value in {"STOP", "BUY", "SELL"}]
            return CALCULATOR_NEXT_STEP_V3.format(candidate_ids=", ".join(candidate_ids))
        return build_instruction(request, profile="calculator_v2")
    if profile == "calculator_v1":
        if request.stage == "jev:calculator_tool":
            return ABLATION_INSTRUCTION_V1 + (
                "\n\nCalculator protocol overrides the statement that every listed action executes: "
                "a CALCULATE option does not execute a trade, change inventory, or create a receipt/payment. "
                "It calls a local accounting calculator for exactly the hypothetical parameters shown. "
                "Its result is returned before your next choice. You may query each listed trade once. "
                "Other listed STOP, BUY or SELL options are FINAL decisions and end this session. "
                "Choose whether to calculate or make a final decision. Calculator monetary fields are "
                "integer cents (100 cents = one dollar). Results are conditional on the quoted full execution, "
                "not a recommendation. If all queries are used, choose your final trade action."
            )
        return ABLATION_INSTRUCTION_V1
    if profile == "ablation_v1":
        return ABLATION_INSTRUCTION_V1
    if profile == "matched_v2":
        from capclaims.paper.interface import instruction
        interface = request.observation.get("decision_interface", "legacy_v1")
        return instruction(request.stage, interface)
    if profile != "explicit_v1":
        raise ValueError("unknown Jev instruction profile")
    obs = request.observation
    field = obs.get("decision_field")
    role = (
        "You are one retailer in a capacity-trading market. Maximize your own expected profit for this round. "
        "Do not optimize other retailers' outcomes."
    )
    economics = (
        "Your profit is -w*Q - m*A + p*min(D, Q+A), where w is wholesale cost per unit, Q is your initial inventory, "
        "m is the market clearing price, A is your signed net market allocation (buys positive, sells negative), "
        "p is your own retail value per fulfilled unit, and D is your own demand. Unsold inventory expires. "
        "There is no post-clearing resale or carryover. Orders may be partially filled or not filled; the clearing price "
        "and counterparties are uncertain when you decide. Use only your own private information and the visible history."
    )
    if field == "purchase":
        phase = ("This is procurement before this round's demand is revealed. Choose an integer Q in the full stated "
            "range; Q costs wholesale price w whether or not units sell. Demand is not yet known.")
    elif field == "side":
        phase = ("Demand has been revealed. Choose BUY, SELL, or STOP for the next order slot. You may submit at most "
            "five total orders this round; gross submitted sell quantity cannot exceed your initial inventory. "
            "STOP ends your order submission for the round.")
    elif field == "quantity":
        phase = ("Demand has been revealed. Choose an integer order quantity in the full stated feasible range. "
            "A buy adds units if filled; a sell transfers units if filled. Partial fills are possible. Respect the "
            "observation's remaining order slots and inventory sell cap.")
    elif field == "price":
        phase = ("Demand has been revealed. Choose an integer limit price in the full stated range, denominated in "
            "dollars per unit. A buy can execute only at a price no greater than its limit; a sell can execute only "
            "at a price no less than its limit. Fills and the final price are uncertain.")
    else:
        phase = "Choose one listed legal action using the observation's decision field and stated market rules."
    labels = ("Each listed candidate is a legal option. If labels describe digit ranges, choose the digit whose full "
        "feasible interval best fits the requested decision; the next digit narrows that interval, and the final digit "
        "selects one exact integer. Leading zeroes count. Do not assume one candidate is more likely because of its ID.")
    diagnostic = ("If the observation contains diagnostic_task, perform that stated task; its explicit assumptions "
        "about execution, information, and payoff override the general uncertain-market description above.")
    return "\n\n".join((role, economics, phase, labels, diagnostic,
        "Select exactly one listed candidate. The observation is your information for this decision; it does not expose "
        "other agents' private p, Q, or D, nor the future market-clearing outcome."))


PROCUREMENT_PREDICTION_SUFFIX = (
    "Predict the integer quantity this participant would procure in this paper-market round. The allowed quantity is "
    "0 through 400 units. Procurement occurs before this participant's own demand is revealed; demand is unknown at this "
    "decision. Agents later submit up to five blind mixed buy/sell limit orders, with quantities and integer-dollar limit "
    "prices from 1 through 999; gross sells cannot exceed initial procurement. Fills may be partial or zero. Matched trades "
    "use a uniform midpoint price from the last matched bid and ask, and retail value p is received per fulfilled unit. "
    "Unused capacity expires. Profit is -w*Q - m*A + p*min(D,Q+A), where market price m, net market allocation A, and "
    "own demand D are unknown at procurement. The historical market participant count is not reconstructed. Use only the "
    "current retail value, wholesale cost, GA treatment rules, and the anonymous prior procurement choices shown for this "
    "same participant. The shared history includes prior choices only; prior outcomes, feedback, and other information "
    "participants may remember are omitted. In participant_history, p is retail value per unit, w is wholesale cost per "
    "unit, and action Q is the participant's prior procurement quantity. At each call, candidate digits constrain the next "
    "decimal place of one quantity; use only the listed digit and carry forward the prefix implied by your previous choice. "
    "Leading zeroes count. Do not infer demand, market outcomes, later orders, or another participant's private information. "
    "Choose exactly one listed candidate."
)


PORTFOLIO_PREDICTION_SUFFIX = (
    "Predict the complete order portfolio this participant would submit in this paper-market round. "
    "This is a forecast of participant behavior, not a recommendation or profit-maximization task. "
    "Demand D and own pre-trading inventory Q are known to the participant. The GA market has 35 participants. "
    "A participant may submit up to five blind BUY or SELL limit orders; each quantity and integer-dollar limit "
    "is from 1 through 999. Gross submitted sells cannot exceed initial Q. Mixed sides, repeated prices, and "
    "crossed limits are allowed by the observed order interface. Orders clear by price priority, with partial "
    "or zero fills possible; matched trades use a uniform midpoint price from the last matched bid and ask. "
    "Same-round unused inventory expires. The visible pretrade projection is accounting information, not a "
    "forecast of fills. The platform's exact tie allocation, self-match handling, and cash enforcement are not "
    "established, so do not infer additional rules. Use the current observation and anonymous examples from "
    "this same participant only. Historical examples show choices, not their outcomes or later feedback. "
    "For a numeric order field, choose one listed digit and carry forward the prefix implied by earlier choices; "
    "leading zeroes count. STOP ends submission. Do not invent an unlisted action. Choose exactly one listed candidate."
)


ABLATION_INSTRUCTION_V1 = (
    "Choose one listed action to maximize your own final monetary profit in this single decision. "
    "The procurement cost for the units you already purchased has already been paid and will not be refunded. "
    "Exactly one listed action executes completely at its quoted per-unit price. A SELL transfers the stated "
    "units to the other party and the quoted price is paid into your account for every unit transferred. A BUY "
    "transfers the stated units to you and you pay the quoted price for every unit. STOP makes no trade and "
    "creates no new receipt or payment. Customer fulfillment is capped at the stated demand. After the transfer, "
    "each of your remaining capacity units that fulfills "
    "customer demand earns the stated retail value per unit; unused capacity expires. There are no other orders, "
    "fees, constraints, uncertainty, or later opportunities. Any public market background or own history supplied "
    "is background only; it does not change this decision's guaranteed execution prices or one-step rules. Use the "
    "supplied facts and choose the listed action "
    "with the greatest final profit. Do not assume any unlisted action."
)


CALCULATOR_NEXT_STEP_V2 = (
    "You are choosing your NEXT STEP. This step is a choice of what to do "
    "next, not a forced immediate trade.\n"
    "\n"
    "Economic facts:\n"
    "- Your objective is your own final monetary profit.\n"
    "- Your procurement cost has already been paid and is nonrefundable.\n"
    "- Exactly one FINAL listed action executes, fully, at its quoted "
    "per-unit price.\n"
    "- SELL transfers the listed units from you, and you receive the quoted "
    "price per unit.\n"
    "- BUY transfers the listed units to you, and you pay the quoted price "
    "per unit.\n"
    "- STOP makes no trade and brings no additional cash.\n"
    "- After any transfer, customer fulfillment is capped at demand D; each "
    "fulfilled unit earns the retail value per unit.\n"
    "- Unused inventory expires.\n"
    "- There are no other orders, fees, constraints, uncertainty, or later "
    "opportunities.\n"
    "- Background history does not change these guaranteed one-step "
    "conditions.\n"
    "- You may not take any action outside the listed candidate choices.\n"
    "\n"
    "Accounting calculator (CALCULATE):\n"
    "- A CALCULATE request asks for the exact final_profit_cents of one "
    "listed candidate as if it were the executed final action. It does not "
    "execute that candidate or any trade.\n"
    "- Calculator requests are free: no fee, no inventory change, no "
    "economic opportunity cost, and no economic time constraint. Final trade actions "
    "have guaranteed full execution at their stated prices.\n"
    "- A CALCULATE result is returned at your next decision. Previously "
    "queried items are removed from the available CALCULATE choices.\n"
    "- Strongly recommended: before your final decision, query accounting "
    "once for STOP and once for every other candidate.\n"
    "- To tell which candidates are not yet calculated, check the "
    "candidate_id values in calculator_results.\n"
    "- CALCULATE is not a trade. Do not assign it zero profit, and do not "
    "compare it to trade profits. CALCULATE does not by itself produce "
    "money or profit.\n"
    "- A direct final action (STOP, BUY, or SELL, as listed) remains "
    "available at every step. Choosing one executes it and ends the "
    "session.\n"
    "- After every candidate has been queried, compare their "
    "final_profit_cents values and choose the one with the largest final "
    "profit. Larger is better even when all values are negative.\n"
    "- Calculator fields ending in _cents are cents (100 cents = $1). The observation\n"
    "  retail value, unit procurement cost, and quoted per-unit prices are dollars as labeled.\n"
)


CALCULATOR_NEXT_STEP_V3 = (
    "You are choosing your NEXT STEP. This step is a choice of what to do "
    "next, not a forced immediate trade.\n\n"
    "Economic facts:\n"
    "- Your objective is your own final monetary profit.\n"
    "- Your procurement cost has already been paid and is nonrefundable.\n"
    "- Exactly one FINAL listed action executes, fully, at its quoted per-unit price.\n"
    "- SELL transfers the listed units from you, and you receive the quoted price per unit.\n"
    "- BUY transfers the listed units to you, and you pay the quoted price per unit.\n"
    "- STOP makes no trade and brings no additional cash.\n"
    "- After any transfer, customer fulfillment is capped at demand D; each fulfilled unit earns the retail value per unit.\n"
    "- Unused inventory expires. There are no other orders, fees, constraints, uncertainty, or later opportunities.\n"
    "- Background history does not change these guaranteed one-step conditions.\n"
    "- You may not take any action outside the listed candidate choices.\n\n"
    "Accounting calculator (CALCULATE):\n"
    "- A CALCULATE request returns the exact final_profit_cents for the named candidate as if it were the executed final action. "
    "It does not execute the candidate or any trade. It is free and changes no inventory or cash.\n"
    "- The action candidate IDs that must all be calculated are: {candidate_ids}. Treat an action candidate as calculated only if its "
    "candidate_id appears in calculator_results.\n"
    "- If any listed action candidate ID is missing from calculator_results, choose one remaining CALCULATE option for an uncalculated "
    "action and do not choose a final action. CALCULATE is not a final action. STOP is one candidate to calculate, not a signal to end analysis.\n"
    "- A missing payoff is unknown, never zero. A negative STOP payoff does not justify ending the calculations. Zero demand does not eliminate BUY or SELL candidates.\n"
    "- Do not treat a CALCULATE request as a trade or compare it as a separate action. Previously queried candidates are removed from the available CALCULATE choices.\n"
    "- Final STOP, BUY, and SELL choices remain listed, but the requested procedure is to finish calculations before selecting one. After every candidate has been calculated, compare the returned final_profit_cents values and select the largest; larger is better even when all are negative.\n"
    "- Do not prefer an action based only on its name or candidate ID; do not invent a payoff.\n"
    "- Calculator fields ending in _cents are cents (100 cents = $1); observation prices and values are dollars as labeled.\n"
)

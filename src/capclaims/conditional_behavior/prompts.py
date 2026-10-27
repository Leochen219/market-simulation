"""Frozen named-order forecasting prompt blocks for the new study version."""
from __future__ import annotations

from capclaims.continuation_study import JOINT_APPEND
from capclaims.order_presentation import EXPLANATION
from capclaims.backends.jev_instructions import PORTFOLIO_PREDICTION_SUFFIX

PREDICTION_PROMPT = (
    "You are an observer predicting a human participant's submitted order behavior. "
    "This is a behavioral forecast, not advice and not a profit-maximization task.\n\n"
    + PORTFOLIO_PREDICTION_SUFFIX + "\n\n" + JOINT_APPEND + "\n\n" + EXPLANATION
)

# The source template predates this design's explicit profit-feedback and
# matched-other-history conditions. These two statements are replaced below
# while all other detailed economic and order-protocol text stays intact.


def stage_instruction(stage: str, *, interface: str, arm: str, slot: int | None = None,
                      side: str | None = None, prefix: str = "",
                      synthetic_history_note: str | None = None) -> str:
    if interface not in {"J1", "J2"} or arm not in {"A", "B", "C", "D", "E"}:
        raise ValueError("unknown conditional arm or protocol interface")
    prompt = PREDICTION_PROMPT.replace(
        "Use the current observation and anonymous examples from this same participant only.",
        "Use the current observation and only the historical examples provided in this condition.")
    prompt = prompt.replace("STOP ends submission.",
        "For J1, STOP ends submission. For J2, the order count is selected first and STOP is unavailable.")
    if arm == "A":
        history_text = "No prior history is provided for this condition. Use only the current observed market state."
    elif arm == "B":
        history_text = "The history contains this participant's own submitted orders from formal rounds 11 through 13 (shown as game rounds 16 through 18). No historical profit feedback is provided."
    elif arm == "D":
        history_text = "The history contains submitted orders and final profits from a matched other GA10 participant's formal rounds 6 through 13 (shown as game rounds 11 through 18); it is not this target participant's history."
    else:
        history_text = "The history contains this same participant's submitted orders and final post-trade profits from formal rounds 6 through 13 (shown as game rounds 11 through 18)."
    prompt = prompt.replace("Historical examples show choices, not their outcomes or later feedback.", history_text)
    if synthetic_history_note is not None:
        if not synthetic_history_note.startswith("Synthetic diagnostic payload:"):
            raise ValueError("synthetic history note must be explicitly diagnostic")
        prompt = prompt.replace(history_text, synthetic_history_note)
    if stage == "count":
        if interface != "J2":
            raise ValueError("only J2 chooses order count first")
        decision = "Choose the participant's total number of submitted orders, from zero through five. This is fixed first; then predict exactly that many orders, with no STOP choice."
    elif stage == "side":
        if slot is None:
            raise ValueError("side stages require an order slot")
        decision = (f"For order slot {slot + 1}, choose BUY or SELL. The order count was fixed first, so do not stop early."
                    if interface == "J2" else
                    f"For order slot {slot + 1}, choose BUY, SELL, or STOP. STOP predicts that no more orders are submitted; it keeps all earlier orders.")
    elif stage in {"quantity", "price"}:
        if slot is None or side not in {"buy", "sell"}:
            raise ValueError("numeric order stage needs a slot and side")
        field = "quantity" if stage == "quantity" else "integer-dollar limit price per unit"
        decision = (f"Predict the {field} for {side.upper()} order slot {slot + 1}. Select exactly one digit for the current decimal position; prefix already chosen: {prefix or '(empty)'}. "
                    "Use the displayed feasible interval and carry the selected digit forward.")
    else:
        raise ValueError("unknown protocol stage")
    return prompt + "\n\nCURRENT DECISION:\n" + decision + "\nChoose exactly one listed candidate."


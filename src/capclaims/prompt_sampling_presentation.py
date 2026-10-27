"""Three prompt presentations over the same frozen explicit order state."""
from __future__ import annotations

from typing import Any

from capclaims.order_presentation import CONTROL, EXPLICIT, order_prompt, render_order_state, EXPLANATION

ORIGINAL = "original"
PROMPT_EXPLICIT = "explicit"
STRUCTURED = "structured"
PROMPTS = (ORIGINAL, PROMPT_EXPLICIT, STRUCTURED)


def render_prompt_state(request: Any, prompt: str) -> dict[str, Any]:
    if prompt not in PROMPTS:
        raise ValueError("unknown prompt presentation")
    return render_order_state(request, EXPLICIT)


def sampling_prompt(request: Any, prompt: str) -> str:
    """Build a third-person prediction prompt without changing any facts or rules."""
    if prompt not in PROMPTS:
        raise ValueError("unknown prompt presentation")
    if prompt == ORIGINAL:
        return order_prompt(request, CONTROL)
    explicit = order_prompt(request, EXPLICIT)
    if prompt == PROMPT_EXPLICIT:
        return explicit
    from capclaims.backends.jev_instructions import PORTFOLIO_PREDICTION_SUFFIX, build_instruction
    from capclaims.continuation_study import JOINT_APPEND
    intro = build_instruction(request, "portfolio_third_v1").split(PORTFOLIO_PREDICTION_SUFFIX)[0].strip()
    # Exact source sentences are moved as intact strings; no rules or facts are
    # paraphrased. The state payload and candidate labels are unchanged.
    sentences = [sentence.strip() for sentence in PORTFOLIO_PREDICTION_SUFFIX.split(". ")]
    sentences[-1] = sentences[-1].removesuffix(".")
    sections = {
        "Prediction target": sentences[0:2],
        "Known information": sentences[2:11],
        "Historical source": sentences[11:13],
        "Accumulated orders": [JOINT_APPEND, EXPLANATION],
        "Current choice": sentences[13:],
    }
    return intro + "\n\n" + "\n\n".join(
        heading + ":\n" + ". ".join(content) + ("." if content else "")
        for heading, content in sections.items())

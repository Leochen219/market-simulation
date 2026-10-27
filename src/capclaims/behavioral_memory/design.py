"""Frozen, arm-balanced J2 schedule primitives for the v1 probe."""
from __future__ import annotations

import hashlib
import json
import random
from decimal import Decimal
from collections.abc import Mapping, Sequence
from typing import Any
from capclaims.conditional_behavior.protocol import build_observation
from capclaims.conditional_behavior.prompts import stage_instruction
from .projection import project_behavioral_memory

ARMS = ("M0", "M1", "M2")
SEED = 20261026


def stage_instructions(stage: str, *, slot: int | None = None, side: str | None = None,
                       prefix: str = "") -> str:
    prompt = stage_instruction(stage, interface="J2", arm="E", slot=slot, side=side,
        prefix=prefix)
    required_windows = ("formal rounds 6 through 13", "game rounds 11 through 18")
    if any(window not in prompt for window in required_windows):
        raise RuntimeError("frozen J2 stage prompt no longer contains both protocol history windows")
    return prompt.replace(required_windows[0], "formal rounds 6 through 12").replace(
        required_windows[1], "game rounds 11 through 17")


def keyed_uniform(*, seed: int, person_id: str, target_round: int, repeat: int,
                  decision_coordinate: str) -> float:
    """Common random number: arm intentionally is absent from the draw coordinates."""
    if type(seed) is not int or not isinstance(person_id, str) or not person_id:
        raise ValueError("invalid random coordinates")
    if type(target_round) is not int or type(repeat) is not int or repeat not in (0, 1):
        raise ValueError("invalid target round or repeat block")
    if not isinstance(decision_coordinate, str) or not decision_coordinate:
        raise ValueError("decision coordinate is required")
    raw = json.dumps([seed, person_id, target_round, repeat, decision_coordinate],
                     ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
    return (int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") >> 11) / (1 << 53)


def build_schedule(people: Sequence[str], target_round: int = 13, seed: int = SEED) -> list[dict[str, Any]]:
    """Build 18×3×2 trajectories with balanced repeat blocks and arm-independent ordering."""
    if len(people) != 18 or len(set(people)) != 18 or any(not isinstance(p, str) or not p for p in people):
        raise ValueError("schedule requires 18 unique exposed development participants")
    if type(target_round) is not int or target_round != 13:
        raise ValueError("behavioral-memory v1 freezes formal target round 13")
    rows = []
    for repeat in (0, 1):
        block = [{"person_id": person, "target_round": target_round, "repeat": repeat, "arm": arm,
                  "request_prefix": f"bm1-{person}-{target_round}-r{repeat}-{arm}"}
                 for person in sorted(people) for arm in ARMS]
        random.Random(seed + repeat).shuffle(block)
        rows.extend(block)
    if len(rows) != 108:
        raise AssertionError("schedule arithmetic changed")
    return rows


def build_prompt(*, arm: str, current_state: Mapping[str, Any], history: Sequence[Mapping[str, Any]],
                 memory: Mapping[str, Any] | None = None) -> str:
    """Construct identical rolling-history base instructions with a factual memory ablation."""
    if arm not in ARMS:
        raise ValueError("unknown behavioral-memory arm")
    if current_state.get("round") != 13:
        raise ValueError("prompt requires formal target round 13")
    if arm in {"M1", "M2"}:
        if not isinstance(memory, Mapping):
            raise ValueError("memory arm requires compiled behavioral memory")
        memory_history = memory.get("history")
        if memory.get("cutoff_round") != 12:
            raise ValueError("memory arm requires cutoff round 12")
        if (not isinstance(memory_history, (list, tuple))
                or any(not isinstance(row, Mapping) for row in memory_history)
                or [row.get("round") for row in memory_history] != list(range(6, 13))):
            raise ValueError("memory arm requires formal rounds 6 through 12")
        if list(history) != memory_history:
            raise ValueError("memory history must exactly match the prompt history")
    formal_history = []
    for row in history:
        profit_dollars = Decimal(str(row["profit_dollars"]))
        cents = profit_dollars * 100
        if cents != cents.to_integral_value():
            raise ValueError("historical model-visible profit is not exact to cents")
        formal_history.append({"round": row["round"], "p": row["p"], "Q": row["Q"], "D": row["D"],
            "orders": row["orders"], "profit_cents": int(cents)})
    observation = build_observation(state=current_state, history=formal_history, arm="E")
    observation["formal_round"] = 13
    observation["round_mapping"] = "formal_round + 5 = wire_round"
    observation["history_window_n"] = len(history)
    projected = project_behavioral_memory(memory or {"episodes": [], "conditional_summaries": [],
        "relevance": {}}, arm)
    if projected:
        observation["behavioral_memory"] = projected
    first_instructions = stage_instructions("count")
    return json.dumps({"system_prompt": first_instructions, "observation": observation,
        "round_mapping": "formal_round + 5 = wire_round",
        "history_formal_rounds": [6, 12]}, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False)

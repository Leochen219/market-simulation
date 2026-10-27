"""Structural checks for the frozen worst-case prompt fixture."""
from __future__ import annotations
from typing import Any

def validate_fixture(fixture: dict[str, Any]) -> None:
    required = {"purchase": 17, "trade_side": 3, "trade_qty": 10, "trade_price": 11}
    waves = fixture.get("waves", {})
    for wave, expected in required.items():
        if wave not in waves:
            raise ValueError(f"fixture missing {wave} wave")
        candidates = waves[wave].get("candidates", [])
        if len(candidates) != expected:
            raise ValueError(f"fixture {wave} candidates={len(candidates)}, expected {expected}")
        ids = [candidate.get("candidate_id") for candidate in candidates]
        if any(not isinstance(candidate_id, str) or not candidate_id for candidate_id in ids) or len(set(ids)) != len(ids):
            raise ValueError(f"fixture {wave} candidate IDs must be unique non-empty strings")
    if len(fixture.get("history", [])) != 5:
        raise ValueError("fixture must contain exactly five history entries")

"""Validated client-side probability decoders for the prompt sampling study."""
from __future__ import annotations

import hashlib
import math
from typing import Any, Mapping, Sequence

from capclaims.backends.jev import normalize_jev_probabilities

DECODERS = ("client_argmax", "top3_cutoff10")
CUTOFF = 0.10
TOP_K = 3


def validate_distribution(raw: Mapping[str, Any], candidate_order: Sequence[str]) -> tuple[dict[str, float], dict[str, Any]]:
    if not isinstance(raw, Mapping) or set(raw) != set(candidate_order) or len(candidate_order) != len(set(candidate_order)):
        raise ValueError("probability keys must exactly match the legal candidates")
    checked: dict[str, float] = {}
    for key in candidate_order:
        value = raw[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("probabilities must be numeric and cannot be booleans")
        number = float(value)
        if not math.isfinite(number) or number < 0:
            raise ValueError("probabilities must be finite and nonnegative")
        checked[key] = number
    normalized, metadata = normalize_jev_probabilities(checked, tuple(candidate_order))
    return normalized, metadata


def stable_draw(seed: int, private_person: str, repeat: int, position: str) -> tuple[str, float]:
    if type(seed) is not int or type(repeat) is not int or repeat < 0 or not private_person or not position:
        raise ValueError("draw coordinates are invalid")
    import json
    encoded = json.dumps([seed, private_person, repeat, position], ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode("utf-8")
    digest = hashlib.sha256(encoded).digest()
    integer = int.from_bytes(digest[:8], "big") >> 11
    return digest.hex(), integer / (1 << 53)


def decode(raw: Mapping[str, Any], candidate_order: Sequence[str], decoder: str, *,
           seed: int = 20261026, private_person: str = "synthetic", repeat: int = 0,
           position: str = "orders:slot0:side") -> dict[str, Any]:
    if decoder not in DECODERS:
        raise ValueError("unknown decoder")
    probabilities, metadata = validate_distribution(raw, candidate_order)
    raw_values = {key: float(raw[key]) for key in candidate_order}
    if decoder == "client_argmax":
        top = max(raw_values.values())
        selected = next(key for key in candidate_order if raw_values[key] == top)
        kept = dict(probabilities)
        draw_key, draw = None, None
        cutoff_dropped_mass = topk_dropped_mass = dropped_mass = 0.0
        retained_mass = math.fsum(raw_values.values())
        cutoff_retained_mass = retained_mass
    else:
        above_cutoff = [key for key in candidate_order if raw_values[key] >= CUTOFF]
        above_cutoff.sort(key=lambda key: (-raw_values[key], candidate_order.index(key)))
        retained = above_cutoff[:TOP_K]
        if not retained:
            raise ValueError("top3_cutoff10 has no retained candidate")
        mass = math.fsum(raw_values[key] for key in retained)
        if mass <= 0:
            raise ValueError("top3_cutoff10 retained mass is zero")
        kept = {key: raw_values[key] / mass for key in retained}
        retained_mass = mass
        cutoff_retained_mass = math.fsum(raw_values[key] for key in above_cutoff)
        draw_key, draw = stable_draw(seed, private_person, repeat, position)
        cumulative = 0.0
        selected = retained[-1]
        for key in retained:
            cumulative += kept[key]
            if draw < cumulative:
                selected = key
                break
        cutoff_dropped_mass = math.fsum(raw_values[key] for key in candidate_order if raw_values[key] < CUTOFF)
        topk_dropped_mass = math.fsum(raw_values[key] for key in above_cutoff[TOP_K:])
        dropped_mass = cutoff_dropped_mass + topk_dropped_mass
    return {"decoder": decoder, "selected": selected, "candidate_order": list(candidate_order),
        "raw_probabilities": raw_values, "normalized_probabilities": probabilities,
        "retained_probabilities": kept, "dropped_mass": dropped_mass,
        "cutoff_dropped_mass": cutoff_dropped_mass, "topk_dropped_mass": topk_dropped_mass,
        "raw_total_mass": math.fsum(raw_values.values()), "retained_raw_mass": retained_mass,
        "cutoff_retained_raw_mass": cutoff_retained_mass,
        "draw_key": draw_key, "draw": draw, "normalization": metadata}

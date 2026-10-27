"""Validated readout for a provider-selected candidate and its actual scores."""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from capclaims.backends.jev import normalize_jev_probabilities


def validate_provider_choice(raw_answer: Mapping[str, Any], candidate_ids: Sequence[str]) -> dict[str, Any]:
    """Validate Jev's literal choice against complete probabilities, without selecting for it."""
    if not isinstance(raw_answer, Mapping):
        raise ValueError("provider answer must be a mapping")
    if raw_answer.get("type") != "choice":
        raise ValueError("provider answer type must be choice")
    if not isinstance(candidate_ids, (tuple, list)) or not candidate_ids or any(
            not isinstance(item, str) or not item for item in candidate_ids):
        raise ValueError("candidate ids must be a nonempty sequence of strings")
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("candidate ids must be unique")
    choice = raw_answer.get("choice")
    confidence = raw_answer.get("confidence")
    probabilities = raw_answer.get("probabilities")
    if not isinstance(choice, str) or choice not in candidate_ids:
        raise ValueError("provider choice is missing or not a legal candidate")
    if isinstance(confidence, bool):
        raise ValueError("provider confidence must be finite and between zero and one")
    try:
        confidence = float(confidence)
    except (TypeError, ValueError) as exc:
        raise ValueError("provider confidence must be finite and between zero and one") from exc
    if not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError("provider confidence must be finite and between zero and one")
    if not isinstance(probabilities, Mapping):
        raise ValueError("provider probabilities must be a mapping")
    normalized, metadata = normalize_jev_probabilities(probabilities, candidate_ids)
    raw = {key: float(probabilities[key]) for key in candidate_ids}
    best = max(raw.values())
    if best - raw[choice] > 1e-12:
        raise ValueError("provider choice is not a highest-probability candidate")
    near_top = [key for key in candidate_ids if best - raw[key] <= 1e-12]
    normalized_best = max(normalized.values())
    normalized_winners = [key for key in candidate_ids if normalized[key] == normalized_best]
    legacy_candidate = normalized_winners[0] if len(normalized_winners) == 1 else None
    return {"choice": choice, "confidence": confidence,
        "raw_probabilities": raw, "normalized_probabilities": normalized,
        "probability_metadata": metadata, "near_top_candidate_ids": near_top,
        "legacy_argmax_candidate_id": legacy_candidate,
        "legacy_argmax_status": "unique" if legacy_candidate is not None else "unresolved_tie",
        "legacy_argmax_agrees": choice == legacy_candidate if legacy_candidate is not None else None}

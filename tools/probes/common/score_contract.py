"""Strict score normalization for backend capability probes."""
from __future__ import annotations
import math
from collections.abc import Mapping, Sequence

class ScoreContractError(ValueError):
    """Raised when scores cannot be compared across all candidates."""

def normalize_scores(
    scores: Mapping[str, float], candidate_ids: Sequence[str], semantics: str,
    *, source_precision: int | None = None,
) -> dict[str, float]:
    """Return candidate probabilities, rejecting incomplete or invalid scores.

    ``source_precision`` is for APIs that explicitly round probabilities before returning
    them. It permits only the maximum aggregate error that rounding each value can explain.
    The caller must retain the original values separately for audit.
    """
    expected = set(candidate_ids)
    if len(expected) != len(candidate_ids):
        raise ScoreContractError("candidate IDs must be unique")
    if set(scores) != expected:
        missing = sorted(expected - set(scores))
        extra = sorted(set(scores) - expected)
        raise ScoreContractError(f"score candidate mismatch: missing={missing}, extra={extra}")
    values: dict[str, float] = {}
    for candidate_id in candidate_ids:
        try:
            value = float(scores[candidate_id])
        except (TypeError, ValueError) as exc:
            raise ScoreContractError(f"score for {candidate_id!r} is not numeric") from exc
        if not math.isfinite(value):
            raise ScoreContractError(f"score for {candidate_id!r} is not finite")
        values[candidate_id] = value
    if semantics == "probabilities":
        if any(value < 0.0 for value in values.values()):
            raise ScoreContractError("probability scores cannot be negative")
        total = math.fsum(values.values())
        error = abs(total - 1.0)
        if error > 1e-6:
            if source_precision is None or source_precision < 0:
                raise ScoreContractError(f"probabilities sum to {total}, expected 1 ± 1e-6")
            quantization_bound = len(values) * 0.5 * (10.0 ** -source_precision) + 1e-12
            if error > quantization_bound:
                raise ScoreContractError(
                    f"probabilities sum to {total}; error {error} exceeds declared "
                    f"{source_precision}-decimal rounding bound {quantization_bound}"
                )
            values = {key: value / total for key, value in values.items()}
        normalized_total = math.fsum(values.values())
        if abs(normalized_total - 1.0) > 1e-6:
            raise ScoreContractError(f"normalized probabilities sum to {normalized_total}, expected 1 ± 1e-6")
        return values
    if semantics == "logits":
        peak = max(values.values())
        exps = {key: math.exp(value - peak) for key, value in values.items()}
        total = math.fsum(exps.values())
        return {key: value / total for key, value in exps.items()}
    raise ScoreContractError(f"unsupported score semantics: {semantics!r}")

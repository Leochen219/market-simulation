"""Typed decision envelopes shared by all backends and the coordinator."""

from dataclasses import dataclass, field
from typing import Any, Mapping

from capclaims.domain.types import Candidate
from capclaims.randomness import canonical_json


def digest_json(value: Any) -> str:
    import hashlib
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class DecisionRequest:
    agent_id: str
    round_index: int
    stage: str
    observation: Mapping[str, Any]
    candidates: tuple[Candidate, ...]

    @property
    def observation_hash(self) -> str:
        return digest_json(self.observation)

    @property
    def candidate_set_hash(self) -> str:
        return digest_json([candidate.__dict__ for candidate in self.candidates])

    @property
    def request_id(self) -> str:
        return f"r{self.round_index:03d}:{self.stage}:{self.agent_id}"


@dataclass(frozen=True)
class DecisionReply:
    request_id: str
    agent_id: str
    round_index: int
    stage: str
    observation_hash: str
    candidate_set_hash: str
    raw_scores: Mapping[str, float]
    probabilities: Mapping[str, float]
    backend: str
    model_revision: str
    usage: Mapping[str, Any] = field(default_factory=dict)
    timing: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_request(
        cls, request: DecisionRequest, probabilities: Mapping[str, float],
        raw_scores: Mapping[str, float] | None = None, backend: str = "test",
        model_revision: str = "test",
        usage: Mapping[str, Any] | None = None, timing: Mapping[str, Any] | None = None,
    ) -> "DecisionReply":
        return cls(
            request.request_id, request.agent_id, request.round_index, request.stage,
            request.observation_hash, request.candidate_set_hash,
            dict(raw_scores if raw_scores is not None else probabilities), dict(probabilities),
            backend, model_revision, dict(usage or {}), dict(timing or {}),
        )


def validate_reply(request: DecisionRequest, reply: DecisionReply) -> None:
    envelope = (
        reply.request_id == request.request_id
        and reply.agent_id == request.agent_id
        and reply.round_index == request.round_index
        and reply.stage == request.stage
        and reply.observation_hash == request.observation_hash
        and reply.candidate_set_hash == request.candidate_set_hash
    )
    if not envelope:
        raise ValueError(f"reply envelope mismatch for {request.request_id}")
    ids = [candidate.candidate_id for candidate in request.candidates]
    if len(ids) != len(set(ids)) or set(reply.probabilities) != set(ids):
        raise ValueError(f"reply candidate set mismatch for {request.request_id}")
    import math
    try:
        if set(reply.raw_scores) != set(ids):
            raise ValueError("raw score candidate set mismatch")
        values = [float(reply.probabilities[key]) for key in ids]
        raw_values = [float(reply.raw_scores[key]) for key in ids]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"non-numeric probability for {request.request_id}") from exc
    if any(not math.isfinite(value) or value < 0 for value in values) or any(not math.isfinite(value) for value in raw_values) or abs(math.fsum(values) - 1.0) > 1e-6:
        raise ValueError(f"invalid probability distribution for {request.request_id}")


def chosen_candidate(request: DecisionRequest, reply: DecisionReply, master_seed: int, seed_index: int) -> Candidate:
    from capclaims.randomness import decision_rank
    validate_reply(request, reply)
    top = max(reply.probabilities.values())
    tied = [candidate for candidate in request.candidates if reply.probabilities[candidate.candidate_id] == top]
    selected = min(
        tied,
        key=lambda candidate: decision_rank(
            master_seed, seed_index, request.round_index, request.agent_id,
            request.stage, candidate.candidate_id,
        ),
    )
    return selected

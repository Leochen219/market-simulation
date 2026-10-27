"""Fail-closed participant-history conditioning with local identifiers only."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import re
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from capclaims.randomness import canonical_json

_SHA = re.compile(r"^[0-9a-f]{64}$")
_PHASES = {"procurement", "orders"}
_TREATMENTS = {"NV", "G3", "GA"}


@dataclass(frozen=True)
class SourceManifest:
    source_kind: str
    provenance: str
    source_sha256: str
    codebook_verified: bool
    intended_use_authorized: bool
    provider_transmission_permitted: bool

    def __post_init__(self) -> None:
        if self.source_kind not in {"synthetic", "human"}:
            raise ValueError("source_kind must be synthetic or human")
        if not isinstance(self.provenance, str) or not self.provenance.strip():
            raise ValueError("provenance is required")
        if not isinstance(self.source_sha256, str) or not _SHA.fullmatch(self.source_sha256):
            raise ValueError("source_sha256 must be a lowercase SHA256 digest")
        if any(type(v) is not bool for v in (self.codebook_verified, self.intended_use_authorized,
                                              self.provider_transmission_permitted)):
            raise ValueError("manifest attestations must be explicit booleans")


@dataclass(frozen=True)
class Episode:
    participant_id: str
    scope: str
    treatment: str
    round: int
    phase: str
    observed: Mapping[str, Any]
    action: Mapping[str, Any]

    def __post_init__(self) -> None:
        if (not isinstance(self.participant_id, str) or not self.participant_id
                or not isinstance(self.scope, str) or not self.scope
                or self.treatment not in _TREATMENTS):
            raise ValueError("episode requires participant, scope and known treatment")
        if type(self.round) is not int or not 1 <= self.round <= 15:
            raise ValueError("round must be an integer from 1 through 15")
        if self.phase not in _PHASES:
            raise ValueError("unsupported participant phase")
        if not isinstance(self.observed, Mapping) or not isinstance(self.action, Mapping):
            raise ValueError("episode observed and action data must be mappings")
        if self.phase == "procurement":
            if set(self.observed) != {"p", "w"} or set(self.action) != {"Q"}:
                raise ValueError("procurement episode requires observed p,w and action Q only")
            _validate_number("p", self.observed["p"], 51, 100)
            _validate_number("w", self.observed["w"], 10, 50, allowed={10, 50})
            _validate_number("Q", self.action["Q"], 0, 400)
        else:
            if set(self.observed) != {"p", "w", "Q", "D"} or set(self.action) != {"orders"}:
                raise ValueError("orders episode requires observed p,w,Q,D and whole orders only")
            for field, value, low, high in (("p", self.observed["p"], 51, 100),
                    ("w", self.observed["w"], 10, 50), ("Q", self.observed["Q"], 0, 400),
                    ("D", self.observed["D"], 0, 200)):
                _validate_number(field, value, low, high, allowed={10, 50} if field == "w" else None)
            _validate_orders(self.action["orders"], self.observed["Q"])
        object.__setattr__(self, "observed", _freeze(self.observed))
        object.__setattr__(self, "action", _freeze(self.action))


@dataclass(frozen=True)
class Target:
    request_id: str
    agent_id: str
    participant_id: str
    scope: str
    treatment: str
    target_round: int
    phase: str
    logical_decision_key: str

    def __post_init__(self) -> None:
        if any(not isinstance(x, str) or not x for x in (self.request_id, self.agent_id,
                self.participant_id, self.scope, self.logical_decision_key)):
            raise ValueError("target identifiers and logical key are required")
        if self.treatment not in _TREATMENTS:
            raise ValueError("unknown target treatment")
        if type(self.target_round) is not int or not 1 <= self.target_round <= 15:
            raise ValueError("target_round must be an integer from 1 through 15")
        if self.phase not in _PHASES:
            raise ValueError("unsupported target phase")


def _validate_number(field: str, value: Any, low: int, high: int, *, allowed: set[int] | None = None) -> None:
    if type(value) is not int or not low <= value <= high or (allowed is not None and value not in allowed):
        raise ValueError(f"{field} must be an integer in its documented range")


def _validate_orders(orders: Any, q: int) -> None:
    if not isinstance(orders, (list, tuple)) or len(orders) > 5:
        raise ValueError("whole portfolio must be a list of at most five orders")
    sell_total = 0
    for order in orders:
        if not isinstance(order, Mapping) or set(order) != {"type", "amount", "price"}:
            raise ValueError("order triples must contain exactly type, amount, price")
        if order["type"] not in {"BUY", "SELL"}:
            raise ValueError("order type must be BUY or SELL")
        for key in ("amount", "price"):
            v = order[key]
            if type(v) is not int or not 1 <= v <= 999:
                raise ValueError("order amount and price must be integer 1..999")
        if order["type"] == "SELL":
            sell_total += order["amount"]
    if sell_total > q:
        raise ValueError("historical sell portfolio exceeds initial procurement sell cap")


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value


class ParticipantConditioner:
    def __init__(self, source: SourceManifest, episodes: Sequence[Episode], *, max_episodes: int = 3,
                 max_context_bytes: int = 4096) -> None:
        if type(max_episodes) is not int or not 1 <= max_episodes <= 3:
            raise ValueError("max_episodes must be 1..3")
        if type(max_context_bytes) is not int or max_context_bytes < 128:
            raise ValueError("max_context_bytes must be at least 128")
        seen = set()
        for ep in episodes:
            key = (ep.participant_id, ep.scope, ep.treatment, ep.round, ep.phase)
            if key in seen:
                raise ValueError("duplicate participant-round-phase episode")
            seen.add(key)
        self.source, self.episodes = source, tuple(episodes)
        self.max_episodes, self.max_context_bytes = max_episodes, max_context_bytes
        self._agent_bindings: dict[str, tuple[str, str, str]] = {}
        self._frozen: dict[tuple[str, str], tuple[Any, ...]] = {}

    def prepare(self, target: Target, *, provider: bool = False) -> dict[str, Any]:
        if self.source.source_kind == "human":
            if not self.source.codebook_verified or not self.source.intended_use_authorized:
                raise PermissionError("human history requires codebook and intended-use attestation")
            if provider and not self.source.provider_transmission_permitted:
                raise PermissionError("provider transmission requires explicit attestation")
        binding = (target.participant_id, target.scope, target.treatment)
        previous = self._agent_bindings.setdefault(target.agent_id, binding)
        if previous != binding:
            raise ValueError("agent participant/scope/treatment binding changed")
        matching_source = any(e.participant_id == target.participant_id and e.scope == target.scope
            and e.treatment == target.treatment and e.phase == target.phase for e in self.episodes)
        if not matching_source:
            raise ValueError("target has no source participant/scope/treatment/phase binding")
        selected = sorted((e for e in self.episodes if e.participant_id == target.participant_id
            and e.scope == target.scope and e.treatment == target.treatment and e.phase == target.phase
            and e.round < target.target_round), key=lambda e: e.round)[-self.max_episodes:]
        rows = [{"history_ordinal": i, "observed": _plain(e.observed), "action": _plain(e.action)}
                for i, e in enumerate(selected, start=1)]
        payload = {"schema": "participant_history_v1", "source_kind": self.source.source_kind,
            "treatment": target.treatment, "phase": target.phase, "episodes": rows}
        encoded = canonical_json(payload).encode("utf-8")
        if len(encoded) > self.max_context_bytes:
            raise ValueError("participant history exceeds context byte limit")
        digest = hashlib.sha256(encoded).hexdigest()
        frozen_key = (target.agent_id, target.logical_decision_key)
        fingerprint = (digest, target.participant_id, target.scope, target.treatment,
                       target.target_round, target.phase)
        if frozen_key in self._frozen and self._frozen[frozen_key] != fingerprint:
            raise ValueError("logical decision context is frozen and cannot change")
        self._frozen[frozen_key] = fingerprint
        return {**payload, "context_sha256": digest, "episode_count": len(rows)}


def export_context(conditioner: ParticipantConditioner, target: Target) -> dict[str, Any]:
    return conditioner.prepare(target, provider=True)

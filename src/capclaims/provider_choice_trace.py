"""Additive trace codec that follows provider selections instead of recomputing argmax."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from capclaims.portfolio_prediction import (_validate_base, _new_trace_state, _current,
    _binding, _apply_choice)
from capclaims.provider_choice import validate_provider_choice
from capclaims.engine.protocol import DecisionReply, validate_reply


def _decode(base: Mapping[str, Any], trace: Sequence[Mapping[str, Any]]):
    checked = _validate_base(base)
    state = _new_trace_state()
    for event in trace:
        current = _current(checked, state)
        if current["kind"] in {"complete", "incomplete"}:
            raise ValueError("trace contains events after terminal or failed state")
        request = current["request"]
        if event.get("binding") != _binding(request):
            raise ValueError("trace request binding differs from reconstructed state")
        if current["kind"] == "forced":
            candidate = current["candidate"]
            if (event.get("kind") != "forced" or event.get("origin") != "constraint"
                    or event.get("candidate_id") != candidate.candidate_id
                    or event.get("value") != candidate.value):
                raise ValueError("trace omitted or changed a singleton constraint")
            _apply_choice(checked, state, request, candidate.value)
            continue
        if event.get("kind") == "failure":
            if event.get("reason") not in {"provider_failure", "invalid_reply"}:
                raise ValueError("unknown trace failure")
            state["failure_reason"] = event["reason"]
            continue
        if event.get("kind") != "choice" or event.get("origin") != "provider":
            raise ValueError("trace requires provider selection for the current request")
        answer = {"type": "choice", "choice": event.get("choice"), "confidence": event.get("confidence"),
                  "probabilities": event.get("raw_probabilities")}
        parsed = validate_provider_choice(answer, [c.candidate_id for c in request.candidates])
        if event.get("candidate_id") != parsed["choice"]:
            raise ValueError("recorded candidate differs from raw provider selection")
        raw_answer = event.get("raw_answer")
        if (not isinstance(raw_answer, Mapping) or raw_answer.get("type") != "choice"
                or raw_answer.get("choice") != answer["choice"]
                or raw_answer.get("confidence") != answer["confidence"]
                or raw_answer.get("probabilities") != answer["probabilities"]):
            raise ValueError("saved raw provider answer differs from its separate raw evidence fields")
        required_derived = {
            "normalized_probabilities": parsed["normalized_probabilities"],
            "near_top_candidate_ids": parsed["near_top_candidate_ids"],
            "legacy_argmax_candidate_id": parsed["legacy_argmax_candidate_id"],
            "legacy_argmax_status": parsed["legacy_argmax_status"],
            "legacy_argmax_agrees": parsed["legacy_argmax_agrees"],
        }
        if any(event.get(field) != expected for field, expected in required_derived.items()):
            raise ValueError("recorded normalized probabilities differ from raw provider scores")
        candidate = next(c for c in request.candidates if c.candidate_id == parsed["choice"])
        if event.get("value") != candidate.value:
            raise ValueError("recorded value differs from selected candidate")
        _apply_choice(checked, state, request, candidate.value)
    return checked, state, _current(checked, state)


def next_provider_trace_action(base: Mapping[str, Any], trace: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    _, state, current = _decode(base, trace)
    if current["kind"] in {"request", "forced"}:
        result = {"kind": current["kind"], "binding": _binding(current["request"])}
        if current["kind"] == "forced": result["value"] = current["candidate"].value
        return result
    return {"kind": current["kind"], "reason": current.get("reason"),
            "completed_order_n": len(state["orders"])}


def build_provider_order_request(base: Mapping[str, Any], trace: Sequence[Mapping[str, Any]]):
    _, _, current = _decode(base, trace)
    return current["request"] if current["kind"] == "request" else None


def advance_provider_trace(base: Mapping[str, Any], trace: Sequence[Mapping[str, Any]],
                           raw_answer: Mapping[str, Any] | None,
                           reply: DecisionReply | None = None) -> tuple[dict[str, Any], ...]:
    checked, state, current = _decode(base, trace)
    if current["kind"] in {"complete", "incomplete"}:
        raise ValueError("cannot advance complete or failed provider trace")
    request = current["request"]
    binding = _binding(request)
    if current["kind"] == "forced":
        if raw_answer is not None or reply is not None:
            raise ValueError("forced singleton does not issue a provider call")
        candidate = current["candidate"]
        event = {"kind": "forced", "origin": "constraint", "binding": binding,
                 "candidate_id": candidate.candidate_id, "value": candidate.value}
    elif raw_answer is None or reply is None:
        event = {"kind": "failure", "reason": "provider_failure", "origin": "provider", "binding": binding}
    else:
        validate_reply(request, reply)
        parsed = validate_provider_choice(raw_answer, [c.candidate_id for c in request.candidates])
        if {k: float(v) for k, v in reply.raw_scores.items()} != parsed["raw_probabilities"]:
            raise ValueError("backend response probabilities differ from request-bound raw provider response")
        candidate = next(c for c in request.candidates if c.candidate_id == parsed["choice"])
        event = {"kind": "choice", "origin": "provider", "binding": binding,
            "candidate_id": candidate.candidate_id, "value": candidate.value,
            "choice": parsed["choice"], "confidence": parsed["confidence"],
            "raw_probabilities": parsed["raw_probabilities"],
            "normalized_probabilities": parsed["normalized_probabilities"],
            "near_top_candidate_ids": parsed["near_top_candidate_ids"],
            "legacy_argmax_candidate_id": parsed["legacy_argmax_candidate_id"],
            "legacy_argmax_status": parsed["legacy_argmax_status"],
            "legacy_argmax_agrees": parsed["legacy_argmax_agrees"],
            "backend": reply.backend, "model_revision": reply.model_revision,
            "request_id": request.request_id, "raw_answer": dict(raw_answer)}
    return tuple([*trace, event])


def replay_provider_trace(base: Mapping[str, Any], events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    _, state, current = _decode(base, events)
    complete = current["kind"] == "complete"
    return {"status": "complete" if complete else "incomplete",
        "reason": state["failure_reason"] or (None if complete else "truncated_trace"),
        "orders": tuple(state["orders"]) if complete else (),
        "completed_order_n": len(state["orders"]),
        "terminal_reason": "stop_or_five_orders" if complete else None}

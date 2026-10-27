"""Client-decoded trace codec over the frozen pure portfolio state machine."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from capclaims import portfolio_prediction as state
from capclaims.prompt_sampling_decoder import decode, validate_distribution
from capclaims.engine.protocol import validate_reply


def _reconstruct(base: Mapping[str, Any], trace: Sequence[Mapping[str, Any]], *,
                 private_person: str | None = None, repeat: int | None = None,
                 seed: int = 20261026) -> tuple[dict[str, Any], dict[str, Any]]:
    checked = state._validate_base(base)
    current_state = state._new_trace_state()
    for event in trace:
        current = state._current(checked, current_state)
        if current["kind"] in {"complete", "incomplete"}:
            raise ValueError("client trace contains events after terminal state")
        request = current["request"]
        binding = state._binding(request)
        if event.get("binding") != binding:
            raise ValueError("client trace request binding differs from reconstructed state")
        if current["kind"] == "forced":
            candidate = current["candidate"]
            expected = {"kind": "forced", "origin": "constraint", "binding": binding,
                "candidate_id": candidate.candidate_id, "value": candidate.value}
            if event != expected:
                raise ValueError("forced transition differs from reconstructed state")
            state._apply_choice(checked, current_state, request, candidate.value)
            continue
        if event.get("kind") != "choice" or event.get("origin") != "client":
            raise ValueError("trace requires a client-decoded choice")
        order = [candidate.candidate_id for candidate in request.candidates]
        if event.get("candidate_order") != order:
            raise ValueError("client trace candidate order differs from reconstructed request")
        if event.get("seed") != seed or (private_person is not None and event.get("private_person") != private_person) \
                or (repeat is not None and event.get("repeat") != repeat):
            raise ValueError("client trace random draw coordinates differ from frozen session")
        raw = event.get("raw_probabilities")
        normalized, normalization = validate_distribution(raw, order)
        decoded = decode(raw, order, event.get("decoder"), seed=event.get("seed", 20261026),
            private_person=event.get("private_person", "synthetic"), repeat=event.get("repeat", 0),
            position=request.stage)
        candidate = next(candidate for candidate in request.candidates if candidate.candidate_id == decoded["selected"])
        service_choice = event.get("raw_choice")
        confidence = event.get("service_confidence")
        if service_choice not in order or isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise ValueError("client trace has invalid provider choice or confidence")
        import math
        if not math.isfinite(float(confidence)) or not 0 <= confidence <= 1:
            raise ValueError("client trace has invalid provider confidence")
        anomaly = raw[service_choice] < max(raw.values())
        tie_disagreement = (service_choice != next(key for key in order if raw[key] == max(raw.values()))
            and raw[service_choice] == max(raw.values()))
        expected = {"kind": "choice", "origin": "client", "binding": binding,
            "candidate_id": candidate.candidate_id, "value": candidate.value,
            "candidate_order": order, **decoded, "raw_choice": service_choice,
            "service_confidence": confidence, "nonmax_provider_choice": anomaly,
            "service_choice_tie_order_disagreement": tie_disagreement,
            "request_id": request.request_id, "seed": event.get("seed", 20261026),
            "private_person": event.get("private_person", "synthetic"), "repeat": event.get("repeat", 0),
            "normalization": normalization}
        if event != expected:
            raise ValueError("client trace contains an altered or inconsistent derived field")
        state._apply_choice(checked, current_state, request, candidate.value)
    return current_state, state._current(checked, current_state)


def next_action(base: Mapping[str, Any], trace: Sequence[Mapping[str, Any]], *,
                person: str | None = None, repeat: int | None = None,
                seed: int = 20261026) -> dict[str, Any]:
    person = person if person is not None else (trace[0].get("private_person") if trace else None)
    repetition = repeat if repeat is not None else (trace[0].get("repeat") if trace else None)
    current_state, current = _reconstruct(base, trace, private_person=person, repeat=repetition, seed=seed)
    if current["kind"] in {"request", "forced"}:
        result = {"kind": current["kind"], "binding": state._binding(current["request"])}
        if current["kind"] == "forced":
            result["value"] = current["candidate"].value
        return result
    return {"kind": current["kind"], "reason": current.get("reason"),
        "completed_order_n": len(current_state["orders"])}


def build_request(base: Mapping[str, Any], trace: Sequence[Mapping[str, Any]], *,
                  person: str | None = None, repeat: int | None = None) -> Any:
    person = person if person is not None else (trace[0].get("private_person") if trace else None)
    repetition = repeat if repeat is not None else (trace[0].get("repeat") if trace else None)
    _, current = _reconstruct(base, trace, private_person=person, repeat=repetition)
    return current["request"] if current["kind"] == "request" else None


def advance(base: Mapping[str, Any], trace: Sequence[Mapping[str, Any]], reply: Any = None,
            *, probabilities: Mapping[str, Any] | None = None, decoder: str | None = None,
            person: str = "synthetic", repeat: int = 0, seed: int = 20261026) -> tuple[dict[str, Any], ...]:
    action = next_action(base, trace, person=person, repeat=repeat, seed=seed)
    if action["kind"] == "forced":
        # Reconstruct once to bind the forced candidate at the exact current prefix.
        checked = state._validate_base(base)
        _, current = _reconstruct(base, trace, private_person=person if trace else None,
            repeat=repeat if trace else None, seed=seed)
        if current["kind"] != "forced":
            raise ValueError("current step is not a forced transition")
        c = current["candidate"]
        event = {"kind": "forced", "origin": "constraint", "binding": state._binding(current["request"]),
            "candidate_id": c.candidate_id, "value": c.value}
        return tuple([*trace, event])
    if action["kind"] != "request":
        raise ValueError("cannot advance terminal prompt-sampling trace")
    request = build_request(base, trace, person=person, repeat=repeat)
    if reply is None or probabilities is None or decoder is None:
        raise ValueError("a validated provider reply and decoder are required")
    validate_reply(request, reply)
    usage = getattr(reply, "usage", {})
    service_choice = usage.get("service_choice")
    confidence = usage.get("service_confidence")
    if service_choice not in [candidate.candidate_id for candidate in request.candidates]:
        raise ValueError("provider service choice is absent or illegal")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        raise ValueError("provider confidence must be finite and between zero and one")
    if not __import__("math").isfinite(float(confidence)):
        raise ValueError("provider confidence must be finite and between zero and one")
    candidate_order = [candidate.candidate_id for candidate in request.candidates]
    selected = decode(probabilities, candidate_order, decoder, seed=seed, private_person=person,
        repeat=repeat, position=request.stage)
    candidate = next(item for item in request.candidates if item.candidate_id == selected["selected"])
    event = {"kind": "choice", "origin": "client", "binding": state._binding(request),
        "candidate_id": candidate.candidate_id, "value": candidate.value,
        "candidate_order": candidate_order, **selected,
        "raw_choice": service_choice,
        "service_confidence": confidence,
        "nonmax_provider_choice": probabilities[service_choice] < max(float(v) for v in probabilities.values()),
        "service_choice_tie_order_disagreement": (service_choice != next(key for key in candidate_order
            if float(probabilities[key]) == max(float(v) for v in probabilities.values()))
            and float(probabilities[service_choice]) == max(float(v) for v in probabilities.values())),
        "request_id": request.request_id, "seed": seed, "private_person": person,
        "repeat": repeat, "normalization": validate_distribution(probabilities, candidate_order)[1]}
    # The client decoder acts on the raw wire probabilities; they must normalize
    # to exactly the same legal distribution accepted by the pinned backend.
    import math
    if any(abs(selected["normalized_probabilities"][key] - reply.probabilities[key]) > 1e-6
            for key in candidate_order):
        raise ValueError("raw provider probabilities differ from accepted SDK distribution")
    return tuple([*trace, event])


def replay(base: Mapping[str, Any], trace: Sequence[Mapping[str, Any]], *, seed: int = 20261026,
           person: str = "synthetic", repeat: int = 0) -> dict[str, Any]:
    current_state, current = _reconstruct(base, trace, private_person=person, repeat=repeat, seed=seed)
    complete = current["kind"] == "complete"
    return {"status": "complete" if complete else "incomplete", "reason": None if complete else current.get("reason", "truncated_trace"),
        "orders": tuple(current_state["orders"]) if complete else (),
        "completed_order_n": len(current_state["orders"]),
        "terminal_reason": "stop_or_five_orders" if complete else None}

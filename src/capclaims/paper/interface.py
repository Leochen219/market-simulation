"""Versioned numeric choice labels for the paper decision interface."""
from __future__ import annotations

from capclaims.domain.types import Candidate

INTERFACES = {"legacy_v1", "economic_v2"}


def validate_interface(value: str) -> str:
    if value not in INTERFACES:
        raise ValueError(f"decision_interface must be one of {', '.join(sorted(INTERFACES))}")
    return value


def numeric_observation(base: dict, prefix: str, minimum: int, maximum: int,
                        suffix: str) -> dict:
    """Add decimal-prefix metadata without changing the legacy observation shape."""
    result = dict(base, numeric_prefix=prefix, choice_minimum=minimum,
                  choice_maximum=maximum, decimal_place=suffix)
    if result.get("decision_interface", "legacy_v1") == "legacy_v1":
        return result
    validate_interface(result["decision_interface"])
    names = {"h": "hundreds", "t": "tens", "u": "ones"}
    if suffix not in names:
        raise ValueError("decimal place must be h, t, or u")
    field = result.get("decision_field")
    value_names = {
        "purchase": ("procurement quantity", "units"),
        "quantity": ("order quantity", "units"),
        "price": ("limit price", "dollars per unit"),
    }
    if field not in value_names:
        raise ValueError("economic_v2 numeric observation requires a known decision_field")
    name, units = value_names[field]
    result.update({"decimal_place_name": names[suffix], "numeric_value_name": name,
        "numeric_units": units})
    return result


def build_candidates(values, observation: dict) -> tuple[Candidate, ...]:
    """Build exact legacy candidates or explicit economic branch labels."""
    interface = validate_interface(observation.get("decision_interface", "legacy_v1"))
    if interface == "legacy_v1":
        return tuple(Candidate(f"c{i}", value, str(value)) for i, value in enumerate(values))

    if observation.get("decision_field") == "side":
        labels = {"buy": "submit a buy limit order", "sell": "submit a sell limit order",
                  "STOP": "submit no more orders this round"}
        result = []
        for value in values:
            if value not in labels:
                raise ValueError("economic_v2 side candidates must be buy, sell, or STOP")
            result.append(Candidate(f"action-{value}", value, f"{value}: {labels[value]}"))
        return tuple(result)

    prefix = observation.get("numeric_prefix", "")
    minimum, maximum = observation.get("choice_minimum"), observation.get("choice_maximum")
    field = observation.get("decision_field")
    value_names = {"purchase": ("procurement quantity", "units"),
        "quantity": ("order quantity", "units"), "price": ("limit price", "dollars per unit")}
    if (not isinstance(prefix, str) or len(prefix) >= 3 or (prefix and not prefix.isdigit()) or
            isinstance(minimum, bool) or not isinstance(minimum, int) or
            isinstance(maximum, bool) or not isinstance(maximum, int) or minimum > maximum or
            field not in value_names):
        raise ValueError("economic_v2 candidate observation has invalid numeric semantics")
    name, units = value_names[field]
    place = 10 ** (2 - len(prefix))
    start = int(prefix) * 10 ** (3 - len(prefix)) if prefix else 0
    candidates = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 9:
            raise ValueError("numeric candidates must be decimal digits")
        low = max(minimum, start + value * place)
        high = min(maximum, start + value * place + place - 1)
        if low > high:
            raise ValueError("candidate digit has no feasible completion in the declared bounds")
        interval = str(low) if low == high else f"{low}..{high}"
        meaning = f"{low} {units}" if low == high else f"{interval} {units}"
        candidates.append(Candidate(f"digit{value}", value, meaning))
    return tuple(candidates)


def instruction(stage: str, interface_version: str = "legacy_v1") -> str:
    validate_interface(interface_version)
    if not stage.startswith("paper:"):
        raise ValueError("production paper instructions require a paper decision stage")
    if interface_version == "legacy_v1":
        return ("Make the paper-profile decision requested by the observation. Choose exactly one listed digit or action. "
                "Use numeric_prefix, choice_minimum, choice_maximum, phase, and slot to interpret the legal choices. "
                "Do not infer or use other participants' private information.")
    return ("Maximize this-round profit; choose one listed option. Labels give each digit's full feasible final "
        "interval and units; after two prefixes, the final digit is exact. Use three digits; leading zeros count. "
        "Follow the observation's field, bounds, phase, and economics. Answer diagnostic_task if present.")

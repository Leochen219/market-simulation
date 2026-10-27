"""Authorized, local-only source adapter for the original 18 development people.

The return values contain participant behavior and must remain in participant-data.
Only the metadata counts/hashes are suitable for a public progress report.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from capclaims.conditional_behavior.protocol import validate_orders, validated_history_row
from capclaims.conditional_behavior.source import (
    DEFAULT_SOURCE, POLICY, REGISTRIES, _verified_history_profit, resolve_exposure_roster,
)
from capclaims.personalization_xlsx import _inner_xlsx_bytes, load_paper_episodes


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _extract_cases(episodes: list[Any], exposed: set[str],
                   profits: dict[tuple[str, int], int]) -> tuple[list[dict], dict[str, dict]]:
    """Extract only exposed r6..13; target labels never enter returned cases."""
    if len(exposed) != 18:
        raise ValueError("behavioral memory requires exactly 18 original development people")
    indexed: dict[tuple[str, int], dict[str, Any]] = {}
    for episode in episodes:
        if (episode.scope != "GA 10" or episode.treatment != "GA"
                or episode.participant_id not in exposed or episode.round not in range(6, 14)):
            continue
        key = (episode.participant_id, episode.round)
        parts = indexed.setdefault(key, {})
        if episode.phase not in {"procurement", "orders"}:
            raise ValueError("source episode has an unknown decision phase")
        if episode.phase in parts:
            raise ValueError("duplicate source participant-round phase")
        parts[episode.phase] = episode
    cases, labels = [], {}
    for person in sorted(exposed):
        alias = "dev-" + _sha(person.encode("utf-8"))[:24]
        if alias in labels:
            raise ValueError("development alias collision")
        history = []
        current = target = None
        for round_no in range(6, 14):
            parts = indexed.get((person, round_no), {})
            if set(parts) != {"procurement", "orders"}:
                raise ValueError("source is missing a required development round or phase")
            procurement, orders_episode = parts["procurement"], parts["orders"]
            observed = orders_episode.observed
            if any(key not in observed for key in ("p", "w", "Q", "D")):
                raise ValueError("source order state is missing required fields")
            state = {key: observed[key] for key in ("p", "w", "Q", "D")}
            if any(type(v) is not int for v in state.values()):
                raise ValueError("source state must use exact integers")
            if not (1 <= state["p"] <= 999 and state["w"] == 10
                    and 0 <= state["Q"] <= 400 and 0 <= state["D"] <= 200):
                raise ValueError("source is outside the verified GA10 market domain")
            if (procurement.action.get("Q") != state["Q"]
                    or any(procurement.observed.get(key) != state[key] for key in ("p", "w"))):
                raise ValueError("source procurement and order states conflict")
            if "orders" not in orders_episode.action:
                raise ValueError("source is missing observed submissions")
            submitted = [{"side": row["type"].lower(), "qty": row["amount"], "price": row["price"]}
                         for row in orders_episode.action["orders"]]
            submitted = list(validate_orders(submitted, initial_q=state["Q"]))
            profit = profits.get((person, round_no))
            if type(profit) is not int:
                raise ValueError("verified final profit is missing or invalid")
            if round_no < 13:
                history.append(validated_history_row({"round": round_no,
                    **{key: state[key] for key in ("p", "Q", "D")},
                    "orders": submitted, "profit_cents": profit}, allow_profit=True))
            else:
                current = {"round": round_no, **state}
                target = {"round": round_no, "orders": submitted, "profit_cents": profit}
        cases.append({"person_id": alias, "history": history, "current_state": current})
        labels[alias] = target
    return cases, labels


def build_authorized_cases(source_path: str | Path = DEFAULT_SOURCE, *,
                           policy_path: str | Path = POLICY,
                           registry_paths: dict[str, Path] | None = None
                           ) -> tuple[list[dict], dict[str, dict], dict[str, Any]]:
    """Validate exact source permission before parsing; return private cases/labels."""
    source_path = Path(source_path)
    policy_path = Path(policy_path)
    policy_bytes = policy_path.read_bytes()
    policy = json.loads(policy_bytes.decode("utf-8"))
    archive_sha = _sha(source_path.read_bytes())
    workbook_sha = _sha(_inner_xlsx_bytes(source_path))
    if (policy.get("source_sha256") != workbook_sha or not all(policy.get(field) is True
            for field in ("codebook_verified", "intended_use_authorized", "provider_transmission_permitted"))):
        raise ValueError("source policy does not authorize the exact workbook and provider use")
    manifest, episodes = load_paper_episodes(source_path, codebook_verified=True,
        intended_use_authorized=True, provider_transmission_permitted=True)
    if manifest.source_sha256 != workbook_sha:
        raise ValueError("parsed workbook differs from authorized source")
    roster = sorted({episode.participant_id for episode in episodes
                     if episode.scope == "GA 10" and episode.treatment == "GA"})
    if len(roster) != 35:
        raise ValueError("GA10 roster no longer matches the verified 35-person market")
    paths = registry_paths or REGISTRIES
    raw = {name: Path(path).read_bytes() for name, path in paths.items()}
    target_used, donor_used = resolve_exposure_roster(roster,
        {name: json.loads(data.decode("utf-8")) for name, data in raw.items()})
    exposed = target_used | donor_used
    profits = _verified_history_profit(source_path, exposed)
    cases, labels = _extract_cases(episodes, exposed, profits)
    registry_sha = {name: _sha(data) for name, data in raw.items()}
    metadata = {"source_sha256": workbook_sha, "source_archive_sha256": archive_sha,
        "source_policy_sha256": _sha(policy_bytes),
        "exclusion_manifest_sha256": registry_sha, "exposure_registry_sha256": registry_sha,
        "exposed_n": len(cases), "history_n": sum(len(case["history"]) for case in cases),
        "source_policy_verified": True,
        "source_paths": {"archive": str(source_path.resolve()), "policy": str(policy_path.resolve()),
            "exposure_registries": {name: str(Path(path).resolve()) for name, path in paths.items()}},
        "source_scope": "GA 10", "target_round": 13, "history_rounds": list(range(6, 13))}
    return cases, labels, metadata

"""Authorized local source ingestion for the frozen GA10 conditional study."""
from __future__ import annotations

import hashlib
import json
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from capclaims.personalization_xlsx import _sheet_rows, _workbook, load_paper_episodes

from .study import build_plan, select_holdout

ROOT = Path(__file__).resolve().parents[3]
PARTICIPANT_DATA = ROOT / "artifacts" / "participant-data"
DEFAULT_SOURCE = Path(r"C:\Users\hongy\Downloads\mnsc.2023.03771.zip")
POLICY = PARTICIPANT_DATA / "quote-perspective-24" / "source-policy-authorized.json"
REGISTRIES = {
    "procurement": PARTICIPANT_DATA / "procurement-preview-24" / "frozen-plan.json",
    "portfolio": PARTICIPANT_DATA / "portfolio-preview-r2-root-24" / "frozen-plan.json",
    "injection": PARTICIPANT_DATA / "injection-preview-root-25-r0" / "manifest.json",
}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _resolve_hash_values(values: Any, roster_hashes: dict[str, str], *, expected_n: int,
                         allow_raw: bool = False) -> set[str]:
    if not isinstance(values, list) or len(values) != expected_n or len(set(values)) != expected_n:
        raise ValueError("prior registry has an invalid participant count")
    result = set()
    for value in values:
        if value in roster_hashes:
            result.add(roster_hashes[value])
        elif allow_raw and value in set(roster_hashes.values()):
            result.add(value)
        else:
            raise ValueError("prior registry participant does not resolve against source roster")
    if len(result) != expected_n:
        raise ValueError("prior registry does not resolve uniquely")
    return result


def resolve_exposure_roster(roster: list[str], registries: dict[str, dict[str, Any]]) -> tuple[set[str], set[str]]:
    """Resolve three exact-byte manifests to the 12 target-used and 6 donor-used cohorts."""
    if set(registries) != {"procurement", "portfolio", "injection"}:
        raise ValueError("all three exposure manifests are required")
    roster_hashes = {_sha(person.encode()): person for person in roster}
    procurement = _resolve_hash_values(registries["procurement"].get("selected_participants"),
        roster_hashes, expected_n=3, allow_raw=True)
    portfolio = _resolve_hash_values(registries["portfolio"].get("selected_participant_hashes"),
        roster_hashes, expected_n=3)
    injection = registries["injection"]
    injected_targets = _resolve_hash_values(injection.get("selected_people"), roster_hashes,
        expected_n=6, allow_raw=True)
    injection_exclusions = _resolve_hash_values(injection.get("exclusions"), roster_hashes,
        expected_n=6, allow_raw=True)
    if injection_exclusions != procurement | portfolio:
        raise ValueError("injection exclusions do not match the exact prior target-study union")
    donor_rows = injection.get("donor_matches")
    if not isinstance(donor_rows, list) or len(donor_rows) != 6:
        raise ValueError("injection donor-history registry must identify six participants")
    donor_values = [row.get("donor_id") for row in donor_rows]
    injected_donors = _resolve_hash_values(donor_values, roster_hashes, expected_n=6, allow_raw=True)
    target_used = procurement | portfolio | injected_targets
    if len(target_used) != 12 or len(target_used & injected_donors):
        raise ValueError("prior target and donor exposure cohorts overlap or have wrong size")
    if len(target_used | injected_donors) != 18:
        raise ValueError("strict exposure union must contain exactly 18 participants")
    return target_used, injected_donors


def _money_cents(value: Any, field: str) -> int:
    try:
        amount = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"source {field} is not a decimal amount") from exc
    cents = amount * 100
    if not amount.is_finite() or cents != cents.to_integral_value():
        raise ValueError(f"source {field} is not exact to cents")
    return int(cents)


def _verified_history_profit(path: str | Path, people: set[str]) -> dict[tuple[str, int], int]:
    """Read only permitted accounting columns for prior GA10 histories, rounds 6..13."""
    archive, shared, members = _workbook(path)
    try:
        rows = _sheet_rows(archive, members["GA 10"], shared)
    finally:
        archive.close()
    if not rows:
        raise ValueError("GA 10 worksheet is empty")
    headers = [str(value) for value in rows[0]]
    needed = ("Player", "Round", "Sales Price", "Order", "Actual Demand", "Sales (#)",
              "Profit Projection ()", "Sales New (#)", "Trade Cost", "Post-trade profit ()")
    if any(headers.count(name) != 1 for name in needed):
        raise ValueError("GA10 source lacks unique cached accounting columns")
    index = {name: headers.index(name) for name in needed}
    result = {}
    for row in rows[1:]:
        person = str(row[index["Player"]]).strip()
        round_no = int(row[index["Round"]])
        if person not in people or round_no not in range(6, 14):
            continue
        def num(field: str) -> int:
            value = row[index[field]]
            try:
                parsed = Decimal(str(value))
                if not parsed.is_finite() or parsed != parsed.to_integral_value():
                    raise ValueError
                return int(parsed)
            except (InvalidOperation, ValueError) as exc:
                raise ValueError(f"cached exposed history field {field} is missing or noninteger") from exc
        p, q, d = num("Sales Price"), num("Order"), num("Actual Demand")
        sales, projection = num("Sales (#)"), num("Profit Projection ()")
        sales_new, trade_cost = num("Sales New (#)"), _money_cents(row[index["Trade Cost"]], "Trade Cost")
        final_profit = _money_cents(row[index["Post-trade profit ()"]], "Post-trade profit")
        if projection != p * min(d, q) - 10 * q:
            raise ValueError("exposed history Profit Projection fails the procurement accounting identity")
        if sales != min(d, q) or final_profit != (p * sales_new - 10 * q) * 100 + trade_cost:
            raise ValueError("exposed history Post-trade profit fails the verified accounting identity")
        key = (person, round_no)
        if key in result:
            raise ValueError("duplicate exposed participant-round profit")
        result[key] = final_profit
    if len(result) != len(people) * 8:
        raise ValueError("training history profits do not cover all 35 participants and eight rounds")
    return result


def build_authorized_plan(source_path: str | Path = DEFAULT_SOURCE, *,
                          registry_paths: dict[str, Path] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load source and all exclusion registries locally; never emit participant identifiers."""
    source_path = Path(source_path)
    policy_bytes = POLICY.read_bytes()
    policy = json.loads(policy_bytes.decode("utf-8"))
    source_bytes_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    from capclaims.personalization_xlsx import _inner_xlsx_bytes
    workbook_sha = hashlib.sha256(_inner_xlsx_bytes(source_path)).hexdigest()
    if policy.get("source_sha256") != workbook_sha or not all(policy.get(field) is True for field in
            ("codebook_verified", "intended_use_authorized", "provider_transmission_permitted")):
        raise ValueError("source-policy attestation does not authorize the exact workbook")
    source_manifest, episodes = load_paper_episodes(source_path, codebook_verified=True,
        intended_use_authorized=True, provider_transmission_permitted=True)
    ga = [episode for episode in episodes if episode.scope == "GA 10" and episode.treatment == "GA"]
    roster = sorted({episode.participant_id for episode in ga})
    paths = registry_paths or REGISTRIES
    raw_registries = {name: path.read_bytes() for name, path in paths.items()}
    registries = {name: json.loads(raw.decode("utf-8")) for name, raw in raw_registries.items()}
    target_used, donor_used = resolve_exposure_roster(roster, registries)
    holdout = select_holdout(roster, target_used, donor_used)
    exposed = target_used | donor_used
    profits = _verified_history_profit(source_path, set(roster))
    indexed: dict[tuple[str, int], dict[str, Any]] = {}
    for episode in ga:
        row = indexed.setdefault((episode.participant_id, episode.round), {})
        if episode.phase == "procurement":
            row.update(episode.observed)
            row["Q"] = episode.action["Q"]
        else:
            row.update(episode.observed)
            row["orders"] = [{"side": order["type"].lower(), "qty": order["amount"],
                "price": order["price"]} for order in episode.action["orders"]]
    histories = {}
    for person in roster:
        values = []
        for round_no in range(1, 14):
            source = indexed.get((person, round_no))
            if not isinstance(source, dict) or "orders" not in source:
                raise ValueError("GA10 history is missing a formal participant-round row")
            row = {key: source[key] for key in ("p", "w", "Q", "D", "orders")}
            row["round"] = round_no
            if round_no in range(6, 14):
                row["profit_cents"] = profits[(person, round_no)]
            # Profit is only required/selected for the exposed 6..13 donor/baseline pool.
            values.append(row)
        histories[person] = values
    targets = {}
    for person in holdout:
        targets[person] = {}
        for round_no in (14, 15):
            source = indexed.get((person, round_no))
            if not isinstance(source, dict) or "orders" not in source:
                raise ValueError("holdout target state is missing a formal order round")
            targets[person][round_no] = {"round": round_no, **{key: source[key]
                for key in ("p", "w", "Q", "D", "orders")}}
    donors = [{"person_id": person, "history": histories[person][5:13]} for person in sorted(exposed)]
    pool = [{"person_id": person, **row} for person in sorted(exposed) for row in histories[person][5:13]]
    plan = build_plan(targets=targets, own_histories=histories, donors=donors,
        public_history_pool=pool, excluded=exposed)
    metadata = {"source_sha256": source_manifest.source_sha256,
        "source_archive_sha256": source_bytes_hash, "source_policy_sha256": _sha(policy_bytes),
        "exclusion_manifest_sha256": {name: _sha(data) for name, data in raw_registries.items()},
        "exposed_people_n": len(exposed), "holdout_people_n": len(holdout)}
    return plan, metadata

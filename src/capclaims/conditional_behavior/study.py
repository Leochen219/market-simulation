"""Frozen design construction, role exclusion, and sealed label separation."""
from __future__ import annotations

import hashlib
import json
import os
import random
import asyncio
import base64
from collections.abc import Mapping, Sequence
from typing import Any

from . import ARMS, SAMPLES_PER_CONTEXT, TARGET_ROUNDS, VERSION
from .baselines import select_donor
from .protocol import build_observation, validated_history_row, validate_orders

SEED = 20261026
ROOT = __import__("pathlib").Path(__file__).resolve().parents[3]


def _code_hashes() -> dict[str, str]:
    """Bind every new module, reused instruction/codec, and design/fixture source."""
    package = ROOT / "src" / "capclaims" / "conditional_behavior"
    paths = {path.relative_to(ROOT).as_posix(): path for path in package.glob("*.py")}
    for relative in (
        "src/capclaims/backends/jev.py", "src/capclaims/backends/jev_instructions.py",
        "src/capclaims/paper/interface.py", "src/capclaims/paper/codec.py",
        "src/capclaims/order_presentation.py", "src/capclaims/continuation_study.py",
        "src/capclaims/personalization_xlsx.py", "src/capclaims/provider_choice.py",
        "docs/research/jev-conditional-distribution-design-2026-10-26.md",
        "docs/engineering/conditional-behavior-cost-gate.md",
        "tools/probes/jev_conditional_behavior.py",
        "configs/jev_conditional_behavior_synthetic_v1.json",
    ):
        path = ROOT / relative
        if path.exists():
            paths[relative] = path
    if not paths:
        raise ValueError("conditional behavior freeze has no code files")
    return {name: _sha_bytes(path.read_bytes()) for name, path in sorted(paths.items())}


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _write_new(path: Any, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


async def _request_previews(inputs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Serialize every stage along a maximum-length legal prefix for each context.

    This is an offline SDK mock-transport pass. It forces five BUY orders with
    999 quantities/prices, exercising the largest history, ledger, prefix and
    candidate envelopes before any provider key is loaded.
    """
    from .sampling import sample_portfolio
    from .transport import MAX_BODY, serialize_body

    class PreviewClient:
        def __init__(self, sell_first: bool):
            self.requests: list[dict[str, Any]] = []
            self.sell_first = sell_first
            self.saw_first_side = False

        async def choose(self, *, request_id: str, state: Any, instructions: str,
                         criteria: dict[str, str]) -> dict[str, Any]:
            body = await serialize_body(state=state, instructions=instructions, criteria=criteria)
            if len(body) > MAX_BODY:
                raise ValueError("a worst-prefix serialized request exceeds 12,000 bytes")
            decision_field = state.get("decision_field")
            if decision_field == "order_count":
                selected = "N5"
            elif decision_field == "side":
                if self.sell_first and not self.saw_first_side and "SELL" in criteria:
                    selected = "SELL"
                else:
                    selected = "BUY"
                self.saw_first_side = True
            elif decision_field in ("quantity", "price"):
                selected = "digit9" if "digit9" in criteria else sorted(criteria)[-1]
            else:
                selected = next(iter(criteria))
            if selected not in criteria:
                selected = next(iter(criteria))
            self.requests.append({"request_id": request_id,
                "body_sha256": _sha_bytes(body), "body_bytes": len(body),
                "body_base64": base64.b64encode(body).decode("ascii")})
            return {"model": "jev-1.14.0", "usage": {"input_tokens": 1, "output_tokens": 1},
                "answers": {"decision": {"type": "choice", "choice": selected,
                    "confidence": 1.0,
                    "probabilities": {key: float(key == selected) for key in criteria}}}}

    # Four samples in a context share identical initial states. Two paths cover
    # five BUYs and the larger SELL-first partial ledger when SELL is legal.
    contexts: dict[tuple[str, int, str], Mapping[str, Any]] = {}
    for row in inputs:
        contexts.setdefault((row["person_id"], row["target_round"], row["arm"]), row)
    previews: list[dict[str, Any]] = []
    for key in sorted(contexts):
        row = dict(contexts[key])
        row["sample"] = 0
        row["case_id"] = hashlib.sha256((row["case_id"] + "\0preview").encode()).hexdigest()
        variants = (False, True) if row["observation"]["own_q_units"] > 0 else (False,)
        for sell_first in variants:
            client = PreviewClient(sell_first)
            variant_row = dict(row)
            variant_row["case_id"] = hashlib.sha256((row["case_id"] +
                ("\0sell_first" if sell_first else "\0buy_max")).encode()).hexdigest()
            await sample_portfolio(client, variant_row, seed=SEED, category="human")
            previews.extend({**request, "path": "sell_first" if sell_first else "buy_max"}
                for request in client.requests)
    if not previews:
        raise ValueError("no complete request previews were generated")
    return previews


def select_holdout(roster: Sequence[str], target_used: set[str], donor_used: set[str]) -> list[str]:
    roster_set = set(roster)
    if len(roster) != 35 or len(roster_set) != 35 or any(not isinstance(x, str) or not x for x in roster):
        raise ValueError("source must contain 35 unique GA10 participants")
    if not target_used <= roster_set or not donor_used <= roster_set:
        raise ValueError("prior-study exclusions must resolve against the exact source roster")
    if len(target_used) != 12 or len(donor_used) != 6 or len(target_used | donor_used) != 18:
        raise ValueError("prior registries must exclude a strict union of 18 participants")
    return sorted(roster_set - (target_used | donor_used))


def _check_rows(rows: Sequence[Mapping[str, Any]], rounds: range, *, profit: bool) -> list[dict[str, Any]]:
    if not isinstance(rows, (list, tuple)):
        raise ValueError("history must be an ordered sequence")
    checked = [validated_history_row(row, allow_profit=profit) for row in rows]
    if [row["round"] for row in checked] != list(rounds):
        raise ValueError("history must contain every expected formal round in order")
    return checked


def _case_id(person: str, target_round: int, arm: str, sample: int) -> str:
    material = f"{VERSION}\0{person}\0{target_round}\0{arm}\0{sample}".encode()
    return hashlib.sha256(material).hexdigest()


def build_plan(*, targets: Mapping[str, Mapping[int, Mapping[str, Any]]],
               own_histories: Mapping[str, Sequence[Mapping[str, Any]]],
               donors: Sequence[Mapping[str, Any]],
               public_history_pool: Sequence[Mapping[str, Any]],
               excluded: set[str], seed: int = SEED) -> dict[str, Any]:
    """Create 680 inference rows while writing target labels into a separate mapping."""
    people = sorted(targets)
    if (len(people) != 17 or len(excluded) != 18 or len(set(people) & excluded)
            or set(people) - set(own_histories)):
        raise ValueError("17 holdout targets must be disjoint from exclusions and have source histories")
    donor_ids = [row.get("person_id") for row in donors]
    if len(donor_ids) != 18 or len(set(donor_ids)) != 18 or set(donor_ids) != excluded:
        raise ValueError("the donor pool must contain exactly 18 distinct exposed participants")
    checked_own: dict[str, list[dict[str, Any]]] = {}
    for person in people:
        checked_own[person] = _check_rows(own_histories[person], range(1, 14), profit=False)
        # Profit is admitted only for the eight rounds actually shown by C/E.
        checked_own[person][5:13] = [validated_history_row(row, allow_profit=True)
                                     for row in own_histories[person][5:13]]
    checked_donors = []
    for donor in donors:
        history = _check_rows(donor.get("history", []), range(6, 14), profit=True)
        checked_donors.append({"person_id": donor["person_id"], "history": history})
    public_rows = []
    for row in public_history_pool:
        if row.get("person_id") not in set(donor_ids) or row.get("round") not in range(6, 14):
            raise ValueError("baseline pool must contain only the fixed exposed history rounds 6 through 13")
        public_rows.append(validated_history_row(row, allow_profit=False) | {"person_id": row["person_id"]})
    if len(public_rows) != 18 * 8:
        raise ValueError("baseline pool must include all 18 exposed histories for rounds 6 through 13")
    labels: dict[str, Any] = {}
    inputs: list[dict[str, Any]] = []
    for person in people:
        states = targets[person]
        if set(states) != set(TARGET_ROUNDS):
            raise ValueError("each holdout participant must have exactly target rounds 14 and 15")
        checked_states: dict[int, dict[str, Any]] = {}
        for target_round in TARGET_ROUNDS:
            state = dict(states[target_round])
            if state.get("round") != target_round:
                raise ValueError("target round labels must be 14 and 15")
            if any(type(state.get(field)) is not int for field in ("p", "w", "Q", "D")):
                raise ValueError("target state must include integer p,w,Q,D")
            if not (1 <= state["p"] <= 999 and 1 <= state["w"] <= 999 and 0 <= state["Q"] <= 400 and 0 <= state["D"] <= 200):
                raise ValueError("target state is outside economic_v2 bounds")
            state["orders"] = list(validate_orders(state.get("orders", []), initial_q=state["Q"]))
            checked_states[target_round] = state
            key = f"{person}:{target_round}"
            labels[key] = {"person_id": person, "target_round": target_round,
                "Q": state["Q"], "D": state["D"], "orders": state["orders"]}
        # Match prior person histories round by round; one donor serves both targets.
        matching_history = checked_own[person][5:13]
        donor = select_donor(matching_history, checked_donors, seed=seed, target_id=person)
        histories = {
            "A": [],
            "B": checked_own[person][10:13],
            "C": checked_own[person][5:13],
            "D": donor["history"],
            "E": checked_own[person][5:13],
        }
        for target_round in TARGET_ROUNDS:
            state = {key: checked_states[target_round][key] for key in ("round", "p", "w", "Q", "D")}
            for arm in ARMS:
                observation = build_observation(state=state, history=histories[arm], arm=arm)
                for sample in range(SAMPLES_PER_CONTEXT):
                    inputs.append({"case_id": _case_id(person, target_round, arm, sample),
                        "person_id": person, "target_round": target_round, "arm": arm,
                        "sample": sample, "interface": observation["protocol_interface"],
                        "donor_id": donor["person_id"] if arm == "D" else None,
                        "donor_match_distance": donor["distance"] if arm == "D" else None,
                        "observation": observation})
    # Deterministic balanced interleaving; no request completion order enters draws.
    rng = random.Random(seed)
    rng.shuffle(inputs)
    if len(inputs) != 680 or len(labels) != 34:
        raise AssertionError("frozen design arithmetic changed")
    return {"schema": VERSION, "seed": seed, "inputs": inputs, "targets": labels,
        "donor_matches": {person: select_donor(checked_own[person][5:13], checked_donors,
            seed=seed, target_id=person)["person_id"]
            for person in people}, "public_history_pool_n": len(public_rows), "baseline_pool": public_rows}


def build_preview(plan: Mapping[str, Any], output_dir: Any, *, source_sha256: str,
                  exclusion_manifest_sha256: Mapping[str, str],
                  source_policy_sha256: str | None = None,
                  source_archive_sha256: str | None = None,
                  source_policy_verified: bool = False) -> dict[str, Any]:
    """Write a sealed label-free inference plan plus separately hashed outcomes."""
    from pathlib import Path
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError("conditional-behavior previews are immutable")
    hashes = dict(exclusion_manifest_sha256)
    if set(hashes) != {"procurement", "portfolio", "injection"} or any(
            not isinstance(value, str) or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value)
            for value in hashes.values()):
        raise ValueError("exact-byte hashes for all three exclusion registries are required")
    for name, value in (("source", source_sha256), ("source policy", source_policy_sha256 or "a" * 64)):
        if not isinstance(value, str) or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
            raise ValueError(f"{name} SHA256 is invalid")
    if source_archive_sha256 is None:
        source_archive_sha256 = source_sha256
    if not isinstance(source_archive_sha256, str) or len(source_archive_sha256) != 64 or any(
            ch not in "0123456789abcdef" for ch in source_archive_sha256):
        raise ValueError("source archive SHA256 is invalid")
    if source_policy_verified is not True:
        raise ValueError("a separately verified source-policy authorization is required")
    if plan.get("schema") != VERSION or len(plan.get("inputs", [])) != 680 or len(plan.get("targets", {})) != 34:
        raise ValueError("frozen plan does not match the 680-case design")
    # Do this before creating the immutable output directory and before any
    # environment/key loading. The preview binds exact SDK bytes for all stages.
    request_previews = asyncio.run(_request_previews(plan["inputs"]))
    request_previews_bytes = b"".join(_canonical(row) + b"\n" for row in request_previews)
    output.mkdir(parents=True, exist_ok=False)
    inference = {"schema": VERSION, "seed": plan["seed"], "inputs": plan["inputs"]}
    input_bytes = _canonical(inference) + b"\n"
    labels_bytes = _canonical(plan["targets"]) + b"\n"
    private_bytes = _canonical({"donor_matches": plan.get("donor_matches", {}),
        "public_history_pool_n": plan.get("public_history_pool_n"),
        "baseline_pool": plan.get("baseline_pool", [])}) + b"\n"
    _write_new(output / "inputs.json", input_bytes)
    _write_new(output / "scoring-targets.json", labels_bytes)
    _write_new(output / "private-metadata.json", private_bytes)
    _write_new(output / "request-previews.jsonl", request_previews_bytes)
    manifest = {"schema": VERSION, "source_sha256": source_sha256,
        "source_archive_sha256": source_archive_sha256,
        "source_policy_sha256": source_policy_sha256 or "a" * 64,
        "source_policy_verified": source_policy_verified,
        "exclusion_manifest_sha256": hashes,
        "inputs_sha256": _sha_bytes(input_bytes), "scoring_targets_sha256": _sha_bytes(labels_bytes),
        "private_metadata_sha256": _sha_bytes(private_bytes),
        "request_previews_sha256": _sha_bytes(request_previews_bytes),
        "request_previews_n": len(request_previews),
        "request_previews_max_body_bytes": max(row["body_bytes"] for row in request_previews),
        "target_n": 34,
        "people_n": 17, "exposed_people_n": 18, "exclusion_union_n": 18,
        "sample_rows": 680, "samples_per_target_arm": 4,
        "arms": list(ARMS), "target_rounds": list(TARGET_ROUNDS), "seed": plan["seed"],
        "model": "jev-1.14.0", "sdk_version": "0.7.5", "request_cap": 29888,
        "prompt_interventions": [
            "Replaced the source template's same-participant-only history instruction with a condition-specific history instruction so A has no history and D can show a matched other participant.",
            "Replaced the source template's choices-only/no-feedback sentence with the frozen A-E history/profit condition description; C/D/E include only past final profit, B has no profit, and no current target outcomes are exposed.",
            "Clarified that J1 permits STOP while J2 fixes order count first and has no STOP option. All remaining detailed instruction text is reused from the prior portfolio prompt blocks."
        ],
        "real_request_cap": 23936, "synthetic_request_cap": 5952,
        "body_bytes_cap": 12000, "max_in_flight": 10, "rps_cap": 40,
        "bytes_per_second_cap": 50000, "wall_seconds_cap": 600,
        "retry_cap": 0, "resume_cap": 0}
    manifest["code_sha256"] = _code_hashes()
    manifest["preview_sha256"] = _sha_bytes(_canonical(manifest))
    _write_new(output / "manifest.json", _canonical(manifest) + b"\n")
    return {key: manifest[key] for key in ("schema", "preview_sha256", "target_n", "people_n", "sample_rows", "request_cap")}


def verify_preview(preview_dir: Any, reviewed_sha256: str) -> dict[str, Any]:
    from pathlib import Path
    root = Path(preview_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    digest = manifest.pop("preview_sha256", None)
    if digest != reviewed_sha256 or digest != _sha_bytes(_canonical(manifest)):
        raise ValueError("preview does not match the reviewed digest")
    manifest["preview_sha256"] = digest
    if manifest.get("code_sha256") != _code_hashes():
        raise ValueError("conditional behavior frozen code, prompt, candidates, or source schema changed")
    for filename, key in (("inputs.json", "inputs_sha256"),
                          ("scoring-targets.json", "scoring_targets_sha256"),
                          ("private-metadata.json", "private_metadata_sha256"),
                          ("request-previews.jsonl", "request_previews_sha256")):
        if _sha_bytes((root / filename).read_bytes()) != manifest.get(key):
            raise ValueError(f"frozen {filename} digest mismatch")
    inputs = json.loads((root / "inputs.json").read_text(encoding="utf-8"))
    labels = json.loads((root / "scoring-targets.json").read_text(encoding="utf-8"))
    if (inputs.get("schema") != VERSION or len(inputs.get("inputs", [])) != 680
            or len(labels) != 34 or inputs.get("seed") != manifest.get("seed")):
        raise ValueError("frozen design cardinality or schema is invalid")
    ids = [row.get("case_id") for row in inputs["inputs"]]
    if len(set(ids)) != len(ids) or any(not isinstance(item, str) for item in ids):
        raise ValueError("case IDs are missing or duplicated")
    return manifest


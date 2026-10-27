"""Counter-keyed random streams; no mutable process-wide RNG is shared."""

import hashlib
import json
import random
from typing import Any


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def hash_int(*parts: Any) -> int:
    digest = hashlib.sha256(canonical_json(list(parts)).encode("utf-8")).digest()
    return int.from_bytes(digest, "big")


def exogenous_value(master_seed: int, seed_index: int, round_index: int, agent_id: str, variable: str) -> int:
    if variable not in {"p", "D"}:
        raise ValueError("variable must be 'p' or 'D'")
    rng = random.Random(hash_int(master_seed, seed_index, round_index, agent_id, variable))
    return rng.randint(51, 100) if variable == "p" else rng.randint(0, 200)


def rng_for_policy(
    master_seed: int, seed_index: int, policy_id: str, round_index: int,
    agent_id: str, stage: str, purpose: str,
) -> random.Random:
    return random.Random(hash_int(master_seed, seed_index, policy_id, round_index, agent_id, stage, purpose))


def auction_rank(master_seed: int, seed_index: int, round_index: int, agent_id: str) -> int:
    return hash_int(master_seed, seed_index, round_index, agent_id)


def decision_rank(
    master_seed: int, seed_index: int, round_index: int,
    agent_id: str, stage: str, candidate_id: str,
) -> int:
    return hash_int(master_seed, seed_index, round_index, agent_id, stage, candidate_id)

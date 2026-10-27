"""Separate full-domain paper baselines; none reuse legacy candidate grids."""
from __future__ import annotations
import random
from functools import lru_cache
from typing import Sequence

from capclaims.engine.protocol import DecisionReply, DecisionRequest
from capclaims.randomness import hash_int


@lru_cache(maxsize=128)
def exact_discrete_newsvendor_q(p: int, wholesale_price: int) -> int:
    if not 51 <= p <= 100 or wholesale_price not in (10, 50):
        raise ValueError("invalid newsvendor inputs")
    # Appendix B's integer demand support is uniform on 0..200 inclusive.
    profits = [sum(p * min(d, q) - wholesale_price * q for d in range(201)) for q in range(401)]
    return max(range(401), key=lambda q: (profits[q], -q))


class PaperBaselineBackend:
    def __init__(self, policy: str, seed: int = 0) -> None:
        if policy not in {"newsvendor", "random_legal", "noisy20"}:
            raise ValueError("unsupported paper baseline policy")
        self.policy, self.seed, self.closed = policy, seed, False

    def _choice(self, request: DecisionRequest):
        candidates = request.candidates
        obs = request.observation
        rng = random.Random(hash_int(self.seed, self.policy, request.round_index, request.agent_id,
                                     request.stage, "paper-baseline"))
        explore = self.policy == "random_legal" or (self.policy == "noisy20" and rng.random() < 0.2)
        if explore:
            return rng.choice(candidates)
        if request.stage.endswith(":side"):
            return next(c for c in candidates if c.value == "STOP")
        if request.stage.startswith("paper:purchase:"):
            prefix = str(obs.get("numeric_prefix", ""))
            digit_position = len(prefix)
            p = int(obs["own_p"])
            wholesale = int(obs["wholesale_price_dollars"])
            branch_values = {}
            for candidate in candidates:
                branch = prefix + str(candidate.value)
                values = [q for q in range(401) if f"{q:03d}".startswith(branch)]
                if not values:
                    continue
                branch_values[candidate.candidate_id] = max(
                    sum(p * min(d, q) - wholesale * q for d in range(201)) for q in values)
            if branch_values:
                best = max(branch_values.values())
                tied = [c for c in candidates if branch_values.get(c.candidate_id) == best]
                return min(tied, key=lambda c: int(c.value))
        # Deterministic non-exploration baseline action for numeric order fields.
        return candidates[0]

    async def decide_many(self, requests: Sequence[DecisionRequest]) -> list[DecisionReply]:
        if self.closed:
            raise RuntimeError("paper baseline backend is closed")
        replies = []
        for request in requests:
            selected = self._choice(request)
            scores = {candidate.candidate_id: float(candidate == selected) for candidate in request.candidates}
            replies.append(DecisionReply.from_request(request, scores, scores,
                backend=f"paper:{self.policy}", model_revision=f"paper-baseline:{self.policy}",
                usage={"score_kind": "policy_one_hot", "model_probability": False,
                       "full_integer_domain": True}))
        return replies

    async def aclose(self) -> None:
        self.closed = True

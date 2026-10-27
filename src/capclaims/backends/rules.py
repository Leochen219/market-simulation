"""Transparent deterministic and keyed-random local policy backend."""

from capclaims.agents.policies import purchase_policy, trade_policy
from capclaims.config import RunConfig
from capclaims.engine.protocol import DecisionReply, DecisionRequest


class RuleBackend:
    def __init__(self, config: RunConfig) -> None:
        self.config = config
        self.closed = False

    async def decide_many(self, requests: list[DecisionRequest]) -> list[DecisionReply]:
        if self.closed:
            raise RuntimeError("backend is closed")
        replies = []
        for request in requests:
            obs = request.observation
            if request.stage == "purchase":
                value = purchase_policy(self.config, request.round_index, request.agent_id, obs["own_p"])
            else:
                action = trade_policy(
                    self.config, request.round_index, request.agent_id, obs["own_p"],
                    obs["own_q"], obs["own_d"],
                )
                value = {
                    "trade_side": action.side,
                    "trade_qty": action.qty,
                    "trade_price": (action.limit_cents // 100) if action.limit_cents is not None else None,
                }[request.stage]
            matching = [candidate for candidate in request.candidates if candidate.value == value]
            if not matching:
                raise ValueError(f"policy selected illegal {request.stage} value {value!r}")
            scores = {candidate.candidate_id: float(candidate is matching[0]) for candidate in request.candidates}
            replies.append(DecisionReply.from_request(
                request, scores, scores, backend=self.config.policy,
                model_revision=f"policy:{self.config.policy}",
                usage={"score_kind": "policy_one_hot", "model_probability": False},
            ))
        return replies

    async def aclose(self) -> None:
        self.closed = True

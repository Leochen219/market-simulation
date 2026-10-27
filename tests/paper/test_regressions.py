import pytest

from capclaims.paper.auction import PaperAuctionResult, PaperTrade
from capclaims.paper.config import PaperConfig
from capclaims.paper.groups import generate_g3_schedule
from capclaims.paper.settlement import settle_paper_round


def test_self_match_is_economically_zero_and_settlement_matches_net_cashflows():
    trade = PaperTrade("a", "a", 2, 5050, "buy", "sell")
    auction = PaperAuctionResult(5050, (trade,), {"a": 0}, {"buy": 0, "sell": 1}, 2)
    result = settle_paper_round({"a": (80, 0, 10)}, auction, 10)
    assert result.profits_cents["a"] == -10000
    cash_balance = sum((entry.amount if entry.destination == "a" else 0) -
                       (entry.amount if entry.source == "a" else 0)
                       for entry in result.entries)
    assert cash_balance == 0


def test_exclusion_policy_fails_at_configuration_if_policy_is_unimplemented():
    with pytest.raises(ValueError, match="not implemented"):
        PaperConfig(treatment="GA", n_agents=10, self_match_policy="exclude")


def test_three_agent_twenty_round_schedule_is_rejected_as_infeasible():
    with pytest.raises(ValueError, match="schedule infeasible"):
        generate_g3_schedule(("a", "b", "c"), rounds=20, seed=2)

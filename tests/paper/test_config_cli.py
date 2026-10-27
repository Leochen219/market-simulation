import pytest

from capclaims.cli import build_parser
from capclaims.paper.config import PaperConfig
from capclaims.paper.baselines import exact_discrete_newsvendor_q


def test_configs_declare_independent_paper_profile_and_default_warmup_counts():
    config = PaperConfig.load("configs/paper-ga-w10.yaml")
    assert config.profile == "paper_mnsc_2023_03771_v1"
    assert config.total_rounds == 20
    assert config.policy == "laya"
    jev = PaperConfig.load("configs/paper-jev-ga-w10.yaml")
    assert jev.policy == "jev" and jev.instruction_profile == "explicit_v1"
    with pytest.raises(ValueError, match="not implemented"):
        PaperConfig(self_match_policy="exclude")
    with pytest.raises(ValueError, match="integer"):
        PaperConfig(seed_index=True)


def test_paper_cli_is_additive_and_exact_newsvendor_stays_in_full_integer_domain():
    parser = build_parser()
    assert parser.parse_args(["paper-run", "--config", "configs/paper-nv-w10.yaml", "--output", "out"]).command == "paper-run"
    assert parser.parse_args(["paper-replay", "--run", "out"]).command == "paper-replay"
    assert parser.parse_args(["paper-summarize", "--run", "out"]).command == "paper-summarize"
    assert parser.parse_args(["run", "--config", "configs/local-laya.yaml", "--output", "old"]).command == "run"
    assert 0 <= exact_discrete_newsvendor_q(80, 10) <= 200

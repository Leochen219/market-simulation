import pytest

from capclaims.paper.codec import legal_prefix_choices, decode_prefix
from capclaims.paper.config import PaperConfig
from capclaims.paper.orders import PaperOrder, validate_portfolio
from capclaims.paper.auction import clear_paper_auction
from capclaims.paper.groups import generate_g3_schedule


def test_full_integer_domains_are_reachable_by_legal_prefixes():
    assert {decode_prefix((a, b, c)) for a in range(5) for b in range(10) for c in range(10)
            if decode_prefix((a, b, c)) <= 400} == set(range(401))
    assert set(legal_prefix_choices("", 999)) == set(range(10))
    assert set(legal_prefix_choices("13", 400)) == {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
    for maximum in (0, 1, 137, 400, 999):
        reachable = set()
        for h in range(10):
            if h not in legal_prefix_choices("", maximum):
                continue
            for t in range(10):
                if t not in legal_prefix_choices(str(h), maximum):
                    continue
                for u in range(10):
                    if u in legal_prefix_choices(f"{h}{t}", maximum):
                        reachable.add(100 * h + 10 * t + u)
        assert reachable == set(range(maximum + 1))


def test_portfolio_accepts_mixed_sides_but_caps_gross_sells_at_initial_q():
    orders = (
        PaperOrder("o1", "a", "buy", 999, 325),
        PaperOrder("o2", "a", "sell", 60, 325),
        PaperOrder("o3", "a", "sell", 40, 1),
    )
    validate_portfolio(orders, owner="a", initial_q=100)
    with pytest.raises(ValueError, match="gross sell"):
        validate_portfolio(orders + (PaperOrder("o4", "a", "sell", 1, 1),), "a", 100)


def test_auction_orders_have_unique_ids_and_support_half_dollar_mcp():
    orders = (
        PaperOrder("b1", "buyer", "buy", 3, 325),
        PaperOrder("s1", "seller", "sell", 2, 324),
    )
    result = clear_paper_auction(orders, master_seed=9, round_index=0)
    assert result.price_cents == 32450
    assert result.trades[0].qty == 2
    assert result.net_allocations == {"buyer": 2, "seller": -2}


def test_paper_config_rejects_n10_g3_and_divisibility_infeasible_schedule():
    with pytest.raises(ValueError, match="divisible by three"):
        PaperConfig(treatment="G3", n_agents=10)
    with pytest.raises(ValueError, match="schedule infeasible"):
        generate_g3_schedule(tuple(f"a{i}" for i in range(3)), rounds=20, seed=7)


def test_g3_schedule_is_seeded_and_never_repeats_exact_triples_over_twice():
    agents = tuple(f"a{i}" for i in range(6))
    one = generate_g3_schedule(agents, rounds=20, seed=7)
    assert one == generate_g3_schedule(agents, rounds=20, seed=7)
    seen = {}
    for round_groups in one:
        assert sorted(x for group in round_groups for x in group) == sorted(agents)
        for group in round_groups:
            key = tuple(sorted(group))
            seen[key] = seen.get(key, 0) + 1
    assert max(seen.values()) <= 2


@pytest.mark.parametrize("n_agents", [30, 39])
def test_g3_paper_schedules_scale_to_original_population_sizes(n_agents):
    agents = tuple(f"a{i:02d}" for i in range(n_agents))
    schedule = generate_g3_schedule(agents, rounds=20, seed=123)
    counts = {}
    for row in schedule:
        assert len(row) == n_agents // 3
        assert sorted(a for group in row for a in group) == sorted(agents)
        for group in row:
            triple = tuple(sorted(group))
            counts[triple] = counts.get(triple, 0) + 1
    assert max(counts.values()) <= 2

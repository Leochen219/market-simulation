from copy import deepcopy

import pytest

from capclaims.behavioral_memory.memory import compile_behavioral_memory


def episode(round_no, p, q, d, orders, profit=1234):
    return {"round": round_no, "p": p, "Q": q, "D": d,
            "orders": orders, "profit_cents": profit}


def test_compiler_preserves_duplicate_submissions_and_derives_episode_relations():
    history = [episode(6, 100, 8, 10, [
        {"side": "buy", "qty": 3, "price": 90},
        {"side": "buy", "qty": 3, "price": 90},
        {"side": "sell", "qty": 2, "price": 110},
    ])]
    before = deepcopy(history)
    memory, audit = compile_behavioral_memory(history, {"round": 13, "p": 105, "Q": 9, "D": 11}, 12)

    derived = memory["episodes"][0]["derived"]
    assert derived["buy_quantity"] == 6
    assert derived["sell_quantity"] == 2
    assert derived["duplicate_submissions"] == 1
    assert derived["order_count"] == 3
    assert derived["D_minus_Q"] == 2
    assert derived["gap"] == 2
    assert derived["surplus"] == 0
    assert derived["buy_weighted_price"] == "90.00"
    assert derived["buy_weighted_price_minus_p"] == "-10.00"
    assert derived["buy_price_range"] == {"min": 90, "max": 90}
    assert audit["exact_arithmetic"]["round:6"]["buy"]["weighted_price"] == {
        "numerator": 540, "denominator": 6, "missing_reason": None}
    assert audit["field_provenance"]["buy_quantity"]["source_refs"] == ["round:6"]
    assert "540/6" in audit["markdown_zh"]
    assert memory["episodes"][0]["source_ref"] == "round:6"
    assert audit["version"] == "jev_behavioral_memory_v1"
    assert history == before


def test_compiler_is_order_invariant_for_statistics_but_retains_raw_order_list():
    first = [episode(6, 100, 4, 2, [
        {"side": "buy", "qty": 1, "price": 90},
        {"side": "sell", "qty": 2, "price": 110},
    ])]
    second = [episode(6, 100, 4, 2, list(reversed(first[0]["orders"]))) ]
    a, _ = compile_behavioral_memory(first, {"round": 13, "p": 100, "Q": 4, "D": 2}, 12)
    b, _ = compile_behavioral_memory(second, {"round": 13, "p": 100, "Q": 4, "D": 2}, 12)
    assert a["episodes"][0]["derived"] == b["episodes"][0]["derived"]
    assert a["episodes"][0]["raw_orders"] != b["episodes"][0]["raw_orders"]
    assert a["wire_history"][0]["round"] == 11
    assert a["wire_history"][0]["formal_round"] == 6


def test_compiler_reports_relevance_as_exact_fraction_and_retains_all_history():
    history = [episode(r, 100 + r, 100, 100, [], r * 100) for r in range(6, 13)]
    memory, _ = compile_behavioral_memory(history, {"round": 13, "p": 113, "Q": 100, "D": 100}, 12)
    assert len(memory["episodes"]) == 7
    assert memory["relevance"]["closest_source_refs"] == ["round:12", "round:11"]
    rel = memory["episodes"][-1]["derived"]["relevance"]
    assert rel["fraction"] == {"numerator": 400, "denominator": 19600}


def test_relevance_ties_use_exact_fraction_then_more_recent_round():
    history = [episode(11, 100, 102, 95, []), episode(12, 100, 100, 94, [])]
    memory, _ = compile_behavioral_memory(history, {"round": 13, "p": 100, "Q": 100, "D": 100}, 12)
    assert [e["derived"]["relevance"]["fraction"]["numerator"] for e in memory["episodes"]] == [588, 588]
    assert memory["relevance"]["closest_source_refs"] == ["round:12", "round:11"]


def test_conditional_price_median_uses_exact_episode_means_and_keeps_rational_audit():
    history = [episode(6, 100, 0, 1, [{"side": "buy", "qty": 2, "price": 90}]),
               episode(7, 100, 0, 1, [{"side": "buy", "qty": 3, "price": 91}])]
    memory, audit = compile_behavioral_memory(history, {"round": 13, "p": 100, "Q": 0, "D": 1}, 12)
    buy = memory["conditional_summaries"][0]["prices"]["buy"]
    assert buy["median"] == "90.50"
    assert audit["conditional_price_exact"]["shortage"]["buy"]["median"] == {
        "numerator": 181, "denominator": 2}
    assert audit["conditional_price_exact"]["shortage"]["buy"]["median_source_refs"] == ["round:6", "round:7"]


@pytest.mark.parametrize("history", [
    [episode(13, 100, 1, 1, [])],
    [episode(6, 100, 1, 1, []) | {"profit_cents": None}],
    [episode(6, 100, 1, 1, [{"side": "hold", "qty": 1, "price": 3}])],
    [episode(6, 100, 1, 1, []) | {"fills": []}],
    [episode(6, 100, 1, 1, []) | {"Q": 1, "q": 2}],
])
def test_compiler_rejects_invalid_or_leaking_history(history):
    with pytest.raises(ValueError):
        compile_behavioral_memory(history, {"round": 13, "p": 100, "Q": 1, "D": 1}, 12)


def test_compiler_rejects_current_outcome_alias_or_cutoff_conflicts():
    history = [episode(6, 100, 1, 1, [])]
    with pytest.raises(ValueError):
        compile_behavioral_memory(history, {"round": 13, "p": 100, "Q": 1, "q": 2, "D": 1}, 12)
    with pytest.raises(ValueError):
        compile_behavioral_memory(history, {"round": 13, "p": 100, "Q": 1, "D": 1,
            "label": {"orders": []}}, 12)
    with pytest.raises(ValueError):
        compile_behavioral_memory(history, {"round": 13, "p": 100, "Q": 1, "D": 1}, 13)

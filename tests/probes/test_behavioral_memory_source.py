from copy import deepcopy
from types import SimpleNamespace

import pytest

from capclaims.behavioral_memory.source import _extract_cases


def source_fixture():
    people = {f"synthetic-person-{i}" for i in range(18)}
    episodes, profits = [], {}
    for person in sorted(people):
        for r in range(6, 14):
            episodes.extend([
                SimpleNamespace(participant_id=person, round=r, scope="GA 10", treatment="GA",
                    phase="procurement", observed={"p": 50, "w": 10}, action={"Q": 100}),
                SimpleNamespace(participant_id=person, round=r, scope="GA 10", treatment="GA",
                    phase="orders", observed={"p": 50, "w": 10, "Q": 100, "D": 130},
                    action={"orders": [{"type": "BUY", "amount": 30, "price": 20}]}),
            ])
            profits[person, r] = 123400
    return episodes, people, profits


def test_source_extracts_18_and_keeps_targets_out_of_history():
    episodes, people, profits = source_fixture()
    original = deepcopy(episodes)
    cases, labels = _extract_cases(episodes, people, profits)
    assert len(cases) == len(labels) == 18
    assert sum(len(c["history"]) for c in cases) == 126
    for c in cases:
        assert c["current_state"] == {"round": 13, "p": 50, "w": 10, "Q": 100, "D": 130}
        assert [r["round"] for r in c["history"]] == list(range(6, 13))
        assert all("w" not in r and r["profit_cents"] == 123400 for r in c["history"])
        assert labels[c["person_id"]]["round"] == 13
    assert episodes == original


def test_target_labels_do_not_change_inference():
    episodes, people, profits = source_fixture()
    cases, labels = _extract_cases(episodes, people, profits)
    for row in episodes:
        if row.round == 13 and row.phase == "orders":
            row.action["orders"] = []
    for key in profits:
        if key[1] == 13:
            profits[key] = 0
    changed, changed_labels = _extract_cases(episodes, people, profits)
    assert changed == cases
    assert changed_labels != labels


def test_source_fails_on_duplicate_phase_conflict_or_missing_profit():
    episodes, people, profits = source_fixture()
    with pytest.raises(ValueError, match="duplicate"):
        _extract_cases(episodes + [deepcopy(episodes[0])], people, profits)
    episodes[1].observed["Q"] = 101
    with pytest.raises(ValueError, match="conflict"):
        _extract_cases(episodes, people, profits)
    episodes, people, profits = source_fixture()
    profits.pop(next(iter(profits)))
    with pytest.raises(ValueError, match="profit"):
        _extract_cases(episodes, people, profits)


def test_unselected_people_and_future_rows_are_excluded_before_validation():
    episodes, people, profits = source_fixture()
    expected = _extract_cases(episodes, people, profits)
    bad = SimpleNamespace(participant_id="unselected", round=13, scope="GA 10", treatment="GA",
        phase="orders", observed={}, action={})
    future = deepcopy(bad)
    future.participant_id = next(iter(people))
    future.round = 14
    assert _extract_cases(episodes + [bad, future], people, profits) == expected

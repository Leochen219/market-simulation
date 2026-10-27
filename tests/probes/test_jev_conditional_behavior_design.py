from collections import Counter

import pytest

from capclaims.conditional_behavior.study import build_plan, select_holdout
from capclaims.conditional_behavior.baselines import select_donor


def _row(round_no, *, p=50, q=100, d=100, profit=0):
    return {"round": round_no, "p": p, "Q": q, "D": d, "orders": [], "profit_cents": profit}


def _fixture():
    people = [f"p{i:02d}" for i in range(35)]
    target_used, donor_used = set(people[:12]), set(people[12:18])
    targets = {person: {round_no: {"round": round_no, "p": 60 + i, "w": 10,
        "Q": 100, "D": 100, "orders": []} for round_no in (14, 15)}
        for i, person in enumerate(people[18:])}
    histories = {person: [_row(round_no, p=60 + i, profit=round_no * 100)
        for round_no in range(1, 14)] for i, person in enumerate(people)}
    donors = [{"person_id": person, "history": histories[person][5:13]}
        for person in people[:18]]
    pool = [{**_row(round_no, p=70 + i, q=50 + i, d=100), "person_id": person}
        for i, person in enumerate(people[:18]) for round_no in range(6, 14)]
    return people, target_used, donor_used, targets, histories, donors, pool


def test_holdout_is_strict_union_of_exclusion_registries_and_exact_seventeen():
    people, target_used, donor_used, *_ = _fixture()
    holdout = select_holdout(people, target_used, donor_used)
    assert len(holdout) == 17 and set(holdout) == set(people) - (target_used | donor_used)
    with pytest.raises(ValueError):
        select_holdout(people, target_used, set(people[11:17]))


def test_plan_has_exactly_680_balanced_cases_and_shared_donor_per_person():
    _, target_used, donor_used, targets, histories, donors, pool = _fixture()
    plan = build_plan(targets=targets, own_histories=histories, donors=donors,
        public_history_pool=pool, excluded=target_used | donor_used, seed=20261026)
    inputs, labels = plan["inputs"], plan["targets"]
    assert len(inputs) == 680 and len(labels) == 34
    counts = Counter((row["person_id"], row["target_round"], row["arm"]) for row in inputs)
    assert set(counts.values()) == {4}
    assert {row["target_round"] for row in inputs} == {14, 15}
    assert all(row["observation"]["market_size"] == 35 for row in inputs)
    donor_by_person = {}
    for row in inputs:
        if row["arm"] == "D":
            donor_by_person.setdefault(row["person_id"], set()).add(row["donor_id"])
    assert all(len(donors_for_person) == 1 for donors_for_person in donor_by_person.values())
    assert all("profit_cents" not in row["observation"] for row in inputs)
    assert all("profit_cents" not in row for row in labels.values())


def test_plan_rejects_roster_overlap_and_wrong_history_windows():
    _, target_used, donor_used, targets, histories, donors, pool = _fixture()
    bad_targets = dict(targets)
    bad_targets[next(iter(bad_targets))][14]["D"] = 201
    with pytest.raises(ValueError):
        build_plan(targets=bad_targets, own_histories=histories, donors=donors,
            public_history_pool=pool, excluded=target_used | donor_used, seed=1)
    with pytest.raises(ValueError):
        build_plan(targets=targets, own_histories=histories, donors=donors[:-1],
            public_history_pool=pool, excluded=target_used | donor_used, seed=1)


def test_donor_matching_uses_aligned_own_history_not_current_target_state():
    target = [_row(round_no, p=50 + round_no, q=70, d=80) for round_no in range(6, 14)]
    exact = {"person_id": "near", "history": [dict(row) for row in target]}
    current_match = {"person_id": "wrong", "history": [_row(round_no, p=90, q=300, d=10)
            for round_no in range(6, 14)]}
    result = select_donor(target, [current_match, exact], seed=1, target_id="t")
    assert result["person_id"] == "near" and result["distance"] == 0


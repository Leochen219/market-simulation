import pytest

from capclaims.behavioral_memory.design import ARMS, build_prompt, build_schedule, keyed_uniform, stage_instructions
from capclaims.behavioral_memory.memory import compile_behavioral_memory


def test_schedule_has_108_interleaved_arm_trajectories_and_two_balanced_blocks():
    people = [f"person-{i}" for i in range(18)]
    schedule = build_schedule(people)
    assert len(schedule) == 108
    assert {arm: sum(r["arm"] == arm for r in schedule) for arm in ARMS} == {"M0": 36, "M1": 36, "M2": 36}
    assert {r for r in (0, 1) for r in (r,)} == {row["repeat"] for row in schedule}
    assert all(sum(row["repeat"] == repeat and row["arm"] == arm for row in schedule) == 18
               for repeat in (0, 1) for arm in ARMS)


def test_common_random_coordinates_do_not_depend_on_arm_and_prompt_mapping_is_explicit():
    x = keyed_uniform(seed=20261026, person_id="p", target_round=13, repeat=1,
                      decision_coordinate="order:0:price")
    assert x == keyed_uniform(seed=20261026, person_id="p", target_round=13, repeat=1,
                              decision_coordinate="order:0:price")
    prompt = build_prompt(arm="M0", current_state={"round": 13, "p": 100, "w": 10, "Q": 9, "D": 7}, history=[])
    assert '"round":18' in prompt
    assert "formal_round + 5 = wire_round" in prompt


def _compiled_memory(history=None):
    history = history if history is not None else [
        {"round": r, "p": 100, "Q": 10, "D": 8, "orders": [], "profit_cents": 0}
        for r in range(6, 13)]
    memory, _ = compile_behavioral_memory(history,
        {"round": 13, "p": 100, "w": 10, "Q": 10, "D": 8}, 12)
    return memory


def test_memory_arms_require_exact_compiled_history_window():
    memory = _compiled_memory()
    with pytest.raises(ValueError, match="memory"):
        build_prompt(arm="M1", current_state={"round": 13, "p": 100, "w": 10, "Q": 10, "D": 8},
                     history=memory["history"])
    with pytest.raises(ValueError, match="history"):
        build_prompt(arm="M2", current_state={"round": 13, "p": 100, "w": 10, "Q": 10, "D": 8},
                     history=memory["history"][:-1], memory=memory)
    wrong_cutoff = {**memory, "cutoff_round": 11}
    with pytest.raises(ValueError, match="cutoff"):
        build_prompt(arm="M1", current_state={"round": 13, "p": 100, "w": 10, "Q": 10, "D": 8},
                     history=memory["history"], memory=wrong_cutoff)
    missing_round = {**memory, "history": memory["history"][:-1]}
    with pytest.raises(ValueError, match="rounds"):
        build_prompt(arm="M2", current_state={"round": 13, "p": 100, "w": 10, "Q": 10, "D": 8},
                     history=memory["history"], memory=missing_round)


def test_dynamic_stage_prompt_preserves_protocol_windows():
    prompt = stage_instructions("count")
    assert "formal rounds 6 through 13" not in prompt
    assert "game rounds 11 through 18" not in prompt
    assert "formal rounds 6 through 12" in prompt
    assert "game rounds 11 through 17" in prompt

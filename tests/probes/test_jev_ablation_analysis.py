import pytest

from tools.analysis.analyze_jev_ablation import paired_effect, repeat_variation


def test_paired_effect_joins_identical_case_order_repeat_blocks():
    rows = [dict(condition=c, case_id="case", order="original", repetition=r,
                 sell_probability=value)
            for c, values in (("base", [0.1, 0.3]), ("changed", [0.4, 0.5]))
            for r, value in enumerate(values)]
    result = paired_effect(list(reversed(rows)), "base", "changed", "sell_probability")
    assert result["n_pairs"] == 2
    assert result["mean_delta"] == pytest.approx(0.25)
    assert result["positive"] == 2


def test_paired_effect_rejects_missing_or_duplicate_blocks():
    row = dict(condition="base", case_id="case", order="original", repetition=0, correct=True)
    with pytest.raises(ValueError):
        paired_effect([row], "base", "changed", "correct")
    with pytest.raises(ValueError):
        paired_effect([row, row, dict(row, condition="changed")], "base", "changed", "correct")


def test_repeat_variation_does_not_treat_tie_break_or_dict_order_as_model_changes():
    row = dict(condition="base", case_id="case", order="original",
               top_candidate_ids=["c0", "c1"], reply={"probabilities": {"c0": 0.5, "c1": 0.5}})
    rows = [dict(row, selected="STOP"), dict(row, selected="SELL", top_candidate_ids=["c1", "c0"],
            reply={"probabilities": {"c1": 0.5, "c0": 0.5}})]
    result = repeat_variation(rows)
    assert result[0]["probability_vectors"] == 1
    assert result[0]["top_sets"] == 1

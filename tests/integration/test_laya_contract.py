import pytest

from capclaims.backends.laya import normalize_laya_probabilities, validate_laya_usage


def test_laya_probability_rounding_is_bounded_and_normalized():
    values = normalize_laya_probabilities({"a": .3333, "b": .3333, "c": .3333}, ["a", "b", "c"])
    assert abs(sum(values.values()) - 1.0) < 1e-12
    with pytest.raises(ValueError, match="candidate"):
        normalize_laya_probabilities({"a": 1.0}, ["a", "b"])
    with pytest.raises(ValueError, match="rounding|sum"):
        normalize_laya_probabilities({"a": .1, "b": .1}, ["a", "b"])


def test_laya_usage_requires_explicit_untruncated_measurements():
    good = {"input_tokens": 4, "output_tokens": 2, "state_tokens": 2,
            "state_tokens_dropped": 0, "truncated": False, "truncated_questions": []}
    assert validate_laya_usage(good)["truncated"] is False
    with pytest.raises(ValueError):
        validate_laya_usage({**good, "truncated": True})
    with pytest.raises(ValueError):
        validate_laya_usage({key: value for key, value in good.items() if key != "state_tokens_dropped"})

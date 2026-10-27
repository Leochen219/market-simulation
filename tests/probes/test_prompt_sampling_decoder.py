import concurrent.futures

import pytest

from capclaims.prompt_sampling_decoder import decode, stable_draw, validate_distribution


def test_threshold_inclusive_and_below_is_dropped():
    result = decode({"a": .10, "b": .099, "c": .801}, ["a", "b", "c"], "top3_cutoff10",
        private_person="p", position="slot0:side")
    assert result["retained_probabilities"] == pytest.approx({"a": .10 / .901, "c": .801 / .901})
    assert result["dropped_mass"] == pytest.approx(.099)


@pytest.mark.parametrize("raw,expected", [({"a": .10, "b": .89}, ["b", "a"]),
    ({"a": .10, "b": .91}, ["b", "a"])])
def test_cutoff_uses_raw_mass_before_allowed_normalization(raw, expected):
    result = decode(raw, list(raw), "top3_cutoff10")
    assert list(result["retained_probabilities"]) == expected
    assert result["raw_total_mass"] == pytest.approx(sum(raw.values()))
    assert result["retained_raw_mass"] + result["cutoff_dropped_mass"] + result["topk_dropped_mass"] == pytest.approx(sum(raw.values()))


def test_argmax_exact_ties_resolve_by_candidate_order():
    result = decode({"later": .5, "first": .5}, ["first", "later"], "client_argmax")
    assert result["selected"] == "first"


def test_singleton_is_deterministic_and_empty_top3_rejects():
    assert decode({"only": 1}, ["only"], "top3_cutoff10")["selected"] == "only"
    with pytest.raises(ValueError, match="no retained"):
        decode({f"c{i}": 1 / 11 for i in range(11)}, [f"c{i}" for i in range(11)], "top3_cutoff10")


@pytest.mark.parametrize("raw", [
    {"a": True, "b": 0}, {"a": -1, "b": 2}, {"a": float("nan"), "b": .5},
    {"a": 1}, {"a": .5, "b": .5, "extra": 0}, {"a": "0.5", "b": .5},
])
def test_invalid_probability_map_is_rejected(raw):
    with pytest.raises(ValueError):
        validate_distribution(raw, ["a", "b"])


def test_sampling_is_stateless_and_order_invariant():
    coords = [(20261026, f"p{i}", i % 2, "orders:slot0:side") for i in range(150)]
    serial = [stable_draw(*coord) for coord in coords]
    with concurrent.futures.ThreadPoolExecutor(10) as pool:
        parallel = list(pool.map(lambda coord: stable_draw(*coord), reversed(coords)))
    assert dict(zip(coords, serial)) == dict(zip(reversed(coords), parallel))


def test_sampling_frequency_tracks_retained_distribution():
    chosen = [decode({"a": .25, "b": .75}, ["a", "b"], "top3_cutoff10", seed=11,
        private_person=f"synthetic-{i}")["selected"] for i in range(1000)]
    assert .68 < chosen.count("b") / len(chosen) < .82

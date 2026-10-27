import unittest

from tools.probes.common.fixture import load_worst_case_fixture
from tools.probes.common.fixture_validation import validate_fixture
from tools.probes.common.result import base_result, sanitize
from tools.probes.common.score_contract import ScoreContractError, normalize_scores


class ScoreContractTests(unittest.TestCase):
    def test_probability_scores_require_full_candidate_set_and_unit_sum(self):
        scores = {"q0": 0.25, "q1": 0.75}
        self.assertEqual(normalize_scores(scores, ["q0", "q1"], "probabilities"), scores)
        for invalid in ({"q0": 1.0}, {"q0": 0.2, "q1": 0.2}, {"q0": -0.1, "q1": 1.1}):
            with self.subTest(invalid=invalid), self.assertRaises(ScoreContractError):
                normalize_scores(invalid, ["q0", "q1"], "probabilities")

    def test_finite_logits_use_stable_softmax(self):
        got = normalize_scores({"a": 1000.0, "b": 999.0}, ["a", "b"], "logits")
        self.assertAlmostEqual(got["a"], 1.0 / (1.0 + 2.718281828459045 ** -1), places=12)
        self.assertAlmostEqual(sum(got.values()), 1.0, places=12)
        with self.assertRaises(ScoreContractError):
            normalize_scores({"a": float("nan"), "b": 0.0}, ["a", "b"], "logits")

    def test_fixture_has_expected_wave_sizes_and_unique_ids(self):
        fixture = load_worst_case_fixture()
        validate_fixture(fixture)
        self.assertEqual(len(fixture["waves"]["purchase"]["candidates"]), 17)
        self.assertEqual(len(fixture["waves"]["trade_side"]["candidates"]), 3)
        self.assertEqual(len(fixture["waves"]["trade_qty"]["candidates"]), 10)
        self.assertEqual(len(fixture["waves"]["trade_price"]["candidates"]), 11)
        self.assertEqual(len(fixture["history"]), 5)

    def test_invalid_fixture_fails_explicitly(self):
        fixture = load_worst_case_fixture()
        fixture["waves"]["purchase"]["candidates"].pop()
        with self.assertRaisesRegex(ValueError, "purchase candidates=16"):
            validate_fixture(fixture)

    def test_result_schema_and_credential_redaction_are_dependency_free(self):
        result = base_result("jev", "abc")
        self.assertEqual(result["status"], "BLOCKED")
        self.assertEqual(result["fixture_hash"], "abc")
        self.assertEqual(sanitize({"api_key": "secret", "nested": [{"token": "secret"}]}),
                         {"api_key": "[REDACTED]", "nested": [{"token": "[REDACTED]"}]})


if __name__ == "__main__":
    unittest.main()

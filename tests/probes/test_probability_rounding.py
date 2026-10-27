import unittest
from tools.probes.common.score_contract import normalize_scores, ScoreContractError

class ProbabilityRoundingTests(unittest.TestCase):
    def test_quantized_probabilities_are_normalized_only_within_rounding_bound(self):
        raw = {f"q{i}": 0.0588 for i in range(17)}
        normalized = normalize_scores(raw, list(raw), "probabilities", source_precision=4)
        self.assertAlmostEqual(sum(normalized.values()), 1.0, places=12)
        self.assertAlmostEqual(raw["q0"], 0.0588)
        with self.assertRaises(ScoreContractError):
            normalize_scores({"a": 0.8, "b": 0.3}, ["a", "b"], "probabilities", source_precision=4)

if __name__ == "__main__":
    unittest.main()

import unittest

from tools.probes.laya_capability_probe import (
    BATCH_ROWS,
    _state_text,
    evaluate_gate,
)
from tools.probes.common.fixture import load_worst_case_fixture


def passing_evidence():
    usage = {"input_tokens": 100, "output_tokens": 0, "state_tokens": 50,
             "state_tokens_dropped": 0, "truncated": False,
             "truncated_questions": []}
    token_row = {"state_tokens_tokenizer_count": 50,
                 "question_tokens_tokenizer_count": 10,
                 "max_candidate_label_tokens": 8}
    wave = {"usage": [dict(usage) for _ in range(BATCH_ROWS)],
            "input_token_counts": [dict(token_row) for _ in range(BATCH_ROWS)],
            "effective_token_limits_verified": {"max_state_tokens": 50,
                "max_candidate_label_tokens": 8, "max_question_tokens": 10},
            "request_kwargs": {"max_len": 1024, "head_max_len": 256,
                               "agent_config_defaults": {"max_len": 1000, "head_max_len": 250}},
            "perturbation_changed_input": True}
    result = {"supports_scores": True, "supports_batch": True, "max_len_support": True,
              "head_max_len_support": True, "score_keysets_match": True,
              "max_probability_error": 0.0, "cross_row_isolation": 0.0,
              "argmax_agreement": 1.0,
              "observed_batch_sizes_by_wave": {"purchase": [BATCH_ROWS]}}
    return result, {"waves": {"purchase": wave}}


class LayaGateTests(unittest.TestCase):
    def test_heterogeneous_rows_have_distinct_content_and_lengths(self):
        fixture = load_worst_case_fixture()
        states = [_state_text(fixture, i) for i in range(BATCH_ROWS)]
        self.assertEqual(len(set(states)), BATCH_ROWS)
        self.assertGreater(len(set(map(len, states))), 1)
        self.assertIn('"probe_row": 9', states[9])

    def test_complete_gate_evidence_passes(self):
        result, raw = passing_evidence()
        self.assertEqual(evaluate_gate(result, raw), (True, []))

    def test_missing_usage_fails_closed(self):
        result, raw = passing_evidence()
        raw["waves"]["purchase"]["usage"][0] = {}
        passed, failures = evaluate_gate(result, raw)
        self.assertFalse(passed)
        self.assertTrue(any("invalid or missing usage" in item for item in failures))

    def test_missing_truncated_questions_fails_closed(self):
        result, raw = passing_evidence()
        del raw["waves"]["purchase"]["usage"][0]["truncated_questions"]
        passed, failures = evaluate_gate(result, raw)
        self.assertFalse(passed)
        self.assertTrue(any("invalid or missing usage" in item for item in failures))

    def test_nonempty_or_malformed_truncated_questions_fails_closed(self):
        for invalid_value in (["decision"], "", None):
            result, raw = passing_evidence()
            raw["waves"]["purchase"]["usage"][0]["truncated_questions"] = invalid_value
            with self.subTest(value=invalid_value):
                passed, failures = evaluate_gate(result, raw)
                self.assertFalse(passed)
                self.assertTrue(any("invalid or missing usage" in item for item in failures))

    def test_truncation_fails(self):
        result, raw = passing_evidence()
        raw["waves"]["purchase"]["usage"][0]["truncated"] = True
        passed, failures = evaluate_gate(result, raw)
        self.assertFalse(passed)
        self.assertTrue(any("truncation observed" in item for item in failures))

    def test_dropped_tokens_fail_even_when_truncated_flag_is_false(self):
        result, raw = passing_evidence()
        raw["waves"]["purchase"]["usage"][0]["state_tokens_dropped"] = 1
        passed, failures = evaluate_gate(result, raw)
        self.assertFalse(passed)
        self.assertTrue(any("truncation observed" in item for item in failures))

    def test_batch_size_one_fails(self):
        result, raw = passing_evidence()
        result["observed_batch_sizes_by_wave"]["purchase"] = [1]
        passed, failures = evaluate_gate(result, raw)
        self.assertFalse(passed)
        self.assertTrue(any("batch size below two" in item for item in failures))

    def test_cross_row_leak_fails(self):
        result, raw = passing_evidence()
        result["cross_row_isolation"] = 0.01
        passed, failures = evaluate_gate(result, raw)
        self.assertFalse(passed)
        self.assertTrue(any("cross-row isolation" in item for item in failures))

    def test_parity_above_tolerance_fails(self):
        result, raw = passing_evidence()
        result["max_probability_error"] = 1.1e-4
        passed, failures = evaluate_gate(result, raw)
        self.assertFalse(passed)
        self.assertTrue(any("parity" in item for item in failures))

    def test_nan_and_none_parity_fail(self):
        for invalid_error in (float("nan"), None):
            result, raw = passing_evidence()
            result["max_probability_error"] = invalid_error
            with self.subTest(value=invalid_error):
                passed, failures = evaluate_gate(result, raw)
                self.assertFalse(passed)
                self.assertTrue(any("parity" in item for item in failures))

    def test_empty_waves_fail(self):
        result, raw = passing_evidence()
        raw["waves"] = {}
        passed, failures = evaluate_gate(result, raw)
        self.assertFalse(passed)
        self.assertEqual(failures, ["no wave measurements"])

    def test_wrong_usage_row_count_fails(self):
        result, raw = passing_evidence()
        raw["waves"]["purchase"]["usage"].pop()
        passed, failures = evaluate_gate(result, raw)
        self.assertFalse(passed)
        self.assertTrue(any("missing per-row usage" in item for item in failures))

    def test_no_high_margin_rows_fails(self):
        result, raw = passing_evidence()
        result["argmax_agreement"] = None
        passed, failures = evaluate_gate(result, raw)
        self.assertFalse(passed)
        self.assertTrue(any("high-margin" in item for item in failures))

    def test_probability_key_mismatch_fails(self):
        result, raw = passing_evidence()
        result["score_keysets_match"] = False
        passed, failures = evaluate_gate(result, raw)
        self.assertFalse(passed)
        self.assertTrue(any("probability keys" in item for item in failures))


if __name__ == "__main__":
    unittest.main()

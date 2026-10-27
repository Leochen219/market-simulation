import asyncio
import math
import unittest
from tools.probes.common.result import sanitize
from tools.probes.jev_capability_probe import ConfiguredRateLimiter

class ProbeUtilityTests(unittest.TestCase):
    def test_jev_rate_limiter_rejects_nonfinite_or_nonpositive_limits(self):
        for rps, tps in ((0, 1), (1, 0), (math.nan, 1), (1, math.inf)):
            with self.subTest(rps=rps, tps=tps), self.assertRaises(ValueError):
                ConfiguredRateLimiter(rps, tps)

    def test_secret_text_is_redacted_from_error_strings(self):
        import os
        old = os.environ.get("TYPESAFE_API_KEY")
        try:
            os.environ["TYPESAFE_API_KEY"] = "private-value"
            self.assertEqual(sanitize("failed private-value"), "failed [REDACTED]")
        finally:
            if old is None:
                os.environ.pop("TYPESAFE_API_KEY", None)
            else:
                os.environ["TYPESAFE_API_KEY"] = old

if __name__ == "__main__":
    unittest.main()

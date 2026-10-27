import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

class ProbeCLITests(unittest.TestCase):
    def test_laya_cli_help_does_not_import_optional_framework(self):
        result = subprocess.run([sys.executable, str(PROJECT_ROOT / "tools" / "probes" / "laya_capability_probe.py"), "--help"],
                                cwd=PROJECT_ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--model-path", result.stdout)

    def test_jev_without_official_key_writes_truthful_blocked_result(self):
        env = {key: value for key, value in os.environ.items() if key not in {"TYPESAFE_API_KEY", "JEV_API_KEY"}}
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "tests") as output_dir:
            result = subprocess.run([
                sys.executable, str(PROJECT_ROOT / "tools" / "probes" / "jev_capability_probe.py"),
                "--output-dir", output_dir,
            ], cwd=PROJECT_ROOT, capture_output=True, text=True, env=env)
            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            self.assertIn('"status": "BLOCKED"', result.stdout)
            self.assertTrue((Path(output_dir) / "jev-raw.json").exists())
            self.assertTrue((Path(output_dir) / "jev-result.json").exists())

    def test_jev_key_without_rps_tps_is_blocked_before_any_sdk_call(self):
        env = {key: value for key, value in os.environ.items() if key not in {"TYPESAFE_API_KEY", "JEV_API_KEY"}}
        env["TYPESAFE_API_KEY"] = "placeholder-for-limits-check"
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "tests") as output_dir:
            result = subprocess.run([
                sys.executable, str(PROJECT_ROOT / "tools" / "probes" / "jev_capability_probe.py"),
                "--output-dir", output_dir,
            ], cwd=PROJECT_ROOT, capture_output=True, text=True, env=env)
            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            self.assertIn("Explicit account RPS and TPS limits are required", result.stdout)
            self.assertIn('"status": "BLOCKED"', result.stdout)

if __name__ == "__main__":
    unittest.main()

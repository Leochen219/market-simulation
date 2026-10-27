import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from tools.probes.laya_capability_probe import load_and_configure_laya_agent

REVISION = "1720e3e3357cfe1e281542e223f8273b0890ca34"

class FakeModel:
    def __init__(self): self.kwargs = None
    def to(self, **kwargs): self.kwargs = kwargs
    def eval(self): return self
    def parameters(self): return []

class FakeAgent:
    def __init__(self):
        self.model = FakeModel()
        self.device = "cuda:0"
        self.dtype = "bfloat16"
        self.amp_enabled = True
    def dtype_for(self, rows): return self.dtype if self.amp_enabled else "float32"

class FakeTorch:
    float32 = "float32"

class LayaProbeSetupTests(unittest.TestCase):
    def test_loaded_agent_is_forced_to_fp32_before_probe_returns_it(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as folder:
            metadata = Path(folder) / ".cache" / "huggingface" / "download" / "config.json.metadata"
            metadata.parent.mkdir(parents=True)
            metadata.write_text(REVISION + "\n", encoding="utf-8")
            agent = FakeAgent()
            calls = []
            fake_laya = SimpleNamespace(load=lambda *args, **kwargs: (calls.append((args, kwargs)) or agent))
            loaded, revision, precision = load_and_configure_laya_agent(Path(folder), "cuda", FakeTorch(), fake_laya)
            self.assertIs(loaded, agent)
            self.assertEqual(revision, REVISION)
            self.assertEqual(calls[0][1]["revision"], REVISION)
            self.assertFalse(loaded.amp_enabled)
            self.assertEqual(loaded.model.kwargs["dtype"], FakeTorch.float32)
            self.assertTrue(precision["dtype_for_large_batch_is_fp32"])

if __name__ == "__main__":
    unittest.main()

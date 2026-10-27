import unittest
from tools.probes.common.precision import configure_fp32

class FakeModel:
    def __init__(self): self.kwargs = None
    def to(self, **kwargs): self.kwargs = kwargs
    def eval(self): return self
    def parameters(self): return []

class FakeTorch:
    float32 = "float32"

class FakeAgent:
    def __init__(self):
        self.model = FakeModel()
        self.device = "cuda:0"
        self.amp_enabled = True
        self.dtype = "bfloat16"
    def dtype_for(self, rows): return self.dtype if self.amp_enabled else "float32"

class PrecisionTests(unittest.TestCase):
    def test_laya_runtime_is_forced_to_full_float32(self):
        agent = FakeAgent()
        metadata = configure_fp32(agent, FakeTorch())
        self.assertFalse(agent.amp_enabled)
        self.assertEqual(agent.dtype, "float32")
        self.assertEqual(agent.model.kwargs, {"device": "cuda:0", "dtype": "float32"})
        self.assertTrue(metadata["dtype_for_large_batch_is_fp32"])

if __name__ == "__main__":
    unittest.main()

import tempfile
import unittest
from pathlib import Path

from tools.probes.laya_capability_probe import _resolve_snapshot_revision

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REVISION = "1720e3e3357cfe1e281542e223f8273b0890ca34"

class SnapshotRevisionTests(unittest.TestCase):
    def test_flat_hf_local_dir_must_have_one_consistent_cached_revision(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / "tests") as folder:
            metadata = Path(folder) / ".cache" / "huggingface" / "download" / "model.safetensors.metadata"
            metadata.parent.mkdir(parents=True)
            metadata.write_text(REVISION + "\n", encoding="utf-8")
            self.assertEqual(_resolve_snapshot_revision(Path(folder)), REVISION)
            (metadata.parent / "config.json.metadata").write_text("f" * 40 + "\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "does not prove one pinned"):
                _resolve_snapshot_revision(Path(folder))

if __name__ == "__main__":
    unittest.main()

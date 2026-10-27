"""Download immutable Laya weights into the project-local ignored model cache."""
from pathlib import Path
from huggingface_hub import snapshot_download

MODEL_ID = "convaiinnovations/laya-multilingual"
REVISION = "1720e3e3357cfe1e281542e223f8273b0890ca34"

if __name__ == "__main__":
    destination = Path(__file__).resolve().parents[1] / ".cache/models/laya-multilingual"
    snapshot_download(
        MODEL_ID, revision=REVISION, local_dir=destination,
        allow_patterns=["*.json", "*.safetensors", "*.bin", "*.txt", "*.model", "*.py", "*.yaml", "*.yml"],
    )
    print(f"Downloaded {MODEL_ID}@{REVISION} to {destination}")

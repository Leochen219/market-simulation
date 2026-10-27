"""Common result envelope and dependency-safe environment capture."""
from __future__ import annotations
import importlib.metadata
import os
import platform
import sys
from typing import Any

RESULT_FIELDS = (
    "backend", "model_revision", "sdk_revision", "os", "python", "cuda", "gpu", "memory",
    "supports_batch", "observed_batch_sizes_by_wave", "forward_count_by_wave", "supports_scores",
    "score_semantics", "tested_candidate_count", "context_limit", "max_len_support", "head_max_len_support",
    "argmax_agreement", "max_probability_error", "cross_row_isolation", "fixture_hash", "cost_sample", "status",
)

def base_result(backend: str, fixture_hash: str) -> dict[str, Any]:
    result = {key: None for key in RESULT_FIELDS}
    result.update({
        "backend": backend,
        "model_revision": None,
        "sdk_revision": None,
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "cuda": None,
        "gpu": None,
        "memory": None,
        "supports_batch": False,
        "observed_batch_sizes_by_wave": {},
        "forward_count_by_wave": {},
        "supports_scores": False,
        "score_semantics": None,
        "tested_candidate_count": None,
        "context_limit": None,
        "max_len_support": None,
        "head_max_len_support": None,
        "argmax_agreement": None,
        "max_probability_error": None,
        "cross_row_isolation": None,
        "fixture_hash": fixture_hash,
        "cost_sample": None,
        "status": "BLOCKED",
    })
    return result

def package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None

def sanitize(value: Any) -> Any:
    """Recursively redact credential-like values from SDK/model diagnostic payloads."""
    secret_names = {"api_key", "apikey", "authorization", "token", "secret", "password"}
    if isinstance(value, dict):
        clean = {}
        for key, item in value.items():
            if str(key).lower().replace("-", "_") in secret_names:
                clean[str(key)] = "[REDACTED]"
            else:
                clean[str(key)] = sanitize(item)
        return clean
    if isinstance(value, (list, tuple)):
        return [sanitize(item) for item in value]
    if isinstance(value, str):
        for env_name in ("TYPESAFE_API_KEY",):
            secret = os.environ.get(env_name)
            if secret:
                value = value.replace(secret, "[REDACTED]")
    return value

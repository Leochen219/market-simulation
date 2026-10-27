"""Real Laya multilingual batch and score capability probe (no market simulation)."""
from __future__ import annotations
import argparse
import hashlib
import json
import subprocess
import math
import re
import time
from pathlib import Path
import sys
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.probes.common.fixture import load_worst_case_fixture, sha256_json, write_json
from tools.probes.common.fixture_validation import validate_fixture
from tools.probes.common.result import base_result, package_version, sanitize
from tools.probes.common.precision import configure_fp32
from tools.probes.common.memory import host_memory_bytes
from tools.probes.common.score_contract import normalize_scores

MODEL_ID = "convaiinnovations/laya-multilingual"
EXPECTED_CONTEXT = 1024
EXPECTED_HEAD = 256
BATCH_ROWS = 10
PROBABILITY_TOLERANCE = 1e-4
HIGH_MARGIN = 2e-4


def _resolve_snapshot_revision(model_path: Path) -> str:
    if re.fullmatch(r"[0-9a-fA-F]{40}", model_path.name):
        return model_path.name.lower()
    metadata = sorted((model_path / ".cache" / "huggingface" / "download").rglob("*.metadata"))
    revisions = set()
    for entry in metadata:
        first_line = entry.read_text(encoding="utf-8").splitlines()
        if first_line and re.fullmatch(r"[0-9a-fA-F]{40}", first_line[0].strip()):
            revisions.add(first_line[0].strip().lower())
    if len(revisions) != 1:
        raise RuntimeError(f"model directory does not prove one pinned 40-hex HF revision: {sorted(revisions)}")
    return next(iter(revisions))


def _code_digest(package_path: Path) -> str:
    package_root = package_path.parent
    digest = hashlib.sha256()
    for source in sorted(package_root.rglob("*.py")):
        relative = source.relative_to(package_root).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(hashlib.sha256(source.read_bytes()).digest())
    return digest.hexdigest()


def _git_revision(package_path: Path) -> str | None:
    try:
        completed = subprocess.run(["git", "-C", str(package_path.parent), "rev-parse", "HEAD"],
                                   check=True, capture_output=True, text=True, timeout=5)
        revision = completed.stdout.strip()
        return revision if re.fullmatch(r"[0-9a-fA-F]{40}", revision) else None
    except (OSError, subprocess.SubprocessError):
        return None


def _state_text(fixture: dict[str, Any], probe_row: int = 0, perturbation: str = "") -> str:
    """Create distinct state rows with variable histories and own values."""
    history_count = probe_row % (len(fixture["history"]) + 1)
    payload = {
        "observation": {**fixture["observation"], "own_p": fixture["observation"]["own_p"] + probe_row,
                       "own_q": fixture["observation"]["own_q"] + probe_row * 3,
                       "probe_row": probe_row},
        "history": fixture["history"][:history_count],
    }
    if perturbation:
        payload["probe_other_row_perturbation"] = perturbation
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _questions(wave: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {"decision": {
        "type": "choice",
        "instructions": wave["question"],
        "criteria": {c["candidate_id"]: c["label"] for c in wave["candidates"]},
    }}


def _sync(device: str, torch: Any) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _validate_usage(result: dict[str, Any], require_untruncated: bool = True) -> dict[str, Any]:
    from collections.abc import Mapping
    usage = result.get("usage")
    required_ints = ("input_tokens", "output_tokens", "state_tokens", "state_tokens_dropped")
    if not isinstance(usage, Mapping):
        raise RuntimeError("Laya response is missing a usage mapping")
    if not isinstance(usage.get("truncated"), bool):
        raise RuntimeError("Laya usage.truncated must be an explicit boolean")
    truncated_questions = usage.get("truncated_questions")
    if not isinstance(truncated_questions, list):
        raise RuntimeError("Laya usage.truncated_questions must be an explicit list")
    if truncated_questions:
        raise RuntimeError(f"Laya truncated question heads: {truncated_questions}")
    for key in required_ints:
        value = usage.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RuntimeError(f"Laya usage.{key} must be an explicit nonnegative integer")
    if usage["input_tokens"] <= 0 or usage["state_tokens"] <= 0:
        raise RuntimeError("Laya input_tokens and state_tokens must be positive")
    if require_untruncated and (usage["truncated"] or usage["state_tokens_dropped"] != 0):
        raise RuntimeError(f"Laya truncated state: {usage}")
    if usage.get("options"):
        raise RuntimeError(f"Laya collapsed candidate options: {usage['options']}")
    return dict(usage)


def _probabilities(result: dict[str, Any], candidate_ids: list[str]) -> dict[str, float]:
    _validate_usage(result)
    answer = (result.get("answers") or {}).get("decision")
    if not isinstance(answer, dict) or answer.get("type") != "choice":
        raise RuntimeError(f"unexpected Laya answer schema: {result!r}")
    raw = answer.get("probabilities")
    if not isinstance(raw, dict):
        raise RuntimeError("Laya choice answer did not include full probabilities")
    if set(raw) != set(candidate_ids):
        raise RuntimeError(f"Laya probability keys mismatch: got={sorted(raw)}, expected={sorted(candidate_ids)}")
    # Record resolution as observed from actual public values, not only from SDK docs.
    if any(round(float(value), 4) != float(value) for value in raw.values()):
        raise RuntimeError("Laya public probability scores contain values finer than four decimal places")
    return normalize_scores(raw, candidate_ids, "probabilities", source_precision=4)


def evaluate_gate(result: dict[str, Any], raw: dict[str, Any]) -> tuple[bool, list[str]]:
    """Pure gate evaluator, kept separate so failure predicates have regression tests."""
    failures: list[str] = []
    waves = raw.get("waves")
    if not isinstance(waves, dict) or not waves:
        return False, ["no wave measurements"]
    if result.get("supports_scores") is not True:
        failures.append("full candidate scores were not verified")
    if result.get("supports_batch") is not True:
        failures.append("batch execution was not verified")
    if result.get("max_len_support") is not True or result.get("head_max_len_support") is not True:
        failures.append("context/head length arguments were not verified")
    if result.get("score_keysets_match") is not True:
        failures.append("candidate probability keys did not match")
    parity_error = result.get("max_probability_error")
    if not isinstance(parity_error, (int, float)) or not math.isfinite(parity_error) or parity_error > PROBABILITY_TOLERANCE:
        failures.append("batch/single parity exceeded tolerance or was unavailable")
    isolation_error = result.get("cross_row_isolation")
    if not isinstance(isolation_error, (int, float)) or not math.isfinite(isolation_error) or isolation_error > PROBABILITY_TOLERANCE:
        failures.append("cross-row isolation exceeded tolerance or was unavailable")
    if result.get("argmax_agreement") != 1.0:
        failures.append("no high-margin agreement evidence or disagreement")
    for name, wave in waves.items():
        if not any(isinstance(size, int) and size >= 2 for size in result.get("observed_batch_sizes_by_wave", {}).get(name, [])):
            failures.append(f"{name}: observed batch size below two")
        usage_rows = wave.get("usage")
        if not isinstance(usage_rows, list) or len(usage_rows) != BATCH_ROWS:
            failures.append(f"{name}: missing per-row usage measurements")
            continue
        for usage in usage_rows:
            try:
                valid_usage = _validate_usage({"usage": usage}, require_untruncated=False)
            except (TypeError, RuntimeError):
                failures.append(f"{name}: invalid or missing usage fields")
                break
            if valid_usage["truncated"] or valid_usage["state_tokens_dropped"] != 0:
                failures.append(f"{name}: truncation observed")
                break
        token_rows = wave.get("input_token_counts")
        limits = wave.get("effective_token_limits_verified")
        request_kwargs = wave.get("request_kwargs")
        if (not isinstance(token_rows, list) or len(token_rows) != BATCH_ROWS or
                not isinstance(limits, dict) or not isinstance(request_kwargs, dict) or
                request_kwargs.get("max_len") != EXPECTED_CONTEXT or
                request_kwargs.get("head_max_len") != EXPECTED_HEAD or
                not isinstance(request_kwargs.get("agent_config_defaults"), dict)):
            failures.append(f"{name}: missing tokenizer measurements")
        elif (limits.get("max_state_tokens", EXPECTED_CONTEXT + 1) > EXPECTED_CONTEXT or
              limits.get("max_candidate_label_tokens", EXPECTED_HEAD + 1) > EXPECTED_HEAD or
              limits.get("max_question_tokens", EXPECTED_HEAD + 1) > EXPECTED_HEAD or
              any(not isinstance(row.get("state_tokens_tokenizer_count"), int) or
                  not isinstance(row.get("question_tokens_tokenizer_count"), int) or
                  not isinstance(row.get("max_candidate_label_tokens"), int) for row in token_rows)):
            failures.append(f"{name}: measured state/head token counts exceed configured limits")
        if wave.get("perturbation_changed_input") is not True:
            failures.append(f"{name}: perturbation did not change encoded input")
    return not failures, failures


def _max_abs_error(left: dict[str, float], right: dict[str, float]) -> float:
    return max(abs(left[key] - right[key]) for key in left)


def _instrument_root_forward(model: Any) -> tuple[list[dict[str, Any]], Any]:
    calls: list[dict[str, Any]] = []
    def dtype_tree(value: Any) -> Any:
        if hasattr(value, "dtype"):
            return str(value.dtype)
        if isinstance(value, (tuple, list)):
            return [dtype_tree(item) for item in value]
        if isinstance(value, dict):
            return {str(key): dtype_tree(item) for key, item in value.items()}
        return None
    def hook(_module: Any, inputs: Any, output: Any) -> None:
        tensor = next((value for value in inputs if hasattr(value, "shape")), None)
        calls.append({"input_shape": list(tensor.shape) if tensor is not None else None,
                      "input_dtypes": dtype_tree(inputs), "output_dtypes": dtype_tree(output)})
    handle = model.register_forward_hook(hook)
    return calls, handle


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(token_id) for token_id in ids]


def _token_counts(agent: Any, state: str, wave: dict[str, Any]) -> dict[str, Any]:
    state_ids = _token_ids(agent.tok, state)
    head_ids = _token_ids(agent.tok, wave["question"])
    candidate_counts = {
        candidate["candidate_id"]: len(_token_ids(agent.tok, candidate["label"]))
        for candidate in wave["candidates"]
    }
    return {"state_tokens_tokenizer_count": len(state_ids),
            "question_tokens_tokenizer_count": len(head_ids),
            "candidate_label_tokens_by_id": candidate_counts,
            "max_candidate_label_tokens": max(candidate_counts.values()),
            "state_token_ids_sha256": hashlib.sha256(json.dumps(state_ids).encode("ascii")).hexdigest()}


def load_and_configure_laya_agent(model_path: Path, device: str, torch: Any, laya: Any) -> tuple[Any, str, dict[str, Any]]:
    if not model_path.is_dir():
        raise FileNotFoundError(f"--model-path is not a local snapshot directory: {model_path}")
    revision = _resolve_snapshot_revision(model_path)
    agent = laya.load(str(model_path), device=device, revision=revision, backend="eager")
    precision = configure_fp32(agent, torch)
    return agent, revision, precision

def run_probe(model_path: Path, device: str, output_dir: Path) -> dict[str, Any]:
    fixture = load_worst_case_fixture()
    validate_fixture(fixture)
    fixture_hash = sha256_json(fixture)
    result = base_result("laya", fixture_hash)
    raw: dict[str, Any] = {"fixture_hash": fixture_hash, "model_id": MODEL_ID, "waves": {}}
    try:
        import torch
        import laya

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested but torch.cuda.is_available() is false ({torch.__version__})")
        agent, snapshot_revision, precision = load_and_configure_laya_agent(model_path, device, torch, laya)
        result.update({"model_revision": snapshot_revision,
                       "laya_source_revision": _git_revision(Path(laya.__file__)),
                       "laya_code_sha256": _code_digest(Path(laya.__file__)),
                       "sdk_revision": getattr(laya, "__version__", None) or package_version("laya"),
                       "precision": precision, "cuda": torch.version.cuda,
                       "gpu": torch.cuda.get_device_name(device) if device.startswith("cuda") else None,
                       "memory": {**host_memory_bytes(), "cuda_total_bytes": torch.cuda.get_device_properties(device).total_memory if device.startswith("cuda") else None},
                       "actual_torch_device": str(agent.device), "observed_batch_sizes_by_wave": {},
                       "forward_count_by_wave": {}, "supports_batch": False,
                       "supports_scores": False, "max_len_support": False,
                       "head_max_len_support": False, "score_keysets_match": False})
        if device.startswith("cuda") and not str(agent.device).startswith("cuda"):
            raise RuntimeError(f"requested CUDA but Laya loaded on {agent.device}")

        # These are actual call arguments; the checkpoint defaults are captured separately.
        result["requested_context_limit"] = EXPECTED_CONTEXT
        result["requested_head_limit"] = EXPECTED_HEAD
        result["agent_config_limits"] = {"max_len": agent.cfg.get("max_len"), "head_max_len": agent.cfg.get("head_max_len")}
        result["tested_candidate_count"] = len(fixture["waves"]["purchase"]["candidates"])
        result["score_semantics"] = "probabilities"
        result["score_source_precision"] = 4

        warmup_wave = fixture["waves"]["purchase"]
        _sync(device, torch)
        agent.predict_batch([_state_text(fixture)], _questions(warmup_wave), batch_size=1,
                            max_len=EXPECTED_CONTEXT, head_max_len=EXPECTED_HEAD)
        _sync(device, torch)
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(device)

        observed_decimal_places: set[int] = set()
        supports_scores = True
        score_keysets_match = True
        for wave_name, wave in fixture["waves"].items():
            ids = [candidate["candidate_id"] for candidate in wave["candidates"]]
            question = _questions(wave)
            states = [_state_text(fixture, probe_row=i) for i in range(BATCH_ROWS)]
            token_counts = [_token_counts(agent, state, wave) for state in states]
            token_lengths = [row["state_tokens_tokenizer_count"] for row in token_counts]
            if len(set(states)) != BATCH_ROWS or len(set(token_lengths)) < 2:
                raise RuntimeError(f"{wave_name}: probe rows are not distinct and variable-length")
            calls, hook = _instrument_root_forward(agent.model)
            _sync(device, torch)
            start = time.perf_counter()
            try:
                batched = agent.predict_batch(states, question, batch_size=BATCH_ROWS,
                                              max_len=EXPECTED_CONTEXT, head_max_len=EXPECTED_HEAD)
            finally:
                _sync(device, torch)
                elapsed = time.perf_counter() - start
                hook.remove()
            if len(batched) != BATCH_ROWS:
                raise RuntimeError(f"{wave_name}: Laya returned {len(batched)} rows, expected {BATCH_ROWS}")
            rows = [_probabilities(row, ids) for row in batched]
            for row in batched:
                observed_decimal_places.update(
                    len(str(value).partition(".")[2]) if "." in str(value) else 0
                    for value in row["answers"]["decision"]["probabilities"].values())
            single_rows, single_outputs = [], []
            for state in states:
                single = agent.predict_batch([state], question, batch_size=1,
                                             max_len=EXPECTED_CONTEXT, head_max_len=EXPECTED_HEAD)[0]
                single_outputs.append(single)
                single_rows.append(_probabilities(single, ids))
            parity_errors = [_max_abs_error(rows[i], single_rows[i]) for i in range(BATCH_ROWS)]
            margins = [sorted(row.values(), reverse=True)[:2] for row in rows]
            high_margin_agreement = [
                max(row, key=row.get) == max(single_rows[i], key=single_rows[i].get)
                for i, row in enumerate(rows)
                if len(margins[i]) > 1 and margins[i][0] - margins[i][1] > HIGH_MARGIN
            ]

            isolated_states = list(states)
            isolated_states[0] = _state_text(fixture, probe_row=0, perturbation="changed-row0-" + ("padding " * 48))
            before_ids = _token_ids(agent.tok, states[0])
            after_ids = _token_ids(agent.tok, isolated_states[0])
            if before_ids == after_ids or len(before_ids) == len(after_ids):
                raise RuntimeError(f"{wave_name}: row-0 perturbation did not change encoded input length/content")
            isolated = agent.predict_batch(isolated_states, question, batch_size=BATCH_ROWS,
                                           max_len=EXPECTED_CONTEXT, head_max_len=EXPECTED_HEAD)
            isolation_errors = [_max_abs_error(rows[i], _probabilities(isolated[i], ids))
                                for i in range(1, BATCH_ROWS)]
            shapes = [entry["input_shape"] for entry in calls]
            row_sizes = [shape[0] for shape in shapes if shape]
            result["observed_batch_sizes_by_wave"][wave_name] = row_sizes
            result["forward_count_by_wave"][wave_name] = len(calls)
            raw["waves"][wave_name] = {
                "batch_raw_outputs": sanitize(batched), "single_raw_outputs": sanitize(single_outputs),
                "perturbed_row_raw_outputs": sanitize(isolated), "instrumented_forward_calls": calls,
                "measured_batch_seconds": elapsed,
                "batch_vs_single_max_abs_probability_error": max(parity_errors),
                "high_margin_argmax_agreement": (sum(high_margin_agreement) / len(high_margin_agreement)) if high_margin_agreement else None,
                "cross_row_isolation_max_abs_probability_error": max(isolation_errors),
                "usage": [sanitize(_validate_usage(row)) for row in batched],
                "input_token_counts": token_counts, "perturbation_changed_input": True,
                "perturbation_input_tokens_before": len(before_ids),
                "perturbation_input_tokens_after": len(after_ids),
                "perturbation_input_ids_sha256_before": hashlib.sha256(json.dumps(before_ids).encode("ascii")).hexdigest(),
                "perturbation_input_ids_sha256_after": hashlib.sha256(json.dumps(after_ids).encode("ascii")).hexdigest(),
                "request_kwargs": {"max_len": EXPECTED_CONTEXT, "head_max_len": EXPECTED_HEAD,
                                   "agent_config_defaults": result["agent_config_limits"]},
                "effective_token_limits_verified": {"context_limit": EXPECTED_CONTEXT,
                    "head_limit": EXPECTED_HEAD, "max_state_tokens": max(token_lengths),
                    "max_question_tokens": max(item["question_tokens_tokenizer_count"] for item in token_counts),
                    "max_candidate_label_tokens": max(item["max_candidate_label_tokens"] for item in token_counts)},
            }
            if not all(len(_probabilities(row, ids)) == len(ids) for row in batched + single_outputs):
                supports_scores = False
            if any(set(row["answers"]["decision"]["probabilities"]) != set(ids) for row in batched + single_outputs):
                score_keysets_match = False

        result["max_probability_error"] = max(w["batch_vs_single_max_abs_probability_error"] for w in raw["waves"].values())
        agreements = [w["high_margin_argmax_agreement"] for w in raw["waves"].values()]
        result["argmax_agreement"] = min(agreements) if agreements and all(value is not None for value in agreements) else None
        result["cross_row_isolation"] = max(w["cross_row_isolation_max_abs_probability_error"] for w in raw["waves"].values())
        result["supports_scores"] = supports_scores
        result["score_keysets_match"] = score_keysets_match
        result["supports_batch"] = all(any(size >= 2 for size in sizes) for sizes in result["observed_batch_sizes_by_wave"].values())
        result["context_limit"] = EXPECTED_CONTEXT
        result["max_len_support"] = all(w["request_kwargs"]["max_len"] == result["context_limit"] for w in raw["waves"].values())
        result["head_max_len_support"] = all(w["request_kwargs"]["head_max_len"] == EXPECTED_HEAD for w in raw["waves"].values())
        result["score_resolution_decimal_places_observed"] = sorted(observed_decimal_places)
        if device.startswith("cuda"):
            result["memory"]["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        result["cost_sample"] = {"currency": None, "usage_by_wave": {name: data["usage"] for name, data in raw["waves"].items()}}
        passed, failures = evaluate_gate(result, raw)
        result["gate_failures"] = failures
        result["status"] = "PASS" if passed else "FAIL"
        if not passed:
            result["supports_batch"] = False
            result["supports_scores"] = False
            result["score_keysets_match"] = False
            result["max_len_support"] = False
            result["head_max_len_support"] = False
    except Exception as exc:
        result["supports_batch"] = False
        result["supports_scores"] = False
        result["score_keysets_match"] = False
        result["max_len_support"] = False
        result["head_max_len_support"] = False
        result["status"] = "FAIL"
        result["error"] = f"{type(exc).__name__}: {exc}"
        raw["error"] = result["error"]
    write_json(output_dir / "laya-result.json", sanitize(result))
    write_json(output_dir / "laya-raw.json", sanitize(raw))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, type=Path, help="resolved immutable HF snapshot directory or pinned local_dir")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    result = run_probe(args.model_path, args.device, args.output_dir)
    print(json.dumps(sanitize(result), indent=2, sort_keys=True))
    return 0 if result.get("status") == "PASS" else 1

if __name__ == "__main__":
    raise SystemExit(main())

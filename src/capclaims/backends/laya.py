"""Batched Laya inference in one persistent, isolated model worker process."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import multiprocessing
import re
import time
import uuid
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

from capclaims.engine.protocol import DecisionReply, DecisionRequest
from capclaims.randomness import canonical_json
from capclaims.paper.interface import instruction as paper_instruction, validate_interface

EXPECTED_REVISION = "1720e3e3357cfe1e281542e223f8273b0890ca34"
EXPECTED_LAYA_VERSION = "0.4.13"
_WORKER: dict[str, Any] = {}


def release_unused_cuda_cache(torch: Any, device: str) -> None:
    """Bound idle allocator blocks across variable-length batches on desktop GPUs.

    This releases only unused cached blocks, never model weights or live tensors.
    Without this bound, many batch shapes exhausted dedicated Windows VRAM even
    though live peak allocation stayed below 2.4 GB.
    """
    if device.startswith("cuda"):
        unused = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
        if unused > 512 * 1024**2:
            with torch.cuda.device(device):
                torch.cuda.empty_cache()


def normalize_laya_probabilities(raw: Mapping[str, float], candidate_ids: Sequence[str]) -> dict[str, float]:
    if len(candidate_ids) != len(set(candidate_ids)) or set(raw) != set(candidate_ids):
        raise ValueError("Laya candidate probability keys do not match the requested candidate set")
    values: dict[str, float] = {}
    for key in candidate_ids:
        value = raw[key]
        if isinstance(value, bool):
            raise ValueError("Laya probabilities must be finite nonnegative numbers")
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("Laya probabilities must be finite nonnegative numbers") from exc
        if not math.isfinite(number) or number < 0:
            raise ValueError("Laya probabilities must be finite nonnegative numbers")
        if round(number, 4) != number:
            raise ValueError("Laya public probabilities exceed verified four-decimal precision")
        values[key] = number
    total = math.fsum(values.values())
    if total <= 0:
        raise ValueError("Laya probability sum must be positive")
    if abs(total - 1.0) > 1e-6:
        bound = len(values) * 0.00005 + 1e-12
        if abs(total - 1.0) > bound:
            raise ValueError(f"Laya probability sum {total} exceeds four-decimal rounding bound {bound}")
        values = {key: value / total for key, value in values.items()}
    return values


def validate_laya_usage(usage: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(usage.get("truncated"), bool):
        raise ValueError("Laya usage.truncated must be an explicit boolean")
    required = ("input_tokens", "output_tokens", "state_tokens", "state_tokens_dropped")
    for name in required:
        value = usage.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"Laya usage.{name} must be an explicit nonnegative integer")
    if usage["input_tokens"] <= 0 or usage["state_tokens"] <= 0:
        raise ValueError("Laya token usage must include positive input and state token counts")
    if usage["truncated"] or usage["state_tokens_dropped"] != 0:
        raise ValueError("Laya truncated or dropped state tokens; refusing reply")
    truncated_questions = usage.get("truncated_questions")
    if not isinstance(truncated_questions, list) or truncated_questions:
        raise ValueError("Laya usage.truncated_questions must be an explicit empty list")
    if usage.get("options"):
        raise ValueError("Laya collapsed one or more candidate options")
    return dict(usage)


def _revision_for(path: Path) -> str:
    if re.fullmatch(r"[0-9a-fA-F]{40}", path.name):
        revision = path.name.lower()
    else:
        metadata_files = sorted((path / ".cache" / "huggingface" / "download").rglob("*.metadata"))
        revisions = set()
        for metadata in metadata_files:
            lines = metadata.read_text(encoding="utf-8").splitlines()
            if lines and re.fullmatch(r"[0-9a-fA-F]{40}", lines[0].strip()):
                revisions.add(lines[0].strip().lower())
        if len(revisions) != 1:
            raise RuntimeError(f"model directory does not prove one immutable Laya revision: {sorted(revisions)}")
        revision = revisions.pop()
    if revision != EXPECTED_REVISION:
        raise RuntimeError(f"Laya checkpoint revision {revision} differs from pinned {EXPECTED_REVISION}")
    return revision


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    result = tokenizer(text, add_special_tokens=False)
    ids = result["input_ids"] if isinstance(result, dict) else result.input_ids
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(item) for item in ids]


def _preflight_laya_question(agent: Any, state: str, question: dict,
                            labels: Sequence[str]) -> None:
    """Prove the complete instruction, options, and state survive Laya assembly."""
    from laya.common import build_sequence, render_options

    q = {"t": "choice", "ins": question["decision"]["instructions"],
         "crit": question["decision"]["criteria"]}
    rendered_options = render_options(q)
    sequence, markers, head_stats, state_stats = build_sequence(
        agent.tok, state, q, max_len=1024, head_max_len=256,
        return_stats=True, return_truncation_stats=True)
    full_instruction = _token_ids(agent.tok, "choice question: " + q["ins"])
    if sequence[1:1 + len(full_instruction)] != full_instruction:
        raise ValueError("Laya choice head would truncate the economic instruction")
    if (head_stats.get("options") != len(labels) or
            head_stats.get("options_distinct") != len(labels) or
            head_stats.get("tokens_per_option") is not None):
        raise ValueError("Laya choice head would collapse or cap economic candidate labels")
    if state_stats.get("state_tokens_dropped") != 0:
        raise ValueError("Laya choice assembly would truncate the economic observation")
    if any(len(_token_ids(agent.tok, " " + option)) > 48 for option in rendered_options):
        raise ValueError("Laya choice candidate label exceeds its per-option token cap")


def production_instruction(stage: str, interface_version: str = "legacy_v1") -> str:
    """Public production wording helper shared with diagnostic probes."""
    if stage.startswith("paper:"):
        return paper_instruction(stage, interface_version)
    if interface_version != "legacy_v1":
        raise ValueError("economic_v2 instructions are available only for paper stages")
    return {
        "purchase": "Choose an initial capacity quantity to maximize expected profit. Demand is uniform from 0 to 200; wholesale price and your sale value are shown. Choose only among the listed quantities.",
        "trade_side": "Choose BUY, SELL, or HOLD based only on your shown inventory and demand. BUY and SELL are limits in a single uniform-price market; choose HOLD when no listed action is preferable.",
        "trade_qty": "Choose the quantity for your already selected trade side. Choose only among the legal listed quantities and never exceed your shown capacity.",
        "trade_price": "Choose a limit price in dollars for your already selected trade side and quantity. Choose only among the listed prices.",
    }[stage]


def _instruction(stage: str, interface_version: str = "legacy_v1") -> str:
    return production_instruction(stage, interface_version)


def _candidate_signature(candidates: Sequence[Any]) -> str:
    """Identity of a choice head includes IDs, values and displayed labels."""
    return canonical_json([candidate.__dict__ for candidate in candidates])


def _worker_init(model_path: str, device: str) -> None:
    import os
    if os.environ.get("CAPCLAIMS_DIAGNOSTICS") == "1":
        import faulthandler
        faulthandler.dump_traceback_later(30, repeat=True)
    import laya
    import torch

    path = Path(model_path)
    revision = _revision_for(path)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable in the Laya worker")
    agent = laya.load(str(path), device=device, revision=revision, backend="eager")
    agent.amp_enabled = False
    agent.dtype = torch.float32
    agent.model.to(device=agent.device, dtype=torch.float32)
    agent.model.eval()
    if device.startswith("cuda") and not str(agent.device).startswith("cuda"):
        raise RuntimeError(f"Laya loaded on {agent.device} after CUDA was requested")
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("highest")
        if torch.get_float32_matmul_precision() != "highest":
            raise RuntimeError("FP32 matmul precision is not highest")
    for owner, attr in ((getattr(torch.backends, "cuda", None), "matmul"),
                        (getattr(torch.backends, "cudnn", None), None)):
        target = getattr(owner, attr, None) if owner is not None and attr else owner
        if target is not None and hasattr(target, "allow_tf32"):
            target.allow_tf32 = False
    parameter_dtypes = {str(parameter.dtype) for parameter in agent.model.parameters()}
    if parameter_dtypes and parameter_dtypes != {"torch.float32"}:
        raise RuntimeError(f"Laya model parameters are not FP32: {sorted(parameter_dtypes)}")
    if agent.dtype_for(170) != torch.float32:
        raise RuntimeError("Laya dtype_for(batch rows) is not FP32")
    if getattr(laya, "__version__", EXPECTED_LAYA_VERSION) != EXPECTED_LAYA_VERSION:
        raise RuntimeError(f"Laya SDK version differs from pinned {EXPECTED_LAYA_VERSION}")
    _WORKER.update({"agent": agent, "torch": torch, "revision": revision, "device": device,
                    "forward_batch_sizes": []})

    def record(_module: Any, inputs: Any, _output: Any) -> None:
        tensor = next((value for value in inputs if hasattr(value, "shape")), None)
        if tensor is None or len(tensor.shape) == 0:
            raise RuntimeError("Laya forward hook did not observe a batched tensor")
        _WORKER["forward_batch_sizes"].append(int(tensor.shape[0]))

    _WORKER["hook"] = agent.model.register_forward_hook(record)


def _predict_group(stage: str, candidates: list[dict[str, Any]], states: list[str]) -> dict[str, Any]:
    import os
    if os.environ.get("CAPCLAIMS_DIAGNOSTICS") == "1":
        import sys
        print(f"LAYA start stage={stage} rows={len(states)}", file=sys.stderr, flush=True)
    agent, torch, device = _WORKER["agent"], _WORKER["torch"], _WORKER["device"]
    interface_versions = set()
    for state in states:
        try:
            observation = json.loads(state)
        except json.JSONDecodeError as exc:
            raise ValueError("Laya paper state is not canonical observation JSON") from exc
        version = observation.get("decision_interface", "legacy_v1")
        interface_versions.add(validate_interface(version))
    if len(interface_versions) != 1:
        raise ValueError("Laya inference group mixes decision interface versions")
    interface_version = interface_versions.pop()
    candidate_ids = [candidate["candidate_id"] for candidate in candidates]
    question = {"decision": {"type": "choice", "instructions": _instruction(stage, interface_version),
                              "criteria": {c["candidate_id"]: c["label"] for c in candidates}}}
    if interface_version == "economic_v2":
        for state in states:
            _preflight_laya_question(agent, state, question,
                                     [candidate["label"] for candidate in candidates])
    for state in states:
        if len(_token_ids(agent.tok, state)) > 1024:
            raise ValueError("Laya observation exceeds 1024 context tokens; refusing to truncate")
    instruction_tokens = len(_token_ids(agent.tok, _instruction(stage, interface_version)))
    labels = [len(_token_ids(agent.tok, c["label"])) for c in candidates]
    if instruction_tokens > 256 or max(labels, default=0) > 256:
        raise ValueError("Laya question or candidate label exceeds 256 head tokens")
    _WORKER["forward_batch_sizes"].clear()
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    raw_rows = agent.predict_batch(states, question, batch_size=len(states), max_len=1024, head_max_len=256)
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    if len(raw_rows) != len(states):
        raise RuntimeError(f"Laya returned {len(raw_rows)} rows for a group of {len(states)}")
    observed = list(_WORKER["forward_batch_sizes"])
    if not observed or any(size != len(states) for size in observed):
        raise RuntimeError(f"Laya forward batch sizes {observed} differ from submitted group size {len(states)}")
    rows = []
    for result in raw_rows:
        answer = (result.get("answers") or {}).get("decision")
        if not isinstance(answer, dict) or answer.get("type") != "choice" or not isinstance(answer.get("probabilities"), Mapping):
            raise ValueError("Laya did not return choice probabilities")
        usage = validate_laya_usage(result.get("usage") or {})
        probs = normalize_laya_probabilities(answer["probabilities"], candidate_ids)
        rows.append({"probabilities": probs, "raw_scores": dict(answer["probabilities"]), "usage": usage})
    memory = torch.cuda.max_memory_allocated(device) if device.startswith("cuda") else None
    reserved_peak = torch.cuda.max_memory_reserved(device) if device.startswith("cuda") else None
    release_unused_cuda_cache(torch, device)
    if os.environ.get("CAPCLAIMS_DIAGNOSTICS") == "1":
        import sys
        reserved = torch.cuda.memory_reserved(device) if device.startswith("cuda") else 0
        print(f"LAYA done stage={stage} seconds={elapsed:.3f} allocated_peak={memory} reserved={reserved}", file=sys.stderr, flush=True)
    return {"rows": rows, "forward_batch_sizes": observed, "group_size": len(states),
            "elapsed_seconds": elapsed, "peak_cuda_bytes": memory,
            "peak_cuda_reserved_bytes": reserved_peak, "revision": _WORKER["revision"],
            "sdk_revision": EXPECTED_LAYA_VERSION, "device": str(agent.device), "dtype": "float32"}


class LayaBackend:
    def __init__(self, model_path: str | Path = ".cache/models/laya-multilingual", device: str = "cuda") -> None:
        self.model_path = Path(model_path).resolve()
        self.device = device
        if not self.model_path.is_dir():
            raise FileNotFoundError(f"Laya model directory does not exist: {self.model_path}")
        self.revision = _revision_for(self.model_path)
        self._executor: ProcessPoolExecutor | None = None
        self._closed = False
        self._worker_busy = False

    def _get_executor(self) -> ProcessPoolExecutor:
        if self._executor is None:
            context = multiprocessing.get_context("spawn")
            self._executor = ProcessPoolExecutor(
                max_workers=1, mp_context=context,
                initializer=_worker_init, initargs=(str(self.model_path), self.device),
            )
        return self._executor

    async def decide_many(self, requests: Sequence[DecisionRequest]) -> list[DecisionReply]:
        if self._closed:
            raise RuntimeError("Laya backend is closed")
        if not requests:
            return []
        groups: dict[tuple[str, str, str], list[tuple[int, DecisionRequest]]] = {}
        for index, request in enumerate(requests):
            signature = _candidate_signature(request.candidates)
            version = validate_interface(request.observation.get("decision_interface", "legacy_v1"))
            groups.setdefault((request.stage, version, signature), []).append((index, request))
        replies: list[DecisionReply | None] = [None] * len(requests)
        loop = asyncio.get_running_loop()
        for (stage, version, _), rows in groups.items():
            candidates = [candidate.__dict__ for candidate in rows[0][1].candidates]
            states = [canonical_json(request.observation) for _, request in rows]
            group_id = uuid.uuid4().hex
            try:
                result = await loop.run_in_executor(self._get_executor(), _predict_group, stage, candidates, states)
            except asyncio.CancelledError:
                self._worker_busy = True
                raise
            for (index, request), row in zip(rows, result["rows"], strict=True):
                replies[index] = DecisionReply.from_request(
                    request, row["probabilities"], row["raw_scores"], backend="laya",
                    model_revision=result["revision"], usage=row["usage"],
                    timing={"group_id": group_id,
                            "peak_cuda_reserved_bytes": result.get("peak_cuda_reserved_bytes"), **{
                        k: result[k] for k in ("forward_batch_sizes", "group_size", "elapsed_seconds", "peak_cuda_bytes", "device", "dtype", "sdk_revision")
                    }},
                )
        if any(reply is None for reply in replies):
            raise RuntimeError("Laya failed to produce one reply per request")
        return [reply for reply in replies if reply is not None]

    async def aclose(self) -> None:
        self._closed = True
        if self._executor is not None:
            executor, self._executor = self._executor, None
            if not self._worker_busy:
                await asyncio.to_thread(executor.shutdown, wait=True, cancel_futures=True)
            else:
                processes = list((executor._processes or {}).values())
                executor.shutdown(wait=False, cancel_futures=True)
                for process in processes:
                    if process.is_alive():
                        process.terminate()
                await asyncio.to_thread(_join_processes, processes)


def _join_processes(processes: list[Any]) -> None:
    for process in processes:
        process.join(timeout=2.0)

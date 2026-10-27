"""Immutable local preview lifecycle. This module never reads credentials or calls a provider."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from . import VERSION
from .design import ARMS, SEED, build_prompt, build_schedule
from .memory import compile_behavioral_memory

MAX_BODY_BYTES = 12_000
MAX_REQUESTS = 3_888
ROOT = Path(__file__).resolve().parents[3]
CODE_FILES = (
    "src/capclaims/behavioral_memory/__init__.py",
    "src/capclaims/behavioral_memory/design.py",
    "src/capclaims/behavioral_memory/lifecycle.py",
    "src/capclaims/behavioral_memory/memory.py",
    "src/capclaims/behavioral_memory/projection.py",
    "src/capclaims/behavioral_memory/sampling.py",
    "src/capclaims/behavioral_memory/runner.py",
    "src/capclaims/behavioral_memory/transport.py",
    "src/capclaims/behavioral_memory/wire_preview.py",
    "src/capclaims/behavioral_memory/cli.py",
    "src/capclaims/behavioral_memory/analysis.py",
    "src/capclaims/behavioral_memory/source.py",
    "src/capclaims/conditional_behavior/protocol.py",
    "src/capclaims/conditional_behavior/source.py",
    "src/capclaims/conditional_behavior/prompts.py",
    "src/capclaims/conditional_behavior/transport.py",
    "src/capclaims/conditional_behavior/scoring.py",
    "src/capclaims/personalization_xlsx.py",
    "src/capclaims/continuation_study.py",
    "src/capclaims/order_presentation.py",
    "src/capclaims/paper/interface.py",
    "src/capclaims/paper/codec.py",
    "src/capclaims/domain/types.py",
    "src/capclaims/backends/jev.py",
    "src/capclaims/backends/jev_instructions.py",
    "docs/engineering/behavioral-memory-cost-gate.md",
    "docs/research/jev-behavioral-memory-design-2026-10-26.md",
    "tools/probes/jev_behavioral_memory.py",
    "pyproject.toml",
)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _code_hashes() -> dict[str, str]:
    result = {}
    for relative in CODE_FILES:
        path = ROOT / relative
        if not path.is_file():
            raise ValueError(f"preview dependency is missing: {relative}")
        result[relative] = _sha(path.read_bytes())
    return result


def _verify_source_paths(metadata: dict[str, Any]) -> None:
    paths = metadata.get("source_paths")
    if not isinstance(paths, dict):
        return
    archive = Path(paths["archive"])
    policy = Path(paths["policy"])
    if _sha(archive.read_bytes()) != metadata.get("source_archive_sha256"):
        raise ValueError("source archive changed after authorized validation")
    from capclaims.personalization_xlsx import _inner_xlsx_bytes
    if _sha(_inner_xlsx_bytes(archive)) != metadata.get("source_sha256"):
        raise ValueError("source workbook bytes changed after authorized validation")
    if _sha(policy.read_bytes()) != metadata.get("source_policy_sha256"):
        raise ValueError("source-policy file changed after authorized validation")
    manifests = paths.get("exposure_registries")
    expected = metadata.get("exposure_registry_sha256")
    if not isinstance(manifests, dict) or not isinstance(expected, dict) or set(manifests) != set(expected):
        raise ValueError("exposure registry paths and hashes are incomplete")
    for name, value in manifests.items():
        if _sha(Path(value).read_bytes()) != expected[name]:
            raise ValueError("exposure registry changed after authorized validation")


def _write_once(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _validate_case(case: dict[str, Any]) -> tuple[str, list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    person = case.get("person_id")
    if not isinstance(person, str) or not person:
        raise ValueError("each development case needs a local person id")
    history = case.get("history")
    state = case.get("current_state")
    label = case.get("label")
    if not isinstance(history, list) or not isinstance(state, dict) or not isinstance(label, dict):
        raise ValueError("case requires history, current_state and separate scorer label")
    if [row.get("round") for row in history] != list(range(6, 13)):
        raise ValueError("each target history must contain formal rounds 6 through 12 in order")
    if state.get("round") != 13 or type(state.get("w")) is not int:
        raise ValueError("target state must be formal round 13 with verified current w")
    if set(label) != {"orders", "profit_cents"}:
        raise ValueError("target scorer label must contain only verified orders and profit_cents")
    if type(label["profit_cents"]) is not int or not isinstance(label["orders"], list):
        raise ValueError("invalid target scorer label")
    return person, history, state, label


def freeze_preview(cases: list[dict[str, Any]], metadata: dict[str, Any], output_dir: str | Path,
                   *, seed: int = SEED,
                   exact_wire_previews: dict[str, Any] | None = None) -> dict[str, Any]:
    """Freeze study inputs and isolated labels before any provider key can be loaded."""
    if len(cases) != 18 or len({case.get("person_id") for case in cases}) != 18:
        raise ValueError("v1 preview requires exactly 18 unique exposed development people")
    if not isinstance(metadata, dict) or metadata.get("source_policy_verified") is not True:
        raise ValueError("authorized source-policy verification is required")
    _verify_source_paths(metadata)
    people = [case["person_id"] for case in cases]
    case_by_person = {}
    labels = {}
    for case in cases:
        person, history, state, label = _validate_case(case)
        case_by_person[person] = (history, state)
        labels[person] = {"round": 13, **label}
    schedule = build_schedule(people, seed=seed)
    if exact_wire_previews is not None and set(exact_wire_previews) != {row["request_prefix"] for row in schedule}:
        raise ValueError("exact wire preview is incomplete for the frozen schedule")
    frozen = []
    prompt_hashes: dict[str, str] = {}
    max_body = 0
    for trajectory in schedule:
        person = trajectory["person_id"]
        history, state = case_by_person[person]
        memory, audit = compile_behavioral_memory(history, state, 12)
        memory_for_arm = memory if trajectory["arm"] in {"M1", "M2"} else None
        prompt = build_prompt(arm=trajectory["arm"], current_state=state, history=memory["history"],
                              memory=memory_for_arm)
        # This canonical envelope is a conservative offline bound. The transport
        # layer must still check the SDK's exact serialized body immediately before send.
        prompt_data = json.loads(prompt)
        request = {"model": "jev-1.14.0", "state": prompt_data["observation"],
                   "instructions": prompt_data["system_prompt"],
                   "criteria": {f"digit{n}": f"units: {n}; full feasible interval 1..999" for n in range(10)}}
        body_n = len(_canonical(request))
        wire_preview = None
        if exact_wire_previews is not None:
            wire_preview = exact_wire_previews[trajectory["request_prefix"]]
            body_n = max(wire_preview["first_count"]["body_bytes"],
                         wire_preview["maximum_legal_prefix"]["body_bytes"])
        max_body = max(max_body, body_n)
        if body_n > MAX_BODY_BYTES:
            raise ValueError("frozen prompt exceeds the 12,000-byte request bound")
        prompt_hashes[trajectory["request_prefix"]] = _sha(prompt.encode("utf-8"))
        frozen.append({**trajectory, "current_state": {k: state[k] for k in ("round", "p", "w", "Q", "D")},
                       "history": memory["wire_history"], "prompt": prompt,
                       "memory": memory if trajectory["arm"] != "M0" else None,
                       "audit": audit, "request_body_bytes_bound": body_n,
                       "wire_preview": wire_preview})
    if len(frozen) * 36 > MAX_REQUESTS:
        raise ValueError("planned request count exceeds the 3,888-call cap")
    labels_bytes = _canonical({"schema": "behavioral-memory-scorer-labels-v1", "labels": labels}) + b"\n"
    candidates = {"order_count": [f"N{n}" for n in range(6)], "side": ["BUY", "SELL"],
        "digits": [f"digit{n}" for n in range(10)]}
    manifest = {"schema": VERSION, "status": "FROZEN", "seed": seed, "model": "jev-1.14.0",
        "arms": list(ARMS), "target_round": 13, "history_rounds": [6, 12],
        "people_n": 18, "trajectory_n": len(frozen), "request_cap": MAX_REQUESTS,
        "calls_per_trajectory_cap": 36, "maximum_request_body_bytes": MAX_BODY_BYTES,
        "wire_preview_mode": "SDK_EXACT" if exact_wire_previews is not None else "CANONICAL_ESTIMATE",
        "frozen_body_bound_bytes": max_body, "metadata": metadata,
        "code_sha256": _code_hashes(), "candidate_order": candidates,
        "candidate_order_sha256": _sha(_canonical(candidates)),
        "scoring_labels_sha256": _sha(labels_bytes),
        "prompt_sha256": prompt_hashes, "schedule": frozen}
    manifest_bytes = _canonical(manifest) + b"\n"
    manifest_hash = _sha(manifest_bytes)
    root = Path(output_dir)
    _write_once(root / "preview.json", manifest_bytes)
    _write_once(root / "scoring-labels.json", labels_bytes)
    _write_once(root / "manifest.sha256", (manifest_hash + "\n").encode())
    return {"status": "FROZEN", "preview_sha256": manifest_hash, "trajectory_n": len(frozen),
            "request_cap": MAX_REQUESTS, "frozen_body_bound_bytes": max_body}


def verify_preview(preview_dir: str | Path, reviewed_sha256: str) -> dict[str, Any]:
    root = Path(preview_dir)
    raw = (root / "preview.json").read_bytes()
    expected = (root / "manifest.sha256").read_text(encoding="ascii").strip()
    actual = _sha(raw)
    if actual != expected or actual != reviewed_sha256:
        raise ValueError("preview digest does not match the reviewed frozen artifact")
    manifest = json.loads(raw.decode("utf-8"))
    if manifest.get("schema") != VERSION or manifest.get("trajectory_n") != 108:
        raise ValueError("unsupported or incomplete behavioral-memory preview")
    if manifest.get("code_sha256") != _code_hashes():
        raise ValueError("implementation dependency changed after preview freeze")
    _verify_source_paths(manifest["metadata"])
    if manifest.get("candidate_order_sha256") != _sha(_canonical(manifest.get("candidate_order"))):
        raise ValueError("frozen candidate order changed")
    if manifest.get("wire_preview_mode") == "SDK_EXACT":
        for trajectory in manifest["schedule"]:
            previews = trajectory.get("wire_preview")
            if not isinstance(previews, dict) or set(previews) != {"first_count", "maximum_legal_prefix"}:
                raise ValueError("SDK-exact request previews are incomplete")
            for item in previews.values():
                body = __import__("base64").b64decode(item["body_base64"], validate=True)
                if (len(body) != item["body_bytes"] or len(body) > MAX_BODY_BYTES
                        or _sha(body) != item["body_sha256"]):
                    raise ValueError("SDK-exact request preview changed or exceeds the body cap")
    labels = json.loads((root / "scoring-labels.json").read_text(encoding="utf-8"))
    if labels.get("schema") != "behavioral-memory-scorer-labels-v1" or len(labels.get("labels", {})) != 18:
        raise ValueError("separate scorer labels are missing or malformed")
    if _sha(_canonical(labels) + b"\n") != manifest.get("scoring_labels_sha256"):
        raise ValueError("separate scorer label artifact changed after preview freeze")
    return {**manifest, "status": "PREVIEW_VALID", "preview_sha256": actual}

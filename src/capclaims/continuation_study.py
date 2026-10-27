"""Frozen provider-choice continuation study lifecycle (additive to legacy studies)."""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from capclaims.backends.jev import MODEL, JevBackend, estimate_input_tokens, load_local_env
from capclaims.backends.jev_instructions import build_instruction
from capclaims.engine.protocol import DecisionReply
from capclaims.injection_context import render_state
from capclaims.injection_study import InMemoryWireRecorder
from capclaims.portfolio_prediction import _validate_base
from capclaims.portfolio_transport import (
    DEFAULT_URL,
    MAX_BODY_BYTES,
    ExpectedWire,
    GuardTransport,
    SharedBudget,
    _url_tuple,
)
from capclaims.provider_choice_trace import (
    advance_provider_trace,
    build_provider_order_request,
    next_provider_trace_action,
    replay_provider_trace,
)
from capclaims.randomness import canonical_json

SEED = 20261025
PREVIEW_SHA = "e42139519d69bbfa4b84fd7d79d410791251ea3580ebc472ca6fd36c159f6434"
JOINT_APPEND = (
    "partial_orders contains your own earlier predictions in this same forecast, not newly observed actions from the participant. "
    "These orders form one joint submitted portfolio. A new BUY or SELL adds an order; it does not replace an earlier order. "
    "STOP means you predict that the participant submits no additional order."
)
ARMS = ("C_base", "C_joint", "E_base", "E_joint")
MAX_SESSIONS = 24
MAX_REQUESTS = 840
MAX_RESERVED_BYTES = 6_300_000
MAX_BODY_BYTES = 7_500
MAX_WALL_SECONDS = 600
MAX_IN_FLIGHT = 10
MAX_RPS = 40
MAX_BYTES_PER_SECOND = 50_000
_ROOT = Path(__file__).resolve().parents[2]
SOURCE_POLICY_PATH = Path("artifacts/participant-data/quote-perspective-24/source-policy-authorized.json")

def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha(value: Any) -> str:
    return _sha_bytes(canonical_json(value).encode("utf-8"))


def _write(path: Path, value: Any) -> None:
    """Create an exclusive JSON artifact and flush its bytes to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write((canonical_json(value) + "\n").encode("utf-8"))
        handle.flush()
        os.fsync(handle.fileno())


def _private(path: str | Path) -> Path:
    """Keep all person-level study artifacts inside the ignored private root."""
    resolved = Path(path).resolve()
    private_root = (_ROOT / "artifacts" / "participant-data").resolve()
    if resolved != private_root and private_root not in resolved.parents:
        raise ValueError("continuation artifacts must be under ignored participant-data")
    return resolved


def _source_policy_path() -> Path:
    """Return the previously authorized source-policy file used by the parent studies."""
    return (_ROOT / SOURCE_POLICY_PATH).resolve()


def _check_source_policy(expected_sha256: str, expected_source: Mapping[str, Any]) -> dict[str, Any]:
    """Recheck the source permission artifact before any credential or provider access."""
    raw = _source_policy_path().read_bytes()
    if _sha_bytes(raw) != expected_sha256:
        raise ValueError("source-policy file no longer matches the frozen preview")
    policy = json.loads(raw.decode("utf-8"))
    required = (
        "codebook_verified",
        "intended_use_authorized",
        "provider_transmission_permitted",
    )
    if not all(policy.get(key) is True for key in required):
        raise ValueError("source-policy file does not authorize this study")
    if (policy.get("source_kind") != expected_source.get("source_kind")
            or policy.get("source_sha256") != expected_source.get("source_sha256")
            or expected_source.get("source_kind") != "human"):
        raise ValueError("source-policy identity differs from the frozen human-data source")
    return policy
def _paths() -> dict[str, str]:
    """Hash every legacy dependency plus each new study-owned executable file."""
    from capclaims.injection_study import _code_hashes as parent_code_hashes

    hashes = parent_code_hashes()
    new_paths = (
        "src/capclaims/provider_choice.py",
        "src/capclaims/provider_choice_trace.py",
        "src/capclaims/continuation_study.py",
        "src/capclaims/continuation_analysis.py",
        "src/capclaims/continuation_transport_audit.py",
        "docs/superpowers/specs/2026-10-25-provider-choice-design.md",
        "tools/probes/jev_continuation.py",
    )
    hashes.update({name: _sha_bytes((_ROOT / name).read_bytes()) for name in new_paths})
    return hashes


def _read_parent(parent: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    # Parent preview is fixed by exact hash. Accept only known frozen injection shape.
    from capclaims.injection_study import verify_preview as verify_legacy_preview

    verify_legacy_preview(parent,PREVIEW_SHA)
    manifest = json.loads((parent / "manifest.json").read_text(encoding="utf-8"))
    claimed = manifest.pop("preview_sha256", None)
    if _sha(manifest) != PREVIEW_SHA or claimed != PREVIEW_SHA:
        raise ValueError("parent preview is not the approved injection-preview-root-25-r0")
    manifest["preview_sha256"] = claimed
    inputs = json.loads((parent / "inputs.json").read_text(encoding="utf-8"))
    if _sha(inputs.get("sessions")) != manifest.get("inputs_file_sha256"):
        raise ValueError("parent inference inputs failed their frozen checksum")
    return manifest, inputs


def build_preview(parent_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Freeze the balanced 24-case preview after source, label and body checks."""
    parent = _private(parent_dir)
    output = _private(output_dir)
    parent_manifest, parent_inputs = _read_parent(parent)
    if output.exists():
        raise FileExistsError("preview is immutable; overwrite is forbidden")

    bases: dict[str, dict[str, Any]] = {}
    for parent_session in parent_manifest["sessions"]:
        target_id = parent_session.get("target_id")
        base = parent_session.get("base") or parent_inputs.get("bases", {}).get(target_id)
        if base is None:
            continue
        person_id = parent_session.get("person_id", target_id)
        bases[person_id] = _validate_base(base)
    if len(bases) != 6:
        raise ValueError("approved parent must provide exactly six development-person bases")

    # Keep each person's four variants together, with randomized order inside each block.
    rng = random.Random(SEED)
    people = sorted(bases)
    rng.shuffle(people)
    sessions = []
    for person in people:
        block = [
            {"session_id": f"{person}:{arm}", "person_id": person, "arm": arm}
            for arm in ARMS
        ]
        rng.shuffle(block)
        sessions.extend(block)

    labels_path = parent / "labels.json"
    label_bytes = labels_path.read_bytes()
    labels = json.loads(label_bytes.decode("utf-8"))
    if (_sha(labels) != parent_manifest.get("labels_sha256")
            or _sha_bytes(label_bytes) != parent_manifest.get("labels_file_sha256")):
        raise ValueError("parent scoring labels do not match the frozen preview")
    scoring_targets = {person: labels[person] for person in bases if person in labels}
    if len(scoring_targets) != 6:
        raise ValueError("parent held-out labels do not cover the six frozen people")

    source = parent_manifest.get("source", {})
    required_permissions = (
        "codebook_verified",
        "intended_use_authorized",
        "provider_transmission_permitted",
    )
    source_verified = all(source.get(name) is True for name in required_permissions)
    if not source_verified:
        raise ValueError("parent source-policy attestations do not authorize this study")
    _check_source_policy(parent_manifest.get("source_policy_sha256", ""), source)

    from capclaims.injection_study import _maximal_legal_requests

    maximum_body = 0
    for base in bases.values():
        for request in _maximal_legal_requests(base["history"]):
            for arm in ARMS:
                wire = asyncio.run(_serialize(request, arm.endswith("joint"), arm))
                maximum_body = max(maximum_body, len(wire.body))
    if maximum_body > MAX_BODY_BYTES:
        raise ValueError("widest continuation SDK request exceeds 7500-byte cap")

    inputs = {"bases": bases, "sessions": sessions}
    output.mkdir(parents=True, exist_ok=False)
    _write(output / "inputs.json", inputs)
    _write(output / "scoring-targets.json", scoring_targets)
    prompt_texts = _prompt_freeze(inputs)
    manifest = {
        "schema": "jev_provider_choice_preview_v1",
        "parent_preview_sha256": PREVIEW_SHA,
        "source_policy_sha256": parent_manifest.get("source_policy_sha256"),
        "source_policy_path": SOURCE_POLICY_PATH.as_posix(),
        "parent_source": source,
        "parent_source_policy_verified": source_verified,
        "sessions": sessions,
        "session_n": MAX_SESSIONS,
        "schedule_seed": SEED,
        "arms": list(ARMS),
        "model": MODEL,
        "sdk_version": "0.7.5",
        "instruction_profile": "portfolio_third_v1",
        "joint_append": JOINT_APPEND,
        "prompt_text_by_condition": prompt_texts,
        "prompt_sha256_by_condition": {
            name: _sha_bytes(text.encode("utf-8"))
            for name, text in prompt_texts.items()
        },
        "request_cap": MAX_REQUESTS,
        "reserved_bytes_cap": MAX_RESERVED_BYTES,
        "body_bytes_cap": MAX_BODY_BYTES,
        "wall_seconds_cap": MAX_WALL_SECONDS,
        "max_in_flight": MAX_IN_FLIGHT,
        "rps_cap": MAX_RPS,
        "bytes_per_second_cap": MAX_BYTES_PER_SECOND,
        "retry_cap": 0,
        "inputs_sha256": _sha_bytes((output / "inputs.json").read_bytes()),
        "preflight_max_body_bytes": maximum_body,
        "scoring_targets_sha256": _sha_bytes((output / "scoring-targets.json").read_bytes()),
        "code_sha256": _paths(),
    }
    manifest["preview_sha256"] = _sha(manifest)
    _write(output / "manifest.json", manifest)
    return manifest


def verify_preview(preview_dir: str | Path, reviewed_sha256: str) -> dict[str, Any]:
    """Verify frozen inference bytes without opening held-out scoring targets."""
    root = Path(preview_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    digest = manifest.pop("preview_sha256", None)
    if digest != reviewed_sha256 or digest != _sha(manifest):
        raise ValueError("preview hash has not been independently reviewed")
    manifest["preview_sha256"] = digest
    if (manifest.get("code_sha256") != _paths()
            or manifest.get("parent_preview_sha256") != PREVIEW_SHA
            or manifest.get("session_n") != MAX_SESSIONS
            or not manifest.get("parent_source_policy_verified")
            or not manifest.get("source_policy_sha256")):
        raise ValueError("preview inputs or code changed after freeze")
    if _sha_bytes((root / "inputs.json").read_bytes()) != manifest.get("inputs_sha256"):
        raise ValueError("preview inputs were modified")
    inputs = json.loads((root / "inputs.json").read_text(encoding="utf-8"))
    if (inputs.get("sessions") != manifest.get("sessions")
            or len(inputs.get("bases", {})) != 6
            or len(inputs.get("sessions", [])) != MAX_SESSIONS
            or len({row.get("session_id") for row in inputs["sessions"]}) != MAX_SESSIONS):
        raise ValueError("frozen schedule or inference inputs are malformed")
    expected = {(person, arm) for person in inputs["bases"] for arm in ARMS}
    actual = {(row.get("person_id"), row.get("arm")) for row in inputs["sessions"]}
    if actual != expected or len(actual) != MAX_SESSIONS:
        raise ValueError("frozen schedule does not contain one of each arm per person")
    prompt_texts = _prompt_freeze(inputs)
    prompt_hashes = {
        name: _sha_bytes(text.encode("utf-8"))
        for name, text in prompt_texts.items()
    }
    if (manifest.get("prompt_text_by_condition") != prompt_texts
            or manifest.get("prompt_sha256_by_condition") != prompt_hashes):
        raise ValueError("frozen full prompts or prompt hashes differ from the current serializer")
    from importlib.metadata import version

    if version("typesafe-sdk") != "0.7.5" or manifest.get("sdk_version") != "0.7.5":
        raise ValueError("pinned TypeSafe SDK version changed")
    # Held-out scoring targets are intentionally not opened in verify/execute/replay.
    # Their frozen digest is checked only by score_run.
    if (type(manifest.get("preflight_max_body_bytes")) is not int
            or manifest["preflight_max_body_bytes"] > MAX_BODY_BYTES):
        raise ValueError("preview request body preflight is missing or over limit")
    return manifest

def _prompt(request:Any,joint:bool)->str:
    p=build_instruction(request,"portfolio_third_v1")
    return p+("\n\n"+JOINT_APPEND if joint else "")


def _prompt_freeze(inputs: Mapping[str, Any]) -> dict[str, str]:
    """Build the exact baseline and joint text from the first scheduled request."""
    first = inputs["sessions"][0]
    base = inputs["bases"][first["person_id"]]
    request_base = {**base, "agent_id": base["agent_id"] + "__" + first["arm"]}
    request = build_provider_order_request(request_base, ())
    if request is None:
        raise ValueError("first frozen session has no initial order request")
    return {"base": _prompt(request, False), "joint": _prompt(request, True)}

async def _serialize(request:Any,joint:bool,arm:str)->ExpectedWire:
    import httpx2
    from typesafe_sdk import AsyncTypeSafeClient,Choice
    from typesafe_sdk._core.retry import RetryPolicy
    captured={}; ids=[c.candidate_id for c in request.candidates]
    async def capture(req):
        captured.update(method=req.method,url=str(req.url),body=await req.aread())
        probs={k:1/len(ids) for k in ids}
        return httpx2.Response(200,json={"model":MODEL,"usage":{"input_tokens":1,"output_tokens":1},
            "answers":{"decision":{"type":"choice","choice":ids[0],"confidence":1,"probabilities":probs}}})
    http=httpx2.AsyncClient(transport=httpx2.MockTransport(capture),follow_redirects=False)
    client=AsyncTypeSafeClient(api_key="dry-run-dummy",model=MODEL,retry=RetryPolicy(max_retries=0),
        http_client=http,base_url="https://api.typesafe.ai",compression=False)
    try:
        state=render_state(request,"C" if arm.startswith("C") else "E")
        await client.system_one(state=state,questions={"decision":Choice(instructions=_prompt(request,joint),
            criteria={c.candidate_id:c.label for c in request.candidates})},model=MODEL)
    finally: await client.aclose()
    body=captured["body"]
    return ExpectedWire(request.request_id,captured["method"],_url_tuple(captured["url"]),body,_sha_bytes(body))

class _PromptBackend(JevBackend):
    def __init__(self,*a:Any,joint:bool,arm:str,**kw:Any):super().__init__(*a,**kw);self.joint=joint;self.arm=arm
    async def decide_many(self,requests):
        requests=tuple(requests); rendered=[(r,_prompt(r,self.joint),render_state(r,"C" if self.arm.startswith("C") else "E")) for r in requests]
        estimates=[estimate_input_tokens(r,p,s) for r,p,s in rendered]; await self._reserve(requests,estimates)
        import asyncio
        return list(await asyncio.gather(*(self._one(r,e,p,s) for (r,p,s),e in zip(rendered,estimates))))

async def execute_preview(preview_dir:str|Path,run_dir:str|Path,reviewed_preview_sha256:str,fake:bool=False,fake_transport:Any=None)->dict[str,Any]:
    m=verify_preview(preview_dir,reviewed_preview_sha256); out=_private(run_dir)
    if out.exists():raise FileExistsError("run directories cannot be resumed or overwritten")
    out.mkdir(parents=True,exist_ok=False)
    # Durable pre-provider receipt precedes all credential/environment access.
    started_monotonic = time.monotonic()
    started_perf_ns = time.perf_counter_ns()
    _write(out/"receipt.json",{"preview_sha256":m["preview_sha256"],"execution_mode":"mock" if fake else "provider",
        "status":"RUNNING","started_at_utc":datetime.now(timezone.utc).isoformat(),"started_monotonic":started_monotonic,
        "started_perf_ns":started_perf_ns,"source_policy_sha256":m.get("source_policy_sha256")})
    inputs=json.loads((Path(preview_dir)/"inputs.json").read_text(encoding="utf-8")); transport=fake_transport
    if not fake:
        if not m.get("parent_source_policy_verified") or not m.get("source_policy_sha256"):
            raise ValueError("frozen source permissions block provider transmission")
        _check_source_policy(m["source_policy_sha256"], m["parent_source"])
        load_local_env()
        if not os.getenv("TYPESAFE_API_KEY"):
            raise ValueError("TYPESAFE_API_KEY unavailable after preview receipt")
    import httpx2
    if transport is None:
        if not fake:transport=httpx2.AsyncHTTPTransport(retries=0)
        else:
            async def mock(req):
                b=json.loads((await req.aread()).decode()); criteria=b["questions"]["decision"]["criteria"]; keys=list(criteria)
                probs={k:1/len(keys) for k in keys}
                return httpx2.Response(200,json={"model":MODEL,"usage":{"input_tokens":2,"output_tokens":1},"answers":{"decision":{"type":"choice","choice":keys[-1],"confidence":.5,"probabilities":probs}}})
            transport=httpx2.MockTransport(mock)
    budget=SharedBudget(out/"attempts.jsonl",max_requests=MAX_REQUESTS,max_bytes=MAX_RESERVED_BYTES,
        rps=10000 if fake else MAX_RPS,bytes_per_second=100000000 if fake else MAX_BYTES_PER_SECOND,max_in_flight=MAX_IN_FLIGHT)
    from capclaims.continuation_transport_audit import HttpOverlapTransport

    observer=HttpOverlapTransport(transport)
    guard=GuardTransport(observer,DEFAULT_URL,budget,max_body_bytes=MAX_BODY_BYTES); recorder=InMemoryWireRecorder(guard)
    from typesafe_sdk import AsyncTypeSafeClient
    from typesafe_sdk._core.retry import RetryPolicy
    api_key="offline-dummy-key" if fake else os.environ["TYPESAFE_API_KEY"]
    http=httpx2.AsyncClient(transport=recorder,follow_redirects=False)
    client=AsyncTypeSafeClient(api_key=api_key,model=MODEL,retry=RetryPolicy(max_retries=0),timeout=30,http_client=http,base_url="https://api.typesafe.ai",compression=False)
    backends={arm:_PromptBackend(api_key=api_key,rps=1000 if fake else 40,tps=10_000_000 if fake else 50_000,max_requests=MAX_REQUESTS,max_estimated_input_tokens=6_300_000,instruction_profile="portfolio_third_v1",objective="portfolio_prediction_v1",client=client,request_timeout=30,max_in_flight=10,load_environment=False,joint=arm.endswith("joint"),arm=arm) for arm in ARMS}
    # Filled in by the runner below; any failed call aborts the complete study.
    rows=[]
    async def run_one(s):
        base={**inputs["bases"][s["person_id"]],"agent_id":inputs["bases"][s["person_id"]]["agent_id"]+"__"+s["arm"]}
        trace=(); events=[]
        while True:
            action=next_provider_trace_action(base,trace)
            if action["kind"]=="complete":return {**s,"status":"complete","trace":events,"orders":[o.__dict__ for o in replay_provider_trace(base,trace)["orders"]]}
            if action["kind"]=="forced": trace=advance_provider_trace(base,trace,None);events.append(trace[-1]);continue
            if action["kind"]!="request":raise RuntimeError("provider trace became incomplete")
            req=build_provider_order_request(base,trace)
            wire=await _serialize(req,s["arm"].endswith("joint"),s["arm"]);guard.register(wire)
            (reply,)=await backends[s["arm"]].decide_many([req])
            from capclaims.portfolio_study import _replay_backend
            record=recorder.rows.get(req.request_id)
            if not record or record["status"]!=200:raise RuntimeError("request-ID-bound HTTP response is missing or unsuccessful")
            body=json.loads(record["response_body_utf8"]); answer=body["answers"]["decision"]
            trace=advance_provider_trace(base,trace,answer,reply);events.append(trace[-1])
    try:
        sessions=inputs["sessions"]
        for start in range(0,len(sessions),MAX_IN_FLIGHT):
            wave=sessions[start:start+MAX_IN_FLIGHT]
            remaining=MAX_WALL_SECONDS-(time.monotonic()-started_monotonic)
            if remaining<=0:raise TimeoutError("study exceeded wall-clock cap before the next wave")
            result=await asyncio.wait_for(asyncio.gather(*(run_one(s) for s in wave),return_exceptions=True),
                timeout=remaining)
            errors=[r for r in result if isinstance(r,BaseException)]
            if errors:
                detail=f"; {str(errors[0])[:180]}" if fake else ""
                raise RuntimeError(f"study wave invalidated ({len(errors)} failures; {type(errors[0]).__name__}{detail})")
            rows.extend(result)
    except BaseException:
        _write(out/"invalid.json",{"status":"INVALID","completed_before_failure":len(rows)})
        raise
    finally:
        for b in backends.values():b._closed=True
        await client.aclose()
    _write(out/"events.json",rows)
    from capclaims.continuation_transport_audit import build_event_chain
    event_chain=build_event_chain(rows)
    _write(out/"events-chain.json",event_chain)
    _write(out/"transport.json",list(guard.http_timings))
    _write(out/"transport-precise.json",observer.rows)
    for source_name in ("attempts.jsonl","transport-requests.jsonl","transport-bodies.jsonl"):
        source=out/source_name
        if not source.exists():source.touch(exist_ok=False)
    receipt_sha256=_sha_bytes((out/"receipt.json").read_bytes())
    terminal={"status":"COMPLETE","execution_mode":"mock" if fake else "provider",
        "preview_sha256":m["preview_sha256"],"receipt_sha256":receipt_sha256,
        "events_chain_sha256":_sha_bytes((out/"events-chain.json").read_bytes()),
        "event_sha256":_sha_bytes((out/"events.json").read_bytes()),
        "attempts_sha256":_sha_bytes((out/"attempts.jsonl").read_bytes()),
        "requests_sha256":_sha_bytes((out/"transport-requests.jsonl").read_bytes()),
        "bodies_sha256":_sha_bytes((out/"transport-bodies.jsonl").read_bytes()),
        "timings_sha256":_sha_bytes((out/"transport.json").read_bytes()),
        "precise_timings_sha256":_sha_bytes((out/"transport-precise.json").read_bytes()),
        "budget":budget.usage}
    max_overlap = observer.peak
    if max_overlap > MAX_IN_FLIGHT:
        _write(out / "invalid.json", {"status": "INVALID", "reason": "concurrency_cap_exceeded"})
        raise RuntimeError("global HTTP concurrency limit was exceeded")
    _write(out/"terminal.json",terminal)
    return {"status":"COMPLETE","session_n":len(rows),"planned_sessions":MAX_SESSIONS,
        "attempted_sessions":len(rows),"unattempted_sessions":0,
        "request_n":budget.usage["requests_charged"],"requests_charged":budget.usage["requests_charged"],
        "reserved_bytes":budget.usage["reserved_bytes"],
        "max_overlap":max_overlap}

def replay_run(preview_dir:str|Path,run_dir:str|Path)->dict[str,Any]:
    pmanifest=json.loads((Path(preview_dir)/"manifest.json").read_text(encoding="utf-8"))
    verify_preview(preview_dir,pmanifest.get("preview_sha256",""))
    root=Path(run_dir); invalid=root/"invalid.json"
    if invalid.exists():raise ValueError("study is INVALID and cannot be scored")
    rows=json.loads((root/"events.json").read_text(encoding="utf-8")); inputs=json.loads((Path(preview_dir)/"inputs.json").read_text(encoding="utf-8"))
    terminal=json.loads((root/"terminal.json").read_text(encoding="utf-8"))
    if terminal.get("preview_sha256")!=pmanifest["preview_sha256"]:raise ValueError("run terminal is bound to another preview")
    for filename,key in (("events.json","event_sha256"),("attempts.jsonl","attempts_sha256"),
            ("transport-requests.jsonl","requests_sha256"),("transport-bodies.jsonl","bodies_sha256"),
            ("transport.json","timings_sha256"),("transport-precise.json","precise_timings_sha256")):
        if _sha_bytes((root/filename).read_bytes())!=terminal.get(key):raise ValueError("run evidence digest mismatch")
    sent={r["request_id"]:r for r in (json.loads(x) for x in (root/"transport-requests.jsonl").read_text().splitlines() if x)}
    bodies={r["request_id"]:r for r in (json.loads(x) for x in (root/"transport-bodies.jsonl").read_text().splitlines() if x)}
    timing={r["request_id"]:r for r in json.loads((root/"transport.json").read_text())}
    if len(rows)!=24 or len({r.get("session_id") for r in rows})!=24:raise ValueError("incomplete or duplicate run sessions")
    if ([{key:row.get(key) for key in ("session_id","person_id","arm")} for row in rows]
            != pmanifest["sessions"]
            or any(row.get("status") != "complete" for row in rows)):
        raise ValueError("run rows do not match all 24 frozen sessions in order")
    global_used=set();portfolios={}
    for row in rows:
        base={**inputs["bases"][row["person_id"]],"agent_id":inputs["bases"][row["person_id"]]["agent_id"]+"__"+row["arm"]}
        trace=tuple(row["trace"]); result=replay_provider_trace(base,trace)
        if result["status"]!="complete":raise ValueError("trace replay incomplete")
        portfolios[row["session_id"]]=result["orders"]
        used=set();prefix=()
        for event in trace:
            if event.get("origin")=="constraint":
                prefix=tuple([*prefix,event]);continue
            if event.get("origin")!="provider":raise ValueError("trace contains non-provider event in successful study")
            rid=event.get("request_id"); body=bodies.get(rid); request=sent.get(rid); t=timing.get(rid)
            if not isinstance(rid,str) or rid in used or not body or not request or not t or t.get("status")!=200:
                raise ValueError("provider selection lacks one unique successful HTTP evidence join")
            if t.get("request_id") != rid:
                raise ValueError("HTTP timing identity differs from the frozen request")
            used.add(rid)
            if rid in global_used:raise ValueError("one request id was reused across sessions")
            global_used.add(rid)
            if hashlib.sha256(request.get("body_utf8"," ").encode()).hexdigest()!=request.get("body_sha256"):
                raise ValueError("sent request body digest is invalid")
            if body.get("request_body_sha256")!=request.get("body_sha256") or body.get("status")!=200:
                raise ValueError("response body is not bound to sent request")
            req=build_provider_order_request(base,prefix)
            if req is None or req.request_id!=rid:raise ValueError("request id differs from reconstructed provider-choice state")
            expected=asyncio.run(_serialize(req,row["arm"].endswith("joint"),row["arm"]))
            if expected.body.decode("utf-8")!=request.get("body_utf8") or expected.body_sha256!=request.get("body_sha256"):
                raise ValueError("saved SDK request differs from reconstructed prompt/state/candidates")
            parsed=json.loads(body["response_body_utf8"])["answers"]["decision"]
            if json.loads(body["response_body_utf8"]).get("model")!=MODEL or parsed.get("type")!="choice":
                raise ValueError("raw response model or choice type is invalid")
            if (event.get("raw_answer") != parsed
                    or parsed.get("choice")!=event.get("choice")
                    or parsed.get("confidence")!=event.get("confidence")
                    or parsed.get("probabilities")!=event.get("raw_probabilities")):
                raise ValueError("trace selection differs from request-bound raw provider response")
            from capclaims.provider_choice import validate_provider_choice
            validate_provider_choice(parsed,[c.candidate_id for c in req.candidates])
            prefix=tuple([*prefix,event])
    if global_used!=set(bodies) or global_used!=set(sent):raise ValueError("unmatched request/response evidence")
    attempts=[json.loads(x) for x in (root/"attempts.jsonl").read_text().splitlines() if x]
    attempt_ids=[r.get("request_id") for r in attempts]
    if len(attempt_ids)!=len(set(attempt_ids)) or set(attempt_ids)!=global_used or len(attempt_ids)>MAX_REQUESTS:
        raise ValueError("attempt ledger does not bind every successful provider call exactly once")
    for sequence, attempt in enumerate(attempts, start=1):
        request = sent[attempt["request_id"]]
        body = request.get("body_utf8", "").encode("utf-8")
        if (attempt.get("seq") != sequence or attempt.get("method") != "POST"
                or _url_tuple(attempt.get("url", "")) != _url_tuple(DEFAULT_URL)
                or attempt.get("expected_sha256") != request.get("body_sha256")
                or attempt.get("actual_sha256") != request.get("body_sha256")
                or attempt.get("body_bytes") != len(body)
                or attempt.get("reserved_bytes") != MAX_BODY_BYTES
                or attempt.get("state") != "attempted"):
            raise ValueError("attempt ledger row differs from the unique sent SDK request")
    if len(timing) != len(global_used) or len(global_used) * MAX_BODY_BYTES > MAX_RESERVED_BYTES:
        raise ValueError("HTTP timing or reserved-byte totals exceed the frozen envelope")
    from capclaims.continuation_transport_audit import audit_transport_evidence
    transport_audit = audit_transport_evidence(root, terminal, global_used, pmanifest["preview_sha256"])
    return {"status":"REPLAY_VALID","session_n":24,"portfolios":portfolios,
        "statuses":[{"session_id":row["session_id"],"status":row["status"]} for row in rows],
        "transport_audit":transport_audit}


def _portfolio_score_components(predicted, human):
    """Return portfolio metrics plus order permutation-independent diagnostics."""
    from capclaims.portfolio_metrics import portfolio_comparison

    comparison = portfolio_comparison(predicted, human)
    signatures = [(order.side, order.qty, order.limit_dollars) for order in predicted]
    duplicate_n = len(signatures) - len(set(signatures))
    return {
        **comparison,
        "predicted_order_count": len(predicted),
        "human_order_count": len(human),
        "duplicate_order_n": duplicate_n,
        "duplicate_order_rate": duplicate_n / len(predicted) if predicted else 0.0,
        "first_order_only_curve_error": portfolio_comparison(predicted[:1], human)["primary_curve_error"],
    }


def score_run(preview_dir:str|Path,run_dir:str|Path)->dict[str,Any]:
    replayed=replay_run(preview_dir,run_dir)
    from capclaims.paper.orders import PaperOrder
    targets=json.loads((Path(preview_dir)/"scoring-targets.json").read_text(encoding="utf-8"))
    if _sha_bytes((Path(preview_dir)/"scoring-targets.json").read_bytes())!=json.loads((Path(preview_dir)/"manifest.json").read_text())["scoring_targets_sha256"]:
        raise ValueError("scoring labels changed after preview")
    rows=json.loads((Path(run_dir)/"events.json").read_text(encoding="utf-8")); grouped={}
    for r in rows:grouped.setdefault(r["arm"],{})[r["person_id"]]=r
    per_arm={};score_rows=[]
    for arm,people in grouped.items():
        errors=[]; agreements=[]
        for person,row in people.items():
            target=targets[person];truth=[]
            for i,o in enumerate(target["orders"]):
                side=o.get("side",o.get("type"));qty=o.get("qty",o.get("amount"));price=o.get("limit_dollars",o.get("price"))
                truth.append(PaperOrder(f"{person}-truth-{i}",person,str(side).lower(),qty,price))
            predicted=[PaperOrder(f"{person}-pred-{i}",person,o.side,o.qty,o.limit_dollars)
                for i,o in enumerate(replayed["portfolios"][row["session_id"]])]
            comp=_portfolio_score_components(predicted,truth);errors.append(comp["primary_curve_error"]);agreements.append(comp["exact_order_multiset_match"])
            choices=[event for event in row["trace"] if event.get("origin")=="provider"]
            score_rows.append({"person_id":person,"arm":arm,"primary_curve_error":comp["primary_curve_error"],
                "retail_window_1_100_curve_error":comp["retail_window_1_100_curve_error"],
                "predicted_gross_buy":comp["predicted_gross_buy"],"human_gross_buy":comp["human_gross_buy"],
                "predicted_gross_sell":comp["predicted_gross_sell"],"human_gross_sell":comp["human_gross_sell"],
                "predicted_side_class":comp["predicted_side_class"],"human_side_class":comp["human_side_class"],
                "order_count_error":comp["order_count_error"],"empty_agreement":comp["empty_agreement"],
                "exact_order_multiset_match":comp["exact_order_multiset_match"],
                "merged_same_side_price_match":comp["merged_same_side_price_match"],
                "first_order_truncation_diagnostic":comp["first_order_only_curve_error"],
                "duplicate_order_n":comp["duplicate_order_n"],
                "duplicate_order_rate":comp["duplicate_order_rate"],
                "predicted_order_count":comp["predicted_order_count"],
                "human_order_count":comp["human_order_count"],
                "provider_choice_n":len(choices),
                "provider_resolved_tie_n":sum(len(event["near_top_candidate_ids"])>1 for event in choices),
                "provider_choice_differs_unique_argmax_n":sum(
                    event.get("legacy_argmax_status")=="unique" and event.get("legacy_argmax_agrees") is False
                    for event in choices)})
        per_arm[arm]={"n":len(errors),"mean_curve_error":sum(errors)/len(errors),"exact_order_match_rate":sum(agreements)/len(agreements)}
        subset=[item for item in score_rows if item["arm"]==arm]
        per_arm[arm].update({
            "mean_retail_window_1_100_curve_error":sum(x["retail_window_1_100_curve_error"] for x in subset)/len(subset),
            "mean_predicted_gross_buy":sum(x["predicted_gross_buy"] for x in subset)/len(subset),
            "mean_human_gross_buy":sum(x["human_gross_buy"] for x in subset)/len(subset),
            "mean_predicted_gross_sell":sum(x["predicted_gross_sell"] for x in subset)/len(subset),
            "mean_human_gross_sell":sum(x["human_gross_sell"] for x in subset)/len(subset),
            "side_class_agreement_rate":sum(x["predicted_side_class"]==x["human_side_class"] for x in subset)/len(subset),
            "mean_order_count_error":sum(x["order_count_error"] for x in subset)/len(subset),
            "mean_predicted_order_count":sum(x["predicted_order_count"] for x in subset)/len(subset),
            "mean_human_order_count":sum(x["human_order_count"] for x in subset)/len(subset),
            "duplicate_order_n":sum(x["duplicate_order_n"] for x in subset),
            "mean_duplicate_order_rate":sum(x["duplicate_order_rate"] for x in subset)/len(subset),
            "empty_agreement_rate":sum(x["empty_agreement"] for x in subset)/len(subset),
            "same_side_price_merge_match_rate":sum(x["merged_same_side_price_match"] for x in subset)/len(subset),
            "mean_first_order_truncation_curve_error":sum(x["first_order_truncation_diagnostic"] for x in subset)/len(subset),
            "provider_choice_n":sum(x["provider_choice_n"] for x in subset),
            "provider_resolved_tie_n":sum(x["provider_resolved_tie_n"] for x in subset),
            "provider_choice_differs_unique_argmax_n":sum(x["provider_choice_differs_unique_argmax_n"] for x in subset),
            "coverage":len(subset)/6,
        })
    contrasts={}
    for prefix in ("C","E"):
        contrasts[prefix]={"joint_minus_base_curve_error":per_arm[f"{prefix}_joint"]["mean_curve_error"]-per_arm[f"{prefix}_base"]["mean_curve_error"]}
    from capclaims.continuation_analysis import summarize_score_rows
    paired=summarize_score_rows(score_rows)
    return {"status":"SCORED_DESCRIPTIVE_DEVELOPMENT_ONLY","per_arm":per_arm,
        "joint_minus_base":contrasts,"paired_analysis":paired,
        "limitations":"six people per arm; descriptive development sample only"}

def audit_legacy_responses(parent_dir:str|Path,legacy_run_dir:str|Path)->dict[str,Any]:
    from capclaims.injection_study import replay_run as replay_legacy
    replay_legacy(parent_dir,legacy_run_dir)
    run=Path(legacy_run_dir); body_rows=[json.loads(x) for x in (run/"transport-bodies.jsonl").read_text(encoding="utf-8").splitlines() if x]
    request_rows=[json.loads(x) for x in (run/"transport-requests.jsonl").read_text(encoding="utf-8").splitlines() if x]
    requests={r["request_id"]:json.loads(r["body_utf8"]) for r in request_rows}
    n=exact_tie=floating_near_tie=provider_near_top=disagree=invalid_choice=invalid_distribution=0
    from capclaims.provider_choice import validate_provider_choice
    from capclaims.backends.jev import normalize_jev_probabilities
    for row in body_rows:
        if not 200<=row.get("status",0)<300:continue
        req=requests[row["request_id"]]; criteria=req["questions"]["decision"]["criteria"]
        answer=json.loads(row["response_body_utf8"])["answers"]["decision"]
        n+=1
        ids=list(criteria)
        try:
            normalized,_=normalize_jev_probabilities(answer.get("probabilities"),ids)
        except (TypeError,ValueError):
            invalid_distribution+=1
            continue
        rawp={k:float(answer["probabilities"][k]) for k in ids}
        best=max(rawp.values());selected=rawp.get(answer.get("choice"),float("nan"))
        max_winner_n=sum(v==best for v in rawp.values());gap=best-selected
        exact_tie+=max_winner_n>1
        floating_near_tie+=max_winner_n==1 and 0<gap<=1e-12
        provider_near_top+=gap<=1e-12
        try:
            parsed=validate_provider_choice(answer,ids)
        except ValueError:
            invalid_choice+=1
        else:
            disagree+=parsed["legacy_argmax_status"]=="unique" and not parsed["legacy_argmax_agrees"]
    if n!=778:raise ValueError(f"legacy wire audit expected 778 successful responses, found {n}")
    return {"response_n":n,"invalid_distribution_n":invalid_distribution,"invalid_choice_n":invalid_choice,
        "valid_choice_n":n-invalid_choice,"exact_probability_tie_n":exact_tie,"floating_near_tie_choice_n":floating_near_tie,
        "provider_choice_within_epsilon_of_max_n":provider_near_top,
        "provider_choice_differs_unique_legacy_argmax_n":disagree}

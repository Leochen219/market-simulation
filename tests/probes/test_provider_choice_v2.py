import pytest
import asyncio, hashlib, json
from pathlib import Path

from capclaims.provider_choice import validate_provider_choice
from capclaims.provider_choice_trace import (advance_provider_trace, build_provider_order_request,
    next_provider_trace_action, replay_provider_trace)
from capclaims.portfolio_prediction import _binding
from capclaims.engine.protocol import DecisionReply


def test_choice_accepts_ties_and_reports_legacy_argmax():
    result = validate_provider_choice({"type": "choice", "choice": "b", "confidence": 0.5,
        "probabilities": {"a": 0.5, "b": 0.5}}, ["a", "b"])
    assert result["choice"] == "b"
    assert result["near_top_candidate_ids"] == ["a", "b"]
    assert result["legacy_argmax_candidate_id"] is None
    assert result["legacy_argmax_status"] == "unresolved_tie"
    assert result["legacy_argmax_agrees"] is None


@pytest.mark.parametrize("answer", [
    {"type": "choice", "choice": "a", "confidence": 0.5, "probabilities": {"a": 0.4, "b": 0.6}},
    {"type": "choice", "choice": "z", "confidence": 0.5, "probabilities": {"a": 0.5, "b": 0.5}},
    {"type": "choice", "choice": "a", "confidence": float("nan"), "probabilities": {"a": 0.5, "b": 0.5}},
    {"type": "choice", "choice": "a", "confidence": 0.5, "probabilities": {"a": 0.5}},
])
def test_choice_rejects_invalid_provider_contract(answer):
    with pytest.raises(ValueError):
        validate_provider_choice(answer, ["a", "b"])


def test_trace_uses_provider_selection_and_replays_raw_evidence():
    base = {"agent_id": "p", "round": 9, "treatment": "GA", "wholesale_price_dollars": 10,
        "p": 60, "Q": 5, "D": 2, "history": []}
    trace = ()
    request = build_provider_order_request(base, trace)
    assert request is not None
    ids = [candidate.candidate_id for candidate in request.candidates]
    chosen = ids[-1]
    probs = {key: 1 / len(ids) for key in ids}
    answer = {"type": "choice", "choice": chosen, "confidence": 0.5, "probabilities": probs}
    reply = DecisionReply.from_request(request, probs, probs, backend="jev", model_revision="jev-1.14.0")
    trace = advance_provider_trace(base, trace, answer, reply)
    assert trace[-1]["choice"] == chosen
    assert next_provider_trace_action(base, trace)["kind"] in {"forced", "request"}
    altered = (dict(trace[-1], choice=ids[0]),)
    with pytest.raises(ValueError):
        replay_provider_trace(base, altered)


def test_trace_rejects_probability_disagreement():
    base = {"agent_id": "p", "round": 9, "treatment": "GA", "wholesale_price_dollars": 10,
        "p": 60, "Q": 5, "D": 2, "history": []}
    request = build_provider_order_request(base, ())
    ids = [candidate.candidate_id for candidate in request.candidates]
    answer = {"type": "choice", "choice": ids[0], "confidence": 1, "probabilities": {k: float(k == ids[0]) for k in ids}}
    wrong = {k: float(k == ids[-1]) for k in ids}
    reply = DecisionReply.from_request(request, wrong, wrong)
    with pytest.raises(ValueError, match="differ"):
        advance_provider_trace(base, (), answer, reply)


def test_float_tail_choice_records_legacy_disagreement_and_raw_answer_type():
    base = {"agent_id": "p", "round": 9, "treatment": "GA", "wholesale_price_dollars": 10,
        "p": 60, "Q": 5, "D": 2, "history": []}
    request = build_provider_order_request(base, ())
    ids = [candidate.candidate_id for candidate in request.candidates]
    probabilities = {ids[0]: 0.5, ids[1]: 0.5 - 5e-13, ids[2]: 0.0}
    answer = {"type": "choice", "choice": ids[1], "confidence": 0.5, "probabilities": probabilities}
    reply = DecisionReply.from_request(request, probabilities, probabilities, backend="jev", model_revision="jev-1.14.0")
    trace = advance_provider_trace(base, (), answer, reply)
    event = trace[-1]
    assert event["legacy_argmax_status"] == "unique"
    assert event["legacy_argmax_candidate_id"] == ids[0]
    assert event["legacy_argmax_agrees"] is False
    assert replay_provider_trace(base, trace)["status"] == "complete"
    with pytest.raises(ValueError, match="normalized probabilities"):
        replay_provider_trace(base, (dict(event, legacy_argmax_agrees=True),))
    altered = (dict(event, raw_answer={**answer, "type": "noul"}),)
    with pytest.raises(ValueError, match="raw provider answer"):
        replay_provider_trace(base, altered)


def test_score_diagnostics_are_permutation_invariant_and_count_duplicate_orders():
    from capclaims.paper.orders import PaperOrder
    from capclaims.continuation_study import _portfolio_score_components

    predicted = [
        PaperOrder("p1", "agent", "sell", 2, 40),
        PaperOrder("p2", "agent", "sell", 2, 40),
    ]
    human = [
        PaperOrder("h1", "human", "buy", 3, 20),
        PaperOrder("h2", "human", "sell", 1, 50),
    ]
    forward = _portfolio_score_components(predicted, human)
    reverse = _portfolio_score_components(predicted, list(reversed(human)))
    assert forward["primary_curve_error"] == reverse["primary_curve_error"]
    assert forward["first_order_only_curve_error"] == reverse["first_order_only_curve_error"]
    assert forward["duplicate_order_n"] == 1
    assert forward["duplicate_order_rate"] == 0.5


def test_24_session_sdk_mock_overlaps_and_replays_raw_response(tmp_path, monkeypatch):
    import capclaims.continuation_study as study
    monkeypatch.setattr(study, "_private", lambda path: Path(path).resolve())
    parent=tmp_path/"parent";parent.mkdir()
    people=[f"p{i}" for i in range(6)]
    bases={p:{"agent_id":p,"round":9,"treatment":"GA","wholesale_price_dollars":10,
        "p":60+i,"Q":5,"D":2,"history":[{"round":r,"p":70,"Q":6,"D":3,
          "orders":[{"type":"SELL","amount":1,"price":20}]} for r in (6,7,8)]}
        for i,p in enumerate(people)}
    labels={p:{"p":bases[p]["p"],"Q":5,"D":2,"orders":[]} for p in people}
    raw=json.dumps(labels,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()
    (parent/"labels.json").write_bytes(raw)
    policy_path=tmp_path/"source-policy.json"
    policy={"codebook_verified":True,"intended_use_authorized":True,"provider_transmission_permitted":True,
        "source_kind":"human","source_sha256":"b"*64}
    policy_raw=json.dumps(policy,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()
    policy_path.write_bytes(policy_raw)
    monkeypatch.setattr(study,"_source_policy_path",lambda:policy_path)
    parent_manifest={"sessions":[],"source":policy,
        "source_policy_sha256":study._sha_bytes(policy_raw),"labels_sha256":study._sha(labels),"labels_file_sha256":study._sha_bytes(raw)}
    sessions=[{"person_id":p,"target_id":p,"base":bases[p]} for p in people]
    parent_manifest["sessions"]=sessions
    monkeypatch.setattr(study,"_read_parent",lambda _: (parent_manifest,{"sessions":sessions}))
    preview=tmp_path/"preview"; manifest=study.build_preview(parent,preview)
    run=tmp_path/"run"
    import httpx2
    active=peak=0
    seen_bodies=set()
    barrier=asyncio.Event()
    async def mock(request):
        nonlocal active,peak
        active+=1;peak=max(peak,active)
        if active==10:barrier.set()
        await asyncio.wait_for(barrier.wait(),timeout=5)
        raw_body=await request.aread();seen_bodies.add(hashlib.sha256(raw_body).hexdigest())
        body=json.loads(raw_body.decode()); criteria=body["questions"]["decision"]["criteria"]
        chosen=next(k for k,v in criteria.items() if "STOP" in v)
        probs={k:1/len(criteria) for k in criteria}
        active-=1
        return httpx2.Response(200,json={"model":study.MODEL,"usage":{"input_tokens":1,"output_tokens":1},
            "answers":{"decision":{"type":"choice","choice":chosen,"confidence":0,"probabilities":probs}}})
    targets_file=preview/"scoring-targets.json";frozen_targets=targets_file.read_bytes()
    targets_file.write_bytes(b"intentionally unavailable during execution and replay")
    result=asyncio.run(study.execute_preview(preview,run,manifest["preview_sha256"],fake=True,fake_transport=httpx2.MockTransport(mock)))
    assert result["status"]=="COMPLETE" and result["session_n"]==24
    assert peak==10 and result["max_overlap"]==10 and len(seen_bodies)==24
    assert study.replay_run(preview,run)["status"]=="REPLAY_VALID"
    with pytest.raises(ValueError):study.score_run(preview,run)
    targets_file.write_bytes(frozen_targets)
    assert study.score_run(preview,run)["status"]=="SCORED_DESCRIPTIVE_DEVELOPMENT_ONLY"

    # Session names alone do not bind the randomized participant/arm assignments.
    from capclaims.continuation_transport_audit import build_event_chain
    evidence=[run/name for name in ("events.json","events-chain.json","terminal.json")]
    saved={path:path.read_bytes() for path in evidence}
    changed=json.loads(saved[evidence[0]])
    changed[0]["arm"]="E_joint" if changed[0]["arm"]!="E_joint" else "C_base"
    evidence[0].write_bytes((study.canonical_json(changed)+"\n").encode())
    evidence[1].write_bytes((study.canonical_json(build_event_chain(changed))+"\n").encode())
    terminal=json.loads(saved[evidence[2]])
    terminal["event_sha256"]=hashlib.sha256(evidence[0].read_bytes()).hexdigest()
    terminal["events_chain_sha256"]=hashlib.sha256(evidence[1].read_bytes()).hexdigest()
    evidence[2].write_bytes((study.canonical_json(terminal)+"\n").encode())
    with pytest.raises(ValueError,match="frozen sessions"):
        study.replay_run(preview,run)
    for path,raw in saved.items():path.write_bytes(raw)

    # A revoked/mismatched private policy must be rejected before even loading keys.
    policy_path.write_bytes(b"{}")
    def no_key_access():
        raise AssertionError("credential access happened before source-policy verification")
    monkeypatch.setattr(study,"load_local_env",no_key_access)
    with pytest.raises(ValueError,match="source-policy"):
        asyncio.run(study.execute_preview(preview,tmp_path/"policy-denied",manifest["preview_sha256"],fake=False))
    policy_path.write_bytes(policy_raw)

    # A rehashed but semantically altered raw response must still be rejected.
    bodies_file=run/"transport-bodies.jsonl"; terminal_file=run/"terminal.json"
    original_bodies=bodies_file.read_bytes(); original_terminal=terminal_file.read_bytes()
    body_rows=[json.loads(line) for line in original_bodies.splitlines()]
    first=body_rows[0]; response=json.loads(first["response_body_utf8"])
    response["answers"]["decision"]["choice"]="action-buy"
    first["response_body_utf8"]=json.dumps(response,separators=(",",":"))
    first["response_body_sha256"]=hashlib.sha256(first["response_body_utf8"].encode()).hexdigest()
    bodies_file.write_bytes(b"".join((json.dumps(row,sort_keys=True,separators=(",",":"))+"\n").encode() for row in body_rows))
    terminal=json.loads(original_terminal);terminal["bodies_sha256"]=hashlib.sha256(bodies_file.read_bytes()).hexdigest()
    terminal_file.write_bytes((json.dumps(terminal,sort_keys=True,separators=(",",":"))+"\n").encode())
    with pytest.raises(ValueError):study.replay_run(preview,run)
    bodies_file.write_bytes(original_bodies);terminal_file.write_bytes(original_terminal)

    # Even if every transport digest is recomputed, a changed prompt fails exact SDK reconstruction.
    requests_file=run/"transport-requests.jsonl"; attempts_file=run/"attempts.jsonl"
    original_requests=requests_file.read_bytes();original_attempts=attempts_file.read_bytes()
    request_rows=[json.loads(line) for line in original_requests.splitlines()]
    altered=request_rows[0];wire=json.loads(altered["body_utf8"])
    wire["questions"]["decision"]["instructions"]=wire["questions"]["decision"]["instructions"].replace("You are","You xre",1)
    altered["body_utf8"]=json.dumps(wire,separators=(",",":"),ensure_ascii=False)
    altered["body_sha256"]=hashlib.sha256(altered["body_utf8"].encode()).hexdigest()
    altered["body_bytes"]=len(altered["body_utf8"].encode())
    requests_file.write_bytes(b"".join((json.dumps(row,sort_keys=True,separators=(",",":"))+"\n").encode() for row in request_rows))
    attempt_rows=[json.loads(line) for line in original_attempts.splitlines()]
    changed_attempt=next(row for row in attempt_rows if row["request_id"]==altered["request_id"])
    changed_attempt["expected_sha256"]=altered["body_sha256"];changed_attempt["actual_sha256"]=altered["body_sha256"]
    changed_attempt["body_bytes"]=altered["body_bytes"]
    attempts_file.write_bytes(b"".join((json.dumps(row,sort_keys=True,separators=(",",":"))+"\n").encode() for row in attempt_rows))
    body_rows=[json.loads(line) for line in original_bodies.splitlines()]
    matched=next(row for row in body_rows if row["request_id"]==altered["request_id"])
    matched["request_body_utf8"]=altered["body_utf8"];matched["request_body_sha256"]=altered["body_sha256"]
    bodies_file.write_bytes(b"".join((json.dumps(row,sort_keys=True,separators=(",",":"))+"\n").encode() for row in body_rows))
    terminal=json.loads(original_terminal)
    terminal["requests_sha256"]=hashlib.sha256(requests_file.read_bytes()).hexdigest()
    terminal["attempts_sha256"]=hashlib.sha256(attempts_file.read_bytes()).hexdigest()
    terminal["bodies_sha256"]=hashlib.sha256(bodies_file.read_bytes()).hexdigest()
    terminal_file.write_bytes((json.dumps(terminal,sort_keys=True,separators=(",",":"))+"\n").encode())
    with pytest.raises(ValueError,match="reconstructed prompt/state/candidates"):
        study.replay_run(preview,run)
    requests_file.write_bytes(original_requests);attempts_file.write_bytes(original_attempts)
    bodies_file.write_bytes(original_bodies);terminal_file.write_bytes(original_terminal)

    # A provider failure invalidates the whole 24-session run; no second wave or retry follows.
    failed_calls=0
    async def fail_http(request):
        nonlocal failed_calls
        failed_calls+=1
        await request.aread()
        return httpx2.Response(500,json={"error":"fixture failure"})
    failed_run=tmp_path/"failed-run"
    with pytest.raises(RuntimeError,match="study wave invalidated"):
        asyncio.run(study.execute_preview(preview,failed_run,manifest["preview_sha256"],fake=True,
            fake_transport=httpx2.MockTransport(fail_http)))
    assert failed_calls==10
    assert (failed_run/"invalid.json").exists()
    with pytest.raises(ValueError,match="INVALID"):
        study.replay_run(preview,failed_run)

"""Actual SDK bytes isolate naming at every model-visible stage and prefix."""
import asyncio
import json
import pytest

from capclaims.continuation_study import _serialize as old_wire
from capclaims.engine.protocol import DecisionReply
from capclaims.field_alignment_context import CONTROL, ALIGNED, RENAMES, render_field_state, inverse_aligned_state
from capclaims.field_alignment_study import serialize_alignment_wire
from capclaims.injection_context import render_state
from capclaims.provider_choice_trace import advance_provider_trace, build_provider_order_request, next_provider_trace_action


@pytest.mark.parametrize("q,p,d", [(0,51,200),(400,100,0)])
def test_all_request_phases_preserve_control_wire_and_treatment_facts(q,p,d):
    async def run():
        base={"agent_id":"synthetic-field-wire","round":9,"treatment":"GA",
              "wholesale_price_dollars":10,"p":p,"Q":q,"D":d,
              "history":[{"round":r,"p":60,"Q":400,"D":100,
                          "orders":[{"type":"BUY","amount":999,"price":1},
                                    {"type":"SELL","amount":1,"price":999}]} for r in (6,7,8)]}
        trace=();phases=set();calls=0
        for _ in range(40):
            action=next_provider_trace_action(base,trace)
            if action["kind"]=="complete":break
            if action["kind"]=="forced":
                trace=advance_provider_trace(base,trace,None);continue
            request=build_provider_order_request(base,trace)
            phases.add(request.observation["decision_field"])
            control=render_field_state(request,CONTROL);aligned=render_field_state(request,ALIGNED)
            assert control==render_state(request,"C")
            assert inverse_aligned_state(aligned)==control
            assert list(aligned)==[RENAMES.get(k,k) for k in control]
            for before,after in RENAMES.items():
                assert aligned[after]==control[before] and type(aligned[after]) is type(control[before])
                assert before not in aligned
            for key in ("own_history","partial_orders","submitted_order_ledger","own_qty_units","own_side"):
                if key in control:assert aligned[key]==control[key]
            a,b,old=await asyncio.gather(serialize_alignment_wire(request,CONTROL),
                serialize_alignment_wire(request,ALIGNED),old_wire(request,True,"C_joint"))
            assert a.body==old.body
            left,right=json.loads(a.body),json.loads(b.body)
            right["state"]=inverse_aligned_state(right["state"])
            assert right==left and list(right["state"])==list(left["state"])
            candidates=request.candidates
            if request.stage.endswith(":side"):
                values={c.value for c in candidates}
                chosen="sell" if request.observation["slot"]%2 and "sell" in values else "buy"
            else:chosen=max(c.value for c in candidates)
            winner=next(c for c in candidates if c.value==chosen)
            probs={c.candidate_id:float(c==winner) for c in candidates}
            raw={"type":"choice","choice":winner.candidate_id,"confidence":1,"probabilities":probs}
            reply=DecisionReply.from_request(request,probs,probs,backend="synthetic",model_revision="jev-1.14.0")
            trace=advance_provider_trace(base,trace,raw,reply);calls+=1
        else:pytest.fail("legal five-order trace did not terminate")
        assert phases=={"side","quantity","price"} and calls<=35
    asyncio.run(run())

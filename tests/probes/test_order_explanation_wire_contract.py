"""Independent contract against actual SDK bytes and real legal trace requests."""
import asyncio
import json
import pytest

from capclaims.continuation_study import _serialize as old_wire
from capclaims.engine.protocol import DecisionReply
from capclaims.order_presentation import CONTROL, EXPLICIT, EXPLANATION, render_order_state, inverse_explicit_state
from capclaims.order_explanation_study import serialize_explanation_wire
from capclaims.provider_choice_trace import advance_provider_trace, build_provider_order_request, next_provider_trace_action


@pytest.mark.parametrize("q,p,d", [(0,51,200),(400,100,0)])
def test_real_all_phase_requests_preserve_control_and_explicit_order_facts(q,p,d):
    async def run():
        base={"agent_id":"synthetic-order-contract","round":9,"treatment":"GA",
            "wholesale_price_dollars":10,"p":p,"Q":q,"D":d,
            "history":[{"round":r,"p":60,"Q":400,"D":100,
                "orders":[{"type":"BUY","amount":999,"price":1},
                          {"type":"SELL","amount":1,"price":999}]} for r in (6,7,8)]}
        trace=(); phases=set(); nonempty=0
        for _ in range(40):
            action=next_provider_trace_action(base,trace)
            if action["kind"]=="complete": break
            if action["kind"]=="forced":
                trace=advance_provider_trace(base,trace,None); continue
            request=build_provider_order_request(base,trace)
            phases.add(request.observation["decision_field"])
            control=render_order_state(request,CONTROL)
            explicit=render_order_state(request,EXPLICIT)
            assert control["partial_order_fields"]=="slot,side,qty units,limit $/unit"
            assert list(inverse_explicit_state(explicit).items())==list(control.items())
            assert list(explicit)==[k for k in control if k!="partial_order_fields"]
            for source, named in zip(control["partial_orders"],explicit["partial_orders"]):
                assert list(named)==["slot","side","quantity_units","limit_price_dollars_per_unit"]
                assert [(type(v),v) for v in named.values()]==[(type(v),v) for v in source]
            if control["partial_orders"]: nonempty+=1
            assert explicit["submitted_order_ledger"]==control["submitted_order_ledger"]
            assert explicit["submitted_order_ledger"]["submitted_order_count"]==len(explicit["partial_orders"])
            a,b,old=await asyncio.gather(serialize_explanation_wire(request,CONTROL),
                serialize_explanation_wire(request,EXPLICIT),old_wire(request,True,"C_joint"))
            assert a.body==old.body
            left,right=json.loads(a.body),json.loads(b.body)
            assert right["questions"]["decision"]["instructions"] == left["questions"]["decision"]["instructions"]+"\n\n"+EXPLANATION
            right["questions"]["decision"]["instructions"]=left["questions"]["decision"]["instructions"]
            right["state"]=inverse_explicit_state(right["state"])
            assert right==left and list(right["state"])==list(left["state"])
            candidates=request.candidates
            if request.stage.endswith(":side"):
                chosen="sell" if request.observation["slot"]%2 and "sell" in {c.value for c in candidates} else "buy"
            else: chosen=max(c.value for c in candidates)
            winner=next(c for c in candidates if c.value==chosen)
            probs={c.candidate_id:float(c==winner) for c in candidates}
            raw={"type":"choice","choice":winner.candidate_id,"confidence":1,"probabilities":probs}
            reply=DecisionReply.from_request(request,probs,probs,backend="synthetic",model_revision="jev-1.14.0")
            trace=advance_provider_trace(base,trace,raw,reply)
        else: pytest.fail("legal five-order trace did not terminate")
        assert phases=={"side","quantity","price"} and nonempty>0
    asyncio.run(run())

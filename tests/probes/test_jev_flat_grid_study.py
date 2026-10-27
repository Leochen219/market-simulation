from copy import deepcopy
import pytest
from capclaims.flat_grid_study import design

BASE={'own_q_units':100,'own_side':'BUY','partial_orders':[], 'submitted_order_ledger':{'submitted_order_count':0,'remaining_order_slots':1,'submitted_buy_quantity_units':0,'submitted_sell_quantity_units':0,'remaining_sell_submission_capacity_units':100},'public_instructions':{'trading':'price choices are complete $2.50 ticks.'},'history':[{'round':1}],'own_p':42}

def test_matched_state_candidates_and_no_mutation():
    before=deepcopy(BASE)
    state, options=design.price_spec({'base_state':BASE,'side':'BUY'}, 20)
    assert BASE==before and state['history']==BASE['history'] and state['own_p']==42
    assert state['own_qty_units']==state['model_chosen_quantity']==20
    assert 'model_chosen_band' not in state and '$5.00' in state['public_instructions']['trading']
    assert len(options)==199 and list(options)[0]=='p005' and list(options)[-1]=='p995'

def test_shared_draw_and_empty_support():
    ids=[f'p{i:03d}' for i in range(5,1000,5)]
    raw={i:0.0 for i in ids}; raw[ids[0]]=raw[ids[1]]=0.5
    row={'person_id':'x','formal_round':11,'repeat':0,'condition':'short3'}
    a=design.decide(raw,ids,row)
    row['condition']='long10'
    assert design.decide(raw,ids,row)==a
    assert design.decide({i:1/199 for i in ids},ids,row)['status']=='client_numeric_abstain'
    raw.pop(ids[0])
    with pytest.raises(ValueError): design.decide(raw,ids,row)

def test_schedule_identity_and_quantity():
    rows=[{'person_id':'x','formal_round':11,'condition':c,'repeat':r,'band_order':[0,1,2,3]} for r in (0,1) for c in ('short3','long10')]
    preds=[dict(r,conditioned_quantity=20,request_hashes={'quantity':'a','price-b0':'b'}) for r in rows]
    result=design.schedule(rows,preds)
    assert len(result)==4 and len({design.request_id(r) for r in result})==4
    assert all(r['conditioned_quantity']==20 for r in result)
    with pytest.raises(ValueError): design.schedule(rows,preds[:-1])

def test_ledger_cap_reservation_and_duplicates(tmp_path):
    from capclaims.flat_grid_study.runner import Ledger, RESERVATION
    ledger=Ledger(tmp_path/'ledger.sqlite',preview_hash='a'*64)
    for i in range(72): ledger.reserve(f'r{i}')
    assert ledger.summary()['physical_requests']==72
    assert ledger.summary()['committed_nanodollars']==72*RESERVATION
    with pytest.raises(ValueError): ledger.reserve('r72')
    with pytest.raises(ValueError): ledger.reserve('r0')

def test_parallel_calls_continue_after_numeric_abstention(tmp_path):
    import asyncio
    from capclaims.flat_grid_study import runner
    class Mock:
        halted=False
        def __init__(self): self.active=0; self.peak=0; self.count=0; self.ready=asyncio.Event()
        async def send_body(self,*,request_id,body):
            self.active+=1; self.count+=1; self.peak=max(self.peak,self.active)
            if self.count==10: self.ready.set()
            await self.ready.wait(); await asyncio.sleep(0)
            self.active-=1
            return {'answers':{'decision':{'choice':'p005','probabilities':{f'p{i:03d}':1/199 for i in range(5,1000,5)}}}}
    async def exercise():
        client=Mock(); ctx={'base_state':BASE,'side':'BUY'}
        rows=[{'person_id':f'x{i}','formal_round':11,'condition':'short3','repeat':0,'conditioned_quantity':20} for i in range(10)]
        results=await asyncio.gather(*(runner.call(client,tmp_path,ctx,r) for r in rows))
        assert client.peak==10 and len(results)==10
        assert all(r['decision']['status']=='client_numeric_abstain' for r in results)
    asyncio.run(exercise())

def test_first_http_failure_survives_partial_replay_and_tamper(tmp_path,monkeypatch):
    import asyncio, httpx2
    from capclaims.flat_grid_study import runner
    async def exercise():
        ctx={'person_id':'x','formal_round':11,'condition':'short3','base_state':BASE,'side':'BUY'}
        row={'person_id':'x','formal_round':11,'condition':'short3','repeat':0,'conditioned_quantity':20}
        rid=design.request_id(row); data=await runner.body(ctx,row)
        plan={'contexts':[ctx],'schedule':[row],'bodies':{rid:{'sha256':runner.sha(data),'bytes':len(data)}}}
        async def valid(_): return plan
        monkeypatch.setattr(runner,'valid_plan',valid)
        (tmp_path/'plan.json').write_bytes(b'plan')
        plan_hash=runner.sha(b'plan'); (tmp_path/'started').write_text(plan_hash+'\n',encoding='ascii')
        ledger=runner.Ledger(tmp_path/'cost-ledger.sqlite',preview_hash=plan_hash)
        async def fail(_): return httpx2.Response(500,content=b'fail')
        async with runner.JevWireClient(api_key='dummy',ledger=ledger,evidence_dir=tmp_path/'wire-evidence',transport=httpx2.MockTransport(fail)) as client:
            with pytest.raises(ValueError): await runner.call(client,tmp_path,ctx,row)
        runner.append(tmp_path/'status.jsonl',{'status':'PARTIAL_TECHNICAL','physical_requests':1})
        _,preds,status=await runner.replay(tmp_path)
        assert not preds and status['status']=='PARTIAL_TECHNICAL'
        assert ledger.summary()['reserved_nanodollars']==runner.RESERVATION
        folder=tmp_path/'wire-evidence'/runner.sha(rid.encode())
        (folder/'request.json').write_bytes(b'tampered')
        with pytest.raises(ValueError): await runner.replay(tmp_path)
    asyncio.run(exercise())

def test_complete_mock_replay_rejects_response_tampering(tmp_path,monkeypatch):
    import asyncio, httpx2, json
    from capclaims.flat_grid_study import runner
    async def exercise():
        contexts=[{'person_id':f'x{i}','formal_round':11,'condition':'short3','base_state':BASE,'side':'BUY'} for i in range(72)]
        rows=[{'person_id':f'x{i}','formal_round':11,'condition':'short3','repeat':0,'conditioned_quantity':20} for i in range(72)]
        bodies={}
        for ctx,row in zip(contexts,rows):
            data=await runner.body(ctx,row); bodies[design.request_id(row)]={'sha256':runner.sha(data),'bytes':len(data)}
        plan={'contexts':contexts,'schedule':rows,'bodies':bodies}
        async def valid(_): return plan
        monkeypatch.setattr(runner,'valid_plan',valid)
        (tmp_path/'plan.json').write_bytes(b'plan'); h=runner.sha(b'plan'); (tmp_path/'started').write_text(h+'\n',encoding='ascii')
        ledger=runner.Ledger(tmp_path/'cost-ledger.sqlite',preview_hash=h)
        async def reply(req):
            body=json.loads(await req.aread()); ids=body['questions']['decision']['criteria']; first=next(iter(ids))
            return httpx2.Response(200,json={'model':runner.MODEL,'usage':{'input_tokens':10,'output_tokens':1},'answers':{'decision':{'choice':first,'probabilities':{k:float(k==first) for k in ids}}}})
        async with runner.JevWireClient(api_key='dummy',ledger=ledger,evidence_dir=tmp_path/'wire-evidence',transport=httpx2.MockTransport(reply)) as client:
            async def no_rate(): pass
            client._rate_slot=no_rate
            for i in range(0,72,10):
                await asyncio.gather(*(runner.call(client,tmp_path,ctx,row) for ctx,row in zip(contexts[i:i+10],rows[i:i+10])))
        runner.append(tmp_path/'status.jsonl',{'status':'COMPLETE','physical_requests':72})
        _,preds,status=await runner.replay(tmp_path)
        assert len(preds)==72 and status['status']=='COMPLETE' and all(p['price']==5 for p in preds)
        folder=tmp_path/'wire-evidence'/runner.sha(design.request_id(rows[0]).encode())
        (folder/'response.json').write_bytes(b'{}')
        with pytest.raises(ValueError): await runner.replay(tmp_path)
    asyncio.run(exercise())

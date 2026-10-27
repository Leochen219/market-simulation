from copy import deepcopy
import asyncio
import pytest
from capclaims.sell_semantics import design, runner
from capclaims.flat_grid_study import design as parent

BASE={'own_q_units':100,'own_side':'SELL','partial_orders':[], 'submitted_order_ledger':{'submitted_order_count':0,'remaining_order_slots':1,'submitted_buy_quantity_units':0,'submitted_sell_quantity_units':0,'remaining_sell_submission_capacity_units':100},'public_instructions':{'trading':'price choices are complete $2.50 ticks.'},'history':[{'round':1}],'own_p':42}
def row(i=0, arm='original', repeat=0, condition='short3'):
    return {'person_id':f'x{i}','formal_round':11,'condition':condition,'repeat':repeat,'conditioned_quantity':20,'arm':arm}
def ctx(i=0): return {'person_id':f'x{i}','formal_round':11,'condition':'short3','base_state':BASE,'side':'SELL'}

def test_single_field_and_exact_parent_body():
    async def exercise():
        c=ctx(); before=deepcopy(c); r=row()
        original,opts=design.price_spec(c,20,'original'); rules,opts2=design.price_spec(c,20,'rules')
        assert rules['public_instructions'].pop('economic_definitions')==design.DEFINITIONS
        assert original==rules and opts==opts2 and len(opts)==199 and c==before
        assert list(opts)[0]=='p005' and list(opts)[-1]=='p995'
        assert await runner.body(c,r)==await runner.serialize_body(state=parent.price_spec(c,20)[0],instructions=parent.PROMPT_PRICE,criteria=opts)
        assert 'arm' not in original and 'arm' not in rules
    asyncio.run(exercise())

def test_schedule_and_paired_draws():
    paths=[{k:v for k,v in row(i,repeat=r,condition=c).items() if k!='arm'} for r in (0,1) for c in ('short3','long10') for i in range(8)]
    s=design.schedule(paths); assert s==design.schedule(paths) and len(s)==64
    for repeat in (0,1):
        block=[r for r in s if r['repeat']==repeat]; pairs={}
        for r in block: pairs.setdefault(design.key(r),[]).append(r['arm'])
        assert len(pairs)==16 and sum(a[0]=='original' for a in pairs.values())==8
    ids=[f'p{i:03d}' for i in range(5,1000,5)]; raw={i:0.0 for i in ids}; raw[ids[0]]=raw[ids[1]]=0.5
    assert design.decide(raw,ids,row(0,'original'))==design.decide(raw,ids,row(0,'rules',condition='long10'))
    assert design.decide({i:1/199 for i in ids},ids,row())['status']=='client_numeric_abstain'
    raw.pop(ids[0])
    with pytest.raises(ValueError): design.decide(raw,ids,row())

def test_ledger_cap(tmp_path):
    ledger=runner.Ledger(tmp_path/'ledger.sqlite',preview_hash='a'*64)
    for i in range(64): ledger.reserve(str(i))
    assert ledger.summary()['committed_nanodollars']==64*runner.RESERVATION
    with pytest.raises(ValueError): ledger.reserve('64')

def test_ten_distinct_concurrent_abstentions(tmp_path):
    class Mock:
        halted=False
        def __init__(self): self.n=0; self.active=0; self.peak=0; self.ready=asyncio.Event(); self.ids=set()
        async def send_body(self,*,request_id,body):
            self.ids.add(request_id); self.n+=1; self.active+=1; self.peak=max(self.peak,self.active)
            if self.n==10: self.ready.set()
            await self.ready.wait(); self.active-=1
            return {'answers':{'decision':{'choice':'p005','probabilities':{f'p{i:03d}':1/199 for i in range(5,1000,5)}}}}
    async def exercise():
        m=Mock(); out=await asyncio.gather(*(runner.call(m,tmp_path,ctx(i),row(i)) for i in range(10)))
        assert m.peak==len(m.ids)==10 and all(x['decision']['status']=='client_numeric_abstain' for x in out)
    asyncio.run(exercise())

def test_first_http_failure_survives_partial_replay_and_tamper(tmp_path,monkeypatch):
    import asyncio, httpx2
    from capclaims.sell_semantics import runner
    async def exercise():
        ctx={'person_id':'x','formal_round':11,'condition':'short3','base_state':BASE,'side':'SELL'}
        row={'person_id':'x','formal_round':11,'condition':'short3','repeat':0,'conditioned_quantity':20,'arm':'original'}
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
    from capclaims.sell_semantics import runner
    async def exercise():
        contexts=[{'person_id':f'x{i}','formal_round':11,'condition':'short3','base_state':BASE,'side':'SELL'} for i in range(64)]
        rows=[{'person_id':f'x{i}','formal_round':11,'condition':'short3','repeat':0,'conditioned_quantity':20,'arm':'original'} for i in range(64)]
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
            for i in range(0,64,10):
                await asyncio.gather(*(runner.call(client,tmp_path,ctx,row) for ctx,row in zip(contexts[i:i+10],rows[i:i+10])))
        runner.append(tmp_path/'status.jsonl',{'status':'COMPLETE','physical_requests':64})
        _,preds,status=await runner.replay(tmp_path)
        assert len(preds)==64 and status['status']=='COMPLETE' and all(p['price']==5 for p in preds)
        folder=tmp_path/'wire-evidence'/runner.sha(design.request_id(rows[0]).encode())
        (folder/'response.json').write_bytes(b'{}')
        with pytest.raises(ValueError): await runner.replay(tmp_path)
    asyncio.run(exercise())

def test_missing_key_does_not_start_or_send(tmp_path,monkeypatch):
    import dotenv
    async def valid(_): return {'contexts': [], 'schedule': []}
    monkeypatch.setattr(runner,'valid_plan',valid)
    monkeypatch.setattr(dotenv,'load_dotenv',lambda *a,**kw: None)
    monkeypatch.delenv('TYPESAFE_API_KEY',raising=False)
    monkeypatch.setattr(runner,'JevWireClient',lambda **kw: pytest.fail('client should not start'))
    with pytest.raises(ValueError,match='provider key missing'): asyncio.run(runner.run(tmp_path))
    assert not (tmp_path/'started').exists() and not (tmp_path/'cost-ledger.sqlite').exists()

def test_predispatch_construction_failure_replays(tmp_path,monkeypatch):
    async def exercise():
        c=ctx(); r=row(); rid=design.request_id(r); data=await runner.body(c,r)
        plan={'contexts':[c],'schedule':[r],'bodies':{rid:{'sha256':runner.sha(data),'bytes':len(data)}}}
        async def valid(_): return plan
        monkeypatch.setattr(runner,'valid_plan',valid)
        (tmp_path/'plan.json').write_bytes(b'plan'); h=runner.sha(b'plan'); (tmp_path/'started').write_text(h+'\n',encoding='ascii')
        runner.Ledger(tmp_path/'cost-ledger.sqlite',preview_hash=h)
        original=runner.body
        async def bad(*args): raise ValueError('construction error')
        monkeypatch.setattr(runner,'body',bad)
        class Client: halted=False
        client=Client()
        with pytest.raises(ValueError): await runner.call(client,tmp_path,c,r)
        assert client.halted
        runner.append(tmp_path/'status.jsonl',{'status':'PARTIAL_TECHNICAL','physical_requests':0,'phase':'dispatch'})
        monkeypatch.setattr(runner,'body',original)
        _,pred,status=await runner.replay(tmp_path)
        assert not pred and status['status']=='PARTIAL_TECHNICAL'
    asyncio.run(exercise())

def test_client_start_failure_records_partial(tmp_path,monkeypatch):
    import dotenv
    async def exercise():
        async def valid(_): return {'contexts': [], 'schedule': []}
        monkeypatch.setattr(runner,'valid_plan',valid)
        monkeypatch.setattr(dotenv,'load_dotenv',lambda *a,**kw: None)
        monkeypatch.setenv('TYPESAFE_API_KEY','dummy')
        (tmp_path/'plan.json').write_bytes(b'plan')
        class Broken:
            def __init__(self,**kw): raise ValueError('client init')
        monkeypatch.setattr(runner,'JevWireClient',Broken)
        with pytest.raises(ValueError,match='client init'): await runner.run(tmp_path)
        _,pred,status=await runner.replay(tmp_path)
        assert status['phase']=='client_start' and status['status']=='PARTIAL_TECHNICAL' and not pred
    asyncio.run(exercise())

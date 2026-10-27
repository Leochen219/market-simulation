import asyncio,hashlib,json,copy
import pytest
from capclaims.long_history import source,design,runner,analysis
from capclaims.grid_sampling_v3 import decode
from tools.probes import jev_grid_probability as v1


@pytest.mark.requires_participant_data
def test_source_window_matching_and_target_separation():
    cases,labels,meta=source.load();contexts,excluded=design.cohort(cases);assert len(cases)==18 and len(labels)==18 and excluded==0 and len(contexts)==36 and len(design.schedule(contexts))==72
    for person in {c['person_id'] for c in contexts}:
        short=next(c for c in contexts if c['person_id']==person and c['condition']=='short3');long=next(c for c in contexts if c['person_id']==person and c['condition']=='long10')
        assert short['current_state']==long['current_state'] and [x['round'] for x in short['history']]==[8,9,10] and [x['round'] for x in long['history']]==list(range(1,11))
        assert short['base_state']['round']==long['base_state']['round']==16 and long['base_state']['formal_round']==11
        assert [x['round'] for x in long['base_state']['history']]==list(range(6,16))
        assert 'orders' not in long['current_state'] and 'profit_cents' not in long['current_state']
    before=copy.deepcopy(contexts); labels[next(iter(labels))]['orders']=[]; assert design.cohort(cases)[0]==before


@pytest.mark.requires_participant_data
def test_seed_excludes_condition_and_body_identity():
    cases,labels,_=source.load();contexts,_=design.cohort(cases);s=contexts[0];l=next(c for c in contexts if c['person_id']==s['person_id'] and c['condition']!=s['condition']);row={'person_id':s['person_id'],'formal_round':11,'repeat':0,'condition':'short3'}
    a=runner.decisions({'a':.6,'b':.4},['a','b'],row,'quantity');b=runner.decisions({'a':.6,'b':.4},['a','b'],dict(row,condition='long10'),'quantity')
    assert a['cutoff3']['draw_sha256']==b['cutoff3']['draw_sha256'] and a['cutoff5']['draw_sha256']==a['cutoff3']['draw_sha256']
    assert runner.rid(row,'quantity')!=runner.rid(dict(row,condition='long10'),'quantity')
    assert s['base_state']['own_p']==l['base_state']['own_p'] and s['base_state']['history_window_n']!=l['base_state']['history_window_n']
    assert v1.quantity_grid('SELL',37)==[10,20,30] and v1.price_grid(301,399)[-1]==399


def test_baseline_legal_last_and_nearest():
    ctx={'side':'SELL','current_state':{'p':50,'Q':20,'D':10},'history':[{'round':8,'p':50,'Q':20,'D':10,'orders':[{'side':'sell','qty':21,'price':7}]},{'round':9,'p':50,'Q':20,'D':10,'orders':[{'side':'sell','qty':10,'price':8},{'side':'sell','qty':10,'price':9}]},{'round':10,'p':50,'Q':20,'D':10,'orders':[{'side':'buy','qty':1,'price':1}]}]}
    assert [o['price'] for o in analysis.baseline(ctx,'last')]==[8,9]
    assert [o['price'] for o in analysis.baseline(ctx,'nearest')]==[8,9]
    ctx['current_state']['Q']=5; assert analysis.baseline(ctx,'last') is None


def test_batch_repeat_barrier_and_ledger_cap(tmp_path,monkeypatch):
    rows=[{'repeat':r,'x':i} for r in (0,1) for i in range(36)];batches=list(runner.batches(rows));assert [len(x) for x in batches]==[10,10,10,6]*2 and all(len({x['repeat'] for x in b})==1 for b in batches)
    monkeypatch.setattr(runner,'LIMIT',2); ledger=runner.Ledger(tmp_path/'ledger.sqlite',preview_hash='a'*64); ledger.reserve('a');ledger.reserve('b')
    with pytest.raises(ValueError):ledger.reserve('c')

class FakeClient:
    def __init__(self,root,empty=False,invalid=False):
        self.root=root;self.empty=empty;self.invalid=invalid;self.halted=False
        self.ledger=runner.Ledger(root/'cost-ledger.sqlite',preview_hash=runner.sha((root/'plan.json').read_bytes()) if (root/'plan.json').exists() else 'a'*64)
    async def send_body(self,request_id,body):
        request=json.loads(body);ids=list(request['questions']['decision']['criteria']);raw={k:float(i==0) for i,k in enumerate(ids)}
        if self.empty and request_id.endswith('quantity'):raw={k:1/len(ids) for k in ids}
        if self.invalid:raw={k:0.0 for k in ids}
        answer={'model':runner.MODEL,'usage':{'input_tokens':10,'output_tokens':1},'answers':{'decision':{'choice':ids[0],'probabilities':raw}}};response=runner.canonical(answer)
        folder=self.root/'wire-evidence'/hashlib.sha256(request_id.encode()).hexdigest();folder.mkdir(parents=True,exist_ok=True)
        self.ledger.reserve(request_id);self.ledger.settle(request_id,10);(folder/'request.json').write_bytes(body);(folder/'response.json').write_bytes(response)
        attempt=runner.canonical({'request_id':request_id,'request_sha256':runner.sha(body),'method':'POST','url':'https://api.typesafe.ai/v1/systemone','body_bytes':len(body),'reserved_bytes':runner.MAX_BODY,'reserved_nanodollars':runner.RESERVATION,'long_context':False,'compression':False,'retry_count':0})
        timing=runner.canonical({'status_code':200,'started_monotonic':1,'finished_monotonic':2,'elapsed_seconds':1});(folder/'attempt.json').write_bytes(attempt);(folder/'timing.json').write_bytes(timing)
        return {**answer,'_wire':{'request_sha256':runner.sha(body),'response_sha256':runner.sha(response),'attempt_sha256':runner.sha(attempt),'timing_sha256':runner.sha(timing)}}


@pytest.mark.requires_participant_data
def test_empty_support_continues_other_context(tmp_path):
    cases,_,_=source.load();contexts,_=design.cohort(cases);ctx=next(c for c in contexts if c['side']=='BUY');row=design.schedule([ctx])[0]
    async def work():
        a=await runner.chain(FakeClient(tmp_path,empty=True),tmp_path,ctx,row)
        b=await runner.chain(FakeClient(tmp_path),tmp_path,ctx,dict(row,repeat=1))
        return a,b
    a,b=asyncio.run(work());assert a['path_status']=='QUANTITY_FAILURE' and b['path_status']=='QUERIED'
    assert len((tmp_path/'trace.jsonl').read_text(encoding='utf-8').splitlines())==7


@pytest.mark.requires_participant_data
def test_invalid_stops_transport_and_records_technical(tmp_path):
    cases,_,_=source.load();ctx=design.cohort(cases)[0][0];row=design.schedule([ctx])[0];client=FakeClient(tmp_path,invalid=True)
    with pytest.raises(ValueError):asyncio.run(runner.chain(client,tmp_path,ctx,row))
    assert client.halted and json.loads((tmp_path/'technical.jsonl').read_text(encoding='utf-8'))['stage']=='quantity'


@pytest.mark.requires_participant_data
def test_complete_four_branches_replay_and_tamper(tmp_path,monkeypatch):
    cases,_,_=source.load();ctx=design.cohort(cases)[0][0];row=design.schedule([ctx])[0];(tmp_path/'plan.json').write_text('{}')
    asyncio.run(runner.chain(FakeClient(tmp_path),tmp_path,ctx,row));(tmp_path/'started').write_text(runner.sha((tmp_path/'plan.json').read_bytes()));(tmp_path/'status.jsonl').write_text(json.dumps({'status':'PARTIAL_TECHNICAL'})+'\n')
    monkeypatch.setattr(runner,'valid_plan',lambda root:{'contexts':[ctx],'schedule':[row]})
    _,pred,_=runner.replay(tmp_path);assert len(pred)==1 and len(pred[0]['request_hashes'])==6 and pred[0]['condition']==ctx['condition']
    lines=(tmp_path/'trace.jsonl').read_text(encoding='utf-8').splitlines();changed=json.loads(lines[-1]);changed['step']['decisions']['argmax']['selected_id']='p999';lines[-1]=json.dumps(changed);(tmp_path/'trace.jsonl').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    with pytest.raises(ValueError,match='decoder'):runner.replay(tmp_path)


def test_actual_transport_ten_distinct_concurrent(tmp_path):
    import httpx2
    async def work():
        arrived=set();bodies={};event=asyncio.Event()
        async def handler(req):
            raw=await req.aread();ident=json.loads(raw)['state']['input_id'];arrived.add(ident);bodies[ident]=raw
            if len(arrived)==10:event.set()
            await asyncio.wait_for(event.wait(),15)
            return httpx2.Response(200,json={'model':runner.MODEL,'usage':{'input_tokens':10,'output_tokens':1},'answers':{'decision':{'type':'choice','choice':'a','probabilities':{'a':1}}}})
        ledger=runner.Ledger(tmp_path/'real.sqlite',preview_hash='a'*64)
        async with runner.JevWireClient(api_key='test-only',ledger=ledger,evidence_dir=tmp_path/'wire-real',transport=httpx2.MockTransport(handler)) as client:
            await asyncio.gather(*(client.choose(request_id=f'id{i}',state={'input_id':i},instructions='Choose',criteria={'a':'A'}) for i in range(10)))
            assert client.peak==10
        assert len(arrived)==10 and len(set(bodies.values()))==10 and ledger.summary()['physical_requests']==10
    asyncio.run(work())

def test_primary_two_way_mask_differs_from_secondary_six_way():
    people=[('a','BUY','shortage'),('b','SELL','surplus')];contexts=[];targets={};pred=[];schedule=[]
    for person,side,stratum in people:
        targets[person]={'round':11,'orders':[{'side':side.lower(),'qty':10,'price':10}]}
        for cond in design.CONDITIONS:
            contexts.append({'person_id':person,'formal_round':11,'condition':cond,'side':side,'stratum':stratum,'current_state':{'p':10,'Q':20,'D':5},'history':[]})
            for repeat in (0,1):
                row={'person_id':person,'condition':cond,'repeat':repeat};schedule.append(row)
                q={arm:10 for arm in design.ARMS};prices={arm:10 for arm in design.ARMS};reasons={arm:None for arm in design.ARMS}
                if person=='a' and cond=='short3' and repeat==0:prices['cutoff5']=None;reasons['cutoff5']='selected_price_decoder'
                pred.append({**row,'formal_round':11,'side':side,'stratum':stratum,'quantities':q,'prices':prices,'price_failure_reason':reasons,'conditioned_quantity':10,'request_hashes':{},'probabilities':{}})
    result=analysis.summarize({'contexts':contexts,'schedule':schedule},pred,targets)
    buy=result['groups']['BUY'];assert buy['metrics']['price']['common_success_slots']==2 and buy['secondary_price_six_way_common']['common_success_slots']==1
    assert buy['metrics']['price']['conditions']['short3']['raw_w1']==0
    assert result['failures']['short3']['cutoff5']['selected_price_decoder_failure_n']==1

def test_pooled_history_balances_person_then_round_then_order():
    people=[[ [{'qty':10},{'qty':20}], [{'qty':30}] ],[ [{'qty':100}] ]]
    points=analysis.pool_weighted(people,'qty')
    assert sum(w for v,w in points if v in (10,20,30))==pytest.approx(.5)
    assert sum(w for v,w in points if v==100)==pytest.approx(.5)
    ctx={'side':'SELL','current_state':{'p':50,'Q':5,'D':1},'history':[{'round':1,'p':50,'Q':5,'D':1,'orders':[{'side':'sell','qty':6,'price':1}]}]}
    assert analysis.history_pool(ctx)==[]

@pytest.mark.requires_participant_data
def test_first_http_failure_replays_unknown_and_rejects_tamper(tmp_path,monkeypatch):
    import httpx2
    cases,_,_=source.load();ctx=design.cohort(cases)[0][0];row=design.schedule([ctx])[0]
    (tmp_path/'plan.json').write_text('{}');(tmp_path/'started').write_text(runner.sha((tmp_path/'plan.json').read_bytes()))
    async def work():
        async def provider(req):return httpx2.Response(503,json={'error':'mock'})
        ledger=runner.Ledger(tmp_path/'cost-ledger.sqlite',preview_hash=runner.sha((tmp_path/'plan.json').read_bytes()))
        async with runner.JevWireClient(api_key='test-only',ledger=ledger,evidence_dir=tmp_path/'wire-evidence',transport=httpx2.MockTransport(provider)) as client:
            with pytest.raises(ValueError):await runner.chain(client,tmp_path,ctx,row)
        assert ledger.summary()['reserved_nanodollars']==runner.RESERVATION
    asyncio.run(work());assert not (tmp_path/'trace.jsonl').exists()
    (tmp_path/'status.jsonl').write_text(json.dumps({'status':'PARTIAL_TECHNICAL','physical_requests':1})+'\n')
    monkeypatch.setattr(runner,'valid_plan',lambda root:{'contexts':[ctx],'schedule':[row]})
    _,pred,status=runner.replay(tmp_path);assert pred==[] and status['status']=='PARTIAL_TECHNICAL'
    event=json.loads((tmp_path/'technical.jsonl').read_text(encoding='utf-8'));folder=tmp_path/'wire-evidence'/hashlib.sha256(event['request_id'].encode()).hexdigest()
    original=(folder/'response.json').read_bytes(); (folder/'response.json').write_text('{}')
    with pytest.raises(ValueError,match='technical evidence hash mismatch'):runner.replay(tmp_path)
    (folder/'response.json').write_bytes(original); (folder/'request.json').write_text('{}')
    with pytest.raises(ValueError,match='technical evidence hash mismatch'):runner.replay(tmp_path)

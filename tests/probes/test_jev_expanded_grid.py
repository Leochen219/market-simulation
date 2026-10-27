import asyncio,hashlib,json
import pytest
from capclaims.behavioral_memory.source import build_authorized_cases
from capclaims.expanded_grid import design,runner,analysis
from capclaims.grid_sampling_v3 import decode
from tools.probes import jev_grid_probability as v1


@pytest.mark.requires_participant_data
def test_cohort_three_history_and_equal_exclusion():
    cases,labels,_=build_authorized_cases(); contexts,targets=design.cohort(cases,labels)
    assert len(contexts)==89 and len(targets)==89 and len(design.schedule(contexts))==178
    assert sum(c['stratum']=='shortage' for c in contexts)==45 and sum(c['stratum']=='surplus' for c in contexts)==44
    assert all([x['round'] for x in c['history']]==list(range(c['formal_round']-3,c['formal_round'])) for c in contexts)
    assert all('orders' not in c['current_state'] and 'profit_cents' not in c['current_state'] for c in contexts)
    assert all(c['base_state']['formal_round']==c['formal_round'] and c['base_state']['round']==c['formal_round']+5 for c in contexts)
    assert len({(x['person_id'],x['formal_round'],x['repeat']) for x in design.schedule(contexts)})==178


def test_grid_boundaries_and_draw_coordinates():
    assert v1.quantity_grid('BUY',0)[-1]==990 and v1.quantity_grid('SELL',37)==[10,20,30]
    assert v1.price_grid(1,100)[0]==1 and v1.price_grid(301,399)[-1]==399
    a=decode({'a':.6,'b':.4},['a','b'],seed=design.SEED,person_id=design.coordinate('p',9),repeat=0,stage='quantity')
    b=decode({'a':.6,'b':.4},['a','b'],seed=design.SEED,person_id=design.coordinate('p',10),repeat=0,stage='quantity')
    assert a['draw_sha256']!=b['draw_sha256']


def test_participant_round_weights():
    points=analysis.balanced([[[1,2],[3]],[[4]]])
    assert sum(w for x,w in points if x in (1,2,3))==pytest.approx(.5)
    assert sum(w for x,w in points if x==4)==pytest.approx(.5)


def test_ledger_duplicate_and_cap(tmp_path,monkeypatch):
    monkeypatch.setattr(runner,'LIMIT',2)
    ledger=runner.Ledger(tmp_path/'ledger.sqlite',preview_hash='a'*64)
    ledger.reserve('a'); ledger.reserve('b')
    with pytest.raises(ValueError): ledger.reserve('c')
    with pytest.raises(ValueError): ledger.reserve('a')

class FakeClient:
    def __init__(self,root,empty=False,invalid=False): self.root=root; self.empty=empty; self.invalid=invalid; self.halted=False; self.active=0; self.max_active=0; self.ledger=runner.Ledger(root/'cost-ledger.sqlite',preview_hash=runner.sha((root/'plan.json').read_bytes()) if (root/'plan.json').exists() else 'a'*64)
    async def send_body(self,request_id,body):
        self.active+=1; self.max_active=max(self.max_active,self.active); await asyncio.sleep(.001)
        request=json.loads(body); ids=list(request['questions']['decision']['criteria']); raw={k:float(i==0) for i,k in enumerate(ids)}
        if self.empty and request_id.endswith('quantity'): raw={k:1/len(ids) for k in ids}
        if self.invalid: raw={k:0.0 for k in ids}
        answer={'model':runner.MODEL,'usage':{'input_tokens':10,'output_tokens':1},'answers':{'decision':{'choice':ids[0],'probabilities':raw}}}
        response=runner.canonical(answer); folder=self.root/'wire-evidence'/hashlib.sha256(request_id.encode()).hexdigest();folder.mkdir(parents=True,exist_ok=True)
        self.ledger.reserve(request_id); self.ledger.settle(request_id,10)
        (folder/'request.json').write_bytes(body);(folder/'response.json').write_bytes(response)
        attempt=runner.canonical({'request_id':request_id,'request_sha256':runner.sha(body),'method':'POST','url':'https://api.typesafe.ai/v1/systemone','body_bytes':len(body),'reserved_bytes':runner.MAX_BODY,'reserved_nanodollars':runner.RESERVATION,'long_context':False,'compression':False,'retry_count':0})
        timing=runner.canonical({'status_code':200,'started_monotonic':1,'finished_monotonic':2,'elapsed_seconds':1})
        (folder/'attempt.json').write_bytes(attempt); (folder/'timing.json').write_bytes(timing); self.active-=1
        return {**answer,'_wire':{'request_sha256':runner.sha(body),'response_sha256':runner.sha(response),'attempt_sha256':runner.sha(attempt),'timing_sha256':runner.sha(timing)}}


@pytest.mark.requires_participant_data
def test_empty_support_skips_branch_and_other_paths_continue(tmp_path):
    cases,labels,_=build_authorized_cases(); contexts,_=design.cohort(cases,labels); ctx=next(c for c in contexts if c['side']=='BUY'); row=next(x for x in design.schedule([ctx]) if x['repeat']==0)
    async def work():
        empty=FakeClient(tmp_path,True); first=await runner.chain(empty,tmp_path,ctx,row)
        other=dict(row,repeat=1); second=await runner.chain(FakeClient(tmp_path),tmp_path,ctx,other)
        return first,second
    first,second=asyncio.run(work()); assert first['path_status']=='QUANTITY_FAILURE' and second['path_status']=='QUERIED'
    assert len((tmp_path/'trace.jsonl').read_text(encoding='utf-8').splitlines())==7


@pytest.mark.requires_participant_data
def test_concurrency_and_complete_branch_replay_tamper(tmp_path,monkeypatch):
    cases,labels,_=build_authorized_cases(); contexts,_=design.cohort(cases,labels); ctx=contexts[0]; row=design.schedule([ctx])[0]
    (tmp_path/'plan.json').write_text('{}')
    asyncio.run(runner.chain(FakeClient(tmp_path),tmp_path,ctx,row))
    (tmp_path/'started').write_text(runner.sha((tmp_path/'plan.json').read_bytes()))
    (tmp_path/'status.jsonl').write_text(json.dumps({'status':'PARTIAL_TECHNICAL'})+'\n')
    monkeypatch.setattr(runner,'valid_plan',lambda root:{'contexts':[ctx],'schedule':[row]})
    _,pred,_=runner.replay(tmp_path); assert len(pred)==1 and len(pred[0]['request_hashes'])==6
    lines=(tmp_path/'trace.jsonl').read_text(encoding='utf-8').splitlines(); altered=json.loads(lines[-1]); altered['step']['decisions']['argmax']['selected_id']='p999';lines[-1]=json.dumps(altered)
    (tmp_path/'trace.jsonl').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    with pytest.raises(ValueError,match='decoder'): runner.replay(tmp_path)


@pytest.mark.requires_participant_data
def test_invalid_probabilities_halt_and_record_technical_event(tmp_path):
    cases,labels,_=build_authorized_cases(); contexts,_=design.cohort(cases,labels); ctx=contexts[0]; row=design.schedule([ctx])[0]; client=FakeClient(tmp_path,invalid=True)
    with pytest.raises(ValueError): asyncio.run(runner.chain(client,tmp_path,ctx,row))
    assert client.halted is True
    record=json.loads((tmp_path/'technical.jsonl').read_text(encoding='utf-8'))
    assert record['stage']=='quantity' and record['error_type']=='ValueError'
    assert not (tmp_path/'trace.jsonl').exists()


def test_batches_do_not_cross_repeat_boundary():
    rows=[{'repeat':repeat,'n':i} for repeat in (0,1) for i in range(89)]
    batches=list(runner.batches(rows))
    assert [len(b) for b in batches]==[10]*8+[9]+[10]*8+[9]
    assert all(len({r['repeat'] for r in batch})==1 for batch in batches)

def test_actual_wire_client_overlaps_ten_distinct_requests(tmp_path):
    import httpx2
    async def work():
        arrived=set(); bodies={}; barrier=asyncio.Event()
        async def provider(request):
            raw=await request.aread(); ident=json.loads(raw)['state']['input_id']; arrived.add(ident); bodies[ident]=raw
            if len(arrived)==10: barrier.set()
            await asyncio.wait_for(barrier.wait(),15)
            return httpx2.Response(200,json={'model':runner.MODEL,'usage':{'input_tokens':10,'output_tokens':1},'answers':{'decision':{'type':'choice','choice':'a','probabilities':{'a':1}}}})
        ledger=runner.Ledger(tmp_path/'actual.sqlite',preview_hash='a'*64)
        async with runner.JevWireClient(api_key='test-only',ledger=ledger,evidence_dir=tmp_path/'actual-wire',transport=httpx2.MockTransport(provider)) as client:
            await asyncio.gather(*(client.choose(request_id=f'i-{i}',state={'input_id':i},instructions='Choose',criteria={'a':'A'}) for i in range(10)))
            assert client.peak==10
        assert len(arrived)==10 and len(set(bodies.values()))==10 and ledger.summary()['physical_requests']==10
    asyncio.run(work())

def test_analysis_common_mask_failures_and_bounds():
    contexts=[{'person_id':p,'formal_round':9,'side':'BUY','stratum':'shortage','current_state':{'Q':20,'p':10}} for p in ('a','b')]
    plan={'contexts':contexts,'schedule':[{'person_id':p,'formal_round':9,'repeat':r} for p in ('a','b') for r in (0,1)]}
    targets={design.coordinate(p,9):{'orders':[{'side':'buy','qty':10,'price':10}]} for p in ('a','b')}
    preds=[]
    for row in plan['schedule']:
        prices={arm:10 for arm in design.ARMS}; reasons={arm:None for arm in design.ARMS}
        if row['person_id']=='a' and row['repeat']==0: prices['cutoff5']=None;reasons['cutoff5']='selected_price_decoder'
        preds.append({**row,'side':'BUY','stratum':'shortage','quantities':{arm:10 for arm in design.ARMS},'prices':prices,'conditioned_quantity':10,'price_failure_reason':reasons,'request_hashes':{},'probabilities':{}})
    result=analysis.summarize(plan,preds,targets); metric=result['groups']['BUY']['metrics']['price']
    assert metric['common_success_slots']==3 and metric['common_success_people']==2
    assert result['failure_counts']['cutoff5']['selected_price_decoder_failure_n']==1
    assert result['failure_counts']['cutoff5']['quantity_decoder_failure_n']==0
    assert metric['arms']['cutoff5']['full_eligible_price_missing_weight']==pytest.approx(.25)
    assert metric['arms']['argmax']['raw_w1']==pytest.approx(0)

def test_no_common_price_still_reports_each_full_eligible_bound():
    contexts=[{'person_id':'p','formal_round':9,'side':'BUY','stratum':'shortage','current_state':{'Q':20,'p':10}}]
    plan={'contexts':contexts,'schedule':[{'person_id':'p','formal_round':9,'repeat':r} for r in (0,1)]}
    targets={design.coordinate('p',9):{'orders':[{'side':'buy','qty':10,'price':10}]}}; preds=[]
    for row in plan['schedule']:
        preds.append({**row,'side':'BUY','stratum':'shortage','quantities':{arm:10 for arm in design.ARMS},'prices':{'argmax':10,'cutoff5':10,'cutoff3':None},'conditioned_quantity':10,'price_failure_reason':{'argmax':None,'cutoff5':None,'cutoff3':'band_decoder'},'request_hashes':{},'probabilities':{}})
    price=analysis.summarize(plan,preds,targets)['groups']['BUY']['metrics']['price']
    assert price['common_success_slots']==0
    assert price['arms']['argmax']['raw_w1'] is None and price['arms']['argmax']['full_eligible_price_missing_weight']==0
    assert price['arms']['argmax']['full_eligible_price_w1_loose_bounds']==pytest.approx([0,0])
    assert price['arms']['cutoff3']['full_eligible_price_missing_weight']==1
    assert len(price['arms']['cutoff3']['full_eligible_price_w1_loose_bounds'])==2

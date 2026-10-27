import asyncio, hashlib, json, math
from pathlib import Path
import pytest
from tools.probes.jev_grid_probability import BANDS, cents, choose, normalized, price_grid, quantity_grid, summarize, GridLedger, RESERVATION, BUDGET, MAX_REQUESTS, select_cases, make_state, _chain, strict_json, sha

def test_grids_are_complete_and_band_partition():
    assert len(quantity_grid('BUY', 12)) == 99
    assert quantity_grid('BUY',12)[-1] == 990
    assert len(price_grid(*BANDS[0])) == 100
    assert [t for band in BANDS for t in price_grid(*band)] == list(range(1,400))
    assert cents(1)==250 and cents(399)==99750

def test_sell_rounddown_and_argmax_stable():
    assert quantity_grid('SELL', 19)==[10]
    assert quantity_grid('SELL', 0)==[]
    assert choose(['a','b','c'],{'a':.4,'b':.4,'c':.2})=='a'

def test_normalization_preserves_threshold_raw_values():
    raw={'a':.10,'b':.90}; valid,meta=normalized(raw,['a','b'])
    assert valid==raw and meta['rounding_normalized'] is False
    stats=summarize([raw],[2])['per_distribution'][0]
    assert stats['mass_ge_10']==1
    assert summarize([{'a':0,'b':1}],[2])['per_distribution'][0]['empty_support']==1
    assert summarize([{'a':.5,'b':.5}],[2])['per_distribution'][0]['max_ties']==2
    assert summarize([{'a':.01,'b':.02}],[2])['per_distribution'][0]['empty_after_cutoff_3']==1
    raw={'a':.10,'b':.91}; valid,meta=normalized(raw,['a','b'])
    assert meta['rounding_normalized'] is True and valid['a']<.10
    raw_stats=summarize([raw],[2])['per_distribution'][0]
    assert raw_stats['count_ge_10']==2 and raw_stats['mass_ge_10']==1.01

def test_selection_is_stable_and_ignores_any_labels():
    cases=[]
    for i in range(12):
        cases.append({'person_id':f'p{i}','history':[],'current_state':{'Q':i,'D':i+1},'target_labels':{'should_not_be_used':i}})
    for i in range(5):
        cases.append({'person_id':f's{i}','history':[],'current_state':{'Q':20+i,'D':2},'target_labels':{'should_not_be_used':i}})
    before=json.dumps(cases,sort_keys=True)
    selected=select_cases(cases)
    assert len(selected)==10 and len({c['person_id'] for c in selected})==10
    assert [c['person_id'] for c in selected]==[c['person_id'] for c in select_cases(cases)]
    assert json.dumps(cases,sort_keys=True)==before

def test_ten_distinct_async_requests_overlap_and_reorder_invariance(tmp_path):
    import httpx2
    from tools.probes.jev_grid_probability import JevWireClient
    async def execute(order,folder):
        arrived=set(); barrier=asyncio.Event()
        async def provider(request):
            data=json.loads(await request.aread()); ident=data['state']['input_id']; arrived.add(ident)
            if len(arrived)==10: barrier.set()
            await asyncio.wait_for(barrier.wait(),5)
            return httpx2.Response(200,json={'model':'jev-1.14.0','usage':{'input_tokens':2,'output_tokens':1},
                'answers':{'decision':{'type':'choice','choice':'a','probabilities':{'a':1.0}}}})
        ledger=GridLedger(folder/'ledger.sqlite',preview_hash='b'*64)
        async with JevWireClient(api_key='offline-test-key',ledger=ledger,evidence_dir=folder/'wire',transport=httpx2.MockTransport(provider)) as client:
            result=await asyncio.gather(*(client.choose(request_id=f'async-{i}',state={'input_id':i},instructions='Choose.',criteria={'a':'A'}) for i in order))
            assert client.peak==10
        return arrived,{json.loads((folder/'wire'/hashlib.sha256(f'async-{i}'.encode()).hexdigest()/'response.json').read_text())['answers']['decision']['choice'] for i in range(10)}
    first=asyncio.run(execute(list(range(10)),tmp_path/'first'))
    second=asyncio.run(execute(list(reversed(range(10))),tmp_path/'second'))
    assert first==second and first[0]==set(range(10))

def test_ledger_independent_budget_and_physical_cap(tmp_path):
    ledger=GridLedger(tmp_path/'ledger.sqlite',preview_hash='a'*64,budget_nanodollars=BUDGET)
    ledger.reserve('one'); ledger.mark_unknown('one','Timeout')
    assert ledger.summary()['reserved_nanodollars']==RESERVATION
    with pytest.raises(ValueError): ledger.reserve('one')
    from datetime import datetime,timezone
    now=datetime.now(timezone.utc).isoformat()
    with ledger._transaction() as db:
        db.executemany('INSERT INTO attempts VALUES (?,?,?,?,?,?,?)',[(f'r{i}','human','reserved',RESERVATION,None,now,None) for i in range(1,MAX_REQUESTS-1)])
    ledger.reserve('last')
    with pytest.raises(ValueError,match='60-request'): ledger.reserve('overflow')

def test_quantity_band_price_chain_uses_only_selected_order_quantity(tmp_path):

    case={'person_id':'synthetic-shortage','current_state':{'round':13,'p':250,'w':10,'Q':37,'D':40},
        'history':[{'round':r,'p':240,'Q':37,'D':40,'orders':[],'profit_cents':0} for r in range(6,13)]}
    base=make_state(case,'BUY')
    assert base['own_q_units']==37 and 'own_qty_units' not in base and 'model_chosen_quantity' not in base
    class MockClient:
        def __init__(self): self.requests=[]
        async def send_body(self,*,request_id,body):
            request=strict_json(body); self.requests.append(request)
            ids=list(request['questions']['decision']['criteria'])
            chosen=next((x for x in ids if x in {'q10','b2','p201'}),ids[0])
            response={'model':'jev-1.14.0','usage':{'input_tokens':1,'output_tokens':1},
                'answers':{'decision':{'type':'choice','choice':chosen,'probabilities':{x:float(x==chosen) for x in ids}}}}
            raw=json.dumps(response,ensure_ascii=False,separators=(',',':'),sort_keys=True,allow_nan=False).encode()
            return {**response,'_wire':{'request_id':request_id,'request_sha256':sha(body),'response_sha256':sha(raw)}}
    client=MockClient(); context={'person_id':case['person_id'],'side':'BUY','base_state':base}
    asyncio.run(_chain(client,tmp_path,context,0,0))
    assert len(client.requests)==3
    quantity_state,band_state,price_state=[r['state'] for r in client.requests]
    assert quantity_state['own_q_units']==37 and 'own_qty_units' not in quantity_state
    assert band_state['own_q_units']==37 and band_state['own_qty_units']==10 and band_state['model_chosen_quantity']==10
    assert price_state['own_q_units']==37 and price_state['own_qty_units']==10 and price_state['model_chosen_quantity']==10
    assert 'model_chosen_band' not in band_state and price_state['model_chosen_band']==2
    assert price_state['own_qty_units']!=price_state['own_q_units']

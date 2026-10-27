import asyncio
from copy import deepcopy
import pytest
from capclaims import flat_price_grid as g


def state():
    return {'own_q_units': 100, 'own_d_units': 20, 'history': [{'round': 1}], 'submitted_order_ledger': {'submitted_sell_quantity_units': 30, 'submitted_buy_quantity_units': 0, 'submitted_order_count': 1, 'remaining_sell_submission_capacity_units': 70, 'remaining_order_slots': 1}, 'partial_orders': [{'slot': 0, 'side': 'SELL', 'quantity_units': 30, 'limit_price_dollars_per_unit': 50}], 'public_instructions': {'trading': 'old $2.50 ticks'}, 'model_chosen_band': 2}


def test_grid_and_state():
    s=state(); original=deepcopy(s)
    q, opts=g.stage_spec(s,'SELL','quantity')
    assert list(opts)==['q'+str(i) for i in range(10,71,10)]
    p, prices=g.stage_spec(s,'SELL','price',50)
    assert len(prices)==199 and prices['p005']=='5.00' and prices['p995']=='995.00'
    assert g.PRICE_VALUES['p005']==5 and g.PRICE_VALUES['p995']==995
    assert 'model_chosen_band' not in p and p['model_chosen_quantity']==50
    assert p['history']==s['history'] and p['submitted_order_ledger']==s['submitted_order_ledger']
    assert '2.50' not in p['public_instructions']['trading'] and s==original


def test_decode_empty_and_invalid():
    ids=list(g.PRICE_VALUES); raw={i:1/199 for i in ids}
    assert g.decode_choice(raw,ids,seed='x',person_id='a',repeat=0,stage='price')['status']=='client_stop'
    raw[ids[0]]=float('nan')
    with pytest.raises(ValueError): g.decode_choice(raw,ids,seed='x',person_id='a',repeat=0,stage='price')


def test_replay_and_concurrency():
    async def run():
        client=g.SyntheticClient(barrier=10)
        paths=[{'person_id':str(i),'repeat':0,'side':'BUY','state':dict(state(), marker=i)} for i in range(10)]
        results=await g.run_batch(client,paths,seed='x')
        assert client.peak==10 and all(len(x['steps'])==2 for x in results)
        assert await g.replay(paths,results,seed='x')==results
        bad=deepcopy(results); bad[0]['steps'][0]['decision']['selected_id']='q990'
        with pytest.raises(ValueError): await g.replay(paths,bad,seed='x')
    asyncio.run(run())


def test_boundaries_and_cutoff():
    assert len(g.stage_spec(state(), 'BUY', 'quantity')[1]) == 99
    with pytest.raises(ValueError): g.stage_spec(state(), 'SELL', 'price', 80)
    ids=list(g.PRICE_VALUES)
    raw={i:0.0 for i in ids}; raw[ids[0]]=0.03; raw[ids[1]]=0.97
    decision=g.decode_choice(raw,ids,seed='s',person_id='p',repeat=0,stage='price')
    assert decision['decision']['eligible_ids']==ids[:2]
    raw[ids[0]]=0.029; raw[ids[1]]=0.971
    assert g.decode_choice(raw,ids,seed='s',person_id='p',repeat=0,stage='price')['decision']['eligible_ids']==ids[1:2]


def test_client_stop_and_transport_error():
    class Client:
        def __init__(self, fail=False): self.fail=fail; self.calls=0
        async def choose(self, **kw):
            self.calls+=1
            if self.fail: raise RuntimeError('transport')
            ids=list(kw['criteria']); return {'answers':{'decision':{'probabilities':{i:1/len(ids) for i in ids}}}}
    async def run():
        p={'person_id':'p','repeat':0,'side':'BUY','state':state()}
        c=Client(); result=await g.predict(c,p,seed='s')
        assert c.calls==1 and result['status']=='client_stop' and result['submitted_orders']==state()['partial_orders']
        c=Client(True)
        with pytest.raises(RuntimeError) as exc: await g.predict(c,p,seed='s')
        assert exc.value.flat_grid_audit['failure']['stage']=='quantity'
    asyncio.run(run())


def test_sdk_body_limits():
    async def run():
        s,o=g.stage_spec(state(),'BUY','price',10)
        assert len(await g.checked_body(s,'price',o)) <= 12000
        with pytest.raises(ValueError): await g.checked_body(dict(s, history=['x'*12000]),'price',o)
        with pytest.raises(ValueError): await g.checked_body(s,'price',{str(i):'x' for i in range(256)})
    asyncio.run(run())


def test_replay_rejects_tampering_and_batch_drains():
    async def run():
        p={'person_id':'p','repeat':0,'side':'BUY','state':state()}
        result=await g.predict(g.SyntheticClient(),p,seed='s')
        for field in ('state','criteria','raw_probabilities','draw','price'):
            bad=deepcopy(result)
            if field=='state': bad['steps'][0]['state']['history']=[]
            elif field=='criteria': bad['steps'][1]['criteria']['p005']='6.00'
            elif field=='raw_probabilities': bad['steps'][0]['raw_probabilities']['q10']=0.5
            elif field=='draw': bad['steps'][0]['decision']['decision']['draw']=0.99
            else: bad['submitted_orders'][-1]['limit_price_dollars_per_unit']=6
            with pytest.raises(ValueError): await g.replay([p],[bad],seed='s')
        class Failing:
            def __init__(self): self.active=0
            async def choose(self,**kw):
                self.active+=1
                try:
                    await asyncio.sleep(0.01)
                    if kw['request_id'].startswith('bad:'): raise RuntimeError('offline failure')
                    first=next(iter(kw['criteria']))
                    return {'answers':{'decision':{'choice':first,'probabilities':{i:float(i==first) for i in kw['criteria']}}}}
                finally: self.active-=1
        client=Failing(); paths=[dict(p,person_id=i) for i in ('bad','good')]
        with pytest.raises(RuntimeError) as exc: await g.run_batch(client,paths,seed='s')
        assert client.active==0 and len(exc.value.outcomes)==2
        assert exc.value.outcomes[0].flat_grid_audit['failure']['type']=='RuntimeError'
        assert exc.value.outcomes[1]['status']=='selected'
    asyncio.run(run())


def test_prior_orders_control_sell_capacity():
    s=state(); s['partial_orders']=[{'slot': 0,'side':'SELL','quantity_units':90,'limit_price_dollars_per_unit':5}]
    s.pop('submitted_order_ledger')
    _, opts=g.stage_spec(s,'SELL','quantity')
    assert list(opts)==['q10']
    s['submitted_order_ledger']={'submitted_sell_quantity_units':0}
    with pytest.raises(ValueError): g.stage_spec(s,'SELL','quantity')
    s.pop('submitted_order_ledger'); s['partial_orders']=[{'slot':0,'side':'SELL','quantity_units':40,'limit_price_dollars_per_unit':5}, {'slot':1,'side':'SELL','quantity_units':40,'limit_price_dollars_per_unit':5}]
    assert list(g.stage_spec(s,'SELL','quantity')[1])==['q10','q20']
    s['partial_orders'][1]['slot']=0
    with pytest.raises(ValueError): g.stage_spec(s,'SELL','quantity')


def test_forecast_and_price_stop_preserve_prefix():
    class PriceStop:
        async def choose(self, **kw):
            ids=list(kw['criteria'])
            raw={i:(1.0 if i==ids[0] else 0.0) for i in ids} if kw['request_id'].endswith('quantity') else {i:1/len(ids) for i in ids}
            return {'answers': {'decision': {'probabilities':raw}}}
    async def run():
        s=state(); before=deepcopy(s); p={'person_id':'p','repeat':0,'side':'SELL','state':s}
        out=await g.predict(PriceStop(),p,seed='s')
        assert out['status']=='client_stop' and out['submitted_orders']==before['partial_orders'] and s==before
        for step in out['steps']:
            assert 'predict' in step['instructions'].lower() and 'not advice' in step['instructions'].lower()
            assert step['instructions'] in step['body']
    asyncio.run(run())

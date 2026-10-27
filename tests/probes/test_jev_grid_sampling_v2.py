import asyncio, hashlib, json, math
import pytest
from capclaims.grid_sampling import decode, CUTOFF, TOLERANCE, TOP_K, SCHEMA
from tools.probes import jev_grid_probability as v1
from tools.probes import jev_grid_probability_v2 as v2

def test_cutoff_edges_use_raw_values_and_frozen_tolerance():
    # Raw .05 remains eligible even though rounded-total validation normalizes it below .05.
    result=decode({'a':.05,'b':.96},['a','b'],seed='s',person_id='p',repeat=0,stage='quantity')
    assert result['normalization']['rounding_normalized'] is True
    assert result['all_candidate_normalized_probabilities']['a'] < .05
    assert 'a' in result['retained_ids']
    edge=decode({'edge':0.049999999999999996,'main':.95},['edge','main'],seed='s',person_id='p',repeat=0,stage='quantity')
    assert 'edge' in edge['retained_ids']
    below=decode({'edge':.05-2*TOLERANCE,'main':.95+2*TOLERANCE},['edge','main'],seed='s',person_id='p',repeat=0,stage='quantity')
    assert 'edge' not in below['eligible_ids']

def test_top3_ties_keep_original_candidate_order_and_all_candidates():
    ids=['z','a','y','b','x']
    raw={key:.2 for key in ids}
    result=decode(raw,ids,seed='tie',person_id='person',repeat=1,stage='price')
    assert result['candidate_ids']==ids
    assert result['retained_ids']==['z','a','y']
    assert result['topk_dropped_ids']==['b','x']
    assert result['draw'] < 1
    assert decode(raw,ids,seed='tie',person_id='person',repeat=1,stage='price')==result
    other=decode(raw,ids,seed='tie',person_id='person',repeat=0,stage='price')
    assert other['draw_sha256']!=result['draw_sha256']

def test_single_survivor_and_empty_support_failure():
    result=decode({'winner':.94,'low1':.03,'low2':.03},['winner','low1','low2'],seed=1,person_id='p',repeat=0,stage='quantity')
    assert result['retained_ids']==['winner'] and result['selected_id']=='winner'
    assert result['retained_normalized_probabilities']=={'winner':1.0,'low1':0.0,'low2':0.0}
    with pytest.raises(ValueError,match='no surviving'):
        decode({f'x{i}':.01 for i in range(100)},[f'x{i}' for i in range(100)],seed=1,person_id='p',repeat=0,stage='price')

def test_invalid_support_and_coordinates_fail_closed():
    with pytest.raises(ValueError): decode({'a':.8},['a','b'],seed=1,person_id='p',repeat=0,stage='quantity')
    with pytest.raises(ValueError): decode({'a':float('nan')},['a'],seed=1,person_id='p',repeat=0,stage='quantity')
    with pytest.raises(ValueError): decode({'a':1.0},['a'],seed=1,person_id='p',repeat=2,stage='quantity')

def test_v1_ten_percent_argmax_path_is_unchanged():
    raw={'a':.60,'b':.25,'c':.15}
    assert v1.choose(['a','b','c'],raw)=='a'
    decoded=decode(raw,['a','b','c'],seed='s',person_id='p',repeat=0,stage='quantity')
    assert decoded['selected_id'] in decoded['retained_ids']
    assert v1.choose(['a','b','c'],raw)=='a'

def test_three_stage_chain_propagates_sampled_quantity_and_tampering_fails(tmp_path,monkeypatch):
    case={'person_id':'synthetic-person','current_state':{'round':13,'p':250,'w':10,'Q':37,'D':40},
        'history':[{'round':r,'p':240,'Q':37,'D':40,'orders':[],'profit_cents':0} for r in range(6,13)]}
    base=v1.make_state(case,'BUY')
    captured=[]
    async def fake_body(state,stage,options):
        return json.dumps({'model':v1.MODEL,'state':state,'questions':{'decision':{'instructions':v2._instructions(stage),'criteria':options}}},ensure_ascii=False,separators=(',',':')).encode()
    class FakeClient:
        async def send_body(self,*,request_id,body):
            request=v1.strict_json(body); captured.append(request)
            ids=list(request['questions']['decision']['criteria'])
            target={'quantity':'q10','band':'b2','price':'p201'}
            stage=next((stage for stage,word in [('quantity','complete quantity'),('band','disjoint price-band'),('price','complete dollar price')] if word in request['questions']['decision']['instructions']),None)
            assert stage is not None
            selected=target[stage]
            if selected not in ids: selected=ids[0]
            raw={key:(1.0 if key==selected else 0.0) for key in ids}
            response={'model':v1.MODEL,'usage':{'input_tokens':1,'output_tokens':1},'answers':{'decision':{'type':'choice','choice':selected,'probabilities':raw}}}
            return {**response,'_wire':{'request_id':request_id,'request_sha256':v2._sha(body),'response_sha256':v2._sha(v2._canonical(response))}}
    monkeypatch.setattr(v2,'_body',fake_body)
    asyncio.run(v2._chain(FakeClient(),tmp_path,{'person_id':case['person_id'],'side':'BUY','base_state':base},0))
    states=[row['state'] for row in captured]
    assert len(states)==3
    assert states[0]['own_q_units']==37 and 'own_qty_units' not in states[0]
    assert states[1]['own_q_units']==37 and states[1]['own_qty_units']==10
    assert states[2]['own_q_units']==37 and states[2]['own_qty_units']==10 and states[2]['model_chosen_band']==2
    trace=[json.loads(line) for line in (tmp_path/'trace.jsonl').read_text().splitlines()]
    row=trace[0]['step']; raw=row['raw_probabilities']
    assert v2._replay_decision(row,raw,person=case['person_id'],repeat=0)['selected_id']==row['selected_id']
    tampered=dict(row,selected_id='q20')
    with pytest.raises(ValueError): v2._replay_decision(tampered,raw,person=case['person_id'],repeat=0)
    altered=dict(captured[1],state=dict(states[1],own_qty_units=20))
    with pytest.raises(ValueError,match='state'): v2._verify_request(altered,'band',states[1],captured[1]['questions']['decision']['criteria'])
    prefixed=dict(captured[0],state=dict(states[0],numeric_prefix='2'))
    with pytest.raises(ValueError,match='state'): v2._verify_request(prefixed,'quantity',states[0],captured[0]['questions']['decision']['criteria'])

def test_full_synthetic_replay_accepts_then_rejects_tampered_sample(tmp_path):
    run_dir=tmp_path/'synthetic-run'; run_dir.mkdir()
    contexts=[]; rows=[]; evidence_root=run_dir/'wire-evidence'; evidence_root.mkdir()
    for person_index in range(10):
        side='BUY' if person_index<5 else 'SELL'; q=40 if side=='BUY' else 100
        context={'person_id':f'synthetic-{person_index}','side':side,'stratum':'shortage' if side=='BUY' else 'surplus',
            'base_state':{'round':18,'own_q_units':q,'own_side':side,'selected_order_count':1,'remaining_planned_order_slots':1,'partial_orders':[]}}
        contexts.append(context)
        for repeat in (0,1):
            quantity=10; band=2
            specs=[('quantity',context['base_state'],v1.criteria(v1.quantity_grid(side,q),'q'),'q10',quantity,0),
                ('band',dict(context['base_state'],own_qty_units=quantity,model_chosen_quantity=quantity),{f'b{i}':f'{lo*2.5:.2f}-{hi*2.5:.2f} dollars' for i,(lo,hi) in enumerate(v1.BANDS)},'b2','b2',2),
                ('price',dict(context['base_state'],own_qty_units=quantity,model_chosen_quantity=quantity,model_chosen_band=band),v1.criteria(v1.price_grid(*v1.BANDS[band]),'p'),'p201',502.5,0)]
            previous={}
            for stage_no,(stage,state,options,selected,value,index) in enumerate(specs):
                ids=list(options); raw={key:float(key==selected) for key in ids}
                decoded=decode(raw,ids,seed=v2.SEED,person_id=context['person_id'],repeat=repeat,stage=stage)
                req_id=f"gpr-v2-r{repeat}-{context['person_id']}-{stage}"
                request={'model':v1.MODEL,'state':state,'questions':{'decision':{'instructions':v2._instructions(stage),'criteria':options}}}
                response={'model':v1.MODEL,'usage':{'input_tokens':1,'output_tokens':1},'answers':{'decision':{'type':'choice','choice':selected,'probabilities':raw}}}
                req_bytes=json.dumps(request,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode(); resp_bytes=v2._canonical(response)
                folder=evidence_root/hashlib.sha256(req_id.encode()).hexdigest(); folder.mkdir()
                (folder/'request.json').write_bytes(req_bytes); (folder/'response.json').write_bytes(resp_bytes)
                argmax=v1.choose(ids,decoded['all_candidate_normalized_probabilities'])
                conditional={'side':side,**({'quantity':quantity} if stage_no>=1 else {}),**({'band':band} if stage_no>=2 else {})}
                step={'stage':stage,'request_id':req_id,'request_sha256':v2._sha(req_bytes),'response_sha256':v2._sha(resp_bytes),'candidate_ids':ids,
                    'raw_probabilities':raw,'raw_sum':math.fsum(raw.values()),'all_candidate_normalized_probabilities':decoded['all_candidate_normalized_probabilities'],
                    'normalization':decoded['normalization'],'decode':decoded,'argmax_id':argmax,'provider_choice':selected,
                    'provider_choice_matches_argmax':selected==argmax,'provider_choice_tie_resolved':False,'provider_choice_strict_nonmax':False,
                    'selected_id':selected,'selected_value':value,'selected_index':index,'conditional_on':conditional}
                rows.append({'schema':'grid-probability-v2-trace','record_type':'STEP','repeat':repeat,'context':context['person_id'],'step':step})
    plan={'schema':v2.PLAN_SCHEMA,'status':'FROZEN','decoder':SCHEMA,'cutoff':CUTOFF,'cutoff_tolerance':TOLERANCE,'top_k':TOP_K,
        'seed':v2.SEED,'model':v1.MODEL,'budget_nanodollars':v2.BUDGET,'request_cap':60,'people_n':10,'repeat_blocks':2,
        'calls_per_chain':3,'planned_requests':60,'contexts':contexts,'code_sha256':v2._hashes()}
    plan_bytes=v2._canonical(plan)+b'\n'; (run_dir/'plan.json').write_bytes(plan_bytes); (run_dir/'manifest.sha256').write_text(v2._sha(plan_bytes)+'\n')
    (run_dir/'started').write_text(v2._sha(plan_bytes)+'\n'); (run_dir/'trace.jsonl').write_bytes(b''.join(v2._canonical(row)+b'\n' for row in rows))
    (run_dir/'status.jsonl').write_text(v2._canonical({'status':'COMPLETE','physical_requests':60,'completed_chains':20}).decode()+'\n')
    _,replayed=v2.replay(run_dir)
    assert len(replayed)==60
    rows[0]['step']['selected_id']='q20'
    (run_dir/'trace.jsonl').write_bytes(b''.join(v2._canonical(row)+b'\n' for row in rows))
    with pytest.raises(ValueError): v2.replay(run_dir)

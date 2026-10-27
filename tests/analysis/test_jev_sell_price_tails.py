import pytest
from tools.analysis.audit_jev_sell_price_tails import interval,band_forces_high,policy_sensitivity,human_summary,reference_rows

def test_original_candidate_order_controls_interval():
    d={'candidate_ids':['b0','b1','b2','b3'],'selected_id':'b2','retained_normalized_probabilities':{'b0':.8,'b1':0,'b2':.05,'b3':.15}}
    assert interval(d)==pytest.approx([.8,.85])
    assert [band_forces_high(i) for i in range(4)]==[False,True,True,True]

def test_threshold_policy_and_empty_support():
    band={'context':'p','repeat':0,'step':{'stage':'band','candidate_ids':['b0','b1','b2','b3'],'raw_probabilities':{'b0':.90,'b1':.04,'b2':.04,'b3':.02}}}
    out=policy_sensitivity([band]); assert out['full']['mean_high_probability']==pytest.approx(.10)
    assert out['cutoff5']['mean_high_probability']==0 and out['cutoff3']['mean_high_probability']==pytest.approx(.08/.98)
    ids=[f'p{i}' for i in range(1,101)]; flat={'context':'p','repeat':0,'step':{'stage':'price','candidate_ids':ids,'raw_probabilities':{k:.01 for k in ids}}}
    out=policy_sensitivity([flat]); assert out['cutoff5']['decode_failures']==1 and out['cutoff3']['decode_failures']==1
    assert out['full']['mean_high_probability']==0

def test_reference_round13_labels_are_not_history_inputs():
    case={'current_state':{'round':13,'p':20,'Q':10,'D':5},'history':[{'round':6,'p':10,'Q':10,'D':5,'orders':[{'side':'sell','qty':1,'price':10}]}]}
    label={'orders':[{'side':'sell','qty':1,'price':300}]}
    cases={'p':case}; labels={'p':label}; hist=reference_rows(cases,labels,{'p'},[6]); current=reference_rows(cases,labels,{'p'},[13])
    assert human_summary(hist)['sell_price_gt250_n']==0
    assert human_summary(current)['sell_price_gt250_n']==1
    assert case['history'][0]['orders'][0]['price']==10

def test_private_failure_case_preserves_wire_evidence():
    import hashlib,json
    from pathlib import Path
    from tools.analysis import audit_jev_sell_price_tails as audit
    path=audit.RUNS/'jev-sell-price-tail-audit-2026-10-27'/'F1-other-context-failure'
    if not path.exists(): pytest.skip('private evidence has not been rendered')
    original=(path/'price-response.json').read_bytes(); response=json.loads(original)
    assert isinstance(response['answers']['decision']['probabilities'],dict)
    bundles=json.loads((path.parent/'full-inputs-private.json').read_text(encoding='utf-8'))['bundles']
    failure=next(x for x in bundles if x['side']=='SELL' and x['stages']['price']['trace'].get('record_type')=='DECODE_FAILURE')
    assert hashlib.sha256(original).hexdigest()==failure['stages']['price']['trace']['step']['response_sha256']
    assert (path/'price-response.json').read_bytes()==original

def test_high_band_argmax_can_differ_from_sample():
    row={'context':'p','repeat':0,'step':{'stage':'band','candidate_ids':['b0','b1','b2','b3'],'raw_probabilities':{'b0':.45,'b1':.02,'b2':.02,'b3':.51}}}
    assert max(row['step']['raw_probabilities'],key=row['step']['raw_probabilities'].get)=='b3'
    result=policy_sensitivity([row])
    assert result['cutoff5']['mean_high_probability']==pytest.approx(.51/.96)
    assert result['full']['mean_high_probability']==pytest.approx(.55)

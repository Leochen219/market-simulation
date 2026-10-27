from copy import deepcopy
import pytest
from capclaims.flat_price_grid import PRICE_VALUES
from capclaims.sell_oracle.analysis import crps, nested_weights, public_aggregate, summarize


def prediction(c, price=10):
    raw={k:float(v==price) for k,v in PRICE_VALUES.items()}
    return {**c,'price':price,'outcome':'selected','raw_probabilities':raw,
            'decision':{'decision':{'retained_normalized_probabilities':raw.copy()}}}


def fixture():
    contexts=[]
    for person,t,has in [('a',11,True),('a',12,True),('b',11,False),('c',11,False)]:
        contexts.append({'person_id':person,'formal_round':t,'current_state':{'p':75,'Q':100,'D':10},
                         'history':[{'orders':[{'side':'sell' if has else 'buy','qty':6,'price':5}]}]})
    rows=[]
    for cid,qid,p,t,q in [('1','a6','a',11,6),('2','a6','a',11,6),('3','a14','a',12,14),('4','b10','b',11,10)]:
        rows.append({'component_id':cid,'query_id':qid,'person_id':p,'formal_round':t,'quantity':q})
    return {'roster':['a','b','c'],'contexts':contexts,'components':rows}, [prediction(c) for c in rows], {'1':5,'2':15,'3':10,'4':20}


def test_nested_weights_equal_people_then_rounds_then_components():
    plan,preds,labels=fixture()
    assert nested_weights(plan['components'])=={'1':.125,'2':.125,'3':.25,'4':.5}
    r=summarize(plan,preds,labels)
    assert r['coverage']['state_selected_events']==4 and r['coverage']['no_sell_events']==1
    assert r['coverage']['physical_queries']==3 and r['coverage']['components']==4
    g=r['groups']['all']
    assert g['human_full']['mean']==15
    assert g['forecasts']['retained']['raw_human_w1']==pytest.approx(6.25)
    assert g['forecasts']['retained']['conditional_crps']==pytest.approx(6.25)
    assert r['groups']['single']['planned_events']==2
    assert r['groups']['multiple']['planned_events']==1


def test_crps_closed_form_and_not_expected_absolute_loss():
    assert crps([(5,.5),(15,.5)],10)==pytest.approx(2.5)
    assert crps([(10,1)],10)==0
    assert crps([(10,1)],20)==10


def test_marginal_matching_can_hide_conditional_error():
    plan,preds,labels=fixture()
    rows=[plan['components'][0],plan['components'][-1]]
    plan['components']=rows
    labels={'1':5,'4':20}
    r=summarize(plan,[prediction(rows[0],20),prediction(rows[1],5)],labels)
    g=r['groups']['single']['forecasts']['retained']
    assert g['raw_human_w1']==0
    assert g['conditional_crps']==15


def test_duplicate_quantities_share_distribution_keep_multiplicity():
    plan,preds,labels=fixture()
    for row in preds[:2]:
        raw={k:0. for k in PRICE_VALUES};raw['p005']=raw['p015']=.5
        row.update(raw_probabilities=raw,decision={'decision':{'retained_normalized_probabilities':raw.copy()}})
        row['price']=5 if row['component_id']=='1' else 15
    r=summarize(plan,preds,labels)
    g=r['groups']['multiple']
    assert g['planned_components']==2 and g['physical_queries']==1
    assert g['forecasts']['retained']['raw_human_w1']==0
    assert g['forecasts']['retained']['conditional_crps']==pytest.approx(2.5)
    assert g['forecasts']['sample']['raw_human_w1']==0


def test_abstained_shared_query_removes_no_prices_as_zero():
    plan,preds,labels=fixture()
    for row in preds[:2]:row.update(price=None,outcome='client_numeric_abstain')
    r=summarize(plan,preds,labels)
    g=r['groups']['all']
    assert g['quoted_components']==2 and g['client_abstained_components']==2
    assert g['forecasts']['retained']['raw_human_w1']==5
    assert g['raw_full']['raw_human_w1']==pytest.approx(6.25)
    assert g['human_common']['mean']==15
    for row in preds:row.update(price=None,outcome='client_numeric_abstain')
    empty=summarize(plan,preds,labels)['groups']['all']
    assert empty['forecasts']['retained']['raw_human_w1'] is None
    assert empty['forecasts']['retained']['distribution']['cdf']==[]
    assert empty['raw_full']['raw_human_w1']==pytest.approx(6.25)


def test_quantity_weighting_changes_only_within_round_component_weights():
    plan,_,_=fixture()
    plan['components'][1]['quantity']=18
    weights=nested_weights(plan['components'],quantity_weighted=True)
    assert weights=={'1':.0625,'2':.1875,'3':.25,'4':.5}


def test_inputs_order_invariance_projection_and_public_suppression():
    plan,preds,labels=fixture();labels['1']=1
    original=deepcopy((plan,preds,labels))
    first=summarize(plan,preds,labels)
    assert (plan,preds,labels)==original
    plan['components'].reverse();preds.reverse()
    assert summarize(plan,preds,labels)==first
    assert first['groups']['all']['human_price_off_grid_components']==1
    public=public_aggregate(first,minimum_people=2)
    assert not public['groups']['all'].get('suppressed')
    assert public['groups']['single']['suppressed']
    assert public['groups']['multiple']['suppressed']
    assert public['groups']['history_yes']['suppressed']
    assert public['groups']['history_no']['suppressed']
    assert 'person_id' not in str(public)


def test_invalid_partial_identity_probability_or_label_cannot_score():
    plan,preds,labels=fixture()
    with pytest.raises(ValueError):summarize(plan,preds[:-1],labels)
    for field,value in [('quantity',99),('price',0),('outcome','technical_failure')]:
        changed=deepcopy(preds);changed[0][field]=value
        with pytest.raises(ValueError):summarize(plan,changed,labels)
    changed=deepcopy(preds);changed[0]['raw_probabilities']['p010']=.5
    with pytest.raises(ValueError):summarize(plan,changed,labels)
    changed=deepcopy(preds)
    changed[0]['decision']['decision']['retained_normalized_probabilities'].update(p010=1.1,p005=-.1)
    with pytest.raises(ValueError):summarize(plan,changed,labels)
    with pytest.raises(ValueError):summarize(plan,preds,{**labels,'1':0})

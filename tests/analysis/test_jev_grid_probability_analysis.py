import json
import pytest
from tools.analysis.analyze_jev_grid_probability import _verify_probability_evidence, _verify_request, _json, _boundary_distribution
from tools.probes import jev_grid_probability as frozen

def _floating_order_fixture():
    ids=[f'a{i:02}' for i in range(10)]
    ordered_values=[.09,.10,.12,.10,.11,.11,.11,.05,.09,.12]
    response_probs={key:value for key,value in zip(reversed(ids),ordered_values)}
    canonical_trace=json.loads(_json({'raw_probabilities':response_probs}))['raw_probabilities']
    assert sum(response_probs.values())==1.0
    assert sum(canonical_trace.values())!=1.0
    norm,meta=frozen.normalized(response_probs,ids)
    step={'candidate_ids':ids,'raw_probabilities':canonical_trace,'raw_sum':sum(response_probs.values()),
          'normalized_probabilities':norm,'normalization':meta,'provider_choice':'a00'}
    answer={'probabilities':response_probs,'choice':'a00'}
    return step,answer

def test_original_response_order_sum_replays_despite_trace_sort_order():
    step,answer=_floating_order_fixture()
    replay=_verify_probability_evidence(step,answer)
    assert replay['raw_sum']==1.0
    assert replay['normalization']['raw_probability_total']==1.0

def test_changed_response_probabilities_are_rejected():
    step,answer=_floating_order_fixture()
    changed=dict(answer,probabilities=dict(answer['probabilities']))
    changed['probabilities']['a00']=.13
    with pytest.raises(ValueError,match='differs'):
        _verify_probability_evidence(step,changed)

def test_changed_quantity_or_numeric_prefix_request_state_is_rejected():
    expected={'own_q_units':37,'own_side':'BUY','selected_order_count':1,'remaining_planned_order_slots':1}
    options={'q10':'10','q20':'20'}
    request={'state':dict(expected),'questions':{'decision':{'criteria':options,'instructions':frozen._instructions('quantity')}}}
    _verify_request(request,'quantity',expected,options)
    altered=dict(request,state=dict(expected,own_qty_units=20))
    with pytest.raises(ValueError,match='state'):
        _verify_request(altered,'quantity',expected,options)
    prefixed=dict(request,state=dict(expected,numeric_prefix='2'))
    with pytest.raises(ValueError,match='state'):
        _verify_request(prefixed,'quantity',expected,options)

def test_boundary_sensitivity_is_separate_from_strict_cutoff():
    result=_boundary_distribution({'top':0.049999999999999996,'other':0.95},.05)
    assert result['strict_support_count']==1
    assert result['tolerant_support_count']==2
    assert result['boundary_candidate_n']==1
    assert result['strict_top3_retained_mass']!=result['tolerant_top3_retained_mass']

import copy
import json
from pathlib import Path
from unittest.mock import patch
import pytest
from tools.analysis import compare_jev_grid_decoders as comp


def fixture():
    contexts=[{'person_id':str(i),'side':'BUY' if i<5 else 'SELL','base_state':{'own_q_units':2}} for i in range(10)]
    plan={'contexts':contexts,'model':'x','prompt':'same','price_bands':[[1,100]]}
    old=[]; new=[]
    for c in contexts:
        for repeat in (0,1):
            for stage,value in (('quantity',2+repeat),('band',0),('price',10+repeat*2.5)):
                row={'context':c['person_id'],'repeat':repeat,'stage':stage,'selected_value':value,'selected_index':0}
                old.append(row)
                new.append(dict(row,record_type='STEP',selected_id='a',argmax_id='a' if repeat==0 else 'b'))
    return plan,old,copy.deepcopy(plan),new


def test_complete_groups_and_toy_statistics():
    result=comp.compare(*fixture())
    assert result['groups']['BUY']['v1_argmax']['quantity']['histogram']=={'2':5,'3':5}
    assert result['groups']['all']['v1_argmax']['price_10_dollar_multiple_n']==10
    assert result['groups']['all']['paired_v2_minus_v1']['quantity']['unchanged_n']==20
    assert result['v2_same_response_decoder_deviation']['price']=={'n':20,'sample_differs_from_same_response_argmax_n':10}
    assert comp._stats([1,2,3,4])['iqr']==2


def test_mismatch_and_incomplete_chains_rejected():
    p1,r1,p2,r2=fixture(); p2['contexts'][0]['base_state']['own_q_units']=99
    with pytest.raises(ValueError,match='context'): comp.compare(p1,r1,p2,r2)
    p1,r1,p2,r2=fixture(); r2.pop()
    with pytest.raises(ValueError,match='incomplete'): comp.compare(p1,r1,p2,r2)
    p1,r1,p2,r2=fixture(); p2['prompt']='changed'
    with pytest.raises(ValueError,match='prompt'): comp.compare(p1,r1,p2,r2)


def test_both_replays_are_required_and_tamper_rejected(tmp_path):
    p1,r1,p2,r2=fixture(); (tmp_path/'status.jsonl').write_text(json.dumps({'status':'COMPLETE'})+'\n')
    with patch.object(comp,'replay_v1',return_value=(p1,r1)) as a, patch.object(comp.v2,'replay',return_value=(p2,r2)) as b:
        assert comp.analyze(tmp_path,tmp_path)['status']=='COMPLETE'; a.assert_called_once(); b.assert_called_once()
    with patch.object(comp,'replay_v1',side_effect=ValueError('wire evidence hash mismatch')):
        with pytest.raises(ValueError,match='hash mismatch'): comp.analyze(tmp_path,tmp_path)
    with patch.object(comp,'replay_v1',return_value=(p1,r1)), patch.object(comp.v2,'replay',side_effect=ValueError('tampered')):
        with pytest.raises(ValueError,match='tampered'): comp.analyze(tmp_path,tmp_path)
    (tmp_path/'status.jsonl').write_text(json.dumps({'status':'RUNNING'})+'\n')
    with patch.object(comp,'replay_v1',return_value=(p1,r1)), patch.object(comp.v2,'replay',return_value=(p2,r2)):
        with pytest.raises(ValueError,match='incomplete'): comp.analyze(tmp_path,tmp_path)

def test_real_v1_replay_rejects_trace_tamper(tmp_path):
    import shutil
    source=comp.ROOT/'artifacts/participant-data/jev-grid-probability-v1-2026-10-27'
    if not source.exists(): pytest.skip('study evidence not present')
    target=tmp_path/'evidence'; shutil.copytree(source,target)
    path=target/'trace.jsonl'; lines=path.read_text(encoding='utf-8').splitlines()
    row=json.loads(lines[0]); row['step']['argmax_id']='tampered'; lines[0]=json.dumps(row)
    path.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    with pytest.raises(ValueError,match='argmax replay mismatch'): comp.replay_v1(target)

def test_partial_price_failure_keeps_matched_denominators():
    p1,r1,p2,r2=fixture()
    failure=next(r for r in r2 if r['context']=='0' and r['repeat']==0 and r['stage']=='price')
    failure.clear(); failure.update(context='0',repeat=0,stage='price',record_type='DECODE_FAILURE',candidate_count=2,raw_probabilities={'a':.02,'b':.03})
    result=comp.compare(p1,r1,p2,r2); group=result['groups']['all']
    assert result['status']=='PARTIAL_FAILURE' and result['price_decode_failures']['n']==1
    assert group['v1_argmax']['price']['n']==20 and group['v2_top3_cutoff5']['price']['n']==19
    assert group['v1_price_matched_to_v2_successes']['n']==19
    assert group['paired_v2_minus_v1']['price']['n']==19
    assert group['v2_top3_cutoff5']['repeat_diversity']['price']['context_n']==9

from tools.analysis import compare_jev_grid_decoders_v3 as comp
from tests.analysis.test_jev_grid_decoder_comparison import fixture

def test_v3_comparison_labels_and_counts():
    result=comp.compare(*fixture())
    assert result['schema']=='jev-grid-decoder-comparison-v3'
    assert result['status']=='COMPLETE'
    assert result['groups']['all']['v3_top3_cutoff3']['quantity']['n']==20
    assert result['v3_same_response_decoder_deviation']['price']['n']==20


def test_same_response_threshold_sensitivity():
    rows=[{'context':'p','repeat':0,'step':{'stage':'quantity','candidate_ids':['a','b','c','d'],'raw_probabilities':{'a':.90,'b':.04,'c':.04,'d':.02}}}]
    out=comp.threshold_sensitivity(rows)['quantity']
    assert out['response_n']==1 and out['both_selected_n']==1
    assert out['support_diff_n']==1 and out['selected_diff_n'] in (0,1)

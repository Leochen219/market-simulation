import unittest
from capclaims.grid_sampling_v3 import decode as old_decode
from capclaims.numeric_stop_policy import decode_or_stop, discard_draft_on_stop

KW = dict(seed='test', person_id='anonymous', repeat=0, stage='price')

def distribution():
    ids=[f'p{i}' for i in range(1,101)]
    return ids, {k:.01 for k in ids}

class NumericStopPolicyTests(unittest.TestCase):
    def test_valid_empty_support_stops(self):
        ids,raw=distribution()
        result=decode_or_stop(raw,ids,**KW)
        self.assertEqual(result['status'],'client_stop')
        self.assertIsNone(result['selected_id'])
        self.assertEqual((result['policy_version'],result['stage'],result['candidate_count'],result['raw_max_probability'],result['stop_origin']),('numeric_empty_support_stop_v1','price',100,.01,'client_empty_support'))

    def test_exact_three_percent_remains_selectable(self):
        ids,raw=distribution()
        # Valid sum with every other choice below threshold.
        raw={k:.0097979797979798 for k in ids}; raw[ids[0]]=.03
        result=decode_or_stop(raw,ids,**KW)
        self.assertEqual(result['status'],'selected'); self.assertEqual(result['selected_id'],'p1')
        self.assertIsNone(result['stop_origin'])

    def test_below_cutoff_beyond_tolerance_stops(self):
        ids,raw=distribution(); raw[ids[0]]=.03-2e-12
        for key in ids[1:]: raw[key]=(1-raw[ids[0]])/99
        self.assertEqual(decode_or_stop(raw,ids,**KW)['status'],'client_stop')

    def test_invalid_distribution_and_candidates_raise(self):
        ids,raw=distribution()
        for bad in ({**raw,ids[0]:-1}, {k:v for k,v in raw.items() if k!=ids[0]}, {**raw,ids[0]:float('nan')}):
            with self.assertRaises(ValueError): decode_or_stop(bad,ids,**KW)
        with self.assertRaises(ValueError): decode_or_stop(raw,ids,**{**KW,'stage':'unknown'})
        with self.assertRaises(ValueError): decode_or_stop(raw,ids[:-1]+['q100'],**KW)
        with self.assertRaises(ValueError): decode_or_stop(raw,ids,**{**KW,'repeat':2})

    def test_selected_matches_unchanged_decoder(self):
        ids=['p1','p2','p3','p4']; raw={'p1':.5,'p2':.2,'p3':.2,'p4':.1}
        self.assertEqual(decode_or_stop(raw,ids,**KW)['decision'],old_decode(raw,ids,**KW))

    def test_stop_preserves_previous_orders_and_discards_draft(self):
        submitted=[{'price':10,'tags':['original']}]; draft={'price':20}
        result=discard_draft_on_stop(submitted,draft)
        self.assertEqual(result,{'submitted_orders':[{'price':10,'tags':['original']}],'pending_draft':None})
        result['submitted_orders'][0]['tags'].append('changed')
        self.assertEqual(submitted,[{'price':10,'tags':['original']}]); self.assertEqual(draft,{'price':20})

if __name__=='__main__': unittest.main()

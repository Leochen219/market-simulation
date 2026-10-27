from copy import deepcopy
import pytest
from capclaims.flat_grid_study import analysis as a


def fixture():
    contexts = [{'person_id': p, 'formal_round': 11, 'condition': 'short3', 'side': 'BUY',
                 'current_state': {'p': 50}} for p in ('a', 'b')]
    old = []; new = []; schedule = []
    for p in ('a', 'b'):
        for r in (0, 1):
            base = dict(person_id=p, formal_round=11, condition='short3', repeat=r)
            schedule.append(base)
            old.append(dict(base, conditioned_quantity=10, prices={'cutoff3': 20 if p == 'a' else 10}))
            new.append(dict(base, fixed_quantity=10, outcome='selected', price=15))
    plan = {'contexts': contexts, 'schedule': schedule}
    labels = {'a': {'orders': [{'side': 'buy', 'price': 10}]}, 'b': {'orders': [{'side': 'buy', 'price': 20}]}}
    return plan, new, plan, old, labels


def test_pooled_distribution_not_mean_person_error():
    out = a.summarize(*fixture())['groups']['short3']['BUY']
    assert out['methods']['hierarchical_2_5']['raw_human_w1'] == 0
    assert out['methods']['flat_5']['raw_human_w1'] == 5
    assert out['flat_minus_hierarchical_raw_w1'] == 5


def test_all_abstain_keeps_planned_and_null_w1():
    args = list(fixture())
    for p in args[1]: p.update(price=None, outcome='client_numeric_abstain')
    out = a.summarize(*args)['groups']['short3']['BUY']
    assert out['common_paths'] == out['common_people'] == 0
    flat = out['methods']['flat_5']
    assert flat['raw_human_w1'] is None and flat['no_quote'] == 4
    assert flat['events']['ge700'] == dict(events=0, planned=4, quoted=0, no_quote=4,
        confirmed_quote_event_per_planned=0, among_quotes=None, hypothetical_unreported_price_bounds=[0, 1])


def test_partial_quotes_shared_mask_and_no_zero():
    args = list(fixture()); args[1][0].update(price=None, outcome='client_numeric_abstain')
    out = a.summarize(*args)['groups']['short3']['BUY']
    assert out['common_people'] == 2 and out['common_paths'] == 3
    assert out['common_distributions']['flat_5']['mean'] == 15
    assert out['methods']['flat_5']['events']['ge700']['hypothetical_unreported_price_bounds'] == [0, .25]
    assert out['common_distributions']['human_raw']['mean'] == 15


def test_threshold_boundaries_projection_and_fixed_quantity():
    assert a.event_counts([699, 700, 760, 800, None], [100]*5, '700_to_800')['events'] == 2
    assert a.event_counts([699, 700, 760, 800, None], [100]*5, 'ge700')['events'] == 3
    assert a.project_price(1, 5) == 5 and a.project_price(999, 5) == 995
    assert a.project_price(12.5, 5) == 15
    args = list(fixture()); args[1][0]['fixed_quantity'] = 20
    with pytest.raises(ValueError, match='quantity'): a.summarize(*args)


def test_human_orders_and_repeats_have_nested_equal_weights():
    args = list(fixture()); args[4]['a']['orders'] = [{'side':'buy','price':5}, {'side':'buy','price':15}]
    out = a.summarize(*args)['groups']['short3']['BUY']
    assert out['human_full_distribution']['mean'] == 15
    before = deepcopy(args)
    a.summarize(*args)
    assert args == before

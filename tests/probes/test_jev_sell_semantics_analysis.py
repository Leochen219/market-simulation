from copy import deepcopy
import pytest

from capclaims.flat_price_grid import PRICE_VALUES
from capclaims.sell_semantics.analysis import public_aggregate, summarize


def fixture():
    contexts, schedule, predictions = [], [], []
    targets = {'a': {'orders': [{'side': 'sell', 'price': 5}]},
               'b': {'orders': [{'side': 'sell', 'price': 15}]}}
    for condition in ('short3', 'long10'):
        for person, has_sell in [('a', True), ('b', False)]:
            context = {'person_id': person, 'formal_round': 11, 'condition': condition,
                       'side': 'SELL', 'current_state': {'p': 20},
                       'history': [{'round': 10, 'orders': [{'side': 'sell' if has_sell else 'buy', 'price': 5, 'qty': 10}]}]}
            contexts.append(context)
            for repeat in (0, 1):
                for arm in ('original', 'rules'):
                    row = {'person_id': person, 'formal_round': 11, 'condition': condition,
                           'repeat': repeat, 'arm': arm, 'conditioned_quantity': 10}
                    schedule.append(row)
                    price = (15 if person == 'a' else 5) if arm == 'original' else 10
                    predictions.append({**row, 'fixed_quantity': 10, 'price': price,
                                        'outcome': 'selected'})
    return {'contexts': contexts, 'schedule': schedule}, predictions, targets


def test_primary_is_pooled_distribution_not_average_person_error():
    plan, predictions, targets = fixture()
    result, private = summarize(plan, predictions, targets)
    for condition in ('short3', 'long10'):
        g = result['groups'][condition]['all']
        assert g['arms']['original']['raw_human_w1'] == 0
        assert g['arms']['rules']['raw_human_w1'] == 5
        assert g['rules_minus_original_raw_w1'] == 5
        assert g['auxiliary_person_w1_delta']['mean'] == -5
    assert len(private) == 4


def test_history_stratum_never_uses_target_and_inputs_are_not_mutated():
    plan, predictions, targets = fixture()
    before = deepcopy((plan, predictions, targets))
    result, _ = summarize(plan, predictions, targets)
    assert (plan, predictions, targets) == before
    for condition in ('short3', 'long10'):
        groups = result['groups'][condition]
        assert groups['has_sell_history']['eligible_people'] == 1
        assert groups['no_sell_history']['eligible_people'] == 1
        assert groups['has_sell_history']['historical_sell_rounds'] == [1]
        assert groups['no_sell_history']['historical_sell_rounds'] == [0]
        assert groups['no_sell_history']['descriptive_single_person'] is True
    targets['b']['orders'][0]['price'] = 995
    changed, _ = summarize(plan, predictions, targets)
    assert changed['groups']['short3']['no_sell_history']['historical_sell_rounds'] == [0]


def test_abstentions_are_not_prices_or_stops_and_all_empty_is_unscored():
    plan, predictions, targets = fixture()
    for row in predictions:
        if row['arm'] == 'rules':
            row.update(price=None, outcome='client_numeric_abstain')
    result, private = summarize(plan, predictions, targets)
    assert not private
    for condition in ('short3', 'long10'):
        g = result['groups'][condition]['all']
        assert g['common_people'] == 0 and g['common_paths'] == 0
        assert g['arms']['rules']['quoted'] == 0
        assert g['arms']['rules']['client_numeric_abstain'] == 4
        assert g['arms']['original']['raw_human_w1'] is None
        assert g['arms']['rules']['raw_human_w1'] is None
        assert g['common_distributions']['rules']['cdf'] == []


def test_one_missing_pair_keeps_person_equal_weight_and_excludes_both_arms():
    plan, predictions, targets = fixture()
    for row in predictions:
        if row['person_id'] == 'a' and row['repeat'] == 0 and row['arm'] == 'rules':
            row.update(price=None, outcome='client_numeric_abstain')
    result, _ = summarize(plan, predictions, targets)
    g = result['groups']['short3']['all']
    assert g['common_people'] == 2 and g['common_paths'] == 3
    assert g['arms']['original']['quoted'] == 4
    assert g['arms']['rules']['quoted'] == 3
    assert g['common_distributions']['original']['mean'] == 10
    assert g['arms']['original']['raw_human_w1'] == 0


def test_score_mass_retained_mass_and_event_boundaries_are_separate():
    plan, predictions, targets = fixture()
    raw = {k: 0.0 for k in PRICE_VALUES}
    raw.update(p005=.03, p010=.25, p080=.70, p995=.02)
    retained = {k: 0.0 for k in PRICE_VALUES}
    retained.update(p005=.03/.98, p010=.25/.98, p080=.70/.98)
    for row in predictions:
        row['raw_probabilities'] = raw.copy()
        row['decision'] = {'decision': {'retained_normalized_probabilities': retained.copy()}}
        if row['arm'] == 'rules':
            row['price'] = 65 if row['person_id'] == 'a' else 700
    result, _ = summarize(plan, predictions, targets)
    g = result['groups']['short3']['all']['arms']['rules']
    assert g['events']['ge65']['events'] == 4
    assert g['events']['ge700']['events'] == 2
    assert g['events']['700_to_800']['events'] == 2
    assert g['scores']['raw_normalized_mass_le20'] == pytest.approx(.28)
    assert g['scores']['raw_normalized_mass_ge65'] == pytest.approx(.72)
    assert g['scores']['retained_normalized_mass_ge700'] == 0
    assert g['scores']['retained_normalized_mass_le20'] == pytest.approx(.28/.98)


def test_failures_duplicates_missing_rows_bad_prices_and_quantities_rejected():
    plan, predictions, targets = fixture()
    with pytest.raises(ValueError):
        summarize(plan, predictions[:-1], targets)
    with pytest.raises(ValueError):
        summarize(plan, predictions + [predictions[0]], targets)
    for field, value in [('price', 1), ('fixed_quantity', 20), ('outcome', 'technical_failure')]:
        changed = deepcopy(predictions)
        changed[0][field] = value
        with pytest.raises(ValueError):
            summarize(plan, changed, targets)


def test_order_permutations_do_not_change_public_distributions():
    plan, predictions, targets = fixture()
    targets['a']['orders'].append({'side': 'sell', 'price': 15})
    first, _ = summarize(plan, predictions, targets)
    targets['a']['orders'].reverse()
    second, _ = summarize(plan, list(reversed(predictions)), targets)
    assert first == second
    text = str(first)
    assert 'person_id' not in text


def test_public_output_suppresses_singleton_and_complement_distributions():
    plan, predictions, targets = fixture()
    full, _ = summarize(plan, predictions, targets)
    before = deepcopy(full)
    public = public_aggregate(full)
    assert full == before
    assert 'no_sell_history' in full['groups']['short3']
    for g in public['groups'].values():
        assert set(g) == {'all', 'stratum_counts'}
        assert g['stratum_counts']['no_sell_history'] == 1
        assert g['stratum_counts']['has_sell_history'] == 1
        assert 'arms' not in g['stratum_counts']
        assert 'historical_sell_rounds' not in g['all']

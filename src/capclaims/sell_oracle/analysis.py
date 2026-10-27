"""Exact nested-weight distributions and conditional CRPS; labels score only."""
from collections import defaultdict
from copy import deepcopy
import asyncio
import json
import math
from pathlib import Path

from capclaims.backends.jev import normalize_jev_probabilities
from capclaims.flat_price_grid import PRICE_VALUES
from capclaims.flat_grid_study.analysis import accounting, distribution, project_price
from tools.analysis.compare_jev_grid_human_distributions import w1


def event_key(row):
    return row['person_id'], row['formal_round']


def nested_weights(components, *, quantity_weighted=False):
    people = defaultdict(lambda: defaultdict(list))
    for c in components:
        people[c['person_id']][c['formal_round']].append(c)
    output = {}
    for rounds in people.values():
        for rows in rounds.values():
            denom = sum(c['quantity'] if quantity_weighted else 1 for c in rows)
            for c in rows:
                within = c['quantity'] if quantity_weighted else 1
                output[c['component_id']] = within / (len(people) * len(rounds) * denom)
    return output


def compress(points):
    masses = defaultdict(list)
    for value, mass in points:
        if mass > 0:
            masses[float(value)].append(mass)
    return [(v, math.fsum(masses[v])) for v in sorted(masses)]


def crps(points, actual):
    """E|X-y| - 1/2 E|X-X'|, in dollars, for a normalized discrete forecast."""
    absolute = math.fsum(p * abs(v - actual) for v, p in points)
    cumulative_mass = cumulative_value = pair_half = 0.0
    for value, mass in sorted(points):
        pair_half += mass * (value * cumulative_mass - cumulative_value)
        cumulative_mass += mass
        cumulative_value += mass * value
    return absolute - pair_half


def mixture(rows, weights, forecasts):
    return compress((value, weights[c['component_id']] * p)
                    for c in rows for value, p in forecasts[c['component_id']])


def actual_distribution(rows, weights, labels, *, projected=False):
    return compress((project_price(labels[c['component_id']], 5) if projected else labels[c['component_id']],
                     weights[c['component_id']]) for c in rows)


def forecast_metrics(rows, weights, forecasts, labels, contexts):
    if not rows:
        return {'raw_human_w1': None, 'grid5_human_w1': None, 'conditional_crps': None,
                'distribution': distribution([]), 'mass_ge65': None, 'mass_ge700': None,
                'mass_above_own_p': None}
    points = mixture(rows, weights, forecasts)
    raw = actual_distribution(rows, weights, labels)
    projected = actual_distribution(rows, weights, labels, projected=True)
    return {'raw_human_w1': w1(raw, points), 'grid5_human_w1': w1(projected, points),
            'conditional_crps': math.fsum(weights[c['component_id']] * crps(forecasts[c['component_id']], labels[c['component_id']]) for c in rows),
            'distribution': distribution(points),
            'mass_ge65': math.fsum(m for x, m in points if x >= 65),
            'mass_ge700': math.fsum(m for x, m in points if x >= 700),
            'mass_above_own_p': math.fsum(weights[c['component_id']] * math.fsum(p for x, p in forecasts[c['component_id']] if x > contexts[event_key(c)]['current_state']['p']) for c in rows)}


def summarize(plan, predictions, labels):
    components = plan['components']
    indexed = {c['component_id']: c for c in components}
    lookup = {p['component_id']: p for p in predictions}
    if len(indexed) != len(components) or len(lookup) != len(predictions) or set(indexed) != set(lookup) or set(labels) != set(indexed):
        raise ValueError('complete unique component coverage required')
    if any(type(y) is not int or not 1 <= y <= 999 for y in labels.values()):
        raise ValueError('invalid raw human price')
    contexts = {event_key(c): c for c in plan['contexts']}
    if len(contexts) != len(plan['contexts']):
        raise ValueError('duplicate context')
    raw_forecasts, retained, samples, query_raw = {}, {}, {}, {}
    for cid, c in indexed.items():
        p = lookup[cid]
        if any(p[k] != c[k] for k in ('person_id', 'formal_round', 'query_id', 'quantity')) or event_key(c) not in contexts:
            raise ValueError('component identity/quantity mismatch')
        raw = p['raw_probabilities']
        normalized, _ = normalize_jev_probabilities(raw, list(PRICE_VALUES))
        if c['query_id'] in query_raw and query_raw[c['query_id']] != raw:
            raise ValueError('shared query has inconsistent probabilities')
        query_raw[c['query_id']] = raw
        raw_forecasts[cid] = [(float(PRICE_VALUES[k]), v) for k, v in normalized.items() if v > 0]
        if p['outcome'] == 'client_numeric_abstain':
            if p['price'] is not None:
                raise ValueError('abstention cannot be a numeric price')
            continue
        if p['outcome'] != 'selected' or type(p['price']) not in (int, float) or p['price'] not in PRICE_VALUES.values():
            raise ValueError('invalid outcome or price')
        d = p['decision'].get('decision', p['decision'])
        kept = d['retained_normalized_probabilities']
        if (set(kept) != set(PRICE_VALUES) or
                any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in kept.values()) or
                abs(math.fsum(kept.values()) - 1) > 1e-9 or kept[f'p{int(p["price"]):03d}'] <= 0):
            raise ValueError('invalid retained distribution')
        retained[cid] = [(float(PRICE_VALUES[k]), v) for k, v in kept.items() if v > 0]
        samples[cid] = [(float(p['price']), 1.0)]
    events = defaultdict(list)
    for c in components:
        events[event_key(c)].append(c)
    group_rows = {'single': [c for c in components if len(events[event_key(c)]) == 1],
                  'multiple': [c for c in components if len(events[event_key(c)]) > 1],
                  'all': list(components)}
    group_rows['history_yes'] = [c for c in components if any(o['side'].lower() == 'sell' for h in contexts[event_key(c)]['history'] for o in h['orders'])]
    yes = {c['component_id'] for c in group_rows['history_yes']}
    group_rows['history_no'] = [c for c in components if c['component_id'] not in yes]
    for name, lo, hi in [('surplus_1_20', 1, 20), ('surplus_21_80', 21, 80), ('surplus_81_plus', 81, 400)]:
        group_rows[name] = [c for c in components if lo <= contexts[event_key(c)]['current_state']['Q']-contexts[event_key(c)]['current_state']['D'] <= hi]
    groups = {}
    for name, rows in group_rows.items():
        if not rows:
            continue
        common = [c for c in rows if c['component_id'] in retained]
        weights = nested_weights(common)
        full_weights = nested_weights(rows)
        quoted_people = {c['person_id'] for c in common}
        record = {'planned_people': len({c['person_id'] for c in rows}), 'planned_events': len({event_key(c) for c in rows}),
                  'planned_components': len(rows), 'physical_queries': len({c['query_id'] for c in rows}),
                  'quoted_people': len(quoted_people), 'quoted_events': len({event_key(c) for c in common}),
                  'quoted_components': len(common), 'client_abstained_components': len(rows)-len(common),
                  'human_full': distribution(actual_distribution(rows, full_weights, labels)),
                  'human_common': distribution(actual_distribution(common, weights, labels)),
                  'human_common_grid5': distribution(actual_distribution(common, weights, labels, projected=True)),
                  'forecasts': {}, 'component_quote_counts': {},
                  'raw_full': forecast_metrics(rows, full_weights, raw_forecasts, labels, contexts)}
        for mode, forecasts in [('raw', raw_forecasts), ('retained', retained), ('sample', samples)]:
            record['forecasts'][mode] = forecast_metrics(common, weights, forecasts, labels, contexts)
        for threshold in (65, 700):
            record['component_quote_counts'][f'ge{threshold}'] = {'high_components': sum(lookup[c['component_id']]['price'] >= threshold for c in common),
                                                                 'planned_components': len(rows), 'quoted_components': len(common)}
        record['human_price_off_grid_components'] = sum(labels[c['component_id']] not in PRICE_VALUES.values() for c in rows)
        record['quantity_weighted_sensitivity'] = forecast_metrics(common, nested_weights(common, quantity_weighted=True), retained, labels, contexts)
        loo = []
        for person in sorted(quoted_people):
            left = [c for c in common if c['person_id'] != person]
            if left:
                loow = nested_weights(left)
                loo.append(w1(actual_distribution(left, loow, labels), mixture(left, loow, retained)))
        record['leave_one_person_out_w1'] = {'n': len(loo), 'minimum': min(loo) if loo else None, 'maximum': max(loo) if loo else None,
                                             'interpretation': 'Descriptive leave-one-person sensitivity, not a confidence interval or causal effect.'}
        groups[name] = record
    priced_events = set(events)
    return {'schema': 'jev-expanded-sell-oracle-results-v1', 'primary_group': 'single',
            'coverage': {'roster_people': len(plan['roster']), 'candidate_person_rounds': len(plan['roster'])*5,
                         'state_selected_people': len({k[0] for k in contexts}), 'state_selected_events': len(contexts),
                         'oracle_priced_people': len({c['person_id'] for c in components}), 'oracle_priced_events': len(events),
                         'no_sell_events': len(contexts)-len(priced_events), 'components': len(components),
                         'physical_queries': len(query_raw), 'quoted_components': len(retained)},
            'groups': groups,
            'interpretation': 'Oracle human quantity is supplied exactly; target prices remain scoring-only. Primary is nested person/round/component-balanced retained-mixture W1 for single-SELL events. Multiple-SELL events miss other current-order context and are secondary. Raw, retained and actual component samples differ; query dedup does not create independent people or model replicates. CRPS is a conditional diagnostic, not proof of calibration. Raw and grid5-projected humans are distinct. This is exposed development data, teacher-forced static prediction, not a deployable complete agent or independent market confirmation. Abstentions are not zero quotes; technical partial results cannot become a complete report.'}


def public_aggregate(result, *, minimum_people=3):
    output = deepcopy(result)
    groups = output['groups']
    suppress = {k for k, g in groups.items() if min(g['planned_people'], g['quoted_people']) < minimum_people}
    for family in [('history_yes', 'history_no'), ('surplus_1_20', 'surplus_21_80', 'surplus_81_plus'), ('single', 'multiple')]:
        if any(k in suppress for k in family):
            suppress.update(k for k in family if k in groups)
    for name in suppress:
        g = groups[name]
        groups[name] = {'suppressed': True, 'planned_people': g['planned_people'], 'planned_events': g['planned_events'],
                        'planned_components': g['planned_components'], 'quoted_components': g['quoted_components']}
    output['publication_note'] = 'Small participant groups and their complementary numeric strata are suppressed; full results stay private. Counts are retained and ALL denominators are not reduced.'
    return output


async def report_async(root, output):
    from capclaims.sell_oracle import runner
    plan, predictions, status = await runner.replay(root)
    if status['status'] != 'COMPLETE':
        raise ValueError('refuse complete-score report for technical-partial study')
    labels = json.loads((Path(root)/'labels-private.json').read_text(encoding='utf-8'))
    full = summarize(plan, predictions, labels)
    full.update(status=status, plan_sha256=runner.sha((Path(root)/'plan.json').read_bytes()), accounting=accounting(root))
    (Path(root)/'analysis-private.json').write_text(json.dumps(full, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(public_aggregate(full), ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    return full


def report(root=None, output=None):
    from capclaims.sell_oracle import runner
    return asyncio.run(report_async(root or runner.OUT, output or runner.ROOT/'docs/analysis/jev-expanded-sell-oracle-2026-10-28.json'))

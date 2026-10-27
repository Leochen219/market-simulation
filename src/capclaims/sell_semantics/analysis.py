"""Frozen paired SELL-price distribution analysis; human labels are scoring-only."""
from __future__ import annotations

import asyncio
from copy import deepcopy
import json
import math
from pathlib import Path
from statistics import median

from capclaims.backends.jev import normalize_jev_probabilities
from capclaims.flat_price_grid import PRICE_VALUES
from capclaims.flat_grid_study.analysis import accounting, distribution, event_counts, project_price
from tools.analysis.compare_jev_grid_human_distributions import weighted, w1

ARMS = ('original', 'rules')
STRATA = ('all', 'has_sell_history', 'no_sell_history')
EVENTS = ('ge700', '700_to_800', 'gt250', 'above_retail')


def identity(row):
    return tuple(row[k] for k in ('person_id', 'formal_round', 'condition', 'repeat', 'arm'))


def has_sell_history(context):
    """Only past submitted orders determine membership; never use a target."""
    return any(o['side'].lower() == 'sell' for h in context['history'] for o in h['orders'])


def history_count(context):
    return sum(any(o['side'].lower() == 'sell' for o in h['orders']) for h in context['history'])


def score_summaries(rows):
    raw, retained, totals = [], [], []
    for row in rows:
        scores = row.get('raw_probabilities')
        if scores is None:
            continue
        normalized, _ = normalize_jev_probabilities(scores, list(PRICE_VALUES))
        totals.append(math.fsum(scores.values()))
        raw.append(normalized)
        d = row.get('decision', {}).get('decision', {})
        kept = d.get('retained_normalized_probabilities')
        if kept is not None:
            retained.append(kept)
    result = {'raw_available_paths': len(raw), 'retained_available_paths': len(retained),
              'raw_score_total_min': min(totals) if totals else None,
              'raw_score_total_max': max(totals) if totals else None}
    for label, predicate in [('le20', lambda x: x <= 20), ('ge65', lambda x: x >= 65),
                             ('ge700', lambda x: x >= 700)]:
        for name, values in [('raw', raw), ('retained', retained)]:
            result[f'{name}_normalized_mass_{label}'] = (
                math.fsum(math.fsum(p for k, p in d.items() if predicate(PRICE_VALUES[k]))
                          for d in values) / len(values) if values else None)
    return result


def summarize(plan, predictions, targets):
    """Return (full aggregate, private person audit), without modifying inputs."""
    scheduled = {identity(r): r for r in plan['schedule']}
    lookup = {identity(r): r for r in predictions}
    if len(scheduled) != len(plan['schedule']) or len(lookup) != len(predictions):
        raise ValueError('duplicate path')
    if set(lookup) != set(scheduled):
        raise ValueError('complete comparison requires all planned outcomes')
    contexts = {(c['person_id'], c['formal_round'], c['condition']): c for c in plan['contexts']}
    if len(contexts) != len(plan['contexts']) or any(c['side'] != 'SELL' for c in contexts.values()):
        raise ValueError('expected unique supplied-SELL contexts')
    for key, row in lookup.items():
        if key[:3] not in contexts or key[4] not in ARMS:
            raise ValueError('unknown context or arm')
        if row['fixed_quantity'] != scheduled[key]['conditioned_quantity']:
            raise ValueError('conditioned quantity changed')
        if row['outcome'] not in ('selected', 'client_numeric_abstain'):
            raise ValueError('technical failure cannot be scored as STOP or quote')
        if row['outcome'] == 'selected':
            if type(row['price']) not in (int, float) or row['price'] not in PRICE_VALUES.values():
                raise ValueError('invalid selected price')
        elif row['price'] is not None:
            raise ValueError('abstention is not a numeric price')
    groups, private = {}, []
    for condition in sorted({k[2] for k in scheduled}):
        window = {k: c for k, c in contexts.items() if k[2] == condition}
        if len({k[0] for k in window}) != len(window):
            raise ValueError('multiple targets per person not supported by this study')
        groups[condition] = {}
        for stratum in STRATA:
            chosen = {k: c for k, c in window.items() if stratum == 'all' or
                      has_sell_history(c) == (stratum == 'has_sell_history')}
            if not chosen:
                continue
            paths = {}
            for key in chosen:
                expected = {(*key, repeat, arm) for repeat in (0, 1) for arm in ARMS}
                actual = {k for k in scheduled if k[:3] == key}
                if actual != expected:
                    raise ValueError('expected both arms and two repeats per context')
                for repeat in (0, 1):
                    if scheduled[(*key, repeat, ARMS[0])]['conditioned_quantity'] != scheduled[(*key, repeat, ARMS[1])]['conditioned_quantity']:
                        raise ValueError('paired arm quantities differ')
                paths[key[0]] = [(*key, repeat) for repeat in (0, 1)]
            people = sorted(paths)
            human = {p: [o['price'] for o in targets[p]['orders'] if o['side'].lower() == 'sell'] for p in people}
            if any(not h for h in human.values()):
                raise ValueError('supplied SELL target has no SELL orders')
            common = {p: [k for k in paths[p] if all(lookup[(*k, arm)]['price'] is not None for arm in ARMS)] for p in people}
            covered = [p for p in people if common[p]]
            raw_human = weighted([human[p] for p in covered]) if covered else []
            grid_human = weighted([[project_price(v, 5) for v in human[p]] for p in covered]) if covered else []
            g = {'eligible_people': len(people), 'planned_paths_per_arm': 2 * len(people),
                 'common_people': len(covered), 'common_paths': sum(len(common[p]) for p in covered),
                 'historical_sell_rounds': sorted(history_count(c) for c in chosen.values()),
                 'descriptive_single_person': len(people) == 1,
                 'human_full_distribution': distribution(weighted([human[p] for p in people])),
                 'common_distributions': {'human_raw': distribution(raw_human), 'human_grid5': distribution(grid_human)},
                 'arms': {}}
            points_by_arm = {}
            for arm in ARMS:
                keys = [(*k, arm) for p in people for k in paths[p]]
                rows = [lookup[k] for k in keys]
                prices = [r['price'] for r in rows]
                retails = [contexts[k[:3]]['current_state']['p'] for k in keys]
                points = weighted([[lookup[(*k, arm)]['price'] for k in common[p]] for p in covered]) if covered else []
                points_by_arm[arm] = points
                es = {name: event_counts(prices, retails, name) for name in EVENTS}
                es['ge65'] = {**event_counts(prices, retails, 'ge700')}
                es['ge65']['events'] = sum(p is not None and p >= 65 for p in prices)
                count, n, quoted = es['ge65']['events'], len(prices), sum(p is not None for p in prices)
                es['ge65'].update(confirmed_quote_event_per_planned=count/n, among_quotes=count/quoted if quoted else None,
                                  hypothetical_unreported_price_bounds=[count/n, (count+n-quoted)/n])
                g['arms'][arm] = {'quoted': quoted, 'client_numeric_abstain': n - quoted,
                                  'raw_human_w1': w1(raw_human, points) if covered else None,
                                  'grid5_human_w1': w1(grid_human, points) if covered else None,
                                  'events': es, 'scores': score_summaries(rows)}
                g['common_distributions'][arm] = distribution(points)
            deltas = []
            for p in covered:
                person_human = weighted([human[p]])
                errors = {arm: w1(person_human, weighted([[lookup[(*k, arm)]['price'] for k in common[p]]])) for arm in ARMS}
                delta = errors['rules'] - errors['original']
                deltas.append(delta)
                if stratum == 'all':
                    key = paths[p][0][:3]
                    private.append({'person_id': p, 'formal_round': key[1], 'condition': condition,
                                    'has_sell_history': has_sell_history(contexts[key]),
                                    'historical_sell_rounds': history_count(contexts[key]),
                                    'common_paths': len(common[p]), 'auxiliary_individual_w1': errors,
                                    'auxiliary_rules_minus_original': delta,
                                    'prices': {arm: [lookup[(*k, arm)]['price'] for k in paths[p]] for arm in ARMS}})
            g['rules_minus_original_raw_w1'] = (g['arms']['rules']['raw_human_w1'] - g['arms']['original']['raw_human_w1']) if covered else None
            g['auxiliary_person_w1_delta'] = {'n': len(deltas), 'mean': math.fsum(deltas)/len(deltas) if deltas else None,
                                             'median': median(deltas) if deltas else None,
                                             'improved_people': sum(v < -1e-10 for v in deltas),
                                             'worsened_people': sum(v > 1e-10 for v in deltas),
                                             'tied_people': sum(abs(v) <= 1e-10 for v in deltas)}
            groups[condition][stratum] = g
    return ({'schema': 'jev-sell-semantics-comparison-v1', 'people': len({k[0] for k in scheduled}),
             'planned_requests': len(scheduled), 'groups': groups,
             'interpretation': 'Primary is person-balanced pooled price W1 on paired successful paths, not mean per-person error. Contemporary arms differ only by a bundle of neutral economic definitions. No quote is not zero, model STOP, or a technical failure. No-SELL-history has one person and is descriptive only. Quantities are old model selections, not human target quantities; human prices can be off-grid. Scores, retained sampling distributions and observed samples are distinct, uncalibrated evidence. Two repeats of eight development people in one market do not identify causal error shares, model capacity or cross-market generalization.'}, private)


def public_aggregate(result):
    """Avoid singleton and ALL-minus-complement disclosure in public results."""
    output = deepcopy(result)
    for condition, groups in output['groups'].items():
        output['groups'][condition] = {
            'all': groups['all'],
            'stratum_counts': {name: groups[name]['eligible_people'] for name in STRATA if name in groups},
        }
        output['groups'][condition]['all'].pop('historical_sell_rounds', None)
    output['publication_note'] = ('Numeric subgroup results are private: one no-SELL-history person and its complement could identify a personal distribution. Public metrics/CDFs cover all eight only; this suppression does not omit anyone from the ALL denominator.')
    return output


async def report_async(root, output):
    from capclaims.sell_semantics import runner
    plan, predictions, status = await runner.replay(root)
    if status['status'] != 'COMPLETE':
        raise ValueError('technical-incomplete study is not a complete comparison')
    targets = json.loads((Path(root) / 'labels-private.json').read_text(encoding='utf-8'))
    result, private = summarize(plan, predictions, targets)
    result.update(status=status, plan_sha256=runner.sha((Path(root) / 'plan.json').read_bytes()),
                  accounting=accounting(root))
    (Path(root) / 'analysis-private.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    public = public_aggregate(result)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(public, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    (Path(root) / 'person-score-private.json').write_text(json.dumps(private, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    return result


def report(root=None, output=None):
    from capclaims.sell_semantics import runner
    return asyncio.run(report_async(root or runner.OUT, output or runner.ROOT / 'docs/analysis/jev-sell-semantics-comparison-2026-10-27.json'))

"""Frozen matched v2/v3 calculator completion contrast; preview is offline."""
from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random

from capclaims.randomness import canonical_json
from tools.probes.jev_calculator import build_plan as legacy_plan, _tree, _write, execute

REQUEST_CAP = 444
TOKEN_CAP = 2_000_000
SEED = 20261024


def build_plan():
    rows = []
    for original in legacy_plan()['sessions']:
        if original['arm'] != 'optional':
            continue
        for condition in ('v2', 'v3'):
            row = deepcopy(original)
            row['prompt_condition'] = condition
            rows.append(row)
    random.Random(SEED).shuffle(rows)
    total_tokens = total_requests = 0
    for sequence, row in enumerate(rows):
        row['sequence'] = sequence
        nodes, cost, count = _tree(row, [])
        row.update(reachable_nodes=nodes, max_estimated_input_tokens=cost, max_requests=count)
        total_tokens += cost
        total_requests += count
    if total_requests != REQUEST_CAP or total_tokens > TOKEN_CAP:
        raise ValueError('completion contrast exceeds frozen budget')
    plan = {'model': 'jev-1.14.0', 'seed': SEED, 'sessions': rows,
            'request_cap': REQUEST_CAP, 'token_cap': TOKEN_CAP,
            'max_estimated_input_tokens': total_tokens, 'max_in_flight': 10,
            'instruction_profile': 'calculator_v3',
            'extra_source_paths': ['tools/probes/jev_calculator_completion.py',
                'tools/probes/jev_calculator_prompt.py', 'tools/analysis/analyze_jev_calculator.py',
                'src/capclaims/domain/types.py']}
    digest = hashlib.sha256(canonical_json(plan).encode()).hexdigest()
    return {**plan, 'plan_sha256': digest}


def analyze(directory):
    from tools.probes.jev_calculator_prompt import analyze as analyze_conditions
    return analyze_conditions(directory, conditions=('v2', 'v3'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--analyze', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if args.analyze:
        if args.execute or args.output_dir or not args.output:
            parser.error('--analyze requires --output and excludes execution/preview')
        result = analyze(args.analyze)
        _write(args.output, result)
        print(json.dumps({k: result[k] for k in ('status', 'conditions', 'actual_http_max_overlap',
                                                 'frozen_source_files_still_match')}))
        return 0
    if not args.output_dir:
        parser.error('--output-dir is required for preview/execution')
    plan = build_plan()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    if args.execute:
        result = asyncio.run(execute(plan, args.output_dir))
        result = {k: result[k] for k in ('status', 'plan_sha256', 'budget_usage')}
    else:
        _write(args.output_dir / 'preview.json', plan)
        result = {'status': 'PREVIEW', 'sessions': len(plan['sessions']),
                  'request_cap': REQUEST_CAP, 'max_estimated_input_tokens': plan['max_estimated_input_tokens'],
                  'plan_sha256': plan['plan_sha256']}
    print(json.dumps(result))
    return 0 if result['status'] in ('PREVIEW', 'COMPLETE') else 1


if __name__ == '__main__':
    raise SystemExit(main())

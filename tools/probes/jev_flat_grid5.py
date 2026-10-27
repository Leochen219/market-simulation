"""Synthetic preview, no-network mock execution, and strict offline replay."""
import argparse
import asyncio
import json
from pathlib import Path
from capclaims import flat_price_grid as grid

SEED = 'synthetic-flat-grid5'

def paths():
    return [{'person_id': f'synthetic-{i}', 'repeat': 0, 'side': 'SELL' if i % 2 else 'BUY',
             'state': {'own_q_units': 100, 'own_d_units': 120, 'history': [{'round': 1, 'price': i}],
                       'submitted_order_ledger': {'submitted_sell_quantity_units': 0}, 'partial_orders': [],
                       'public_instructions': {}}} for i in range(10)]

async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['preview', 'mock-run', 'replay'])
    parser.add_argument('--file', type=Path, help='Mock evidence path (required for mock-run and replay)')
    args = parser.parse_args()
    if args.mode == 'preview':
        sizes = []
        for p in paths():
            for stage, quantity in [('quantity', None), ('price', 10)]:
                state, options = grid.stage_spec(p['state'], p['side'], stage, quantity)
                sizes.append(len(await grid.checked_body(state, stage, options)))
        print(json.dumps({'version': grid.VERSION, 'paths': 10, 'price_choices': len(grid.PRICE_VALUES), 'max_body_bytes': max(sizes)}))
    elif args.mode == 'mock-run':
        if args.file is None: parser.error('--file is required')
        client = grid.SyntheticClient(barrier=10)
        rows = paths(); results = await grid.run_batch(client, rows, seed=SEED)
        evidence = {'version': grid.VERSION, 'seed': SEED, 'paths': rows, 'results': results, 'peak': client.peak}
        with args.file.open('x', encoding='utf-8') as f: json.dump(evidence, f, ensure_ascii=False, allow_nan=False)
        print(json.dumps({'paths': len(rows), 'peak': client.peak, 'requests': sum(len(x['steps']) for x in results)}))
    else:
        if args.file is None: parser.error('--file is required')
        with args.file.open(encoding='utf-8') as f: evidence = json.load(f)
        if set(evidence) != {'version','seed','paths','results','peak'} or evidence['version'] != grid.VERSION or evidence['seed'] != SEED or evidence['paths'] != paths() or evidence['peak'] != 10:
            raise ValueError('mock evidence envelope mismatch')
        await grid.replay(evidence['paths'], evidence['results'], seed=evidence['seed'])
        print(json.dumps({'status': 'replay_pass', 'paths': len(evidence['paths'])}))

if __name__ == '__main__': asyncio.run(main())

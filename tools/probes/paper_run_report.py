"""Extract a sanitized, replay-verified technical acceptance report from a run."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from capclaims.paper.analysis import summarize_events
from capclaims.paper.replay import replay_paper


def report(root: Path) -> dict:
    replay = replay_paper(root)
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    audits = [json.loads(row) for row in (root / 'audit.jsonl').read_text(encoding='utf-8').splitlines()]
    groups = {row['telemetry']['group_id']: row['telemetry'] for row in audits
              if row.get('telemetry', {}).get('group_id')}
    usage = [row['usage'] for row in audits if row['backend'] == 'laya']
    summary = summarize_events(replay.events, manifest['config']['wholesale_price'])
    return {
        'status': 'PASS', 'profile': manifest['profile'], 'config': manifest['config'],
        'rounds': replay.rounds, 'formal_rounds': replay.formal_rounds,
        'trades_all_rounds': replay.trades, 'volume_all_rounds': replay.volume,
        'formal_profit_cents': replay.profit_cents, 'final_hash': replay.final_hash,
        'decision_count': len(audits), 'inference_group_count': len(groups),
        'observed_forward_batch_sizes': sorted({size for group in groups.values()
                                               for size in group['forward_batch_sizes']}),
        'sum_inference_group_seconds': sum(group['elapsed_seconds'] for group in groups.values()),
        'peak_cuda_allocated_bytes': max((group['peak_cuda_bytes'] or 0 for group in groups.values()), default=0),
        'peak_cuda_reserved_bytes': max((group['peak_cuda_reserved_bytes'] for group in groups.values()
                                         if group.get('peak_cuda_reserved_bytes') is not None), default=None),
        'max_input_tokens': max((row['input_tokens'] for row in usage), default=None),
        'max_state_tokens': max((row['state_tokens'] for row in usage), default=None),
        'truncated_rows': sum(bool(row['truncated']) or bool(row['state_tokens_dropped']) for row in usage),
        'model_revisions': sorted({row['model_revision'] for row in audits}),
        'maximum_orders_observed': max((sum(o['owner'] == owner for o in event['orders'])
                                       for event in replay.events for owner in event['procurement']), default=0),
        'formal_aggregate': summary['aggregate'],
        'scope': 'Technical pilot; no human-behavior or treatment-effect reproduction claim.',
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = report(args.run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()

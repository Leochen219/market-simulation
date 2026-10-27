"""Post-hoc aggregate-only audit of flat-grid raw scores and retained support."""
import asyncio
import json
from capclaims.flat_grid_study import runner


async def main():
    _, predictions, status = await runner.replay()
    if status['status'] != 'COMPLETE':
        raise ValueError('Complete strict replay is required')
    traces = runner.readlines(runner.OUT / 'trace.jsonl')
    high_scores = [p for row in traces for k, p in row['raw_probabilities'].items()
                   if int(k[1:]) >= 700]
    retained_high = [k for row in traces
                     for k, p in row['decision']['decision']['retained_normalized_probabilities'].items()
                     if int(k[1:]) >= 700 and p > 0]
    result = {
        'schema': 'jev-flat-grid5-posthoc-tail-audit-v1',
        'plan_sha256': runner.sha((runner.OUT / 'plan.json').read_bytes()),
        'requests': len(traces),
        'price_min': min(p['price'] for p in predictions if p['price'] is not None),
        'price_max': max(p['price'] for p in predictions if p['price'] is not None),
        'max_raw_single_candidate_score_ge700': max(high_scores),
        'raw_candidates_ge700_with_score_at_least_003': sum(p >= .03 for p in high_scores),
        'retained_positive_candidates_ge700': len(retained_high),
        'provider_choice_nonmax': sum(row['provider_nonmax'] for row in traces),
        'provider_choice_outside_candidates': sum(row['provider_choice'] not in row['candidate_ids'] for row in traces),
        'interpretation': 'Post-hoc mechanism diagnostic; aggregate raw tail mass is not zero even when every individual tail option fails cutoff. Not a calibration test or extra model call.'}
    output = runner.ROOT / 'docs/analysis/jev-flat-grid5-comparison-2026-10-27-tail-audit.json'
    output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    asyncio.run(main())

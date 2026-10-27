"""Expanded paper-domain M0: direct integer choices and a conditional-digit route.

Full-range candidates implement paper §4.1. Conditional digits are an explicitly
declared AI interface adaptation, not a flat joint-argmax implementation.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time


def margin_agreement(batch: list[dict], singles: list[dict]) -> tuple[int, float | None]:
    eligible = []
    for row, single in zip(batch,singles,strict=True):
        ranked = sorted(row,key=row.get,reverse=True)
        if len(ranked)>1 and row[ranked[0]]-row[ranked[1]]>2e-4:
            eligible.append(ranked[0]==max(single,key=single.get))
    return len(eligible), sum(eligible)/len(eligible) if eligible else None


def full_integer_criteria(low: int, high: int) -> dict[str, str]:
    return {f'v{value}': str(value) for value in range(low, high + 1)}


def build_paper_states(decision_interface: str = "legacy_v1") -> list[str]:
    """Actual production observation builder, varied histories and full-width orders."""
    from capclaims.paper.observations import build_paper_observation
    from capclaims.randomness import canonical_json
    rows = []
    for index in range(10):
        owner = f'agent_{index:02d}'
        def portfolio(round_index):
            return [dict(order_id=f'r{round_index:03d}:{owner}:slot{slot}', owner=owner,
                         side='buy' if slot % 2 == 0 else 'sell',
                         qty=999 if slot % 2 == 0 else 100, limit_dollars=999)
                    for slot in range(5)]
        history = [dict(round=r+1,p=100,Q=400,D=200,orders=portfolio(r),
                        profit_cents=-19980000,net_quantity=-200,mcp_cents=99900)
                   for r in range(18-index % 3,18)]
        extra = {} if decision_interface == "legacy_v1" else {"decision_interface":decision_interface}
        row = build_paper_observation(agent_id=owner,round_index=18,treatment='GA',
            n_agents=10,wholesale_price=50,p=100-index,history=history,
            partial_orders=portfolio(18)[:4],phase='orders',q=400,d=199-index,
            slot=4,decision_field='price',own_side='buy',own_qty=999,**extra)
        row.update(numeric_prefix='99',choice_minimum=1,choice_maximum=999,
                   legal_prefix_mask=list(range(10)),decimal_place='u')
        if decision_interface != "legacy_v1":
            row.pop('legal_prefix_mask', None)
            from capclaims.paper.interface import numeric_observation
            row = numeric_observation(row,'99',1,999,'u')
        rows.append(canonical_json(row))
    return rows


def run(model_path: Path, device: str, output_dir: Path) -> dict:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    import torch
    import laya
    from tools.probes.laya_capability_probe import (
        load_and_configure_laya_agent, _instrument_root_forward, _probabilities, _token_ids,
    )
    from capclaims.backends.laya import _instruction
    agent, revision, precision = load_and_configure_laya_agent(model_path, device, torch, laya)
    calls, hook = _instrument_root_forward(agent.model)
    from importlib.metadata import version
    import hashlib
    result = {'status':'FAIL','revision':revision,'precision':precision,'direct':{},'prefix':{},
              'sdk_version':version('laya'),'torch':torch.__version__,'cuda':torch.version.cuda,
              'gpu':torch.cuda.get_device_name(device) if device.startswith('cuda') else None,
              'probe_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    raw = {}
    states = build_paper_states()

    def sync():
        if device.startswith('cuda'):
            torch.cuda.synchronize(device)

    def predict(states_, criteria, instruction):
        calls.clear()
        sync()
        start = time.perf_counter()
        answers = agent.predict_batch(states_, {'decision':{'type':'choice','instructions':instruction,
                                                           'criteria':criteria}},
                                      batch_size=len(states_),max_len=1024,head_max_len=256)
        sync()
        shapes = [c['input_shape'][0] for c in calls if c['input_shape']]
        probabilities = [_probabilities(answer, list(criteria)) for answer in answers]
        return answers, probabilities, shapes, time.perf_counter()-start

    try:
        if device.startswith('cuda'):
            torch.cuda.reset_peak_memory_stats(device)
        # Probe direct candidate fit before choosing the adapted conditional route.
        for label, low, high in [('purchase',0,400),('price',1,999)]:
            try:
                answers, probabilities, sizes, elapsed = predict(
                    states, full_integer_criteria(low,high),
                    'Choose the listed integer value to maximize your profit.')
                result['direct'][label] = {'status':'PASS','candidate_count':high-low+1,
                                          'forward_batch_sizes':sizes,'seconds':elapsed}
                raw['direct_'+label] = answers
            except (ValueError, RuntimeError) as exc:
                result['direct'][label] = {'status':'FAIL','candidate_count':high-low+1,
                                          'reason':str(exc),'forward_count':len(calls)}
        criteria = {f'd{n}':str(n) for n in range(10)}
        instruction = _instruction('paper:slot4:price:units')
        answers, probs, sizes, elapsed = predict(states,criteria,instruction)
        singles = [predict([state],criteria,instruction)[1][0] for state in states]
        parity = max(abs(p[k]-s[k]) for p,s in zip(probs,singles) for k in criteria)
        high_margin_count, agreement = margin_agreement(probs,singles)
        changed = list(states)
        changed[0] = changed[0][:-1] + ',"probe_row_perturbation":"different observation with additional tokens"}'
        if _token_ids(agent.tok,changed[0]) == _token_ids(agent.tok,states[0]):
            raise RuntimeError('perturbation did not change input tokens')
        perturbed = predict(changed,criteria,instruction)[1]
        isolation = max(abs(probs[i][k]-perturbed[i][k]) for i in range(1,10) for k in criteria)
        state_tokens = [len(_token_ids(agent.tok,s)) for s in states]
        result['prefix'] = {'status':'PASS' if sizes==[10] and parity<=1e-4 and isolation<=1e-4 and agreement==1.0 else 'FAIL',
                            'forward_batch_sizes':sizes,'batch_single_max_error':parity,
                            'high_margin_row_count':high_margin_count,'high_margin_argmax_agreement':agreement,
                            'other_row_isolation_error':isolation,'seconds':elapsed,
                            'state_tokens':state_tokens,'usage':[a['usage'] for a in answers],
                            'candidate_count':10,'quantization_decimals':4,
                            'interface_adaptation':'conditional digits; not joint flat argmax'}
        raw['prefix_answers'] = answers
        raw['prefix_states'] = states
        raw['prefix_single_probabilities'] = singles
        raw['prefix_perturbed_probabilities'] = perturbed
        result['status'] = result['prefix']['status']
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        hook.remove()
        result['peak_cuda_bytes'] = torch.cuda.max_memory_allocated(device) if device.startswith('cuda') else None
        for name, content in [('result',result),('raw',raw)]:
            (output_dir/f'paper-{name}.json').write_text(json.dumps(content,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path',type=Path,default=Path('.cache/models/laya-multilingual'))
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    result=run(args.model_path,args.device,args.output_dir)
    print(json.dumps(result,sort_keys=True,indent=2))
    return 0 if result['status']=='PASS' else 1


if __name__=='__main__':
    raise SystemExit(main())

"""Build readable private case evidence from replay-validated private bundles."""
import hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
PRIVATE=ROOT/'artifacts/participant-data/jev-sell-price-tail-audit-2026-10-27'
RUNS=ROOT/'artifacts/participant-data'
RUN_NAMES={'argmax':'jev-grid-probability-v1-2026-10-27','cutoff5':'jev-grid-probability-v2-2026-10-27','cutoff3':'jev-grid-probability-v3-2026-10-27'}

def write_bundle(alias,bundle):
    destination=PRIVATE/alias; destination.mkdir(exist_ok=True)
    qstate=bundle['stages']['quantity']['request']['state']
    orders=[order for h in qstate['history'] for order in h['orders']]
    history_summary=[]
    for side in ('SELL','BUY'):
        prices=[o['limit_price_dollars_per_unit'] for o in orders if o['side']==side]
        history_summary.append(f'{side} {len(prices)} 笔，价格范围 {min(prices)}–{max(prices)}' if prices else f'{side} 0 笔')
    control_note='C1 与高价路径的当前状态相同，但其数量和价格条件可不同；仅 S1 与 S2 的区间请求体经哈希核对完全相同。' if alias.startswith('C1') else ('R1 来自不同当前情境：区间 argmax 为高价 b3，实际抽样为 b0。' if alias.startswith('R1') else 'F1 来自不同的当前情境，不是同一人的对照。' if alias.startswith('F1') else 'S1 与 S2 的区间请求体经哈希核对完全相同；其他条件请求不应自动视为相同。')
    lines=[f'# {alias}: {bundle["arm"]}, repeat {bundle["repeat"]}', '',control_note,'',
        f'当前 retail p={qstate["own_p"]}, wholesale w={qstate["wholesale_price_dollars"]}, Q={qstate["own_q_units"]}, D={qstate["own_d_units"]}, Q−D={qstate["own_q_units"]-qstate["own_d_units"]}。',
        '保存历史订单：'+'；'.join(history_summary)+'。',
        '', '以下是该路径真实保存的状态和提示；每步链接的 JSON 是从 wire 证据逐字节复制的原始文件，响应包含完整概率。', '']
    for stage in ('quantity','band','price'):
        entry=bundle['stages'][stage]; step=entry['trace']['step']; rid=step['request_id']; wire=RUNS/RUN_NAMES[bundle['arm']]/'wire-evidence'/hashlib.sha256(rid.encode()).hexdigest()
        for kind in ('request','response'):
            data=(wire/f'{kind}.json').read_bytes(); expected=step[f'{kind}_sha256']; actual=hashlib.sha256(data).hexdigest()
            if actual!=expected: raise ValueError('wire hash mismatch while copying')
            (destination/f'{stage}-{kind}.json').write_bytes(data)
        question=entry['request']['questions']['decision']; lines += [f'## {stage}', '',f'[原始请求]({stage}-request.json) SHA-256 `{step["request_sha256"]}`；[原始响应]({stage}-response.json) SHA-256 `{step["response_sha256"]}`。','', '实际问题指令：','',f'> {question["instructions"]}','', '实际完整状态：','', '```json',json.dumps(entry['request']['state'],ensure_ascii=False,indent=2),'```','']
        if entry['trace'].get('record_type')=='DECODE_FAILURE':
            lines += [f'解码失败：`{step["decode_error"]}`。原始概率见响应。','']
        elif 'decode' in step:
            d=step['decode']; sel=step['selected_id']; interval=entry['selected_interval']
            lines += [f'选择 `{sel}`，值 `{step["selected_value"]}`；同响应 argmax `{step["argmax_id"]}`；原始概率 `{step["raw_probabilities"][sel]}`，保留后概率 `{d["retained_normalized_probabilities"][sel]}`；保留质量 `{d["retained_mass"]}`。',f'抽样值 `{d["draw"]}`，选中区间 `{interval}`，按原始候选顺序计算。','']
        else: lines += [f'Argmax `{step["argmax_id"]}`；值 `{step["selected_value"]}`。','']
    (destination/'case.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

def main():
    bundles=json.loads((PRIVATE/'full-inputs-private.json').read_text(encoding='utf-8'))['bundles']
    high={b['alias']:b for b in bundles if b.get('alias') in ('S1','S2','S3')}; person=high['S1']['context']
    control=next(b for b in bundles if b['context']==person and b['arm']=='cutoff5' and b['repeat']==1 and b['stages']['band']['trace']['step']['selected_id']=='b0')
    failure=next(b for b in bundles if b['side']=='SELL' and b['stages']['price']['trace'].get('record_type')=='DECODE_FAILURE')
    if failure['context']==person: raise ValueError('failure control unexpectedly matches high case')
    counterexample=next(b for b in bundles if b['arm']=='cutoff5' and b['side']=='SELL' and b['stages']['band']['trace']['step']['argmax_id']=='b3' and b['stages']['band']['trace']['step']['selected_id']=='b0')
    if counterexample['context']==person: raise ValueError('counterexample unexpectedly shares high-price context')
    for alias,bundle in {**high,'C1-same-context-low-band':control,'F1-other-context-failure':failure,'R1-high-band-argmax':counterexample}.items(): write_bundle(alias,bundle)
if __name__=='__main__': main()

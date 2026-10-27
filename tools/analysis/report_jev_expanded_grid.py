"""Render a Chinese aggregate report and chart from the frozen public JSON only."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'docs/analysis/jev-expanded-grid-results-2026-10-27'
ARMS=('argmax','cutoff5','cutoff3')
LABELS={'argmax':'Argmax','cutoff5':'Top 3 / 5%','cutoff3':'Top 3 / 3%'}
COLORS={'argmax':'#0072B2','cutoff5':'#D55E00','cutoff3':'#009E73'}

def number(value,places=2):
    return '不可用' if value is None else f'{value:.{places}f}'
def pct(value): return '不可用' if value is None else f'{100*value:.1f}%'

def plot_cdf(data):
    style={'human_raw':('black','-','Human raw'),'human_grid':('grey','--','Human projected'),
        'argmax':('#0072B2','-','Argmax'),'cutoff5':('#D55E00','-','Top 3 / 5%'),'cutoff3':('#009E73','-','Top 3 / 3%')}
    fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    qmax=1.05*max((r['value'] for side in ('BUY','SELL') for curve in data['groups'][side]['quantity']['curves'].values() for r in curve['cdf']),default=1)
    for i,side in enumerate(('BUY','SELL')):
        for j,stage in enumerate(('quantity','price','price')):
            ax=axes[i,j]; xmax=qmax if j==0 else 1000 if j==1 else 100; curves=data['groups'][side][stage]['curves']
            for name,(color,ls,label) in style.items():
                rows=[r for r in curves.get(name,{}).get('cdf',[]) if r['value']<=xmax]
                ax.step([0]+[r['value'] for r in rows]+[xmax],[0]+[r['cdf'] for r in rows]+[rows[-1]['cdf'] if rows else 0],where='post',color=color,linestyle=ls,label=label)
            ax.set_xlim(0,xmax); ax.set_ylim(0,1.02); ax.grid(alpha=.2); ax.set_title(f"{side} — "+('Quantity observed range' if j==0 else 'Price full' if j==1 else 'Price 0–100 zoom'))
            ax.set_ylabel('participant-balanced CDF'); ax.set_xlabel('units' if j==0 else 'dollars')
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='outside lower center',ncol=5)
    fig.suptitle('Human and model common-success distributions; prices conditional on 3% quantity')
    fig.savefig(BASE.parent/(BASE.name+'-cdf.png'),dpi=160);plt.close(fig)


def render(data):
    if data.get('schema')!='jev-expanded-grid-analysis-v1': raise ValueError('unexpected aggregate report schema')
    status=data['status']; groups=data['groups']
    lines=['# 扩展 Jev 网格研究结果 — 2026-10-27','',
        f"运行状态：`{status['status']}`；共有 {data['total_contexts']} 个情境、计划 {data['planned_paths']} 条重复路径，实际请求 {status.get('physical_requests','不可用')} 次。费用见运行记录。",
        '', '**主要结论：** 数量分布中抽样规则的差距略小，但价格分布和解码失败使它不能被认定为整体更好；共同成功价格样本还会掩盖部分较大偏差。',
        '', '样本来自 18 位原开发参与者，正式第 9–13 轮，排除一条库存等于需求的情境；每个保留情境两次重复。输入使用此前固定三轮历史，服务端分别返回数量、区间和全部四个价格分支。',
        '', '**3% 是完整预测流程。** 其他两种价格解码均以同一个 3% 数量为条件，不能作为一般意义上的完整 Argmax 或 5% 预测流程；数量决策单独比较。本样本的 5% 与 3% 数量选择恰好 178/178 一致，因此该条件在本样本中也与 5% 实际数量一致，但这不构成对一般 5% 全流程的验证。旧研究使用七轮历史，不构成历史长度的因果对照。',
        '', '人类和模型按参与者、该参与者合格轮次、该轮订单或重复依次等权。人类侧仅纳入实际存在指定方向订单的情境；该方向由库存/需求确定，不是方向预测。价格使用所有方法共同成功的同一组情境—重复，因失败而选择出的样本不等于完整样本。W1 越小代表该样本的一维分布越接近，数量单位为件、价格单位为美元，不是逐人误差或校准。',
        '', f"人类订单覆盖（全部 {data['total_contexts']} 情境）：指定方向单笔 {data['coverage'].get('one_given_side',0)}、多笔 {data['coverage'].get('multiple_given_side',0)}、仅反方向 {data['coverage'].get('opposite_only',0)}、混合 {data['coverage'].get('mixed',0)}、无订单 {data['coverage'].get('empty',0)}；按方向和轮次的细分见 JSON。",'',
        '| 方向 | 指标 | 合格人 / 情境 | 共同成功路径 / 计划路径 | Argmax 原始 / 网格 W1 | 5% 原始 / 网格 W1 | 3% 原始 / 网格 W1 |',
        '|---|---|---:|---:|---:|---:|---:|']
    for side in ('BUY','SELL'):
        group=groups[side]
        for stage,label in (('quantity','数量'),('price','价格')):
            m=group['metrics'][stage]; vals=[]
            for arm in ARMS:
                x=m['arms'].get(arm,{})
                vals.append(f"{number(x.get('raw_w1'))} / {number(x.get('projected_w1'))}")
            lines.append(f"| {side} | {label} | {group['eligible_people']} / {group['eligible_contexts']} | {m['common_success_slots']} / {m['denominator_planned_slots']} | {' | '.join(vals)} |")
    lines += ['', '人类网格投影仅是表示方式的敏感性分析：逐笔投影到各自可用网格并保留所有订单，并不表示人类已适应网格；原始人类值是主要比较。逐人总订单量相对于模型单笔的次要比较、留一人敏感性及各分层见 JSON。','',
        '| 方法 | 数量解码失败 | 共用 3% 数量失败 | 区间解码失败 | 所选价格解码失败 | 基础设施不可用 | 计划路径 |', '|---|---:|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        f=data['failure_counts'][arm]
        lines.append(f"| {LABELS[arm]} | {f['quantity_decoder_failure_n']} | {f['shared_3pct_quantity_failure_n']} | {f['band_decoder_failure_n']} | {f['selected_price_decoder_failure_n']} | {f['unavailable_infrastructure_n']} | {f['planned_path_denominator']} |")
    lines += ['', f"3% 完整流程完成 {data['planned_paths']-data['failure_counts']['cutoff3']['shared_3pct_quantity_failure_n']-data['failure_counts']['cutoff3']['band_decoder_failure_n']-data['failure_counts']['cutoff3']['selected_price_decoder_failure_n']-data['failure_counts']['cutoff3']['unavailable_infrastructure_n']}/{data['planned_paths']} 条；5% 价格条件对照完成 {data['planned_paths']-data['failure_counts']['cutoff5']['shared_3pct_quantity_failure_n']-data['failure_counts']['cutoff5']['band_decoder_failure_n']-data['failure_counts']['cutoff5']['selected_price_decoder_failure_n']-data['failure_counts']['cutoff5']['unavailable_infrastructure_n']}/{data['planned_paths']} 条。", '', '数量自身的解码失败与作为价格上游条件的 3% 数量失败分别计数；上游失败不会错误归因于其他方法的数量解码器。价格失败不会填成零。下表为每个方法原始可用价格构造的保守全合格样本 W1 宽界；它不是置信区间，也不保证排名稳定。','',
        '| 方向 / 方法 | 缺失权重 | 全合格价格 W1 保守界 | 共同成功价格均值 | 共同成功 >$250 权重 | 共同成功高于当轮零售价权重 |', '|---|---:|---:|---:|---:|---:|']
    for side in ('BUY','SELL'):
        m=groups[side]['metrics']['price']
        for arm in ARMS:
            a=m['arms'].get(arm,{}); bounds=a.get('full_eligible_price_w1_loose_bounds'); b='不可用' if bounds is None else f'[{number(bounds[0])}, {number(bounds[1])}]'
            lines.append(f"| {side} / {LABELS[arm]} | {pct(a.get('full_eligible_price_missing_weight'))} | {b} | {number(a.get('weighted_mean'))} | {pct(a.get('high_price_share'))} | {pct(a.get('price_above_retail_share'))} |")
    lines += ['', '重复稳定性只在**请求体 SHA-256 完全相同**的响应对上计算原始概率 L1 均值；其他重复因条件变化不计入这一比较。','', '| 节点 | 可比请求对 / 有两次请求的对数 | 同请求体概率 L1 均值 |','|---|---:|---:|']
    for node,x in data['request_body_stability'].items(): lines.append(f"| {node} | {x['same_request_body_sha_pairs']} / {x['available_pairs']} | {number(x['same_body_raw_probability_l1_mean'],4)} |")
    lines += ['', '![按方向与变量比较的原始人类 W1](jev-expanded-grid-results-2026-10-27.png)', '', '这是描述性开发样本分析；重复和轮次并非独立参与者。结果不能直接证明统计显著、个体准确性、校准或对另一轮历史长度的因果效果。完整分母、留一人和网格敏感性结果见 [聚合 JSON](jev-expanded-grid-results-2026-10-27.json)。','',
        '图表与本报告只读取聚合 JSON；可使用系统 Python（需 matplotlib）离线重建：','', '```powershell','python tools/analysis/report_jev_expanded_grid.py','```','']
    supplement_path=BASE.parent/(BASE.name+'-supplement.json')
    if supplement_path.exists():
        extra=json.loads(supplement_path.read_text(encoding='utf-8'))
        if extra.get('schema')!='jev-expanded-grid-run-supplement-v1': raise ValueError('unexpected accounting supplement')
        n=extra['quantity_decision_n']; equal=extra['quantity_5_3_agreement_n']
        lines += ['', f"同一数量响应上，5% 与 3% 解码选择一致 {equal}/{n} 次；此次没有观察到这两个阈值造成的数量选择差异。", '',
            f"共 {extra['settled_request_n']} 次请求完成结算，输入 token {extra['settled_input_tokens']:,}，费用 USD {extra['settled_cost_nanodollars']/1e9:.9f}；首个请求开始至最后请求结束约 {extra['first_to_last_http_elapsed_seconds']:.1f} 秒。", '']
    buy=groups['BUY']['metrics']['price']['arms']; sell=groups['SELL']['metrics']['price']['arms']
    if all(arm in buy and arm in sell and buy[arm].get('raw_w1') is not None and sell[arm].get('raw_w1') is not None for arm in ARMS):
        bb=buy['cutoff3']['full_eligible_price_w1_loose_bounds']; sb=sell['cutoff5']['full_eligible_price_w1_loose_bounds']; sa=sell['argmax']['full_eligible_price_w1_loose_bounds']
        lines += ['', f"需要特别注意成功样本的选择：BUY 的 3% 在共同成功样本的价格 W1 为 {number(buy['cutoff3']['raw_w1'])}，但其自身全合格样本的保守界为 [{number(bb[0])}, {number(bb[1])}]。共同掩码也剔除了 5% 失败而 3% 成功的路径，所以共同样本结果可能掩盖这些 3% 输出的较大偏差；这不是把缺失价格视为零。", '',
            f"SELL 共同成功样本中 5% 的价格 W1 为 {number(sell['cutoff5']['raw_w1'])}，低于 Argmax 的 {number(sell['argmax']['raw_w1'])}；但全合格样本保守界 [{number(sb[0])}, {number(sb[1])}] 与 Argmax 的 [{number(sa[0])}, {number(sa[1])}] 重叠，不能据此认定 5% 的完整样本价格更好。", '']
    lines += ['', '对缺失价格假设只限合法价格区间时，3% 价格的全合格样本 W1 界在 BUY 和 SELL 两侧均高于条件 Argmax 的对应值；这仍然是本开发样本的描述性结论，不说明原因或对持出人群的表现。SELL 的 5% 与 Argmax 全样本界重叠，分布拟合排序仍未确定。', '',
        '两次经授权的 Claude Opus 5.5 咨询分别讨论设计和结果解释；结果咨询仅发送匿名聚合摘要，没有发送人类记录或代码，也未使用咨询侧工具或 MCP。结果建议与上述 BUY、3% 缺失界及 SELL 5% 不确定性一致；留一人敏感性由本地分析提供，未发送给结果咨询。结果咨询的 [收据](../../artifacts/jev-expanded-grid-2026-10-27/opus-results-result/receipt.json) 和 [建议](../../artifacts/jev-expanded-grid-2026-10-27/opus-results-result/advice.md) 可供核对。', '',
        '下一步可在已查询的完整分支上离线探索区间 argmax/抽样 × 区间内 argmax/抽样的 2×2 分解，不需新 API；它是事后探索，不是默认生产规则。每单位价格、SELL 最低收款与区间界限的文字澄清是尚未验证的假设，需另行受控对照。直接拟合只使用本次 89 情境的对应人类参考，不把其他轮次的历史订单并入主要比较。', '']
    cdf_path=BASE.parent/(BASE.name+'-cdf.json')
    if cdf_path.exists():
        cdf_data=json.loads(cdf_path.read_text(encoding='utf-8'))
        if cdf_data.get('schema')!='jev-expanded-grid-cdf-v1': raise ValueError('unexpected CDF schema')
        lines += ['', '人类与模型的参与者加权均值（与上表使用相同共同成功掩码）：', '', '| 方向 / 变量 | 人类原始 | 人类网格 | Argmax | 5% | 3% |', '|---|---:|---:|---:|---:|---:|']
        for side in ('BUY','SELL'):
            for stage in ('quantity','price'):
                curves=cdf_data['groups'][side][stage]['curves']
                values=[number(curves.get(name,{}).get('weighted_mean')) for name in ('human_raw','human_grid','argmax','cutoff5','cutoff3')]
                lines.append(f"| {side} / {'数量' if stage=='quantity' else '价格'} | {' | '.join(values)} |")
        lines += ['', '![共同成功样本的加权累积分布](jev-expanded-grid-results-2026-10-27-cdf.png)', '', 'CDF 数据经独立离线重建，并逐组核对原始和网格 W1 与冻结报告一致。', '']
        plot_cdf(cdf_data)
    fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    for i,side in enumerate(('BUY','SELL')):
        for j,stage in enumerate(('quantity','price')):
            ax=axes[i,j]; m=groups[side]['metrics'][stage]
            vals=[m['arms'].get(arm,{}).get('raw_w1') for arm in ARMS]
            for k,(arm,value) in enumerate(zip(ARMS,vals)):
                if value is not None:
                    ax.bar(k,value,color=COLORS[arm]); ax.annotate(f'{value:.2f}',(k,value),xytext=(0,3),textcoords='offset points',ha='center',fontsize=9)
                else: ax.text(k,.02,'N/A',ha='center',transform=ax.get_xaxis_transform())
            ax.set_xticks(range(3),[LABELS[a] for a in ARMS],rotation=10); ax.set_ylabel('W1 (units)' if stage=='quantity' else 'W1 (dollars)')
            ax.set_title(f'{side}: '+('quantity' if stage=='quantity' else 'price, common successes')+f" (n={m['common_success_slots']})")
            maximum=max((v for v in vals if v is not None),default=0); ax.set_ylim(0,maximum*1.18 if maximum>0 else 1); ax.grid(axis='y',alpha=.2)
    fig.suptitle('Participant-balanced W1 to human orders; prices conditional on 3% quantity')
    fig.savefig(BASE.with_suffix('.png'),dpi=160); plt.close(fig)
    BASE.with_suffix('.md').write_text('\n'.join(lines),encoding='utf-8')

if __name__=='__main__':
    data=json.loads(BASE.with_suffix('.json').read_text(encoding='utf-8')); render(data)

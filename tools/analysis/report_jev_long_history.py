"""Render the approved long-history public aggregate JSON into Chinese report and chart."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'docs/analysis/jev-long-history-results-2026-10-27'
CONDS=('short3','long10'); LABEL={'short3':'短历史 3 轮','long10':'长历史 10 轮'};COLORS=('#0072B2','#D55E00')
def fmt(x,n=2):return '不可用' if x is None else f'{x:.{n}f}'
def pct(x):return '不可用' if x is None else f'{100*x:.1f}%'

def plot_cdf(data):
    if data.get('schema')!='jev-long-history-cdf-v1':raise ValueError('unexpected CDF schema')
    style={'human_raw':('black','-','Human raw'),'human_grid':('grey','--','Human projected'),'short3':('#0072B2','-','Short: 3 rounds'),'long10':('#D55E00','-','Long: 10 rounds')}
    qmax=1.05*max((r['value'] for side in ('BUY','SELL') for curve in data['groups'][side]['quantity']['curves'].values() for r in curve['cdf']),default=1)
    fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    for i,side in enumerate(('BUY','SELL')):
        for j,stage in enumerate(('quantity','price','price')):
            ax=axes[i,j];xmax=qmax if j==0 else 1000 if j==1 else 100;curves=data['groups'][side][stage]['curves']
            for name,(color,ls,label) in style.items():
                rows=[r for r in curves.get(name,{}).get('cdf',[]) if r['value']<=xmax]
                ax.step([0]+[r['value'] for r in rows]+[xmax],[0]+[r['cdf'] for r in rows]+[rows[-1]['cdf'] if rows else 0],where='post',color=color,linestyle=ls,label=label)
            ax.set_xlim(0,xmax);ax.set_ylim(0,1.02);ax.grid(alpha=.2);ax.set_title(f"{side} — "+('Quantity observed range' if j==0 else 'Price full' if j==1 else 'Price 0–100 zoom'))
            ax.set_ylabel('participant-balanced CDF');ax.set_xlabel('units' if j==0 else 'dollars')
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='outside lower center',ncol=4);fig.suptitle('Human and 3% model distributions: same-person, two-window common successes')
    fig.savefig(BASE.parent/(BASE.name+'-cdf.png'),dpi=160);plt.close(fig)


def render(data):
    if data.get('schema')!='jev-long-history-analysis-v1':raise ValueError('unexpected schema')
    status=data['status'];groups=data['groups'];cov=data['coverage']
    lines=['# Jev 长历史对照：第 16 游戏轮','',
        f"状态 `{status['status']}`，{data['people']} 位原开发参与者，每种历史条件各两次重复；实际请求 {status.get('physical_requests','不可用')} 次。",'',
        '目标是**正式第 11 轮，也就是游戏第 16 轮**。短历史为正式 8–10 / 游戏 13–15；长历史为正式 1–10 / 游戏 6–15。两组使用同一参与者的同一目标状态，唯一设计差异是历史窗口及其派生记忆。17 位保留参与者未用于此测试。','',
        '主要比较为 3% 完整流程，对每位参与者先等权，再在其实际同方向订单或两个模型重复中等权。方向由当前库存与需求给定，模型没有预测方向或订单数。价格主比较只取**两种历史都成功**的相同人—重复组合；缺失值不当作零。W1 越小表示样本的一维分布越接近，不是逐人误差或校准。','',
        f"目标订单覆盖：指定方向单笔 {cov.get('one_given_side',0)} 人、多笔 {cov.get('multiple_given_side',0)} 人、仅反方向 {cov.get('opposite_only',0)} 人、混合 {cov.get('mixed',0)} 人、无订单 {cov.get('empty',0)} 人。",'',
        '| 方向 / 指标 | 合格人数 | 两组共同成功人数 / 重复路径 | 每组计划路径 | 短历史原始 / 网格 W1 | 长历史原始 / 网格 W1 | 人类 / 短 / 长均值 |','|---|---:|---:|---:|---:|---:|---:|']
    for side in ('BUY','SELL'):
        group=groups[side]
        for stage,label in (('quantity','数量'),('price','价格')):
            m=group['metrics'][stage];a=m['conditions'].get('short3',{});b=m['conditions'].get('long10',{})
            lines.append(f"| {side} / {label} | {group['eligible_people']} | {m['common_success_people']} / {m['common_success_slots']} | {m['planned_slots_per_window']} | {fmt(a.get('raw_w1'))} / {fmt(a.get('projected_w1'))} | {fmt(b.get('raw_w1'))} / {fmt(b.get('projected_w1'))} | {fmt(a.get('human_weighted_mean'))} / {fmt(a.get('model_weighted_mean'))} / {fmt(b.get('model_weighted_mean'))} |")
    lines += ['', '网格投影仅是表示方式敏感性分析，不表示观察到人类适应了诊断网格；原始人类订单是主要参考。结果只覆盖当前目标轮，数量及价格均不选择“第一笔”人类订单。','',
        '| 方向 / 历史 | 实际价格成功 / 计划 | 缺失个人权重 | 全合格样本价格 W1 保守界 | 共同成功 >$250 / 高于当轮零售价 |','|---|---:|---:|---:|---:|']
    for side in ('BUY','SELL'):
        m=groups[side]['metrics']['price']
        for cond in CONDS:
            a=m['conditions'].get(cond,{});b=a.get('full_price_w1_loose_bounds');bound='不可用' if b is None else f'[{fmt(b[0])}, {fmt(b[1])}]'
            lines.append(f"| {side} / {LABEL[cond]} | {a.get('full_price_observed_slots',0)} / {m['planned_slots_per_window']} | {pct(a.get('full_price_missing_weight'))} | {bound} | {pct(a.get('above_250_share'))} / {pct(a.get('above_retail_share'))} |")
    lines += ['', '完整样本价格界采用合法模型价格范围和人类值端点构造宽松 CDF 包络；它不是置信区间。共同成功样本可能因失败而发生选择偏差，不能用其单一排名替代缺失值分析。','',
        '| 历史 / 规则 | 数量解码失败 | 共享 3% 数量失败 | 区间失败 | 所选价格失败 | 计划路径 |','|---|---:|---:|---:|---:|---:|']
    for cond in CONDS:
        for arm in ('argmax','cutoff5','cutoff3'):
            x=data['failures'][cond][arm];lines.append(f"| {LABEL[cond]} / {arm} | {x['quantity_decoder_failure_n']} | {x['shared_quantity_failure_n']} | {x['band_decoder_failure_n']} | {x['selected_price_decoder_failure_n']} | {x['planned_paths']} |")
    lines += ['', '**3% 是唯一完整预测流程。** Argmax 与 5% 的价格均以同一个 3% 数量为条件；六路共同成功对照是次要分析，不能称为独立完整流程。','',
        '## 历史分布参照','', '“最近”与“最近状态”各从对应窗口选一个有同方向订单的历史轮，保留该轮全部同方向订单；“窗口池”按人、历史轮、订单依次等权，并在短、长窗口及模型均有覆盖的相同人群比较。SELL 历史总量若超过当前库存则排除。无可用历史即缺失，不用跨人替代。历史参照可能输出多笔订单，而模型每条路径只有一笔；两者的订单单位不同。','',
        '| 方向 / 指标 | 窗口池共同覆盖人 / 路径 | 短：历史 / 模型 W1 | 长：历史 / 模型 W1 |','|---|---:|---:|---:|']
    for side in ('BUY','SELL'):
        for stage,label in (('quantity','数量'),('price','价格')):
            x=groups[side]['pooled_history_baseline'][stage];a=x['conditions'].get('short3',{});b=x['conditions'].get('long10',{})
            lines.append(f"| {side} / {label} | {x['common_covered_people']} / {x['common_model_slots']} | {fmt(a.get('historical_raw_w1'))} / {fmt(a.get('model_same_covered_people_raw_w1'))} | {fmt(b.get('historical_raw_w1'))} / {fmt(b.get('model_same_covered_people_raw_w1'))} |")
    lines += ['', '| 方向 / 历史 / 方法 / 指标 | 覆盖人 / 合格人 | 同覆盖人群：历史 / 模型 W1 |','|---|---:|---:|']
    for side in ('BUY','SELL'):
        for cond in CONDS:
            for method,mlabel in (('last','最近有订单轮'),('nearest','最近状态')):
                for stage,label in (('quantity','数量'),('price','价格')):
                    x=groups[side]['baselines'][f'{cond}_{method}_{stage}']
                    lines.append(f"| {side} / {LABEL[cond]} / {mlabel} / {label} | {x['coverage_people']} / {x['eligible_people']} | {fmt(x.get('historical_raw_w1'))} / {fmt(x.get('model_same_covered_people_raw_w1'))} |")
    lines += ['', '各窗口的最近/最近状态覆盖人群可能不同，不直接跨窗口比较其分数。窗口池表使用两窗口共同覆盖人群。历史多订单参照较接近人类时，只说明可用历史中存在相关信号，不能证明人类非随机或模型固有能力不足，也不能分配误差原因。逐人总订单量对模型单笔的次要距离见 JSON。','',
        '长窗口同时增加早期学习经历与个体历史信息；即使出现差异，也不能归因于某一种信息或推断个体特征。两次模型重复不是人类随机性的校准；18 位开发参与者不是独立持出样本。留一人结果仅作描述，不作为显著性检验。','',
        '![短、长历史在共同成功样本中的 W1](jev-long-history-results-2026-10-27.png)','', '完整覆盖、每项失败、留一人和六路次要对照见 [聚合 JSON](jev-long-history-results-2026-10-27.json)。绘图脚本只读取该公开汇总。','']
    diagnostic=BASE.parent/(BASE.name+'-argmax-diagnostic.json')
    if diagnostic.exists():
        d=json.loads(diagnostic.read_text(encoding='utf-8'))
        if d.get('schema')!='jev-long-history-full-conditional-argmax-price-v1':raise ValueError('invalid argmax diagnostic')
        lines += ['', '## 事后探索：完整覆盖的条件 Argmax 价格诊断', '', '同一 3% 数量下，先取区间 argmax，再取该区间内价格 argmax；这是事后探索、完整覆盖的人类合格样本诊断，既不是全局最高概率价格，也不是独立的完整 argmax 数量—价格流程。它与上文的 3% 主流程是不同的解码规则，不能将两者差异归因于单一阶段或某个阈值。', '', '| 方向 / 历史 | 人数 / 重复 | 原始人类价格 W1 | 人类 / 模型加权均价 | 模型 >$250 权重 |', '|---|---:|---:|---:|---:|']
        for side in ('BUY','SELL'):
            for cond in CONDS:
                v=d['groups'][side][cond];lines.append(f"| {side} / {LABEL[cond]} | {v['people']} / {v['slots']} | {fmt(v['raw_w1'])} | {fmt(v['human_mean'])} / {fmt(v['model_mean'])} | {pct(v['model_price_above_250_share'])} |")
        lines += ['', '条件 argmax 价格差距小于本次 3% 流程，说明解码选择值得进一步检查；但它仍有高价尾部，不能据此认定高价仅由抽样造成，也不能将全部误差归因于解码器。', '']
    lines += ['', '## 结果解释与咨询', '',
        '在本次 3% Jev 配置下，历史价格信号的利用不足：同覆盖 10 名 BUY 参与者时，长历史最近状态参照的价格 W1 为 1.90，模型为 210.50；同覆盖 7 名 SELL 参与者时，长历史最近同方向订单轮参照为 0.86，模型为 277.11。历史参照可含多笔订单，而模型仅给出一笔，差距不能解释为逐笔预测准确率。这个证据不否定人类行为中的随机性，也不能证明模型的固有能力上限或锁定单一误差原因。', '',
        '长历史下 BUY 和 SELL 的数量 W1 均下降；BUY 的两组模型数量均值均为 58，均值相同不代表分布相同。BUY 3% 价格 W1 从 198.40 增至 210.50；SELL 的共同成功样本由 323.59 降至 252.34（15/16 路径），而全样本缺失价格界如上表，显示结果对缺失处理敏感。', '',
        '经授权的两次 Claude Opus 5.5 咨询（`claude-opus-5-5`，`claude-vps`）分别讨论研究设计和结果解释，均已完成；仅发送脱敏设计或聚合摘要，没有发送源数据或原始参与者记录，也未调用咨询侧工具或 MCP。设计咨询请求 SHA-256 为 `403aa39d6038a80edf168a25f99a0706fafcad1afd8705971546b7185d48de6e`，见 [设计收据](../../artifacts/jev-long-history-2026-10-27/opus-design-result/receipt.json)；结果咨询为 `c9e00786c70b7c4107c29e11d35e3a624a4f1c599c86bf05d0f1c311d45673ad`，见 [结果收据](../../artifacts/jev-long-history-2026-10-27/opus-results-result/receipt.json) 和 [建议](../../artifacts/jev-long-history-2026-10-27/opus-results-result/advice.md)。收据分别报告约 USD 0.11444 和 USD 0.109124 的提供方标价；会员额度实际扣减未报告，这些数额不是本次 Jev 已结算费用。', '',
        '下一步可在已保存的分支概率上离线计算区间与区间内选择的混合分布，以区分局部解码与响应形状；这只是零新增 API 的建议，本报告未执行该分析。', '']
    extra=BASE.parent/(BASE.name+'-supplement.json')
    if extra.exists():
        x=json.loads(extra.read_text(encoding='utf-8'));lines += [f"结算：{x['settled_requests']} 次请求，输入 token {x['input_tokens']:,}，USD {x['cost_nanodollars']/1e9:.9f}；首个请求至最后响应约 {x['elapsed_seconds']:.1f} 秒。",'']
    cdf_path=BASE.parent/(BASE.name+'-cdf.json')
    if cdf_path.exists():
        cdf=json.loads(cdf_path.read_text(encoding='utf-8'));plot_cdf(cdf)
        lines += ['', '![人类与短、长历史的共同成功累积分布](jev-long-history-results-2026-10-27-cdf.png)', '', 'CDF 从私有证据离线重建，仅输出聚合；其原始与投影 W1 已逐组核对冻结分析。', '']
    fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    for i,side in enumerate(('BUY','SELL')):
        for j,stage in enumerate(('quantity','price')):
            ax=axes[i,j];m=groups[side]['metrics'][stage];vals=[m['conditions'].get(c,{}).get('raw_w1') for c in CONDS]
            for k,val in enumerate(vals):
                if val is None:ax.text(k,.02,'N/A',ha='center',transform=ax.get_xaxis_transform())
                else:ax.bar(k,val,color=COLORS[k]);ax.annotate(f'{val:.2f}',(k,val),xytext=(0,3),textcoords='offset points',ha='center')
            maximum=max((v for v in vals if v is not None),default=0);ax.set_ylim(0,maximum*1.18 if maximum>0 else 1)
            ax.set_xticks((0,1),('3 rounds','10 rounds'));ax.set_ylabel('W1 (units)' if stage=='quantity' else 'W1 (dollars)');ax.set_title(f"{side} {stage} — common successes n={m['common_success_slots']}");ax.grid(axis='y',alpha=.2)
    fig.suptitle('3% full pipeline: participant-balanced distribution distance to human orders')
    fig.savefig(BASE.with_suffix('.png'),dpi=160);plt.close(fig);BASE.with_suffix('.md').write_text('\n'.join(lines),encoding='utf-8')
if __name__=='__main__':render(json.loads(BASE.with_suffix('.json').read_text(encoding='utf-8')))

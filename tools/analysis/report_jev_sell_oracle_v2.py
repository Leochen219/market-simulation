"""Render only the privacy-filtered aggregate SELL oracle V2 JSON."""
from pathlib import Path
import json

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'docs/analysis/jev-expanded-sell-oracle-v2-2026-10-28'
GROUPS=(('single','单笔SELL（主指标）'),('multiple','多笔SELL（次要描述）'),('all','全部有SELL事件'))
MODES=(('raw','原始分数'),('retained','截断后分布'),('sample','一次抽样'))

def fmt(x):return '不可用' if x is None else f'{x:.2f}'
def percent(x):return '不可用' if x is None else f'{100*x:.2f}%'

def render(data):
    if data.get('status',{}).get('status')!='COMPLETE': raise ValueError('incomplete study cannot be rendered')
    groups=data['groups'];coverage=data['coverage'];account=data['accounting']
    if any(groups[k].get('suppressed') for k,_ in GROUPS): raise ValueError('required aggregate group is suppressed')
    lines=['# 扩展SELL价格诊断：给定真人数量的结果','',
      '本研究使用既有GA10开发数据，预测价格时直接提供真人该笔SELL的精确数量；数量并非模型预测。输入还包括本人此前十轮实际历史及已核验历史利润、当前状态和中性规则说明。未提供目标价格、目标利润或当前其他订单。后续目标轮可看到此前目标轮的真实历史，因此这是逐轮静态预测，不是闭环市场模拟。','',
      f"覆盖：{coverage['roster_people']}人、{coverage['candidate_person_rounds']}个人轮次，先按当前 Q>D 选出{coverage['state_selected_people']}人、{coverage['state_selected_events']}事件；其中{coverage['oracle_priced_people']}人、{coverage['oracle_priced_events']}事件含{coverage['components']}笔实际SELL。其余{coverage['no_sell_events']}个无SELL事件保留在覆盖分母，不视为0元报价。相同事件内相同数量共用一次推断，共{coverage['physical_queries']}次物理请求；{coverage['quoted_components']}/{coverage['components']}笔获得报价。",'',
      '主分析限定一笔SELL的事件；多笔事件及全部有SELL事件分别为次要描述。每人等权、每人内事件等权、事件内SELL分量等权；一次共用推断不增加独立样本。多笔事件中的其他当前订单未提供。','',
      '## 价格分布与条件误差','',
      '| 分组 | 人 / 事件 / SELL笔数 | 分布或抽样 | W1：真人原始价 | W1：真人5元投影 | 真人均价 / P95 | 模型均价 / P95 | 条件CRPS |',
      '|---|---:|---|---:|---:|---:|---:|---:|']
    for key,label in GROUPS:
        g=groups[key];h=g['human_common']; count=f"{g['planned_people']} / {g['planned_events']} / {g['planned_components']}"
        for mode,name in MODES:
            m=g['forecasts'][mode];d=m['distribution']
            lines.append(f"| {label} | {count} | {name} | {fmt(m['raw_human_w1'])} | {fmt(m['grid5_human_w1'])} | {fmt(h['mean'])} / {fmt(h['p95'])} | {fmt(d['mean'])} / {fmt(d['p95'])} | {fmt(m['conditional_crps'])} |")
    primary=groups['single']; retained=primary['forecasts']['retained']; raw=primary['forecasts']['raw']; sample=primary['forecasts']['sample']; multi=groups['multiple']['forecasts']
    def mass_le20(d): return sum(row['mass'] for row in d['cdf'] if row['value']<=20)
    lines += ['', '## 如何理解主结果', '',
      f"单笔SELL中，真人及截断后模型的中位数均为{fmt(primary['human_common']['median'])}美元：模型识别到低价中心，但高价质量仍偏多。真人价格不超过20美元的质量为{percent(mass_le20(primary['human_common']))}，模型截断后为{percent(mass_le20(retained['distribution']))}；截断后≥65美元的质量为{percent(retained['mass_ge65'])}。",
      f"主指标对真人原价W1为{fmt(retained['raw_human_w1'])}，对5元投影为{fmt(retained['grid5_human_w1'])}，网格投影对这一差距的影响较小。单次抽样W1为{fmt(sample['raw_human_w1'])}，低于截断分布的{fmt(retained['raw_human_w1'])}；多笔事件对应{fmt(multi['sample']['raw_human_w1'])}与{fmt(multi['retained']['raw_human_w1'])}。一次预定抽样并不能说明分布稳定改善。",
      f"原始≥700美元质量{percent(raw['mass_ge700'])}，截断后为{percent(retained['mass_ge700'])}，因此本次抽样没有≥700美元报价；这不表示所有高价尾部均已消失。逐人留一敏感性W1范围{fmt(primary['leave_one_person_out_w1']['minimum'])}–{fmt(primary['leave_one_person_out_w1']['maximum'])}；每次移除一人后边际差距仍存在，但该范围不是置信区间。", '',
      '下一步优先离线关联高价质量与当前价格、状态及本人历史SELL，并比较匹配历史检索基线；随后单独设计历史使用干预。这里不追加调用、不改变默认方案或缩减候选网格。']
    lines += ['', '价格与距离单位均为每单位美元。W1比较按上述权重合并的人类与模型价格分布，越低表示边际分布越接近；主指标是单笔SELL、截断后分布对真人原始价格的W1。条件CRPS逐笔以该笔真人原价为观测值，再按同样的嵌套权重汇总；单次抽样的CRPS是点预测绝对误差，不证明校准。','',
      '原始分数先经合法性检查和归一化，但没有证据证明它们是已校准的人类选择概率。截断后分布是实际客户端规则：按原始分数筛选达到3%的候选、最多保留前三项并重新归一化；抽样是一条预定随机数实现。三者不可互换。真人5元网格投影仅作敏感性分析；例如1元投影到5元相差4元。','',
      '![单笔SELL价格CDF](jev-expanded-sell-oracle-v2-2026-10-28.png)','',
      '图左放大0–110美元，图右展示完整0–1000美元区间。曲线为加权累积分布；原始分数的远端尾部与截断后的实际客户端分布分开显示。','',
      '## 尾部与局限','',
      '| 分组 | 表示 | ≥65质量 | ≥700质量 | 高于本人零售价的质量 |',
      '|---|---|---:|---:|---:|']
    for key,label in GROUPS:
        for mode,name in MODES:
            m=groups[key]['forecasts'][mode];lines.append(f"| {label} | {name} | {percent(m['mass_ge65'])} | {percent(m['mass_ge700'])} | {percent(m['mass_above_own_p'])} |")
    lines += ['', '尾部事件是描述统计；SELL限价不等于成交价，高于本人零售价也不自动意味着错误。真人原始限价可不在5元网格上；模型候选仍只有5元网格，未作抖动。','',
      '所有参与者此前均已在本项目中使用，不能称为新留出集或独立确认。此前8人研究使用旧模型选出的固定数量，本研究使用真人数量、更多个人轮次与不同条件；两者不是配对因果对照，不能据此断言数量改变导致效果改善。也不能把结果解释为完整代理的表现或模型能力上限。','',
      '## 运行与复核','',
      f"- {account['requests']}次物理请求，{account['input_tokens']:,}输入tokens，核算USD{account['cost_nanodollars']/1e9:.9f}；费用硬上限USD0.30。",
      f"- 10路并发上限，实际峰值{account['observed_peak_inflight']}；HTTP区间{account['elapsed_seconds']:.2f}秒，最大滚动一秒{account['max_rolling_second_requests']}请求及{account['max_rolling_reserved_bytes']:,}预留字节。",
      '- 请求、响应、费用与抽样经严格回放；旧版本的失败证据保留。本次没有执行20轮市场。',
      '- 完整运行与本地命令参见 [V2操作说明](../engineering/jev-sell-oracle-v2.md)。公开渲染仅读取隐私过滤后的汇总JSON。','',
      '```powershell','python -m tools.probes.jev_sell_oracle_v2 preview','python -m tools.probes.jev_sell_oracle_v2 replay','python -m tools.probes.jev_sell_oracle_v2 report','python -m tools.analysis.report_jev_sell_oracle_v2','```','',
      '预览及回放需要本机已授权数据或私有证据；不要重复启动已运行目录。','']
    BASE.with_suffix('.md').write_text('\n'.join(lines),encoding='utf-8')
    chart(groups['single'])

def chart(group):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    series=[('Human orders',group['human_common'],'#222222'),('Raw scores',group['forecasts']['raw']['distribution'],'#d55e00'),('Retained top-3 / 3%',group['forecasts']['retained']['distribution'],'#0072b2'),('One sample',group['forecasts']['sample']['distribution'],'#009e73')]
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),constrained_layout=True)
    for ax,xmax,title in zip(axes,(110,1000),('Detail: $0–110','Full grid: $0–1000')):
        for label,distribution,color in series:
            rows=distribution['cdf']; x=[0]+[r['value'] for r in rows]+[1000];y=[0]+[r['cdf'] for r in rows]+[1]
            ax.step(x,y,where='post',label=label,color=color,linewidth=1.9)
        ax.set(xlim=(0,xmax),ylim=(0,1.02),xlabel='Submitted SELL limit price ($ / unit)',ylabel='Cumulative mass',title=title);ax.grid(alpha=.22)
    axes[1].legend(loc='lower right',fontsize=8)
    fig.suptitle('Single-SELL events: person-balanced price distributions',fontsize=12)
    fig.savefig(BASE.with_suffix('.png'),dpi=180);plt.close(fig)

def main():
    data=json.loads(BASE.with_suffix('.json').read_text(encoding='utf-8'));render(data)
if __name__=='__main__':main()

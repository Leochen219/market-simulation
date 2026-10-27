"""Render verified aggregate files; singleton/complement detail stays private."""
import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PUBLIC = ROOT / 'docs/analysis/jev-sell-semantics-comparison-2026-10-27'
PRIVATE = ROOT / 'artifacts/participant-data/jev-sell-semantics-v1-2026-10-27'
WINDOWS = ('short3', 'long10')
LABEL = {'short3': '三轮历史', 'long10': '十轮历史'}
STRATA = ('all', 'has_sell_history', 'no_sell_history')
STRATUM_LABEL = {'all': '全部', 'has_sell_history': '有本人SELL历史', 'no_sell_history': '无本人SELL历史'}
ARMS = ('original', 'rules')


def fmt(x):
    return '不可用' if x is None else str(Decimal(str(x)).quantize(Decimal('.01'), rounding=ROUND_HALF_UP))


def pct(x):
    return '不可用' if x is None else f'{100*x:.2f}%'


def groups(data, private):
    for condition in WINDOWS:
        for stratum in STRATA if private else ('all',):
            if stratum in data['groups'][condition]:
                yield condition, stratum, data['groups'][condition][stratum]


def render(data, *, private=False):
    if data['status']['status'] != 'COMPLETE':
        raise ValueError('Do not render partial experiment as complete')
    prompt = json.loads((ROOT/'docs/prompts/sell-semantics-v1.json').read_text(encoding='utf-8'))
    lines = ['# SELL规则语义：与同期原版的配对对照', '',
             '同原批次8位卖方，预测正式第11轮（游戏第16轮）；三轮历史为正式8–10轮，十轮为正式1–10轮，各两次重复。每个窗口、每个版本16条价格预测，共64次真实请求。', '',
             '**原版重新调用，作为同期对照，不是复用上次32条卖价。** 每对的本人历史、当前状态、方向、旧路径已选数量、199个完整价格5–995、原始top-3＋3% cutoff及抽样随机数相同。版本与窗口不改变抽样坐标；两个重复分时间块，版本交错。只有规则版在状态中追加一个中性定义字段。没有新增成交反馈、他人历史、收益计算或推荐动作。', '',
             '## 本次效果', '',
             '| 历史 | 原版价格分布W1 | 规则版W1 | 规则−原版 | 原版／规则版有效报价 |',
             '|---|---:|---:|---:|---:|']
    for c in WINDOWS:
        g=data['groups'][c]['all']; a,b=(g['arms'][k] for k in ARMS)
        lines.append(f"| {LABEL[c]} | {fmt(a['raw_human_w1'])} | {fmt(b['raw_human_w1'])} | {fmt(g['rules_minus_original_raw_w1'])} | {a['quoted']}/{g['planned_paths_per_arm']} ／ {b['quoted']}/{g['planned_paths_per_arm']} |")
    lines += ['', '主指标是按人等权的**合并报价分布**与真人原始同方向限价的W1，单位元，越低越近；不是平均逐人误差。每行两版取完全相同的共同成功路径，真人同一人多单时内部订单等权、模型重复内部等权。', '',
              '结论：三轮历史出现小幅开发集改善，十轮历史的总体分布距离未变；补齐这组规则尚未解决剩余卖价偏高。总体W1不变不代表每个人或每个原始概率都不变。两个重复不足以校准模型概率或确认稳定效果，不能据此宣布语义无作用或模型存在固有能力上限。', '',
              '## 本人SELL历史分组', '']
    if private:
        lines += ['本地私有报告，包含1人小组与7人补集的数值；不上传GitHub或Notion。无SELL组只有同一人，描述性案例变化不等于“缺SELL历史”的因果效应。', '',
                  '| 历史 / 组别 | 人数 / 共同路径 | 原版W1 | 规则版W1 | 规则−原版 |',
                  '|---|---:|---:|---:|---:|']
        for c,s,g in groups(data,True):
            a,b=(g['arms'][k] for k in ARMS)
            lines.append(f"| {LABEL[c]} / {STRATUM_LABEL[s]} | {g['eligible_people']} / {g['common_paths']} | {fmt(a['raw_human_w1'])} | {fmt(b['raw_human_w1'])} | {fmt(g['rules_minus_original_raw_w1'])} |")
    else:
        lines += ['每个历史窗口均为7人有本人SELL记录、1人无SELL记录。完整分组结果已保存本机私有报告。为了避免通过“全部8人−其中7人”重建单人的行为数据，公开报告不展示两个子组的数值分布。总体分母仍保留全部8人。', '',
                  '分组定性结果：有历史组在短窗口改善，长窗口与真人的总体分布距离未变，并不表示CDF相同；无历史的单人案例在短窗口略有改变，长窗口样本价格未改变，其高价偏差仍明显。不能用7对1的开发比较建立组间因果结论。']
    lines += ['', '## 均价、尾部和网格敏感性', '',
              '| 历史 / 组别 | 真人均价 | 原版均价 / P95 | 规则版均价 / P95 | 原版／规则版对5元投影真人W1 |',
              '|---|---:|---:|---:|---:|']
    for c,s,g in groups(data,private):
        d=g['common_distributions'];a,b=(g['arms'][k] for k in ARMS)
        lines.append(f"| {LABEL[c]} / {STRATUM_LABEL[s]} | {fmt(d['human_raw']['mean'])} | {fmt(d['original']['mean'])} / {fmt(d['original']['p95'])} | {fmt(d['rules']['mean'])} / {fmt(d['rules']['p95'])} | {fmt(a['grid5_human_w1'])} ／ {fmt(b['grid5_human_w1'])} |")
    lines += ['', '| 历史 / 组别 / 版本 | ≥65元 / 计划 | ≥700元 / 计划 | >本人零售价 / 计划 | 客户端弃权 |',
              '|---|---:|---:|---:|---:|']
    for c,s,g in groups(data,private):
        for arm in ARMS:
            a=g['arms'][arm];n=g['planned_paths_per_arm']
            lines.append(f"| {LABEL[c]} / {STRATUM_LABEL[s]} / {'原版' if arm=='original' else '规则版'} | {a['events']['ge65']['events']}/{n} | {a['events']['ge700']['events']}/{n} | {a['events']['above_retail']['events']}/{n} | {a['client_numeric_abstain']} |")
    lines += ['', '尾部是描述，不把SELL高于本人p自动视为错误：其他参与者的p不同。报价不等于成交价，提交不等于成交。弃权不是0元或真人STOP，技术失败不能被当作空单。网格投影只衡量表示敏感性，不等同于观察真人适应新网格。', '',
              '## 原始分数与截断抽样', '',
              '| 历史 / 组别 / 版本 | 截断前≤20质量 | 截断后≤20质量 | 截断前≥65质量 | 截断后≥65质量 |',
              '|---|---:|---:|---:|---:|']
    for c,s,g in groups(data,private):
        for arm in ARMS:
            a=g['arms'][arm]['scores']
            lines.append(f"| {LABEL[c]} / {STRATUM_LABEL[s]} / {'原版' if arm=='original' else '规则版'} | {pct(a['raw_normalized_mass_le20'])} | {pct(a['retained_normalized_mass_le20'])} | {pct(a['raw_normalized_mass_ge65'])} | {pct(a['retained_normalized_mass_ge65'])} |")
    lines += ['', '这里先验证并归一化API返回的完整分数，再分别统计截断前与top-3＋3%后的分布；并非模型已校准的真人选择概率。截断前后、实际抽到的价格是三个不同对象。完整原始总和、支持与随机数留在私有证据中。', '',
              '## 本次只增加的原文', '', '```text', prompt['economic_definitions_added_only_in_rules_arm'], '```', '',
              '这组文字未提示低价、最优策略或目标答案。结果仅评价这组定义的整体作用，不能分解哪句话有用。', '',
              '## 边界、验证和本地复核', '',
              '目标是条件行为分布。当前仍固定旧模型数量，而真人可能选择不同数量；数量和价格的相关关系尚未处理，1元等真人报价也不在新网格内。真实人的规则知识与完整经历无法由这些输入完全恢复。一个市场的8名开发参与者、两个重复不构成独立确认；不得从无改善唯一归因为模型能力、训练数据或个体异质性。', '']
    ac=data['accounting']
    lines += [f"- {ac['requests']}次请求、{ac['input_tokens']:,}输入tokens，按冻结价表核算USD{ac['cost_nanodollars']/1e9:.9f}；发送前硬上限USD0.25。",
              f"- 10路并发上限，实测峰值{ac['observed_peak_inflight']}，耗时{ac['elapsed_seconds']:.3f}秒；最大滚动一秒{ac['max_rolling_second_requests']}请求、{ac['max_rolling_reserved_bytes']:,}预留字节。",
              '- 旧实验及提示未改。64请求体最大11609字节，未截断；零自动重试。严格回放校验请求/响应/抽样/费用，独立复算价格距离。',
              '- 离线共享回归63项通过；生命周期修复与隐私过滤后的17项新测试通过，最终过滤小改后的8项分析测试再通过。未运行20轮市场。',
              f"- 冻结SHA256：`{data['plan_sha256']}`。", '',
              '建议保留这次语义完整版本作为开发记录，但不称作已验证的新默认。下一步先离线核对仍高价者的SELL历史、原始价格分数、数量条件和真实信息缺口；本报告不自动追加调用或改变抽样。', '',
              '```powershell', 'python -m tools.probes.jev_sell_semantics replay',
              'python -m tools.probes.jev_sell_semantics report',
              'python -m tools.analysis.report_jev_sell_semantics', '```', '',
              '前两条需要本机完整私有冻结证据；渲染公开报告只需已有汇总JSON、提示快照及Matplotlib。不要对已经启动的目录再次执行run。']
    base=PRIVATE/'comparison-private' if private else PUBLIC
    lines += ['', f"![价格累计分布]({base.name}.png)", '']
    base.with_suffix('.md').write_text('\n'.join(lines).rstrip()+'\n',encoding='utf-8')
    chart(data,base,private=private)


def chart(data,base,*,private=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if private:
        rows=3;fig,axes=plt.subplots(rows,2,figsize=(12,11),squeeze=False)
        panels=[(i,j,c,s,100) for i,s in enumerate(STRATA) for j,c in enumerate(WINDOWS)]
    else:
        rows=2;fig,axes=plt.subplots(rows,2,figsize=(12,8),squeeze=False)
        panels=[(i,j,c,'all',1000 if i==0 else 100) for i in range(2) for j,c in enumerate(WINDOWS)]
    colors={'human_raw':'#343B43','original':'#BD692D','rules':'#2676A8'}
    names={'human_raw':'Human raw','original':'Contemporary original','rules':'Economic definitions'}
    styles={'human_raw':'-','original':'--','rules':'-.'}
    for i,j,c,s,limit in panels:
        ax=axes[i,j];g=data['groups'][c][s]
        for name in ('human_raw',*ARMS):
            points=g['common_distributions'][name]['cdf']
            if points:ax.step([0]+[p['value'] for p in points]+[1000], [0]+[p['cdf'] for p in points]+[1],where='post',label=names[name],color=colors[name],linestyle=styles[name],linewidth=2)
        who={'all':'All','has_sell_history':'Has SELL history','no_sell_history':'No SELL history (one case)'}[s]
        ax.set_title(f"{'3' if c=='short3' else '10'} history rounds | {who} | {g['common_people']} people/{g['common_paths']} pairs",fontsize=10.5,loc='left')
        ax.set_xlim(0,limit);ax.set_ylim(0,1.03)
        ax.grid(axis='y',color='#DEE2E6',linewidth=.7)
        ax.spines[['top','right']].set_visible(False)
        if j==0:ax.set_ylabel('Cumulative person-balanced mass')
        if i==rows-1:ax.set_xlabel('Submitted limit price (dollars/unit)')
    h,l=axes[0,0].get_legend_handles_labels()
    fig.legend(h,l,loc='upper center',bbox_to_anchor=(.5,.945),ncol=3,frameon=False)
    fig.suptitle('SELL-price distributions: original versus neutral economic definitions',fontsize=14,y=.985)
    note=('Private strata; one-person group is descriptive only. All panels show $0-$100 detail with full CDF denominators.' if private else 'Top: full $0-$1000 domain. Bottom: $0-$100 detail, same CDF denominators. Public output omits singleton/complement.')
    fig.text(.07,.025,note+'\nSame old quantity, history and paired draw; top-3 + 3%. No quote is never a zero price.',fontsize=9)
    fig.subplots_adjust(left=.075,right=.98,top=.86,bottom=.13,hspace=.32,wspace=.16)
    fig.savefig(base.with_suffix('.png'),dpi=180,facecolor='white');plt.close(fig)


if __name__=='__main__':
    render(json.loads(PUBLIC.with_suffix('.json').read_text(encoding='utf-8')))
    delivered={'public_report':str(PUBLIC.with_suffix('.md'))}
    if (PRIVATE/'analysis-private.json').is_file():
        render(json.loads((PRIVATE/'analysis-private.json').read_text(encoding='utf-8')),private=True)
        delivered['private_report']=str(PRIVATE/'comparison-private.md')
    print(json.dumps(delivered))

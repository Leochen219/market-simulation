"""Render only the strictly replayed safe aggregate; never read participant data."""
import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'docs/analysis/jev-flat-grid5-comparison-2026-10-27'
LABEL = {'short3': '三轮历史', 'long10': '十轮历史'}
METHODS = ('hierarchical_2_5', 'flat_5')


def fmt(x):
    return '不可用' if x is None else str(Decimal(str(x)).quantize(Decimal('.01'), rounding=ROUND_HALF_UP))


def pct(x):
    return '不可用' if x is None else f'{100*x:.1f}%'


def render(data):
    if data['status']['status'] != 'COMPLETE':
        raise ValueError('Refuse to render partial experiment as complete')
    groups = data['groups']
    totals = {m: {e: sum(g['methods'][m]['events'][e]['events']
                         for gs in groups.values() for g in gs.values())
                  for e in ('ge700', 'gt250')} for m in METHODS}
    lines = [
        '# 5元直接网格：匹配数量的真实Jev价格对照', '',
        '同18名开发参与者，预测正式第11轮（游戏第16轮）。短历史为正式8–10轮，长历史为正式1–10轮；每人每条件两次重复，共72条路径。', '',
        '**本次只重新请求价格。** 每条路径的真人当前状态、历史、给定方向以及旧3%流程已选数量保持不变。新接口一次选择5、10、…、995这199个完整价格；旧接口先选价带、再从价带选2.5元网格价格。两组都使用top-3＋原始概率至少3%的截断采样。', '',
        '保留原观察者预测提示，仅修改末尾价格选择说明及网格文字。没有重跑数量、方向、STOP、多订单或20轮市场。网格精度和分层生成接口同时改变，不能把结果唯一归因于5元间距。', '',
        '## 本次发现', '',
        f"四个分组的价格分布W1均明显下降；≥700元实际报价从{totals[METHODS[0]]['ge700']}/72降至{totals[METHODS[1]]['ge700']}/72，>250元从{totals[METHODS[0]]['gt250']}/72降至{totals[METHODS[1]]['gt250']}/72。新接口72/72都给出价格，改善并非通过增加弃权实现。",
        '', '这表明，在相同数量和可见历史下，替换价格生成方式能大幅改变Jev的拟真人表现；不能把旧高价直接解释为模型普遍不能处理数字。旧新调用发生在不同时间，且生成接口、网格及必要措辞共同变化；尚未唯一定位因果机制。', '',
        '卖价仍偏高：共同成功口径下，三轮/十轮新卖价均值分别33.13/29.38元，真人8.75元；新卖价P95均为90元，真人20元。投影到5元后的卖价W1仍为23.13/19.38元，因此余下差距主要不是几元舍入可以解释。买价长历史均价18.00元接近真人19.75元，但整体接近不证明每位参与者均被准确预测，也未检验完整条件分布校准。', '',
        '## 价格分布与覆盖', '',
        'W1为按人等权的合并价格分布与同一批真人原始报价之间的Wasserstein距离，单位为元，越小越接近；不是个人逐笔命中或平均个人误差。真人一人多单时，该人内部订单等权；模型两次重复在共同成功路径中等权。每行旧、新使用完全相同的人和成功路径。', '',
        '| 历史 / 方向 | 共同人数 / 路径 | 旧有效 / 计划 | 新有效 / 计划 | 旧W1 | 新W1 | 新−旧 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for cond in ('short3', 'long10'):
        for side in ('BUY', 'SELL'):
            g = groups[cond][side]
            a, b = (g['methods'][m] for m in METHODS)
            n = g['planned_paths_per_method']
            lines.append(f"| {LABEL[cond]} / {side} | {g['common_people']} / {g['common_paths']} | {a['quoted']}/{n} | {b['quoted']}/{n} | {fmt(a['raw_human_w1'])} | {fmt(b['raw_human_w1'])} | {fmt(g['flat_minus_hierarchical_raw_w1'])} |")
    lines += ['', '有效报价率须与分布距离一起阅读。所有候选低于3%时是客户端数值弃权，不是0元、真人STOP或技术失败；不把它补零，也不通过删除弃权宣称完整行为改善。', '',
              '| 历史 / 方向 | 真人均价 | 旧均价 / 中位数 / P95 | 新均价 / 中位数 / P95 | 旧 / 新：对5元投影真人W1 |',
              '|---|---:|---:|---:|---:|']
    for cond in ('short3', 'long10'):
        for side in ('BUY', 'SELL'):
            g = groups[cond][side]
            d = g['common_distributions']
            stat = lambda m: ' / '.join(fmt(d[m][k]) for k in ('mean', 'median', 'p95'))
            a, b = (g['methods'][m] for m in METHODS)
            lines.append(f"| {LABEL[cond]} / {side} | {fmt(d['human_raw']['mean'])} | {stat(METHODS[0])} | {stat(METHODS[1])} | {fmt(a['common_grid5_human_w1'])} / {fmt(b['common_grid5_human_w1'])} |")
    lines += ['', '5元投影仅检验网格表示误差，保留完整的原始真人报价作为主要参考；不是观察到真人在新网格下的行为。', '',
              '## 高价尾部', '',
              '以下计数使用每种方法的全部计划路径，不限共同成功子集。≥700包括800元以上；“700多”单独定义为700≤价格<800，800元不属于该窄区间。事件数/计划数不把弃权当低价；另报有效报价中的比例。', '',
              '| 历史 / 方向 / 接口 | ≥700：次数/计划（有效报价内） | 700–<800：次数/计划 | >250：次数/计划 | 高于当轮p：次数/计划 | 弃权 |',
              '|---|---:|---:|---:|---:|---:|']
    for cond in ('short3', 'long10'):
        for side in ('BUY', 'SELL'):
            g = groups[cond][side]
            for m, label in zip(METHODS, ('旧分区间', '新直接5元')):
                a = g['methods'][m]
                es = a['events']
                count = lambda name: f"{es[name]['events']}/{es[name]['planned']}"
                lines.append(f"| {LABEL[cond]} / {side} / {label} | {count('ge700')}（{pct(es['ge700']['among_quotes'])}） | {count('700_to_800')} | {count('gt250')} | {count('above_retail')} | {a['no_quote']} |")
    lines += ['', '完整JSON另保留弃权所对应假设未报价格的敏感性界；该界不是置信区间，也不声称弃权实际有一个隐藏报价。', '',
              '| 历史 / 方向 | 旧未截断模型分数≥700质量 | 新未截断模型分数≥700质量 |',
              '|---|---:|---:|']
    for cond in ('short3', 'long10'):
        for side in ('BUY', 'SELL'):
            g = groups[cond][side]
            a, b = (g['methods'][m] for m in METHODS)
            lines.append(f"| {LABEL[cond]} / {side} | {pct(a['raw_model_score_mass_ge700'])} | {pct(b['raw_model_score_mass_ge700'])} |")
    lines += ['', '旧分数质量按价带概率×各价带条件价格概率求和，新分数直接来自199项；均条件于相同旧数量、位于截断前。两者不是已经校准的真人行为概率，不能将分数变化解释成已证实的心理机制。', '',
              '事后阈值核对（另存tail-audit JSON）：本批所有≥700元候选的单项原始分数最高2%，均低于3%门槛，所以在本批截断支持里没有高价候选。高价合计分数仍有约2.6%–4.9%；0次高价不能改写为模型高价概率为0。实际新报价范围10–95元，provider choice非最大项及域外异常均为0。该补充是离线诊断，没有新增调用。', '',
              '![按人等权的价格累计分布](jev-flat-grid5-comparison-2026-10-27.png)', '',
              '![0–100元范围细节，累计质量分母不变](jev-flat-grid5-comparison-2026-10-27-zoom.png)', '',
              '## 执行与可复现性', '']
    ac = data['accounting']
    lines += [f"- 实际请求{ac['requests']}次，输入{ac['input_tokens']:,} tokens，按冻结价表核算USD {ac['cost_nanodollars']/1e9:.6f}，硬上限USD 0.25。",
              f"- 10路并发上限；按真实HTTP时间重建峰值{ac['observed_peak_inflight']}路，耗时{ac['elapsed_seconds']:.2f}秒；最大滚动一秒{ac['max_rolling_second_requests']}请求、{ac['max_rolling_reserved_bytes']:,}预留字节。",
              '- jev-1.14.0；原始响应、候选顺序、截断支持、随机数、选择、请求/响应哈希及预算均严格回放。无自动重试或补跑。',
              '- 离线定向回归47项通过，另经独立本地审查。计划和源代码冻结后执行；本次未新增Opus咨询。',
              f"- 冻结计划SHA256：`{data['plan_sha256']}`。",
              '- 逐人记录、完整状态、API请求和密钥只留本机；此报告与JSON仅为安全汇总。', '',
              '## 解释边界与下一步', '',
              '这是同市场开发样本上的条件选价诊断，两个重复不增加独立真人数，也不足以验证概率校准。降低高价尾部不等于整组订单像真人；接近总体分布也不意味着逐个人预测准确。覆盖下降、低价集中或条件分布丢失都须与W1一起判断。', '',
              '建议将直接5元网格保留为新的开发候选，旧版冻结不变。下一步若要识别原因，应在相同5元叶节点和措辞下对比分层/直接生成；若优先改善拟真人效果，应先离线检查剩余65–90元SELL报价及历史低价信息的使用。两种目的分开，尚未执行新增对照或自动修改阈值。完整多单和闭环市场需另行冻结验证。', '',
              '本地只读复核（已配置项目环境后，在仓库根目录）：', '',
              '```powershell', 'python -m tools.probes.jev_flat_grid5_study replay',
              'python -m tools.probes.jev_flat_grid5_study report',
              'python -m tools.analysis.report_jev_flat_grid5', '```', '',
              '`run`具有一次性启动标记，不可用于自动重跑；私有源数据与既有冻结证据不随GitHub分发。']
    BASE.with_suffix('.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def chart(data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), sharex=True, sharey=True)
    colors = {'human_raw': '#343B43', 'hierarchical_2_5': '#BD692D', 'flat_5': '#2676A8'}
    names = {'human_raw': 'Human (raw)', 'hierarchical_2_5': 'Old: bands, $2.50 grid', 'flat_5': 'New: direct $5 grid'}
    styles = {'human_raw': '-', 'hierarchical_2_5': '--', 'flat_5': '-.'}
    for row, cond in enumerate(('short3', 'long10')):
        for col, side in enumerate(('BUY', 'SELL')):
            ax = axes[row, col]
            g = data['groups'][cond][side]
            for m in ('human_raw', *METHODS):
                c = g['common_distributions'][m]['cdf']
                if c:
                    ax.step([0]+[v['value'] for v in c]+[1000], [0]+[v['cdf'] for v in c]+[1], where='post', color=colors[m], linestyle=styles[m], linewidth=2, label=names[m])
            ax.axvline(700, color='#8B9298', linewidth=.9, linestyle=':')
            old, new = (g['methods'][m] for m in METHODS)
            ax.set_title(f"{side} | {'3' if row == 0 else '10'} history rounds | shared {g['common_people']} people / {g['common_paths']} paths", fontsize=11, loc='left')
            ax.text(.98, .08, f"Quoted: old {old['quoted']}/{g['planned_paths_per_method']}; new {new['quoted']}/{g['planned_paths_per_method']}", transform=ax.transAxes, ha='right', fontsize=9)
            ax.set_xlim(0, 1000); ax.set_ylim(0, 1.03)
            ax.set_xticks([0, 250, 500, 700, 1000])
            ax.grid(axis='y', color='#DEE2E6', linewidth=.7)
            ax.spines[['top', 'right']].set_visible(False)
            if col == 0: ax.set_ylabel('Cumulative person-balanced mass')
            if row == 1: ax.set_xlabel('Limit price (dollars per unit)')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, bbox_to_anchor=(.5, .945), frameon=False)
    fig.suptitle('Matched conditional price distributions: hierarchical vs direct grid', fontsize=15, y=.985)
    fig.text(.07, .025, 'Same old-policy quantity and direction; top-3 + 3% cutoff. Shared quoted paths only; no quote is never price zero.\n18 development people, one market, two repeats; descriptive comparison, not calibrated probabilities.', fontsize=9)
    fig.subplots_adjust(top=.86, bottom=.14, left=.07, right=.98, hspace=.26, wspace=.12)
    fig.savefig(BASE.with_suffix('.png'), dpi=180, facecolor='white')
    for ax in axes.flat:
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 20, 40, 60, 80, 100])
    fig.suptitle('Matched price distributions: $0–$100 detail (same CDF denominator)', fontsize=15, y=.985)
    fig.savefig(BASE.parent / (BASE.name + '-zoom.png'), dpi=180, facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    result = json.loads(BASE.with_suffix('.json').read_text(encoding='utf-8'))
    render(result)
    chart(result)
    print(json.dumps({'report': str(BASE.with_suffix('.md')), 'chart': str(BASE.with_suffix('.png'))}))

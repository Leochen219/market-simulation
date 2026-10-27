"""Prepare bounded reviewed chart/receipt inputs, without model calls."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "analysis"


def write(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    data = json.loads((OUT / "pilot-behavior-data.json").read_text(encoding="utf-8"))
    raw = ROOT / "artifacts/runs/paper-ga-laya-bounded-24/economic.jsonl"
    events = [json.loads(x) for x in raw.read_text(encoding="utf-8").splitlines()]
    rows = []
    for event, summary in zip(events, data["round_trends"], strict=True):
        before = sum(max(event["exogenous"][a]["D"] - q, 0) for a, q in event["procurement"].items())
        after = sum(event["settlement"]["shortages"].values())
        assert (before, after) == (summary["shortage_before"], summary["shortage_after"])
        for stage, value in (("交易前", before), ("交易后", after)):
            rows.append({"round": str(event["round_index"] + 1), "stage": stage, "shortage": value})
    source = {
        "label": "本地 Laya GA-w10 模拟日志",
        "files": [{"label": "economic.jsonl"}, {"label": "pilot-behavior-data.json"}],
        "metricDefinitions": [{"label": "每轮总缺货", "definition": "交易前缺货按每位 agent 的 max(D−Q,0) 求和；交易后缺货按 max(D−Q−A,0) 求和。"}],
        "filters": ["10 个 AI agent", "批发价 $10", "展示第 1–20 轮；第 1–5 轮为热身"],
        "caveats": ["来自单个共享模型、相互作用的市场轨迹；不是 20 个独立实验。"],
        "evidenceFlow": [{"kind": "validation", "title": "数值核对", "detail": "原始日志逐轮独立计算与分析汇总一致；完整场次通过无模型回放。"}],
    }
    write("shortage-chart-input.json", {
        "schemaVersion": 1, "id": "pilot-shortage-before-after",
        "title": "总缺货：交易前与交易后（第 1–5 轮为热身）",
        "chart": {"type": "line", "x": "round", "y": "shortage", "series": "stage",
                  "xLabel": "轮次", "yLabel": "每轮缺货总单位（10 agents）"},
        "rows": rows, "source": source,
    })
    write("shortage-sources-input.json", {"schemaVersion": 1, "items": [{
        "id": "pilot-shortage-before-after", "title": "交易前后缺货的逐轮变化",
        "queries": [{"id": "round-shortage", "source": source, "rows": rows,
                     "columns": ["round", "stage", "shortage"], "reportingPeriod": "模拟第 1–20 轮",
                     "methods": [{"language": "calculation", "code": "before = sum(max(D_i-Q_i, 0))\nafter = sum(max(D_i-Q_i-A_i, 0))"}]}],
    }]})
    paper = json.loads((OUT / "paper-comparison-facts.json").read_text(encoding="utf-8"))
    comparisons = [
        {"metric": "平均采购单位", "local": data["formal"]["Q"]["mean"], "paper": paper["mean_procurement_units"]["value"]},
        {"metric": "期末剩余单位/人轮", "local": data["formal"]["post_market_leftover_units"]/150, "paper": paper["mean_end_leftover_units_per_player_round"]["value"]},
        {"metric": "期末缺货单位/人轮", "local": data["formal"]["post_market_shortage_units"]/150, "paper": paper["mean_end_shortage_units_per_player_round"]["value"]},
        {"metric": "成交价均值/美元", "local": data["execution"]["MCP_dollars"]["mean"], "paper": paper["mean_clearing_price_dollars"]["value"]},
    ]
    write("analysis-sources-input.json", {"schemaVersion": 1, "items": [{
        "id": "paper-comparison", "title": "本地正式轮次与论文 GA-10 对照",
        "queries": [{"id": "comparison", "source": {
            "label": "本地试验与论文结果表",
            "files": [{"label": "pilot-behavior-data.json"}, {"label": "paper-comparison-facts.json"}],
            "links": [{"label": "Wendt 等，Management Science", "href": paper["source"]}],
            "metricDefinitions": [{"label": "GA-10 对照", "definition": "采购、剩余和缺货是每人每正式轮均值；价格是每市场每正式轮均值。论文后缀 10 表示批发价。"}],
            "caveats": ["本地 10 个 AI、150 个正式记录；论文 GA-10 处理样本 35 位人类、525 个正式记录。这是描述性参照，不是匹配的因果对照。", "论文表 4 和 D.3 未明确 ddof，本地报告同时列出总体和样本标准差口径。"],
        }, "columns": ["metric", "local", "paper"], "rows": comparisons,
           "reportingPeriod": "本地第 6–20 轮与论文正式轮次"}],
    }, {
        "id": "economic-behavior", "title": "收益、交易合理性与数字模式",
        "queries": [{"id": "profit-check", "source": {
            "label": "本地市场与同采购量不交易计算",
            "files": [{"label": "economic.jsonl"}, {"label": "audit.jsonl"}, {"label": "analyze_pilot.py"}],
            "metricDefinitions": [{"label": "同 Q 收益比较", "definition": "固定每位 agent 的实际采购量、需求和零售价，比较实际市场与全部不下交易单；市场现金抵消，供应链收益等于客户销售收入。"}],
            "filters": ["正式第 6–20 轮", "10 个 AI agent", "批发价 $10"],
            "caveats": ["全部不下单的条件计算不模拟采购策略改变；单方 STOP 后悔值之和不等于整体收益损失。", "333 模式支持接口或历史反馈待检验假说，不能确定其因果来源。"],
            "evidenceFlow": [{"kind": "validation", "title": "独立核查", "detail": "原始日志独立核对了收益恒等式、40 个净买入记录的 STOP 收益、数字模式和自成交；修正了自成交名义额与真实现金转移的标签。"}],
        }, "reportingPeriod": "模拟第 6–20 轮", "columns": ["scenario", "retailer_profit_dollars", "supply_chain_dollars"],
        "rows": [{"scenario": "同 Q 全部不下单", "retailer_profit_dollars": data["no_trade_counterfactual_at_realized_Q"]["sum_profit_cents"]/100, "supply_chain_dollars": data["fixed_total_Q_allocation_efficiency"]["no_trade_sales_revenue_cents"]/100},
                 {"scenario": "实际 Laya 市场", "retailer_profit_dollars": data["accounting_scope"]["retailer_profit_cents"]/100, "supply_chain_dollars": data["accounting_scope"]["combined_chain_profit_cents"]/100}],
        "methods": [{"language": "calculation", "code": "retailer_profit = sum(-10*Q_i - m*A_i + p_i*min(D_i,Q_i+A_i))\nsupplier_receipts = 10*sum(Q_i)\nsupply_chain = retailer_profit + supplier_receipts"}]}],
    }]})


if __name__ == "__main__":
    main()

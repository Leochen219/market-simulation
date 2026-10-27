"""Paired descriptive analysis of a complete, frozen Jev input ablation."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


def paired_effect(rows, baseline, treatment, metric):
    indexed = {baseline: {}, treatment: {}}
    for row in rows:
        if row["condition"] not in indexed:
            continue
        block = (row["case_id"], row["order"], row["repetition"])
        group = indexed[row["condition"]]
        if block in group:
            raise ValueError("duplicate paired block")
        group[block] = float(row[metric])
    if not indexed[baseline] or indexed[baseline].keys() != indexed[treatment].keys():
        raise ValueError("paired coverage differs")
    deltas = [indexed[treatment][key] - value for key, value in indexed[baseline].items()]
    return {"baseline": baseline, "treatment": treatment, "metric": metric, "n_pairs": len(deltas),
            "mean_delta": sum(deltas) / len(deltas), "minimum": min(deltas), "maximum": max(deltas),
            "positive": sum(d > 1e-12 for d in deltas), "negative": sum(d < -1e-12 for d in deltas),
            "unchanged": sum(abs(d) <= 1e-12 for d in deltas)}


def repeat_variation(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["condition"], row["case_id"], row["order"]].append(row)
    return [{"condition": key[0], "case_id": key[1], "order": key[2], "n": len(group),
             "probability_vectors": len({tuple(sorted(r["reply"]["probabilities"].items())) for r in group}),
             "top_sets": len({tuple(sorted(r["top_candidate_ids"])) for r in group})}
            for key, group in sorted(grouped.items())]


def analyze(directory):
    result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
    plan = json.loads((directory / "frozen-plan.json").read_text(encoding="utf-8"))
    rows = json.loads((directory / "raw-evidence.json").read_text(encoding="utf-8"))
    if result["status"] != "COMPLETE" or len(rows) != 396:
        raise ValueError("complete 396-row evidence required")
    planned = {f'r{r["request"]["round_index"]:03d}:{r["request"]["stage"]}:{r["request"]["agent_id"]}': r
               for wave in plan["waves"] for r in wave}
    seen = set()
    for row in rows:
        identity = row["reply"]["request_id"]
        if identity in seen or identity not in planned:
            raise ValueError("unexpected or repeated request identity")
        seen.add(identity)
        source = planned[identity]
        if row["request"] != source["request"] or any(row[k] != source[k] for k in (
                "condition", "case_id", "repetition", "order", "expected", "cohort")):
            raise ValueError("raw request or metadata differs from frozen plan")
        probabilities = row["reply"]["probabilities"]
        by_action = {c["value"]: probabilities[c["candidate_id"]] for c in row["request"]["candidates"]}
        expected_probability = by_action[row["expected"]]
        row["correct_margin"] = expected_probability - max(v for k, v in by_action.items() if k != row["expected"])
        if row["correct"] != (row["correct_margin"] > 0):
            raise ValueError("saved strict correctness differs from probability ordering")
        row["sell_probability"] = by_action.get("SELL", 0.0)
    if seen != planned.keys():
        raise ValueError("missing planned requests")
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["condition"], row["case_id"]].append(row)
    cells = [{"condition": condition, "case_id": case, "cohort": group[0]["cohort"],
              "expected": group[0]["expected"], "n": len(group),
              "correct": sum(r["correct"] for r in group),
              "selected": dict(Counter(r["selected"] for r in group)),
              "mean_sell_probability": sum(r["sell_probability"] for r in group) / len(group),
              "mean_correct_margin": sum(r["correct_margin"] for r in group) / len(group)}
             for (condition, case), group in sorted(grouped.items())]
    links = []
    for mcp in ("none", "10", "333", "777"):
        links.append((f"market_0_history_{mcp}", f"market_1_history_{mcp}"))
    for market in (0, 1):
        for mcp in ("10", "333", "777"):
            links.append((f"market_{market}_history_none", f"market_{market}_history_{mcp}"))
        links.extend((f"market_{market}_history_10", f"market_{market}_history_{mcp}") for mcp in ("333", "777"))
    links.extend(("clean_labels_bare", treatment) for treatment in (
        "clean_labels_sell_cash", "clean_labels_stop_cost", "market_0_history_none"))
    links.extend((label, "market_0_history_none") for label in ("clean_labels_sell_cash", "clean_labels_stop_cost"))
    effects = []
    for baseline, treatment in links:
        for cohort in ("all", "seen", "new"):
            subset = rows if cohort == "all" else [r for r in rows if r["cohort"] == cohort]
            for metric in ("correct", "sell_probability", "correct_margin"):
                effects.append({"cohort": cohort, **paired_effect(subset, baseline, treatment, metric)})
    return {"source": directory.as_posix(), "matrixhash": plan["matrixhash"],
            "orderedwirehash": plan["orderedwirehash"],
            "raw_evidence_sha256": hashlib.sha256((directory / "raw-evidence.json").read_bytes()).hexdigest(),
            "result_sha256": hashlib.sha256((directory / "result.json").read_bytes()).hexdigest(),
            "budget_usage": result["budget_usage"], "n": len(rows), "cells": cells,
            "paired_effects": effects, "repeat_variation": repeat_variation(rows),
            "interpretation": "Paired descriptive differences over six fixtures; repetitions are not independent economic scenarios. No trading readiness claim."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()

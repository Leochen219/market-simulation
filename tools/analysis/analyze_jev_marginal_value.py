"""Audit frozen marginal-value evidence; label post-data hypotheses explicitly."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from capclaims.engine.protocol import digest_json


def average_value_action(row, weak=False):
    """Exploratory AFTER-data rule; not an oracle and never sent to Jev."""
    left = row["m"] * row["Q"]
    right = row["p"] * min(row["D"], row["Q"])
    return "SELL" if (left >= right if weak else left > right) else "STOP"


def analyze(directory):
    plan = json.loads((directory / "frozen-plan.json").read_text(encoding="utf-8"))
    result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
    raw_path = directory / "raw-evidence.json"
    rows = json.loads(raw_path.read_text(encoding="utf-8"))
    if result["status"] != "COMPLETE" or len(rows) != 456:
        raise ValueError("complete 456-call evidence required")
    planned = {f'r{r["request"]["round_index"]:03d}:{r["request"]["stage"]}:{r["request"]["agent_id"]}': r
               for wave in plan["waves"] for r in wave}
    seen = set()
    grouped = defaultdict(list)
    for row in rows:
        identity = row["reply"]["request_id"]
        if identity in seen or identity not in planned:
            raise ValueError("unexpected or duplicated reply identity")
        seen.add(identity)
        frozen = planned[identity]
        if row["request"] != frozen["request"]:
            raise ValueError("actual request differs from frozen plan")
        for key in ("case_id", "Q", "D", "p", "m", "w", "x", "order", "repetition"):
            if row[key] != frozen[key]:
                raise ValueError(f"metadata differs: {key}")
        # Independent lost-sales expression, rather than the runner's min-difference formula.
        delta = row["m"] * row["x"] - row["p"] * min(row["x"], max(0, row["D"] - row["Q"] + row["x"]))
        expected = ["SELL"] if delta > 0 else ["STOP"] if delta < 0 else ["STOP", "SELL"]
        if row["delta_profit"] != delta or row["expected"] != expected:
            raise ValueError("external oracle differs")
        if digest_json(row["request"]["observation"]) != row["reply"]["observation_hash"]:
            raise ValueError("observation hash differs")
        probs = row["reply"]["probabilities"]
        by_action = {c["value"]: probs[c["candidate_id"]] for c in row["request"]["candidates"]}
        correct = None if delta == 0 else by_action[expected[0]] > by_action["STOP" if expected[0] == "SELL" else "SELL"]
        if row["correct"] != correct or row["sell_probability"] != by_action["SELL"]:
            raise ValueError("saved strict score differs")
        grouped[row["case_id"]].append(row)
    if seen != planned.keys() or any(len(group) != 6 for group in grouped.values()):
        raise ValueError("scenario coverage differs")
    unique = [r for r in rows if not r["economic_tie"]]
    if len(unique) != 450:
        raise ValueError("economic tie denominator differs")
    exploratory = {}
    for weak in (False, True):
        key = "average_value_ge" if weak else "average_value_gt"
        mismatches = [r for r in unique if r["selected"] != average_value_action(r, weak)]
        exploratory[key] = {"n": len(unique), "agreements": len(unique) - len(mismatches),
            "status": "POST_HOC_EXPLORATORY_NOT_PREREGISTERED_OR_INDEPENDENTLY_VALIDATED",
            "counterexamples": [{k: r[k] for k in ("case_id", "Q", "D", "p", "m", "order", "repetition", "selected")}
                                for r in mismatches]}
    summary = result["summary"]
    if summary["unique_accuracy"]["n"] != len(unique) or summary["unique_accuracy"]["correct"] != sum(r["correct"] is True for r in unique):
        raise ValueError("headline accuracy differs from raw evidence")
    for name, weak in (("strict_m_gt_p", False), ("weak_m_ge_p", True)):
        predict = lambda r: "SELL" if (r["m"] >= r["p"] if weak else r["m"] > r["p"]) else "STOP"
        informative = [r for r in unique if r["expected"] != [predict(r)]]
        saved = summary["price_heuristics"][name]
        if (saved["n"] != len(unique) or saved["agreements"] != sum(r["selected"] == predict(r) for r in unique)
                or saved["discriminating_n"] != len(informative)
                or saved["discriminating_agreements"] != sum(r["selected"] == predict(r) for r in informative)):
            raise ValueError("predeclared heuristic summary differs from raw evidence")
    points = {p["case_id"]: dict(p) for p in summary["scenario_points"]}
    for case, group in grouped.items():
        points[case]["selected_counts"] = dict(Counter(r["selected"] for r in group))
    series = []
    for grid in summary["series"]:
        series.append({**{k: grid[k] for k in ("name", "axis")}, "points": [
            {**p, "n": len(grouped[p["case_id"]]), "selected_counts": points[p["case_id"]]["selected_counts"]}
            for p in grid["points"]]})
    return {"source": directory.as_posix(), "model": result["model"], "count": len(rows),
            "matrixhash": plan["matrixhash"], "orderedwirehash": plan["orderedwirehash"],
            "raw_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
            "result_sha256": hashlib.sha256((directory / "result.json").read_bytes()).hexdigest(),
            "instruction": plan["instruction"], "source_sha256": plan["source_sha256"],
            "budget_usage": result["budget_usage"], "unique_accuracy": summary["unique_accuracy"],
            "economic_ties": summary["economic_ties"], "predeclared_price_heuristics": summary["price_heuristics"],
            "exploratory_average_value_heuristics": exploratory, "series": series,
            "scenario_points": list(points.values()), "repeat_variation": summary["repeat_variation"],
            "interpretation": "Repeated black-box response experiment, no proof of model internals; reused series points are not independent cases. No20-round market."}


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

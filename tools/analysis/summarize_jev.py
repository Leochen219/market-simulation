"""Summarize saved Jev controls without exposing local credentials or raw requests."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def summarize(directory):
    path = directory / "result.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    keep = ("status", "suite", "instruction_profile", "model", "real_api_calls",
            "numeric_correct", "numeric_total", "economic_correct", "economic_total",
            "wire_plan_hash", "fixture_scoring_hash", "sdk_coroutine_parallelism",
            "http_transport_parallelism", "budget_usage", "probe_source_sha256")
    summary = {k: result[k] for k in keep}
    summary.update(directory=directory.as_posix(), result_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    summary["cases"] = {}
    for family in ("numeric", "economic"):
        rows = json.loads((directory / f"jev-{family}-raw.json").read_text(encoding="utf-8"))
        assert len(rows) == result[f"{family}_total"]
        assert sum(r["correct"] for r in rows) == result[f"{family}_correct"]
        summary[family + "_rounded_responses"] = sum(r["reply"]["usage"]["rounding_normalized"] for r in rows)
        summary[family + "_failures"] = [{k: r[k] for k in (
            "family", "case", "past", "field", "target", "suffix", "variant", "selected", "expected", "strict_ranking") if k in r}
            for r in rows if not r["correct"]]
        if family == "economic":
            for case in sorted({r["case"] for r in rows}):
                subset = [r for r in rows if r["case"] == case]
                summary["cases"][case] = {"correct": sum(r["correct"] for r in subset),
                    "total": len(subset), "expected": subset[0]["expected"],
                    "selected": dict(Counter(r["selected"] for r in subset))}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matched", type=Path, required=True)
    parser.add_argument("--explicit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {"matched": summarize(args.matched), "explicit": summarize(args.explicit),
              "confirmation_and_twenty_rounds": "NOT_RUN: both development gates failed"}
    assert result["matched"]["fixture_scoring_hash"] == result["explicit"]["fixture_scoring_hash"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()

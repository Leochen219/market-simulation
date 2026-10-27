"""Build a small, inspectable report from saved actual-model readiness evidence."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def summarize(directory: Path, interface: str) -> dict:
    result_path = directory / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    source = result["interfaces"][interface]
    summary = {"directory": directory.as_posix(), "interface": interface,
               "result_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
               "suite": result["suite"], "fixtures_hash": result["fixtures_hash"],
               "probe_source_sha256": result["probe_source_sha256"]}
    for key in ("status", "production_capability", "numeric_correct", "numeric_total",
                "economic_correct", "economic_total", "history_sensitivity"):
        summary[key] = source[key]
    summary["presentations"] = {}
    for family in ("numeric", "economic"):
        rows = json.loads((directory / f"{interface}-{family}-raw.json").read_text(encoding="utf-8"))
        assert len(rows) == source[f"{family}_total"]
        assert sum(r["correct"] for r in rows) == source[f"{family}_correct"]
        summary["presentations"][family] = {
            variant: {"correct": sum(r["correct"] for r in subset), "total": len(subset),
                      "selected": dict(Counter(str(r["selected"]) for r in subset))}
            for variant in sorted({r["variant"] for r in rows})
            if (subset := [r for r in rows if r["variant"] == variant])
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy", type=Path, required=True)
    parser.add_argument("--economic", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {"legacy": summarize(args.legacy, "legacy_v1"),
              "economic": summarize(args.economic, "economic_v2"),
              "twenty_round_rerun": "NOT_RUN: mandatory development readiness failed",
              "confirmation": "NOT_RUN: development gate failed",
              "interpretation": "ID remapping and order reversal are combined; their effects are not separately identified."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()

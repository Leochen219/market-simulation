"""Safe paired summaries for provider-choice continuation experiments."""
from __future__ import annotations

from statistics import mean, median
from typing import Any, Mapping, Sequence


def summarize_score_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize one-person-per-arm rows while preserving paired study contrasts."""
    by_arm: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        arm, person = row.get("arm"), row.get("person_id")
        if arm not in {"C_base", "C_joint", "E_base", "E_joint"} or not isinstance(person, str):
            raise ValueError("score row has unknown arm or participant key")
        if person in by_arm.setdefault(arm, {}):
            raise ValueError("duplicate person within study arm")
        by_arm[arm][person] = row
    if any(len(by_arm.get(arm, {})) != 6 for arm in ("C_base", "C_joint", "E_base", "E_joint")):
        raise ValueError("each study arm must include all six development people")
    output = {}
    for arm, people in by_arm.items():
        vals = [people[p]["primary_curve_error"] for p in sorted(people)]
        output[arm] = {"n": len(vals), "mean_primary_curve_error": mean(vals),
                       "median_primary_curve_error": median(vals)}
    paired = {}
    for view in ("C", "E"):
        b, j = by_arm[f"{view}_base"], by_arm[f"{view}_joint"]
        differences = [j[p]["primary_curve_error"] - b[p]["primary_curve_error"] for p in sorted(b)]
        paired[view] = {"n": len(differences), "mean_joint_minus_base": mean(differences),
                        "median_joint_minus_base": median(differences)}
    did = paired["C"]["mean_joint_minus_base"] - paired["E"]["mean_joint_minus_base"]
    return {"arms": output, "paired_joint_minus_base": paired,
            "difference_in_differences_C_minus_E": did,
            "interpretation": "descriptive development sample; not confirmatory evidence"}

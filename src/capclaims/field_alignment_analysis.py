"""Descriptive repeat-collapsed summaries for the field-name ablation."""
from __future__ import annotations
from statistics import mean, median
from typing import Any, Mapping, Sequence

ARMS = ("C_joint_control", "C_joint_aligned")
METRICS = (
    "primary_curve_error", "retail_window_1_100_curve_error", "first_order_only_curve_error",
    "duplicate_order_n", "duplicate_order_rate", "predicted_order_count", "human_order_count",
    "predicted_gross_buy", "human_gross_buy", "signed_buy_bias", "absolute_buy_error",
    "positive_excess_buy", "predicted_gross_sell", "human_gross_sell", "empty_agreement", "exact_order_multiset_match", "merged_same_side_price_match",
    "side_class_agreement", "five_order_rate",
)
BOOLEAN_METRICS = {"empty_agreement", "exact_order_multiset_match", "merged_same_side_price_match", "side_class_agreement"}


def summarize_repeated_pairs(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Average repeats within person/arm before paired summaries; keep detailed rows private."""
    keyed: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    for row in rows:
        person, arm, repeat = row.get("person_id"), row.get("arm"), row.get("repeat")
        if not isinstance(person, str) or arm not in ARMS or type(repeat) is not int or repeat not in (0, 1):
            raise ValueError("score row has invalid person, arm, or repeat")
        key = (person, arm, repeat)
        if key in keyed:
            raise ValueError("duplicate person/arm/repeat score")
        if type(row.get("primary_curve_error")) not in (int, float):
            raise ValueError("every score row needs a numeric primary error")
        keyed[key] = row
    people = sorted({key[0] for key in keyed})
    if len(people) != 6 or len(keyed) != 24 or any(
        (p, arm, repeat) not in keyed for p in people for arm in ARMS for repeat in (0, 1)):
        raise ValueError("complete six-person, two-arm, two-repeat coverage is required")

    collapsed: dict[tuple[str, str], dict[str, float]] = {}
    private_repeat_rows = []
    for person in people:
        for arm in ARMS:
            a, b = keyed[(person, arm, 0)], keyed[(person, arm, 1)]
            vals = {}
            for metric in METRICS:
                if metric in a and metric in b and all(
                    (type(x.get(metric)) in (int, float) or
                     (metric in BOOLEAN_METRICS and type(x.get(metric)) is bool)) for x in (a, b)):
                    vals[metric] = (float(a[metric]) + float(b[metric])) / 2
            collapsed[(person, arm)] = vals
            siga, sigb = a.get("order_multiset_signature"), b.get("order_multiset_signature")
            private_repeat_rows.append({"person_id": person, "arm": arm,
                "exact_multiset_match": siga == sigb if siga is not None and sigb is not None else None,
                "absolute_primary_error_difference": abs(float(a["primary_curve_error"])-float(b["primary_curve_error"]))})

    per_arm = {}
    paired = {}
    for arm in ARMS:
        per_arm[arm] = {metric: mean(collapsed[(p, arm)][metric] for p in people)
            for metric in METRICS if all(metric in collapsed[(p, arm)] for p in people)}
        per_arm[arm]["median_primary_curve_error"] = median(collapsed[(p, arm)]["primary_curve_error"] for p in people)
    for metric in METRICS:
        if all(metric in collapsed[(p, arm)] for p in people for arm in ARMS):
            deltas = [collapsed[(p, "C_joint_aligned")][metric]-collapsed[(p, "C_joint_control")][metric]
                for p in people]
            loo = [mean(delta for j, delta in enumerate(deltas) if j != i) for i in range(len(deltas))]
            paired[metric] = {"mean_aligned_minus_control": mean(deltas),
                "median_aligned_minus_control": median(deltas),
                "aligned_lower_n": sum(d < 0 for d in deltas), "equal_n": sum(d == 0 for d in deltas),
                "aligned_higher_n": sum(d > 0 for d in deltas),
                "leave_one_person_out_mean_min": min(loo), "leave_one_person_out_mean_max": max(loo)}
    exact = [r["exact_multiset_match"] for r in private_repeat_rows if r["exact_multiset_match"] is not None]
    return {"status": "DESCRIPTIVE_DEVELOPMENT_ONLY", "n_people": 6, "repeat_n_per_person_arm": 2,
        "per_arm": per_arm, "paired_aligned_minus_control": paired,
        "repeat_consistency": {"observed_pair_n": len(exact), "exact_multiset_match_n": sum(exact),
            "mean_absolute_primary_error_difference": mean(r["absolute_primary_error_difference"] for r in private_repeat_rows)},
        "private_person_scores": [{"person_id": p, "arm": arm, **collapsed[(p, arm)]}
            for p in people for arm in ARMS],
        "private_repeat_rows": private_repeat_rows,
        "interpretation": "six development people; repeats collapsed before paired summaries; no population inference"}

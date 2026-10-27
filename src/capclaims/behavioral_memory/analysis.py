"""Predeclared two-draw participant-level scores and scrubbed aggregate reporting."""
from __future__ import annotations

import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from capclaims.conditional_behavior.scoring import fair_brier, fair_crps, fair_energy_score, feature_vector

ARMS = ("M0", "M1", "M2")
PRIMARY = (("M1", "M0"), ("M2", "M1"))
SECONDARY = (("M2", "M0"),)


def _curve(orders: Sequence[Mapping[str, Any]]) -> tuple[int, ...]:
    buys = [sum(o["qty"] for o in orders if o["side"] == "buy" and o["price"] >= price)
            for price in range(1, 1000)]
    sells = [sum(o["qty"] for o in orders if o["side"] == "sell" and o["price"] <= price)
             for price in range(1, 1000)]
    return tuple(buys + sells)


def _energy(samples: Sequence[Sequence[float]], truth: Sequence[float]) -> float:
    n = len(samples)
    dist = lambda a, b: math.sqrt(math.fsum((x - y) ** 2 for x, y in zip(a, b)))
    first = math.fsum(dist(row, truth) for row in samples) / n
    pair = math.fsum(dist(samples[i], samples[j]) for i in range(n) for j in range(n) if i != j)
    return first - pair / (2 * n * (n - 1))


def _signature(orders: Sequence[Mapping[str, Any]]) -> tuple[tuple[str, int, int], ...]:
    return tuple(sorted((o["side"], o["qty"], o["price"]) for o in orders))


def _features(orders: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    buy = [o for o in orders if o["side"] == "buy"]
    sell = [o for o in orders if o["side"] == "sell"]
    def v(side):
        rows = buy if side == "buy" else sell
        q = sum(o["qty"] for o in rows)
        return {"n": len(rows), "q": q,
            "p": sum(o["qty"] * o["price"] for o in rows) / q if q else 0.0}
    b, s = v("buy"), v("sell")
    return {"order_count": len(orders), "buy_count": b["n"], "sell_count": s["n"],
        "buy_quantity": b["q"], "sell_quantity": s["q"],
        "buy_price": b["p"] if b["n"] else None, "sell_price": s["p"] if s["n"] else None,
        "has_buy": float(b["n"] > 0), "has_sell": float(s["n"] > 0),
        "empty": float(not orders)}


def _person_score(samples: list[dict[str, Any]], label: Mapping[str, Any]) -> dict[str, Any]:
    if len(samples) != 2 or sorted(s["repeat"] for s in samples) != [0, 1]:
        raise ValueError("each person and arm requires two distinct complete repeat blocks")
    samples.sort(key=lambda row: row["repeat"])
    portfolios = [s["orders"] for s in samples]
    truth = label["orders"]
    z = [feature_vector(orders) for orders in portfolios]
    truth_z = feature_vector(truth)
    curves = [_curve(orders) for orders in portfolios]
    curve_truth = _curve(truth)
    metrics = {
        "fair_energy_score": fair_energy_score(z, truth_z),
        "fair_energy_full_curve": _energy(curves, curve_truth),
        "fair_brier_buy": fair_brier([bool(_features(x)["has_buy"]) for x in portfolios],
                                     bool(_features(truth)["has_buy"])),
        "fair_brier_sell": fair_brier([bool(_features(x)["has_sell"]) for x in portfolios],
                                      bool(_features(truth)["has_sell"])),
        "fair_brier_empty": fair_brier([bool(_features(x)["empty"]) for x in portfolios],
                                       bool(_features(truth)["empty"])),
        "fair_crps_order_count": fair_crps([len(x) for x in portfolios], len(truth)),
        "fair_crps_buy_quantity": fair_crps([_features(x)["buy_quantity"] for x in portfolios],
                                            _features(truth)["buy_quantity"]),
        "fair_crps_sell_quantity": fair_crps([_features(x)["sell_quantity"] for x in portfolios],
                                             _features(truth)["sell_quantity"]),
    }
    actual = _features(truth)
    predicted = [_features(x) for x in portfolios]
    for name in ("order_count", "buy_quantity", "sell_quantity", "buy_price", "sell_price"):
        vals = [x[name] for x in predicted if x[name] is not None]
        errors = ([abs(x[name] - actual[name]) for x in predicted
                   if x[name] is not None and actual[name] is not None])
        metrics[f"mean_{name}"] = statistics.fmean(vals) if vals else None
        metrics[f"abs_error_{name}"] = statistics.fmean(errors) if errors else None
        metrics[f"sd_{name}"] = statistics.pstdev(vals) if vals else None
        if name.endswith("_price"):
            metrics[f"prediction_coverage_{name}"] = sum(x[name] is not None for x in predicted) / 2
            metrics[f"paired_price_error_coverage_{name}"] = len(errors) / 2
    duplicate_counts = [sum(n - 1 for n in Counter(
        (o["side"], o["qty"], o["price"]) for o in portfolio).values()) for portfolio in portfolios]
    metrics["mean_duplicate_order_count"] = statistics.fmean(duplicate_counts)
    metrics["any_duplicate_order_rate"] = sum(n > 0 for n in duplicate_counts) / 2
    provider_n = provider_present_n = provider_missing_n = sample_disagreement_n = decision_n = 0
    for sample in samples:
        for step in sample.get("trace", []):
            probabilities = step["raw_probabilities"]
            best_probability = max(probabilities.values())
            choice = step.get("provider_choice")
            if choice is None:
                provider_missing_n += 1
            else:
                provider_present_n += 1
                provider_n += int(choice not in probabilities or
                                  probabilities[choice] < best_probability - 1e-12)
                sample_disagreement_n += int(step["selected"] != choice)
            decision_n += 1
    metrics["provider_choice_present_n"] = provider_present_n
    metrics["provider_choice_missing_n"] = provider_missing_n
    metrics["provider_choice_missing_rate"] = provider_missing_n / decision_n if decision_n else None
    metrics["provider_choice_non_map_rate"] = provider_n / provider_present_n if provider_present_n else None
    metrics["sample_differs_from_provider_choice_rate"] = (
        sample_disagreement_n / provider_present_n if provider_present_n else None)
    metrics["exact_multiset_matches_n"] = sum(_signature(x) == _signature(truth) for x in portfolios)
    metrics["repeat_exact_multiset_match"] = _signature(portfolios[0]) == _signature(portfolios[1])
    metrics["five_order_n"] = sum(len(x) == 5 for x in portfolios)
    metrics["portfolios"] = portfolios
    return metrics


def score_trajectories(schedule: Sequence[Mapping[str, Any]], labels: Mapping[str, Mapping[str, Any]],
                       results: Sequence[Mapping[str, Any]], *, bootstrap_draws: int = 10_000,
                       bootstrap_seed: int = 20261026) -> dict[str, Any]:
    """Score fixed 108-trajectory denominator; incomplete runs yield no reduced-K score."""
    if bootstrap_draws != 10_000:
        raise ValueError("the frozen report uses exactly 10,000 person reweightings")
    planned = {(r["person_id"], r["repeat"], r["arm"]) for r in schedule}
    if len(schedule) != 108 or len(planned) != 108:
        raise ValueError("score requires 108 unique planned trajectory identities")
    valid_statuses = {"COMPLETE", "FAILED", "INCOMPLETE"}
    by_key: dict[tuple[str, int, str], Mapping[str, Any]] = {}
    result_status_counts = {status: 0 for status in (*sorted(valid_statuses), "ABSENT")}
    for row in results:
        try:
            key = (row["person_id"], row["repeat"], row["arm"])
            status = row["status"]
        except (KeyError, TypeError) as exc:
            raise ValueError("trajectory result has no complete identity or status") from exc
        if (not isinstance(key[0], str) or type(key[1]) is not int or key[1] not in (0, 1)
                or key[2] not in ARMS or not isinstance(status, str)
                or key not in planned or key in by_key or status not in valid_statuses):
            raise ValueError("duplicate or invalid trajectory evidence")
        by_key[key] = row
        result_status_counts[status] += 1
    absent = planned - set(by_key)
    result_status_counts["ABSENT"] = len(absent)
    complete = {key: row for key, row in by_key.items() if row["status"] == "COMPLETE"}
    if len(complete) != 108:
        return {"status": "INCOMPLETE", "planned_trajectories": 108,
            "completed_trajectories": len(complete),
            "missing_trajectory_n": 108 - len(complete),
            "failed_trajectory_n": result_status_counts["FAILED"],
            "incomplete_trajectory_n": result_status_counts["INCOMPLETE"],
            "absent_trajectory_n": result_status_counts["ABSENT"],
            "result_status_counts": result_status_counts,
            "scoring_note": "No reduced-K or complete-case conclusion is reported."}
    people = sorted(labels)
    if len(people) != 18 or any(sum(key[0] == person for key in planned) != 6 for person in people):
        raise ValueError("score requires exactly 18 people with six planned arm/repeat rows each")
    per_person: dict[str, dict[str, dict[str, Any]]] = {person: {} for person in people}
    for person in people:
        for arm in ARMS:
            samples = [complete[(person, repeat, arm)] for repeat in (0, 1)]
            per_person[person][arm] = _person_score(samples, labels[person])
    metric_names = tuple(k for k in per_person[people[0]]["M0"] if k not in {"portfolios"})
    mean_defined = lambda values: statistics.fmean(value for value in values if value is not None) \
        if any(value is not None for value in values) else None
    by_arm = {arm: {metric: mean_defined([per_person[p][arm][metric] for p in people])
                    for metric in metric_names} for arm in ARMS}
    contrasts = {}
    interval_rng = random.Random(bootstrap_seed)
    for left, right in (*PRIMARY, *SECONDARY):
        name = f"{left}-{right}"
        delta = {metric: [None if per_person[p][left][metric] is None or per_person[p][right][metric] is None
                          else per_person[p][left][metric] - per_person[p][right][metric] for p in people]
                 for metric in metric_names}
        summary = {metric: mean_defined(values) for metric, values in delta.items()}
        summary["paired_person_n_by_metric"] = {metric: sum(v is not None for v in values)
                                                 for metric, values in delta.items()}
        summary["per_person_deltas"] = {p: {metric: delta[metric][i] for metric in metric_names}
                                        for i, p in enumerate(people)}
        summary["improved_n_fair_energy"] = sum(value < 0 for value in delta["fair_energy_score"])
        summary["leave_one_person_out_fair_energy_range"] = [
            min(statistics.fmean(delta["fair_energy_score"][:i] + delta["fair_energy_score"][i + 1:])
                for i in range(18)),
            max(statistics.fmean(delta["fair_energy_score"][:i] + delta["fair_energy_score"][i + 1:])
                for i in range(18))]
        bootstrap = sorted(statistics.fmean(interval_rng.choice(delta["fair_energy_score"])
                                            for _ in range(18)) for _ in range(bootstrap_draws))
        summary["fixed_seed_person_reweighting_95_interval_fair_energy"] = [
            bootstrap[int(.025 * (bootstrap_draws - 1))], bootstrap[int(.975 * (bootstrap_draws - 1))]]
        contrasts[name] = summary
    return {"status": "COMPLETE", "planned_trajectories": 108, "completed_trajectories": 108,
        "person_n": 18, "per_arm": by_arm, "contrasts": contrasts,
        "per_person": per_person,
        "metric_definitions": {"fair_energy_score": "Two independent draws scored together on the frozen eight-dimensional feature vector; lower is better.",
            "fair_energy_full_curve": "Fair Euclidean energy score over 1..999 cumulative BUY/SELL curves.",
            "fair_brier_*": "Fair Brier score for direction presence or an empty portfolio.",
            "fair_crps_*": "Fair CRPS for order count and BUY/SELL quantity.",
            "primary_contrasts": ["M1-M0", "M2-M1"], "secondary_contrast": "M2-M0",
            "bootstrap": "10,000 fixed-seed resamples of the 18 participants; descriptive interval, not calibration evidence."}}


def public_report(score: Mapping[str, Any]) -> dict[str, Any]:
    """Scrub all participant identifiers and rows from the shareable aggregate report."""
    if score.get("status") != "COMPLETE":
        return {key: score[key] for key in ("status", "planned_trajectories", "completed_trajectories",
            "missing_trajectory_n", "failed_trajectory_n", "incomplete_trajectory_n",
            "absent_trajectory_n", "result_status_counts", "scoring_note") if key in score}
    contrasts = {}
    for name, value in score["contrasts"].items():
        contrasts[name] = {key: item for key, item in value.items() if key != "per_person_deltas"}
    return {"status": score["status"], "planned_trajectories": score["planned_trajectories"],
        "completed_trajectories": score["completed_trajectories"], "person_n": score["person_n"],
        "per_arm": score["per_arm"], "contrasts": contrasts,
        "metric_definitions": score["metric_definitions"]}


def write_report(run_dir: str | Path, score: Mapping[str, Any], output: str | Path | None = None) -> Path:
    run = Path(run_dir)
    destination = Path(output) if output else run / "report.json"
    if destination.exists():
        raise FileExistsError("aggregate report is immutable")
    safe = public_report(score)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(safe, ensure_ascii=False, sort_keys=True, indent=2,
                                      allow_nan=False) + "\n", encoding="utf-8")
    return destination


def chinese_report(score: Mapping[str, Any]) -> str:
    if score.get("status") != "COMPLETE":
        return (f"# 行为记忆研究报告\n\n状态：{score.get('status')}。计划轨迹108条，"
                f"完成{score.get('completed_trajectories', 0)}条。由于存在未完成轨迹，不报告降K或完整案例结论。")
    lines = ["# 行为记忆研究报告", "", "状态：完整。每位参与者的两次独立组合共同评分；参与者等权。",
        "主要比较：M1-M0、M2-M1；次要比较：M2-M0。公平能量分数越低越好。", ""]
    for name, result in score["contrasts"].items():
        interval = result["fixed_seed_person_reweighting_95_interval_fair_energy"]
        lines.append(f"- {name}：八维公平能量差 {result['fair_energy_score']:.6f}；"
                     f"18人中改善 {result['improved_n_fair_energy']} 人；"
                     f"描述性重加权95%区间 [{interval[0]:.6f}, {interval[1]:.6f}]。")
    lines.extend(["", "该区间为描述性参与者重加权区间；两次样本不代表校准结论。"])
    return "\n".join(lines)

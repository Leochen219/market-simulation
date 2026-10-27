"""Fixed participant-first scoring, baselines, sensitivity, and safe reporting."""
from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .baselines import baseline_distributions, expected_fair_score
from .runner import replay_run
from .scoring import (curve_distance, fair_brier, fair_crps, fair_energy_score,
                      feature_vector, incomplete_energy_interval, medoid_index)

FEATURE_NAMES = ("has_buy", "has_sell", "buy_count_5", "sell_count_5",
                 "buy_qty_200", "sell_qty_200", "buy_mean_price_100", "sell_mean_price_100")
FEATURE_MAXIMA = (1.0, 1.0, 1.0, 1.0, 24.975, 2.0, 9.99, 9.99)
METRIC_REGISTRY = (
    "fair_energy_score", "fair_crps_buy", "fair_crps_sell", "fair_crps_order_count",
    "fair_brier_empty", "fair_brier_buy", "fair_brier_sell", "fair_energy_full_curve",
    "fair_energy_price_window_1_100", "fair_energy_price_window_101_999",
    "price_shape_l1_mean", "price_shape_coverage",
    "mean_abs_order_count_error", "mean_abs_buy_quantity_error", "mean_abs_sell_quantity_error",
)


def _metric_interval(metric: str, q: int) -> list[float]:
    if metric == "fair_energy_score":
        return list(incomplete_energy_interval(q))
    if metric == "fair_energy_full_curve":
        return [0.0, math.sqrt(999 * 4995 ** 2 + 999 * q ** 2)]
    if metric == "fair_energy_price_window_1_100":
        return [0.0, math.sqrt(100 * 4995 ** 2 + 100 * q ** 2)]
    if metric == "fair_energy_price_window_101_999":
        return [0.0, math.sqrt(899 * 4995 ** 2 + 899 * q ** 2)]
    if metric in {"fair_crps_buy", "mean_abs_buy_quantity_error"}:
        return [0.0, 4995.0]
    if metric in {"fair_crps_sell", "mean_abs_sell_quantity_error"}:
        return [0.0, float(q)]
    if metric in {"fair_crps_order_count", "mean_abs_order_count_error"}:
        return [0.0, 5.0]
    if metric.startswith("fair_brier_") or metric == "price_shape_coverage":
        return [0.0, 1.0]
    if metric == "price_shape_l1_mean":
        return [0.0, 999 * (4995 + q) / 1998]
    raise ValueError(f"missing frozen score interval for {metric}")


def _difference_summary(bounds_by_person: dict[str, list[float]], *, seed: int) -> dict[str, Any]:
    """Summarize paired 17-person intervals without dropping incomplete participants."""
    if len(bounds_by_person) != 17:
        raise ValueError("paired study contrasts require all 17 planned participants")
    people = sorted(bounds_by_person)
    lower = [bounds_by_person[person][0] for person in people]
    upper = [bounds_by_person[person][1] for person in people]
    exact = all(lo == hi for lo, hi in bounds_by_person.values())
    point = [bounds_by_person[person][0] for person in people] if exact else None
    loo_bounds = []
    for omitted in range(17):
        loo_bounds.append([sum(value for index, value in enumerate(lower) if index != omitted) / 16,
                           sum(value for index, value in enumerate(upper) if index != omitted) / 16])
    rng = random.Random(seed)
    boot_lows, boot_highs = [], []
    for _ in range(10_000):
        picks = [rng.randrange(17) for _ in range(17)]
        boot_lows.append(sum(lower[index] for index in picks) / 17)
        boot_highs.append(sum(upper[index] for index in picks) / 17)
    boot_lows.sort()
    boot_highs.sort()
    return {"participant_differences": {person: bounds_by_person[person] for person in people},
        "mean_difference_interval": [sum(lower) / 17, sum(upper) / 17],
        "median_difference_interval": [sorted(lower)[8], sorted(upper)[8]],
        "leave_one_participant_out_intervals": loo_bounds,
        "leave_one_out_min_lower_bound": min(value[0] for value in loo_bounds),
        "leave_one_out_max_upper_bound": max(value[1] for value in loo_bounds),
        "participant_reweighting_95_interval": [boot_lows[249], boot_highs[9749]],
        "complete_case_mean": _average(point) if point is not None else None,
        "complete_case_median": sorted(point)[8] if point is not None else None,
        "complete_case_n": len(point) if point is not None else 0,
        "all_17_exact": exact}


def _feature_summary(context_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Participant weighted feature means and dispersion with conservative missing bounds."""
    by_person: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in context_rows:
        by_person[row["person_id"]].append(row)
    means = {}
    within_dispersion = {}
    between_dispersion = {}
    for index, name in enumerate(FEATURE_NAMES):
        participant_bounds = []
        within_bounds = []
        exact_values = []
        for person in sorted(by_person):
            rows = by_person[person]
            mean_rows = [row["sample_feature_means_interval"][name] for row in rows]
            participant_bounds.append([sum(value[0] for value in mean_rows) / len(mean_rows),
                                       sum(value[1] for value in mean_rows) / len(mean_rows)])
            sd_rows = [row["sample_feature_population_sd_interval"][name] for row in rows]
            within_bounds.append([sum(value[0] for value in sd_rows) / len(sd_rows),
                                  sum(value[1] for value in sd_rows) / len(sd_rows)])
            if participant_bounds[-1][0] == participant_bounds[-1][1]:
                exact_values.append(participant_bounds[-1][0])
        means[name] = {"interval": [sum(value[0] for value in participant_bounds) / len(participant_bounds),
                                    sum(value[1] for value in participant_bounds) / len(participant_bounds)]}
        within_dispersion[name] = {"interval": [
            sum(value[0] for value in within_bounds) / len(within_bounds),
            sum(value[1] for value in within_bounds) / len(within_bounds)]}
        if len(exact_values) == len(by_person):
            center = sum(exact_values) / len(exact_values)
            between_dispersion[name] = {"interval": [math.sqrt(sum((value-center)**2 for value in exact_values)
                / len(exact_values))] * 2}
        else:
            between_dispersion[name] = {"interval": [0.0, FEATURE_MAXIMA[index] / 2]}
    return {"unique_people_n": len(by_person), "target_context_n": len(context_rows),
        "complete_context_n": sum(row["observed_K"] == 4 for row in context_rows),
        "observed_sample_n": sum(row["observed_K"] for row in context_rows),
        "mean_feature_values": means, "within_forecast_population_sd": within_dispersion,
        "between_participant_sd": between_dispersion}


def _curves(orders):
    buy = tuple(sum(row["qty"] for row in orders if row["side"] == "buy" and row["price"] >= p)
                for p in range(1, 1000))
    sell = tuple(sum(row["qty"] for row in orders if row["side"] == "sell" and row["price"] <= p)
                 for p in range(1, 1000))
    return buy, sell


def _curve_l2(left, right) -> float:
    """Exact Euclidean distance between two 999-price step curves."""
    boundaries = {1, 1000}
    for portfolio in (left, right):
        for order in portfolio:
            price = order["price"]
            boundaries.add(price)
            boundaries.add(min(1000, price + 1))
    ordered = sorted(boundaries)
    squared = 0.0
    for lo, hi in zip(ordered, ordered[1:]):
        if lo >= hi:
            continue
        for side in ("buy", "sell"):
            if side == "buy":
                a = sum(order["qty"] for order in left if order["side"] == side and order["price"] >= lo)
                b = sum(order["qty"] for order in right if order["side"] == side and order["price"] >= lo)
            else:
                a = sum(order["qty"] for order in left if order["side"] == side and order["price"] <= lo)
                b = sum(order["qty"] for order in right if order["side"] == side and order["price"] <= lo)
            squared += (hi - lo) * (a - b) ** 2
    return math.sqrt(squared)


def _expected_energy_curve(records, truth) -> float:
    values = [row["orders"] for row in records]
    if not values:
        raise ValueError("curve baseline needs at least one record")
    first = sum(_curve_l2(row, truth) for row in values) / len(values)
    pair = sum(_curve_l2(left, right) for left in values for right in values) / (len(values) ** 2)
    return first - 0.5 * pair


def _expected_fair_scalar(records, truth, score) -> float:
    return expected_fair_score(records, truth, score,
        lambda left, right: abs(left - right))


def _expected_fair_brier(records, truth, predicate) -> float:
    values = [bool(predicate(row["orders"])) for row in records]
    p = sum(values) / len(values)
    return (p - int(bool(truth))) ** 2


def _weighted_wasserstein_1d(left_values, left_weights, right_values, right_weights) -> float:
    """Exact 1-D W1 between weighted empirical distributions on sorted support."""
    if len(left_values) != len(left_weights) or len(right_values) != len(right_weights):
        raise ValueError("support and weight vectors must have equal lengths")
    if not left_values or not right_values:
        raise ValueError("both empirical distributions need positive support")
    def normalize(values, weights):
        mass: dict[float, float] = defaultdict(float)
        for value, weight in zip(values, weights, strict=True):
            value, weight = float(value), float(weight)
            if not math.isfinite(value) or not math.isfinite(weight) or weight < 0:
                raise ValueError("empirical support and weights must be finite and nonnegative")
            mass[value] += weight
        total = sum(mass.values())
        if total <= 0:
            raise ValueError("empirical distribution must have positive mass")
        return {value: weight / total for value, weight in mass.items()}
    left, right = normalize(left_values, left_weights), normalize(right_values, right_weights)
    points = sorted(set(left) | set(right))
    cdf_left = cdf_right = distance = 0.0
    previous = points[0]
    for point in points:
        distance += abs(cdf_left - cdf_right) * (point - previous)
        cdf_left += left.get(point, 0.0)
        cdf_right += right.get(point, 0.0)
        previous = point
    return distance


def _side_price_w1(human_orders, predicted_orders, side: str) -> float | None:
    """Quantity-normalized price-distribution W1, or NA if either side is absent."""
    if side not in {"buy", "sell"}:
        raise ValueError("price W1 side must be BUY or SELL")
    human = [row for row in human_orders if row["side"] == side]
    predicted = [row for row in predicted_orders if row["side"] == side]
    if not human or not predicted:
        return None
    htotal, ptotal = sum(row["qty"] for row in human), sum(row["qty"] for row in predicted)
    return _weighted_wasserstein_1d([row["price"] for row in human],
        [row["qty"] / htotal for row in human], [row["price"] for row in predicted],
        [row["qty"] / ptotal for row in predicted])


def _portfolio_signature(orders):
    return tuple(sorted((row["side"], row["qty"], row["price"]) for row in orders))


def _portfolio_exactness(portfolios, truth) -> dict[str, Any]:
    truth_signature = _portfolio_signature(truth)
    signatures = [_portfolio_signature(orders) for orders in portfolios]
    repeat_counts = [len(orders) - len(set(_portfolio_signature(orders))) for orders in portfolios]
    return {"medoid_exact_match_human": False,
        "mean_random_draw_exact_match_human": sum(value == truth_signature for value in signatures) / len(signatures),
        "duplicate_forecast_sample_n": len(signatures) - len(set(signatures)),
        "distinct_forecast_portfolios_n": len(set(signatures)),
        "within_portfolio_repeated_order_counts": repeat_counts,
        "mean_within_portfolio_repeated_order_count": sum(repeat_counts) / len(repeat_counts),
        "within_portfolio_repeated_any_rate": sum(value > 0 for value in repeat_counts) / len(repeat_counts)}


def _weighted_moments(values, weights) -> dict[str, float]:
    total = sum(weights)
    if total <= 0:
        raise ValueError("weighted moments require positive mass")
    mean = sum(value * weight for value, weight in zip(values, weights, strict=True)) / total
    variance = sum(weight * (value - mean) ** 2 for value, weight in zip(values, weights, strict=True)) / total
    return {"mean": mean, "population_sd": math.sqrt(variance), "n": len(values),
        "unique_support_n": len(set(values)), "weight_sum": total}


def _pooled_raw_w1(contexts, labels, completed_samples, *, stratum: str | None = None) -> dict[str, Any]:
    """Compare pooled raw BUY/SELL totals with frozen participant-first weights."""
    selected = [key for key in contexts if stratum is None or contexts[key]["stratum"] == stratum]
    people = sorted({key[0] for key in selected})
    complete_n = sum(contexts[key]["K"] == 4 for key in selected)
    max_q = max((labels[f"{person}:{round_no}"]["Q"] for person, round_no, _ in selected), default=0)
    required_people = 5 if stratum is not None else 1
    if len(people) < required_people:
        return {"status": "INSUFFICIENT_UNIQUE_PEOPLE", "unique_people_n": len(people),
            "required_unique_people_n": required_people}
    metrics = {}
    for side in ("buy", "sell"):
        metric = {}
        if complete_n != len(selected):
            max_range = 4995 if side == "buy" else max_q
            metric = {"status": "INCOMPLETE", "w1": None, "w1_interval": [0.0, float(max_range)],
                "human": None, "model": None, "planned_context_n": len(selected),
                "complete_context_n": complete_n, "human_context_n": len(selected),
                "observed_model_draw_n": sum(contexts[key]["K"] for key in selected),
                "coverage": complete_n / len(selected)}
            metrics[side] = metric
            continue
        human_values, human_weights = [], []
        model_values, model_weights = [], []
        for person in people:
            person_keys = [key for key in selected if key[0] == person]
            context_weight = (1 / len(people)) / len(person_keys)
            for _, round_no, arm in person_keys:
                label = labels[f"{person}:{round_no}"]
                truth_total = sum(row["qty"] for row in label["orders"] if row["side"] == side)
                human_values.append(truth_total)
                human_weights.append(context_weight)
                for orders in completed_samples[(person, round_no, arm)]:
                    model_values.append(sum(row["qty"] for row in orders if row["side"] == side))
                    model_weights.append(context_weight / 4)
        metric = {"status": "SCORED",
            "w1": _weighted_wasserstein_1d(human_values, human_weights, model_values, model_weights),
            "human": _weighted_moments(human_values, human_weights),
            "model": _weighted_moments(model_values, model_weights),
            "planned_context_n": len(selected), "complete_context_n": complete_n,
            "unique_people_n": len(people), "coverage": 1.0}
        metrics[side] = metric
    return {"status": "SCORED" if complete_n == len(selected) else "INCOMPLETE",
        "unique_people_n": len(people), "planned_context_n": len(selected),
        "complete_context_n": complete_n, "coverage": complete_n / len(selected) if selected else 0.0,
        "by_side": metrics}


def _pooled_price_w1(contexts, labels, completed_samples, *, stratum: str | None = None) -> dict[str, Any]:
    selected = [key for key in contexts if stratum is None or contexts[key]["stratum"] == stratum]
    people = sorted({key[0] for key in selected})
    complete_n = sum(contexts[key]["K"] == 4 for key in selected)
    if complete_n != len(selected):
        return {"status": "INCOMPLETE", "by_side": {
            side: {"w1": None, "status": "INCOMPLETE", "possible_sample_pairs": len(selected) * 4,
                "eligible_sample_pairs": None, "coverage": None,
                "w1_interval": [0.0, 998.0],
                "observed_sample_pair_n": sum(contexts[key]["K"] for key in selected)}
                for side in ("buy", "sell")},
            "planned_context_n": len(selected), "complete_context_n": complete_n,
            "coverage": complete_n / len(selected) if selected else 0.0}
    output = {}
    for side in ("buy", "sell"):
        values, weights = [], []
        eligible = human_side_pairs = 0
        possible = 4 * len(selected)
        for person in people:
            person_keys = [key for key in selected if key[0] == person]
            context_weight = (1 / len(people)) / len(person_keys)
            for _, round_no, arm in person_keys:
                truth = labels[f"{person}:{round_no}"]["orders"]
                truth_has = any(row["side"] == side for row in truth)
                for orders in completed_samples[(person, round_no, arm)]:
                    if truth_has:
                        human_side_pairs += 1
                    value = _side_price_w1(truth, orders, side)
                    if value is not None:
                        values.append(value)
                        weights.append(context_weight / 4)
                        eligible += 1
        if values:
            output[side] = {"status": "SCORED", "weighted_mean_w1": _weighted_moments(values, weights)["mean"],
                "eligible_sample_pairs": eligible, "possible_sample_pairs": possible,
                "human_side_sample_pairs": human_side_pairs,
                "coverage": eligible / possible if possible else 0.0,
                "coverage_given_human_side": eligible / human_side_pairs if human_side_pairs else None,
                "eligible_weight": sum(weights)}
        else:
            output[side] = {"status": "NO_ELIGIBLE_DIRECTION", "weighted_mean_w1": None,
                "eligible_sample_pairs": 0, "possible_sample_pairs": possible,
                "human_side_sample_pairs": human_side_pairs,
                "coverage": 0.0, "coverage_given_human_side": 0.0 if human_side_pairs else None,
                "eligible_weight": 0.0}
    return {"status": "SCORED", "planned_context_n": len(selected),
        "complete_context_n": complete_n, "coverage": complete_n / len(selected) if selected else 0.0,
        "by_side": output}


def _fair_sample_distance(samples, truth, distance):
    if len(samples) < 2:
        return None
    n = len(samples)
    return (sum(distance(x, truth) for x in samples) / n
            - sum(distance(samples[i], samples[j]) for i in range(n) for j in range(n) if i != j)
            / (2 * n * (n - 1)))


def _average(values):
    return sum(values) / len(values) if values else None


def _bootstrap_interval(values: list[float], *, seed: int, draws: int = 10_000) -> tuple[float, float] | None:
    if not values:
        return None
    rng = random.Random(seed)
    n = len(values)
    estimates = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(draws))
    return estimates[int(0.025 * (draws - 1))], estimates[int(0.975 * (draws - 1))]


def score_run(preview_dir: str | Path, run_dir: str | Path, *, bootstrap_draws: int = 10_000) -> dict[str, Any]:
    """Score every target with fixed K and keep missing contexts as conservative intervals."""
    if bootstrap_draws != 10_000:
        raise ValueError("the frozen report uses exactly 10,000 participant reweightings")
    replayed = replay_run(preview_dir, run_dir)
    preview, run = Path(preview_dir), Path(run_dir)
    manifest = json.loads((preview / "manifest.json").read_text(encoding="utf-8"))
    labels = json.loads((preview / "scoring-targets.json").read_text(encoding="utf-8"))
    inputs = json.loads((preview / "inputs.json").read_text(encoding="utf-8"))["inputs"]
    private = json.loads((preview / "private-metadata.json").read_text(encoding="utf-8"))
    event_by_case = {event["case_id"]: event for event in replayed["events"] if event.get("status") == "complete"}
    rows_by_context: dict[tuple[str, int, str], list[tuple[dict[str, Any], dict[str, Any] | None]]] = defaultdict(list)
    for row in inputs:
        rows_by_context[(row["person_id"], row["target_round"], row["arm"])].append((row, event_by_case.get(row["case_id"])))
    target_rows: dict[tuple[str, str], dict[str, Any]] = {}
    sample_orders: dict[tuple[str, int, str], list[list[dict[str, Any]]]] = {}
    for key, context in rows_by_context.items():
        person, target_round, arm = key
        truth_row = labels[f"{person}:{target_round}"]
        truth = truth_row["orders"]
        ordered_context = sorted(context, key=lambda item: item[0]["sample"])
        sample_ids = [row["sample"] for row, _ in ordered_context]
        if len(sample_ids) != 4 or len(set(sample_ids)) != 4:
            raise ValueError("every context must have four distinct frozen sample numbers")
        complete_pairs = [(row["sample"], event["orders"]) for row, event in ordered_context if event is not None]
        complete_pairs.sort(key=lambda item: item[0])
        complete_sample_ids = [sample_id for sample_id, _ in complete_pairs]
        complete = [orders for _, orders in complete_pairs]
        sample_orders[key] = complete
        k = len(complete)
        truth_features = feature_vector(truth)
        q, d = truth_row["Q"], truth_row["D"]
        d_q_stratum = "D>Q" if d > q else "D<Q" if d < q else "D=Q"
        result: dict[str, Any] = {"person_id": person, "target_round": target_round,
            "arm": arm, "planned_K": 4, "observed_K": k, "sample_ids": complete_sample_ids,
            "D_Q_stratum": d_q_stratum}
        if k == 4:
            z = [feature_vector(orders) for orders in complete]
            result["fair_energy_score"] = fair_energy_score(z, truth_features)
            result["fair_crps_buy"] = fair_crps([sum(o["qty"] for o in p if o["side"] == "buy") for p in complete],
                sum(o["qty"] for o in truth if o["side"] == "buy"))
            result["fair_crps_sell"] = fair_crps([sum(o["qty"] for o in p if o["side"] == "sell") for p in complete],
                sum(o["qty"] for o in truth if o["side"] == "sell"))
            result["fair_crps_order_count"] = fair_crps([len(p) for p in complete], len(truth))
            result["fair_brier_empty"] = fair_brier([not p for p in complete], not truth)
            result["fair_brier_buy"] = fair_brier([any(o["side"] == "buy" for o in p) for p in complete],
                any(o["side"] == "buy" for o in truth))
            result["fair_brier_sell"] = fair_brier([any(o["side"] == "sell" for o in p) for p in complete],
                any(o["side"] == "sell" for o in truth))
            curves = [_curves(p) for p in complete]
            truth_curve = _curves(truth)
            flatten = lambda value: tuple(value[0]) + tuple(value[1])
            euclid = lambda a, b: math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
            result["fair_energy_full_curve"] = _fair_sample_distance([flatten(c) for c in curves], flatten(truth_curve), euclid)
            for name, low, high in (("1_100", 1, 100), ("101_999", 101, 999)):
                d = lambda a, b, lo=low, hi=high: math.sqrt(sum((a[s][p - 1] - b[s][p - 1]) ** 2
                    for s in (0, 1) for p in range(lo, hi + 1)))
                result[f"fair_energy_price_window_{name}"] = _fair_sample_distance(curves, truth_curve, d)
            idx = medoid_index(complete, sample_ids=complete_sample_ids)
            medoid = complete[idx]
            result["medoid_sample"] = complete_sample_ids[idx]
            result["medoid_curve_l1"] = curve_distance(medoid, truth) / 1998
            result["mean_random_draw_curve_l1"] = sum(curve_distance(portfolio, truth) / 1998
                for portfolio in complete) / 4
            result["random_draw_curve_l1"] = result["mean_random_draw_curve_l1"]
            result["sample_order_counts"] = [len(portfolio) for portfolio in complete]
            result["sample_buy_sell_quantities"] = [
                [sum(row["qty"] for row in portfolio if row["side"] == side)
                 for side in ("buy", "sell")] for portfolio in complete]
            multiset_counts = Counter(tuple(sorted((row["side"], row["qty"], row["price"])
                for row in portfolio)) for portfolio in complete)
            result["exact_portfolio_multiset_counts"] = [
                {"orders": [{"side": side, "qty": qty, "price": price} for side, qty, price in signature],
                 "sample_count": count} for signature, count in sorted(multiset_counts.items())]
            exactness = _portfolio_exactness(complete, truth)
            exactness["medoid_exact_match_human"] = _portfolio_signature(medoid) == _portfolio_signature(truth)
            result.update(exactness)
            result["five_order_samples_n"] = sum(len(portfolio) == 5 for portfolio in complete)
            result["order_count_histogram"] = {str(count): sum(len(portfolio) == count for portfolio in complete)
                for count in range(6)}
            count_errors = [abs(len(portfolio) - len(truth)) for portfolio in complete]
            buy_errors = [abs(sum(row["qty"] for row in portfolio if row["side"] == "buy")
                - sum(row["qty"] for row in truth if row["side"] == "buy")) for portfolio in complete]
            sell_errors = [abs(sum(row["qty"] for row in portfolio if row["side"] == "sell")
                - sum(row["qty"] for row in truth if row["side"] == "sell")) for portfolio in complete]
            result["mean_abs_order_count_error"] = _average(count_errors)
            result["mean_abs_buy_quantity_error"] = _average(buy_errors)
            result["mean_abs_sell_quantity_error"] = _average(sell_errors)
            feature_means = [sum(vector[index] for vector in z) / 4 for index in range(8)]
            result["sample_feature_means"] = dict(zip(FEATURE_NAMES, feature_means, strict=True))
            result["sample_feature_population_sd"] = dict(zip(FEATURE_NAMES,
                [math.sqrt(sum((vector[index] - feature_means[index]) ** 2 for vector in z) / 4)
                 for index in range(8)], strict=True))
            result["sample_feature_means_interval"] = {name: [feature_means[index], feature_means[index]]
                for index, name in enumerate(FEATURE_NAMES)}
            result["sample_feature_population_sd_interval"] = {name: [
                result["sample_feature_population_sd"][name], result["sample_feature_population_sd"][name]]
                for name in FEATURE_NAMES}
            sample_min = [[min(curve[side][price] for curve in curves) for price in range(999)]
                          for side in (0, 1)]
            sample_max = [[max(curve[side][price] for curve in curves) for price in range(999)]
                          for side in (0, 1)]
            covered = sum(sample_min[side][price] <= truth_curve[side][price] <= sample_max[side][price]
                          for side in (0, 1) for price in range(999))
            result["price_shape_l1_mean"] = sum(curve_distance(portfolio, truth) / 1998
                for portfolio in complete) / 4
            result["price_shape_coverage"] = covered / 1998
        else:
            for metric in METRIC_REGISTRY:
                result[f"{metric}_interval"] = _metric_interval(metric, q)
            observed_features = [feature_vector(portfolio) for portfolio in complete]
            result["sample_feature_means_interval"] = {}
            result["sample_feature_population_sd_interval"] = {}
            for index, name in enumerate(FEATURE_NAMES):
                observed_sum = sum(vector[index] for vector in observed_features)
                result["sample_feature_means_interval"][name] = [
                    observed_sum / 4, (observed_sum + (4-k) * FEATURE_MAXIMA[index]) / 4]
                result["sample_feature_population_sd_interval"][name] = [0.0, FEATURE_MAXIMA[index] / 2]
        target_rows[(f"{person}:{target_round}", arm)] = result

    # Exact expectation baselines over historical records, with no K-sample Monte Carlo.
    baseline_rows = []
    pool = private.get("baseline_pool", [])
    for _, label in labels.items():
        person, target_round = label["person_id"], label["target_round"]
        input_e = next(row for row in inputs if row["person_id"] == person and row["target_round"] == int(target_round) and row["arm"] == "E")
        personal = input_e["observation"]["history"]
        distributions = baseline_distributions(target={"p": input_e["observation"]["own_p"],
            "Q": label["Q"], "D": input_e["observation"]["own_d_units"]},
            target_id=person, public_history_pool=pool, personal_history=personal, seed=manifest["seed"])
        input_b = next(row for row in inputs if row["person_id"] == person and row["target_round"] == int(target_round) and row["arm"] == "B")
        short_distributions = baseline_distributions(target={"p": input_e["observation"]["own_p"],
            "Q": label["Q"], "D": input_e["observation"]["own_d_units"]},
            target_id=person, public_history_pool=pool,
            personal_history=input_b["observation"]["history"], seed=manifest["seed"])
        truth_feature = feature_vector(label["orders"])
        for name, data in (("population", distributions["population"]),
                           ("state", distributions["state"]),
                           ("personal_long8", distributions["personal"]),
                           ("personal_short3", short_distributions["personal"])):
            score = expected_fair_score(data["records"], truth_feature, feature_vector,
                lambda a, b: math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b))), empty_fallback=data["fallback_empty"])
            full_curve = _expected_energy_curve(data["records"], label["orders"])
            buy_truth = sum(order["qty"] for order in label["orders"] if order["side"] == "buy")
            sell_truth = sum(order["qty"] for order in label["orders"] if order["side"] == "sell")
            count_truth = len(label["orders"])
            buy_predicate = lambda orders: sum(order["qty"] for order in orders if order["side"] == "buy")
            sell_predicate = lambda orders: sum(order["qty"] for order in orders if order["side"] == "sell")
            count_predicate = lambda orders: len(orders)
            baseline_rows.append({"person_id": person, "target_round": int(target_round),
                "baseline": name, "fair_energy_score": score, "candidate_n": data["source_n"],
                "fallback_empty": data["fallback_empty"],
                "fair_energy_full_curve": full_curve,
                "fair_crps_buy": _expected_fair_scalar(data["records"], buy_truth, buy_predicate),
                "fair_crps_sell": _expected_fair_scalar(data["records"], sell_truth, sell_predicate),
                "fair_crps_order_count": _expected_fair_scalar(data["records"], count_truth, count_predicate),
                "fair_brier_empty": _expected_fair_brier(data["records"], not label["orders"], lambda orders: not orders),
                "fair_brier_buy": _expected_fair_brier(data["records"],
                    any(order["side"] == "buy" for order in label["orders"]),
                    lambda orders: any(order["side"] == "buy" for order in orders)),
                "fair_brier_sell": _expected_fair_brier(data["records"],
                    any(order["side"] == "sell" for order in label["orders"]),
                    lambda orders: any(order["side"] == "sell" for order in orders))})

    # Target scores are averaged within person before participants receive equal weight.
    per_arm = {}
    per_arm_bounds = {}
    participants_by_arm = {}
    metric_names = list(METRIC_REGISTRY)
    per_arm_complete_case = {}
    for arm in "ABCDE":
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for ((target_key, row_arm), result) in target_rows.items():
            if row_arm == arm:
                grouped[target_key.split(":")[0]].append(result)
        participant_metrics = {}
        participant_bounds = {}
        for person, rows in grouped.items():
            participant_metrics[person] = {}
            participant_bounds[person] = {}
            for metric in metric_names:
                bounds = []
                for row in rows:
                    value = row.get(metric)
                    if isinstance(value, (int, float)):
                        bounds.append((float(value), float(value)))
                    else:
                        interval = row.get(f"{metric}_interval")
                        if interval is None:
                            interval = _metric_interval(metric, next(
                                label["Q"] for label in labels.values()
                                if label["person_id"] == person and int(label["target_round"]) == row["target_round"]))
                        bounds.append((float(interval[0]), float(interval[1])))
                # Each formal target round receives exactly half the person's weight.
                if len(bounds) == 2:
                    participant_bounds[person][metric] = [sum(pair[i] for pair in bounds) / 2 for i in (0, 1)]
                    participant_metrics[person][metric] = (participant_bounds[person][metric][0]
                        if participant_bounds[person][metric][0] == participant_bounds[person][metric][1] else None)
                else:
                    participant_bounds[person][metric] = [0.0, 1.0]
                    participant_metrics[person][metric] = None
        participants_by_arm[arm] = participant_metrics
        participants_by_arm[arm + "_bounds"] = participant_bounds
        per_arm_complete_case[arm] = {metric: {"mean": _average([values[metric] for values in participant_metrics.values()
            if values[metric] is not None]), "n": sum(values[metric] is not None for values in participant_metrics.values())}
            for metric in metric_names}
        per_arm[arm] = {metric: (per_arm_complete_case[arm][metric]["mean"]
            if per_arm_complete_case[arm][metric]["n"] == 17 else None) for metric in metric_names}
        per_arm_bounds[arm] = {metric: [_average([value[metric][bound] for value in participant_bounds.values()])
                                              for bound in (0, 1)] for metric in metric_names}

    baseline_by_name: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in baseline_rows:
        baseline_by_name[row["baseline"]][row["person_id"]].append(row)

    def baseline_person_bounds(name: str, person: str, metric: str) -> list[float]:
        rows = baseline_by_name[name][person]
        values = [float(row[metric]) for row in rows]
        mean = _average(values)
        return [mean, mean]

    baseline_registry = ("fair_energy_score", "fair_energy_full_curve", "fair_crps_buy",
        "fair_crps_sell", "fair_crps_order_count", "fair_brier_empty", "fair_brier_buy", "fair_brier_sell")
    baseline_participant_first = {}
    for name, by_person in baseline_by_name.items():
        baseline_participant_first[name] = {}
        for metric in baseline_registry:
            person_values = [_average([row[metric] for row in by_person[person]]) for person in sorted(by_person)]
            baseline_participant_first[name][metric] = {"mean": _average(person_values),
                "participant_n": len(person_values)}

    def source_bounds(source: str, person: str, metric: str) -> list[float]:
        if source in "ABCDE" and len(source) == 1:
            return participants_by_arm[source + "_bounds"][person][metric]
        baseline_metric = "fair_energy_score" if metric == "fair_energy_score" else metric
        return baseline_person_bounds(source, person, baseline_metric)

    contrast_pairs = (("E", "state"), ("E", "population"), ("C", "D"),
                      ("E", "C"), ("B", "A"), ("C", "B"))
    contrasts = {}
    for left, right in contrast_pairs:
        bounds = {}
        for person in sorted({row["person_id"] for row in inputs}):
            lbound = source_bounds(left, person, "fair_energy_score")
            rbound = source_bounds(right, person, "fair_energy_score")
            bounds[person] = [lbound[0] - rbound[1], lbound[1] - rbound[0]]
        contrasts[f"{left}_minus_{right}"] = _difference_summary(bounds,
            seed=int(manifest["seed"]) + sum(ord(ch) for ch in left + right))

    primary_summary = contrasts["E_minus_state"]
    participant_bounds_for_E = participants_by_arm["E_bounds"]
    exact_E = all(participant_bounds_for_E[person]["fair_energy_score"][0]
                   == participant_bounds_for_E[person]["fair_energy_score"][1]
                   for person in participant_bounds_for_E)
    e_mean = per_arm["E"]["fair_energy_score"] if exact_E else None
    state_mean = _average([_average([row["fair_energy_score"] for row in rows])
        for rows in baseline_by_name["state"].values()])
    improve = (1 - e_mean / state_mean) if e_mean is not None and state_mean not in (None, 0) else None
    all_planned = replayed["mode"] == "provider" and replayed["status"] == "REPLAY_VALID" and replayed["sample_n"] == 680
    complete_contexts = (len(rows_by_context) == 34 * 5 and all(len(sample_orders[key]) == 4 for key in rows_by_context))
    all_people = len({row["person_id"] for row in inputs}) == 17
    gate = bool(all_planned and complete_contexts and all_people and improve is not None and improve >= 0.10
        and primary_summary["participant_reweighting_95_interval"][1] < 0
        and primary_summary["mean_difference_interval"][1] < 0
        and primary_summary["leave_one_out_max_upper_bound"] < 0)

    context_metadata = {(result["person_id"], result["target_round"], result["arm"]):
        {"K": result["observed_K"], "stratum": result["D_Q_stratum"]}
        for result in target_rows.values()}
    weighted_order_totals_w1_by_arm = {}
    weighted_price_w1_by_arm = {}
    portfolio_diagnostics_by_arm = {}
    feature_moments_by_arm = {}
    for arm in "ABCDE":
        arm_contexts = [result for result in target_rows.values() if result["arm"] == arm]
        complete_arm_contexts = [result for result in arm_contexts if result["observed_K"] == 4]
        complete_sample_n = 4 * len(complete_arm_contexts)
        repeated_order_n = sum(sum(result.get("within_portfolio_repeated_order_counts", []))
            for result in complete_arm_contexts)
        repeated_any_n = sum(sum(count > 0 for count in result.get("within_portfolio_repeated_order_counts", []))
            for result in complete_arm_contexts)
        count_hist = {str(number): sum(result["order_count_histogram"].get(str(number), 0)
            for result in complete_arm_contexts) for number in range(6)}
        portfolio_diagnostics_by_arm[arm] = {
            "planned_contexts": len(arm_contexts),
            "complete_contexts": len(complete_arm_contexts),
            "incomplete_contexts": len(arm_contexts) - len(complete_arm_contexts),
            "observed_sample_n": sum(result["observed_K"] for result in arm_contexts),
            "distinct_multisets_across_contexts": sum(result.get("distinct_forecast_portfolios_n", 0)
                for result in complete_arm_contexts),
            "duplicate_forecast_sample_n": sum(result.get("duplicate_forecast_sample_n", 0)
                for result in complete_arm_contexts),
            "within_portfolio_repeated_order_count": sum(
                sum(result.get("within_portfolio_repeated_order_counts", [])) for result in complete_arm_contexts),
            "within_portfolio_repeated_any_sample_n": sum(
                sum(count > 0 for count in result.get("within_portfolio_repeated_order_counts", []))
                for result in complete_arm_contexts),
            "complete_sample_n": complete_sample_n,
            "mean_within_portfolio_repeated_order_count": repeated_order_n / complete_sample_n
                if complete_sample_n else None,
            "within_portfolio_repeated_any_rate": repeated_any_n / complete_sample_n
                if complete_sample_n else None,
            "medoid_exact_match_human_n": sum(bool(result.get("medoid_exact_match_human"))
                for result in complete_arm_contexts),
            "mean_random_draw_exact_match_human": (_average([result["mean_random_draw_exact_match_human"]
                for result in complete_arm_contexts]) if len(complete_arm_contexts) == len(arm_contexts) else None),
            "five_order_sample_n": sum(result.get("five_order_samples_n", 0) for result in complete_arm_contexts),
            "complete_sample_order_count_histogram": count_hist,
            "participant_first_mean_abs_order_count_error_interval": per_arm_bounds[arm]["mean_abs_order_count_error"],
            "participant_first_mean_abs_buy_quantity_error_interval": per_arm_bounds[arm]["mean_abs_buy_quantity_error"],
            "participant_first_mean_abs_sell_quantity_error_interval": per_arm_bounds[arm]["mean_abs_sell_quantity_error"],
        }
        feature_moments_by_arm[arm] = _feature_summary(arm_contexts)
        arm_keys = [key for key in context_metadata if key[2] == arm]
        arm_context_metadata = {key: context_metadata[key] for key in arm_keys}
        weighted_order_totals_w1_by_arm[arm] = {"overall": _pooled_raw_w1(
            arm_context_metadata, labels, sample_orders)}
        weighted_price_w1_by_arm[arm] = {"overall": _pooled_price_w1(
            arm_context_metadata, labels, sample_orders)}
        for stratum in ("D>Q", "D=Q", "D<Q"):
            stratum_keys = [key for key in arm_keys if context_metadata[key]["stratum"] == stratum]
            people = {key[0] for key in stratum_keys}
            if len(people) < 5:
                weighted_order_totals_w1_by_arm[arm][stratum] = {
                    "status": "INSUFFICIENT_UNIQUE_PEOPLE", "unique_people_n": len(people),
                    "required_unique_people_n": 5}
                weighted_price_w1_by_arm[arm][stratum] = weighted_order_totals_w1_by_arm[arm][stratum]
                continue
            weighted_order_totals_w1_by_arm[arm][stratum] = _pooled_raw_w1(
                arm_context_metadata, labels, sample_orders, stratum=stratum)
            weighted_price_w1_by_arm[arm][stratum] = _pooled_price_w1(
                arm_context_metadata, labels, sample_orders, stratum=stratum)
            feature_moments_by_arm[arm][stratum] = _feature_summary([
                result for result in arm_contexts if result["D_Q_stratum"] == stratum])
    return {"status": "DESCRIPTIVE_CONDITIONAL_DISTRIBUTION", "preview_sha256": manifest["preview_sha256"],
        "execution_mode": replayed["mode"], "planned_targets": 34, "completed_sample_rows": replayed["sample_n"],
        "physical_attempted_requests": replayed.get("physical_attempted_requests"),
        "completed_trace_requests": replayed.get("completed_trace_requests"),
        "case_state_counts": replayed.get("case_state_counts"),
        "provider_attempt_evidence": replayed.get("provider_attempt_evidence"),
        "target_rows": list(target_rows.values()), "baseline_target_rows": baseline_rows,
        "per_arm_participant_first": per_arm, "per_arm_participant_first_bounds": per_arm_bounds,
        "per_arm_complete_case": per_arm_complete_case,
        "participants_by_arm": participants_by_arm,
        "paired_energy_contrasts": contrasts,
        "weighted_order_totals_w1_by_arm": weighted_order_totals_w1_by_arm,
        "weighted_price_w1_by_arm": weighted_price_w1_by_arm,
        "weighted_feature_moments_by_arm": feature_moments_by_arm,
        "portfolio_diagnostics_by_arm": portfolio_diagnostics_by_arm,
        "primary_E_vs_state": {"relative_improvement": improve,
            "mean_paired_difference": primary_summary["complete_case_mean"],
            "median_paired_difference": primary_summary["complete_case_median"],
            "participant_differences": primary_summary["participant_differences"],
            "participant_reweighting_95_interval": primary_summary["participant_reweighting_95_interval"],
            "leave_one_participant_out_intervals": primary_summary["leave_one_participant_out_intervals"],
            "leave_one_participant_out_min_lower_bound": primary_summary["leave_one_out_min_lower_bound"],
            "leave_one_participant_out_max_upper_bound": primary_summary["leave_one_out_max_upper_bound"],
            "aggregate_difference_interval": primary_summary["mean_difference_interval"],
            "aggregate_difference_upper_bound": primary_summary["mean_difference_interval"][1],
            "all_680_provider_rows_valid": all_planned,
            "all_34x5_contexts_K4": bool(complete_contexts and all_people),
            "meets_10pct_and_interval_gate": gate},
        "score_registry": list(METRIC_REGISTRY),
        "baseline_score_registry": ["fair_energy_score", "fair_energy_full_curve", "fair_crps_buy",
            "fair_crps_sell", "fair_crps_order_count", "fair_brier_empty", "fair_brier_buy", "fair_brier_sell"],
        "baseline_participant_first": baseline_participant_first,
        "metric_definitions": {
            "raw_order_total_w1": "Exact 1-D Wasserstein distance obtained by integrating absolute CDF differences over pooled empirical raw BUY or SELL totals.",
            "raw_order_total_w1_weights": "Overall human context weights are 1/17 person times 1/2 target round; each of four model draws receives 1/17 times 1/2 times 1/4. Within D/Q strata, person weight is equal, then context weight equal within person, then four draws equal within context.",
            "price_w1": "Per-side quantity-normalized Wasserstein distance over limit prices, only for context/sample pairs where both sides exist.",
            "price_shape_l1_mean": "Mean sample-to-truth L1 distance over both 999-price cumulative curves, divided by 1998.",
            "price_shape_coverage": "Fraction of the 1998 side-price curve coordinates where truth lies inside the four-sample min/max envelope.",
            "participant_weighting": "Each person has weight 1/17; each target round has weight 1/2 within that person.",
            "d_q_strata": "Equal participant weights and equal eligible-context weights within participant; only strata with at least five unique people are scored."
        },
        "limitations": ["10,000 participant reweightings are sensitivity intervals, not guaranteed-coverage confidence intervals.",
            "Four samples per context do not estimate tail probabilities reliably.",
            "Each arm metric uses fixed half weight for rounds 14 and 15 within person, then equal weight for all 17 people.",
            "Pooled W1 distributions are marked INCOMPLETE when any contributing context lacks four valid draws; missing contexts are never dropped.",
            "Only strict complete replays receive exact scores; missing samples retain fixed-registry conservative intervals."]}

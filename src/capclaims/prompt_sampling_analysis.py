"""Descriptive, participant-paired summaries for the six-cell sampling design."""
from __future__ import annotations

from collections import defaultdict
from statistics import mean, median
from typing import Any, Mapping, Sequence

from capclaims.prompt_sampling_presentation import PROMPTS
from capclaims.prompt_sampling_decoder import DECODERS


def summarize_cells(rows: Sequence[Mapping[str, Any]], metric: str = "primary_curve_error") -> dict[str, Any]:
    if len(rows) != 24:
        raise ValueError("each prompt batch requires 24 scored forecasts")
    roster = sorted({row.get("person_id") for row in rows})
    if len(roster) != 6 or None in roster:
        raise ValueError("each prompt batch requires exactly six private participants")
    expected = {(person, decoder, repeat) for person in roster for decoder in DECODERS for repeat in (0, 1)}
    observed = set()
    people: dict[tuple[str, str], list[float]] = defaultdict(list)
    repeats: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        if row.get("prompt") not in PROMPTS or row.get("decoder") not in DECODERS:
            raise ValueError("scored row has an unknown prompt or decoder")
        factor = (row["person_id"], row["decoder"], row.get("repeat"))
        if factor not in expected or factor in observed:
            raise ValueError("prompt batch has a duplicate or invalid participant/decoder/repeat row")
        observed.add(factor)
        value = float(row[metric])
        if not __import__("math").isfinite(value):
            raise ValueError("scored metric must be finite")
        people[(row["prompt"], row["decoder"], row["person_id"])].append(value)
        repeats[(row["prompt"], row["decoder"], row["person_id"])].append(value)
    if observed != expected:
        raise ValueError("prompt batch is missing one or more participant/decoder/repeat rows")
    person_means = {f"{p}/{d}/{person}": mean(values)
        for (p, d, person), values in people.items()}
    present_prompts = sorted({row["prompt"] for row in rows}, key=PROMPTS.index)
    aggregate = {f"{p}/{d}": mean(person_means[f"{p}/{d}/{person}"] for person in roster)
        for p in present_prompts for d in DECODERS}
    return {"metric": metric, "cell_means": aggregate, "per_person_means": person_means,
        "rows_n": len(rows), "paired_differences": _paired(rows, metric),
        "repeat_consistency": _repeat_consistency(rows)}


def _repeat_consistency(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by = defaultdict(dict)
    for row in rows:
        by[(row["prompt"], row["decoder"], row["person_id"])][row["repeat"]] = row
    same_order = 0
    compared = 0
    for pair in by.values():
        if 0 not in pair or 1 not in pair:
            raise ValueError("repeat consistency needs exactly both repeats")
        # Full order signatures are private row values supplied by score().
        compared += 1
        same_order += pair[0].get("order_multiset_signature") == pair[1].get("order_multiset_signature")
    return {"same_portfolio_n": same_order, "person_decoder_pairs_n": compared,
        "same_portfolio_rate": same_order / compared if compared else None}


def _paired(rows: Sequence[Mapping[str, Any]], metric: str) -> dict[str, Any]:
    by = defaultdict(list)
    for row in rows:
        by[(row["person_id"], row["prompt"], row["decoder"])].append(float(row[metric]))
    person_means = {key: mean(vals) for key, vals in by.items()}
    decoding = {}
    present_prompts = [prompt for prompt in PROMPTS if any(row.get("prompt") == prompt for row in rows)]
    for prompt in present_prompts:
        differences = [person_means[(person, prompt, "top3_cutoff10")] - person_means[(person, prompt, "client_argmax")]
            for person in sorted({k[0] for k in person_means})]
        decoding[prompt] = _contrast(differences)
    prompts = {}
    for decoder in DECODERS:
        for left, right in ((PROMPTS[0], PROMPTS[1]), (PROMPTS[0], PROMPTS[2]), (PROMPTS[1], PROMPTS[2])):
            if left not in present_prompts or right not in present_prompts:
                continue
            diff = [person_means[(person, left, decoder)] - person_means[(person, right, decoder)]
                for person in sorted({k[0] for k in person_means})]
            prompts[f"{left}_minus_{right}/{decoder}"] = _contrast(diff)
    return {"decoder_within_prompt": decoding, "prompt_within_decoder": prompts}


def _contrast(values: Sequence[float]) -> dict[str, Any]:
    leave_one_out = [mean([value for j, value in enumerate(values) if j != omitted])
        for omitted in range(len(values))] if len(values) > 1 else []
    return {"mean": mean(values), "median": median(values),
        "lower_equal_higher": {"lower": sum(value < 0 for value in values),
            "equal": sum(value == 0 for value in values), "higher": sum(value > 0 for value in values)},
        "person_differences": list(values),
        "leave_one_person_out_mean_range": [min(leave_one_out), max(leave_one_out)] if leave_one_out else None}


def compare_three_runs(run_summaries: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    if set(run_summaries) != set(PROMPTS):
        raise ValueError("comparison requires exactly original, explicit, and structured batches")
    cells = {}
    for prompt in PROMPTS:
        summary = run_summaries[prompt]
        if summary.get("status") != "SCORED_OFFLINE":
            raise ValueError("all three run summaries must be replayed and scored")
        for decoder, value in summary.get("configs", {}).items():
            cells[f"{prompt}/{decoder}"] = value
    ordered = sorted(cells, key=lambda name: (cells[name]["mean_primary_curve_error"], name))
    if ordered:
        best_value = cells[ordered[0]]["mean_primary_curve_error"]
        tied = [name for name in ordered if cells[name]["mean_primary_curve_error"] == best_value]
    else:
        tied = []
    return {"status": "DESCRIPTIVE_DEVELOPMENT_ONLY", "configs": cells,
        "ranking_by_participant_averaged_primary_error": ordered, "tied_best_configs": tied,
        "selection_scope": "development only; no promotion", "repeat_count": 2,
        "limitations": ["Six selected people; no population-equivalence claim.",
            "Two repeats support only small descriptive sampling diagnostics.",
            "Prompt conditions differ in length and organization, so mechanism cannot be isolated."]}


def summarize_three_prompt_cases(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate the private 72 case rows without returning participant-level values."""
    if len(rows) != 72:
        raise ValueError("three-prompt comparison requires exactly 72 scored forecasts")
    roster = sorted({row.get("person_id") for row in rows})
    if len(roster) != 6 or None in roster:
        raise ValueError("three-prompt comparison requires the same six participants")
    expected = {(person, prompt, decoder, repeat) for person in roster for prompt in PROMPTS
        for decoder in DECODERS for repeat in (0, 1)}
    observed = set()
    by = {}
    ids = set()
    for row in rows:
        factor = (row.get("person_id"), row.get("prompt"), row.get("decoder"), row.get("repeat"))
        session_key = (row.get("prompt"), row.get("session_id"))
        if factor not in expected or factor in observed or session_key in ids:
            raise ValueError("three-prompt rows contain duplicate or invalid factors")
        if row.get("session_id") is None:
            raise ValueError("three-prompt case lacks its unique session id")
        observed.add(factor)
        ids.add(session_key)
        value = float(row["primary_curve_error"])
        if not __import__("math").isfinite(value):
            raise ValueError("primary score must be finite")
        by[factor] = row
    if observed != expected:
        raise ValueError("three-prompt comparison is missing one or more factors")

    person_scores = {}
    configs = {}
    repeat_consistency = {}
    for prompt in PROMPTS:
        for decoder in DECODERS:
            key = f"{prompt}/{decoder}"
            means, abs_repeat_errors, same = [], [], 0
            for person in roster:
                first = by[(person, prompt, decoder, 0)]
                second = by[(person, prompt, decoder, 1)]
                person_mean = (float(first["primary_curve_error"]) + float(second["primary_curve_error"])) / 2
                means.append(person_mean)
                person_scores[(person, prompt, decoder)] = person_mean
                abs_repeat_errors.append(abs(float(first["primary_curve_error"]) - float(second["primary_curve_error"])))
                same += first.get("order_multiset_signature") == second.get("order_multiset_signature")
            configs[key] = {"mean_primary_curve_error": mean(means),
                "median_participant_primary_curve_error": median(means),
                "mean_absolute_repeat_primary_error": mean(abs_repeat_errors),
                "person_n": 6, "forecast_n": 12}
            repeat_consistency[key] = {"same_portfolio_n": same, "participant_decoder_pairs_n": 6,
                "same_portfolio_rate": same / 6,
                "mean_absolute_repeat_primary_error": mean(abs_repeat_errors),
                "median_absolute_repeat_primary_error": median(abs_repeat_errors)}

    decoding = {}
    for prompt in PROMPTS:
        diffs = [person_scores[(person, prompt, "top3_cutoff10")] - person_scores[(person, prompt, "client_argmax")]
            for person in roster]
        decoding[prompt] = _safe_contrast(diffs)
    prompts = {}
    pairs = ((PROMPTS[0], PROMPTS[1]), (PROMPTS[0], PROMPTS[2]), (PROMPTS[1], PROMPTS[2]))
    for decoder in DECODERS:
        for left, right in pairs:
            diffs = [person_scores[(person, left, decoder)] - person_scores[(person, right, decoder)]
                for person in roster]
            prompts[f"{left}_minus_{right}/{decoder}"] = _safe_contrast(diffs)
    ranking = sorted(configs, key=lambda key: (configs[key]["mean_primary_curve_error"], key))
    best = configs[ranking[0]]["mean_primary_curve_error"]
    tied = [key for key in ranking if configs[key]["mean_primary_curve_error"] == best]
    return {"status": "DESCRIPTIVE_DEVELOPMENT_ONLY", "forecast_n": 72, "person_n": 6,
        "configs": configs, "decoder_within_prompt": decoding,
        "prompt_within_decoder": prompts, "repeat_consistency_by_config": repeat_consistency,
        "ranking_by_participant_averaged_primary_error": ranking, "tied_best_configs": tied,
        "selection_scope": "development only; no promotion",
        "limitations": ["Six selected participants; no population-equivalence claim.",
            "Two repeats support descriptive sampling diagnostics only.",
            "Prompt conditions differ in length and organization, so mechanism cannot be isolated."]}


def _safe_contrast(values: Sequence[float]) -> dict[str, Any]:
    leave_one_out = [mean([value for j, value in enumerate(values) if j != omitted])
        for omitted in range(len(values))] if len(values) > 1 else []
    return {"mean": mean(values), "median": median(values),
        "lower_equal_higher": {"lower": sum(value < 0 for value in values),
            "equal": sum(value == 0 for value in values), "higher": sum(value > 0 for value in values)},
        "leave_one_person_out_mean_range": [min(leave_one_out), max(leave_one_out)] if leave_one_out else None,
        "paired_person_n": len(values)}

"""Analyze two representation-feature sweep blocks."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

try:
    from analysis.analyze_transmutation import (
        ACTION_INDEX,
        design_matrix,
        fit_fourier,
        pointwise_overdispersion_ratio,
    )
except ModuleNotFoundError:
    from analyze_transmutation import (
        ACTION_INDEX,
        design_matrix,
        fit_fourier,
        pointwise_overdispersion_ratio,
    )


def _canonical_id(specification: str | dict) -> str:
    if isinstance(specification, str):
        return specification
    kind = specification["kind"]
    n_bins = int(specification.get("n_bins", 24))
    decimals = int(specification.get("decimals", 6))
    shift = float(specification.get("origin_shift_bins", 0.0))
    if kind == "intervals":
        return f"intervals_n{n_bins}_d{decimals}_o{shift:g}"
    if kind == "centers":
        return f"centers_n{n_bins}_d{decimals}_o{shift:g}"
    if kind == "counts":
        total = int(specification.get("count_total", 100))
        return f"counts_n{n_bins}_total{total}_o{shift:g}"
    order = int(specification.get("moment_order", 3))
    return f"moments_m1_m{order}_d{decimals}"


def bootstrap_smoothed_condition(
    counts: np.ndarray,
    offsets: np.ndarray,
    order: int,
    samples: int,
    pseudocount: float,
    rng: np.random.Generator,
) -> tuple[dict[str, np.ndarray], dict]:
    totals = np.sum(counts, axis=1).astype(np.int64)
    probability = (counts + pseudocount) / (
        totals[:, None] + 3.0 * pseudocount
    )
    draws = np.empty((samples, offsets.size, 3), dtype=np.int16)
    for index, (total, cell_probability) in enumerate(
        zip(totals, probability)
    ):
        draws[:, index, :] = rng.multinomial(
            int(total),
            cell_probability,
            size=samples,
        )
    g = (draws[:, :, 2] - draws[:, :, 0]) / totals[None, :]
    inverse = np.linalg.pinv(design_matrix(offsets, order))
    beta = g @ inverse.T
    values = {
        "a0": beta[:, 0],
        "a_sin_1": beta[:, 1],
        "b_cos_1": beta[:, 2],
        "a_sin_2": beta[:, 3],
        "b_cos_2": beta[:, 4],
        "mean_activity": np.mean(
            (draws[:, :, 0] + draws[:, :, 2]) / totals[None, :],
            axis=1,
        ),
    }
    summary = {
        name: {
            "median": float(np.quantile(value, 0.5)),
            "ci95_low": float(np.quantile(value, 0.025)),
            "ci95_high": float(np.quantile(value, 0.975)),
        }
        for name, value in values.items()
    }
    return values, summary


def classify_a1(samples: np.ndarray, rules: dict) -> dict:
    probabilities = {
        "polar": float(
            np.mean(samples > float(rules["minimum_polar_a1"]))
        ),
        "suppressed": float(
            np.mean(
                np.abs(samples)
                <= float(rules["maximum_suppressed_abs_a1"])
            )
        ),
        "sign_reversing": float(
            np.mean(
                samples < float(rules["maximum_sign_reversing_a1"])
            )
        ),
    }
    label = max(probabilities, key=probabilities.get)
    if probabilities[label] < float(rules["minimum_class_probability"]):
        label = "unresolved"
    return {"label": label, "probabilities": probabilities}


def adjacent_diagnostic(
    left: np.ndarray,
    right: np.ndarray,
    left_class: str,
    right_class: str,
    rules: dict,
) -> dict:
    difference = right - left
    low = float(np.quantile(difference, 0.025))
    high = float(np.quantile(difference, 0.975))
    jump_probability = float(
        np.mean(
            np.abs(difference)
            >= float(rules["minimum_jump_magnitude"])
        )
    )
    opposite = {left_class, right_class} == {
        "polar",
        "sign_reversing",
    }
    return {
        "median_delta_a1": float(np.quantile(difference, 0.5)),
        "ci95_low": low,
        "ci95_high": high,
        "jump_probability": jump_probability,
        "operational_jump": (
            jump_probability >= float(rules["minimum_jump_probability"])
        ),
        "sign_transition": opposite,
    }


def _representation_ids(protocol: dict) -> list[str]:
    return [
        _canonical_id(specification)
        for specification in protocol["representations"]
    ]


def load_block(run_dir: Path) -> tuple[dict, dict[str, dict[str, np.ndarray]]]:
    config = json.loads(
        (run_dir / "resolved_config.json").read_text(encoding="utf-8")
    )
    protocol = config["protocol"]
    identifiers = _representation_ids(protocol)
    counts = {
        identifier: {
            profile: np.zeros(
                (len(protocol["offset_radians"]), 3),
                dtype=np.int64,
            )
            for profile in protocol["concentration_profiles"]
        }
        for identifier in identifiers
    }
    valid = 0
    total = 0
    with (run_dir / "trace.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            total += 1
            if not record.get("valid"):
                continue
            valid += 1
            counts[record["representation"]][record["profile"]][
                int(record["offset_index"])
            ][ACTION_INDEX[record["action_label"]]] += 1
    minimum = min(
        int(np.min(np.sum(cells, axis=1)))
        for representation in counts.values()
        for cells in representation.values()
    )
    operational = {
        "run_dir": str(run_dir),
        "seed_block_id": protocol["seed_block_id"],
        "n_records": total,
        "n_valid": valid,
        "valid_rate": valid / total,
        "minimum_cell_samples": minimum,
        "passes": (
            valid / total >= float(protocol["criteria"]["minimum_valid_rate"])
            and minimum
            >= int(protocol["criteria"]["minimum_valid_samples_per_condition"])
        ),
    }
    return {**config, "operational": operational}, counts


def analyze(
    run_dirs: list[Path],
    output_dir: Path,
    reference_analysis: Path | None = None,
) -> dict:
    if len(run_dirs) != 2:
        raise ValueError("exactly two feature-block runs are required")
    loaded = [load_block(path) for path in run_dirs]
    configs = [loaded_item[0] for loaded_item in loaded]
    count_blocks = [loaded_item[1] for loaded_item in loaded]
    protocols = [config["protocol"] for config in configs]
    shared = (
        "offset_radians",
        "concentration_profiles",
        "representations",
        "feature_axes",
        "repetitions_per_condition",
        "primary_fourier_order",
        "sensitivity_fourier_orders",
        "classification_rules",
        "criteria",
    )
    for key in shared:
        if protocols[0][key] != protocols[1][key]:
            raise ValueError(f"feature protocols differ at {key}")
    protocol = protocols[0]
    identifiers = _representation_ids(protocol)
    offsets = np.asarray(protocol["offset_radians"], dtype=np.float64)
    profiles = sorted(
        protocol["concentration_profiles"],
        key=lambda profile: protocol["concentration_profiles"][profile],
    )
    order = int(protocol["primary_fourier_order"])
    samples = int(protocol["bootstrap_samples"])
    pseudocount = float(protocol.get("bootstrap_pseudocount", 0.5))
    rules = protocol["classification_rules"]
    block_results = {}
    block_samples = []
    rows = []
    maximum_sensitivity = {"a1": 0.0, "a2": 0.0}
    for block_index, (config, counts) in enumerate(loaded):
        block_id = config["protocol"]["seed_block_id"]
        rng = np.random.default_rng(2026152300 + block_index)
        block_results[block_id] = {
            "operational": config["operational"],
            "conditions": {},
        }
        sample_store = {}
        for identifier in identifiers:
            block_results[block_id]["conditions"][identifier] = {}
            sample_store[identifier] = {}
            for profile in profiles:
                cell_counts = counts[identifier][profile]
                estimate = fit_fourier(cell_counts, offsets, order)
                action_probability = np.mean(
                    cell_counts
                    / np.sum(cell_counts, axis=1, keepdims=True),
                    axis=0,
                )
                estimate["mean_action_probabilities"] = {
                    "retard": float(action_probability[0]),
                    "stay": float(action_probability[1]),
                    "advance": float(action_probability[2]),
                }
                values, interval = bootstrap_smoothed_condition(
                    cell_counts,
                    offsets,
                    order,
                    samples,
                    pseudocount,
                    rng,
                )
                classification = classify_a1(values["a_sin_1"], rules)
                sensitivity = {
                    str(sensitivity_order): fit_fourier(
                        cell_counts,
                        offsets,
                        int(sensitivity_order),
                    )
                    for sensitivity_order in protocol[
                        "sensitivity_fourier_orders"
                    ]
                }
                range_a1 = float(
                    np.ptp(
                        [
                            result["a_sin_1"]
                            for result in sensitivity.values()
                        ]
                    )
                )
                range_a2 = float(
                    np.ptp(
                        [
                            result["a_sin_2"]
                            for result in sensitivity.values()
                        ]
                    )
                )
                maximum_sensitivity["a1"] = max(
                    maximum_sensitivity["a1"],
                    range_a1,
                )
                maximum_sensitivity["a2"] = max(
                    maximum_sensitivity["a2"],
                    range_a2,
                )
                block_results[block_id]["conditions"][identifier][profile] = {
                    "concentration": float(
                        protocol["concentration_profiles"][profile]
                    ),
                    "estimate": estimate,
                    "bootstrap": interval,
                    "classification": classification,
                    "sensitivity_orders": sensitivity,
                    "sensitivity_range_a1": range_a1,
                    "sensitivity_range_a2": range_a2,
                }
                sample_store[identifier][profile] = values
                rows.append(
                    {
                        "block": block_id,
                        "representation": identifier,
                        "profile": profile,
                        "concentration": protocol[
                            "concentration_profiles"
                        ][profile],
                        "a0": estimate["a0"],
                        "a1": estimate["a_sin_1"],
                        "a2": estimate["a_sin_2"],
                        "b1": estimate["b_cos_1"],
                        "b2": estimate["b_cos_2"],
                        "activity": estimate["mean_activity"],
                        "a1_class": classification["label"],
                        "a1_ci_low": interval["a_sin_1"]["ci95_low"],
                        "a1_ci_high": interval["a_sin_1"]["ci95_high"],
                    }
                )
        block_samples.append(sample_store)

    axis_results = {}
    replicated_jumps = []
    replicated_transitions = []
    trajectory_correlations = []
    for axis, points in protocol["feature_axes"].items():
        axis_results[axis] = {}
        for profile in profiles:
            condition_key = f"{profile}"
            per_block = {}
            for block_index, config in enumerate(configs):
                block_id = config["protocol"]["seed_block_id"]
                diagnostics = []
                for left, right in zip(points[:-1], points[1:]):
                    left_id = left["representation"]
                    right_id = right["representation"]
                    left_class = block_results[block_id]["conditions"][
                        left_id
                    ][profile]["classification"]["label"]
                    right_class = block_results[block_id]["conditions"][
                        right_id
                    ][profile]["classification"]["label"]
                    diagnostic = adjacent_diagnostic(
                        block_samples[block_index][left_id][profile][
                            "a_sin_1"
                        ],
                        block_samples[block_index][right_id][profile][
                            "a_sin_1"
                        ],
                        left_class,
                        right_class,
                        rules,
                    )
                    diagnostic.update(
                        {
                            "left_value": left["value"],
                            "right_value": right["value"],
                            "left_representation": left_id,
                            "right_representation": right_id,
                            "left_class": left_class,
                            "right_class": right_class,
                        }
                    )
                    diagnostics.append(diagnostic)
                per_block[block_id] = diagnostics
            correlations = {}
            for coefficient in ("a_sin_1", "a_sin_2", "a0"):
                first = [
                    block_results[protocols[0]["seed_block_id"]][
                        "conditions"
                    ][point["representation"]][profile]["estimate"][coefficient]
                    for point in points
                ]
                second = [
                    block_results[protocols[1]["seed_block_id"]][
                        "conditions"
                    ][point["representation"]][profile]["estimate"][coefficient]
                    for point in points
                ]
                correlation = float(np.corrcoef(first, second)[0, 1])
                correlations[coefficient] = correlation
                if (
                    coefficient in ("a_sin_1", "a_sin_2")
                    and np.isfinite(correlation)
                ):
                    trajectory_correlations.append(correlation)
            for pair_index in range(len(points) - 1):
                first_diagnostic = per_block[
                    protocols[0]["seed_block_id"]
                ][pair_index]
                second_diagnostic = per_block[
                    protocols[1]["seed_block_id"]
                ][pair_index]
                key = {
                    "axis": axis,
                    "profile": profile,
                    "left_value": points[pair_index]["value"],
                    "right_value": points[pair_index + 1]["value"],
                }
                if (
                    first_diagnostic["operational_jump"]
                    and second_diagnostic["operational_jump"]
                ):
                    replicated_jumps.append(key)
                if (
                    first_diagnostic["sign_transition"]
                    and second_diagnostic["sign_transition"]
                ):
                    replicated_transitions.append(key)
            axis_results[axis][condition_key] = {
                "block_diagnostics": per_block,
                "trajectory_correlations": correlations,
            }

    block_comparison = {}
    maximum_overdispersion = 0.0
    for identifier in identifiers:
        block_comparison[identifier] = {}
        for profile in profiles:
            ratio = pointwise_overdispersion_ratio(
                count_blocks[0][identifier][profile],
                count_blocks[1][identifier][profile],
            )
            maximum_overdispersion = max(maximum_overdispersion, ratio)
            block_comparison[identifier][profile] = {
                "pointwise_overdispersion_ratio": ratio,
                "a1_block2_minus_block1": {
                    **{
                        "median": float(
                            np.quantile(
                                block_samples[1][identifier][profile][
                                    "a_sin_1"
                                ]
                                - block_samples[0][identifier][profile][
                                    "a_sin_1"
                                ],
                                0.5,
                            )
                        ),
                        "ci95_low": float(
                            np.quantile(
                                block_samples[1][identifier][profile][
                                    "a_sin_1"
                                ]
                                - block_samples[0][identifier][profile][
                                    "a_sin_1"
                                ],
                                0.025,
                            )
                        ),
                        "ci95_high": float(
                            np.quantile(
                                block_samples[1][identifier][profile][
                                    "a_sin_1"
                                ]
                                - block_samples[0][identifier][profile][
                                    "a_sin_1"
                                ],
                                0.975,
                            )
                        ),
                    }
                },
            }
    mean_feature_trajectories = {}
    exploratory_axis_spans = []
    for axis, points in protocol["feature_axes"].items():
        mean_feature_trajectories[axis] = {}
        for profile in profiles:
            trajectory = []
            block_a1_values = {block_id: [] for block_id in block_results}
            for point in points:
                identifier = point["representation"]
                block_estimates = [
                    block_results[block_id]["conditions"][identifier][profile][
                        "estimate"
                    ]
                    for block_id in block_results
                ]
                for block_id, estimate in zip(
                    block_results,
                    block_estimates,
                ):
                    block_a1_values[block_id].append(
                        estimate["a_sin_1"]
                    )
                trajectory.append(
                    {
                        "value": point["value"],
                        "representation": identifier,
                        "a0": float(
                            np.mean(
                                [
                                    estimate["a0"]
                                    for estimate in block_estimates
                                ]
                            )
                        ),
                        "a1": float(
                            np.mean(
                                [
                                    estimate["a_sin_1"]
                                    for estimate in block_estimates
                                ]
                            )
                        ),
                        "a2": float(
                            np.mean(
                                [
                                    estimate["a_sin_2"]
                                    for estimate in block_estimates
                                ]
                            )
                        ),
                    }
                )
            spans = {
                block_id: float(max(values) - min(values))
                for block_id, values in block_a1_values.items()
            }
            mean_feature_trajectories[axis][profile] = trajectory
            exploratory_axis_spans.append(
                {
                    "axis": axis,
                    "profile": profile,
                    "block_a1_spans": spans,
                    "replicated_point_span_at_least_0.25": all(
                        span >= 0.25 for span in spans.values()
                    ),
                }
            )
    operational_pass = all(
        config["operational"]["passes"] for config in configs
    )
    minimum_correlation = min(trajectory_correlations)
    event_keys = {
        (event["axis"], event["profile"])
        for event in replicated_jumps + replicated_transitions
    }
    event_correlations = [
        axis_results[axis][profile]["trajectory_correlations"]["a_sin_1"]
        for axis, profile in event_keys
    ]
    minimum_event_correlation = (
        min(event_correlations) if event_correlations else None
    )
    overall_pass = (
        operational_pass
        and bool(replicated_jumps or replicated_transitions)
        and minimum_event_correlation is not None
        and minimum_event_correlation
        >= float(
            protocol["criteria"]["minimum_replication_trajectory_correlation"]
        )
    )
    report = {
        "protocol_name": protocol["protocol_name"],
        "feature_axes": protocol["feature_axes"],
        "classification_rules": rules,
        "bootstrap_pseudocount": pseudocount,
        "blocks": block_results,
        "axis_results": axis_results,
        "block_comparison": block_comparison,
        "mean_feature_trajectories": mean_feature_trajectories,
        "exploratory_axis_spans": exploratory_axis_spans,
        "replicated_operational_jumps": replicated_jumps,
        "replicated_sign_transitions": replicated_transitions,
        "minimum_feature_trajectory_correlation": minimum_correlation,
        "minimum_event_trajectory_correlation": minimum_event_correlation,
        "maximum_pointwise_overdispersion_ratio": maximum_overdispersion,
        "maximum_fourier_order_sensitivity_range": maximum_sensitivity,
        "operational_pass": operational_pass,
        "overall_feature_gate_pass": overall_pass,
    }
    if reference_analysis is not None:
        reference = json.loads(reference_analysis.read_text(encoding="utf-8"))
        report["reference_analysis_path"] = str(reference_analysis)
        report["reference_mean_trajectories"] = reference[
            "mean_trajectories"
        ]
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "feature_sweep_analysis.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "feature_sweep_endpoints.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--reference-analysis", type=Path)
    args = parser.parse_args()
    report = analyze(args.run_dir, args.out_dir, args.reference_analysis)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

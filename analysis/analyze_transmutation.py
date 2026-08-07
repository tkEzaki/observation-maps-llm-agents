"""Analyze two nonuniform-offset representation-transmutation seed blocks."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ACTION_INDEX = {"retard": 0, "stay": 1, "advance": 2}


def design_matrix(offsets: np.ndarray, order: int) -> np.ndarray:
    columns = [np.ones(offsets.size)]
    for harmonic in range(1, order + 1):
        columns.append(np.sin(harmonic * offsets))
        columns.append(np.cos(harmonic * offsets))
    return np.column_stack(columns)


def fit_fourier(
    counts: np.ndarray,
    offsets: np.ndarray,
    order: int,
) -> dict[str, float]:
    totals = np.sum(counts, axis=1)
    g = (counts[:, 2] - counts[:, 0]) / totals
    matrix = design_matrix(offsets, order)
    beta, _, _, _ = np.linalg.lstsq(matrix, g, rcond=None)
    result = {
        "a0": float(beta[0]),
        "mean_activity": float(
            np.mean((counts[:, 0] + counts[:, 2]) / totals)
        ),
        "design_condition_number": float(np.linalg.cond(matrix)),
    }
    for harmonic in range(1, order + 1):
        result[f"a_sin_{harmonic}"] = float(beta[2 * harmonic - 1])
        result[f"b_cos_{harmonic}"] = float(beta[2 * harmonic])
    return result


def bootstrap_condition(
    counts: np.ndarray,
    offsets: np.ndarray,
    order: int,
    samples: int,
    rng: np.random.Generator,
) -> tuple[dict[str, np.ndarray], dict]:
    totals = np.sum(counts, axis=1).astype(np.int64)
    probability = counts / totals[:, None]
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
    matrix = design_matrix(offsets, order)
    inverse = np.linalg.pinv(matrix)
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


def load_block(run_dir: Path) -> tuple[dict, dict[str, dict[str, np.ndarray]]]:
    config = json.loads(
        (run_dir / "resolved_config.json").read_text(encoding="utf-8")
    )
    protocol = config["protocol"]
    offsets = protocol["offset_radians"]
    n_offsets = len(offsets)
    if "stimulus_profiles" in protocol:
        profiles = list(protocol["stimulus_profiles"])
    else:
        profiles = list(protocol["concentration_profiles"])
    counts = {
        representation: {
            profile: np.zeros((n_offsets, 3), dtype=np.int64)
            for profile in profiles
        }
        for representation in protocol["representations"]
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
    minimum = int(protocol["criteria"]["minimum_valid_samples_per_condition"])
    cell_minimum = min(
        int(np.min(np.sum(profile_counts, axis=1)))
        for representation_counts in counts.values()
        for profile_counts in representation_counts.values()
    )
    operational = {
        "run_dir": str(run_dir),
        "seed_block_id": protocol["seed_block_id"],
        "n_records": total,
        "n_valid": valid,
        "valid_rate": valid / total,
        "minimum_cell_samples": cell_minimum,
        "passes": (
            valid / total >= float(protocol["criteria"]["minimum_valid_rate"])
            and cell_minimum >= minimum
        ),
    }
    return {**config, "operational": operational}, counts


def phenotype_probability(
    representation: str,
    condition_samples: dict[str, dict[str, np.ndarray]],
    rule: dict,
) -> float:
    broad = condition_samples["kappa_2"]
    narrow = condition_samples["kappa_12"]
    if representation == "intervals_24_decimal6":
        passing = (
            (broad["a_sin_1"] >= float(rule["minimum_a1_kappa_2"]))
            & (
                np.abs(narrow["a_sin_1"])
                <= float(rule["maximum_abs_a1_kappa_12"])
            )
            & (
                narrow["a_sin_2"] - broad["a_sin_2"]
                >= float(rule["minimum_a2_increment"])
            )
        )
    elif representation == "moments_m1_m3":
        passing = (
            narrow["a_sin_1"]
            >= float(rule["minimum_a1_kappa_12"])
        )
    elif representation == "centers_24_standard":
        passing = (
            narrow["a_sin_1"]
            <= float(rule["maximum_a1_kappa_12"])
        )
    else:
        raise ValueError(f"no phenotype rule for {representation}")
    return float(np.mean(passing))


def pointwise_overdispersion_ratio(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    n_first = np.sum(first, axis=1, keepdims=True)
    n_second = np.sum(second, axis=1, keepdims=True)
    pooled = (first + second) / (n_first + n_second)
    expected_first = n_first * pooled
    expected_second = n_second * pooled
    statistic = 0.0
    for observed, expected in (
        (first, expected_first),
        (second, expected_second),
    ):
        mask = expected > 0
        statistic += float(
            np.sum((observed[mask] - expected[mask]) ** 2 / expected[mask])
        )
    degrees = max(1, 2 * first.shape[0])
    return statistic / degrees


def _interval(values: np.ndarray) -> dict:
    return {
        "median": float(np.quantile(values, 0.5)),
        "ci95_low": float(np.quantile(values, 0.025)),
        "ci95_high": float(np.quantile(values, 0.975)),
        "compatible_with_zero": bool(
            np.quantile(values, 0.025) <= 0 <= np.quantile(values, 0.975)
        ),
    }


def analyze(
    run_dirs: list[Path],
    output_dir: Path,
    historical_analysis: Path | None = None,
) -> dict:
    if len(run_dirs) != 2:
        raise ValueError("exactly two seed-block run directories are required")
    loaded = [load_block(path) for path in run_dirs]
    configs = [item[0] for item in loaded]
    count_blocks = [item[1] for item in loaded]
    protocols = [config["protocol"] for config in configs]
    shared_keys = (
        "offset_radians",
        "concentration_profiles",
        "representations",
        "repetitions_per_condition",
        "primary_fourier_order",
        "sensitivity_fourier_orders",
        "phenotype_rules",
        "replication_rules",
    )
    for key in shared_keys:
        if protocols[0][key] != protocols[1][key]:
            raise ValueError(f"block protocols differ at {key}")
    protocol = protocols[0]
    offsets = np.asarray(protocol["offset_radians"], dtype=np.float64)
    order = int(protocol["primary_fourier_order"])
    samples = int(protocol["bootstrap_samples"])
    profile_order = sorted(
        protocol["concentration_profiles"],
        key=lambda profile: protocol["concentration_profiles"][profile],
    )
    block_results = {}
    bootstrap_values: list[dict] = []
    rows = []
    for block_index, (config, counts) in enumerate(loaded):
        block_id = config["protocol"]["seed_block_id"]
        rng = np.random.default_rng(
            2026122300 + block_index
        )
        block_results[block_id] = {
            "operational": config["operational"],
            "representations": {},
        }
        block_samples = {}
        for representation in protocol["representations"]:
            representation_result = {
                "conditions": {},
            }
            representation_samples = {}
            for profile in profile_order:
                cell_counts = counts[representation][profile]
                estimate = fit_fourier(cell_counts, offsets, order)
                values, interval = bootstrap_condition(
                    cell_counts,
                    offsets,
                    order,
                    samples,
                    rng,
                )
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
                sensitivity_a1 = [
                    result["a_sin_1"] for result in sensitivity.values()
                ]
                sensitivity_a2 = [
                    result["a_sin_2"] for result in sensitivity.values()
                ]
                representation_result["conditions"][profile] = {
                    "concentration": float(
                        protocol["concentration_profiles"][profile]
                    ),
                    "estimate": estimate,
                    "bootstrap": interval,
                    "sensitivity_orders": sensitivity,
                    "sensitivity_range_a1": float(
                        max(sensitivity_a1) - min(sensitivity_a1)
                    ),
                    "sensitivity_range_a2": float(
                        max(sensitivity_a2) - min(sensitivity_a2)
                    ),
                }
                representation_samples[profile] = values
                rows.append(
                    {
                        "block": block_id,
                        "representation": representation,
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
                        "a1_ci_low": interval["a_sin_1"]["ci95_low"],
                        "a1_ci_high": interval["a_sin_1"]["ci95_high"],
                        "a2_ci_low": interval["a_sin_2"]["ci95_low"],
                        "a2_ci_high": interval["a_sin_2"]["ci95_high"],
                    }
                )
            probability = phenotype_probability(
                representation,
                representation_samples,
                protocol["phenotype_rules"][representation],
            )
            representation_result["phenotype_probability"] = probability
            representation_result["phenotype_pass"] = (
                probability
                >= float(
                    protocol["replication_rules"][
                        "minimum_phenotype_probability"
                    ]
                )
            )
            block_results[block_id]["representations"][
                representation
            ] = representation_result
            block_samples[representation] = representation_samples
        bootstrap_values.append(block_samples)

    comparison = {}
    compatible_count = 0
    comparison_count = 0
    trajectory_pass_count = 0
    for representation in protocol["representations"]:
        representation_comparison = {
            "conditions": {},
            "trajectory_correlations": {},
        }
        for profile in profile_order:
            condition_comparison = {
                "pointwise_overdispersion_ratio": (
                    pointwise_overdispersion_ratio(
                        count_blocks[0][representation][profile],
                        count_blocks[1][representation][profile],
                    )
                )
            }
            for coefficient in ("a0", "a_sin_1", "a_sin_2"):
                difference = (
                    bootstrap_values[1][representation][profile][coefficient]
                    - bootstrap_values[0][representation][profile][coefficient]
                )
                condition_comparison[
                    f"{coefficient}_block2_minus_block1"
                ] = _interval(difference)
                comparison_count += 1
                compatible_count += int(
                    condition_comparison[
                        f"{coefficient}_block2_minus_block1"
                    ]["compatible_with_zero"]
                )
            representation_comparison["conditions"][
                profile
            ] = condition_comparison
        for coefficient in ("a_sin_1", "a_sin_2"):
            first = [
                block_results[protocols[0]["seed_block_id"]][
                    "representations"
                ][representation]["conditions"][profile]["estimate"][
                    coefficient
                ]
                for profile in profile_order
            ]
            second = [
                block_results[protocols[1]["seed_block_id"]][
                    "representations"
                ][representation]["conditions"][profile]["estimate"][
                    coefficient
                ]
                for profile in profile_order
            ]
            correlation = float(np.corrcoef(first, second)[0, 1])
            representation_comparison["trajectory_correlations"][
                coefficient
            ] = correlation
        if min(
            representation_comparison["trajectory_correlations"].values()
        ) >= float(
            protocol["replication_rules"]["minimum_trajectory_correlation"]
        ):
            trajectory_pass_count += 1
        comparison[representation] = representation_comparison

    replicated_classes = [
        representation
        for representation in protocol["representations"]
        if all(
            block_results[block_id]["representations"][representation][
                "phenotype_pass"
            ]
            for block_id in block_results
        )
    ]
    operational_pass = all(
        config["operational"]["passes"] for config in configs
    )
    mean_trajectories = {}
    for representation in protocol["representations"]:
        mean_trajectories[representation] = {}
        for profile in profile_order:
            mean_trajectories[representation][profile] = {
                coefficient: float(
                    np.mean(
                        [
                            block_results[block_id]["representations"][
                                representation
                            ]["conditions"][profile]["estimate"][coefficient]
                            for block_id in block_results
                        ]
                    )
                )
                for coefficient in (
                    "a0",
                    "a_sin_1",
                    "a_sin_2",
                    "b_cos_1",
                    "b_cos_2",
                    "mean_activity",
                )
            }
    maximum_sensitivity_range = {
        coefficient: max(
            block_results[block_id]["representations"][representation][
                "conditions"
            ][profile][f"sensitivity_range_{coefficient}"]
            for block_id in block_results
            for representation in protocol["representations"]
            for profile in profile_order
        )
        for coefficient in ("a1", "a2")
    }
    maximum_overdispersion_ratio = max(
        condition["pointwise_overdispersion_ratio"]
        for representation in comparison.values()
        for condition in representation["conditions"].values()
    )
    overall_pass = (
        operational_pass
        and len(replicated_classes)
        >= int(protocol["replication_rules"]["minimum_classes_replicated"])
        and trajectory_pass_count
        >= int(protocol["replication_rules"]["minimum_classes_replicated"])
    )
    report = {
        "protocol_name": protocol["protocol_name"],
        "offset_design": protocol["offset_design"],
        "primary_fourier_order": order,
        "sensitivity_fourier_orders": protocol[
            "sensitivity_fourier_orders"
        ],
        "blocks": block_results,
        "block_comparison": comparison,
        "mean_trajectories": mean_trajectories,
        "replicated_phenotype_classes": replicated_classes,
        "trajectory_pass_count": trajectory_pass_count,
        "coefficient_compatibility_fraction": (
            compatible_count / comparison_count
        ),
        "maximum_pointwise_overdispersion_ratio": (
            maximum_overdispersion_ratio
        ),
        "maximum_fourier_order_sensitivity_range": (
            maximum_sensitivity_range
        ),
        "operational_pass": operational_pass,
        "overall_replication_gate_pass": overall_pass,
    }
    if historical_analysis is not None:
        report["historical_analysis_path"] = str(historical_analysis)
        report["historical_analysis"] = json.loads(
            historical_analysis.read_text(encoding="utf-8")
        )["results"]
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "transmutation_analysis.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "transmutation_endpoints.csv").open(
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
    parser.add_argument(
        "--run-dir",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--historical-analysis", type=Path)
    args = parser.parse_args()
    report = analyze(
        args.run_dir,
        args.out_dir,
        args.historical_analysis,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

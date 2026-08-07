"""Analyze a completed 36-point representation-invariance grid."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from analyze_representation_screen import (
    ACTION_INDEX,
    bootstrap_variant,
    coefficients,
)


def _grid_index(offset: float, grid_size: int) -> int:
    coordinate = (offset + np.pi) * grid_size / (2.0 * np.pi)
    return int(round(coordinate)) % grid_size


def load_representation_counts(
    run_dirs: list[Path],
    *,
    grid_size: int,
) -> dict[str, dict[str, np.ndarray]]:
    grouped: dict[str, dict[str, dict[int, np.ndarray]]] = {}
    for run_dir in run_dirs:
        with (run_dir / "trace.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                if not record.get("valid"):
                    continue
                representation = record["representation"]
                profile = record["profile"]
                index = _grid_index(float(record["offset_radians"]), grid_size)
                counts = (
                    grouped
                    .setdefault(representation, {})
                    .setdefault(profile, {})
                    .setdefault(index, np.zeros(3, dtype=np.int64))
                )
                counts[ACTION_INDEX[record["action_label"]]] += 1
    result = {}
    expected = set(range(grid_size))
    for representation, profiles in grouped.items():
        result[representation] = {}
        for profile, offsets in profiles.items():
            if set(offsets) != expected:
                missing = sorted(expected - set(offsets))
                raise ValueError(
                    f"{representation}/{profile} missing grid indices {missing}"
                )
            result[representation][profile] = np.stack(
                [offsets[index] for index in range(grid_size)]
            )
    return result


def load_full_baseline(
    run_dirs: list[Path],
) -> dict[str, np.ndarray]:
    grouped: dict[str, dict[int, np.ndarray]] = {}
    for run_dir in run_dirs:
        with (run_dir / "trace.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                if not record.get("valid"):
                    continue
                profile = record["profile"]
                index = int(record["offset_index"])
                counts = (
                    grouped
                    .setdefault(profile, {})
                    .setdefault(index, np.zeros(3, dtype=np.int64))
                )
                counts[ACTION_INDEX[record["action_label"]]] += 1
    return {
        profile: np.stack([offsets[index] for index in range(48)])
        for profile, offsets in grouped.items()
    }


def analyze(
    representation_dirs: list[Path],
    baseline_dirs: list[Path],
    output_dir: Path,
) -> dict:
    config = json.loads(
        (representation_dirs[0] / "resolved_config.json").read_text(
            encoding="utf-8"
        )
    )
    protocol = config["protocol"]
    rule = protocol["qualitative_switching_rule"]
    samples = int(protocol["bootstrap_samples"])
    grid_size = 36
    counts = load_representation_counts(
        representation_dirs,
        grid_size=grid_size,
    )
    counts["baseline_intervals_24_decimal6"] = load_full_baseline(
        baseline_dirs
    )
    results = {}
    rows = []
    for variant_index, variant in enumerate(sorted(counts)):
        broad = counts[variant]["broad_kappa_2"]
        narrow = counts[variant]["narrow_kappa_12"]
        n_offsets = broad.shape[0]
        delta = -np.pi + np.arange(n_offsets) * 2.0 * np.pi / n_offsets
        broad_coefficient = coefficients(broad, delta)
        narrow_coefficient = coefficients(narrow, delta)
        bootstrap = bootstrap_variant(
            broad,
            narrow,
            delta,
            rule,
            samples=samples,
            seed=20260923 + variant_index,
        )
        result = {
            "n_offsets": n_offsets,
            "n_per_condition": int(np.sum(broad[0])),
            "broad": broad_coefficient,
            "narrow": narrow_coefficient,
            "narrow_minus_broad_a2": (
                narrow_coefficient["a_sin_2"]
                - broad_coefficient["a_sin_2"]
            ),
            "bootstrap": bootstrap,
        }
        results[variant] = result
        rows.append(
            {
                "representation": variant,
                "n_offsets": n_offsets,
                "broad_a1": broad_coefficient["a_sin_1"],
                "narrow_a1": narrow_coefficient["a_sin_1"],
                "broad_a2": broad_coefficient["a_sin_2"],
                "narrow_a2": narrow_coefficient["a_sin_2"],
                "narrow_minus_broad_a2": (
                    result["narrow_minus_broad_a2"]
                ),
                "broad_a0": broad_coefficient["a0"],
                "narrow_a0": narrow_coefficient["a0"],
                "broad_activity": broad_coefficient["mean_activity"],
                "narrow_activity": narrow_coefficient["mean_activity"],
                "switching_probability": (
                    bootstrap["switching_rule_probability"]
                ),
                "passes": bootstrap["passes_probability_gate"],
            }
        )
    variants = protocol["representations"]
    passing = [
        variant
        for variant in variants
        if results[variant]["bootstrap"]["passes_probability_gate"]
    ]
    report = {
        "representation_runs": [str(path) for path in representation_dirs],
        "baseline_runs": [str(path) for path in baseline_dirs],
        "representation_grid_size": grid_size,
        "baseline_grid_size": 48,
        "primary_rule": rule,
        "results": results,
        "new_variants_passing": passing,
        "new_variant_count": len(variants),
        "passing_count": len(passing),
        "all_new_variants_pass": len(passing) == len(variants),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "representation_grid_analysis.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "representation_grid_endpoints.csv").open(
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
        "--representation-run",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument(
        "--baseline-run",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(
        args.representation_run,
        args.baseline_run,
        args.out_dir,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

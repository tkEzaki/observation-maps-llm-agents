"""Analyze low-order harmonic invariance across observation representations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ACTION_INDEX = {"retard": 0, "stay": 1, "advance": 2}


def load_new_counts(
    run_dir: Path,
) -> dict[str, dict[str, np.ndarray]]:
    grouped: dict[str, dict[str, dict[int, np.ndarray]]] = {}
    with (run_dir / "trace.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if not record.get("valid"):
                continue
            representation = record["representation"]
            profile = record["profile"]
            offset_index = int(record["offset_index"])
            counts = (
                grouped
                .setdefault(representation, {})
                .setdefault(profile, {})
                .setdefault(offset_index, np.zeros(3, dtype=np.int64))
            )
            counts[ACTION_INDEX[record["action_label"]]] += 1
    return {
        representation: {
            profile: np.stack(
                [offsets[index] for index in sorted(offsets)]
            )
            for profile, offsets in profiles.items()
        }
        for representation, profiles in grouped.items()
    }


def load_baseline_counts(run_dirs: list[Path]) -> dict[str, np.ndarray]:
    grouped: dict[str, dict[int, np.ndarray]] = {}
    for run_dir in run_dirs:
        with (run_dir / "trace.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                if not record.get("valid"):
                    continue
                original_index = int(record["offset_index"])
                if original_index % 4:
                    continue
                profile = record["profile"]
                offset_index = original_index // 4
                counts = (
                    grouped
                    .setdefault(profile, {})
                    .setdefault(offset_index, np.zeros(3, dtype=np.int64))
                )
                counts[ACTION_INDEX[record["action_label"]]] += 1
    return {
        profile: np.stack([offsets[index] for index in sorted(offsets)])
        for profile, offsets in grouped.items()
    }


def coefficients(counts: np.ndarray, delta: np.ndarray) -> dict[str, float]:
    n = np.sum(counts, axis=1)
    g = (counts[:, 2] - counts[:, 0]) / n
    return {
        "a0": float(np.mean(g)),
        "a_sin_1": float(2.0 * np.mean(g * np.sin(delta))),
        "b_cos_1": float(2.0 * np.mean(g * np.cos(delta))),
        "a_sin_2": float(2.0 * np.mean(g * np.sin(2.0 * delta))),
        "b_cos_2": float(2.0 * np.mean(g * np.cos(2.0 * delta))),
        "mean_activity": float(
            np.mean((counts[:, 0] + counts[:, 2]) / n)
        ),
    }


def bootstrap_variant(
    broad: np.ndarray,
    narrow: np.ndarray,
    delta: np.ndarray,
    rule: dict,
    *,
    samples: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    broad_probability = broad / np.sum(broad, axis=1, keepdims=True)
    narrow_probability = narrow / np.sum(narrow, axis=1, keepdims=True)
    n_broad = int(np.sum(broad[0]))
    n_narrow = int(np.sum(narrow[0]))
    values = np.empty((samples, 4))
    for sample_index in range(samples):
        broad_draw = np.stack(
            [rng.multinomial(n_broad, p) for p in broad_probability]
        )
        narrow_draw = np.stack(
            [rng.multinomial(n_narrow, p) for p in narrow_probability]
        )
        broad_coefficient = coefficients(broad_draw, delta)
        narrow_coefficient = coefficients(narrow_draw, delta)
        values[sample_index] = [
            broad_coefficient["a_sin_1"],
            narrow_coefficient["a_sin_1"],
            broad_coefficient["a_sin_2"],
            narrow_coefficient["a_sin_2"],
        ]
    names = ("broad_a1", "narrow_a1", "broad_a2", "narrow_a2")
    intervals = {
        name: {
            "median": float(np.quantile(values[:, index], 0.5)),
            "ci95_low": float(np.quantile(values[:, index], 0.025)),
            "ci95_high": float(np.quantile(values[:, index], 0.975)),
        }
        for index, name in enumerate(names)
    }
    switching = (
        (values[:, 0] >= float(rule["minimum_broad_a1"]))
        & (
            np.abs(values[:, 1])
            <= float(rule["maximum_abs_narrow_a1"])
        )
        & (values[:, 2] > float(rule["minimum_broad_a2"]))
        & (
            values[:, 3] - values[:, 2]
            >= float(rule["minimum_narrow_minus_broad_a2"])
        )
    )
    probability = float(np.mean(switching))
    return {
        "samples": samples,
        "coefficient_intervals": intervals,
        "switching_rule_probability": probability,
        "passes_probability_gate": (
            probability >= float(rule["minimum_bootstrap_probability"])
        ),
    }


def analyze(
    run_dir: Path,
    baseline_dirs: list[Path],
    output_dir: Path,
) -> dict:
    config = json.loads(
        (run_dir / "resolved_config.json").read_text(encoding="utf-8")
    )
    protocol = config["protocol"]
    counts = load_new_counts(run_dir)
    counts["baseline_intervals_24_decimal6"] = load_baseline_counts(
        baseline_dirs
    )
    delta = -np.pi + np.arange(12) * 2.0 * np.pi / 12
    rule = protocol["qualitative_switching_rule"]
    samples = int(protocol["bootstrap_samples"])
    results = {}
    rows = []
    for variant_index, variant in enumerate(sorted(counts)):
        broad = counts[variant]["broad_kappa_2"]
        narrow = counts[variant]["narrow_kappa_12"]
        broad_coefficient = coefficients(broad, delta)
        narrow_coefficient = coefficients(narrow, delta)
        bootstrap = bootstrap_variant(
            broad,
            narrow,
            delta,
            rule,
            samples=samples,
            seed=20260823 + variant_index,
        )
        results[variant] = {
            "n_per_condition_broad": int(np.sum(broad[0])),
            "n_per_condition_narrow": int(np.sum(narrow[0])),
            "broad": broad_coefficient,
            "narrow": narrow_coefficient,
            "narrow_minus_broad_a2": (
                narrow_coefficient["a_sin_2"]
                - broad_coefficient["a_sin_2"]
            ),
            "bootstrap": bootstrap,
        }
        rows.append(
            {
                "representation": variant,
                "broad_a1": broad_coefficient["a_sin_1"],
                "narrow_a1": narrow_coefficient["a_sin_1"],
                "broad_a2": broad_coefficient["a_sin_2"],
                "narrow_a2": narrow_coefficient["a_sin_2"],
                "narrow_minus_broad_a2": (
                    narrow_coefficient["a_sin_2"]
                    - broad_coefficient["a_sin_2"]
                ),
                "broad_activity": broad_coefficient["mean_activity"],
                "narrow_activity": narrow_coefficient["mean_activity"],
                "switching_probability": (
                    bootstrap["switching_rule_probability"]
                ),
                "passes": bootstrap["passes_probability_gate"],
            }
        )
    new_variants = protocol["representations"]
    passing = [
        variant
        for variant in new_variants
        if results[variant]["bootstrap"]["passes_probability_gate"]
    ]
    report = {
        "representation_run": str(run_dir),
        "baseline_runs": [str(path) for path in baseline_dirs],
        "primary_rule": rule,
        "results": results,
        "new_variants_passing": passing,
        "new_variant_count": len(new_variants),
        "passing_count": len(passing),
        "all_new_variants_pass": len(passing) == len(new_variants),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "representation_analysis.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "representation_endpoints.csv").open(
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
    parser.add_argument("representation_run", type=Path)
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

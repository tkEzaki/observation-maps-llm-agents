"""Compare two independent Stage B response-law measurements."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load_curve(run_dir: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    with (run_dir / "response_curve.csv").open(
        newline="",
        encoding="utf-8",
    ) as handle:
        rows = list(csv.DictReader(handle))
    result = {}
    for profile in sorted({row["profile"] for row in rows}):
        selected = [row for row in rows if row["profile"] == profile]
        delta = np.asarray(
            [float(row["offset_radians"]) for row in selected],
            dtype=np.float64,
        )
        g = np.asarray(
            [float(row["g_mean"]) for row in selected],
            dtype=np.float64,
        )
        result[profile] = (delta, g)
    return result


def load_actions(
    run_dir: Path,
) -> dict[str, dict[float, np.ndarray]]:
    grouped: dict[str, dict[float, list[float]]] = {}
    with (run_dir / "trace.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if not record.get("valid"):
                continue
            profile = record["profile"]
            offset = float(record["offset_radians"])
            grouped.setdefault(profile, {}).setdefault(offset, []).append(
                float(record["action_value"])
            )
    return {
        profile: {
            offset: np.asarray(values, dtype=np.float64)
            for offset, values in offsets.items()
        }
        for profile, offsets in grouped.items()
    }


def fourier_coefficients(
    delta: np.ndarray,
    g: np.ndarray,
    max_harmonic: int,
) -> tuple[list[str], np.ndarray]:
    names = ["a0"]
    values = [float(np.mean(g))]
    for harmonic in range(1, max_harmonic + 1):
        names.extend([f"a_sin_{harmonic}", f"b_cos_{harmonic}"])
        values.extend(
            [
                float(2.0 * np.mean(g * np.sin(harmonic * delta))),
                float(2.0 * np.mean(g * np.cos(harmonic * delta))),
            ]
        )
    return names, np.asarray(values)


def bootstrap_fourier_difference(
    first: dict[float, np.ndarray],
    second: dict[float, np.ndarray],
    *,
    max_harmonic: int,
    n_bootstrap: int,
    seed: int,
) -> dict:
    offsets = np.asarray(sorted(first))
    if set(first) != set(second):
        raise ValueError("runs do not contain identical offset grids")
    rng = np.random.default_rng(seed)
    names = None
    differences = []
    for _ in range(n_bootstrap):
        g_first = np.asarray(
            [
                np.mean(rng.choice(first[offset], size=first[offset].size))
                for offset in offsets
            ]
        )
        g_second = np.asarray(
            [
                np.mean(rng.choice(second[offset], size=second[offset].size))
                for offset in offsets
            ]
        )
        names, coefficient_first = fourier_coefficients(
            offsets,
            g_first,
            max_harmonic,
        )
        _, coefficient_second = fourier_coefficients(
            offsets,
            g_second,
            max_harmonic,
        )
        differences.append(coefficient_second - coefficient_first)
    samples = np.asarray(differences)
    intervals = []
    for index, name in enumerate(names or []):
        low, median, high = np.quantile(
            samples[:, index],
            [0.025, 0.5, 0.975],
        )
        intervals.append(
            {
                "coefficient": name,
                "difference_replication_minus_first": float(median),
                "ci95_low": float(low),
                "ci95_high": float(high),
                "compatible_with_zero_difference": bool(low <= 0 <= high),
            }
        )
    return {
        "n_bootstrap": n_bootstrap,
        "intervals": intervals,
        "compatible_count": sum(
            item["compatible_with_zero_difference"] for item in intervals
        ),
        "coefficient_count": len(intervals),
    }


def compare_runs(
    first_dir: Path,
    second_dir: Path,
    *,
    max_harmonic: int = 6,
    n_bootstrap: int = 2000,
) -> dict:
    curves_first = load_curve(first_dir)
    curves_second = load_curve(second_dir)
    actions_first = load_actions(first_dir)
    actions_second = load_actions(second_dir)
    if set(curves_first) != set(curves_second):
        raise ValueError("runs do not contain identical profiles")

    profiles = {}
    for profile_index, profile in enumerate(sorted(curves_first)):
        delta_first, g_first = curves_first[profile]
        delta_second, g_second = curves_second[profile]
        np.testing.assert_allclose(delta_first, delta_second, atol=0, rtol=0)
        correlation = float(np.corrcoef(g_first, g_second)[0, 1])
        rmse = float(np.sqrt(np.mean((g_second - g_first) ** 2)))
        mae = float(np.mean(np.abs(g_second - g_first)))
        informative = (np.abs(g_first) >= 0.2) & (np.abs(g_second) >= 0.2)
        directional_agreement = (
            float(np.mean(np.sign(g_first[informative]) == np.sign(g_second[informative])))
            if np.any(informative)
            else None
        )
        names, coefficients_first = fourier_coefficients(
            delta_first,
            g_first,
            max_harmonic,
        )
        _, coefficients_second = fourier_coefficients(
            delta_second,
            g_second,
            max_harmonic,
        )
        pooled_g = (g_first + g_second) / 2.0
        _, coefficients_pooled = fourier_coefficients(
            delta_first,
            pooled_g,
            max_harmonic,
        )
        mirrored_indices = [
            int(
                np.argmin(
                    np.abs(
                        (delta_first + offset + np.pi) % (2.0 * np.pi)
                        - np.pi
                    )
                )
            )
            for offset in delta_first
        ]
        mirrored = pooled_g[mirrored_indices]
        pooled_odd = (pooled_g - mirrored) / 2.0
        pooled_even = (pooled_g + mirrored) / 2.0
        bootstrap = bootstrap_fourier_difference(
            actions_first[profile],
            actions_second[profile],
            max_harmonic=max_harmonic,
            n_bootstrap=n_bootstrap,
            seed=20260723 + profile_index,
        )
        profiles[profile] = {
            "curve_correlation": correlation,
            "rmse": rmse,
            "mean_absolute_difference": mae,
            "directional_agreement_for_abs_g_at_least_0_2": (
                directional_agreement
            ),
            "informative_offset_count": int(np.sum(informative)),
            "fourier_first": dict(zip(names, coefficients_first.tolist())),
            "fourier_replication": dict(
                zip(names, coefficients_second.tolist())
            ),
            "pooled_n_per_condition": 100,
            "pooled_fourier": dict(
                zip(names, coefficients_pooled.tolist())
            ),
            "pooled_structure": {
                "correlation_with_sin_delta": float(
                    np.corrcoef(pooled_g, np.sin(delta_first))[0, 1]
                ),
                "mean_absolute_adjacent_change": float(
                    np.mean(np.abs(np.roll(pooled_g, -1) - pooled_g))
                ),
                "maximum_adjacent_change": float(
                    np.max(np.abs(np.roll(pooled_g, -1) - pooled_g))
                ),
                "odd_rms": float(np.sqrt(np.mean(pooled_odd**2))),
                "even_rms": float(np.sqrt(np.mean(pooled_even**2))),
            },
            "fourier_bootstrap_difference": bootstrap,
            "curve_correlation_gate_at_least_0_8": correlation >= 0.8,
        }

    return {
        "first_run": str(first_dir),
        "replication_run": str(second_dir),
        "profiles": profiles,
        "primary_replication_gate": {
            "rule": "curve correlation >= 0.8 for every profile",
            "passed": all(
                result["curve_correlation_gate_at_least_0_8"]
                for result in profiles.values()
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first_run", type=Path)
    parser.add_argument("replication_run", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    report = compare_runs(
        args.first_run,
        args.replication_run,
        n_bootstrap=args.bootstrap,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

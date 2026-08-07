"""Extended Stage B diagnostics requested before representation experiments."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ACTION_INDEX = {"retard": 0, "stay": 1, "advance": 2}


def load_counts(run_dir: Path) -> dict[str, dict[float, np.ndarray]]:
    grouped: dict[str, dict[float, np.ndarray]] = {}
    with (run_dir / "trace.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if not record.get("valid"):
                continue
            profile = record["profile"]
            offset = float(record["offset_radians"])
            counts = grouped.setdefault(profile, {}).setdefault(
                offset,
                np.zeros(3, dtype=np.int64),
            )
            counts[ACTION_INDEX[record["action_label"]]] += 1
    return grouped


def mean_and_variance(counts: np.ndarray) -> tuple[float, float]:
    n = int(np.sum(counts))
    mean = float((counts[2] - counts[0]) / n)
    activity = float((counts[0] + counts[2]) / n)
    variance = n / (n - 1) * (activity - mean**2) if n > 1 else 0.0
    return mean, variance


def harmonic_coefficients(
    delta: np.ndarray,
    values: np.ndarray,
    max_harmonic: int,
) -> dict[str, float]:
    result = {"a0": float(np.mean(values))}
    for harmonic in range(1, max_harmonic + 1):
        result[f"a_sin_{harmonic}"] = float(
            2.0 * np.mean(values * np.sin(harmonic * delta))
        )
        result[f"b_cos_{harmonic}"] = float(
            2.0 * np.mean(values * np.cos(harmonic * delta))
        )
    return result


def sample_from_probabilities(
    rng: np.random.Generator,
    probabilities: np.ndarray,
    sample_size: int,
) -> np.ndarray:
    return np.asarray(
        [rng.multinomial(sample_size, p) for p in probabilities],
        dtype=np.int64,
    )


def vectors_from_counts(
    counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n = np.sum(counts, axis=1)
    mean = (counts[:, 2] - counts[:, 0]) / n
    activity = (counts[:, 0] + counts[:, 2]) / n
    variance = n / (n - 1) * (activity - mean**2)
    return mean, variance


def standardized_replication_test(
    first: np.ndarray,
    second: np.ndarray,
    *,
    simulations: int,
    seed: int,
) -> dict:
    n_first = int(np.sum(first[0]))
    n_second = int(np.sum(second[0]))
    g_first, variance_first = vectors_from_counts(first)
    g_second, variance_second = vectors_from_counts(second)
    denominator = variance_first / n_first + variance_second / n_second
    observed_terms = np.divide(
        (g_first - g_second) ** 2,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    observed_q = float(np.sum(observed_terms))
    observed_correlation = float(np.corrcoef(g_first, g_second)[0, 1])
    pooled_probabilities = (first + second) / (
        np.sum(first, axis=1, keepdims=True)
        + np.sum(second, axis=1, keepdims=True)
    )

    rng = np.random.default_rng(seed)
    simulated_q = np.empty(simulations)
    simulated_correlation = np.empty(simulations)
    for index in range(simulations):
        draw_first = sample_from_probabilities(
            rng,
            pooled_probabilities,
            n_first,
        )
        draw_second = sample_from_probabilities(
            rng,
            pooled_probabilities,
            n_second,
        )
        mean_first, var_first = vectors_from_counts(draw_first)
        mean_second, var_second = vectors_from_counts(draw_second)
        draw_denominator = var_first / n_first + var_second / n_second
        terms = np.divide(
            (mean_first - mean_second) ** 2,
            draw_denominator,
            out=np.zeros_like(draw_denominator),
            where=draw_denominator > 0,
        )
        simulated_q[index] = np.sum(terms)
        simulated_correlation[index] = np.corrcoef(
            mean_first,
            mean_second,
        )[0, 1]

    correlation_interval = np.quantile(
        simulated_correlation,
        [0.025, 0.5, 0.975],
    )
    return {
        "q_observed": observed_q,
        "q_null_mean": float(np.mean(simulated_q)),
        "q_parametric_bootstrap_p": float(
            (np.sum(simulated_q >= observed_q) + 1) / (simulations + 1)
        ),
        "curve_correlation_observed": observed_correlation,
        "curve_correlation_null_95_interval": (
            correlation_interval.tolist()
        ),
        "curve_correlation_percentile_under_same_law": float(
            np.mean(simulated_correlation <= observed_correlation)
        ),
        "simulations": simulations,
    }


def von_mises_moment(concentration: float, harmonic: int) -> float:
    angle = np.linspace(-np.pi, np.pi, 262_144, endpoint=False)
    weight = np.exp(concentration * (np.cos(angle) - 1.0))
    return float(
        np.sum(weight * np.cos(harmonic * angle)) / np.sum(weight)
    )


def bootstrap_scaling_test(
    broad_counts: np.ndarray,
    narrow_counts: np.ndarray,
    delta: np.ndarray,
    *,
    harmonic: int,
    scaling: float,
    simulations: int,
    seed: int,
) -> dict:
    broad_probabilities = broad_counts / np.sum(
        broad_counts,
        axis=1,
        keepdims=True,
    )
    narrow_probabilities = narrow_counts / np.sum(
        narrow_counts,
        axis=1,
        keepdims=True,
    )
    n_broad = int(np.sum(broad_counts[0]))
    n_narrow = int(np.sum(narrow_counts[0]))
    broad_g, _ = vectors_from_counts(broad_counts)
    narrow_g, _ = vectors_from_counts(narrow_counts)
    broad_coefficient = harmonic_coefficients(
        delta,
        broad_g,
        harmonic,
    )[f"a_sin_{harmonic}"]
    narrow_coefficient = harmonic_coefficients(
        delta,
        narrow_g,
        harmonic,
    )[f"a_sin_{harmonic}"]
    observed_difference = narrow_coefficient - scaling * broad_coefficient

    rng = np.random.default_rng(seed)
    differences = np.empty(simulations)
    for index in range(simulations):
        broad_draw = sample_from_probabilities(
            rng,
            broad_probabilities,
            n_broad,
        )
        narrow_draw = sample_from_probabilities(
            rng,
            narrow_probabilities,
            n_narrow,
        )
        broad_sample, _ = vectors_from_counts(broad_draw)
        narrow_sample, _ = vectors_from_counts(narrow_draw)
        broad_value = harmonic_coefficients(
            delta,
            broad_sample,
            harmonic,
        )[f"a_sin_{harmonic}"]
        narrow_value = harmonic_coefficients(
            delta,
            narrow_sample,
            harmonic,
        )[f"a_sin_{harmonic}"]
        differences[index] = narrow_value - scaling * broad_value
    interval = np.quantile(differences, [0.025, 0.5, 0.975])
    return {
        "harmonic": harmonic,
        "broad_observed": broad_coefficient,
        "fixed_kernel_scaling": scaling,
        "narrow_fixed_kernel_prediction": scaling * broad_coefficient,
        "narrow_observed": narrow_coefficient,
        "observed_minus_prediction": observed_difference,
        "bootstrap_difference_95_interval": interval.tolist(),
        "fixed_kernel_prediction_compatible": bool(
            interval[0] <= 0 <= interval[2]
        ),
    }


def parity_and_spectrum(
    pooled_counts: np.ndarray,
    delta: np.ndarray,
    *,
    simulations: int,
    seed: int,
) -> tuple[dict, list[dict]]:
    probabilities = pooled_counts / np.sum(
        pooled_counts,
        axis=1,
        keepdims=True,
    )
    sample_size = int(np.sum(pooled_counts[0]))
    g, _ = vectors_from_counts(pooled_counts)
    parity_vector = (-1.0) ** np.arange(g.size)
    parity_coefficient = float(np.mean(g * parity_vector))
    one_step = float(np.mean(np.abs(np.roll(g, -1) - g)))
    two_step = float(np.mean(np.abs(np.roll(g, -2) - g)))

    rng = np.random.default_rng(seed)
    parity_samples = np.empty(simulations)
    for index in range(simulations):
        draw = sample_from_probabilities(rng, probabilities, sample_size)
        sampled_g, _ = vectors_from_counts(draw)
        parity_samples[index] = np.mean(sampled_g * parity_vector)
    parity_interval = np.quantile(parity_samples, [0.025, 0.5, 0.975])

    transform = np.fft.rfft(g) / g.size
    amplitudes = np.abs(transform)
    amplitudes[1:-1] *= 2.0
    spectrum = [
        {
            "harmonic": harmonic,
            "amplitude": float(amplitudes[harmonic]),
            "real": float(transform[harmonic].real),
            "imaginary": float(transform[harmonic].imag),
        }
        for harmonic in range(amplitudes.size)
    ]
    top_nonzero = sorted(
        spectrum[1:],
        key=lambda item: item["amplitude"],
        reverse=True,
    )[:8]
    return (
        {
            "parity_coefficient_mean_g_times_minus1_q": parity_coefficient,
            "parity_coefficient_95_interval": parity_interval.tolist(),
            "nyquist_m24_amplitude": float(amplitudes[-1]),
            "mean_absolute_one_point_difference": one_step,
            "mean_absolute_two_point_difference": two_step,
            "two_point_difference_per_grid_step": two_step / 2.0,
            "top_nonzero_harmonics": top_nonzero,
        },
        spectrum,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first_run", type=Path)
    parser.add_argument("replication_run", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--simulations", type=int, default=10_000)
    args = parser.parse_args()

    first = load_counts(args.first_run)
    second = load_counts(args.replication_run)
    if set(first) != set(second):
        raise ValueError("profile sets differ between runs")

    profiles = {}
    spectra = []
    probability_rows = []
    pooled_by_profile = {}
    delta_by_profile = {}
    for profile_index, profile in enumerate(sorted(first)):
        offsets = np.asarray(sorted(first[profile]))
        if set(first[profile]) != set(second[profile]):
            raise ValueError(f"offset grids differ for {profile}")
        first_counts = np.stack([first[profile][x] for x in offsets])
        second_counts = np.stack([second[profile][x] for x in offsets])
        pooled_counts = first_counts + second_counts
        pooled_by_profile[profile] = pooled_counts
        delta_by_profile[profile] = offsets
        pooled_probabilities = pooled_counts / np.sum(
            pooled_counts,
            axis=1,
            keepdims=True,
        )
        pooled_g, _ = vectors_from_counts(pooled_counts)
        pooled_activity = pooled_probabilities[:, 0] + pooled_probabilities[:, 2]
        parity, spectrum = parity_and_spectrum(
            pooled_counts,
            offsets,
            simulations=args.simulations,
            seed=20260750 + profile_index,
        )
        replication = standardized_replication_test(
            first_counts,
            second_counts,
            simulations=args.simulations,
            seed=20260760 + profile_index,
        )
        profiles[profile] = {
            "standardized_replication": replication,
            "parity_and_spectrum": parity,
            "activity": {
                "mean": float(np.mean(pooled_activity)),
                "minimum": float(np.min(pooled_activity)),
                "maximum": float(np.max(pooled_activity)),
                "at_delta_zero": float(
                    pooled_activity[np.argmin(np.abs(offsets))]
                ),
                "at_delta_pi": float(pooled_activity[0]),
                "harmonics_through_6": harmonic_coefficients(
                    offsets,
                    pooled_activity,
                    6,
                ),
            },
        }
        for item in spectrum:
            spectra.append({"profile": profile, **item})
        for index, offset in enumerate(offsets):
            probability_rows.append(
                {
                    "profile": profile,
                    "offset_radians": float(offset),
                    "p_retard": float(pooled_probabilities[index, 0]),
                    "p_stay": float(pooled_probabilities[index, 1]),
                    "p_advance": float(pooled_probabilities[index, 2]),
                    "g": float(pooled_g[index]),
                    "activity": float(pooled_activity[index]),
                }
            )

    broad_name = "broad_kappa_2"
    narrow_name = "narrow_kappa_12"
    concentration_broad = 2.0
    concentration_narrow = 12.0
    scaling_tests = []
    moment_scaling = {}
    for harmonic in (1, 2):
        broad_moment = von_mises_moment(concentration_broad, harmonic)
        narrow_moment = von_mises_moment(concentration_narrow, harmonic)
        scaling = narrow_moment / broad_moment
        moment_scaling[f"m{harmonic}"] = {
            "broad_moment": broad_moment,
            "narrow_moment": narrow_moment,
            "narrow_over_broad": scaling,
        }
        scaling_tests.append(
            bootstrap_scaling_test(
                pooled_by_profile[broad_name],
                pooled_by_profile[narrow_name],
                delta_by_profile[broad_name],
                harmonic=harmonic,
                scaling=scaling,
                simulations=args.simulations,
                seed=20260770 + harmonic,
            )
        )

    report = {
        "first_run": str(args.first_run),
        "replication_run": str(args.replication_run),
        "simulations": args.simulations,
        "profiles": profiles,
        "fixed_pairwise_kernel_test": {
            "von_mises_moment_scaling": moment_scaling,
            "sine_harmonic_tests": scaling_tests,
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "extended_diagnostics.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    with (args.out_dir / "full_spectrum.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=spectra[0].keys())
        writer.writeheader()
        writer.writerows(spectra)
    with (args.out_dir / "action_probabilities.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=probability_rows[0].keys(),
        )
        writer.writeheader()
        writer.writerows(probability_rows)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

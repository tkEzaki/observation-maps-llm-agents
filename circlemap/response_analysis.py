"""Stage B response-law aggregation and preregistered diagnostics."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from .observation import wrap_phase


def _phase_dependence_permutation(
    offsets: np.ndarray,
    actions: np.ndarray,
    *,
    n_permutations: int = 5000,
    seed: int = 20260723,
) -> dict:
    unique_offsets, group_index = np.unique(offsets, return_inverse=True)
    group_counts = np.bincount(
        group_index,
        minlength=unique_offsets.size,
    )

    def statistic(values: np.ndarray) -> float:
        group_sums = np.bincount(
            group_index,
            weights=values,
            minlength=unique_offsets.size,
        )
        means = group_sums / group_counts
        return float(np.var(means))

    observed = statistic(actions)
    rng = np.random.default_rng(seed)
    exceedances = 0
    for _ in range(n_permutations):
        if statistic(rng.permutation(actions)) >= observed:
            exceedances += 1
    p_value = (exceedances + 1) / (n_permutations + 1)
    return {
        "statistic_between_offset_variance": observed,
        "permutations": n_permutations,
        "p_value": p_value,
    }


def analyze_response_records(
    records: Iterable[dict],
    output_dir: Path,
    *,
    max_harmonic: int = 6,
    n_permutations: int = 5000,
    minimum_valid_rate: float = 0.99,
    minimum_valid_samples_per_condition: int = 40,
    phase_dependence_family_alpha: float = 0.01,
    minimum_response_range: float = 0.2,
    minimum_mirrored_valid_fraction: float = 0.9,
) -> dict:
    """Aggregate valid actions and write response curve and Fourier outputs."""
    records = list(records)
    output_dir.mkdir(parents=True, exist_ok=True)
    valid = [record for record in records if record.get("valid")]
    profiles = sorted({record["profile"] for record in records})
    summary_rows = []
    profile_results = {}

    for profile_index, profile in enumerate(profiles):
        profile_all = [r for r in records if r["profile"] == profile]
        profile_valid = [r for r in valid if r["profile"] == profile]
        offsets = sorted({float(r["offset_radians"]) for r in profile_all})
        curve = []
        for offset in offsets:
            planned = [
                r
                for r in profile_all
                if float(r["offset_radians"]) == offset
            ]
            samples = [
                r
                for r in profile_valid
                if float(r["offset_radians"]) == offset
            ]
            actions = np.asarray(
                [r["action_value"] for r in samples],
                dtype=np.float64,
            )
            counts = {
                label: sum(r["action_label"] == label for r in samples)
                for label in ("retard", "stay", "advance")
            }
            n_valid = len(samples)
            mean = float(np.mean(actions)) if n_valid else float("nan")
            standard_error = (
                float(np.std(actions, ddof=1) / np.sqrt(n_valid))
                if n_valid > 1
                else float("nan")
            )
            row = {
                "profile": profile,
                "offset_radians": offset,
                "n_planned": len(planned),
                "n_valid": n_valid,
                "p_retard": counts["retard"] / n_valid if n_valid else None,
                "p_stay": counts["stay"] / n_valid if n_valid else None,
                "p_advance": (
                    counts["advance"] / n_valid if n_valid else None
                ),
                "g_mean": mean,
                "g_standard_error": standard_error,
            }
            curve.append(row)
            summary_rows.append(row)

        delta = np.asarray([row["offset_radians"] for row in curve])
        g = np.asarray([row["g_mean"] for row in curve])
        finite = np.isfinite(g)
        if not np.all(finite):
            odd = np.full_like(g, np.nan)
            even = np.full_like(g, np.nan)
            fourier = None
        else:
            mirrored_indices = [
                int(
                    np.argmin(
                        np.abs(
                            wrap_phase(delta - float(wrap_phase(-offset)))
                        )
                    )
                )
                for offset in delta
            ]
            mirrored = g[mirrored_indices]
            odd = (g - mirrored) / 2.0
            even = (g + mirrored) / 2.0
            fourier = {
                "a0": float(np.mean(g)),
                "harmonics": [
                    {
                        "m": harmonic,
                        "a_sin": float(
                            2.0 * np.mean(g * np.sin(harmonic * delta))
                        ),
                        "b_cos": float(
                            2.0 * np.mean(g * np.cos(harmonic * delta))
                        ),
                    }
                    for harmonic in range(1, max_harmonic + 1)
                ],
            }
        for row, odd_value, even_value in zip(curve, odd, even):
            row["g_odd"] = float(odd_value)
            row["g_even"] = float(even_value)

        if profile_valid:
            valid_offsets = np.asarray(
                [r["offset_radians"] for r in profile_valid],
                dtype=np.float64,
            )
            valid_actions = np.asarray(
                [r["action_value"] for r in profile_valid],
                dtype=np.float64,
            )
            dependence = _phase_dependence_permutation(
                valid_offsets,
                valid_actions,
                n_permutations=n_permutations,
                seed=20260723 + profile_index,
            )
            response_range = float(np.max(g) - np.min(g))
        else:
            dependence = None
            response_range = float("nan")

        profile_results[profile] = {
            "curve": curve,
            "fourier": fourier,
            "phase_dependence": dependence,
            "response_range": response_range,
        }

    fieldnames = [
        "profile",
        "offset_radians",
        "n_planned",
        "n_valid",
        "p_retard",
        "p_stay",
        "p_advance",
        "g_mean",
        "g_standard_error",
        "g_odd",
        "g_even",
    ]
    with (output_dir / "response_curve.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    valid_rate = len(valid) / len(records) if records else 0.0
    alpha = phase_dependence_family_alpha / max(1, len(profiles))
    phase_dependent = any(
        result["phase_dependence"] is not None
        and result["phase_dependence"]["p_value"] < alpha
        and result["response_range"] >= minimum_response_range
        for result in profile_results.values()
    )
    mirrored_sufficient = all(
        all(
            row["n_valid"]
            >= minimum_mirrored_valid_fraction * row["n_planned"]
            for row in result["curve"]
        )
        for result in profile_results.values()
    )
    sampling_sufficient = all(
        all(
            row["n_valid"] >= minimum_valid_samples_per_condition
            for row in result["curve"]
        )
        for result in profile_results.values()
    )
    report = {
        "n_planned": len(records),
        "n_valid": len(valid),
        "valid_rate": valid_rate,
        "profiles": profile_results,
        "criteria": {
            "valid_rate_sufficient": valid_rate >= minimum_valid_rate,
            "sampling_per_condition_sufficient": sampling_sufficient,
            "phase_dependence_detected": phase_dependent,
            "mirrored_offsets_sufficient": mirrored_sufficient,
            "all_stage_b_data_criteria_pass": (
                valid_rate >= minimum_valid_rate
                and sampling_sufficient
                and phase_dependent
                and mirrored_sufficient
            ),
        },
        "phase_dependence_rule": {
            "permutation_alpha_bonferroni": alpha,
            "minimum_response_range": minimum_response_range,
            "n_permutations": n_permutations,
        },
        "frozen_thresholds": {
            "minimum_valid_rate": minimum_valid_rate,
            "minimum_valid_samples_per_condition": (
                minimum_valid_samples_per_condition
            ),
            "phase_dependence_family_alpha": (
                phase_dependence_family_alpha
            ),
            "minimum_mirrored_valid_fraction": (
                minimum_mirrored_valid_fraction
            ),
        },
    }
    with (output_dir / "response_analysis.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    return report

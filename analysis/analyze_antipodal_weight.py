"""Analyze dense antipodal weight-imbalance (P1) traces."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from analysis.analyze_complex_kernel import (
    _endpoint_row,
    _pairwise_distances,
    _probability_matrix,
    _write_csv,
)
from analysis.analyze_transmutation import load_block
from analysis.complex_kernel import (
    bootstrap_complex_coefficients,
    covariance_ellipse_95,
    mean_js_divergence,
    mean_total_variation,
    polar,
    summarize_condition,
)

ROOT = Path(__file__).resolve().parents[1]


def _epsilon_from_profile(profile: str, protocol: dict) -> float:
    mapping = protocol.get("epsilon_by_profile", {})
    if profile in mapping:
        return float(mapping[profile])
    raise KeyError(f"missing epsilon for profile {profile}")


def _ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return float("nan")
    return float(np.polyfit(x, y, 1)[0])


def analyze_weight_sweep(
    run_dirs: list[Path],
    output_dir: Path,
    *,
    bootstrap_samples: int = 400,
) -> dict:
    if len(run_dirs) != 2:
        raise ValueError("exactly two run directories required")
    loaded = [load_block(path) for path in run_dirs]
    protocol = loaded[0][0]["protocol"]
    offsets = np.asarray(protocol["offset_radians"], dtype=np.float64)
    profiles = list(protocol["stimulus_profiles"])
    representations = list(protocol["representations"])
    primary_order = int(protocol["primary_fourier_order"])
    sensitivity_orders = [
        int(order) for order in protocol.get("sensitivity_fourier_orders", [2, 4, 6, 8, 12])
    ]

    endpoint_rows: list[dict] = []
    block_payload: dict = {}
    for block_index, (config, counts) in enumerate(loaded):
        block_id = config["protocol"]["seed_block_id"]
        rng = np.random.default_rng(2026072400 + block_index)
        block_payload[block_id] = {"representations": {}}
        for representation in representations:
            representation_result = {"conditions": {}}
            for profile in profiles:
                epsilon = _epsilon_from_profile(profile, protocol)
                cell = counts[representation][profile]
                summary = summarize_condition(
                    cell,
                    offsets,
                    primary_order=primary_order,
                    sensitivity_orders=sensitivity_orders,
                )
                boot = bootstrap_complex_coefficients(
                    cell,
                    offsets,
                    order=primary_order,
                    samples=bootstrap_samples,
                    rng=rng,
                )
                ellipses = {
                    "1": covariance_ellipse_95(
                        np.column_stack([boot["a1"], boot["b1"]])
                    ),
                    "2": covariance_ellipse_95(
                        np.column_stack([boot["a2"], boot["b2"]])
                    ),
                }
                representation_result["conditions"][profile] = {
                    "epsilon": epsilon,
                    "concentration": float(protocol.get("epsilon_by_profile", {}).get(profile, 0.0)),
                    **summary,
                    "bootstrap_ellipses": ellipses,
                }
                row = _endpoint_row(
                    block_id=block_id,
                    representation=representation,
                    profile=profile,
                    concentration=epsilon,
                    summary=summary,
                    ellipse1=ellipses["1"],
                    ellipse2=ellipses["2"],
                )
                row["epsilon"] = epsilon
                endpoint_rows.append(row)
            block_payload[block_id]["representations"][representation] = (
                representation_result
            )

    # Across-block mean trajectories vs epsilon.
    trajectory_rows = []
    susceptibilities = []
    for representation in representations:
        epsilons = []
        a1_vals = []
        b1_vals = []
        r1_vals = []
        r2_vals = []
        a0_vals = []
        activity_vals = []
        for profile in profiles:
            epsilon = _epsilon_from_profile(profile, protocol)
            a1 = []
            b1 = []
            r2 = []
            a0 = []
            activity = []
            for block_id in block_payload:
                condition = block_payload[block_id]["representations"][representation][
                    "conditions"
                ][profile]
                harmonics = condition["harmonics"]
                a1.append(harmonics["a_sin"][1])
                b1.append(harmonics["b_cos"][1])
                r2.append(harmonics["polar"]["2"]["R"])
                a0.append(harmonics["a0"])
                activity.append(condition["activity"]["mean_activity"])
            mean_a1 = float(np.mean(a1))
            mean_b1 = float(np.mean(b1))
            radius, phase = polar(mean_a1, mean_b1)
            epsilons.append(epsilon)
            a1_vals.append(mean_a1)
            b1_vals.append(mean_b1)
            r1_vals.append(radius)
            r2_vals.append(float(np.mean(r2)))
            a0_vals.append(float(np.mean(a0)))
            activity_vals.append(float(np.mean(activity)))
            trajectory_rows.append(
                {
                    "representation": representation,
                    "profile": profile,
                    "epsilon": epsilon,
                    "a1": mean_a1,
                    "b1": mean_b1,
                    "R1": radius,
                    "phi1_degrees": math.degrees(phase),
                    "R2": float(np.mean(r2)),
                    "a0": float(np.mean(a0)),
                    "activity": float(np.mean(activity)),
                }
            )
        eps = np.asarray(epsilons, dtype=np.float64)
        a1_arr = np.asarray(a1_vals, dtype=np.float64)
        b1_arr = np.asarray(b1_vals, dtype=np.float64)
        act = np.asarray(activity_vals, dtype=np.float64)
        chi_a1 = _ols_slope(eps, a1_arr)
        chi_b1 = _ols_slope(eps, b1_arr)
        by_eps = {
            float(row["epsilon"]): row
            for row in trajectory_rows
            if row["representation"] == representation
        }
        even_pairs = []
        for epsilon in sorted({abs(float(value)) for value in eps if abs(value) > 0}):
            plus = by_eps.get(epsilon)
            minus = by_eps.get(-epsilon)
            if plus is None or minus is None:
                continue
            even_pairs.append(
                {
                    "epsilon": float(epsilon),
                    "a1_odd_residual": abs(plus["a1"] + minus["a1"]),
                    "b1_odd_residual": abs(plus["b1"] + minus["b1"]),
                    "R2_even_residual": abs(plus["R2"] - minus["R2"]),
                    "a0_even_residual": abs(plus["a0"] - minus["a0"]),
                    "A_even_residual": abs(plus["activity"] - minus["activity"]),
                }
            )
        a_at_0 = float(act[np.argmin(np.abs(eps))])
        a_max = float(np.max(act))
        mid = 0.5 * (a_at_0 + a_max)
        abs_eps = np.abs(eps)
        order = np.argsort(abs_eps)
        eps_half = float("nan")
        for index in order:
            if abs_eps[index] <= 0:
                continue
            if act[index] >= mid:
                eps_half = float(abs_eps[index])
                break
        near_sorted = sorted(
            [
                (abs(float(e)), float(a))
                for e, a in zip(eps, act)
                if abs(float(e)) > 0
            ],
            key=lambda item: item[0],
        )[:3]
        if len(near_sorted) >= 2:
            slope_abs = _ols_slope(
                np.asarray([item[0] for item in near_sorted]),
                np.asarray([item[1] for item in near_sorted]),
            )
        else:
            slope_abs = float("nan")
        susceptibilities.append(
            {
                "representation": representation,
                "chi_a1": chi_a1,
                "chi_b1": chi_b1,
                "chi_parallel_a1": chi_a1,
                "activity_at_0": a_at_0,
                "activity_max": a_max,
                "activity_delta_max": a_max - a_at_0,
                "eps_half_activation": eps_half,
                "activity_abs_eps_near0_slope": slope_abs,
                "mean_a1_odd_residual": float(
                    np.mean([pair["a1_odd_residual"] for pair in even_pairs])
                )
                if even_pairs
                else float("nan"),
                "mean_A_even_residual": float(
                    np.mean([pair["A_even_residual"] for pair in even_pairs])
                )
                if even_pairs
                else float("nan"),
                "mean_R2_even_residual": float(
                    np.mean([pair["R2_even_residual"] for pair in even_pairs])
                )
                if even_pairs
                else float("nan"),
            }
        )

    # Distances at epsilon=0 profile.
    zero_profile = next(
        profile
        for profile in profiles
        if abs(_epsilon_from_profile(profile, protocol)) < 1e-15
    )
    concentrations = {
        profile: abs(_epsilon_from_profile(profile, protocol))
        for profile in profiles
    }
    distance_rows = []
    for block_id in block_payload:
        distance_rows.extend(
            _pairwise_distances(
                block_id=block_id,
                representations=representations,
                profiles=[zero_profile],
                concentrations=concentrations,
                block_payload=block_payload,
                offsets=offsets,
            )
        )

    report = {
        "family": "antipodal_weight_sweep",
        "protocol_name": protocol["protocol_name"],
        "zero_profile": zero_profile,
        "epsilons": [
            _epsilon_from_profile(profile, protocol) for profile in profiles
        ],
        "susceptibilities": susceptibilities,
        "interpretation_hooks": {
            "signed_C1": "chi_a1 / chi_parallel_a1",
            "activity": "eps_half_activation and activity_abs_eps_near0_slope",
            "evenness": "mean_*_even_residual / a1_odd_residual",
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "weight_sweep_analysis.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_csv(output_dir / "complex_endpoints.csv", endpoint_rows)
    _write_csv(output_dir / "epsilon_trajectories.csv", trajectory_rows)
    _write_csv(output_dir / "susceptibilities.csv", susceptibilities)
    _write_csv(output_dir / "distances_at_epsilon0.csv", distance_rows)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "analysis" / "complex_kernel_antipodal_weight",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=400)
    args = parser.parse_args()
    report = analyze_weight_sweep(
        args.run_dir,
        args.out_dir,
        bootstrap_samples=args.bootstrap_samples,
    )
    print(json.dumps({"wrote": str(args.out_dir), "n_susc": len(report["susceptibilities"])}, indent=2))


if __name__ == "__main__":
    main()

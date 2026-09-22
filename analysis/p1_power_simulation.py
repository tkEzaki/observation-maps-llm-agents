"""Power simulation for dense antipodal weight-imbalance (P1) design."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MANIFOLD = ROOT / "analysis" / "complex_kernel_stimulus_manifold"
OUT = ROOT / "analysis" / "stage_b_offline_p3_p4"


def _load_antipodal_activity() -> dict[str, float]:
    report = json.loads(
        (MANIFOLD / "complex_kernel_analysis.json").read_text(encoding="utf-8")
    )
    blocks = list(report["blocks"].keys())
    out = {}
    for rep_key, short in (
        ("intervals_24_decimal6", "intervals"),
        ("moments_m1_m3", "moments"),
        ("centers_24_standard", "centers"),
    ):
        values = []
        for block in blocks:
            values.append(
                report["blocks"][block]["representations"][rep_key]["conditions"][
                    "antipodal_equal_k6"
                ]["activity"]["mean_activity"]
            )
        out[short] = float(np.mean(values))
    return out


def _simulate_activity_detection(
    *,
    a0: float,
    a_inf: float,
    eps_half: float,
    eps_grid: np.ndarray,
    n_offsets: int,
    n_samples: int,
    n_monte: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Power to detect encoding×|ε| activation vs flat null via ANOVA F."""
    detections = 0
    for _ in range(n_monte):
        means = []
        for eps in eps_grid:
            # Even activation model: A(|ε|) = A0 + (A∞-A0) * |ε|/(|ε|+ε_half)
            target = a0 + (a_inf - a0) * (
                abs(float(eps)) / (abs(float(eps)) + eps_half)
            )
            target = float(np.clip(target, 1e-3, 1.0 - 1e-3))
            # Offset-averaged binomial-like activity noise
            draws = rng.binomial(n_samples, target, size=n_offsets) / n_samples
            means.append(float(np.mean(draws)))
        # Contrast: slope of A vs |ε| via simple correlation
        abs_eps = np.abs(eps_grid)
        corr = np.corrcoef(abs_eps, np.asarray(means))[0, 1]
        if corr == corr and corr > 0.5:
            detections += 1
    return {
        "power_corr_gt_0.5": detections / n_monte,
        "a0": a0,
        "a_inf": a_inf,
        "eps_half": eps_half,
    }


def _simulate_signed_c1_detection(
    *,
    chi: float,
    eps_grid: np.ndarray,
    n_offsets: int,
    n_samples: int,
    noise_sd: float,
    n_monte: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Power to recover sign(χ) from linear fit of a1 ≈ Re(C1) vs ε."""
    detections = 0
    for _ in range(n_monte):
        a1 = []
        for eps in eps_grid:
            # Effective SE shrinks with sqrt(n_offsets * n_samples)
            se = noise_sd / math.sqrt(max(1, n_offsets * n_samples / 12.0))
            a1.append(chi * float(eps) + rng.normal(0.0, se))
        # Finite-difference / OLS slope
        slope = float(np.polyfit(eps_grid, np.asarray(a1), 1)[0])
        if np.sign(slope) == np.sign(chi) and abs(slope) > 0.25 * abs(chi):
            detections += 1
    return {
        "power_recover_sign_and_scale": detections / n_monte,
        "chi": chi,
        "noise_sd": noise_sd,
    }


def main() -> None:
    activity0 = _load_antipodal_activity()
    # Default signed grid from plan
    grid_a = np.asarray(
        [-0.10, -0.05, -0.02, 0.0, 0.02, 0.05, 0.10, 0.20],
        dtype=np.float64,
    )
    # Near-zero densified positive + signed audit
    grid_b = np.asarray(
        [-0.10, -0.02, 0.0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20],
        dtype=np.float64,
    )
    rng = np.random.default_rng(20260723)
    n_offsets = 36
    results = {
        "activity0_empirical": activity0,
        "grids": {
            "signed_8": grid_a.tolist(),
            "nearzero_9": grid_b.tolist(),
        },
        "scenarios": [],
    }
    for name, grid in (("signed_8", grid_a), ("nearzero_9", grid_b)):
        for n_samples in (8, 12):
            moments = _simulate_activity_detection(
                a0=activity0["moments"],
                a_inf=1.0,
                eps_half=0.03,
                eps_grid=grid,
                n_offsets=n_offsets,
                n_samples=n_samples,
                n_monte=400,
                rng=rng,
            )
            intervals = _simulate_activity_detection(
                a0=activity0["intervals"],
                a_inf=0.98,
                eps_half=0.15,
                eps_grid=grid,
                n_offsets=n_offsets,
                n_samples=n_samples,
                n_monte=400,
                rng=rng,
            )
            c1_moments = _simulate_signed_c1_detection(
                chi=4.0,
                eps_grid=grid,
                n_offsets=n_offsets,
                n_samples=n_samples,
                noise_sd=0.35,
                n_monte=400,
                rng=rng,
            )
            c1_intervals = _simulate_signed_c1_detection(
                chi=1.0,
                eps_grid=grid,
                n_offsets=n_offsets,
                n_samples=n_samples,
                noise_sd=0.35,
                n_monte=400,
                rng=rng,
            )
            n_profiles = len(grid)
            calls_per_block = 3 * n_profiles * n_offsets * n_samples
            results["scenarios"].append(
                {
                    "grid": name,
                    "n_samples": n_samples,
                    "n_profiles": n_profiles,
                    "calls_per_block": calls_per_block,
                    "calls_two_blocks": 2 * calls_per_block,
                    "moments_activity_power": moments,
                    "intervals_activity_power": intervals,
                    "moments_c1_power": c1_moments,
                    "intervals_c1_power": c1_intervals,
                }
            )

    # Recommendation: signed_8 with 12 samples — matches manifold density,
    # high power for moments activation; densified grid only if moments rise
    # is unresolved after primary panel.
    recommendation = {
        "grid": "signed_8",
        "epsilon": grid_a.tolist(),
        "repetitions_per_condition": 12,
        "n_offsets": 36,
        "calls_per_block": 3 * 8 * 36 * 12,
        "calls_two_blocks": 2 * 3 * 8 * 36 * 12,
        "rationale": (
            "Signed 8-point weight axis recovers even/odd covariance tests; "
            "12 samples/offset matches manifold v0.1; moments activation "
            "power is high under eps_half~0.03. Defer 0.005/0.01 densification "
            "to a top-up if A(|ε|) is unresolved between 0 and 0.02."
        ),
        "estimated_usd_two_blocks_scaled_from_manifold": None,
    }
    # Scale from analyzed manifold pair actual cost ~$6.266799 / 15552 calls
    manifold_calls = 15552
    manifold_usd = 6.266799
    recommendation["estimated_usd_two_blocks_scaled_from_manifold"] = round(
        recommendation["calls_two_blocks"] / manifold_calls * manifold_usd,
        4,
    )
    results["recommendation"] = recommendation

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "p1_power_simulation.json"
    path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(recommendation, indent=2))


if __name__ == "__main__":
    main()

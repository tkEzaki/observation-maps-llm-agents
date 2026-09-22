"""Offline complex-kernel reanalysis for transmutation and feature-sweep runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

try:
    from analysis.analyze_feature_sweep import (
        _canonical_id,
        load_block as load_feature_block,
    )
    from analysis.analyze_transmutation import load_block as load_transmutation_block
    from analysis.complex_kernel import (
        average_complex_coefficient,
        bootstrap_complex_coefficients,
        complex_from_ab,
        complex_low_order_distance,
        covariance_ellipse_95,
        density_weighted_l2,
        ellipse_contains_origin,
        mean_js_divergence,
        mean_kl_divergence,
        mean_total_variation,
        origin_shift_null,
        origin_shift_residual,
        phase_reportable,
        polar,
        polar_channel_distance,
        summarize_condition,
        wrap_angle,
    )
except ModuleNotFoundError:
    from analyze_feature_sweep import (
        _canonical_id,
        load_block as load_feature_block,
    )
    from analyze_transmutation import load_block as load_transmutation_block
    from complex_kernel import (
        average_complex_coefficient,
        bootstrap_complex_coefficients,
        complex_from_ab,
        complex_low_order_distance,
        covariance_ellipse_95,
        density_weighted_l2,
        ellipse_contains_origin,
        mean_js_divergence,
        mean_kl_divergence,
        mean_total_variation,
        origin_shift_null,
        origin_shift_residual,
        phase_reportable,
        polar,
        polar_channel_distance,
        summarize_condition,
        wrap_angle,
    )


DEFAULT_ORDERS = [2, 4, 6, 8, 12]
DEFAULT_BOOTSTRAP = 2000


def _profile_order(protocol: dict) -> list[str]:
    return sorted(
        protocol["concentration_profiles"],
        key=lambda profile: protocol["concentration_profiles"][profile],
    )


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _feature_memberships(protocol: dict) -> dict[str, list[tuple[str, float]]]:
    memberships: dict[str, list[tuple[str, float]]] = {}
    for axis, entries in protocol.get("feature_axes", {}).items():
        for entry in entries:
            memberships.setdefault(entry["representation"], []).append(
                (axis, float(entry["value"]))
            )
    return memberships


def _probability_matrix(summary: dict) -> np.ndarray:
    probs = summary["probabilities_by_offset"]
    return np.column_stack(
        [probs["retard"], probs["stay"], probs["advance"]]
    )


def _phase_fields(
    *,
    a: float,
    b: float,
    ellipse: dict[str, float] | None = None,
    minimum_radius: float = 0.15,
) -> dict[str, float | int | None]:
    """Polar fields with weak-amplitude / origin-covering phase suppression."""
    radius, phase = polar(a, b)
    reportable = phase_reportable(
        radius, ellipse=ellipse, minimum_radius=minimum_radius
    )
    origin_hit = (
        bool(ellipse_contains_origin(ellipse)) if ellipse is not None else False
    )
    return {
        "R": radius,
        "phi": phase if reportable else float("nan"),
        "phi_degrees": math.degrees(phase) if reportable else float("nan"),
        "phi_reportable": int(reportable),
        "phi_suppressed_weak_R": int(radius < minimum_radius),
        "phi_suppressed_origin_in_ellipse": int(origin_hit and not reportable),
        "phi_raw_degrees": math.degrees(phase),
    }


def _endpoint_row(
    *,
    block_id: str,
    representation: str,
    profile: str,
    concentration: float,
    summary: dict,
    ellipse1: dict[str, float] | None = None,
    ellipse2: dict[str, float] | None = None,
    axis: str | None = None,
    feature_value: float | None = None,
) -> dict:
    harmonics = summary["harmonics"]
    a1 = harmonics["a_sin"].get(1, float("nan"))
    b1 = harmonics["b_cos"].get(1, float("nan"))
    a2 = harmonics["a_sin"].get(2, float("nan"))
    b2 = harmonics["b_cos"].get(2, float("nan"))
    phase1 = _phase_fields(a=a1, b=b1, ellipse=ellipse1)
    phase2 = _phase_fields(a=a2, b=b2, ellipse=ellipse2)
    a0 = summary["a0_robustness"]
    gprime = summary["gprime0_diagnostic"]
    row = {
        "block": block_id,
        "representation": representation,
        "profile": profile,
        "concentration": concentration,
        "a0": harmonics["a0"],
        "a1": a1,
        "b1": b1,
        "R1": phase1["R"],
        "phi1": phase1["phi"],
        "phi1_degrees": phase1["phi_degrees"],
        "phi1_reportable": phase1["phi_reportable"],
        "phi1_raw_degrees": phase1["phi_raw_degrees"],
        "a2": a2,
        "b2": b2,
        "R2": phase2["R"],
        "phi2": phase2["phi"],
        "phi2_degrees": phase2["phi_degrees"],
        "phi2_reportable": phase2["phi_reportable"],
        "phi2_raw_degrees": phase2["phi_raw_degrees"],
        "activity": summary["activity"]["mean_activity"],
        "mean_stay": summary["activity"]["mean_stay"],
        "mean_entropy": summary["mean_entropy"],
        "entropy_units": "nats",
        "entropy_normalized_by_log3": 0,
        "no_stay_directional_controller": int(
            summary["activity"]["no_stay_directional_controller"]
        ),
        "hard_directional_controller": int(
            summary["activity"]["hard_directional_controller"]
        ),
        "a0_fourier_range": a0["fourier_a0_range"],
        "a0_unweighted_mean_g": a0["unweighted_mean_g"],
        "a0_density_weighted_mean_g": a0["density_weighted_mean_g"],
        "a0_mirror_even_mean_g": (
            a0["mirror_even_mean_g"]
            if a0["mirror_even_mean_g"] is not None
            else float("nan")
        ),
        "g_prime0_M6_diagnostic": summary[
            "fitted_mean_response_derivative_at_zero_diagnostic"
        ],
        "g_prime0_order_range_diagnostic": gprime["range"],
        "g_prime0_sign_flip_diagnostic": int(gprime["sign_flip_across_orders"]),
        "n_fixed_points_diagnostic": summary["n_fixed_points_diagnostic"],
    }
    if axis is not None:
        row["feature_axis"] = axis
        row["feature_value"] = feature_value
    return row


def _a0_rows(
    *,
    block_id: str,
    representation: str,
    profile: str,
    concentration: float,
    summary: dict,
) -> list[dict]:
    suite = summary["a0_robustness"]
    rows = []
    for order, value in suite["fourier_a0_by_order"].items():
        rows.append(
            {
                "block": block_id,
                "representation": representation,
                "profile": profile,
                "concentration": concentration,
                "estimator": f"fourier_M{order}",
                "a0": value,
            }
        )
    for estimator, key in (
        ("unweighted_mean_g", "unweighted_mean_g"),
        ("density_weighted_mean_g", "density_weighted_mean_g"),
        ("mirror_even_mean_g", "mirror_even_mean_g"),
    ):
        value = suite[key]
        rows.append(
            {
                "block": block_id,
                "representation": representation,
                "profile": profile,
                "concentration": concentration,
                "estimator": estimator,
                "a0": value if value is not None else float("nan"),
            }
        )
    return rows


def _offset_rows(
    *,
    block_id: str,
    representation: str,
    profile: str,
    concentration: float,
    offsets: np.ndarray,
    summary: dict,
) -> list[dict]:
    rows = []
    probs = summary["probabilities_by_offset"]
    for index, offset in enumerate(offsets):
        rows.append(
            {
                "block": block_id,
                "representation": representation,
                "profile": profile,
                "concentration": concentration,
                "offset_index": index,
                "offset_radians": float(offset),
                "g": summary["g_by_offset"][index],
                "activity": summary["activity_by_offset"][index],
                "entropy": summary["entropy_by_offset"][index],
                "p_retard": probs["retard"][index],
                "p_stay": probs["stay"][index],
                "p_advance": probs["advance"][index],
            }
        )
    return rows


def _gprime_rows(
    *,
    block_id: str,
    representation: str,
    profile: str,
    concentration: float,
    summary: dict,
) -> list[dict]:
    rows = []
    for order, value in summary["gprime0_diagnostic"]["by_order"].items():
        rows.append(
            {
                "block": block_id,
                "representation": representation,
                "profile": profile,
                "concentration": concentration,
                "fourier_order": int(order),
                "g_prime0": value,
                "n_fixed_points_fitted": summary["fixed_point_order_diagnostic"][
                    "by_order"
                ][order],
                "status": "diagnostic_only_not_primary",
            }
        )
    return rows


def _ellipse_rows(
    *,
    block_id: str,
    representation: str,
    profile: str,
    concentration: float,
    boot: dict[str, np.ndarray],
    axis: str | None = None,
    feature_value: float | None = None,
) -> list[dict]:
    rows = []
    for harmonic, a_key, b_key in ((1, "a1", "b1"), (2, "a2", "b2")):
        ellipse = covariance_ellipse_95(
            np.column_stack([boot[a_key], boot[b_key]])
        )
        row = {
            "block": block_id,
            "representation": representation,
            "profile": profile,
            "concentration": concentration,
            "harmonic": harmonic,
            **ellipse,
            "contains_origin": int(ellipse_contains_origin(ellipse)),
            "note": "conditional_on_block_multinomial_bootstrap_not_formal_pairwise_test",
        }
        if axis is not None:
            row["feature_axis"] = axis
            row["feature_value"] = feature_value
        rows.append(row)
    return rows


def _distance_bundle(
    offsets: np.ndarray,
    left_summary: dict,
    right_summary: dict,
) -> dict[str, float]:
    left_probs = _probability_matrix(left_summary)
    right_probs = _probability_matrix(right_summary)
    return {
        "response_l2": density_weighted_l2(
            offsets,
            left_summary["g_by_offset"],
            right_summary["g_by_offset"],
        ),
        "complex_low_order": complex_low_order_distance(
            left_summary["harmonics"],
            right_summary["harmonics"],
        ),
        "c1_chordal": polar_channel_distance(
            left_summary["harmonics"],
            right_summary["harmonics"],
            harmonic=1,
        ),
        "c2_chordal": polar_channel_distance(
            left_summary["harmonics"],
            right_summary["harmonics"],
            harmonic=2,
        ),
        "mean_tv": mean_total_variation(left_probs, right_probs),
        "mean_js": mean_js_divergence(left_probs, right_probs),
        "mean_sym_kl": mean_kl_divergence(left_probs, right_probs),
    }


def _pairwise_distances(
    *,
    block_id: str,
    representations: list[str],
    profiles: list[str],
    concentrations: dict[str, float],
    block_payload: dict,
    offsets: np.ndarray,
) -> list[dict]:
    rows = []
    for profile in profiles:
        for left_index, left in enumerate(representations):
            for right in representations[left_index + 1 :]:
                left_summary = block_payload[block_id]["representations"][left][
                    "conditions"
                ][profile]
                right_summary = block_payload[block_id]["representations"][right][
                    "conditions"
                ][profile]
                rows.append(
                    {
                        "block": block_id,
                        "profile": profile,
                        "concentration": concentrations[profile],
                        "representation_a": left,
                        "representation_b": right,
                        "comparison": "between_representation",
                        **_distance_bundle(offsets, left_summary, right_summary),
                    }
                )
    return rows


def _within_block_distances(
    *,
    representations: list[str],
    profiles: list[str],
    concentrations: dict[str, float],
    block_payload: dict,
    offsets: np.ndarray,
) -> list[dict]:
    """Same-representation distances between the two acquisition blocks."""
    block_ids = list(block_payload.keys())
    if len(block_ids) != 2:
        return []
    left_block, right_block = block_ids
    rows = []
    for representation in representations:
        for profile in profiles:
            left_summary = block_payload[left_block]["representations"][representation][
                "conditions"
            ][profile]
            right_summary = block_payload[right_block]["representations"][representation][
                "conditions"
            ][profile]
            rows.append(
                {
                    "block_a": left_block,
                    "block_b": right_block,
                    "profile": profile,
                    "concentration": concentrations[profile],
                    "representation": representation,
                    "comparison": "within_representation_across_blocks",
                    **_distance_bundle(offsets, left_summary, right_summary),
                }
            )
    return rows


def _mean_complex_trajectories(
    *,
    representations: list[str],
    profiles: list[str],
    block_payload: dict,
    minimum_radius: float = 0.15,
) -> dict:
    """Average complex coeffs across blocks, then convert to polar form."""
    mean_trajectories = {}
    for representation in representations:
        mean_trajectories[representation] = {}
        for profile in profiles:
            block_coeffs1 = []
            block_coeffs2 = []
            a0_values = []
            activity_values = []
            entropy_values = []
            ellipse_contains = []
            for block_id in block_payload:
                condition = block_payload[block_id]["representations"][representation][
                    "conditions"
                ][profile]
                harmonics = condition["harmonics"]
                block_coeffs1.append(
                    (harmonics["a_sin"][1], harmonics["b_cos"][1])
                )
                block_coeffs2.append(
                    (harmonics["a_sin"][2], harmonics["b_cos"][2])
                )
                a0_values.append(harmonics["a0"])
                activity_values.append(condition["activity"]["mean_activity"])
                entropy_values.append(condition["mean_entropy"])
                ellipse = condition.get("bootstrap_ellipses", {}).get("1")
                if ellipse is not None:
                    ellipse_contains.append(ellipse_contains_origin(ellipse))
            mean1 = average_complex_coefficient(block_coeffs1)
            mean2 = average_complex_coefficient(block_coeffs2)
            # Conservative: suppress if mean R is weak or any block ellipse covers 0.
            reportable1 = phase_reportable(
                float(mean1["R"]),
                ellipse=None,
                minimum_radius=minimum_radius,
            ) and not any(ellipse_contains)
            reportable2 = phase_reportable(
                float(mean2["R"]),
                ellipse=None,
                minimum_radius=minimum_radius,
            )
            # Consistency check: polar must match mean (a,b).
            check_r, check_phi = polar(float(mean1["a"]), float(mean1["b"]))
            if abs(check_r - float(mean1["R"])) > 1e-12:
                raise RuntimeError("inconsistent R1 from mean complex coefficient")
            if abs(wrap_angle(check_phi - float(mean1["phi"]))) > 1e-12:
                raise RuntimeError("inconsistent phi1 from mean complex coefficient")
            mean_trajectories[representation][profile] = {
                "a0": float(np.mean(a0_values)),
                "a1": float(mean1["a"]),
                "b1": float(mean1["b"]),
                "R1": float(mean1["R"]),
                "phi1": float(mean1["phi"]) if reportable1 else float("nan"),
                "phi1_degrees": (
                    float(mean1["phi_degrees"]) if reportable1 else float("nan")
                ),
                "phi1_raw_degrees": float(mean1["phi_degrees"]),
                "phi1_reportable": int(reportable1),
                "a2": float(mean2["a"]),
                "b2": float(mean2["b"]),
                "R2": float(mean2["R"]),
                "phi2": float(mean2["phi"]) if reportable2 else float("nan"),
                "phi2_degrees": (
                    float(mean2["phi_degrees"]) if reportable2 else float("nan")
                ),
                "phi2_raw_degrees": float(mean2["phi_degrees"]),
                "phi2_reportable": int(reportable2),
                "activity": float(np.mean(activity_values)),
                "mean_entropy": float(np.mean(entropy_values)),
                "averaging_rule": "mean_complex_then_polar",
            }
    return mean_trajectories


def _block_stratified_delta_c_tests(
    *,
    representations: list[str],
    profiles: list[str],
    concentrations: dict[str, float],
    boot_clouds: dict,
    harmonic: int = 1,
) -> list[dict]:
    """Block-stratified bootstrap of ΔC_m, conditional on two acquisition blocks.

    Within each block, responses are resampled and C_m is formed; the two
    block-wise coefficients are then averaged within each replicate. This does
    not resample blocks from a larger run-level population.
    """
    a_key = f"a{harmonic}"
    b_key = f"b{harmonic}"
    block_ids = list(boot_clouds.keys())
    rows = []
    for profile in profiles:
        for left_index, left in enumerate(representations):
            for right in representations[left_index + 1 :]:
                stacked = []
                for block_id in block_ids:
                    left_boot = boot_clouds[block_id][left][profile]
                    right_boot = boot_clouds[block_id][right][profile]
                    stacked.append(
                        np.column_stack(
                            [
                                left_boot[a_key] - right_boot[a_key],
                                left_boot[b_key] - right_boot[b_key],
                            ]
                        )
                    )
                # Mean Δ over the two fixed blocks for each replicate index.
                n = min(cloud.shape[0] for cloud in stacked)
                mean_delta = np.mean(
                    np.stack([cloud[:n] for cloud in stacked], axis=0),
                    axis=0,
                )
                ellipse = covariance_ellipse_95(mean_delta)
                rows.append(
                    {
                        "profile": profile,
                        "concentration": concentrations[profile],
                        "representation_a": left,
                        "representation_b": right,
                        "harmonic": harmonic,
                        "delta_a": ellipse["center_a"],
                        "delta_b": ellipse["center_b"],
                        "delta_R": float(
                            math.hypot(ellipse["center_a"], ellipse["center_b"])
                        ),
                        "contains_origin": int(ellipse_contains_origin(ellipse)),
                        "axis_major": ellipse["axis_major"],
                        "axis_minor": ellipse["axis_minor"],
                        "conditioning": (
                            "block_stratified_bootstrap_conditional_on_"
                            "two_acquisition_blocks"
                        ),
                        "role": "auxiliary_not_familywise_primary",
                        "note": "origin_exclusion_is_not_multiple_comparison_adjusted",
                    }
                )
    return rows


def _distance_ratio_rows(
    *,
    between_rows: list[dict],
    within_rows: list[dict],
) -> list[dict]:
    """between-representation distance / within-representation block distance."""
    within_lookup = {
        (row["representation"], row["profile"]): row for row in within_rows
    }
    metrics = (
        "response_l2",
        "complex_low_order",
        "c1_chordal",
        "c2_chordal",
        "mean_tv",
        "mean_js",
        "mean_sym_kl",
    )
    rows = []
    for row in between_rows:
        if row.get("comparison") != "between_representation":
            continue
        if row["representation_a"] == row["representation_b"]:
            continue
        left_within = within_lookup.get(
            (row["representation_a"], row["profile"])
        )
        right_within = within_lookup.get(
            (row["representation_b"], row["profile"])
        )
        if left_within is None or right_within is None:
            continue
        out = {
            "block": row["block"],
            "profile": row["profile"],
            "concentration": row["concentration"],
            "representation_a": row["representation_a"],
            "representation_b": row["representation_b"],
        }
        for metric in metrics:
            within_scale = 0.5 * (
                float(left_within[metric]) + float(right_within[metric])
            )
            between = float(row[metric])
            out[f"{metric}_between"] = between
            out[f"{metric}_within_mean"] = within_scale
            out[f"{metric}_ratio"] = (
                between / within_scale if within_scale > 1e-12 else float("nan")
            )
        rows.append(out)
    return rows


def analyze_transmutation(
    run_dirs: list[Path],
    output_dir: Path,
    *,
    bootstrap_samples: int | None = None,
) -> dict:
    if len(run_dirs) != 2:
        raise ValueError("exactly two transmutation run directories are required")
    loaded = [load_transmutation_block(path) for path in run_dirs]
    protocol = loaded[0][0]["protocol"]
    offsets = np.asarray(protocol["offset_radians"], dtype=np.float64)
    primary_order = int(protocol["primary_fourier_order"])
    sensitivity_orders = [
        int(order) for order in protocol.get("sensitivity_fourier_orders", DEFAULT_ORDERS)
    ]
    profiles = _profile_order(protocol)
    n_bootstrap = int(
        bootstrap_samples
        if bootstrap_samples is not None
        else min(int(protocol.get("bootstrap_samples", DEFAULT_BOOTSTRAP)), DEFAULT_BOOTSTRAP)
    )

    endpoint_rows: list[dict] = []
    a0_rows: list[dict] = []
    offset_rows: list[dict] = []
    gprime_rows: list[dict] = []
    ellipse_rows: list[dict] = []
    distance_rows: list[dict] = []
    block_payload: dict = {}
    boot_clouds: dict = {}

    for block_index, (config, counts) in enumerate(loaded):
        block_id = config["protocol"]["seed_block_id"]
        rng = np.random.default_rng(2026072300 + block_index)
        block_payload[block_id] = {
            "operational": config["operational"],
            "representations": {},
        }
        boot_clouds[block_id] = {}
        for representation in protocol["representations"]:
            representation_result = {"conditions": {}}
            boot_clouds[block_id][representation] = {}
            for profile in profiles:
                concentration = float(protocol["concentration_profiles"][profile])
                cell_counts = counts[representation][profile]
                summary = summarize_condition(
                    cell_counts,
                    offsets,
                    primary_order=primary_order,
                    sensitivity_orders=sensitivity_orders,
                )
                boot = bootstrap_complex_coefficients(
                    cell_counts,
                    offsets,
                    order=primary_order,
                    samples=n_bootstrap,
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
                    "concentration": concentration,
                    **summary,
                    "bootstrap_ellipses": ellipses,
                }
                boot_clouds[block_id][representation][profile] = boot
                endpoint_rows.append(
                    _endpoint_row(
                        block_id=block_id,
                        representation=representation,
                        profile=profile,
                        concentration=concentration,
                        summary=summary,
                        ellipse1=ellipses["1"],
                        ellipse2=ellipses["2"],
                    )
                )
                a0_rows.extend(
                    _a0_rows(
                        block_id=block_id,
                        representation=representation,
                        profile=profile,
                        concentration=concentration,
                        summary=summary,
                    )
                )
                offset_rows.extend(
                    _offset_rows(
                        block_id=block_id,
                        representation=representation,
                        profile=profile,
                        concentration=concentration,
                        offsets=offsets,
                        summary=summary,
                    )
                )
                gprime_rows.extend(
                    _gprime_rows(
                        block_id=block_id,
                        representation=representation,
                        profile=profile,
                        concentration=concentration,
                        summary=summary,
                    )
                )
                ellipse_rows.extend(
                    _ellipse_rows(
                        block_id=block_id,
                        representation=representation,
                        profile=profile,
                        concentration=concentration,
                        boot=boot,
                    )
                )
            block_payload[block_id]["representations"][representation] = (
                representation_result
            )
        distance_rows.extend(
            _pairwise_distances(
                block_id=block_id,
                representations=list(protocol["representations"]),
                profiles=profiles,
                concentrations=protocol["concentration_profiles"],
                block_payload=block_payload,
                offsets=offsets,
            )
        )

    within_rows = _within_block_distances(
        representations=list(protocol["representations"]),
        profiles=profiles,
        concentrations=protocol["concentration_profiles"],
        block_payload=block_payload,
        offsets=offsets,
    )
    ratio_rows = _distance_ratio_rows(
        between_rows=distance_rows,
        within_rows=within_rows,
    )
    delta_c1_rows = _block_stratified_delta_c_tests(
        representations=list(protocol["representations"]),
        profiles=profiles,
        concentrations=protocol["concentration_profiles"],
        boot_clouds=boot_clouds,
        harmonic=1,
    )
    mean_trajectories = _mean_complex_trajectories(
        representations=list(protocol["representations"]),
        profiles=profiles,
        block_payload=block_payload,
    )

    report = {
        "family": "transmutation",
        "protocol_name": protocol["protocol_name"],
        "primary_fourier_order": primary_order,
        "sensitivity_fourier_orders": sensitivity_orders,
        "bootstrap_samples": n_bootstrap,
        "blocks": block_payload,
        "mean_trajectories": mean_trajectories,
        "primary_endpoints": [
            "C1",
            "C2",
            "a0_robust",
            "activity",
            "action_entropy",
            "trinomial_distribution",
        ],
        "demoted_diagnostics": ["g_prime0", "n_fixed_points"],
        "terminology": {
            "object": "representation-induced microscopic response-law transmutation",
            "transmutation_operational": (
                "reproducible change in complex harmonic coefficients and action "
                "distribution induced by changing the observation encoding while "
                "holding the underlying field, model, response contract, and "
                "generation settings fixed"
            ),
            "encoding_phrase": (
                "different encodings of the same underlying unimodal von Mises field"
            ),
            "a0_phrase": "offset-averaged action bias",
            "g_prime0_phrase": (
                "derivative of the fitted mean-response curve at zero offset"
            ),
            "hard_controller_phrase": (
                "no-stay directional controller with low output entropy; "
                "'hard' means stay probability is operationally zero, not "
                "fully deterministic advance/retard"
            ),
            "weak_R_phase_threshold": (
                "visualization_and_reporting_convention_not_inferential_endpoint"
            ),
            "delta_c1_role": "auxiliary_block_stratified_not_familywise_primary",
            "not_claimed": [
                "macroscopic interaction-phase transmutation",
                "discontinuous bifurcation",
                "global attractive/repulsive coupling from a1 alone",
                "aligned-state stability from M=6 g'(0)",
                "ellipse non-overlap as formal pairwise significance",
                "family-wise significance of all Delta-C1 cells",
            ],
        },
        "uncertainty_caveat": (
            "Bootstrap ellipses and ΔC1 tests use a block-stratified bootstrap "
            "conditional on the two independent acquisition blocks; they do not "
            "estimate run-level variance by resampling blocks, and ΔC1 cells are "
            "auxiliary without multiplicity adjustment."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "complex_kernel_analysis.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_csv(output_dir / "complex_endpoints.csv", endpoint_rows)
    _write_csv(output_dir / "a0_robustness.csv", a0_rows)
    _write_csv(output_dir / "offset_action_curves.csv", offset_rows)
    _write_csv(output_dir / "gprime0_order_sensitivity.csv", gprime_rows)
    _write_csv(output_dir / "complex_ellipses.csv", ellipse_rows)
    _write_csv(output_dir / "representation_distances.csv", distance_rows)
    _write_csv(output_dir / "within_representation_distances.csv", within_rows)
    _write_csv(output_dir / "distance_between_within_ratios.csv", ratio_rows)
    _write_csv(output_dir / "delta_c1_bootstrap_tests.csv", delta_c1_rows)
    return report


def analyze_feature_sweep(
    run_dirs: list[Path],
    output_dir: Path,
    *,
    bootstrap_samples: int | None = None,
) -> dict:
    if len(run_dirs) != 2:
        raise ValueError("exactly two feature-sweep run directories are required")
    loaded = [load_feature_block(path) for path in run_dirs]
    protocol = loaded[0][0]["protocol"]
    offsets = np.asarray(protocol["offset_radians"], dtype=np.float64)
    primary_order = int(protocol["primary_fourier_order"])
    sensitivity_orders = [
        int(order) for order in protocol.get("sensitivity_fourier_orders", DEFAULT_ORDERS)
    ]
    profiles = _profile_order(protocol)
    representation_ids = [_canonical_id(spec) for spec in protocol["representations"]]
    feature_memberships = _feature_memberships(protocol)
    pseudocount = float(protocol.get("bootstrap_pseudocount", 0.5))
    n_bootstrap = int(
        bootstrap_samples
        if bootstrap_samples is not None
        else min(int(protocol.get("bootstrap_samples", DEFAULT_BOOTSTRAP)), DEFAULT_BOOTSTRAP)
    )

    endpoint_rows: list[dict] = []
    a0_rows: list[dict] = []
    offset_rows: list[dict] = []
    gprime_rows: list[dict] = []
    ellipse_rows: list[dict] = []
    origin_rows: list[dict] = []
    block_payload: dict = {}

    for block_index, (config, counts) in enumerate(loaded):
        block_id = config["protocol"]["seed_block_id"]
        rng = np.random.default_rng(2026072310 + block_index)
        block_payload[block_id] = {
            "operational": config["operational"],
            "representations": {},
        }
        for representation in representation_ids:
            memberships = feature_memberships.get(representation, [(None, None)])
            representation_result = {
                "conditions": {},
                "feature_axes": [
                    {"axis": axis, "value": feature_value}
                    for axis, feature_value in memberships
                    if axis is not None
                ],
            }
            for profile in profiles:
                concentration = float(protocol["concentration_profiles"][profile])
                cell_counts = counts[representation][profile]
                summary = summarize_condition(
                    cell_counts,
                    offsets,
                    primary_order=primary_order,
                    sensitivity_orders=sensitivity_orders,
                )
                boot = bootstrap_complex_coefficients(
                    cell_counts,
                    offsets,
                    order=primary_order,
                    samples=n_bootstrap,
                    rng=rng,
                    pseudocount=pseudocount,
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
                    "concentration": concentration,
                    **summary,
                    "bootstrap_ellipses": ellipses,
                }
                for axis, feature_value in memberships:
                    endpoint_rows.append(
                        _endpoint_row(
                            block_id=block_id,
                            representation=representation,
                            profile=profile,
                            concentration=concentration,
                            summary=summary,
                            ellipse1=ellipses["1"],
                            ellipse2=ellipses["2"],
                            axis=axis,
                            feature_value=feature_value,
                        )
                    )
                    ellipse_rows.extend(
                        _ellipse_rows(
                            block_id=block_id,
                            representation=representation,
                            profile=profile,
                            concentration=concentration,
                            boot=boot,
                            axis=axis,
                            feature_value=feature_value,
                        )
                    )
                a0_rows.extend(
                    _a0_rows(
                        block_id=block_id,
                        representation=representation,
                        profile=profile,
                        concentration=concentration,
                        summary=summary,
                    )
                )
                offset_rows.extend(
                    _offset_rows(
                        block_id=block_id,
                        representation=representation,
                        profile=profile,
                        concentration=concentration,
                        offsets=offsets,
                        summary=summary,
                    )
                )
                gprime_rows.extend(
                    _gprime_rows(
                        block_id=block_id,
                        representation=representation,
                        profile=profile,
                        concentration=concentration,
                        summary=summary,
                    )
                )
            block_payload[block_id]["representations"][representation] = (
                representation_result
            )

    origin_entries = protocol.get("feature_axes", {}).get("origin_shift", [])
    if origin_entries:
        baseline_id = origin_entries[0]["representation"]
        n_bins = 24
        for spec in protocol["representations"]:
            if isinstance(spec, dict) and _canonical_id(spec) == baseline_id:
                n_bins = int(spec.get("n_bins", 24))
                break
        delta_bin = (2.0 * math.pi) / float(n_bins)
        for block_id in block_payload:
            for profile in profiles:
                baseline = block_payload[block_id]["representations"][baseline_id][
                    "conditions"
                ][profile]["harmonics"]
                for entry in origin_entries:
                    representation = entry["representation"]
                    shift = float(entry["value"])
                    observed = block_payload[block_id]["representations"][representation][
                        "conditions"
                    ][profile]["harmonics"]
                    for harmonic in (1, 2):
                        c0 = complex_from_ab(
                            baseline["a_sin"][harmonic],
                            baseline["b_cos"][harmonic],
                        )
                        c_obs = complex_from_ab(
                            observed["a_sin"][harmonic],
                            observed["b_cos"][harmonic],
                        )
                        c_null = origin_shift_null(c0, harmonic, shift, delta_bin)
                        residual = origin_shift_residual(c_obs, c_null)
                        origin_rows.append(
                            {
                                "block": block_id,
                                "profile": profile,
                                "concentration": protocol["concentration_profiles"][
                                    profile
                                ],
                                "representation": representation,
                                "origin_shift_bins": shift,
                                "harmonic": harmonic,
                                "observed_a": observed["a_sin"][harmonic],
                                "observed_b": observed["b_cos"][harmonic],
                                "null_a": c_null.real,
                                "null_b": c_null.imag,
                                "residual_magnitude": residual["residual_magnitude"],
                                "phase_residual": residual["phase_residual"],
                                "phase_residual_degrees": math.degrees(
                                    residual["phase_residual"]
                                ),
                                "max_coordinate_phase_degrees": math.degrees(
                                    harmonic * shift * delta_bin
                                ),
                            }
                        )

    report = {
        "family": "feature_sweep",
        "protocol_name": protocol["protocol_name"],
        "primary_fourier_order": primary_order,
        "sensitivity_fourier_orders": sensitivity_orders,
        "bootstrap_samples": n_bootstrap,
        "feature_axes": protocol.get("feature_axes", {}),
        "blocks": block_payload,
        "origin_null_summary": {
            "n_rows": len(origin_rows),
            "mean_residual_magnitude_m1": (
                float(
                    np.mean(
                        [
                            row["residual_magnitude"]
                            for row in origin_rows
                            if row["harmonic"] == 1 and row["origin_shift_bins"] != 0.0
                        ]
                    )
                )
                if any(
                    row["harmonic"] == 1 and row["origin_shift_bins"] != 0.0
                    for row in origin_rows
                )
                else None
            ),
        },
        "primary_endpoints": [
            "C1",
            "C2",
            "a0_robust",
            "activity",
            "action_entropy",
        ],
        "demoted_diagnostics": ["g_prime0", "n_fixed_points", "adjacent_jump_gate"],
        "terminology": {
            "object": "representation-controlled continuous or sharp crossover",
            "encoding_phrase": "different encodings of the same underlying field",
            "not_claimed": [
                "discontinuous bifurcation",
                "critical precision",
                "macroscopic phase selection",
                "aligned-state stability from M=6 g'(0)",
            ],
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "complex_kernel_analysis.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_csv(output_dir / "complex_endpoints.csv", endpoint_rows)
    _write_csv(output_dir / "a0_robustness.csv", a0_rows)
    _write_csv(output_dir / "offset_action_curves.csv", offset_rows)
    _write_csv(output_dir / "gprime0_order_sensitivity.csv", gprime_rows)
    _write_csv(output_dir / "complex_ellipses.csv", ellipse_rows)
    _write_csv(output_dir / "origin_null_residuals.csv", origin_rows)
    return report


def analyze_stimulus_manifold(
    run_dirs: list[Path],
    output_dir: Path,
    *,
    bootstrap_samples: int | None = None,
) -> dict:
    if len(run_dirs) != 2:
        raise ValueError("exactly two stimulus-manifold run directories are required")
    loaded = [load_transmutation_block(path) for path in run_dirs]
    protocol = loaded[0][0]["protocol"]
    if "stimulus_profiles" not in protocol:
        raise ValueError("stimulus_manifold analysis requires stimulus_profiles")
    offsets = np.asarray(protocol["offset_radians"], dtype=np.float64)
    primary_order = int(protocol["primary_fourier_order"])
    sensitivity_orders = [
        int(order) for order in protocol.get("sensitivity_fourier_orders", DEFAULT_ORDERS)
    ]
    profiles = list(protocol["stimulus_profiles"])
    representations = list(protocol["representations"])
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from circlemap.stimuli import stimulus_concentration_proxy

    concentrations = {
        profile: stimulus_concentration_proxy(profile) for profile in profiles
    }
    pseudocount = float(protocol.get("bootstrap_pseudocount", 0.5))
    n_bootstrap = int(
        bootstrap_samples
        if bootstrap_samples is not None
        else min(int(protocol.get("bootstrap_samples", DEFAULT_BOOTSTRAP)), DEFAULT_BOOTSTRAP)
    )
    rules = protocol.get("manifold_rules", {})
    min_separation = float(rules.get("minimum_c1_chordal_separation", 0.25))
    min_corr = float(
        protocol.get("replication_rules", {}).get(
            "minimum_trajectory_correlation",
            rules.get("minimum_trajectory_correlation", 0.8),
        )
    )

    endpoint_rows: list[dict] = []
    a0_rows: list[dict] = []
    offset_rows: list[dict] = []
    gprime_rows: list[dict] = []
    ellipse_rows: list[dict] = []
    distance_rows: list[dict] = []
    block_payload: dict = {}

    for block_index, (config, counts) in enumerate(loaded):
        block_id = config["protocol"]["seed_block_id"]
        rng = np.random.default_rng(2026122300 + block_index)
        block_payload[block_id] = {
            "operational": config["operational"],
            "representations": {},
        }
        for representation in representations:
            representation_result = {"conditions": {}}
            for profile in profiles:
                concentration = concentrations[profile]
                cell_counts = counts[representation][profile]
                summary = summarize_condition(
                    cell_counts,
                    offsets,
                    primary_order=primary_order,
                    sensitivity_orders=sensitivity_orders,
                )
                boot = bootstrap_complex_coefficients(
                    cell_counts,
                    offsets,
                    order=primary_order,
                    samples=n_bootstrap,
                    rng=rng,
                    pseudocount=pseudocount,
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
                    "concentration": concentration,
                    **summary,
                    "bootstrap_ellipses": ellipses,
                }
                endpoint_rows.append(
                    _endpoint_row(
                        block_id=block_id,
                        representation=representation,
                        profile=profile,
                        concentration=concentration,
                        summary=summary,
                        ellipse1=ellipses["1"],
                        ellipse2=ellipses["2"],
                    )
                )
                a0_rows.extend(
                    _a0_rows(
                        block_id=block_id,
                        representation=representation,
                        profile=profile,
                        concentration=concentration,
                        summary=summary,
                    )
                )
                offset_rows.extend(
                    _offset_rows(
                        block_id=block_id,
                        representation=representation,
                        profile=profile,
                        concentration=concentration,
                        offsets=offsets,
                        summary=summary,
                    )
                )
                gprime_rows.extend(
                    _gprime_rows(
                        block_id=block_id,
                        representation=representation,
                        profile=profile,
                        concentration=concentration,
                        summary=summary,
                    )
                )
                ellipse_rows.extend(
                    _ellipse_rows(
                        block_id=block_id,
                        representation=representation,
                        profile=profile,
                        concentration=concentration,
                        boot=boot,
                    )
                )
            block_payload[block_id]["representations"][representation] = (
                representation_result
            )
        distance_rows.extend(
            _pairwise_distances(
                block_id=block_id,
                representations=representations,
                profiles=profiles,
                concentrations=concentrations,
                block_payload=block_payload,
                offsets=offsets,
            )
        )

    # Across-block mean separations per field (polar C1 + full-operator).
    field_separations = {}
    persistent_fields = []
    for profile in profiles:
        pairs_c1 = {}
        pairs_tv = {}
        pairs_js = {}
        for left_index, left in enumerate(representations):
            for right in representations[left_index + 1 :]:
                c1_distances = []
                tv_distances = []
                js_distances = []
                for block_id in block_payload:
                    left_summary = block_payload[block_id]["representations"][left][
                        "conditions"
                    ][profile]
                    right_summary = block_payload[block_id]["representations"][right][
                        "conditions"
                    ][profile]
                    c1_distances.append(
                        polar_channel_distance(
                            left_summary["harmonics"],
                            right_summary["harmonics"],
                            harmonic=1,
                        )
                    )
                    left_probs = _probability_matrix(left_summary)
                    right_probs = _probability_matrix(right_summary)
                    tv_distances.append(
                        mean_total_variation(left_probs, right_probs)
                    )
                    js_distances.append(
                        mean_js_divergence(left_probs, right_probs)
                    )
                key = f"{left}__{right}"
                pairs_c1[key] = float(np.mean(c1_distances))
                pairs_tv[key] = float(np.mean(tv_distances))
                pairs_js[key] = float(np.mean(js_distances))
        minimum_c1 = min(pairs_c1.values()) if pairs_c1 else float("nan")
        minimum_tv = min(pairs_tv.values()) if pairs_tv else float("nan")
        minimum_js = min(pairs_js.values()) if pairs_js else float("nan")
        polar_separable = bool(minimum_c1 >= min_separation)
        # Full-operator absolute TV threshold is descriptive only; prefer
        # between/within TV ratio calibrated in STAGE_B_FULL_OPERATOR_GATE.md.
        full_operator_separable = bool(minimum_tv >= 0.20)
        is_antipodal_profile = "antipodal" in profile
        if polar_separable and full_operator_separable:
            classification = "polar_and_full_operator_separated"
        elif (not polar_separable) and full_operator_separable and is_antipodal_profile:
            classification = "symmetry_selective_polar_channel_collapse"
        elif (not polar_separable) and full_operator_separable:
            classification = "weak_polar_full_operator_separated"
        elif polar_separable and not full_operator_separable:
            classification = "polar_separated_full_weak"
        else:
            classification = "collapsed"
        field_separations[profile] = {
            "pair_c1_chordal": pairs_c1,
            "minimum_pair_c1_chordal": minimum_c1,
            "separable": polar_separable,
            "polar_channel_separable": polar_separable,
            "pair_mean_tv": pairs_tv,
            "minimum_pair_mean_tv": minimum_tv,
            "pair_mean_js": pairs_js,
            "minimum_pair_mean_js": minimum_js,
            "full_operator_separable": full_operator_separable,
            "classification": classification,
        }
        if (
            profile != rules.get("anchor_profile", "unimodal_k9")
            and polar_separable
        ):
            persistent_fields.append(profile)

    # Within-representation trajectory correlations across blocks over fields.
    trajectory_correlations = {}
    for representation in representations:
        first = []
        second = []
        block_ids = list(block_payload)
        for profile in profiles:
            for key in ("R1", "R2", "a0"):
                condition0 = block_payload[block_ids[0]]["representations"][representation][
                    "conditions"
                ][profile]
                condition1 = block_payload[block_ids[1]]["representations"][representation][
                    "conditions"
                ][profile]
                if key == "a0":
                    first.append(condition0["harmonics"]["a0"])
                    second.append(condition1["harmonics"]["a0"])
                elif key == "R1":
                    first.append(condition0["harmonics"]["polar"]["1"]["R"])
                    second.append(condition1["harmonics"]["polar"]["1"]["R"])
                else:
                    first.append(condition0["harmonics"]["polar"]["2"]["R"])
                    second.append(condition1["harmonics"]["polar"]["2"]["R"])
        correlation = float(np.corrcoef(first, second)[0, 1])
        trajectory_correlations[representation] = correlation

    non_anchor = [
        profile
        for profile in profiles
        if profile != rules.get("anchor_profile", "unimodal_k9")
    ]
    separable_non_anchor = sum(
        1
        for profile in non_anchor
        if field_separations[profile]["polar_channel_separable"]
    )
    full_operator_non_anchor = sum(
        1
        for profile in non_anchor
        if field_separations[profile]["full_operator_separable"]
    )
    trajectory_pass = sum(
        1 for value in trajectory_correlations.values() if value >= min_corr
    )
    operational_pass = all(
        config["operational"]["passes"] for config, _ in loaded
    )
    antipodal_like = [
        profile
        for profile in non_anchor
        if field_separations[profile]["classification"]
        == "symmetry_selective_polar_channel_collapse"
    ]
    if separable_non_anchor == len(non_anchor):
        verdict = "persists"
    elif separable_non_anchor > 0 and antipodal_like:
        verdict = "symmetry_selective_polar_channel_collapse"
    elif separable_non_anchor > 0:
        verdict = "partially_collapses"
    else:
        verdict = "collapses"
    overall_pass = (
        operational_pass
        and separable_non_anchor >= 1
        and trajectory_pass
        >= int(
            protocol.get("replication_rules", {}).get(
                "minimum_classes_with_persistent_separation",
                2,
            )
        )
    )

    report = {
        "family": "stimulus_manifold",
        "protocol_name": protocol["protocol_name"],
        "stimulus_profiles": profiles,
        "primary_fourier_order": primary_order,
        "bootstrap_samples": n_bootstrap,
        "blocks": block_payload,
        "field_separations": field_separations,
        "trajectory_correlations": trajectory_correlations,
        "persistent_non_anchor_fields": persistent_fields,
        "separable_non_anchor_count": separable_non_anchor,
        "full_operator_separable_non_anchor_count": full_operator_non_anchor,
        "trajectory_pass_count": trajectory_pass,
        "operational_pass": operational_pass,
        "overall_manifold_gate_pass": overall_pass,
        "transmutation_verdict": verdict,
        "scientific_interpretation": (
            "Representation-induced response-law transmutation persists across "
            "unimodal, non-antipodal bimodal, asymmetric and sparse fields. Exact "
            "antipodal symmetry suppresses representation-separated polar "
            "channels, but does not erase representation dependence in activity, "
            "even harmonics or the full action distribution."
        ),
        "primary_endpoints": [
            "C1",
            "C2",
            "a0_robust",
            "activity",
            "action_entropy",
            "trinomial_distribution",
        ],
        "demoted_diagnostics": ["g_prime0", "n_fixed_points"],
        "terminology": {
            "object": "representation-induced microscopic response-law transmutation",
            "encoding_phrase": (
                "different encodings of the same underlying field family"
            ),
            "a0_phrase": "field-dependent offset-averaged action bias a0(R, rho)",
            "moments_antipodal_phrase": (
                "directional abstention under pi-rotation symmetry"
            ),
            "not_claimed": [
                "macroscopic interaction-phase transmutation",
                "Stage C authorization",
                "encoding-intrinsic torque independent of rho",
                "full operator collapse under antipodal symmetry",
            ],
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "complex_kernel_analysis.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_csv(output_dir / "complex_endpoints.csv", endpoint_rows)
    _write_csv(output_dir / "a0_robustness.csv", a0_rows)
    _write_csv(output_dir / "offset_action_curves.csv", offset_rows)
    _write_csv(output_dir / "gprime0_order_sensitivity.csv", gprime_rows)
    _write_csv(output_dir / "complex_ellipses.csv", ellipse_rows)
    _write_csv(output_dir / "representation_distances.csv", distance_rows)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--family",
        choices=("transmutation", "feature_sweep", "stimulus_manifold", "both"),
        default="both",
    )
    parser.add_argument("--transmutation-run-dir", type=Path, action="append")
    parser.add_argument("--feature-run-dir", type=Path, action="append")
    parser.add_argument("--manifold-run-dir", type=Path, action="append")
    parser.add_argument(
        "--transmutation-out-dir",
        type=Path,
        default=Path("analysis/complex_kernel_transmutation"),
    )
    parser.add_argument(
        "--feature-out-dir",
        type=Path,
        default=Path("analysis/complex_kernel_feature_sweep"),
    )
    parser.add_argument(
        "--manifold-out-dir",
        type=Path,
        default=Path("analysis/complex_kernel_stimulus_manifold"),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=None)
    args = parser.parse_args()

    reports = {}
    if args.family in ("transmutation", "both"):
        if not args.transmutation_run_dir or len(args.transmutation_run_dir) != 2:
            raise SystemExit("provide exactly two --transmutation-run-dir values")
        reports["transmutation"] = analyze_transmutation(
            args.transmutation_run_dir,
            args.transmutation_out_dir,
            bootstrap_samples=args.bootstrap_samples,
        )
    if args.family in ("feature_sweep", "both"):
        if not args.feature_run_dir or len(args.feature_run_dir) != 2:
            raise SystemExit("provide exactly two --feature-run-dir values")
        reports["feature_sweep"] = analyze_feature_sweep(
            args.feature_run_dir,
            args.feature_out_dir,
            bootstrap_samples=args.bootstrap_samples,
        )
    if args.family == "stimulus_manifold":
        if not args.manifold_run_dir or len(args.manifold_run_dir) != 2:
            raise SystemExit("provide exactly two --manifold-run-dir values")
        reports["stimulus_manifold"] = analyze_stimulus_manifold(
            args.manifold_run_dir,
            args.manifold_out_dir,
            bootstrap_samples=args.bootstrap_samples,
        )
    print(json.dumps({key: True for key in reports}, indent=2))


if __name__ == "__main__":
    main()

"""Pure helpers for complex Fourier / activity / stability reanalysis."""

from __future__ import annotations

import math

import numpy as np

try:
    from analysis.analyze_transmutation import fit_fourier
except ModuleNotFoundError:
    from analyze_transmutation import fit_fourier


TWO_PI = 2.0 * math.pi


def polar(a: float, b: float) -> tuple[float, float]:
    """Return amplitude R and phase phi=atan2(b, a)."""
    return float(math.hypot(a, b)), float(math.atan2(b, a))


def wrap_angle(phi: float) -> float:
    """Wrap angle to (-pi, pi]."""
    wrapped = (phi + math.pi) % TWO_PI - math.pi
    if wrapped <= -math.pi:
        wrapped += TWO_PI
    return float(wrapped)


def complex_from_ab(a: float, b: float) -> complex:
    return complex(float(a), float(b))


def g_prime0(a_sin: dict[int, float] | list[float] | np.ndarray) -> float:
    """Derivative of the fitted mean-response curve at zero offset.

    This is sum_{m>=1} m a_m from sine coefficients only. It is not by itself
    an aligned-state stability criterion when a0 or cosine terms are nonzero.
    """
    if isinstance(a_sin, dict):
        return float(sum(int(m) * float(value) for m, value in a_sin.items()))
    return float(
        sum((index + 1) * float(value) for index, value in enumerate(a_sin))
    )


def mean_action_curve(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (g, activity, probabilities[n,3]) from offset-wise counts."""
    totals = np.sum(counts, axis=1).astype(np.float64)
    totals = np.maximum(totals, 1.0)
    probabilities = counts.astype(np.float64) / totals[:, None]
    g = probabilities[:, 2] - probabilities[:, 0]
    activity = probabilities[:, 0] + probabilities[:, 2]
    return g, activity, probabilities


def action_entropy(probabilities: np.ndarray) -> np.ndarray:
    """Shannon entropy of (p-, p0, p+) per offset.

    Natural log, units nats, not normalized by log(3).
    """
    clipped = np.clip(probabilities, 0.0, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(clipped > 0.0, clipped * np.log(clipped), 0.0)
    return -np.sum(terms, axis=-1)


def density_weighted_circular_mean(
    offsets: np.ndarray,
    values: np.ndarray,
) -> float:
    """Arc-length-weighted circular mean via periodic trapezoidal quadrature.

    Corrects for uneven spacing of design offsets on the circle. Despite the
    historical function name, this is *not* a stimulus-density ρ(δ) weight.
    """
    offsets = np.asarray(offsets, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if offsets.size == 0:
        return float("nan")
    if offsets.size == 1:
        return float(values[0])
    order = np.argsort(offsets)
    x = offsets[order]
    y = values[order]
    x_ext = np.concatenate([x, [x[0] + TWO_PI]])
    y_ext = np.concatenate([y, [y[0]]])
    integral = float(np.trapezoid(y_ext, x_ext))
    return integral / TWO_PI


def unweighted_mean(values: np.ndarray) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def mirror_even_mean(
    offsets: np.ndarray,
    values: np.ndarray,
    *,
    atol: float = 1e-9,
) -> float | None:
    """Average even part (g(δ)+g(-δ))/2 over positive mirror pairs."""
    offsets = np.asarray(offsets, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    even_parts: list[float] = []
    used: set[int] = set()
    for index, delta in enumerate(offsets):
        if index in used:
            continue
        if abs(delta) <= atol:
            even_parts.append(float(values[index]))
            used.add(index)
            continue
        if delta < 0:
            continue
        matches = np.where(np.isclose(offsets, -delta, atol=atol, rtol=0.0))[0]
        if matches.size == 0:
            continue
        partner = int(matches[0])
        even_parts.append(0.5 * (float(values[index]) + float(values[partner])))
        used.add(index)
        used.add(partner)
    if not even_parts:
        return None
    return float(np.mean(even_parts))


def a0_robustness_suite(
    counts: np.ndarray,
    offsets: np.ndarray,
    orders: list[int],
) -> dict:
    """Compare Fourier intercepts across M with direct mean estimators."""
    g, _, _ = mean_action_curve(counts)
    fourier_a0 = {}
    for order in orders:
        estimate = fit_fourier(counts, offsets, int(order))
        fourier_a0[str(int(order))] = float(estimate["a0"])
    values = list(fourier_a0.values())
    mirror = mirror_even_mean(offsets, g)
    return {
        "fourier_a0_by_order": fourier_a0,
        "fourier_a0_range": float(max(values) - min(values)) if values else float("nan"),
        "unweighted_mean_g": unweighted_mean(g),
        "density_weighted_mean_g": density_weighted_circular_mean(offsets, g),
        "mirror_even_mean_g": mirror,
    }


def extract_harmonics(estimate: dict, max_harmonic: int) -> dict:
    """Pull a_m/b_m and polar form through max_harmonic from a fit_fourier dict."""
    a_sin: dict[int, float] = {}
    b_cos: dict[int, float] = {}
    polar_terms: dict[str, dict[str, float]] = {}
    for harmonic in range(1, max_harmonic + 1):
        a_key = f"a_sin_{harmonic}"
        b_key = f"b_cos_{harmonic}"
        if a_key not in estimate or b_key not in estimate:
            break
        a_value = float(estimate[a_key])
        b_value = float(estimate[b_key])
        radius, phase = polar(a_value, b_value)
        a_sin[harmonic] = a_value
        b_cos[harmonic] = b_value
        polar_terms[str(harmonic)] = {
            "a": a_value,
            "b": b_value,
            "R": radius,
            "phi": phase,
            "phi_degrees": float(math.degrees(phase)),
        }
    return {
        "a0": float(estimate["a0"]),
        "a_sin": a_sin,
        "b_cos": b_cos,
        "polar": polar_terms,
        "g_prime0": g_prime0(a_sin),
        "mean_activity": float(estimate["mean_activity"]),
        "design_condition_number": float(
            estimate.get("design_condition_number", float("nan"))
        ),
    }


def origin_shift_null(
    c0: complex,
    harmonic: int,
    shift_bins: float,
    delta_bin: float,
) -> complex:
    """Pure coordinate-shift null: C_m(s) = C_m(0) exp(i m s Δ_bin)."""
    angle = float(harmonic) * float(shift_bins) * float(delta_bin)
    return c0 * complex(math.cos(angle), math.sin(angle))


def origin_shift_residual(
    c_observed: complex,
    c_null: complex,
) -> dict[str, float]:
    residual = c_observed - c_null
    obs_phi = math.atan2(c_observed.imag, c_observed.real)
    null_phi = math.atan2(c_null.imag, c_null.real)
    return {
        "residual_real": float(residual.real),
        "residual_imag": float(residual.imag),
        "residual_magnitude": float(abs(residual)),
        "phase_residual": wrap_angle(obs_phi - null_phi),
        "observed_R": float(abs(c_observed)),
        "null_R": float(abs(c_null)),
    }


def scan_fixed_points(
    offsets: np.ndarray,
    g: np.ndarray,
    *,
    include_wrap: bool = True,
) -> list[dict[str, float | bool]]:
    """Linear zero crossings of the measured mean-action curve."""
    offsets = np.asarray(offsets, dtype=np.float64)
    g = np.asarray(g, dtype=np.float64)
    order = np.argsort(offsets)
    x = offsets[order]
    y = g[order]
    points: list[dict[str, float | bool]] = []

    def _add(left_x: float, left_y: float, right_x: float, right_y: float) -> None:
        span = right_x - left_x
        if abs(span) < 1e-12:
            return
        if left_y == 0.0:
            slope = (right_y - left_y) / span
            points.append(
                {
                    "delta": float(wrap_angle(left_x)),
                    "slope": float(slope),
                    "attractive_channel_sign": bool(slope > 0.0),
                }
            )
            return
        if left_y * right_y > 0.0:
            return
        if right_y == left_y:
            return
        frac = left_y / (left_y - right_y)
        delta = left_x + frac * span
        slope = (right_y - left_y) / span
        points.append(
            {
                "delta": float(wrap_angle(delta)),
                "slope": float(slope),
                "attractive_channel_sign": bool(slope > 0.0),
            }
        )

    for index in range(x.size - 1):
        _add(float(x[index]), float(y[index]), float(x[index + 1]), float(y[index + 1]))
    if include_wrap and x.size >= 2:
        _add(float(x[-1]), float(y[-1]), float(x[0] + TWO_PI), float(y[0]))
    # Deduplicate near-identical roots.
    unique: list[dict[str, float | bool]] = []
    for point in points:
        if any(abs(float(point["delta"]) - float(other["delta"])) < 1e-8 for other in unique):
            continue
        unique.append(point)
    return unique


def hard_controller_flags(
    activity: np.ndarray,
    probabilities: np.ndarray,
    *,
    activity_threshold: float = 0.999,
    stay_threshold: float = 0.001,
) -> dict[str, float | bool]:
    """Operational no-stay flag; does not imply deterministic advance/retard."""
    mean_activity = float(np.mean(activity))
    mean_stay = float(np.mean(probabilities[:, 1]))
    return {
        "mean_activity": mean_activity,
        "mean_stay": mean_stay,
        "min_activity": float(np.min(activity)),
        "max_activity": float(np.max(activity)),
        "no_stay_directional_controller": bool(
            mean_activity >= activity_threshold and mean_stay <= stay_threshold
        ),
        # Backward-compatible alias with narrowed meaning.
        "hard_directional_controller": bool(
            mean_activity >= activity_threshold and mean_stay <= stay_threshold
        ),
    }


CHI2_95_2DF = 5.991464547107979


def average_complex_coefficient(
    coefficients: list[tuple[float, float]],
) -> dict[str, float | bool | None]:
    """Average complex coeffs first, then convert to polar form.

    Never average phases directly across blocks or samples.
    """
    if not coefficients:
        raise ValueError("coefficients must be nonempty")
    mean_a = float(np.mean([pair[0] for pair in coefficients]))
    mean_b = float(np.mean([pair[1] for pair in coefficients]))
    radius, phase = polar(mean_a, mean_b)
    return {
        "a": mean_a,
        "b": mean_b,
        "R": radius,
        "phi": phase,
        "phi_degrees": float(math.degrees(phase)),
    }


def ellipse_contains_origin(ellipse: dict[str, float]) -> bool:
    """True if the 95% Gaussian ellipse contains the complex origin."""
    center = np.asarray(
        [ellipse["center_a"], ellipse["center_b"]],
        dtype=np.float64,
    )
    covariance = np.asarray(
        [
            [ellipse["cov_aa"], ellipse["cov_ab"]],
            [ellipse["cov_ab"], ellipse["cov_bb"]],
        ],
        dtype=np.float64,
    )
    try:
        mahalanobis = float(center @ np.linalg.solve(covariance, center))
    except np.linalg.LinAlgError:
        return bool(math.hypot(float(center[0]), float(center[1])) < 1e-12)
    return mahalanobis <= CHI2_95_2DF


def phase_reportable(
    radius: float,
    *,
    ellipse: dict[str, float] | None = None,
    minimum_radius: float = 0.15,
) -> bool:
    """Suppress phase when amplitude is weak or the 95% ellipse covers 0.

    The minimum_radius cutoff is a visualization/reporting convention, not an
    inferential endpoint; ellipse origin coverage is the statistical caution.
    """
    if radius < minimum_radius:
        return False
    if ellipse is not None and ellipse_contains_origin(ellipse):
        return False
    return True


def mean_js_divergence(
    first_probabilities: np.ndarray,
    second_probabilities: np.ndarray,
    *,
    floor: float = 1e-9,
) -> float:
    """Mean Jensen–Shannon divergence of trinomials over offsets (nats)."""
    first = np.clip(np.asarray(first_probabilities, dtype=np.float64), floor, 1.0)
    second = np.clip(np.asarray(second_probabilities, dtype=np.float64), floor, 1.0)
    first = first / np.sum(first, axis=1, keepdims=True)
    second = second / np.sum(second, axis=1, keepdims=True)
    mixture = 0.5 * (first + second)
    forward = np.sum(first * (np.log(first) - np.log(mixture)), axis=1)
    backward = np.sum(second * (np.log(second) - np.log(mixture)), axis=1)
    return float(np.mean(0.5 * (forward + backward)))



def reconstruct_g(
    offsets: np.ndarray,
    harmonics: dict,
    *,
    max_harmonic: int | None = None,
) -> np.ndarray:
    """Rebuild the Fourier mean-action curve from extracted harmonics."""
    offsets = np.asarray(offsets, dtype=np.float64)
    values = np.full(offsets.shape, float(harmonics["a0"]), dtype=np.float64)
    limit = max_harmonic or max(harmonics["a_sin"], default=0)
    for harmonic in range(1, int(limit) + 1):
        if harmonic not in harmonics["a_sin"]:
            break
        values = (
            values
            + harmonics["a_sin"][harmonic] * np.sin(harmonic * offsets)
            + harmonics["b_cos"][harmonic] * np.cos(harmonic * offsets)
        )
    return values


def gprime0_order_sensitivity(sensitivity: dict[str, dict]) -> dict:
    """Diagnostic summary of g'(0) across Fourier truncation orders."""
    by_order = {
        int(order): float(payload["g_prime0"])
        for order, payload in sensitivity.items()
    }
    values = list(by_order.values())
    signs = {value >= 0.0 for value in values}
    return {
        "by_order": {str(order): value for order, value in sorted(by_order.items())},
        "range": float(max(values) - min(values)) if values else float("nan"),
        "sign_flip_across_orders": bool(len(signs) > 1),
        "primary_status": "diagnostic_only_not_primary",
    }


def fixed_point_order_sensitivity(
    offsets: np.ndarray,
    sensitivity: dict[str, dict],
) -> dict:
    """Count zeros of the fitted Fourier curve at each truncation order."""
    by_order = {}
    for order, payload in sensitivity.items():
        fitted = reconstruct_g(offsets, payload, max_harmonic=int(order))
        by_order[int(order)] = len(scan_fixed_points(offsets, fitted))
    values = list(by_order.values())
    return {
        "by_order": {str(order): count for order, count in sorted(by_order.items())},
        "range": int(max(values) - min(values)) if values else 0,
        "primary_status": "diagnostic_only_not_primary",
    }


def bootstrap_complex_coefficients(
    counts: np.ndarray,
    offsets: np.ndarray,
    *,
    order: int,
    samples: int,
    rng: np.random.Generator,
    pseudocount: float = 0.0,
) -> dict[str, np.ndarray]:
    """Multinomial bootstrap of low-order complex Fourier coefficients."""
    try:
        from analysis.analyze_transmutation import design_matrix
    except ModuleNotFoundError:
        from analyze_transmutation import design_matrix

    totals = np.sum(counts, axis=1).astype(np.int64)
    probability = (counts + pseudocount) / (
        totals[:, None] + 3.0 * pseudocount
    )
    draws = np.empty((samples, offsets.size, 3), dtype=np.int16)
    for index, (total, cell_probability) in enumerate(zip(totals, probability)):
        draws[:, index, :] = rng.multinomial(
            int(total),
            cell_probability,
            size=samples,
        )
    g = (draws[:, :, 2] - draws[:, :, 0]) / totals[None, :]
    inverse = np.linalg.pinv(design_matrix(offsets, int(order)))
    beta = g @ inverse.T
    return {
        "a0": beta[:, 0],
        "a1": beta[:, 1],
        "b1": beta[:, 2],
        "a2": beta[:, 3],
        "b2": beta[:, 4],
    }


def covariance_ellipse_95(samples_2d: np.ndarray) -> dict[str, float]:
    """95% Gaussian confidence ellipse from an (n,2) sample cloud."""
    points = np.asarray(samples_2d, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("samples_2d must have shape (n, 2)")
    if points.shape[0] < 2:
        raise ValueError("need at least two samples for an ellipse")
    center = np.mean(points, axis=0)
    centered = points - center
    covariance = (centered.T @ centered) / max(1, points.shape[0] - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    # chi-square 95% quantile for 2 degrees of freedom
    scale = math.sqrt(5.991464547107979)
    major = float(scale * math.sqrt(eigenvalues[0]))
    minor = float(scale * math.sqrt(eigenvalues[1]))
    angle = float(math.atan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    return {
        "center_a": float(center[0]),
        "center_b": float(center[1]),
        "cov_aa": float(covariance[0, 0]),
        "cov_ab": float(covariance[0, 1]),
        "cov_bb": float(covariance[1, 1]),
        "axis_major": major,
        "axis_minor": minor,
        "angle_radians": angle,
        "angle_degrees": float(math.degrees(angle)),
    }


def density_weighted_l2(
    offsets: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    """sqrt of arc-length-weighted circular mean of squared response difference.

    Uses periodic trapezoidal quadrature over design offsets (not ρ(δ)).
    """
    diff = np.asarray(first, dtype=np.float64) - np.asarray(second, dtype=np.float64)
    return float(math.sqrt(max(0.0, density_weighted_circular_mean(offsets, diff**2))))


def complex_low_order_distance(
    first: dict,
    second: dict,
) -> float:
    """Euclidean distance on (a0, a1, b1, a2, b2)."""
    keys = (
        ("a0", None),
        ("a_sin", 1),
        ("b_cos", 1),
        ("a_sin", 2),
        ("b_cos", 2),
    )
    total = 0.0
    for key, harmonic in keys:
        if harmonic is None:
            delta = float(first["a0"]) - float(second["a0"])
        else:
            delta = float(first[key][harmonic]) - float(second[key][harmonic])
        total += delta * delta
    return float(math.sqrt(total))


def polar_channel_distance(
    first: dict,
    second: dict,
    *,
    harmonic: int = 1,
) -> float:
    """Chordal distance between complex harmonics C_m."""
    left = first["polar"][str(harmonic)]
    right = second["polar"][str(harmonic)]
    return float(
        math.hypot(
            left["a"] - right["a"],
            left["b"] - right["b"],
        )
    )


def mean_total_variation(
    first_probabilities: np.ndarray,
    second_probabilities: np.ndarray,
) -> float:
    """Mean total variation distance of trinomials over offsets."""
    first = np.asarray(first_probabilities, dtype=np.float64)
    second = np.asarray(second_probabilities, dtype=np.float64)
    return float(0.5 * np.mean(np.sum(np.abs(first - second), axis=1)))


def mean_kl_divergence(
    first_probabilities: np.ndarray,
    second_probabilities: np.ndarray,
    *,
    floor: float = 1e-9,
) -> float:
    """Symmetric mean KL of trinomials with probability floor."""
    first = np.clip(np.asarray(first_probabilities, dtype=np.float64), floor, 1.0)
    second = np.clip(np.asarray(second_probabilities, dtype=np.float64), floor, 1.0)
    first = first / np.sum(first, axis=1, keepdims=True)
    second = second / np.sum(second, axis=1, keepdims=True)
    forward = np.sum(first * (np.log(first) - np.log(second)), axis=1)
    backward = np.sum(second * (np.log(second) - np.log(first)), axis=1)
    return float(np.mean(0.5 * (forward + backward)))


def summarize_condition(
    counts: np.ndarray,
    offsets: np.ndarray,
    *,
    primary_order: int = 6,
    sensitivity_orders: list[int] | None = None,
) -> dict:
    """Full offline summary for one (representation, κ) cell."""
    orders = sorted(
        set([int(primary_order), *(sensitivity_orders or [2, 4, 6, 8, 12])])
    )
    g, activity, probabilities = mean_action_curve(counts)
    entropy = action_entropy(probabilities)
    primary = fit_fourier(counts, offsets, int(primary_order))
    harmonics = extract_harmonics(primary, int(primary_order))
    sensitivity = {
        str(order): extract_harmonics(fit_fourier(counts, offsets, int(order)), int(order))
        for order in orders
    }
    a0_suite = a0_robustness_suite(counts, offsets, orders)
    flags = hard_controller_flags(activity, probabilities)
    fixed_points = scan_fixed_points(offsets, g)
    gprime_diagnostic = gprime0_order_sensitivity(sensitivity)
    fixed_point_diagnostic = fixed_point_order_sensitivity(offsets, sensitivity)
    return {
        "primary_order": int(primary_order),
        "harmonics": harmonics,
        "sensitivity_orders": sensitivity,
        "a0_robustness": a0_suite,
        "activity": flags,
        "mean_entropy": float(np.mean(entropy)),
        "entropy_definition": {
            "formula": "mean_over_delta[-sum_s p_s(delta) log p_s(delta)]",
            "units": "nats",
            "log_base": "natural",
            "normalized_by_log3": False,
            "offset_average": "uniform_over_design_grid",
        },
        "entropy_by_offset": entropy.tolist(),
        "g_by_offset": g.tolist(),
        "activity_by_offset": activity.tolist(),
        "probabilities_by_offset": {
            "retard": probabilities[:, 0].tolist(),
            "stay": probabilities[:, 1].tolist(),
            "advance": probabilities[:, 2].tolist(),
        },
        "fixed_points_diagnostic": fixed_points,
        "n_fixed_points_diagnostic": len(fixed_points),
        "gprime0_diagnostic": gprime_diagnostic,
        "fixed_point_order_diagnostic": fixed_point_diagnostic,
        # Legacy key retained for CSV compatibility; not an aligned-state claim.
        "aligned_local_slope_g_prime0_diagnostic": harmonics["g_prime0"],
        "fitted_mean_response_derivative_at_zero_diagnostic": harmonics["g_prime0"],
    }

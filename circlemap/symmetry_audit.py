"""Serialization-level symmetry audits for antipodal / ε-breaking stimuli."""

from __future__ import annotations

import math
import re

import numpy as np

from circlemap.observation import RelativePhaseHistogram
from circlemap.representations import build_representation_prompt_from_histogram
from circlemap.stimuli import StimulusSpec, build_stimulus_histogram


_MOMENT_LINE = re.compile(
    r"^moment_(\d+)_(cos|sin):\s*([+-]?\d+(?:\.\d+)?)\s*$"
)


def half_turn_bin_permutation(n_bins: int) -> np.ndarray:
    """Index map sending bin centers θ → θ+π on a uniform circular grid."""
    if n_bins % 2 != 0:
        raise ValueError("half-turn permutation requires even n_bins")
    shift = n_bins // 2
    return (np.arange(n_bins) + shift) % n_bins


def histogram_pi_period_residual(
    first: RelativePhaseHistogram,
    second: RelativePhaseHistogram,
) -> float:
    """L1 residual between histograms at δ and δ+π (equal antipodal ⇒ 0)."""
    if first.fractions.size != second.fractions.size:
        raise ValueError("bin counts must match")
    return float(np.sum(np.abs(first.fractions - second.fractions)))


def histogram_half_turn_invariance_residual(
    histogram: RelativePhaseHistogram,
) -> float:
    """L1 residual of f vs half-turn(f); π-symmetric densities are invariant."""
    perm = half_turn_bin_permutation(int(histogram.fractions.size))
    return float(np.sum(np.abs(histogram.fractions - histogram.fractions[perm])))


def parse_moment_prompt_payload(prompt: str) -> dict[tuple[int, str], float]:
    """Parse moment_m_cos/sin lines from a moments representation prompt."""
    values: dict[tuple[int, str], float] = {}
    for line in prompt.splitlines():
        match = _MOMENT_LINE.match(line.strip())
        if match is None:
            continue
        harmonic = int(match.group(1))
        part = match.group(2)
        values[(harmonic, part)] = float(match.group(3))
    if not values:
        raise ValueError("no moment lines found in prompt")
    return values


def moments_half_turn_string_residuals(
    prompt_delta: str,
    prompt_delta_plus_pi: str,
) -> dict[str, float]:
    """Check z_m(δ+π)=(-1)^m z_m(δ) at rounded prompt string values."""
    first = parse_moment_prompt_payload(prompt_delta)
    second = parse_moment_prompt_payload(prompt_delta_plus_pi)
    residuals = {}
    for (harmonic, part), value in first.items():
        sign = 1.0 if harmonic % 2 == 0 else -1.0
        expected = sign * value
        observed = second[(harmonic, part)]
        residuals[f"moment_{harmonic}_{part}"] = abs(expected - observed)
    return residuals


def antipodal_dense_spec(
    *,
    concentration: float = 6.0,
    epsilon: float = 0.0,
) -> StimulusSpec:
    """Equal or weight-imbalanced antipodal mixture (modes at 0 and π)."""
    weight = 0.5 + float(epsilon)
    if weight <= 0.0 or weight >= 1.0:
        raise ValueError("epsilon must keep both weights in (0,1)")
    return StimulusSpec(
        kind="mixture",
        concentration=float(concentration),
        mode_offsets=(0.0, float(np.pi)),
        weights=(weight, 1.0 - weight),
    )


def assert_exact_antipodal_serialization(
    *,
    offset: float = 0.4,
    concentration: float = 6.0,
    moment_decimals: int = 6,
) -> None:
    """Raise AssertionError if ε=0 antipodal serialization breaks π-laws."""
    spec = antipodal_dense_spec(concentration=concentration, epsilon=0.0)
    hist = build_stimulus_histogram(spec, offset)
    hist_pi = build_stimulus_histogram(spec, offset + math.pi)

    period_residual = histogram_pi_period_residual(hist, hist_pi)
    if period_residual > 1e-12:
        raise AssertionError(
            f"equal antipodal hist(δ) vs hist(δ+π) residual {period_residual}"
        )

    invariance = histogram_half_turn_invariance_residual(hist)
    if invariance > 1e-12:
        raise AssertionError(
            f"equal antipodal half-turn invariance residual {invariance}"
        )

    moments_name = "moments_m1_m3"
    prompt_m = build_representation_prompt_from_histogram(moments_name, hist)
    prompt_m_pi = build_representation_prompt_from_histogram(moments_name, hist_pi)
    residuals = moments_half_turn_string_residuals(prompt_m, prompt_m_pi)
    tol = 0.5 * 10 ** (-moment_decimals) + 1e-12
    bad = {key: value for key, value in residuals.items() if value > tol}
    if bad:
        raise AssertionError(f"moments half-turn string residuals: {bad}")

    # Odd moments should be ~0 after rounding for exact antipodal balance.
    parsed = parse_moment_prompt_payload(prompt_m)
    odd_bad = {
        f"moment_{h}_{part}": value
        for (h, part), value in parsed.items()
        if h % 2 == 1 and abs(value) > tol
    }
    if odd_bad:
        raise AssertionError(f"equal antipodal odd moments not ~0: {odd_bad}")


def assert_epsilon_prompts_not_collapsed(
    *,
    epsilon: float = 0.02,
    offset: float = 0.4,
    concentration: float = 6.0,
) -> None:
    """Small nonzero ε must not serialize identically to ε=0 for each encoding."""
    zero = antipodal_dense_spec(concentration=concentration, epsilon=0.0)
    tipped = antipodal_dense_spec(concentration=concentration, epsilon=epsilon)
    hist0 = build_stimulus_histogram(zero, offset)
    hist_eps = build_stimulus_histogram(tipped, offset)
    for name in (
        "intervals_24_decimal6",
        "centers_24_standard",
        "moments_m1_m3",
    ):
        p0 = build_representation_prompt_from_histogram(name, hist0)
        p_eps = build_representation_prompt_from_histogram(name, hist_eps)
        if p0 == p_eps:
            raise AssertionError(
                f"{name}: epsilon={epsilon} collapsed to identical prompt as epsilon=0"
            )


def assert_signed_epsilon_covariance_hint(
    *,
    epsilon: float = 0.10,
    offset: float = 0.4,
    concentration: float = 6.0,
) -> None:
    """ρ_{-ε,δ} should match ρ_{ε,δ+π} at the histogram level."""
    plus = antipodal_dense_spec(concentration=concentration, epsilon=epsilon)
    minus = antipodal_dense_spec(concentration=concentration, epsilon=-epsilon)
    hist_minus = build_stimulus_histogram(minus, offset)
    hist_plus_shifted = build_stimulus_histogram(plus, offset + math.pi)
    residual = histogram_pi_period_residual(hist_minus, hist_plus_shifted)
    if residual > 1e-12:
        raise AssertionError(
            f"signed-epsilon covariance residual {residual} "
            f"(expected ρ(-ε,δ)=ρ(+ε,δ+π))"
        )

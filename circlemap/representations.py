"""Equivalent relative-phase representations for invariance experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping

import numpy as np

from .observation import RelativePhaseHistogram, wrap_phase
from .response import (
    build_response_prompt,
    fixed_field_histogram,
)
from .stimuli import build_stimulus_histogram, coerce_stimulus_spec


@dataclass(frozen=True)
class RepresentationSpec:
    """A versionable operator that converts a phase field into LLM text."""

    kind: Literal["intervals", "centers", "counts", "moments"]
    n_bins: int = 24
    decimals: int = 6
    origin_shift_bins: float = 0.0
    count_total: int = 100
    moment_order: int = 3
    legacy_prompt: bool = False

    @property
    def canonical_id(self) -> str:
        if self.legacy_prompt:
            return f"intervals_n{self.n_bins}_d{self.decimals}_legacy"
        if self.kind == "intervals":
            return (
                f"intervals_n{self.n_bins}_d{self.decimals}"
                f"_o{self.origin_shift_bins:g}"
            )
        if self.kind == "centers":
            return (
                f"centers_n{self.n_bins}_d{self.decimals}"
                f"_o{self.origin_shift_bins:g}"
            )
        if self.kind == "counts":
            return (
                f"counts_n{self.n_bins}_total{self.count_total}"
                f"_o{self.origin_shift_bins:g}"
            )
        return f"moments_m1_m{self.moment_order}_d{self.decimals}"


REPRESENTATION_ALIASES = {
    "intervals_24_decimal6": RepresentationSpec(
        kind="intervals",
        n_bins=24,
        decimals=6,
        legacy_prompt=True,
    ),
    "centers_24_standard": RepresentationSpec(kind="centers"),
    "centers_24_half_shift": RepresentationSpec(
        kind="centers",
        origin_shift_bins=0.5,
    ),
    "intervals_12": RepresentationSpec(kind="intervals", n_bins=12),
    "intervals_48": RepresentationSpec(kind="intervals", n_bins=48),
    "intervals_24_decimal4": RepresentationSpec(
        kind="intervals",
        decimals=4,
    ),
    "counts_100": RepresentationSpec(kind="counts"),
    "moments_m1_m3": RepresentationSpec(kind="moments"),
}

REPRESENTATION_VARIANTS = tuple(
    name
    for name in REPRESENTATION_ALIASES
    if name != "intervals_24_decimal6"
)
CANONICAL_TRANSMUTATION_CLASSES = (
    "intervals_24_decimal6",
    "moments_m1_m3",
    "centers_24_standard",
)


def coerce_representation_spec(
    value: str | RepresentationSpec | Mapping[str, object],
) -> tuple[str, RepresentationSpec]:
    if isinstance(value, str):
        try:
            return value, REPRESENTATION_ALIASES[value]
        except KeyError as exc:
            raise ValueError(f"unknown representation variant: {value}") from exc
    if isinstance(value, Mapping):
        spec = RepresentationSpec(**value)
    elif isinstance(value, RepresentationSpec):
        spec = value
    else:
        raise TypeError("representation must be a name, mapping, or spec")
    return spec.canonical_id, spec


def _instruction(observation_description: str) -> str:
    return (
        "You are an interaction operator for a phase on a circle.\n"
        f"{observation_description}\n"
        "Negative relative phase is behind you; positive relative phase is "
        "ahead of you.\n"
        "Choose one social correction in response to this distribution:\n"
        '- \"advance\": move in the positive phase direction\n'
        '- \"stay\": apply no social correction\n'
        '- \"retard\": move in the negative phase direction\n'
        "Use only the relative distribution below. Return exactly one JSON "
        "object and no other text.\n"
        'Required schema: {\"social_action\":\"advance|stay|retard\"}\n'
    )


def _center_distribution(
    offset_radians: float,
    concentration: float,
    *,
    n_bins: int,
    origin_shift_bins: float,
) -> tuple[np.ndarray, np.ndarray]:
    width = 2.0 * np.pi / n_bins
    centers = (
        -np.pi
        + width * (np.arange(n_bins) + 0.5 + origin_shift_bins)
    )
    if origin_shift_bins:
        centers = (centers + np.pi) % (2.0 * np.pi) - np.pi
        centers.sort()
    log_weights = concentration * np.cos(centers - offset_radians)
    log_weights -= np.max(log_weights)
    fractions = np.exp(log_weights)
    fractions /= np.sum(fractions)
    return centers, fractions


def _rebinned_centers(
    histogram: RelativePhaseHistogram,
    *,
    n_bins: int,
    origin_shift_bins: float,
) -> tuple[np.ndarray, np.ndarray]:
    width = 2.0 * np.pi / n_bins
    centers = (
        -np.pi
        + width * (np.arange(n_bins) + 0.5 + origin_shift_bins)
    )
    if origin_shift_bins:
        centers = (centers + np.pi) % (2.0 * np.pi) - np.pi
        centers.sort()
    if (
        n_bins == histogram.fractions.size
        and abs(float(origin_shift_bins)) < 1e-12
        and np.allclose(histogram.edges, np.linspace(-np.pi, np.pi, n_bins + 1))
    ):
        return centers, np.asarray(histogram.fractions, dtype=np.float64)
    source_centers = (histogram.edges[:-1] + histogram.edges[1:]) / 2.0
    assigned = np.zeros(n_bins, dtype=np.float64)
    for source_center, mass in zip(source_centers, histogram.fractions):
        deltas = np.abs(wrap_phase(centers - source_center))
        assigned[int(np.argmin(deltas))] += float(mass)
    assigned /= np.sum(assigned)
    return centers, assigned


def _serialize_intervals_histogram(
    histogram: RelativePhaseHistogram,
    *,
    decimals: int,
) -> str:
    lines = []
    for index, value in enumerate(histogram.fractions):
        lines.append(
            f"bin_{index:02d} "
            f"[{histogram.edges[index]:+.{decimals}f},"
            f"{histogram.edges[index + 1]:+.{decimals}f}): "
            f"{value:.{decimals}f}"
        )
    return "\n".join(lines)


def _serialize_intervals(
    offset_radians: float,
    concentration: float,
    *,
    n_bins: int,
    decimals: int,
) -> str:
    histogram = fixed_field_histogram(
        offset_radians,
        concentration,
        n_bins=n_bins,
    )
    return _serialize_intervals_histogram(histogram, decimals=decimals)


def _serialize_centers_histogram(
    histogram: RelativePhaseHistogram,
    *,
    n_bins: int,
    decimals: int,
    origin_shift_bins: float,
) -> str:
    centers, fractions = _rebinned_centers(
        histogram,
        n_bins=n_bins,
        origin_shift_bins=origin_shift_bins,
    )
    return "\n".join(
        f"bin_{index:02d} center {center:+.{decimals}f}: "
        f"{value:.{decimals}f}"
        for index, (center, value) in enumerate(zip(centers, fractions))
    )


def _serialize_centers(
    offset_radians: float,
    concentration: float,
    *,
    n_bins: int,
    decimals: int,
    origin_shift_bins: float,
) -> str:
    centers, fractions = _center_distribution(
        offset_radians,
        concentration,
        n_bins=n_bins,
        origin_shift_bins=origin_shift_bins,
    )
    return "\n".join(
        f"bin_{index:02d} center {center:+.{decimals}f}: "
        f"{value:.{decimals}f}"
        for index, (center, value) in enumerate(zip(centers, fractions))
    )


def _integer_counts(fractions: np.ndarray, total: int) -> np.ndarray:
    raw = fractions * total
    counts = np.floor(raw).astype(np.int64)
    remainder = total - int(np.sum(counts))
    order = np.argsort(-(raw - counts), kind="stable")
    counts[order[:remainder]] += 1
    return counts


def _serialize_counts_histogram(
    histogram: RelativePhaseHistogram,
    *,
    total: int,
    decimals: int,
) -> str:
    counts = _integer_counts(histogram.fractions, total)
    width = max(1, len(str(total)))
    return "\n".join(
        f"bin_{index:02d} "
        f"[{histogram.edges[index]:+.{decimals}f},"
        f"{histogram.edges[index + 1]:+.{decimals}f}): "
        f"{count:0{width}d}"
        for index, count in enumerate(counts)
    )


def _serialize_counts(
    offset_radians: float,
    concentration: float,
    *,
    n_bins: int,
    total: int,
    decimals: int,
) -> str:
    histogram = fixed_field_histogram(
        offset_radians,
        concentration,
        n_bins=n_bins,
    )
    return _serialize_counts_histogram(
        histogram,
        total=total,
        decimals=decimals,
    )


def _serialize_moments_histogram(
    histogram: RelativePhaseHistogram,
    *,
    order: int,
    decimals: int,
) -> str:
    centers = (histogram.edges[:-1] + histogram.edges[1:]) / 2.0
    weight = np.asarray(histogram.fractions, dtype=np.float64)
    weight = weight / np.sum(weight)
    lines = []
    for harmonic in range(1, order + 1):
        cosine = float(np.sum(weight * np.cos(harmonic * centers)))
        sine = float(np.sum(weight * np.sin(harmonic * centers)))
        lines.append(f"moment_{harmonic}_cos: {cosine:+.{decimals}f}")
        lines.append(f"moment_{harmonic}_sin: {sine:+.{decimals}f}")
    return "\n".join(lines)


def _serialize_moments(
    offset_radians: float,
    concentration: float,
    *,
    order: int,
    decimals: int,
) -> str:
    angle = np.linspace(-np.pi, np.pi, 65_536, endpoint=False)
    weight = np.exp(concentration * (np.cos(angle - offset_radians) - 1.0))
    weight /= np.sum(weight)
    lines = []
    for harmonic in range(1, order + 1):
        cosine = float(np.sum(weight * np.cos(harmonic * angle)))
        sine = float(np.sum(weight * np.sin(harmonic * angle)))
        lines.append(f"moment_{harmonic}_cos: {cosine:+.{decimals}f}")
        lines.append(f"moment_{harmonic}_sin: {sine:+.{decimals}f}")
    return "\n".join(lines)


def build_representation_prompt_from_histogram(
    variant: str | RepresentationSpec | Mapping[str, object],
    histogram: RelativePhaseHistogram,
) -> str:
    """Encode an arbitrary relative-phase histogram under a representation."""
    representation_name, spec = coerce_representation_spec(variant)
    if spec.n_bins <= 0 or spec.decimals < 0:
        raise ValueError("n_bins must be positive and decimals non-negative")
    if spec.kind in {"intervals", "counts"} and histogram.fractions.size != spec.n_bins:
        raise ValueError("histogram bin count must match representation n_bins")

    if spec.legacy_prompt:
        if (
            histogram.fractions.size == spec.n_bins
            and int(histogram.peer_count) == 240
            and np.allclose(
                histogram.edges,
                np.linspace(-np.pi, np.pi, spec.n_bins + 1),
            )
        ):
            return build_response_prompt(histogram)
        description = (
            "The table gives the normalized distribution of other phases "
            f"relative to you in {spec.n_bins} circular intervals in radians."
        )
        payload = _serialize_intervals_histogram(
            histogram,
            decimals=spec.decimals,
        )
        return (
            _instruction(description)
            + "Relative phase distribution:\n"
            + payload
        )

    if spec.kind == "centers":
        description = (
            f"The table gives normalized relative-phase mass at {spec.n_bins} circular "
            "bin centers in radians."
        )
        payload = _serialize_centers_histogram(
            histogram,
            n_bins=spec.n_bins,
            decimals=spec.decimals,
            origin_shift_bins=spec.origin_shift_bins,
        )
        label = "Relative phase distribution by bin center:"
    elif spec.kind == "intervals":
        if spec.origin_shift_bins != 0:
            raise ValueError("shifted interval encoding is not implemented")
        description = (
            "The table gives the normalized distribution of other phases "
            f"relative to you in {spec.n_bins} circular intervals in radians."
        )
        payload = _serialize_intervals_histogram(
            histogram,
            decimals=spec.decimals,
        )
        label = "Relative phase distribution:"
    elif spec.kind == "counts":
        if spec.origin_shift_bins != 0:
            raise ValueError("shifted count encoding is not implemented")
        description = (
            "The table gives a fixed-total count representation of the "
            f"relative phase distribution. Counts sum to {spec.count_total}."
        )
        payload = _serialize_counts_histogram(
            histogram,
            total=spec.count_total,
            decimals=spec.decimals,
        )
        label = "Relative phase distribution as counts:"
    else:
        if representation_name == "moments_m1_m3":
            description = (
                "The values are the first three normalized circular moments "
                "of the distribution of other phases relative to you."
            )
        else:
            description = (
                f"The values are the first {spec.moment_order} normalized "
                "circular moments of the distribution of other phases "
                "relative to you."
            )
        payload = _serialize_moments_histogram(
            histogram,
            order=spec.moment_order,
            decimals=spec.decimals,
        )
        label = "Relative phase distribution as circular moments:"

    return _instruction(description) + label + "\n" + payload


def build_representation_prompt(
    variant: str | RepresentationSpec | Mapping[str, object],
    offset_radians: float,
    concentration: float,
    *,
    stimulus: str | Mapping[str, object] | None = None,
) -> str:
    """Build a frozen prompt from a named or structured representation.

    If ``stimulus`` is provided, the field comes from the stimulus catalog and
    ``concentration`` is ignored except as an optional trace proxy.
    """
    if stimulus is not None:
        _, stim_spec = coerce_stimulus_spec(stimulus)
        _, spec = coerce_representation_spec(variant)
        if (
            spec.legacy_prompt
            and stim_spec.kind == "unimodal"
            and stim_spec.peer_count == 240
            and stim_spec.n_bins == 24
        ):
            return build_response_prompt(
                fixed_field_histogram(
                    offset_radians,
                    stim_spec.concentration,
                    n_bins=24,
                )
            )
        histogram = build_stimulus_histogram(stimulus, offset_radians)
        return build_representation_prompt_from_histogram(variant, histogram)

    representation_name, spec = coerce_representation_spec(variant)
    if spec.n_bins <= 0 or spec.decimals < 0:
        raise ValueError("n_bins must be positive and decimals non-negative")
    if spec.legacy_prompt:
        histogram = fixed_field_histogram(
            offset_radians,
            concentration,
            n_bins=spec.n_bins,
        )
        return build_response_prompt(histogram)

    if spec.kind == "centers":
        description = (
            f"The table gives normalized relative-phase mass at {spec.n_bins} circular "
            "bin centers in radians."
        )
        payload = _serialize_centers(
            offset_radians,
            concentration,
            n_bins=spec.n_bins,
            decimals=spec.decimals,
            origin_shift_bins=spec.origin_shift_bins,
        )
        label = "Relative phase distribution by bin center:"
    elif spec.kind == "intervals":
        if spec.origin_shift_bins != 0:
            raise ValueError("shifted interval encoding is not implemented")
        description = (
            "The table gives the normalized distribution of other phases "
            f"relative to you in {spec.n_bins} circular intervals in radians."
        )
        payload = _serialize_intervals(
            offset_radians,
            concentration,
            n_bins=spec.n_bins,
            decimals=spec.decimals,
        )
        label = "Relative phase distribution:"
    elif spec.kind == "counts":
        if spec.origin_shift_bins != 0:
            raise ValueError("shifted count encoding is not implemented")
        description = (
            "The table gives a fixed-total count representation of the "
            f"relative phase distribution. Counts sum to {spec.count_total}."
        )
        payload = _serialize_counts(
            offset_radians,
            concentration,
            n_bins=spec.n_bins,
            total=spec.count_total,
            decimals=spec.decimals,
        )
        label = "Relative phase distribution as counts:"
    else:
        if representation_name == "moments_m1_m3":
            description = (
                "The values are the first three normalized circular moments "
                "of the distribution of other phases relative to you."
            )
        else:
            description = (
                f"The values are the first {spec.moment_order} normalized "
                "circular moments of the distribution of other phases "
                "relative to you."
            )
        payload = _serialize_moments(
            offset_radians,
            concentration,
            order=spec.moment_order,
            decimals=spec.decimals,
        )
        label = "Relative phase distribution as circular moments:"

    return _instruction(description) + label + "\n" + payload

"""Gauge-invariant observations of relative phase distributions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

TAU = 2.0 * np.pi
DEFAULT_N_BINS = 24
BIN_COORDINATE_DECIMALS = 12


def wrap_phase(value: ArrayLike) -> NDArray[np.float64]:
    """Wrap radians to the half-open interval [-pi, pi)."""
    array = np.asarray(value, dtype=np.float64)
    return (array + np.pi) % TAU - np.pi


@dataclass(frozen=True)
class RelativePhaseHistogram:
    """Canonical fixed-width histogram seen by one focal agent."""

    edges: NDArray[np.float64]
    fractions: NDArray[np.float64]
    peer_count: int

    def __post_init__(self) -> None:
        if self.edges.ndim != 1 or self.fractions.ndim != 1:
            raise ValueError("edges and fractions must be one-dimensional")
        if self.edges.size != self.fractions.size + 1:
            raise ValueError("edges must contain one more value than fractions")
        if self.peer_count < 0:
            raise ValueError("peer_count must be non-negative")


def relative_phase_histogram(
    phases: ArrayLike,
    focal_index: int,
    *,
    n_bins: int = DEFAULT_N_BINS,
) -> RelativePhaseHistogram:
    """Build a normalized histogram of peer phases relative to one agent."""
    phase_array = np.asarray(phases, dtype=np.float64)
    if phase_array.ndim != 1:
        raise ValueError("phases must be one-dimensional")
    if phase_array.size == 0:
        raise ValueError("phases must contain at least one agent")
    if not 0 <= focal_index < phase_array.size:
        raise IndexError("focal_index is outside the phase array")
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")

    peer_mask = np.ones(phase_array.size, dtype=bool)
    peer_mask[focal_index] = False
    relative = wrap_phase(phase_array[peer_mask] - phase_array[focal_index])
    edges = np.linspace(-np.pi, np.pi, n_bins + 1, dtype=np.float64)
    # Assign bins in dimensionless coordinates. A common rotation should
    # cancel algebraically; canonical rounding removes machine-epsilon
    # subtraction differences exactly at an edge. Modulo maps a coordinate
    # rounded to n_bins back onto the half-open -pi boundary.
    bin_coordinate = (relative + np.pi) * n_bins / TAU
    bin_coordinate = np.round(
        bin_coordinate,
        decimals=BIN_COORDINATE_DECIMALS,
    )
    bin_indices = np.floor(bin_coordinate).astype(np.int64) % n_bins
    counts = np.bincount(bin_indices, minlength=n_bins)
    peer_count = int(relative.size)
    if peer_count:
        fractions = counts.astype(np.float64) / peer_count
    else:
        fractions = np.zeros(n_bins, dtype=np.float64)

    return RelativePhaseHistogram(
        edges=edges,
        fractions=fractions,
        peer_count=peer_count,
    )


def all_relative_histograms(
    phases: ArrayLike,
    *,
    n_bins: int = DEFAULT_N_BINS,
) -> tuple[RelativePhaseHistogram, ...]:
    """Build all focal observations from one immutable phase snapshot."""
    phase_array = np.asarray(phases, dtype=np.float64)
    if phase_array.ndim != 1:
        raise ValueError("phases must be one-dimensional")
    return tuple(
        relative_phase_histogram(phase_array, i, n_bins=n_bins)
        for i in range(phase_array.size)
    )


def serialize_histogram(
    histogram: RelativePhaseHistogram,
    *,
    decimals: int = 6,
) -> str:
    """Serialize every bin in canonical order and fixed decimal precision."""
    if decimals < 0:
        raise ValueError("decimals must be non-negative")
    lines = []
    for index, fraction in enumerate(histogram.fractions):
        left = histogram.edges[index]
        right = histogram.edges[index + 1]
        lines.append(
            f"bin_{index:02d} "
            f"[{left:+.{decimals}f},{right:+.{decimals}f}): "
            f"{fraction:.{decimals}f}"
        )
    return "\n".join(lines)

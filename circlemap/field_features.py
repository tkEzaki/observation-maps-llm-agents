"""Field descriptors for Stage 3 full-trinomial surrogates.

These features are computed from the observed relative-phase histogram.
They must not rely on experiment-only labels such as ε; production
collective fields have no explicit imbalance tag.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .observation import RelativePhaseHistogram
from .representations import REPRESENTATION_ALIASES, coerce_representation_spec


TAU = 2.0 * np.pi
COMMON_FEATURE_NAMES: tuple[str, ...] = (
    "re_z1",
    "im_z1",
    "abs_z1",
    "re_z2",
    "im_z2",
    "abs_z2",
    "re_z3",
    "im_z3",
    "abs_z3",
    "circular_entropy",
    "antipodal_balance",
    "asymmetry",
    "bimodality",
    "sparsity",
    "peer_count",
)


@dataclass(frozen=True)
class FieldDescriptors:
    """Common + optional representation-native feature vectors."""

    common: NDArray[np.float64]
    common_names: tuple[str, ...]
    native: NDArray[np.float64]
    native_names: tuple[str, ...]
    estimated_eps: float
    unresolved_near_zero: bool


def _bin_centers(histogram: RelativePhaseHistogram) -> NDArray[np.float64]:
    return 0.5 * (histogram.edges[:-1] + histogram.edges[1:])


def circular_moments(
    histogram: RelativePhaseHistogram,
    *,
    order: int = 3,
) -> list[complex]:
    """Normalized circular moments from the histogram mass."""
    fractions = np.asarray(histogram.fractions, dtype=np.float64)
    centers = _bin_centers(histogram)
    moments: list[complex] = []
    for harmonic in range(1, order + 1):
        z = complex(
            float(np.sum(fractions * np.cos(harmonic * centers))),
            float(np.sum(fractions * np.sin(harmonic * centers))),
        )
        moments.append(z)
    return moments


def circular_entropy(histogram: RelativePhaseHistogram) -> float:
    """Shannon entropy of ρ in nats (zeros contribute nothing)."""
    fractions = np.asarray(histogram.fractions, dtype=np.float64)
    positive = fractions[fractions > 0.0]
    if positive.size == 0:
        return 0.0
    return float(-np.sum(positive * np.log(positive)))


def antipodal_half_masses(
    histogram: RelativePhaseHistogram,
) -> tuple[float, float]:
    """Mass near 0 vs near ±π using angular proximity on the circle."""
    fractions = np.asarray(histogram.fractions, dtype=np.float64)
    centers = _bin_centers(histogram)
    # Distance to 0 and to π (same as −π).
    dist_zero = np.abs(centers)
    dist_pi = np.pi - np.abs(centers)
    near_zero = dist_zero <= dist_pi
    mass_zero = float(np.sum(fractions[near_zero]))
    mass_pi = float(np.sum(fractions[~near_zero]))
    return mass_zero, mass_pi


def estimated_weight_imbalance(histogram: RelativePhaseHistogram) -> float:
    """Proxy ε̂ = mass_near_0 − 0.5 for antipodal-like fields.

    For unimodal fields this is not a true stimulus ε; it remains a useful
    descriptor of polar mass imbalance and is used only for Policy-A OOD
    tagging of the unresolved near-zero band.
    """
    mass_zero, _mass_pi = antipodal_half_masses(histogram)
    return float(mass_zero - 0.5)


def common_field_vector(
    histogram: RelativePhaseHistogram,
) -> tuple[NDArray[np.float64], float]:
    """Return (common feature vector, estimated ε̂)."""
    moments = circular_moments(histogram, order=3)
    mass_zero, mass_pi = antipodal_half_masses(histogram)
    total_anti = mass_zero + mass_pi
    if total_anti <= 0.0:
        balance = 0.0
        asymmetry = 0.0
    else:
        # 1 when equal antipodal halves; 0 when all mass on one pole.
        balance = 1.0 - abs(mass_zero - mass_pi) / total_anti
        asymmetry = mass_zero - mass_pi
    abs_z = [abs(z) for z in moments]
    # Positive when second harmonic dominates first (bimodal signature).
    bimodality = float(abs_z[1] - abs_z[0])
    peer_count = float(histogram.peer_count)
    sparsity = 1.0 / max(peer_count, 1.0)
    eps_hat = float(mass_zero - 0.5)
    values = np.asarray(
        [
            moments[0].real,
            moments[0].imag,
            abs_z[0],
            moments[1].real,
            moments[1].imag,
            abs_z[1],
            moments[2].real,
            moments[2].imag,
            abs_z[2],
            circular_entropy(histogram),
            balance,
            asymmetry,
            bimodality,
            sparsity,
            peer_count,
        ],
        dtype=np.float64,
    )
    return values, eps_hat


def native_feature_vector(
    representation: str,
    histogram: RelativePhaseHistogram,
) -> tuple[NDArray[np.float64], tuple[str, ...]]:
    """Representation-native payload actually shown (bins or moments)."""
    name, spec = coerce_representation_spec(representation)
    if name not in REPRESENTATION_ALIASES and not isinstance(representation, str):
        pass
    if spec.kind == "moments":
        moments = circular_moments(histogram, order=spec.moment_order)
        values: list[float] = []
        names: list[str] = []
        for index, z in enumerate(moments, start=1):
            values.extend([z.real, z.imag])
            names.extend([f"moment_{index}_cos", f"moment_{index}_sin"])
        return np.asarray(values, dtype=np.float64), tuple(names)
    # intervals / centers / counts: use the presented bin mass vector.
    fractions = np.asarray(histogram.fractions, dtype=np.float64)
    names = tuple(f"bin_{index:02d}" for index in range(fractions.size))
    return fractions.copy(), names


def unresolved_near_zero_band(
    eps_hat: float,
    *,
    antipodal_balance: float,
    abs_z1: float,
    lower: float = 1e-6,
    upper: float = 0.02,
    balance_min: float = 0.85,
    abs_z1_max: float = 0.25,
) -> bool:
    """Policy A: flag unresolved 0 < |ε| < 0.02-like fields without ε labels."""
    magnitude = abs(float(eps_hat))
    return bool(
        lower < magnitude < upper
        and float(antipodal_balance) >= balance_min
        and float(abs_z1) <= abs_z1_max
    )


def extract_field_descriptors(
    representation: str,
    histogram: RelativePhaseHistogram,
) -> FieldDescriptors:
    """Build common + native descriptors for one (representation, field)."""
    common, eps_hat = common_field_vector(histogram)
    native, native_names = native_feature_vector(representation, histogram)
    unresolved = unresolved_near_zero_band(
        eps_hat,
        antipodal_balance=float(common[COMMON_FEATURE_NAMES.index("antipodal_balance")]),
        abs_z1=float(common[COMMON_FEATURE_NAMES.index("abs_z1")]),
    )
    return FieldDescriptors(
        common=common,
        common_names=COMMON_FEATURE_NAMES,
        native=native,
        native_names=native_names,
        estimated_eps=eps_hat,
        unresolved_near_zero=unresolved,
    )


def combined_feature_matrix(
    descriptors: list[FieldDescriptors],
) -> tuple[NDArray[np.float64], tuple[str, ...]]:
    """Stack common∥native rows; native width must match within a batch."""
    if not descriptors:
        return np.zeros((0, 0), dtype=np.float64), ()
    native_width = descriptors[0].native.size
    names = descriptors[0].common_names + descriptors[0].native_names
    rows = []
    for item in descriptors:
        if item.native.size != native_width:
            raise ValueError("native feature width mismatch within batch")
        rows.append(np.concatenate([item.common, item.native]))
    return np.asarray(rows, dtype=np.float64), names

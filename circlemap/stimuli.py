"""Frozen stimulus-field catalog for Stage B manifold expansion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping

import numpy as np

from .observation import RelativePhaseHistogram, wrap_phase
from .response import fixed_field_histogram


@dataclass(frozen=True)
class StimulusSpec:
    """A named continuous or sparse relative-phase field."""

    kind: Literal["unimodal", "mixture", "fixed"]
    concentration: float
    peer_count: int = 240
    n_bins: int = 24
    mode_offsets: tuple[float, ...] = ()
    weights: tuple[float, ...] = ()
    mode_counts: tuple[int, ...] = ()
    realization_seed: int | None = None
    partner_paired: bool = False
    fixed_fractions: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if self.n_bins <= 0:
            raise ValueError("n_bins must be positive")
        if self.peer_count <= 0:
            raise ValueError("peer_count must be positive")
        if not np.isfinite(self.concentration) or self.concentration < 0:
            raise ValueError("concentration must be finite and non-negative")
        if self.kind == "fixed":
            if len(self.fixed_fractions) != self.n_bins:
                raise ValueError("fixed_fractions must match n_bins")
            if abs(sum(self.fixed_fractions) - 1.0) > 1e-6:
                raise ValueError("fixed_fractions must sum to one")
            if any(value < 0 for value in self.fixed_fractions):
                raise ValueError("fixed_fractions must be non-negative")
            return
        if self.mode_counts:
            if any(count < 0 for count in self.mode_counts):
                raise ValueError("mode_counts must be non-negative")
            if sum(self.mode_counts) != self.peer_count:
                raise ValueError("mode_counts must sum to peer_count")
        if self.kind == "unimodal":
            if self.mode_offsets or self.weights or self.mode_counts:
                raise ValueError("unimodal stimuli do not take mode mixtures")
            return
        if len(self.mode_offsets) < 2:
            raise ValueError("mixture stimuli need at least two modes")
        if self.mode_counts:
            if len(self.mode_counts) != len(self.mode_offsets):
                raise ValueError("mode_counts must match mode_offsets")
            return
        if len(self.mode_offsets) != len(self.weights):
            raise ValueError("mode_offsets and weights must match")
        if any(weight < 0 for weight in self.weights):
            raise ValueError("weights must be non-negative")
        if abs(sum(self.weights) - 1.0) > 1e-9:
            raise ValueError("weights must sum to one")


STIMULUS_CATALOG: dict[str, StimulusSpec] = {
    "unimodal_k9": StimulusSpec(kind="unimodal", concentration=9.0),
    "bimodal_equal_sep2_k6": StimulusSpec(
        kind="mixture",
        concentration=6.0,
        mode_offsets=(-1.0, 1.0),
        weights=(0.5, 0.5),
    ),
    "antipodal_equal_k6": StimulusSpec(
        kind="mixture",
        concentration=6.0,
        mode_offsets=(0.0, float(np.pi)),
        weights=(0.5, 0.5),
    ),
    "asymmetric_w075_sep2_k6": StimulusSpec(
        kind="mixture",
        concentration=6.0,
        mode_offsets=(-1.0, 1.0),
        weights=(0.75, 0.25),
    ),
    "sparse_N8_unimodal_k6": StimulusSpec(
        kind="unimodal",
        concentration=6.0,
        peer_count=8,
    ),
    "sparse_N16_antipodal_k6": StimulusSpec(
        kind="mixture",
        concentration=6.0,
        peer_count=16,
        mode_offsets=(0.0, float(np.pi)),
        weights=(0.5, 0.5),
    ),
}


def antipodal_weight_epsilon_id(epsilon: float, *, concentration: float = 6.0) -> str:
    """Stable catalog id for dense antipodal weight imbalance."""
    milli = int(round(float(epsilon) * 1000))
    sign = "m" if milli < 0 else "p"
    return f"antipodal_weps_{sign}{abs(milli):03d}_k{int(concentration)}"


def register_antipodal_weight_epsilon_grid(
    epsilons: list[float] | tuple[float, ...],
    *,
    concentration: float = 6.0,
) -> list[str]:
    """Ensure catalog entries exist for each ε; return ordered ids."""
    ids = []
    for epsilon in epsilons:
        name = antipodal_weight_epsilon_id(epsilon, concentration=concentration)
        weight = 0.5 + float(epsilon)
        if weight <= 0.0 or weight >= 1.0:
            raise ValueError(f"epsilon={epsilon} leaves a weight outside (0,1)")
        STIMULUS_CATALOG[name] = StimulusSpec(
            kind="mixture",
            concentration=float(concentration),
            mode_offsets=(0.0, float(np.pi)),
            weights=(weight, 1.0 - weight),
        )
        ids.append(name)
    return ids


# Default P1 dense signed weight axis (locked by power simulation).
P1_DENSE_WEIGHT_EPSILONS: tuple[float, ...] = (
    -0.10,
    -0.05,
    -0.02,
    0.0,
    0.02,
    0.05,
    0.10,
    0.20,
)
register_antipodal_weight_epsilon_grid(P1_DENSE_WEIGHT_EPSILONS)

# Step D nested sparse antipodal panel.
D_SPARSE_PEER_COUNT = 16
D_SPARSE_IMBALANCES: tuple[tuple[int, int], ...] = ((8, 8), (9, 7), (10, 6))
D_SPARSE_N_REALIZATIONS = 6
D_SPARSE_REPETITIONS = 4


def sparse_peer_antipodal_id(
    count_a: int,
    count_b: int,
    realization: int,
    *,
    peer_count: int = D_SPARSE_PEER_COUNT,
    concentration: float = 6.0,
) -> str:
    if count_a + count_b != peer_count:
        raise ValueError("count_a + count_b must equal peer_count")
    return (
        f"sparse_peer{peer_count}_antipodal_{count_a}_{count_b}"
        f"_r{int(realization):02d}_k{int(concentration)}"
    )


def register_sparse_peer_antipodal_panel(
    *,
    imbalances: tuple[tuple[int, int], ...] = D_SPARSE_IMBALANCES,
    n_realizations: int = D_SPARSE_N_REALIZATIONS,
    peer_count: int = D_SPARSE_PEER_COUNT,
    concentration: float = 6.0,
    seed_base: int = 2026072316,
) -> list[str]:
    """Register imbalance × realization sparse antipodal stimuli."""
    ids: list[str] = []
    for count_a, count_b in imbalances:
        for realization in range(n_realizations):
            name = sparse_peer_antipodal_id(
                count_a,
                count_b,
                realization,
                peer_count=peer_count,
                concentration=concentration,
            )
            STIMULUS_CATALOG[name] = StimulusSpec(
                kind="mixture",
                concentration=float(concentration),
                peer_count=int(peer_count),
                mode_offsets=(0.0, float(np.pi)),
                mode_counts=(int(count_a), int(count_b)),
                realization_seed=int(seed_base + 1000 * count_a + 10 * count_b + realization),
                partner_paired=True,
            )
            ids.append(name)
    return ids


D_SPARSE_PROFILE_IDS = register_sparse_peer_antipodal_panel()


def coerce_stimulus_spec(
    value: str | StimulusSpec | Mapping[str, object],
) -> tuple[str, StimulusSpec]:
    if isinstance(value, str):
        try:
            return value, STIMULUS_CATALOG[value]
        except KeyError as exc:
            raise ValueError(f"unknown stimulus profile: {value}") from exc
    if isinstance(value, Mapping):
        payload = dict(value)
        name = str(payload.pop("id", "custom"))
        spec = StimulusSpec(**payload)  # type: ignore[arg-type]
        return name, spec
    if isinstance(value, StimulusSpec):
        return "custom", value
    raise TypeError("stimulus must be a name, mapping, or StimulusSpec")


def _von_mises_fractions(
    centers: np.ndarray,
    offset_radians: float,
    concentration: float,
) -> np.ndarray:
    log_weights = concentration * np.cos(wrap_phase(centers - offset_radians))
    log_weights -= np.max(log_weights)
    weights = np.exp(log_weights)
    return weights / np.sum(weights)


def _integerize_fractions(fractions: np.ndarray, total: int) -> np.ndarray:
    raw = fractions * total
    counts = np.floor(raw).astype(np.int64)
    remainder = total - int(np.sum(counts))
    order = np.argsort(-(raw - counts), kind="stable")
    counts[order[:remainder]] += 1
    return counts.astype(np.float64) / float(total)


def _bin_phases(
    phases: np.ndarray,
    *,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    edges = np.linspace(-np.pi, np.pi, n_bins + 1, dtype=np.float64)
    wrapped = wrap_phase(np.asarray(phases, dtype=np.float64))
    # digitize expects right edges; map pi endpoint into last bin.
    indices = np.digitize(wrapped, edges[1:-1], right=False)
    counts = np.bincount(indices, minlength=n_bins).astype(np.float64)
    fractions = counts / max(1.0, float(np.sum(counts)))
    return edges, fractions


def _sample_sparse_mixture_phases(
    spec: StimulusSpec,
    offset_radians: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if not spec.mode_counts:
        raise ValueError("sparse mixture sampling requires mode_counts")
    kappa = float(spec.concentration)
    phases: list[float] = []
    if spec.partner_paired and len(spec.mode_offsets) == 2:
        count_a, count_b = (int(spec.mode_counts[0]), int(spec.mode_counts[1]))
        mode_a, mode_b = (float(spec.mode_offsets[0]), float(spec.mode_offsets[1]))
        n_pair = min(count_a, count_b)
        if n_pair:
            base = rng.vonmises(
                wrap_phase(offset_radians + mode_a),
                kappa,
                size=n_pair,
            )
            phases.extend(float(value) for value in base)
            partners = wrap_phase(np.asarray(base, dtype=np.float64) + np.pi)
            phases.extend(float(value) for value in partners)
        leftover = abs(count_a - count_b)
        if leftover:
            majority_mode = mode_a if count_a >= count_b else mode_b
            extra = rng.vonmises(
                wrap_phase(offset_radians + majority_mode),
                kappa,
                size=leftover,
            )
            phases.extend(float(value) for value in extra)
    else:
        for mode_offset, count in zip(spec.mode_offsets, spec.mode_counts):
            if count == 0:
                continue
            draws = rng.vonmises(
                wrap_phase(offset_radians + float(mode_offset)),
                kappa,
                size=int(count),
            )
            phases.extend(float(value) for value in draws)
    if len(phases) != spec.peer_count:
        raise RuntimeError("sparse sample size mismatch")
    return np.asarray(phases, dtype=np.float64)


def build_stimulus_histogram(
    stimulus: str | StimulusSpec | Mapping[str, object],
    offset_radians: float,
) -> RelativePhaseHistogram:
    """Build a deterministic histogram rotated so the reference equals offset."""
    _, spec = coerce_stimulus_spec(stimulus)
    if not np.isfinite(offset_radians):
        raise ValueError("offset_radians must be finite")

    if spec.kind == "fixed":
        # Frozen empirical field already expressed in the focal frame.
        # Nonzero offset rotates the mass for gauge checks / paired designs.
        fractions = np.asarray(spec.fixed_fractions, dtype=np.float64)
        if abs(float(offset_radians)) > 1e-15:
            shift_bins = int(
                np.round(float(offset_radians) * spec.n_bins / (2.0 * np.pi))
            )
            fractions = np.roll(fractions, -shift_bins)
        edges = np.linspace(-np.pi, np.pi, spec.n_bins + 1, dtype=np.float64)
        return RelativePhaseHistogram(
            edges=edges,
            fractions=fractions,
            peer_count=spec.peer_count,
        )

    # Exact legacy path for dense unimodal fields.
    if (
        spec.kind == "unimodal"
        and spec.peer_count == 240
        and spec.n_bins == 24
        and not spec.mode_counts
    ):
        return fixed_field_histogram(
            offset_radians,
            spec.concentration,
            n_bins=spec.n_bins,
            synthetic_peer_count=spec.peer_count,
        )

    # Sparse count mixtures: sample peer phases (optionally partner-paired).
    if spec.kind == "mixture" and spec.mode_counts:
        if spec.realization_seed is None:
            raise ValueError("sparse mixture stimuli require realization_seed")
        # Mix seed with a stable hash of the offset so each offset is independent
        # but reproducible across runs.
        offset_key = int(np.round(float(offset_radians) * 1_000_000.0)) % (2**31 - 1)
        rng = np.random.default_rng(
            (int(spec.realization_seed) * 1_000_003 + offset_key) % (2**32 - 1)
        )
        phases = _sample_sparse_mixture_phases(spec, offset_radians, rng)
        edges, fractions = _bin_phases(phases, n_bins=spec.n_bins)
        return RelativePhaseHistogram(
            edges=edges,
            fractions=fractions,
            peer_count=spec.peer_count,
        )

    edges = np.linspace(-np.pi, np.pi, spec.n_bins + 1, dtype=np.float64)
    centers = (edges[:-1] + edges[1:]) / 2.0
    if spec.kind == "unimodal":
        fractions = _von_mises_fractions(
            centers,
            offset_radians,
            spec.concentration,
        )
    else:
        fractions = np.zeros(spec.n_bins, dtype=np.float64)
        for mode_offset, weight in zip(spec.mode_offsets, spec.weights):
            mode = wrap_phase(offset_radians + mode_offset)
            fractions += weight * _von_mises_fractions(
                centers,
                float(mode),
                spec.concentration,
            )
        fractions /= np.sum(fractions)

    if spec.peer_count != 240:
        fractions = _integerize_fractions(fractions, spec.peer_count)

    return RelativePhaseHistogram(
        edges=edges,
        fractions=fractions,
        peer_count=spec.peer_count,
    )


def stimulus_concentration_proxy(
    stimulus: str | StimulusSpec | Mapping[str, object],
) -> float:
    """Numeric stand-in stored on tasks for backward-compatible traces."""
    _, spec = coerce_stimulus_spec(stimulus)
    return float(spec.concentration)

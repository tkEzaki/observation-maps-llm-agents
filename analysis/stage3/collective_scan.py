"""Surrogate-only collective coverage scan (Stage 3A gate before freeze)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from circlemap.field_features import extract_field_descriptors
from circlemap.observation import RelativePhaseHistogram, wrap_phase

from .models import TrinomialModel
from .ood import OODDetector


ACTIONS = np.asarray([-1.0, 0.0, 1.0], dtype=np.float64)


@dataclass
class CoverageScanResult:
    q_ood: float
    n_agent_steps: int
    n_ood: int
    mean_ood_score: float
    final_order_parameter: float


def _histogram_from_phases(
    phases: NDArray[np.float64],
    focal: int,
    *,
    n_bins: int = 24,
) -> RelativePhaseHistogram:
    peer_mask = np.ones(phases.size, dtype=bool)
    peer_mask[focal] = False
    relative = wrap_phase(phases[peer_mask] - phases[focal])
    edges = np.linspace(-np.pi, np.pi, n_bins + 1, dtype=np.float64)
    # Same binning convention as observation.relative_phase_histogram.
    bin_coordinate = (relative + np.pi) * n_bins / (2.0 * np.pi)
    bin_indices = np.floor(np.round(bin_coordinate, decimals=12)).astype(np.int64) % n_bins
    counts = np.bincount(bin_indices, minlength=n_bins)
    peer_count = int(relative.size)
    fractions = counts.astype(np.float64) / max(peer_count, 1)
    return RelativePhaseHistogram(edges=edges, fractions=fractions, peer_count=peer_count)


def run_surrogate_collective_scan(
    model: TrinomialModel,
    ood_detector: OODDetector,
    *,
    representation: str,
    n_agents: int = 24,
    n_steps: int = 40,
    coupling: float = 0.15,
    omega_scale: float = 0.02,
    seed: int = 0,
    feature_builder=None,
    ood_feature_builder=None,
) -> CoverageScanResult:
    """Closed-loop surrogate pilot; records OOD rate on realized fields.

    ``feature_builder(representation, histogram) -> (x_row, unresolved_flag)``
    must match the training feature layout used by ``model``.
    ``ood_feature_builder`` may return a (possibly lower-dimensional) row for
    the OOD detector; defaults to the model features.
    """
    if feature_builder is None:
        raise ValueError("feature_builder is required to match training features")
    if ood_feature_builder is None:
        ood_feature_builder = feature_builder

    rng = np.random.default_rng(seed)
    phases = rng.uniform(-np.pi, np.pi, size=n_agents)
    omega = rng.normal(0.0, omega_scale, size=n_agents)
    n_ood = 0
    scores: list[float] = []
    total = n_agents * n_steps

    for _ in range(n_steps):
        actions = np.zeros(n_agents, dtype=np.float64)
        unresolved_flags = []
        feature_rows = []
        ood_rows = []
        for agent in range(n_agents):
            histogram = _histogram_from_phases(phases, agent)
            x_row, unresolved = feature_builder(representation, histogram)
            ood_row, _ = ood_feature_builder(representation, histogram)
            feature_rows.append(x_row)
            ood_rows.append(ood_row)
            unresolved_flags.append(unresolved)
        x = np.asarray(feature_rows, dtype=np.float64)
        x_ood = np.asarray(ood_rows, dtype=np.float64)
        unresolved = np.asarray(unresolved_flags, dtype=bool)
        ood_scores = ood_detector.score(x_ood, unresolved_near_zero=unresolved)
        ood_mask = ood_scores > ood_detector.threshold
        n_ood += int(np.sum(ood_mask))
        scores.extend(float(v) for v in ood_scores)
        probs = model.predict_proba(x)
        for agent in range(n_agents):
            actions[agent] = float(rng.choice(ACTIONS, p=probs[agent]))
        phases = wrap_phase(phases + omega + coupling * actions)

    order = abs(np.mean(np.exp(1j * phases)))
    return CoverageScanResult(
        q_ood=float(n_ood / max(total, 1)),
        n_agent_steps=int(total),
        n_ood=int(n_ood),
        mean_ood_score=float(np.mean(scores)) if scores else float("nan"),
        final_order_parameter=float(order),
    )


def default_feature_builder(representation: str, histogram: RelativePhaseHistogram):
    """Common∥native vector matching Stage3 dataset packing."""
    descriptors = extract_field_descriptors(representation, histogram)
    x = np.concatenate([descriptors.common, descriptors.native])
    return x, descriptors.unresolved_near_zero


def common_ood_feature_builder(representation: str, histogram: RelativePhaseHistogram):
    """Rotation-aware common descriptors only (exclude native bin padding)."""
    descriptors = extract_field_descriptors(representation, histogram)
    return descriptors.common, descriptors.unresolved_near_zero

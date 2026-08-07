"""Collective observables computed from unwrapped phase trajectories."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .engine import SimulationResult


def order_parameter(phases: ArrayLike) -> complex:
    """Return the complex Kuramoto order parameter for one phase snapshot."""
    phase_array = np.asarray(phases, dtype=np.float64)
    if phase_array.ndim != 1 or phase_array.size == 0:
        raise ValueError("phases must be a non-empty one-dimensional array")
    return complex(np.mean(np.exp(1j * phase_array)))


@dataclass(frozen=True)
class CollectiveBalance:
    """Measured collective frequency and its additive decomposition."""

    effective_frequencies: NDArray[np.float64]
    collective_frequency: float
    mean_natural_frequency: float
    mean_social_torque: float
    predicted_collective_frequency: float
    residual: float


def collective_balance(
    result: SimulationResult,
    *,
    start_step: int = 0,
    end_step: int | None = None,
) -> CollectiveBalance:
    """Evaluate the exact frequency-torque balance on one update window."""
    if end_step is None:
        end_step = result.n_steps
    if not 0 <= start_step < end_step <= result.n_steps:
        raise ValueError(
            "window must satisfy 0 <= start_step < end_step <= n_steps"
        )

    duration = end_step - start_step
    effective = (
        result.unwrapped_phases[end_step]
        - result.unwrapped_phases[start_step]
    ) / duration
    collective = float(np.mean(effective))
    mean_natural = float(np.mean(result.natural_frequencies))
    mean_torque = float(
        np.mean(result.social_actions[start_step:end_step])
    )
    predicted = mean_natural + result.coupling * mean_torque

    return CollectiveBalance(
        effective_frequencies=effective,
        collective_frequency=collective,
        mean_natural_frequency=mean_natural,
        mean_social_torque=mean_torque,
        predicted_collective_frequency=predicted,
        residual=collective - predicted,
    )

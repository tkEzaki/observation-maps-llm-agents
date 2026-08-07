"""Deterministic continuous-phase engine with synchronous updates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .observation import (
    DEFAULT_N_BINS,
    RelativePhaseHistogram,
    all_relative_histograms,
)

ActionProvider = Callable[
    [int, tuple[RelativePhaseHistogram, ...]],
    ArrayLike,
]


@dataclass(frozen=True)
class SimulationResult:
    """Complete state and action history for one deterministic run."""

    unwrapped_phases: NDArray[np.float64]
    natural_frequencies: NDArray[np.float64]
    social_actions: NDArray[np.float64]
    coupling: float

    @property
    def n_steps(self) -> int:
        return int(self.social_actions.shape[0])

    @property
    def n_agents(self) -> int:
        return int(self.natural_frequencies.size)


def _vector(name: str, values: ArrayLike) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.size == 0:
        raise ValueError(f"{name} must contain at least one agent")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _validated_actions(actions: ArrayLike, n_agents: int) -> NDArray[np.float64]:
    action_array = _vector("social_actions", actions)
    if action_array.size != n_agents:
        raise ValueError("social_actions must have one value per agent")
    if not np.all(np.isin(action_array, (-1.0, 0.0, 1.0))):
        raise ValueError("social_actions must contain only -1, 0, or +1")
    return action_array


def step(
    unwrapped_phases: ArrayLike,
    natural_frequencies: ArrayLike,
    coupling: float,
    social_actions: ArrayLike,
) -> NDArray[np.float64]:
    """Apply one additive map update from a single state snapshot."""
    phases = _vector("unwrapped_phases", unwrapped_phases)
    frequencies = _vector("natural_frequencies", natural_frequencies)
    if phases.size != frequencies.size:
        raise ValueError("phase and frequency arrays must have equal length")
    if not np.isfinite(coupling):
        raise ValueError("coupling must be finite")
    actions = _validated_actions(social_actions, phases.size)
    return phases + frequencies + float(coupling) * actions


def simulate(
    initial_unwrapped_phases: ArrayLike,
    natural_frequencies: ArrayLike,
    coupling: float,
    n_steps: int,
    action_provider: ActionProvider,
    *,
    n_bins: int = DEFAULT_N_BINS,
) -> SimulationResult:
    """Run synchronous updates using observations from each pre-update state."""
    phases = _vector("initial_unwrapped_phases", initial_unwrapped_phases).copy()
    frequencies = _vector("natural_frequencies", natural_frequencies).copy()
    if phases.size != frequencies.size:
        raise ValueError("phase and frequency arrays must have equal length")
    if not np.isfinite(coupling):
        raise ValueError("coupling must be finite")
    if not isinstance(n_steps, int) or isinstance(n_steps, bool) or n_steps < 0:
        raise ValueError("n_steps must be a non-negative integer")
    if not isinstance(n_bins, int) or isinstance(n_bins, bool) or n_bins <= 0:
        raise ValueError("n_bins must be a positive integer")
    if not callable(action_provider):
        raise TypeError("action_provider must be callable")

    trajectory = np.empty((n_steps + 1, phases.size), dtype=np.float64)
    actions_history = np.empty((n_steps, phases.size), dtype=np.float64)
    trajectory[0] = phases

    for time_index in range(n_steps):
        observations = all_relative_histograms(phases, n_bins=n_bins)
        actions = _validated_actions(
            action_provider(time_index, observations),
            phases.size,
        )
        actions_history[time_index] = actions
        phases = step(phases, frequencies, coupling, actions)
        trajectory[time_index + 1] = phases

    return SimulationResult(
        unwrapped_phases=trajectory,
        natural_frequencies=frequencies,
        social_actions=actions_history,
        coupling=float(coupling),
    )

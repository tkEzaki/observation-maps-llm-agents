"""Shared helpers for Stage 3A coverage / OOD diagnosis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from circlemap.field_features import COMMON_FEATURE_NAMES, extract_field_descriptors
from circlemap.observation import RelativePhaseHistogram, wrap_phase

from .ood import OODDetector


N_COMMON = len(COMMON_FEATURE_NAMES)
PEER_INDEX = COMMON_FEATURE_NAMES.index("peer_count")
SPARSITY_INDEX = COMMON_FEATURE_NAMES.index("sparsity")
# Shape / symmetry descriptors (exclude peer-size channels).
CANONICAL_INDICES = tuple(
    i for i in range(N_COMMON) if i not in {PEER_INDEX, SPARSITY_INDEX}
)


def histogram_from_phases(
    phases: NDArray[np.float64],
    focal: int,
    *,
    n_bins: int = 24,
) -> RelativePhaseHistogram:
    peer_mask = np.ones(phases.size, dtype=bool)
    peer_mask[focal] = False
    relative = wrap_phase(phases[peer_mask] - phases[focal])
    edges = np.linspace(-np.pi, np.pi, n_bins + 1, dtype=np.float64)
    bin_coordinate = (relative + np.pi) * n_bins / (2.0 * np.pi)
    bin_indices = np.floor(np.round(bin_coordinate, decimals=12)).astype(np.int64) % n_bins
    counts = np.bincount(bin_indices, minlength=n_bins)
    peer_count = int(relative.size)
    fractions = counts.astype(np.float64) / max(peer_count, 1)
    return RelativePhaseHistogram(edges=edges, fractions=fractions, peer_count=peer_count)


def split_feature_views(
    x_full: NDArray[np.float64],
) -> dict[str, NDArray[np.float64]]:
    """Split packed common∥native rows into OOD component views."""
    common = x_full[:, :N_COMMON]
    native = x_full[:, N_COMMON:]
    return {
        "canonical": common[:, list(CANONICAL_INDICES)],
        "native": native,
        "peer": common[:, [PEER_INDEX]],
        "common": common,
        "full": x_full,
    }


@dataclass
class ComponentOOD:
    """Separate detectors for canonical / native / peer channels."""

    canonical: OODDetector
    native: OODDetector
    peer_train_values: NDArray[np.float64]
    peer_tol: float = 0.51  # absolute peer-count tolerance
    x_train_full: NDArray[np.float64] | None = None
    train_meta_indices: NDArray[np.int64] | None = None

    @classmethod
    def fit(cls, x_full: NDArray[np.float64]) -> "ComponentOOD":
        views = split_feature_views(x_full)
        return cls(
            canonical=OODDetector.fit(views["canonical"]),
            native=OODDetector.fit(views["native"]),
            peer_train_values=np.unique(np.round(views["peer"].ravel()).astype(np.float64)),
            x_train_full=x_full,
            train_meta_indices=np.arange(x_full.shape[0], dtype=np.int64),
        )

    def distances(
        self,
        x_full: NDArray[np.float64],
        *,
        unresolved_near_zero: NDArray[np.bool_] | None = None,
    ) -> dict[str, NDArray[np.float64]]:
        views = split_feature_views(x_full)
        d_can = self.canonical.score(views["canonical"])
        d_nat = self.native.score(views["native"])
        peer = views["peer"].ravel()
        # Distance to nearest training peer_count (0 if exact match).
        d_peer = np.min(
            np.abs(peer[:, None] - self.peer_train_values[None, :]),
            axis=1,
        )
        policy = (
            unresolved_near_zero.astype(np.float64)
            if unresolved_near_zero is not None
            else np.zeros(x_full.shape[0], dtype=np.float64)
        )
        # Combined: max of standardized exceedances (soft OR).
        can_ex = d_can / max(self.canonical.threshold, 1e-12)
        nat_ex = d_nat / max(self.native.threshold, 1e-12)
        peer_ex = d_peer / max(self.peer_tol, 1e-12)
        policy_ex = policy  # already 0/1
        combined = np.maximum.reduce([can_ex, nat_ex, peer_ex, policy_ex])
        return {
            "d_canonical": d_can,
            "d_native": d_nat,
            "d_peer": d_peer,
            "d_policy_a": policy,
            "d_combined": combined,
        }

    def flags(
        self,
        x_full: NDArray[np.float64],
        *,
        unresolved_near_zero: NDArray[np.bool_] | None = None,
    ) -> dict[str, NDArray[np.bool_]]:
        dist = self.distances(x_full, unresolved_near_zero=unresolved_near_zero)
        return {
            "ood_canonical": dist["d_canonical"] > self.canonical.threshold,
            "ood_native": dist["d_native"] > self.native.threshold,
            "ood_peer_count": dist["d_peer"] > self.peer_tol,
            "ood_near_zero": dist["d_policy_a"] > 0.5,
            "ood_combined": (
                (dist["d_canonical"] > self.canonical.threshold)
                | (dist["d_native"] > self.native.threshold)
                | (dist["d_peer"] > self.peer_tol)
                | (dist["d_policy_a"] > 0.5)
            ),
        }

    def nearest_training_indices(
        self,
        x_full: NDArray[np.float64],
        *,
        view: str = "canonical",
    ) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
        """Return (indices into training rows, distances) for nearest neighbors."""
        if self.x_train_full is None:
            raise RuntimeError("ComponentOOD fitted without training index map")
        views_q = split_feature_views(x_full)
        views_t = split_feature_views(self.x_train_full)
        q = views_q[view]
        t = views_t[view]
        mean = np.mean(t, axis=0)
        scale = np.std(t, axis=0)
        scale = np.where(scale < 1e-8, 1.0, scale)
        qs = (q - mean) / scale
        ts = (t - mean) / scale
        n = qs.shape[0]
        idx = np.empty(n, dtype=np.int64)
        dist = np.empty(n, dtype=np.float64)
        chunk = 64
        for start in range(0, n, chunk):
            stop = min(start + chunk, n)
            diff = qs[start:stop, None, :] - ts[None, :, :]
            d2 = np.sum(diff * diff, axis=-1)
            local = np.argmin(d2, axis=1)
            idx[start:stop] = local
            dist[start:stop] = np.sqrt(d2[np.arange(stop - start), local])
        return idx, dist


def feature_builder(representation: str, histogram: RelativePhaseHistogram):
    descriptors = extract_field_descriptors(representation, histogram)
    x = np.concatenate([descriptors.common, descriptors.native])
    return x, descriptors.unresolved_near_zero, descriptors


def tv_distance(p: NDArray[np.float64], q: NDArray[np.float64]) -> NDArray[np.float64]:
    return 0.5 * np.sum(np.abs(p - q), axis=-1)

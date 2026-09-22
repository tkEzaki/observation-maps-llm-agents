"""Interval-serialization discrete features (not a centers transplant)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from analysis.centers.discrete_bin_features import _circular_zero_run
from analysis.stage3.applicability_domain import N_COMMON

INTERVAL_DISCRETE_NAMES: tuple[str, ...] = (
    "n_occupied_intervals",
    "n_zero_intervals",
    "max_zero_run",
    "max_bin_mass",
    "top2_mass_gap",
    "peak_interval_distance",
    "half_turn_antipodal_overlap",
    "focal_and_adjacent_mass",
    "boundary_adjacent_mass",
    "interval_edge_asymmetry",
    "boundary_perturbation_sensitivity",
    "cummass_l1_halfturn",
)


def interval_discrete_features(native: NDArray[np.float64]) -> NDArray[np.float64]:
    """Features from the presented 24-bin interval mass vector."""
    p = np.asarray(native, dtype=np.float64)
    single = p.ndim == 1
    if single:
        p = p[None, :]
    n, d = p.shape
    out = np.empty((n, len(INTERVAL_DISCRETE_NAMES)), dtype=np.float64)
    mid = d // 2
    for i in range(n):
        row = p[i]
        s = float(np.sum(row))
        frac = row / s if s > 0 else row
        occ = frac > 1e-12
        n_occ = int(np.sum(occ))
        n_zero = int(d - n_occ)
        max_zero = _circular_zero_run(~occ)
        order = np.argsort(frac)[::-1]
        max_mass = float(frac[order[0]])
        second = float(frac[order[1]]) if d > 1 else 0.0
        top2 = max_mass - second
        peak_i = int(order[0])
        peak_j = int(order[1]) if d > 1 else peak_i
        peak_dist = float(min(abs(peak_i - peak_j), d - abs(peak_i - peak_j)))
        rolled = np.roll(frac, d // 2)
        antipodal = float(np.sum(np.minimum(frac, rolled)))
        focal = float(frac[(mid - 1) % d] + frac[mid % d] + frac[(mid + 1) % d])
        boundary = 0.0
        for b in range(d):
            if not occ[b]:
                continue
            if (not occ[(b - 1) % d]) or (not occ[(b + 1) % d]):
                boundary += float(frac[b])
        # Edge asymmetry: left-half vs right-half mass around focal 0
        left = float(np.sum(frac[:mid]))
        right = float(np.sum(frac[mid:]))
        edge_asym = left - right
        # Boundary perturbation sensitivity: L1 change if mass on peak spills ±1 bin equally
        sens = 0.0
        if max_mass > 0:
            spill = frac.copy()
            move = 0.05 * max_mass
            spill[peak_i] -= move
            spill[(peak_i - 1) % d] += 0.5 * move
            spill[(peak_i + 1) % d] += 0.5 * move
            spill = np.maximum(spill, 0.0)
            spill = spill / max(float(np.sum(spill)), 1e-12)
            sens = float(0.5 * np.sum(np.abs(frac - spill)))
        cum = np.cumsum(frac)
        cum_ht = np.cumsum(rolled)
        cum_l1 = float(np.mean(np.abs(cum - cum_ht)))
        out[i] = (
            float(n_occ),
            float(n_zero),
            float(max_zero),
            max_mass,
            top2,
            peak_dist,
            antipodal,
            focal,
            boundary,
            edge_asym,
            sens,
            cum_l1,
        )
    return out[0] if single else out


class DiscreteAugmentedKernel:
    """Canonical + circular-EMD with interval discrete features on the canonical side."""

    def __init__(self):
        self.name = "canonical_emd_plus_interval_discrete"

    @classmethod
    def fit(cls, x: NDArray[np.float64], counts: NDArray[np.float64], **kwargs):
        from analysis.centers.distances import circular_emd_pairwise
        from analysis.centers.peer_stratified_models import _standardize_fit
        from analysis.stage3.models import counts_to_probs

        native = x[:, N_COMMON:]
        common = x[:, :N_COMMON]
        disc = interval_discrete_features(native)
        xs, mean, scale = _standardize_fit(np.concatenate([common, disc], axis=1))
        obj = cls()
        obj.common_train = xs
        obj.common_mean = mean
        obj.common_scale = scale
        obj.native_train = native.astype(np.float64)
        obj.p_train = counts_to_probs(counts, alpha=0.5)
        obj.length_scale_native = float(kwargs.get("length_scale_native", 0.75))
        obj.length_scale_common = float(kwargs.get("length_scale_common", 1.5))
        obj.k_neighbors = int(kwargs.get("k_neighbors", 32))
        return obj

    def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        from analysis.centers.distances import circular_emd_pairwise
        from analysis.centers.peer_stratified_models import _standardize_apply

        native = x[:, N_COMMON:]
        common = x[:, :N_COMMON]
        disc = interval_discrete_features(native)
        xs = _standardize_apply(
            np.concatenate([common, disc], axis=1), self.common_mean, self.common_scale
        )
        n = x.shape[0]
        out = np.empty((n, 3), dtype=np.float64)
        k = min(self.k_neighbors, self.native_train.shape[0])
        inv_n = 1.0 / (self.length_scale_native**2)
        inv_c = 1.0 / (self.length_scale_common**2)
        for start in range(0, n, 64):
            stop = min(start + 64, n)
            d_n = circular_emd_pairwise(native[start:stop], self.native_train)
            diff = xs[start:stop, None, :] - self.common_train[None, :, :]
            dist2 = (d_n**2) * inv_n + np.sum(diff * diff, axis=-1) * inv_c
            if k < self.native_train.shape[0]:
                idx = np.argpartition(dist2, kth=k - 1, axis=1)[:, :k]
                rows = np.arange(stop - start)[:, None]
                dist2_k = dist2[rows, idx]
                p_k = self.p_train[idx]
            else:
                dist2_k = dist2
                p_k = np.repeat(self.p_train[None, :, :], stop - start, axis=0)
            w = np.exp(-0.5 * dist2_k)
            w = w / np.maximum(np.sum(w, axis=1, keepdims=True), 1e-12)
            out[start:stop] = np.sum(w[:, :, None] * p_k, axis=1)
        return out

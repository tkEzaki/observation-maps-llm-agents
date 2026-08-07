"""Centers-specific discrete features from presented 24-bin vectors."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from analysis.stage3.applicability_domain import N_COMMON

DISCRETE_FEATURE_NAMES: tuple[str, ...] = (
    "n_occupied_bins",
    "n_zero_bins",
    "max_zero_run",
    "max_bin_mass",
    "top2_peak_gap",
    "peak_bin_distance",
    "antipodal_pairing",
    "focal_bin_mass",  # bins adjacent to wrap origin (focal frame)
    "bin_boundary_proximity",  # mass near bin edges via peak sharpness proxy
)


def _circular_zero_run(mask_zero: NDArray[np.bool_]) -> int:
    n = int(mask_zero.size)
    if n == 0:
        return 0
    if bool(np.all(mask_zero)):
        return n
    # Linearize by rotating so a False starts the array when possible.
    doubled = np.concatenate([mask_zero, mask_zero])
    best = 0
    cur = 0
    for flag in doubled:
        if flag:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return int(min(best, n))


def discrete_bin_features_from_native(
    native: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute discrete serialization features; native shape (n, 24) or (24,)."""
    p = np.asarray(native, dtype=np.float64)
    single = p.ndim == 1
    if single:
        p = p[None, :]
    n, d = p.shape
    out = np.empty((n, len(DISCRETE_FEATURE_NAMES)), dtype=np.float64)
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
        top2_gap = max_mass - second
        peak_i = int(order[0])
        peak_j = int(order[1]) if d > 1 else peak_i
        peak_dist = float(min(abs(peak_i - peak_j), d - abs(peak_i - peak_j)))
        # Antipodal pairing: corr of mass with π-shifted bins
        shift = d // 2
        rolled = np.roll(frac, shift)
        antipodal = float(np.sum(np.minimum(frac, rolled)))
        # Focal neighborhood: bins near wrap origin indices 0 and d-1 plus mid? 
        # For centers serialization, edges[0]=-π; focal relative 0 is near bin center.
        # Use mass in bins whose centers are closest to 0: typically bins d/2-1 and d/2.
        mid = d // 2
        focal = float(frac[(mid - 1) % d] + frac[mid % d] + frac[(mid + 1) % d])
        # Boundary proximity proxy: how much mass sits in bins with neighbors empty
        boundary = 0.0
        for b in range(d):
            if not occ[b]:
                continue
            left = occ[(b - 1) % d]
            right = occ[(b + 1) % d]
            if not left or not right:
                boundary += float(frac[b])
        out[i] = (
            float(n_occ),
            float(n_zero),
            float(max_zero),
            max_mass,
            top2_gap,
            peak_dist,
            antipodal,
            focal,
            boundary,
        )
    return out[0] if single else out


def append_discrete_features(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Append discrete features to common∥native feature matrix."""
    native = x[:, N_COMMON:]
    disc = discrete_bin_features_from_native(native)
    return np.concatenate([x, disc], axis=1)

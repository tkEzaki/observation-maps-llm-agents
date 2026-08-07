"""OOD detector for Stage 3A (Policy A near-zero unresolved band)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


def _standardize(
    x: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    mean = np.mean(x, axis=0)
    scale = np.std(x, axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    return (x - mean) / scale, mean, scale


def _unique_rows(x: NDArray[np.float64], *, decimals: int = 6) -> NDArray[np.float64]:
    """Drop exact / near-exact duplicate feature rows (e.g. repeated blocks)."""
    rounded = np.round(x, decimals=decimals)
    _, indices = np.unique(rounded, axis=0, return_index=True)
    return x[np.sort(indices)]


def _kth_neighbor_distance(
    query: NDArray[np.float64],
    ref: NDArray[np.float64],
    *,
    k: int,
) -> NDArray[np.float64]:
    """Distance from each query row to its k-th nearest ref row (1-based k)."""
    k_eff = max(1, min(int(k), ref.shape[0]))
    out = np.empty(query.shape[0], dtype=np.float64)
    chunk = 128
    for start in range(0, query.shape[0], chunk):
        stop = min(start + chunk, query.shape[0])
        diff = query[start:stop, None, :] - ref[None, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=-1))
        # k-th neighbor via partition; if query is a subset of ref, k=1 can be 0.
        partitioned = np.partition(dist, kth=k_eff - 1, axis=1)
        out[start:stop] = partitioned[:, k_eff - 1]
    return out


@dataclass
class OODDetector:
    """Distance-to-manifold OOD score with unresolved-band inflation."""

    x_ref: NDArray[np.float64]
    mean: NDArray[np.float64]
    scale: NDArray[np.float64]
    threshold: float
    k: int = 5
    unresolved_boost: float = 2.0

    @classmethod
    def fit(
        cls,
        x_train: NDArray[np.float64],
        *,
        quantile: float = 0.95,
        k: int = 5,
        max_train: int = 4000,
        seed: int = 0,
    ) -> "OODDetector":
        rng = np.random.default_rng(seed)
        xs, mean, scale = _standardize(x_train)
        unique = _unique_rows(xs)
        if unique.shape[0] > max_train:
            idx = rng.choice(unique.shape[0], size=max_train, replace=False)
            ref = unique[idx]
        else:
            ref = unique
        # Self-scores: distance to k-th neighbor within the unique manifold.
        # Use k+1 when scoring ref against itself so the zero self-match is skipped.
        self_k = min(k + 1, ref.shape[0])
        scores = _kth_neighbor_distance(ref, ref, k=self_k)
        # Guard against all-zero scores (pathological collapse).
        positive = scores[scores > 1e-12]
        if positive.size == 0:
            threshold = 1.0
        else:
            threshold = float(np.quantile(positive, quantile))
            threshold = max(threshold, float(np.median(positive)) * 1e-3)
        return cls(
            x_ref=ref,
            mean=mean,
            scale=scale,
            threshold=threshold,
            k=int(k),
            unresolved_boost=2.0,
        )

    def score(
        self,
        x: NDArray[np.float64],
        *,
        unresolved_near_zero: NDArray[np.bool_] | None = None,
    ) -> NDArray[np.float64]:
        xs = (x - self.mean) / self.scale
        raw = _kth_neighbor_distance(xs, self.x_ref, k=self.k)
        if unresolved_near_zero is not None:
            boost = np.where(unresolved_near_zero, self.unresolved_boost, 1.0)
            raw = raw * boost
        return raw

    def is_ood(
        self,
        x: NDArray[np.float64],
        *,
        unresolved_near_zero: NDArray[np.bool_] | None = None,
    ) -> NDArray[np.bool_]:
        return self.score(x, unresolved_near_zero=unresolved_near_zero) > self.threshold


def ood_error_calibration(
    ood_scores: NDArray[np.float64],
    tv_errors: NDArray[np.float64],
    *,
    n_bins: int = 8,
) -> list[dict[str, float]]:
    """Bin OOD scores and report mean predictive TV error per bin."""
    order = np.argsort(ood_scores)
    scores = ood_scores[order]
    errors = tv_errors[order]
    edges = np.linspace(0, scores.size, n_bins + 1, dtype=int)
    rows = []
    for left, right in zip(edges[:-1], edges[1:]):
        if right <= left:
            continue
        rows.append(
            {
                "bin_left": float(scores[left]),
                "bin_right": float(scores[right - 1]),
                "mean_ood": float(np.mean(scores[left:right])),
                "mean_tv": float(np.mean(errors[left:right])),
                "n": float(right - left),
            }
        )
    return rows

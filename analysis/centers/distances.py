"""Centers-branch histogram distances (native 24-bin support)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _normalize(p: NDArray[np.float64], *, eps: float = 1e-12) -> NDArray[np.float64]:
    p = np.asarray(p, dtype=np.float64)
    if p.ndim == 1:
        s = float(np.sum(p))
        return p / max(s, eps)
    s = np.sum(p, axis=-1, keepdims=True)
    return p / np.maximum(s, eps)


def hellinger(p: NDArray[np.float64], q: NDArray[np.float64]) -> float:
    """Hellinger distance in [0, 1]."""
    a = np.sqrt(_normalize(p))
    b = np.sqrt(_normalize(q))
    return float(np.sqrt(np.clip(0.5 * np.sum((a - b) ** 2), 0.0, 1.0)))


def hellinger_pairwise(
    query: NDArray[np.float64], ref: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Pairwise Hellinger; shapes (n_q, d), (n_r, d) → (n_q, n_r)."""
    a = np.sqrt(_normalize(query))
    b = np.sqrt(_normalize(ref))
    # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a·b; ||a||^2≈1
    dots = a @ b.T
    dist2 = np.clip(1.0 - dots, 0.0, 1.0)
    return np.sqrt(dist2)


def js_distance(p: NDArray[np.float64], q: NDArray[np.float64]) -> float:
    """Jensen–Shannon distance (sqrt of JS divergence, base-e)."""
    a = _normalize(p)
    b = _normalize(q)
    m = 0.5 * (a + b)

    def _kl(u: NDArray[np.float64], v: NDArray[np.float64]) -> float:
        mask = u > 0
        return float(np.sum(u[mask] * np.log(u[mask] / np.maximum(v[mask], 1e-12))))

    js = 0.5 * _kl(a, m) + 0.5 * _kl(b, m)
    return float(np.sqrt(max(js, 0.0)))


def circular_emd(p: NDArray[np.float64], q: NDArray[np.float64]) -> float:
    """Circular 1-Wasserstein on equal bins (bin units), fixed alignment.

    Does **not** minimize over cyclic shifts: centers serialization is
    orientation-sensitive, so the presented 24-bin vectors are compared as-is.
    """
    a = _normalize(p)
    b = _normalize(q)
    cdf = np.cumsum(a - b)
    cdf = cdf - np.mean(cdf)
    return float(np.mean(np.abs(cdf)))


def circular_emd_pairwise(
    query: NDArray[np.float64],
    ref: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Vectorized pairwise circular EMD (fixed alignment)."""
    q = _normalize(query)
    r = _normalize(ref)
    # For each pair, CDF of difference — compute via broadcasting carefully.
    # Use: for each query row against all refs.
    n_q = q.shape[0]
    n_r = r.shape[0]
    out = np.empty((n_q, n_r), dtype=np.float64)
    for i in range(n_q):
        diff = q[i][None, :] - r
        cdf = np.cumsum(diff, axis=1)
        cdf = cdf - np.mean(cdf, axis=1, keepdims=True)
        out[i] = np.mean(np.abs(cdf), axis=1)
    return out


def js_pairwise(
    query: NDArray[np.float64], ref: NDArray[np.float64]
) -> NDArray[np.float64]:
    q = _normalize(query)
    r = _normalize(ref)
    n_q = q.shape[0]
    n_r = r.shape[0]
    out = np.empty((n_q, n_r), dtype=np.float64)
    for i in range(n_q):
        for j in range(n_r):
            out[i, j] = js_distance(q[i], r[j])
    return out

"""Applicability-domain layers for Stage 3A (not a single OR-ed OOD flag).

Layers
------
1. unsupported peer count (hard gate)
2. shape support (rotation-invariant magnitudes / symmetry)
3. native support (representation-specific distances)
4. Policy-A near-zero unresolved band

Predictive risk R(x) is calibrated separately from grouped OOF residuals.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from circlemap.field_features import COMMON_FEATURE_NAMES

N_COMMON = len(COMMON_FEATURE_NAMES)
PEER_INDEX = COMMON_FEATURE_NAMES.index("peer_count")

# Rotation-invariant shape channels (exclude Re/Im orientation).
SHAPE_NAMES: tuple[str, ...] = (
    "abs_z1",
    "abs_z2",
    "abs_z3",
    "circular_entropy",
    "antipodal_balance",
    "bimodality",
    "sparsity",
)
SHAPE_INDICES = tuple(COMMON_FEATURE_NAMES.index(name) for name in SHAPE_NAMES)

# Orientation channels (arg proxies via Re/Im pairs) — for orientation support only.
ORIENT_INDICES = tuple(
    COMMON_FEATURE_NAMES.index(name)
    for name in ("re_z1", "im_z1", "re_z2", "im_z2", "re_z3", "im_z3")
)

DEFAULT_TRAIN_PEERS: tuple[int, ...] = (8, 16, 240)


def shape_matrix(x_full: NDArray[np.float64]) -> NDArray[np.float64]:
    return x_full[:, list(SHAPE_INDICES)]


def orientation_matrix(x_full: NDArray[np.float64]) -> NDArray[np.float64]:
    return x_full[:, list(ORIENT_INDICES)]


def native_matrix(x_full: NDArray[np.float64]) -> NDArray[np.float64]:
    return x_full[:, N_COMMON:]


def peer_vector(x_full: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.round(x_full[:, PEER_INDEX]).astype(np.float64)


def unsupported_peer(
    peers: NDArray[np.float64],
    train_peers: NDArray[np.float64] | tuple[int, ...] = DEFAULT_TRAIN_PEERS,
) -> NDArray[np.bool_]:
    allowed = np.asarray(train_peers, dtype=np.float64)
    return ~np.isin(np.round(peers), allowed)


def _standardize(
    x: NDArray[np.float64],
    mean: NDArray[np.float64] | None = None,
    scale: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    if mean is None:
        mean = np.mean(x, axis=0)
    if scale is None:
        scale = np.std(x, axis=0)
        scale = np.where(scale < 1e-8, 1.0, scale)
    return (x - mean) / scale, mean, scale


def _unique_rows(x: NDArray[np.float64], *, decimals: int = 5) -> NDArray[np.float64]:
    rounded = np.round(x, decimals=decimals)
    _, indices = np.unique(rounded, axis=0, return_index=True)
    return x[np.sort(indices)]


def _kth_distance(
    query: NDArray[np.float64],
    ref: NDArray[np.float64],
    *,
    k: int,
) -> NDArray[np.float64]:
    k_eff = max(1, min(int(k), ref.shape[0]))
    out = np.empty(query.shape[0], dtype=np.float64)
    chunk = 128
    for start in range(0, query.shape[0], chunk):
        stop = min(start + chunk, query.shape[0])
        diff = query[start:stop, None, :] - ref[None, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=-1))
        part = np.partition(dist, kth=k_eff - 1, axis=1)
        out[start:stop] = part[:, k_eff - 1]
    return out


def hellinger_to_ref(
    query: NDArray[np.float64],
    ref: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Min Hellinger distance from each query histogram row to the ref set."""
    q = np.clip(query, 0.0, None)
    r = np.clip(ref, 0.0, None)
    q = q / np.maximum(np.sum(q, axis=1, keepdims=True), 1e-12)
    r = r / np.maximum(np.sum(r, axis=1, keepdims=True), 1e-12)
    sqrt_q = np.sqrt(q)
    sqrt_r = np.sqrt(r)
    out = np.empty(query.shape[0], dtype=np.float64)
    chunk = 64
    for start in range(0, query.shape[0], chunk):
        stop = min(start + chunk, query.shape[0])
        # ||sqrt_q - sqrt_r|| / sqrt(2)
        diff = sqrt_q[start:stop, None, :] - sqrt_r[None, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=-1) / 2.0)
        out[start:stop] = np.min(dist, axis=1)
    return out


@dataclass
class SupportLayer:
    """Density-normalized support score for a fixed feature view."""

    ref: NDArray[np.float64]
    mean: NDArray[np.float64]
    scale: NDArray[np.float64]
    local_scale: float
    k: int
    kind: str = "euclidean"  # or hellinger

    @classmethod
    def fit_euclidean(
        cls,
        x: NDArray[np.float64],
        *,
        k: int = 5,
        max_ref: int = 3000,
        seed: int = 0,
    ) -> "SupportLayer":
        rng = np.random.default_rng(seed)
        xs, mean, scale = _standardize(x)
        unique = _unique_rows(xs)
        if unique.shape[0] > max_ref:
            unique = unique[rng.choice(unique.shape[0], size=max_ref, replace=False)]
        self_k = min(k + 1, unique.shape[0])
        self_d = _kth_distance(unique, unique, k=self_k)
        positive = self_d[self_d > 1e-12]
        local_scale = float(np.median(positive)) if positive.size else 1.0
        local_scale = max(local_scale, 1e-6)
        return cls(
            ref=unique,
            mean=mean,
            scale=scale,
            local_scale=local_scale,
            k=int(k),
            kind="euclidean",
        )

    @classmethod
    def fit_hellinger(
        cls,
        x: NDArray[np.float64],
        *,
        max_ref: int = 2500,
        seed: int = 0,
    ) -> "SupportLayer":
        rng = np.random.default_rng(seed)
        unique = _unique_rows(x, decimals=6)
        if unique.shape[0] > max_ref:
            unique = unique[rng.choice(unique.shape[0], size=max_ref, replace=False)]
        # Local scale from leave-one-ish min Hellinger among ref.
        sample = unique
        if sample.shape[0] > 400:
            sample = unique[rng.choice(unique.shape[0], size=400, replace=False)]
        d = hellinger_to_ref(sample, unique)
        # Self matches ~0; use upper quantiles of positive distances.
        positive = d[d > 1e-8]
        local_scale = float(np.median(positive)) if positive.size else 0.1
        local_scale = max(local_scale, 1e-6)
        return cls(
            ref=unique,
            mean=np.zeros(x.shape[1]),
            scale=np.ones(x.shape[1]),
            local_scale=local_scale,
            k=1,
            kind="hellinger",
        )

    def distance(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.kind == "hellinger":
            raw = hellinger_to_ref(x, self.ref)
        else:
            xs = (x - self.mean) / self.scale
            raw = _kth_distance(xs, self.ref, k=self.k)
        return raw / self.local_scale

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            ref=self.ref,
            mean=self.mean,
            scale=self.scale,
            local_scale=np.asarray(self.local_scale, dtype=np.float64),
            k=np.asarray(self.k, dtype=np.int64),
            kind=np.asarray(self.kind),
        )

    @classmethod
    def load(cls, path: Path) -> "SupportLayer":
        payload = np.load(Path(path), allow_pickle=False)
        return cls(
            ref=np.asarray(payload["ref"], dtype=np.float64),
            mean=np.asarray(payload["mean"], dtype=np.float64),
            scale=np.asarray(payload["scale"], dtype=np.float64),
            local_scale=float(payload["local_scale"]),
            k=int(payload["k"]),
            kind=str(payload["kind"]),
        )


@dataclass
class ApplicabilityDomain:
    """Four-layer AD plus optional orientation support."""

    train_peers: NDArray[np.float64]
    shape: SupportLayer
    orientation: SupportLayer
    native: SupportLayer
    representation: str
    is_histogram_native: bool

    @classmethod
    def fit(
        cls,
        x_full: NDArray[np.float64],
        *,
        representation: str,
        train_peers: tuple[int, ...] = DEFAULT_TRAIN_PEERS,
    ) -> "ApplicabilityDomain":
        is_hist = "moments" not in representation
        native = native_matrix(x_full)
        if is_hist:
            native_layer = SupportLayer.fit_hellinger(native)
        else:
            native_layer = SupportLayer.fit_euclidean(native)
        return cls(
            train_peers=np.asarray(train_peers, dtype=np.float64),
            shape=SupportLayer.fit_euclidean(shape_matrix(x_full)),
            orientation=SupportLayer.fit_euclidean(orientation_matrix(x_full)),
            native=native_layer,
            representation=representation,
            is_histogram_native=is_hist,
        )

    def score(
        self,
        x_full: NDArray[np.float64],
        *,
        unresolved_near_zero: NDArray[np.bool_] | None = None,
    ) -> dict[str, NDArray[np.float64] | NDArray[np.bool_]]:
        peers = peer_vector(x_full)
        g_peer = unsupported_peer(peers, self.train_peers)
        d_shape = self.shape.distance(shape_matrix(x_full))
        d_orient = self.orientation.distance(orientation_matrix(x_full))
        d_native = self.native.distance(native_matrix(x_full))
        policy = (
            unresolved_near_zero.astype(np.float64)
            if unresolved_near_zero is not None
            else np.zeros(x_full.shape[0], dtype=np.float64)
        )
        # Local density proxy: inverse of shape distance (clipped).
        local_density = 1.0 / np.maximum(d_shape, 1e-6)
        return {
            "g_peer": g_peer,
            "d_shape": d_shape,
            "d_orientation": d_orient,
            "d_native": d_native,
            "policy_a": policy,
            "local_density": local_density,
            "peer_count": peers,
        }

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.shape.save(directory / "shape.npz")
        self.orientation.save(directory / "orientation.npz")
        self.native.save(directory / "native.npz")
        meta = {
            "representation": self.representation,
            "is_histogram_native": bool(self.is_histogram_native),
            "train_peers": [int(p) for p in self.train_peers.tolist()],
        }
        (directory / "meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: Path) -> "ApplicabilityDomain":
        directory = Path(directory)
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        return cls(
            train_peers=np.asarray(meta["train_peers"], dtype=np.float64),
            shape=SupportLayer.load(directory / "shape.npz"),
            orientation=SupportLayer.load(directory / "orientation.npz"),
            native=SupportLayer.load(directory / "native.npz"),
            representation=str(meta["representation"]),
            is_histogram_native=bool(meta["is_histogram_native"]),
        )

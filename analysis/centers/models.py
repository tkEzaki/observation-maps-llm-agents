"""Centers-branch response models (do not clone moments hurdle by default)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from analysis.centers.distances import circular_emd_pairwise, hellinger_pairwise
from analysis.stage3.applicability_domain import N_COMMON
from analysis.stage3.models import (
    KernelHurdleMultinomial,
    KernelNeighborBaseline,
    SoftmaxStumpBoost,
    counts_to_probs,
)
from circlemap.field_features import COMMON_FEATURE_NAMES


ABS_Z1 = COMMON_FEATURE_NAMES.index("abs_z1")
ABS_Z2 = COMMON_FEATURE_NAMES.index("abs_z2")
ABS_Z3 = COMMON_FEATURE_NAMES.index("abs_z3")
ANTIPODAL = COMMON_FEATURE_NAMES.index("antipodal_balance")


def _standardize_fit(x: NDArray[np.float64]):
    mean = np.mean(x, axis=0)
    scale = np.std(x, axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    return (x - mean) / scale, mean, scale


def _standardize_apply(
    x: NDArray[np.float64], mean: NDArray[np.float64], scale: NDArray[np.float64]
) -> NDArray[np.float64]:
    return (x - mean) / scale


@dataclass
class HybridNativeKernel:
    """Canonical Euclidean + native Hellinger/circular-EMD kernel multinomial."""

    name: str
    x_common_train: NDArray[np.float64]
    native_train: NDArray[np.float64]
    p_train: NDArray[np.float64]
    common_mean: NDArray[np.float64]
    common_scale: NDArray[np.float64]
    length_scale_common: float
    length_scale_native: float
    k_neighbors: int
    native_metric: str  # hellinger | circular_emd

    @classmethod
    def fit(
        cls,
        x: NDArray[np.float64],
        counts: NDArray[np.float64],
        *,
        native_metric: str = "hellinger",
        length_scale_common: float = 1.5,
        length_scale_native: float = 0.75,
        k_neighbors: int = 32,
        alpha: float = 0.5,
    ) -> "HybridNativeKernel":
        common = x[:, :N_COMMON]
        native = x[:, N_COMMON:]
        xs, mean, scale = _standardize_fit(common)
        return cls(
            name=f"hybrid_kernel_{native_metric}",
            x_common_train=xs,
            native_train=native.astype(np.float64),
            p_train=counts_to_probs(counts, alpha=alpha),
            common_mean=mean,
            common_scale=scale,
            length_scale_common=float(length_scale_common),
            length_scale_native=float(length_scale_native),
            k_neighbors=int(k_neighbors),
            native_metric=native_metric,
        )

    def _native_dist(
        self, native_q: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        if self.native_metric == "hellinger":
            return hellinger_pairwise(native_q, self.native_train)
        if self.native_metric == "circular_emd":
            return circular_emd_pairwise(native_q, self.native_train)
        raise KeyError(self.native_metric)

    def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        common = _standardize_apply(x[:, :N_COMMON], self.common_mean, self.common_scale)
        native = x[:, N_COMMON:]
        n = x.shape[0]
        out = np.empty((n, self.p_train.shape[1]), dtype=np.float64)
        k = min(self.k_neighbors, self.x_common_train.shape[0])
        inv_c = 1.0 / (self.length_scale_common**2)
        inv_n = 1.0 / (self.length_scale_native**2)
        chunk = 64
        for start in range(0, n, chunk):
            stop = min(start + chunk, n)
            diff = common[start:stop, None, :] - self.x_common_train[None, :, :]
            d_c2 = np.sum(diff * diff, axis=-1)
            d_n = self._native_dist(native[start:stop])
            dist2 = d_c2 * inv_c + (d_n**2) * inv_n
            if k < self.x_common_train.shape[0]:
                idx = np.argpartition(dist2, kth=k - 1, axis=1)[:, :k]
                rows = np.arange(stop - start)[:, None]
                dist2_k = dist2[rows, idx]
                p_k = self.p_train[idx]
            else:
                dist2_k = dist2
                p_k = np.repeat(self.p_train[None, :, :], stop - start, axis=0)
            weights = np.exp(-0.5 * dist2_k)
            weights = weights / np.maximum(np.sum(weights, axis=1, keepdims=True), 1e-12)
            out[start:stop] = np.sum(weights[:, :, None] * p_k, axis=1)
        return out

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            directory / "weights.npz",
            x_common_train=self.x_common_train,
            native_train=self.native_train,
            p_train=self.p_train,
            common_mean=self.common_mean,
            common_scale=self.common_scale,
        )
        meta = {
            "name": self.name,
            "length_scale_common": self.length_scale_common,
            "length_scale_native": self.length_scale_native,
            "k_neighbors": self.k_neighbors,
            "native_metric": self.native_metric,
        }
        (directory / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> "HybridNativeKernel":
        directory = Path(directory)
        payload = np.load(directory / "weights.npz")
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        return cls(
            name=str(meta["name"]),
            x_common_train=payload["x_common_train"],
            native_train=payload["native_train"],
            p_train=payload["p_train"],
            common_mean=payload["common_mean"],
            common_scale=payload["common_scale"],
            length_scale_common=float(meta["length_scale_common"]),
            length_scale_native=float(meta["length_scale_native"]),
            k_neighbors=int(meta["k_neighbors"]),
            native_metric=str(meta["native_metric"]),
        )


@dataclass
class SoftRegimeMixture:
    """Soft mixture of local multinomial kernels gated by field descriptors."""

    name: str
    experts: list[KernelNeighborBaseline]
    gate_mean: NDArray[np.float64]
    gate_scale: NDArray[np.float64]
    gate_centers: NDArray[np.float64]
    gate_temperature: float
    gate_indices: tuple[int, ...]

    @staticmethod
    def _gate_features(x: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.column_stack(
            [
                x[:, ABS_Z1],
                x[:, ABS_Z2],
                x[:, ABS_Z3],
                x[:, ANTIPODAL],
            ]
        )

    @classmethod
    def fit(
        cls,
        x: NDArray[np.float64],
        counts: NDArray[np.float64],
        *,
        n_experts: int = 4,
        gate_temperature: float = 0.35,
        alpha: float = 0.5,
    ) -> "SoftRegimeMixture":
        g = cls._gate_features(x)
        gs, mean, scale = _standardize_fit(g)
        # Initialize centers by quantiles on abs_z1 × antipodal plane.
        abs_z1 = gs[:, 0]
        bal = gs[:, 3]
        centers = []
        for a_lo, a_hi, b_lo, b_hi in (
            (0.50, 1.01, -np.inf, np.inf),  # strong polar
            (-0.25, 0.50, -np.inf, 0.0),  # weak / even-ish
            (-np.inf, -0.25, -np.inf, np.inf),  # low |z1|
            (-0.10, 0.60, 0.5, np.inf),  # antipodal-ish
        ):
            mask = (abs_z1 >= a_lo) & (abs_z1 < a_hi) & (bal >= b_lo) & (bal < b_hi)
            if int(np.sum(mask)) < 20:
                mask = np.ones(gs.shape[0], dtype=bool)
            centers.append(np.mean(gs[mask], axis=0))
        centers_arr = np.asarray(centers[:n_experts], dtype=np.float64)
        # Soft assignment for expert training weights via nearest center.
        experts = []
        for e in range(n_experts):
            d2 = np.sum((gs - centers_arr[e]) ** 2, axis=1)
            # Take closest 40% mass or at least 80 rows
            order = np.argsort(d2)
            n_take = max(80, int(0.4 * gs.shape[0]))
            idx = order[: min(n_take, gs.shape[0])]
            experts.append(
                KernelNeighborBaseline.fit(
                    x[idx], counts[idx], length_scale=1.4, k_neighbors=24, alpha=alpha
                )
            )
        return cls(
            name="soft_regime_mixture",
            experts=experts,
            gate_mean=mean,
            gate_scale=scale,
            gate_centers=centers_arr,
            gate_temperature=float(gate_temperature),
            gate_indices=(ABS_Z1, ABS_Z2, ABS_Z3, ANTIPODAL),
        )

    def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        g = _standardize_apply(self._gate_features(x), self.gate_mean, self.gate_scale)
        # Softmax over negative distances to centers
        d2 = np.sum((g[:, None, :] - self.gate_centers[None, :, :]) ** 2, axis=-1)
        logits = -d2 / max(self.gate_temperature, 1e-6)
        logits = logits - np.max(logits, axis=1, keepdims=True)
        w = np.exp(logits)
        w = w / np.maximum(np.sum(w, axis=1, keepdims=True), 1e-12)
        preds = [expert.predict_proba(x) for expert in self.experts]
        stacked = np.stack(preds, axis=1)  # (n, e, 3)
        return np.sum(w[:, :, None] * stacked, axis=1)


def fit_centers_candidates(
    x: NDArray[np.float64],
    counts: NDArray[np.float64],
) -> dict[str, object]:
    """Fit the CENT-1 candidate set."""
    models: dict[str, object] = {
        "global_empirical": __import__(
            "analysis.stage3.models", fromlist=["GlobalEmpiricalBaseline"]
        ).GlobalEmpiricalBaseline.fit(counts),
        "kernel_nn_euclidean": KernelNeighborBaseline.fit(x, counts),
        "hybrid_hellinger": HybridNativeKernel.fit(
            x, counts, native_metric="hellinger"
        ),
        "hybrid_circular_emd": HybridNativeKernel.fit(
            x, counts, native_metric="circular_emd", k_neighbors=24
        ),
        "activity_direction_hurdle": KernelHurdleMultinomial.fit(x, counts),
        "soft_regime_mixture": SoftRegimeMixture.fit(x, counts),
        "softmax_stump_boost": SoftmaxStumpBoost.fit(x, counts, n_estimators=30),
    }
    return models

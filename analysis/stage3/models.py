"""Full-trinomial surrogate candidates (numpy-only; no deep nets)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray


def _softmax(logits: NDArray[np.float64]) -> NDArray[np.float64]:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def _standardize_fit(x: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    mean = np.mean(x, axis=0)
    scale = np.std(x, axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    return (x - mean) / scale, mean, scale


def _standardize_apply(
    x: NDArray[np.float64],
    mean: NDArray[np.float64],
    scale: NDArray[np.float64],
) -> NDArray[np.float64]:
    return (x - mean) / scale


def counts_to_probs(
    counts: NDArray[np.float64],
    *,
    alpha: float | tuple[float, ...] | NDArray[np.float64] = 0.5,
) -> NDArray[np.float64]:
    """Dirichlet-smoothed empirical multinomials (scalar or per-class alpha)."""
    smoothed = counts.astype(np.float64) + np.asarray(alpha, dtype=np.float64)
    return smoothed / np.sum(smoothed, axis=1, keepdims=True)


class TrinomialModel(Protocol):
    name: str

    def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        ...


@dataclass
class GlobalEmpiricalBaseline:
    """Constant multinomial = training-set empirical distribution."""

    name: str
    probs: NDArray[np.float64]

    @classmethod
    def fit(cls, counts: NDArray[np.float64]) -> "GlobalEmpiricalBaseline":
        totals = np.sum(counts, axis=0).astype(np.float64) + 0.5
        probs = totals / np.sum(totals)
        return cls(name="global_empirical", probs=probs)

    def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        n = int(x.shape[0])
        return np.repeat(self.probs[None, :], n, axis=0)


@dataclass
class KernelNeighborBaseline:
    """RBF kernel-weighted average of training trinomials."""

    name: str
    x_train: NDArray[np.float64]
    p_train: NDArray[np.float64]
    mean: NDArray[np.float64]
    scale: NDArray[np.float64]
    length_scale: float
    k_neighbors: int

    @classmethod
    def fit(
        cls,
        x: NDArray[np.float64],
        counts: NDArray[np.float64],
        *,
        length_scale: float = 1.5,
        k_neighbors: int = 32,
        alpha: float | tuple[float, ...] | NDArray[np.float64] = 0.5,
    ) -> "KernelNeighborBaseline":
        xs, mean, scale = _standardize_fit(x)
        return cls(
            name="kernel_nn",
            x_train=xs,
            p_train=counts_to_probs(counts, alpha=alpha),
            mean=mean,
            scale=scale,
            length_scale=float(length_scale),
            k_neighbors=int(k_neighbors),
        )

    def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        xs = _standardize_apply(x, self.mean, self.scale)
        n_classes = int(self.p_train.shape[1])
        # Chunked distance to limit peak memory on large training sets.
        out = np.empty((xs.shape[0], n_classes), dtype=np.float64)
        chunk = 256
        k = min(self.k_neighbors, self.x_train.shape[0])
        inv_ls2 = 1.0 / (self.length_scale ** 2)
        for start in range(0, xs.shape[0], chunk):
            stop = min(start + chunk, xs.shape[0])
            diff = xs[start:stop, None, :] - self.x_train[None, :, :]
            dist2 = np.sum(diff * diff, axis=-1)
            if k < self.x_train.shape[0]:
                idx = np.argpartition(dist2, kth=k - 1, axis=1)[:, :k]
                rows = np.arange(stop - start)[:, None]
                dist2_k = dist2[rows, idx]
                p_k = self.p_train[idx]
            else:
                dist2_k = dist2
                p_k = self.p_train[None, :, :].repeat(stop - start, axis=0)
            weights = np.exp(-0.5 * dist2_k * inv_ls2)
            weights = weights / np.maximum(np.sum(weights, axis=1, keepdims=True), 1e-12)
            out[start:stop] = np.sum(weights[:, :, None] * p_k, axis=1)
        return out

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            name=np.asarray(self.name),
            x_train=self.x_train,
            p_train=self.p_train,
            mean=self.mean,
            scale=self.scale,
            length_scale=np.asarray(self.length_scale, dtype=np.float64),
            k_neighbors=np.asarray(self.k_neighbors, dtype=np.int64),
        )

    @classmethod
    def load(cls, path: Path) -> "KernelNeighborBaseline":
        payload = np.load(Path(path), allow_pickle=False)
        return cls(
            name=str(payload["name"]),
            x_train=np.asarray(payload["x_train"], dtype=np.float64),
            p_train=np.asarray(payload["p_train"], dtype=np.float64),
            mean=np.asarray(payload["mean"], dtype=np.float64),
            scale=np.asarray(payload["scale"], dtype=np.float64),
            length_scale=float(payload["length_scale"]),
            k_neighbors=int(payload["k_neighbors"]),
        )


@dataclass
class MultinomialLogisticL2:
    """Regularized multinomial logistic regression via gradient descent."""

    name: str
    weights: NDArray[np.float64]  # (n_features+1, 3) with bias row
    mean: NDArray[np.float64]
    scale: NDArray[np.float64]

    @classmethod
    def fit(
        cls,
        x: NDArray[np.float64],
        counts: NDArray[np.float64],
        *,
        l2: float = 1.0,
        lr: float = 0.05,
        steps: int = 250,
        seed: int = 0,
    ) -> "MultinomialLogisticL2":
        rng = np.random.default_rng(seed)
        xs, mean, scale = _standardize_fit(x)
        n, d = xs.shape
        design = np.concatenate([np.ones((n, 1)), xs], axis=1)
        weights = 0.01 * rng.normal(size=(d + 1, 3))
        totals = np.sum(counts, axis=1, keepdims=True)
        totals = np.maximum(totals, 1.0)
        y = counts / totals
        sample_w = totals.ravel() / np.mean(totals)
        for _ in range(steps):
            logits = design @ weights
            probs = _softmax(logits)
            grad = design.T @ ((probs - y) * sample_w[:, None]) / n
            grad[1:, :] += l2 * weights[1:, :]
            weights -= lr * grad
            # Remove gauge freedom: center class logits.
            weights -= np.mean(weights, axis=1, keepdims=True)
        return cls(name="multinomial_logistic_l2", weights=weights, mean=mean, scale=scale)

    def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        xs = _standardize_apply(x, self.mean, self.scale)
        design = np.concatenate([np.ones((xs.shape[0], 1)), xs], axis=1)
        return _softmax(design @ self.weights)


@dataclass
class SoftmaxStumpBoost:
    """Shallow multinomial boosting with axis-aligned stumps (depth-1)."""

    name: str
    stumps: list[dict]
    init_logits: NDArray[np.float64]
    mean: NDArray[np.float64]
    scale: NDArray[np.float64]
    learning_rate: float

    @classmethod
    def fit(
        cls,
        x: NDArray[np.float64],
        counts: NDArray[np.float64],
        *,
        n_estimators: int = 40,
        learning_rate: float = 0.35,
        n_thresholds: int = 8,
        seed: int = 0,
    ) -> "SoftmaxStumpBoost":
        rng = np.random.default_rng(seed)
        xs, mean, scale = _standardize_fit(x)
        n, d = xs.shape
        totals = np.sum(counts, axis=1, keepdims=True)
        totals = np.maximum(totals, 1.0)
        y = counts / totals
        sample_w = totals.ravel()
        sample_w = sample_w / np.mean(sample_w)
        # Start at global empirical logits.
        global_p = np.average(y, axis=0, weights=sample_w)
        global_p = np.clip(global_p, 1e-6, 1.0)
        global_p = global_p / np.sum(global_p)
        init_logits = np.log(global_p)
        logits = np.repeat(init_logits[None, :], n, axis=0)
        stumps: list[dict] = []
        feature_order = np.arange(d)
        for round_index in range(n_estimators):
            probs = _softmax(logits)
            residual = (y - probs) * sample_w[:, None]
            rng.shuffle(feature_order)
            best = None
            best_score = -np.inf
            for feature_index in feature_order[: min(d, 24)]:
                column = xs[:, feature_index]
                qs = np.quantile(column, np.linspace(0.15, 0.85, n_thresholds))
                for threshold in qs:
                    left = column <= threshold
                    right = ~left
                    if not np.any(left) or not np.any(right):
                        continue
                    left_val = np.average(residual[left], axis=0)
                    right_val = np.average(residual[right], axis=0)
                    # Score: explained residual energy.
                    pred = np.empty_like(residual)
                    pred[left] = left_val
                    pred[right] = right_val
                    score = -float(np.mean((residual - pred) ** 2))
                    if score > best_score:
                        best_score = score
                        best = {
                            "feature_index": int(feature_index),
                            "threshold": float(threshold),
                            "left": left_val.astype(np.float64),
                            "right": right_val.astype(np.float64),
                            "round": int(round_index),
                        }
            if best is None:
                break
            stumps.append(best)
            column = xs[:, best["feature_index"]]
            left = column <= best["threshold"]
            update = np.empty((n, 3), dtype=np.float64)
            update[left] = best["left"]
            update[~left] = best["right"]
            logits = logits + learning_rate * update
        return cls(
            name="softmax_stump_boost",
            stumps=stumps,
            init_logits=init_logits.astype(np.float64),
            mean=mean,
            scale=scale,
            learning_rate=float(learning_rate),
        )

    def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        xs = _standardize_apply(x, self.mean, self.scale)
        logits = np.repeat(self.init_logits[None, :], xs.shape[0], axis=0)
        for stump in self.stumps:
            left = xs[:, stump["feature_index"]] <= stump["threshold"]
            update = np.empty((xs.shape[0], 3), dtype=np.float64)
            update[left] = stump["left"]
            update[~left] = stump["right"]
            logits = logits + self.learning_rate * update
        return _softmax(logits)


@dataclass
class HurdleMultinomial:
    """Two-stage stay vs move, then signed direction given move.

    p0 = s(X), p+ = (1-s) q(X), p- = (1-s)(1-q).
    Useful for moments abstention / activation structure.
    """

    name: str
    stay_weights: NDArray[np.float64]
    stay_mean: NDArray[np.float64]
    stay_scale: NDArray[np.float64]
    dir_weights: NDArray[np.float64]
    dir_mean: NDArray[np.float64]
    dir_scale: NDArray[np.float64]

    @staticmethod
    def _fit_binary(
        x: NDArray[np.float64],
        y: NDArray[np.float64],
        sample_w: NDArray[np.float64],
        *,
        l2: float,
        lr: float,
        steps: int,
        seed: int,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Fit P(y=1|x) with logistic; returns (weights[d+1], mean, scale)."""
        rng = np.random.default_rng(seed)
        xs, mean, scale = _standardize_fit(x)
        n, d = xs.shape
        design = np.concatenate([np.ones((n, 1)), xs], axis=1)
        w = 0.01 * rng.normal(size=d + 1)
        for _ in range(steps):
            logits = design @ w
            probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -40, 40)))
            grad = design.T @ ((probs - y) * sample_w) / n
            grad[1:] += l2 * w[1:]
            w -= lr * grad
        return w, mean, scale

    @classmethod
    def fit(
        cls,
        x: NDArray[np.float64],
        counts: NDArray[np.float64],
        *,
        l2: float = 1.0,
        lr: float = 0.05,
        steps: int = 250,
        seed: int = 0,
    ) -> "HurdleMultinomial":
        totals = np.sum(counts, axis=1)
        totals = np.maximum(totals, 1.0)
        stay = counts[:, 1] / totals
        move = totals - counts[:, 1]
        denom = np.maximum(counts[:, 0] + counts[:, 2], 1e-9)
        q = counts[:, 2] / denom
        sample_w = totals / np.mean(totals)
        stay_w, stay_mean, stay_scale = cls._fit_binary(
            x, stay, sample_w, l2=l2, lr=lr, steps=steps, seed=seed
        )
        dir_w_sample = (move / totals) * sample_w
        dir_w_sample = dir_w_sample / max(float(np.mean(dir_w_sample)), 1e-12)
        dir_weights, dir_mean, dir_scale = cls._fit_binary(
            x, q, dir_w_sample, l2=l2, lr=lr, steps=steps, seed=seed + 1
        )
        return cls(
            name="hurdle_multinomial",
            stay_weights=stay_w,
            stay_mean=stay_mean,
            stay_scale=stay_scale,
            dir_weights=dir_weights,
            dir_mean=dir_mean,
            dir_scale=dir_scale,
        )

    def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        xs_s = _standardize_apply(x, self.stay_mean, self.stay_scale)
        design_s = np.concatenate([np.ones((xs_s.shape[0], 1)), xs_s], axis=1)
        logits_s = design_s @ self.stay_weights
        s = 1.0 / (1.0 + np.exp(-np.clip(logits_s, -40, 40)))
        xs_d = _standardize_apply(x, self.dir_mean, self.dir_scale)
        design_d = np.concatenate([np.ones((xs_d.shape[0], 1)), xs_d], axis=1)
        logits_d = design_d @ self.dir_weights
        q = 1.0 / (1.0 + np.exp(-np.clip(logits_d, -40, 40)))
        p0 = s
        pp = (1.0 - s) * q
        pm = (1.0 - s) * (1.0 - q)
        return np.column_stack([pm, p0, pp])


@dataclass
class KernelHurdleMultinomial:
    """Two-stage hurdle with kernel-NN stay gate and kernel-NN direction."""

    name: str
    stay_model: KernelNeighborBaseline
    dir_model: KernelNeighborBaseline
    stay_feature_indices: tuple[int, ...] | None = None

    @classmethod
    def fit(
        cls,
        x: NDArray[np.float64],
        counts: NDArray[np.float64],
        *,
        stay_feature_indices: tuple[int, ...] | None = None,
        length_scale_stay: float = 1.2,
        length_scale_dir: float = 1.5,
        k_neighbors: int = 32,
        alpha: float = 0.5,
    ) -> "KernelHurdleMultinomial":
        totals = np.sum(counts, axis=1, keepdims=True)
        totals = np.maximum(totals, 1.0)
        stay_mass = counts[:, [1]]
        move_mass = counts[:, [0]] + counts[:, [2]]
        stay_counts = np.concatenate([move_mass, stay_mass], axis=1)
        dir_counts = counts[:, [0, 2]]
        x_stay = x if stay_feature_indices is None else x[:, list(stay_feature_indices)]
        stay_model = KernelNeighborBaseline.fit(
            x_stay,
            stay_counts,
            length_scale=length_scale_stay,
            k_neighbors=k_neighbors,
            alpha=alpha,
        )
        dir_model = KernelNeighborBaseline.fit(
            x,
            dir_counts,
            length_scale=length_scale_dir,
            k_neighbors=k_neighbors,
            alpha=alpha,
        )
        return cls(
            name="kernel_hurdle",
            stay_model=stay_model,
            dir_model=dir_model,
            stay_feature_indices=stay_feature_indices,
        )

    def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        x_stay = (
            x
            if self.stay_feature_indices is None
            else x[:, list(self.stay_feature_indices)]
        )
        stay_bin = self.stay_model.predict_proba(x_stay)
        s = stay_bin[:, 1]
        direction = self.dir_model.predict_proba(x)
        q = direction[:, 1] / np.maximum(direction[:, 0] + direction[:, 1], 1e-12)
        p0 = s
        pp = (1.0 - s) * q
        pm = (1.0 - s) * (1.0 - q)
        return np.column_stack([pm, p0, pp])

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.stay_model.save(directory / "stay_kernel.npz")
        self.dir_model.save(directory / "dir_kernel.npz")
        meta = {
            "name": self.name,
            "stay_feature_indices": (
                list(self.stay_feature_indices)
                if self.stay_feature_indices is not None
                else None
            ),
        }
        (directory / "meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: Path) -> "KernelHurdleMultinomial":
        directory = Path(directory)
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        indices = meta.get("stay_feature_indices")
        return cls(
            name=str(meta.get("name", "kernel_hurdle")),
            stay_model=KernelNeighborBaseline.load(directory / "stay_kernel.npz"),
            dir_model=KernelNeighborBaseline.load(directory / "dir_kernel.npz"),
            stay_feature_indices=(
                tuple(int(i) for i in indices) if indices is not None else None
            ),
        )


@dataclass
class RegimeAwareKernelHurdle:
    """Kernel hurdle with separate stay experts for abstention vs active fields.

    Regime is inferred only from observation descriptors (no K / t / source):
    high antipodal_balance and low |z1| → abstention expert; else active.
    """

    name: str
    stay_abstain: KernelNeighborBaseline
    stay_active: KernelNeighborBaseline
    dir_model: KernelNeighborBaseline
    stay_feature_indices: tuple[int, ...]
    abs_z1_index: int
    antipodal_balance_index: int
    balance_min: float
    abs_z1_max: float

    @staticmethod
    def regime_mask(
        x: NDArray[np.float64],
        *,
        abs_z1_index: int,
        antipodal_balance_index: int,
        balance_min: float,
        abs_z1_max: float,
    ) -> NDArray[np.bool_]:
        """True = abstention regime."""
        return (x[:, antipodal_balance_index] >= balance_min) & (
            x[:, abs_z1_index] <= abs_z1_max
        )

    @classmethod
    def fit(
        cls,
        x: NDArray[np.float64],
        counts: NDArray[np.float64],
        *,
        stay_feature_indices: tuple[int, ...],
        abs_z1_index: int,
        antipodal_balance_index: int,
        balance_min: float = 0.95,
        abs_z1_max: float = 0.12,
        length_scale_stay_abstain: float = 1.2,
        length_scale_stay_active: float = 0.25,
        length_scale_dir: float = 1.5,
        k_neighbors_abstain: int = 24,
        k_neighbors_active: int = 6,
        alpha: float = 0.5,
        # Asymmetric Dirichlet on [move, stay] for the active expert only.
        alpha_active_stay: tuple[float, float] = (1.5, 0.15),
    ) -> "RegimeAwareKernelHurdle":
        totals = np.sum(counts, axis=1, keepdims=True)
        totals = np.maximum(totals, 1.0)
        stay_mass = counts[:, [1]]
        move_mass = counts[:, [0]] + counts[:, [2]]
        stay_counts = np.concatenate([move_mass, stay_mass], axis=1)
        dir_counts = counts[:, [0, 2]]
        x_stay = x[:, list(stay_feature_indices)]
        abstain = cls.regime_mask(
            x,
            abs_z1_index=abs_z1_index,
            antipodal_balance_index=antipodal_balance_index,
            balance_min=balance_min,
            abs_z1_max=abs_z1_max,
        )
        # Ensure both pools have enough mass; fall back by empirical stay.
        emp_stay = (counts[:, 1] / totals[:, 0]) >= 0.5
        if int(np.sum(abstain)) < 30:
            abstain = emp_stay
        active = ~abstain
        if int(np.sum(active)) < 30:
            active = ~emp_stay
            abstain = emp_stay
        stay_abstain = KernelNeighborBaseline.fit(
            x_stay[abstain],
            stay_counts[abstain],
            length_scale=length_scale_stay_abstain,
            k_neighbors=k_neighbors_abstain,
            alpha=alpha,
        )
        stay_active = KernelNeighborBaseline.fit(
            x_stay[active],
            stay_counts[active],
            length_scale=length_scale_stay_active,
            k_neighbors=k_neighbors_active,
            alpha=alpha_active_stay,
        )
        dir_model = KernelNeighborBaseline.fit(
            x,
            dir_counts,
            length_scale=length_scale_dir,
            k_neighbors=32,
            alpha=alpha,
        )
        return cls(
            name="regime_kernel_hurdle",
            stay_abstain=stay_abstain,
            stay_active=stay_active,
            dir_model=dir_model,
            stay_feature_indices=stay_feature_indices,
            abs_z1_index=abs_z1_index,
            antipodal_balance_index=antipodal_balance_index,
            balance_min=float(balance_min),
            abs_z1_max=float(abs_z1_max),
        )

    def predict_proba(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        x_stay = x[:, list(self.stay_feature_indices)]
        abstain = self.regime_mask(
            x,
            abs_z1_index=self.abs_z1_index,
            antipodal_balance_index=self.antipodal_balance_index,
            balance_min=self.balance_min,
            abs_z1_max=self.abs_z1_max,
        )
        s = np.empty(x.shape[0], dtype=np.float64)
        if np.any(abstain):
            s[abstain] = self.stay_abstain.predict_proba(x_stay[abstain])[:, 1]
        if np.any(~abstain):
            s[~abstain] = self.stay_active.predict_proba(x_stay[~abstain])[:, 1]
        direction = self.dir_model.predict_proba(x)
        q = direction[:, 1] / np.maximum(direction[:, 0] + direction[:, 1], 1e-12)
        p0 = s
        pp = (1.0 - s) * q
        pm = (1.0 - s) * (1.0 - q)
        return np.column_stack([pm, p0, pp])

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.stay_abstain.save(directory / "stay_abstain.npz")
        self.stay_active.save(directory / "stay_active.npz")
        self.dir_model.save(directory / "dir_kernel.npz")
        meta = {
            "name": self.name,
            "stay_feature_indices": list(self.stay_feature_indices),
            "abs_z1_index": self.abs_z1_index,
            "antipodal_balance_index": self.antipodal_balance_index,
            "balance_min": self.balance_min,
            "abs_z1_max": self.abs_z1_max,
        }
        (directory / "meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: Path) -> "RegimeAwareKernelHurdle":
        directory = Path(directory)
        meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        return cls(
            name=str(meta.get("name", "regime_kernel_hurdle")),
            stay_abstain=KernelNeighborBaseline.load(directory / "stay_abstain.npz"),
            stay_active=KernelNeighborBaseline.load(directory / "stay_active.npz"),
            dir_model=KernelNeighborBaseline.load(directory / "dir_kernel.npz"),
            stay_feature_indices=tuple(int(i) for i in meta["stay_feature_indices"]),
            abs_z1_index=int(meta["abs_z1_index"]),
            antipodal_balance_index=int(meta["antipodal_balance_index"]),
            balance_min=float(meta["balance_min"]),
            abs_z1_max=float(meta["abs_z1_max"]),
        )


def fit_candidate_models(
    x: NDArray[np.float64],
    counts: NDArray[np.float64],
    *,
    include_hurdle: bool = True,
    include_kernel_hurdle: bool = False,
    stay_feature_indices: tuple[int, ...] | None = None,
) -> dict[str, TrinomialModel]:
    """Fit the Stage-3A candidate set plus the global empirical baseline."""
    models: dict[str, TrinomialModel] = {
        "global_empirical": GlobalEmpiricalBaseline.fit(counts),
        "kernel_nn": KernelNeighborBaseline.fit(x, counts),
        "multinomial_logistic_l2": MultinomialLogisticL2.fit(x, counts),
        "softmax_stump_boost": SoftmaxStumpBoost.fit(x, counts),
    }
    if include_hurdle:
        models["hurdle_multinomial"] = HurdleMultinomial.fit(x, counts)
    if include_kernel_hurdle:
        models["kernel_hurdle"] = KernelHurdleMultinomial.fit(
            x, counts, stay_feature_indices=stay_feature_indices
        )
    return models

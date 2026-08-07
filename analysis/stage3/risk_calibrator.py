"""Calibrated predictive risk R(x) from grouped OOF residuals."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


FEATURE_KEYS = (
    "d_shape",
    "d_native",
    "local_density",
    "u_ensemble",
    "d_orientation",
    "policy_a",
)


@dataclass
class RiskCalibrator:
    """Ridge linear model on log1p features predicting TV residual."""

    weights: NDArray[np.float64]  # including bias at index 0
    feature_keys: tuple[str, ...]
    target: str = "e_tv"
    train_spearman: float = float("nan")
    bin_calibration: list[dict] | None = None

    @staticmethod
    def _design(rows: list[dict], keys: tuple[str, ...]) -> NDArray[np.float64]:
        cols = []
        for key in keys:
            values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
            if key == "local_density":
                cols.append(np.log1p(values))
            else:
                cols.append(np.log1p(np.maximum(values, 0.0)))
        matrix = np.column_stack(cols)
        return np.concatenate([np.ones((matrix.shape[0], 1)), matrix], axis=1)

    @classmethod
    def fit(
        cls,
        rows: list[dict],
        *,
        target: str = "e_tv",
        l2: float = 1e-2,
        feature_keys: tuple[str, ...] = FEATURE_KEYS,
    ) -> "RiskCalibrator":
        if len(rows) < 20:
            raise ValueError("need at least 20 OOF rows to fit risk calibrator")
        # Exclude unsupported-peer rows from shape risk fit (hard gate elsewhere).
        usable = [row for row in rows if int(row.get("g_peer", 0)) == 0]
        if len(usable) < 20:
            usable = rows
        design = cls._design(usable, feature_keys)
        y = np.asarray([float(row[target]) for row in usable], dtype=np.float64)
        d = design.shape[1]
        xtx = design.T @ design
        xtx[1:, 1:] += l2 * np.eye(d - 1)
        weights = np.linalg.solve(xtx, design.T @ y)
        pred = design @ weights
        calibrator = cls(
            weights=weights,
            feature_keys=feature_keys,
            target=target,
            train_spearman=_spearman(pred, y),
        )
        calibrator.bin_calibration = _bin_means(pred, y, n_bins=8)
        return calibrator

    def predict_rows(self, rows: list[dict]) -> NDArray[np.float64]:
        design = self._design(rows, self.feature_keys)
        return design @ self.weights

    def predict_arrays(
        self,
        *,
        d_shape: NDArray[np.float64],
        d_native: NDArray[np.float64],
        local_density: NDArray[np.float64],
        u_ensemble: NDArray[np.float64],
        d_orientation: NDArray[np.float64] | None = None,
        policy_a: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        n = d_shape.shape[0]
        if d_orientation is None:
            d_orientation = np.zeros(n)
        if policy_a is None:
            policy_a = np.zeros(n)
        payload = {
            "d_shape": d_shape,
            "d_native": d_native,
            "local_density": local_density,
            "u_ensemble": u_ensemble,
            "d_orientation": d_orientation,
            "policy_a": policy_a,
        }
        cols = []
        for key in self.feature_keys:
            values = np.asarray(payload[key], dtype=np.float64)
            if key == "local_density":
                cols.append(np.log1p(values))
            else:
                cols.append(np.log1p(np.maximum(values, 0.0)))
        design = np.column_stack([np.ones(n), *cols])
        return design @ self.weights

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "weights": self.weights.tolist(),
            "feature_keys": list(self.feature_keys),
            "target": self.target,
            "train_spearman": self.train_spearman,
            "bin_calibration": self.bin_calibration,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "RiskCalibrator":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            weights=np.asarray(payload["weights"], dtype=np.float64),
            feature_keys=tuple(payload["feature_keys"]),
            target=payload.get("target", "e_tv"),
            train_spearman=float(payload.get("train_spearman", float("nan"))),
            bin_calibration=payload.get("bin_calibration"),
        )


def _spearman(x: NDArray[np.float64], y: NDArray[np.float64]) -> float:
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = np.sqrt(np.sum(rx * rx) * np.sum(ry * ry))
    if denom <= 0:
        return float("nan")
    return float(np.sum(rx * ry) / denom)


def _bin_means(
    score: NDArray[np.float64],
    error: NDArray[np.float64],
    *,
    n_bins: int,
) -> list[dict]:
    order = np.argsort(score)
    s = score[order]
    e = error[order]
    edges = np.linspace(0, s.size, n_bins + 1, dtype=int)
    rows = []
    for left, right in zip(edges[:-1], edges[1:]):
        if right <= left:
            continue
        rows.append(
            {
                "mean_R": float(np.mean(s[left:right])),
                "mean_error": float(np.mean(e[left:right])),
                "n": int(right - left),
            }
        )
    return rows


def monotone_ok(bin_calibration: list[dict] | None, *, tol: float = 0.02) -> bool:
    """True if consecutive bin mean errors are non-decreasing up to tol dips."""
    if not bin_calibration or len(bin_calibration) < 3:
        return False
    errors = [row["mean_error"] for row in bin_calibration]
    violations = 0
    for left, right in zip(errors, errors[1:]):
        if right + tol < left:
            violations += 1
    return violations <= 1

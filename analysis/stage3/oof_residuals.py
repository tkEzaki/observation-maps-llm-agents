"""Grouped out-of-fold residuals for predictive-risk calibration."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from analysis.stage3.applicability_domain import ApplicabilityDomain
from analysis.stage3.models import (
    KernelHurdleMultinomial,
    KernelNeighborBaseline,
    MultinomialLogisticL2,
    counts_to_probs,
)
from analysis.stage3.validation import unique_groups
from circlemap.field_features import COMMON_FEATURE_NAMES


def tv_row(pred: NDArray[np.float64], empir: NDArray[np.float64]) -> NDArray[np.float64]:
    return 0.5 * np.sum(np.abs(pred - empir), axis=1)


def nll_row(pred: NDArray[np.float64], counts: NDArray[np.float64]) -> NDArray[np.float64]:
    totals = np.maximum(np.sum(counts, axis=1), 1.0)
    return -np.sum(counts * np.log(np.clip(pred, 1e-12, 1.0)), axis=1) / totals


def ensemble_disagreement(
    preds: list[NDArray[np.float64]],
) -> NDArray[np.float64]:
    if len(preds) < 2:
        return np.zeros(preds[0].shape[0], dtype=np.float64)
    acc = np.zeros(preds[0].shape[0], dtype=np.float64)
    pairs = 0
    for i in range(len(preds)):
        for j in range(i + 1, len(preds)):
            acc += 0.5 * np.sum(np.abs(preds[i] - preds[j]), axis=1)
            pairs += 1
    return acc / max(pairs, 1)


def default_stay_indices() -> tuple[int, ...]:
    names = COMMON_FEATURE_NAMES
    wanted = ("abs_z1", "antipodal_balance", "sparsity", "peer_count")
    return tuple(names.index(name) for name in wanted)


def build_oof_residual_table(
    x: NDArray[np.float64],
    counts: NDArray[np.float64],
    group_maps: dict[str, list[str]],
    *,
    representation: str,
    unresolved: NDArray[np.bool_] | None = None,
    schemes: tuple[str, ...] = (
        "leave_one_profile_out",
        "leave_one_eps_level_out",
        "sparse_realization_holdout",
        "offset_group_holdout",
        "acquisition_block_holdout",
    ),
) -> list[dict]:
    """Fit kernel ensemble OOF; attach AD distances from the training fold."""
    if unresolved is None:
        unresolved = np.zeros(x.shape[0], dtype=bool)
    rows: list[dict] = []
    stay_idx = default_stay_indices()

    for scheme in schemes:
        groups = group_maps[scheme]
        if scheme in {"leave_one_eps_level_out", "sparse_realization_holdout"}:
            active = [i for i, g in enumerate(groups) if g != "na"]
            if len(active) < 20 or len(unique_groups([groups[i] for i in active])) < 2:
                continue
            indices = np.asarray(active, dtype=np.int64)
        else:
            indices = np.arange(x.shape[0], dtype=np.int64)
            if len(unique_groups([groups[i] for i in indices])) < 2:
                continue

        x_s = x[indices]
        c_s = counts[indices]
        u_s = unresolved[indices]
        g_s = [groups[int(i)] for i in indices]

        for held in unique_groups(g_s):
            test_local = np.asarray([g == held for g in g_s], dtype=bool)
            train_local = ~test_local
            if not np.any(test_local) or not np.any(train_local):
                continue
            x_tr, c_tr = x_s[train_local], c_s[train_local]
            x_te, c_te = x_s[test_local], c_s[test_local]
            u_te = u_s[test_local]

            kernel = KernelNeighborBaseline.fit(x_tr, c_tr)
            logistic = MultinomialLogisticL2.fit(x_tr, c_tr, steps=120)
            khurdle = KernelHurdleMultinomial.fit(
                x_tr, c_tr, stay_feature_indices=stay_idx
            )
            pred_k = kernel.predict_proba(x_te)
            pred_s = logistic.predict_proba(x_te)
            pred_h = khurdle.predict_proba(x_te)
            disagreement = ensemble_disagreement([pred_k, pred_s, pred_h])

            empir = counts_to_probs(c_te, alpha=0.0)
            tv = tv_row(pred_k, empir)
            nll = nll_row(pred_k, c_te)

            ad = ApplicabilityDomain.fit(x_tr, representation=representation)
            scores = ad.score(x_te, unresolved_near_zero=u_te)

            global_idx = indices[test_local]
            for local in range(x_te.shape[0]):
                rows.append(
                    {
                        "representation": representation,
                        "scheme": scheme,
                        "fold": held,
                        "row_index": int(global_idx[local]),
                        "e_tv": float(tv[local]),
                        "e_nll": float(nll[local]),
                        "d_shape": float(scores["d_shape"][local]),
                        "d_native": float(scores["d_native"][local]),
                        "d_orientation": float(scores["d_orientation"][local]),
                        "local_density": float(scores["local_density"][local]),
                        "u_ensemble": float(disagreement[local]),
                        "g_peer": int(bool(scores["g_peer"][local])),
                        "policy_a": float(scores["policy_a"][local]),
                        "peer_count": float(scores["peer_count"][local]),
                    }
                )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

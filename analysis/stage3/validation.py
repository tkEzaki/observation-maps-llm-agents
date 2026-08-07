"""Grouped validation metrics for Stage 3A (no response-level random splits)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from circlemap.field_features import COMMON_FEATURE_NAMES

from .models import (
    GlobalEmpiricalBaseline,
    HurdleMultinomial,
    KernelHurdleMultinomial,
    KernelNeighborBaseline,
    MultinomialLogisticL2,
    RegimeAwareKernelHurdle,
    SoftmaxStumpBoost,
    TrinomialModel,
    counts_to_probs,
)


def _fit_one(
    name: str,
    x: NDArray[np.float64],
    counts: NDArray[np.float64],
    *,
    stay_feature_indices: tuple[int, ...] | None = None,
) -> TrinomialModel:
    if name == "global_empirical":
        return GlobalEmpiricalBaseline.fit(counts)
    if name == "kernel_nn":
        return KernelNeighborBaseline.fit(x, counts)
    if name == "multinomial_logistic_l2":
        return MultinomialLogisticL2.fit(x, counts)
    if name == "softmax_stump_boost":
        return SoftmaxStumpBoost.fit(x, counts)
    if name == "hurdle_multinomial":
        return HurdleMultinomial.fit(x, counts)
    if name == "kernel_hurdle":
        return KernelHurdleMultinomial.fit(
            x, counts, stay_feature_indices=stay_feature_indices
        )
    if name == "regime_kernel_hurdle":
        if stay_feature_indices is None:
            raise ValueError("regime_kernel_hurdle requires stay_feature_indices")
        return RegimeAwareKernelHurdle.fit(
            x,
            counts,
            stay_feature_indices=stay_feature_indices,
            abs_z1_index=COMMON_FEATURE_NAMES.index("abs_z1"),
            antipodal_balance_index=COMMON_FEATURE_NAMES.index("antipodal_balance"),
        )
    raise KeyError(name)


def multinomial_log_loss(
    probs: NDArray[np.float64],
    counts: NDArray[np.float64],
) -> float:
    clipped = np.clip(probs, 1e-12, 1.0)
    totals = np.sum(counts, axis=1)
    if float(np.sum(totals)) <= 0:
        return float("nan")
    # Mean per-trial NLL.
    nll = -np.sum(counts * np.log(clipped)) / np.sum(totals)
    return float(nll)


def js_divergence(
    p: NDArray[np.float64],
    q: NDArray[np.float64],
) -> NDArray[np.float64]:
    p = np.clip(p, 1e-12, 1.0)
    q = np.clip(q, 1e-12, 1.0)
    p = p / np.sum(p, axis=-1, keepdims=True)
    q = q / np.sum(q, axis=-1, keepdims=True)
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log(p / m), axis=-1)
    kl_qm = np.sum(q * np.log(q / m), axis=-1)
    return 0.5 * (kl_pm + kl_qm)


def tv_distance(p: NDArray[np.float64], q: NDArray[np.float64]) -> NDArray[np.float64]:
    return 0.5 * np.sum(np.abs(p - q), axis=-1)


def brier_score(probs: NDArray[np.float64], counts: NDArray[np.float64]) -> float:
    empir = counts_to_probs(counts, alpha=0.0)
    return float(np.mean(np.sum((probs - empir) ** 2, axis=-1)))


def action_metrics(
    probs: NDArray[np.float64],
    counts: NDArray[np.float64],
) -> dict[str, float]:
    empir = counts_to_probs(counts, alpha=0.0)
    g_hat = probs[:, 2] - probs[:, 0]
    g = empir[:, 2] - empir[:, 0]
    a_hat = probs[:, 0] + probs[:, 2]
    a = empir[:, 0] + empir[:, 2]
    return {
        "g_rmse": float(np.sqrt(np.mean((g_hat - g) ** 2))),
        "activity_rmse": float(np.sqrt(np.mean((a_hat - a) ** 2))),
        "mean_tv": float(np.mean(tv_distance(probs, empir))),
        "mean_js": float(np.mean(js_divergence(probs, empir))),
        "log_loss": multinomial_log_loss(probs, counts),
        "brier": brier_score(probs, counts),
    }


@dataclass
class GroupSplitResult:
    scheme: str
    model: str
    fold: str
    n_train: int
    n_test: int
    metrics: dict[str, float]


def unique_groups(labels: list[str]) -> list[str]:
    seen: list[str] = []
    for label in labels:
        if label not in seen:
            seen.append(label)
    return seen


def leave_one_group_out_predictions(
    model_factory,
    x: NDArray[np.float64],
    counts: NDArray[np.float64],
    groups: list[str],
) -> tuple[NDArray[np.float64], list[GroupSplitResult]]:
    """Fit per fold; return stacked OOF probabilities for the named model factory.

    ``model_factory`` maps (x_train, counts_train) -> model with predict_proba.
    """
    n = x.shape[0]
    oof = np.full((n, 3), np.nan, dtype=np.float64)
    results: list[GroupSplitResult] = []
    for held in unique_groups(groups):
        test_mask = np.asarray([g == held for g in groups], dtype=bool)
        train_mask = ~test_mask
        if not np.any(test_mask) or not np.any(train_mask):
            continue
        model = model_factory(x[train_mask], counts[train_mask])
        pred = model.predict_proba(x[test_mask])
        oof[test_mask] = pred
        metrics = action_metrics(pred, counts[test_mask])
        results.append(
            GroupSplitResult(
                scheme="loo_group",
                model=getattr(model, "name", "model"),
                fold=str(held),
                n_train=int(np.sum(train_mask)),
                n_test=int(np.sum(test_mask)),
                metrics=metrics,
            )
        )
    return oof, results


def evaluate_grouped_schemes(
    x: NDArray[np.float64],
    counts: NDArray[np.float64],
    group_maps: dict[str, list[str]],
    *,
    model_names: tuple[str, ...] = (
        "global_empirical",
        "kernel_nn",
        "multinomial_logistic_l2",
        "softmax_stump_boost",
    ),
    stay_feature_indices: tuple[int, ...] | None = None,
) -> dict[str, object]:
    """Run all grouped schemes × candidate models; summarize mean metrics."""
    summary: dict[str, object] = {"folds": [], "by_scheme_model": {}}
    for scheme, groups in group_maps.items():
        if scheme in {"sparse_realization_holdout", "leave_one_eps_level_out"}:
            # Only evaluate on rows that carry the grouping label.
            active = [i for i, g in enumerate(groups) if g != "na"]
            if len(active) < 10 or len(unique_groups([groups[i] for i in active])) < 2:
                continue
            x_s = x[active]
            c_s = counts[active]
            g_s = [groups[i] for i in active]
        else:
            x_s, c_s, g_s = x, counts, groups
            if len(unique_groups(g_s)) < 2:
                continue

        for model_name in model_names:

            def factory(xt, ct, _name=model_name):
                return _fit_one(
                    _name, xt, ct, stay_feature_indices=stay_feature_indices
                )

            _oof, fold_results = leave_one_group_out_predictions(
                factory, x_s, c_s, g_s
            )
            for item in fold_results:
                item.scheme = scheme
                summary["folds"].append(
                    {
                        "scheme": item.scheme,
                        "model": item.model,
                        "fold": item.fold,
                        "n_train": item.n_train,
                        "n_test": item.n_test,
                        **item.metrics,
                    }
                )
            if fold_results:
                keys = fold_results[0].metrics.keys()
                means = {
                    key: float(np.nanmean([f.metrics[key] for f in fold_results]))
                    for key in keys
                }
                summary["by_scheme_model"][f"{scheme}::{model_name}"] = means
    return summary


def moments_abstention_activation_check(
    rows: list,
    probs: NDArray[np.float64],
    *,
    representation: str = "moments_m1_m3",
) -> dict[str, float]:
    """Require high stay at exact balance and low stay under small imbalance."""
    mask_rep = [row.representation == representation for row in rows]
    if not any(mask_rep):
        return {}
    indices = [i for i, flag in enumerate(mask_rep) if flag]
    stay = probs[indices, 1]
    eps_levels = [rows[i].epsilon_level for i in indices]
    profiles = [rows[i].profile for i in indices]

    def mean_stay(predicate) -> float:
        selected = [stay[j] for j, i in enumerate(indices) if predicate(rows[i])]
        return float(np.mean(selected)) if selected else float("nan")

    exact = mean_stay(
        lambda row: row.epsilon_level == "+0.000"
        or "antipodal_equal" in row.profile
        or "_8_8_" in row.profile
    )
    # Prefer dense |ε|=0.02 when available.
    small = mean_stay(
        lambda row: row.epsilon_level in {"+0.020", "-0.020"}
        or "_9_7_" in row.profile
    )
    return {
        "moments_stay_exact_balance": exact,
        "moments_stay_small_imbalance": small,
        "moments_stay_gap": (
            float(exact - small)
            if np.isfinite(exact) and np.isfinite(small)
            else float("nan")
        ),
    }

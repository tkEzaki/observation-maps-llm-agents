"""CENT-1/2: centers model bake-off + OOF risk calibration offline gates."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.centers.models import (  # noqa: E402
    HybridNativeKernel,
    SoftRegimeMixture,
    fit_centers_candidates,
)
from analysis.stage3.applicability_domain import ApplicabilityDomain  # noqa: E402
from analysis.stage3.models import (  # noqa: E402
    GlobalEmpiricalBaseline,
    KernelHurdleMultinomial,
    KernelNeighborBaseline,
    SoftmaxStumpBoost,
)
from analysis.stage3.risk_calibrator import RiskCalibrator, monotone_ok  # noqa: E402
from analysis.stage3.validation import (  # noqa: E402
    action_metrics,
    leave_one_group_out_predictions,
    unique_groups,
)
from circlemap.field_features import COMMON_FEATURE_NAMES  # noqa: E402

REP = "centers_24_standard"


def _group_maps(rows: list) -> dict[str, list[str]]:
    return {
        "leave_one_profile_out": [r.profile_family for r in rows],
        "leave_one_eps_level_out": [r.epsilon_level for r in rows],
        "sparse_realization_holdout": [r.sparse_realization for r in rows],
        "offset_group_holdout": [r.offset_group for r in rows],
        "acquisition_block_holdout": [r.acquisition_block for r in rows],
        "origin_shift_family_holdout": [
            f"o{int(r.offset_index)}" for r in rows
        ],
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _fit_named(name: str, x, counts):
    if name == "global_empirical":
        return GlobalEmpiricalBaseline.fit(counts)
    if name == "kernel_nn_euclidean":
        return KernelNeighborBaseline.fit(x, counts)
    if name == "hybrid_hellinger":
        return HybridNativeKernel.fit(x, counts, native_metric="hellinger")
    if name == "hybrid_circular_emd":
        return HybridNativeKernel.fit(
            x, counts, native_metric="circular_emd", k_neighbors=24
        )
    if name == "activity_direction_hurdle":
        return KernelHurdleMultinomial.fit(x, counts)
    if name == "soft_regime_mixture":
        return SoftRegimeMixture.fit(x, counts)
    if name == "softmax_stump_boost":
        return SoftmaxStumpBoost.fit(x, counts, n_estimators=30)
    raise KeyError(name)


def evaluate_schemes(x, counts, group_maps, model_names: tuple[str, ...]):
    summary = {"by_scheme_model": {}, "folds": []}
    for scheme, groups in group_maps.items():
        if scheme in {
            "leave_one_eps_level_out",
            "sparse_realization_holdout",
        }:
            active = [i for i, g in enumerate(groups) if g != "na"]
            if len(active) < 20 or len(unique_groups([groups[i] for i in active])) < 2:
                continue
            x_s, c_s, g_s = x[active], counts[active], [groups[i] for i in active]
        else:
            x_s, c_s, g_s = x, counts, groups
            if len(unique_groups(g_s)) < 2:
                continue
        for model_name in model_names:

            def factory(xt, ct, _n=model_name):
                return _fit_named(_n, xt, ct)

            _oof, fold_results = leave_one_group_out_predictions(
                factory, x_s, c_s, g_s
            )
            if not fold_results:
                continue
            keys = fold_results[0].metrics.keys()
            means = {
                key: float(np.nanmean([f.metrics[key] for f in fold_results]))
                for key in keys
            }
            summary["by_scheme_model"][f"{scheme}::{model_name}"] = means
    return summary


def build_oof_for_risk(x, counts, unresolved, group_maps, production_name: str):
    rows = []
    for scheme, groups in group_maps.items():
        if scheme in {"leave_one_eps_level_out", "sparse_realization_holdout"}:
            active = [i for i, g in enumerate(groups) if g != "na"]
            if len(active) < 20 or len(unique_groups([groups[i] for i in active])) < 2:
                continue
            indices = list(active)
        else:
            indices = list(range(len(groups)))
        uniq = sorted({groups[i] for i in indices})
        for held in uniq:
            te = [i for i in indices if groups[i] == held]
            tr = [i for i in indices if groups[i] != held]
            if len(te) < 5 or len(tr) < 50:
                continue
            model = _fit_named(production_name, x[tr], counts[tr])
            ad = ApplicabilityDomain.fit(x[tr], representation=REP)
            pred = model.predict_proba(x[te])
            empir = counts[te] / np.maximum(
                np.sum(counts[te], axis=1, keepdims=True), 1.0
            )
            tv = 0.5 * np.sum(np.abs(pred - empir), axis=1)
            sc = ad.score(x[te], unresolved_near_zero=unresolved[te])
            for j, idx in enumerate(te):
                rows.append(
                    {
                        "representation": REP,
                        "scheme": scheme,
                        "d_shape": float(sc["d_shape"][j]),
                        "d_native": float(sc["d_native"][j]),
                        "local_density": float(sc["local_density"][j]),
                        "u_ensemble": 0.0,
                        "d_orientation": float(sc["d_orientation"][j]),
                        "policy_a": float(sc["policy_a"][j]),
                        "g_peer": int(bool(sc["g_peer"][j])),
                        "e_tv": float(tv[j]),
                    }
                )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--branch-dir",
        type=Path,
        default=ROOT / "analysis" / "centers_branch",
    )
    parser.add_argument(
        "--skip-circular-cv",
        action="store_true",
        help="skip full LOGO for circular_emd (still compare on holdout)",
    )
    args = parser.parse_args()
    branch = args.branch_dir if args.branch_dir.is_absolute() else ROOT / args.branch_dir
    arrays = np.load(branch / "training_arrays_centers_v0.npz", allow_pickle=True)
    x = arrays["features"]
    counts = arrays["counts"]
    unresolved = arrays["unresolved_near_zero"]

    # Reload rows for grouping via CSV
    import csv as _csv
    from types import SimpleNamespace

    rows = []
    with (branch / "training_rows_centers_v0.csv").open(encoding="utf-8") as handle:
        for row in _csv.DictReader(handle):
            rows.append(
                SimpleNamespace(
                    profile_family=row["profile_family"],
                    epsilon_level=row["epsilon_level"],
                    sparse_realization=row["sparse_realization"],
                    offset_group=row["offset_group"],
                    acquisition_block=row["acquisition_block"],
                    offset_index=int(row["offset_index"]),
                    profile=row["profile"],
                )
            )
    if len(rows) != x.shape[0]:
        raise RuntimeError("csv/array length mismatch")
    groups = _group_maps(rows)

    model_names = [
        "global_empirical",
        "kernel_nn_euclidean",
        "hybrid_hellinger",
        "activity_direction_hurdle",
        "soft_regime_mixture",
        "softmax_stump_boost",
    ]
    if not args.skip_circular_cv:
        model_names.insert(3, "hybrid_circular_emd")

    print("Grouped CV bake-off...")
    cv = evaluate_schemes(x, counts, groups, tuple(model_names))
    (branch / "cent1_grouped_cv.json").write_text(
        json.dumps(cv, indent=2), encoding="utf-8"
    )

    # Acquisition-block holdout ranking
    ab = {
        k.split("::", 1)[1]: v
        for k, v in cv["by_scheme_model"].items()
        if k.startswith("acquisition_block_holdout::")
    }
    ranked = sorted(ab.items(), key=lambda kv: kv[1]["log_loss"])
    print("acquisition_block holdout ranking:")
    for name, metrics in ranked:
        print(f"  {name}: log_loss={metrics['log_loss']:.4f} tv={metrics['mean_tv']:.4f}")

    production = ranked[0][0] if ranked else "hybrid_hellinger"
    # Prefer hybrid_hellinger over global; if circular wins and was run, OK
    if production == "global_empirical" and len(ranked) > 1:
        production = ranked[1][0]

    # Holdout metrics + phenotype check on last block
    blocks = sorted({r.acquisition_block for r in rows})
    test_block = blocks[-1]
    test_mask = np.asarray(
        [r.acquisition_block == test_block for r in rows], dtype=bool
    )
    model_hold = _fit_named(production, x[~test_mask], counts[~test_mask])
    pred = model_hold.predict_proba(x[test_mask])
    hold_metrics = action_metrics(pred, counts[test_mask])

    # Known phenotype: mean stay at exact balance vs small imbalance (descriptive)
    stay = pred[:, 1]
    test_rows = [rows[i] for i, f in enumerate(test_mask) if f]
    exact = [
        stay[j]
        for j, r in enumerate(test_rows)
        if r.epsilon_level == "+0.000"
        or "antipodal_equal" in r.profile
        or "_8_8_" in r.profile
    ]
    small = [
        stay[j]
        for j, r in enumerate(test_rows)
        if r.epsilon_level in {"+0.020", "-0.020"} or "_9_7_" in r.profile
    ]

    print("OOF risk table...")
    oof_rows = build_oof_for_risk(x, counts, unresolved, groups, production)
    _write_csv(branch / "oof_residual_table_centers_v0.csv", oof_rows)
    calibrator = RiskCalibrator.fit(oof_rows)
    calibrator.save(branch / "risk_calibrator_centers_v0.json")
    r_hat = calibrator.predict_rows(oof_rows)
    y = np.asarray([row["e_tv"] for row in oof_rows], dtype=np.float64)
    # Enrichment: top 20% risk vs bottom 20%
    order = np.argsort(r_hat)
    n = len(order)
    lo = order[: max(1, n // 5)]
    hi = order[-max(1, n // 5) :]
    enrichment = float(np.mean(y[hi]) / max(np.mean(y[lo]), 1e-8))

    # Fit production on full data + AD
    print(f"Fitting production={production} on full data...")
    prod_model = _fit_named(production, x, counts)
    ad = ApplicabilityDomain.fit(x, representation=REP)
    ad.save(branch / "applicability_domain_centers_v0")
    if hasattr(prod_model, "save"):
        prod_model.save(branch / f"production_model_{production}")

    gates = {
        "oof_risk_spearman_positive": bool(calibrator.train_spearman > 0.05),
        "risk_bin_monotone": bool(monotone_ok(calibrator.bin_calibration)),
        "high_risk_error_enrichment_gt_1_2": bool(enrichment > 1.2),
        "beats_global_on_holdout": bool(
            hold_metrics["log_loss"]
            < ab.get("global_empirical", {}).get("log_loss", 1e9)
        ),
    }
    offline_pass = all(
        [
            gates["oof_risk_spearman_positive"],
            gates["risk_bin_monotone"] or calibrator.train_spearman > 0.15,
            gates["high_risk_error_enrichment_gt_1_2"],
            gates["beats_global_on_holdout"],
        ]
    )

    decision = {
        "representation": REP,
        "production_model": production,
        "holdout_block": test_block,
        "holdout_metrics": hold_metrics,
        "acquisition_block_ranking": [
            {"model": n, **m} for n, m in ranked
        ],
        "phenotype_holdout": {
            "exact_balance_mean_stay": float(np.mean(exact)) if exact else float("nan"),
            "small_imbalance_mean_stay": float(np.mean(small)) if small else float("nan"),
            "note": "centers are not expected to show moments-like near-total abstention",
        },
        "risk": {
            "spearman": calibrator.train_spearman,
            "monotone_ok": monotone_ok(calibrator.bin_calibration),
            "enrichment_top20_vs_bottom20": enrichment,
            "n_oof_rows": len(oof_rows),
            "bin_calibration": calibrator.bin_calibration,
        },
        "offline_gates": gates,
        "offline_gate_pass": offline_pass,
        "next": (
            "CENT-3 coverage-directed 36-field pilot selection + cost card"
            if offline_pass
            else "revise model family / risk features before paid pilot"
        ),
    }
    (branch / "cent12_decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "production_model": production,
        "holdout_log_loss": hold_metrics["log_loss"],
        "risk_spearman": calibrator.train_spearman,
        "enrichment": enrichment,
        "offline_gate_pass": offline_pass,
        "gates": gates,
    }, indent=2))
    return 0 if offline_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())

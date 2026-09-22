"""Post-pilot refit: include pilot rows, recalibrate R, recompute Q_risk.

Offline only. Writes post-refit artifacts; does not authorize Stage C.
"""

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

from analysis.stage3.applicability_domain import ApplicabilityDomain  # noqa: E402
from analysis.stage3.dataset import (  # noqa: E402
    build_unified_dataset,
    post_pilot_source_runs,
    write_dataset_csv,
)
from analysis.stage3.models import (  # noqa: E402
    KernelHurdleMultinomial,
    KernelNeighborBaseline,
    SoftmaxStumpBoost,
)
from analysis.stage3.oof_residuals import (  # noqa: E402
    build_oof_residual_table,
    default_stay_indices,
    write_csv,
)
from analysis.stage3.rebuild_applicability import risk_diagnose_regime  # noqa: E402
from analysis.stage3.risk_calibrator import RiskCalibrator, monotone_ok  # noqa: E402
from analysis.stage3.validation import (  # noqa: E402
    action_metrics,
    evaluate_grouped_schemes,
    moments_abstention_activation_check,
)

REPS = (
    "intervals_24_decimal6",
    "moments_m1_m3",
    "centers_24_standard",
)


def _group_maps(rows: list) -> dict[str, list[str]]:
    return {
        "leave_one_profile_out": [row.profile_family for row in rows],
        "leave_one_eps_level_out": [row.epsilon_level for row in rows],
        "sparse_realization_holdout": [row.sparse_realization for row in rows],
        "offset_group_holdout": [row.offset_group for row in rows],
        "acquisition_block_holdout": [row.acquisition_block for row in rows],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "analysis" / "stage3a_artifacts",
    )
    parser.add_argument("--qrisk-steps", type=int, default=20)
    args = parser.parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Building post-pilot unified dataset...")
    sources = post_pilot_source_runs(ROOT)
    for family, runs in sources.items():
        print(f"  {family}: {len(runs)} runs")
    all_rows, packed = build_unified_dataset(ROOT, source_runs=sources)
    write_dataset_csv(all_rows, out_dir / "training_rows_post_pilot.csv")
    stay_idx = default_stay_indices()

    oof_all: list[dict] = []
    summary: dict = {
        "n_rows_total": len(all_rows),
        "n_rows_by_source": {},
        "representations": {},
    }
    for row in all_rows:
        summary["n_rows_by_source"][row.source_family] = (
            summary["n_rows_by_source"].get(row.source_family, 0) + 1
        )

    model_compare: list[dict] = []
    qrisk_rows: list[dict] = []

    for representation in REPS:
        print(f"\n=== {representation} ===")
        rep_rows = [r for r in all_rows if r.representation == representation]
        x = packed[representation]["features"]
        counts = packed[representation]["counts"]
        unresolved = packed[representation]["unresolved_near_zero"]

        print("  grouped OOF + risk recalibration...")
        oof_rows = build_oof_residual_table(
            x,
            counts,
            _group_maps(rep_rows),
            representation=representation,
            unresolved=unresolved,
        )
        oof_all.extend(oof_rows)
        calibrator = RiskCalibrator.fit(oof_rows)
        calibrator.save(out_dir / f"risk_calibrator_post_{representation}.json")
        spearman = calibrator.train_spearman
        mono = monotone_ok(calibrator.bin_calibration)
        print(f"  risk Spearman={spearman:.3f} monotone={mono}")

        print("  grouped CV (incl. kernel_hurdle)...")
        cv_summary = evaluate_grouped_schemes(
            x,
            counts,
            _group_maps(rep_rows),
            model_names=(
                "global_empirical",
                "kernel_nn",
                "kernel_hurdle",
                "softmax_stump_boost",
                "multinomial_logistic_l2",
            ),
            stay_feature_indices=stay_idx,
        )

        # Acquisition-block holdout metrics + moments stay
        blocks = sorted({r.acquisition_block for r in rep_rows})
        # Prefer holding out a pilot block if present.
        test_block = next(
            (b for b in blocks if "block_2" in b or b == "block_2"),
            blocks[-1],
        )
        test_mask = np.asarray(
            [r.acquisition_block == test_block for r in rep_rows], dtype=bool
        )
        train_mask = ~test_mask
        fitted = {
            "kernel_nn": KernelNeighborBaseline.fit(x[train_mask], counts[train_mask]),
            "kernel_hurdle": KernelHurdleMultinomial.fit(
                x[train_mask],
                counts[train_mask],
                stay_feature_indices=stay_idx,
            ),
            "softmax_stump_boost": SoftmaxStumpBoost.fit(
                x[train_mask], counts[train_mask], n_estimators=25
            ),
        }
        holdout = {}
        for name, model in fitted.items():
            pred = model.predict_proba(x[test_mask])
            metrics = action_metrics(pred, counts[test_mask])
            stay = moments_abstention_activation_check(
                [rep_rows[i] for i, flag in enumerate(test_mask) if flag],
                pred,
                representation=representation,
            )
            holdout[name] = {**metrics, **stay}
            model_compare.append(
                {
                    "representation": representation,
                    "model": name,
                    "holdout_block": test_block,
                    **metrics,
                    **stay,
                }
            )
            print(
                f"  {name}: log_loss={metrics['log_loss']:.3f} "
                f"stay_gap={stay.get('moments_stay_gap', float('nan'))}"
            )

        # Full fit for Q_risk
        ad = ApplicabilityDomain.fit(x, representation=representation)
        full_models = {
            "kernel_nn": KernelNeighborBaseline.fit(x, counts),
            "kernel_hurdle": KernelHurdleMultinomial.fit(
                x, counts, stay_feature_indices=stay_idx
            ),
            "softmax_stump_boost": SoftmaxStumpBoost.fit(x, counts, n_estimators=25),
        }
        r_oof = calibrator.predict_rows(oof_rows)
        risk_threshold = float(np.quantile(r_oof, 0.8))

        for n_agents in (9, 17):
            for regime in ("t0_only", "k0_drift", "closed_loop"):
                row = risk_diagnose_regime(
                    ad=ad,
                    calibrator=calibrator,
                    models=full_models,
                    representation=representation,
                    n_agents=n_agents,
                    regime=regime,
                    n_steps=args.qrisk_steps,
                    seed=200 + n_agents,
                    risk_threshold=risk_threshold,
                )
                qrisk_rows.append(row)
                print(
                    f"  Q_risk N={n_agents} {regime}: "
                    f"mean_Q={row['mean_Q_risk']:.3f}"
                )

        # Production pick for this encoding
        knn_loss = holdout["kernel_nn"]["log_loss"]
        kh_loss = holdout["kernel_hurdle"]["log_loss"]
        global_ref = cv_summary.get("by_scheme_model", {}).get(
            "acquisition_block_holdout::global_empirical", {}
        ).get("log_loss", 1e9)
        production = "kernel_hurdle" if kh_loss <= knn_loss * 1.02 else "kernel_nn"
        stay_gap = holdout[production].get("moments_stay_gap", float("nan"))
        direction_ok = bool(np.isfinite(stay_gap) and stay_gap > 0)

        summary["representations"][representation] = {
            "n_rows": len(rep_rows),
            "risk_spearman_oof": spearman,
            "risk_monotone_ok": mono,
            "risk_threshold_oof80": risk_threshold,
            "holdout_block": test_block,
            "holdout": holdout,
            "cv_summary_keys": list(cv_summary.get("by_scheme_model", {}).keys())[:12],
            "cv_acquisition_block": {
                k.split("::", 1)[1]: v["log_loss"]
                for k, v in cv_summary.get("by_scheme_model", {}).items()
                if k.startswith("acquisition_block_holdout::")
            },
            "production_model": production,
            "beats_global": bool(
                holdout[production]["log_loss"] < global_ref
                if np.isfinite(global_ref)
                else True
            ),
            "moments_direction_ok": direction_ok,
            "moments_stay_gap": float(stay_gap) if np.isfinite(stay_gap) else None,
        }

    write_csv(out_dir / "oof_residual_table_post_pilot.csv", oof_all)
    write_csv(out_dir / "post_pilot_model_compare.csv", model_compare)
    qrisk_out = [
        {
            "representation": r["representation"],
            "n_agents": r["n_agents"],
            "peer_count": r["peer_count"],
            "regime": r["regime"],
            "mean_Q_risk": r["mean_Q_risk"],
            "mean_q_high_risk": r["mean_q_high_risk"],
            "final_q_high_risk": r["final_q_high_risk"],
            "mean_R": r["mean_R"],
            "mean_q_unsupported_peer": r["mean_q_unsupported_peer"],
        }
        for r in qrisk_rows
    ]
    write_csv(out_dir / "post_pilot_qrisk.csv", qrisk_out)

    moments = summary["representations"]["moments_m1_m3"]
    closed = [
        r
        for r in qrisk_out
        if r["representation"] == "moments_m1_m3"
        and r["regime"] == "closed_loop"
        and r["n_agents"] in (9, 17)
    ]
    mean_q = float(np.mean([r["mean_Q_risk"] for r in closed])) if closed else float("nan")

    moments_freeze = bool(
        moments["risk_spearman_oof"] > 0.2
        and moments["risk_monotone_ok"]
        and moments["moments_direction_ok"]
        and moments["beats_global"]
        and mean_q < 0.35
    )
    decision = {
        "status": "post_pilot_refit_complete",
        "stage_3b_full_freeze": False,
        "stage_3b_partial_freeze_moments": moments_freeze,
        "moments_production_model": moments["production_model"],
        "moments_risk_spearman": moments["risk_spearman_oof"],
        "moments_closed_loop_mean_Q_risk_N9_N17": mean_q,
        "gates_moments": {
            "risk_spearman_gt_0.2": moments["risk_spearman_oof"] > 0.2,
            "risk_monotone_ok": moments["risk_monotone_ok"],
            "moments_direction_ok": moments["moments_direction_ok"],
            "beats_global": moments["beats_global"],
            "Q_risk_closed_lt_0.35": mean_q < 0.35,
        },
        "per_representation": summary["representations"],
        "n_rows_by_source": summary["n_rows_by_source"],
        "stage_c_blocked": True,
        "note": (
            "Partial freeze authorizes moments production use under peer hard-gate "
            "and calibrated R; intervals/centers remain exploratory."
        ),
    }
    (out_dir / "stage3b_decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )
    print("\n" + json.dumps({
        "partial_freeze_moments": moments_freeze,
        "production": moments["production_model"],
        "spearman": moments["risk_spearman_oof"],
        "mean_Q_closed": mean_q,
        "gates": decision["gates_moments"],
        "n_rows_by_source": summary["n_rows_by_source"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

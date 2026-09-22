"""Stage 3A runner: dataset integration, candidate surrogates, grouped CV, OOD scan."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.stage3.collective_scan import (  # noqa: E402
    common_ood_feature_builder,
    default_feature_builder,
    run_surrogate_collective_scan,
)
from analysis.stage3.dataset import (  # noqa: E402
    build_unified_dataset,
    save_packed_npz,
    write_dataset_csv,
)
from analysis.stage3.models import fit_candidate_models  # noqa: E402
from analysis.stage3.ood import OODDetector, ood_error_calibration  # noqa: E402
from analysis.stage3.validation import (  # noqa: E402
    action_metrics,
    evaluate_grouped_schemes,
    moments_abstention_activation_check,
    tv_distance,
)
from circlemap.field_features import COMMON_FEATURE_NAMES  # noqa: E402


def _group_maps(rows: list) -> dict[str, list[str]]:
    return {
        "leave_one_profile_out": [row.profile_family for row in rows],
        "leave_one_eps_level_out": [row.epsilon_level for row in rows],
        "acquisition_block_holdout": [row.acquisition_block for row in rows],
        "sparse_realization_holdout": [row.sparse_realization for row in rows],
        "offset_group_holdout": [row.offset_group for row in rows],
    }


def _select_model(summary: dict, representation: str) -> str:
    """Pick simplest model that beats global and is within best CV log-loss.

    Kernel-NN is an allowed production candidate (not a baseline to beat).
    """
    by = summary.get("by_scheme_model", {})
    candidates = [
        "kernel_nn",
        "multinomial_logistic_l2",
        "hurdle_multinomial",
        "softmax_stump_boost",
    ]
    key_block = "acquisition_block_holdout"
    scores = {}
    for name in ["global_empirical", *candidates]:
        key = f"{key_block}::{name}"
        if key in by:
            scores[name] = by[key]["log_loss"]
    if not scores:
        return "kernel_nn"
    global_loss = scores.get("global_empirical", 1e9)
    eligible = {
        name: loss
        for name, loss in scores.items()
        if name != "global_empirical" and loss < global_loss
    }
    if not eligible:
        return min(
            (name for name in candidates if name in scores),
            key=lambda name: scores[name],
            default="kernel_nn",
        )
    best_loss = min(eligible.values())
    # Prefer simpler models if within 2% of best.
    preference = (
        "kernel_nn",
        "hurdle_multinomial",
        "multinomial_logistic_l2",
        "softmax_stump_boost",
    )
    close = {
        name: loss
        for name, loss in eligible.items()
        if loss <= best_loss * 1.02
    }
    for name in preference:
        if name in close:
            return name
    return min(eligible, key=eligible.get)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "analysis" / "stage3a_artifacts",
    )
    parser.add_argument(
        "--skip-stump-boost-cv",
        action="store_true",
        help="Skip expensive stump-boost leave-one-group CV (still fit full model).",
    )
    parser.add_argument("--coverage-steps", type=int, default=30)
    parser.add_argument("--coverage-agents", type=int, default=16)
    args = parser.parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Building unified Stage 3A dataset...")
    rows, packed = build_unified_dataset(ROOT)
    write_dataset_csv(rows, out_dir / "training_rows.csv")
    save_packed_npz(packed, out_dir / "training_arrays.npz")
    print(f"  rows={len(rows)} representations={list(packed)}")

    report: dict = {
        "status": "stage_3a_not_frozen",
        "policy_near_zero": "A_unresolved_ood",
        "n_rows": len(rows),
        "representations": {},
        "notes": [
            "chi_a1 and eps_1/2 are not model inputs; full trinomials are targets.",
            "Production surrogate must not depend on labeled epsilon alone.",
            "Stage 3B freeze requires grouped-validation + closed-loop OOD gates.",
        ],
    }

    for representation, arrays in packed.items():
        print(f"\n=== {representation} ===")
        x = arrays["features"]
        counts = arrays["counts"]
        rep_rows = [row for row in rows if row.representation == representation]
        models = fit_candidate_models(x, counts)
        model_names = list(models)
        if args.skip_stump_boost_cv:
            cv_names = tuple(
                name for name in model_names if name != "softmax_stump_boost"
            )
        else:
            cv_names = tuple(model_names)

        summary = evaluate_grouped_schemes(
            x,
            counts,
            _group_maps(rep_rows),
            model_names=cv_names,
        )

        # In-sample diagnostics + OOD calibration using held-out block if possible.
        blocks = sorted({row.acquisition_block for row in rep_rows})
        if len(blocks) >= 2:
            test_block = blocks[-1]
            test_mask = np.asarray(
                [row.acquisition_block == test_block for row in rep_rows],
                dtype=bool,
            )
            train_mask = ~test_mask
            fitted = fit_candidate_models(x[train_mask], counts[train_mask])
            n_common = len(COMMON_FEATURE_NAMES)
            x_ood_train = x[train_mask][:, :n_common]
            x_ood_test = x[test_mask][:, :n_common]
            ood = OODDetector.fit(x_ood_train)
            chosen_name = _select_model(summary, representation)
            if chosen_name not in fitted:
                chosen_name = "multinomial_logistic_l2"
            chosen = fitted[chosen_name]
            pred = chosen.predict_proba(x[test_mask])
            empir = counts[test_mask] / np.maximum(
                np.sum(counts[test_mask], axis=1, keepdims=True), 1.0
            )
            tv = tv_distance(pred, empir)
            ood_scores = ood.score(
                x_ood_test,
                unresolved_near_zero=arrays["unresolved_near_zero"][test_mask],
            )
            calibration = ood_error_calibration(ood_scores, tv)
            holdout_metrics = action_metrics(pred, counts[test_mask])
            moments_check = moments_abstention_activation_check(
                [rep_rows[i] for i, flag in enumerate(test_mask) if flag],
                pred,
                representation=representation,
            )
            # Coverage scan with full-data fit of chosen model.
            full_fit = models[chosen_name]
            full_ood = OODDetector.fit(x[:, :n_common])
            coverage = run_surrogate_collective_scan(
                full_fit,
                full_ood,
                representation=representation,
                n_agents=args.coverage_agents,
                n_steps=args.coverage_steps,
                feature_builder=default_feature_builder,
                ood_feature_builder=common_ood_feature_builder,
            )
            coverage_payload = {
                "q_ood": coverage.q_ood,
                "n_agent_steps": coverage.n_agent_steps,
                "n_ood": coverage.n_ood,
                "mean_ood_score": coverage.mean_ood_score,
                "final_order_parameter": coverage.final_order_parameter,
            }
        else:
            chosen_name = "multinomial_logistic_l2"
            holdout_metrics = {}
            moments_check = {}
            calibration = []
            coverage_payload = {}

        beats = {}
        for scheme_model, metrics in summary.get("by_scheme_model", {}).items():
            scheme, model = scheme_model.split("::", 1)
            base_g = summary["by_scheme_model"].get(f"{scheme}::global_empirical")
            base_k = summary["by_scheme_model"].get(f"{scheme}::kernel_nn")
            if base_g is None or base_k is None:
                continue
            if model in {"global_empirical", "kernel_nn"}:
                continue
            beats[scheme_model] = {
                "log_loss": metrics["log_loss"],
                "beats_global": metrics["log_loss"] < base_g["log_loss"],
                "beats_kernel": metrics["log_loss"] < base_k["log_loss"],
            }

        report["representations"][representation] = {
            "n_rows": int(x.shape[0]),
            "n_features": int(x.shape[1]),
            "common_features": list(COMMON_FEATURE_NAMES),
            "selected_model_provisional": chosen_name,
            "grouped_validation": summary["by_scheme_model"],
            "baseline_comparisons": beats,
            "acquisition_holdout_metrics": holdout_metrics,
            "moments_abstention_activation": moments_check,
            "ood_error_calibration": calibration,
            "closed_loop_coverage": coverage_payload,
            "freeze_ready": False,
        }
        print(
            f"  provisional={chosen_name} "
            f"coverage_q_ood={coverage_payload.get('q_ood', float('nan')):.3f}"
        )

    report_path = out_dir / "stage3a_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {report_path}")
    print("Stage 3A complete: models are NOT frozen (Stage 3B gates pending).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

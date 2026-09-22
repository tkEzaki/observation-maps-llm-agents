"""Stage C v0.2a post-run analysis vs locked v1/v2 surrogates."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.stage3.applicability_domain import ApplicabilityDomain  # noqa: E402
from analysis.stage3.models import KernelHurdleMultinomial, RegimeAwareKernelHurdle  # noqa: E402
from analysis.stage3.risk_calibrator import RiskCalibrator  # noqa: E402
from analysis.stage_c.offline_diagnostics_v0_1 import (  # noqa: E402
    analyze_llm_risk_and_teacher,
    analyze_trajectories,
)


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _nanmean(vals: list[float]) -> float:
    arr = np.asarray(vals, dtype=np.float64)
    return float(np.nanmean(arr)) if arr.size else float("nan")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--session",
        type=Path,
        default=ROOT
        / "runs"
        / "stage_c"
        / "stage-c-v0.2a_ec22717957a1"
        / "20260724T014052Z",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "analysis" / "stage_c_artifacts" / "stage_c_v0_2a",
    )
    parser.add_argument(
        "--verdict-name",
        type=str,
        default="v0_2a_verdict.json",
    )
    args = parser.parse_args()
    args.out_dir = args.out_dir if args.out_dir.is_absolute() else (ROOT / args.out_dir)
    args.session = args.session if args.session.is_absolute() else (ROOT / args.session)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    session = args.session
    summary = json.loads((session / "session_summary.json").read_text(encoding="utf-8"))
    protocol = json.loads((session / "protocol.json").read_text(encoding="utf-8"))

    traj = analyze_trajectories(session, args.out_dir)

    # Cost from actual tokens
    in_tok = sum(int(r["actual_input_tokens"]) for r in summary["runs"])
    out_tok = sum(int(r["actual_output_tokens"]) for r in summary["runs"])
    cost = (in_tok * 0.75 + out_tok * 4.5) / 1_000_000
    cost_payload = {
        "total_calls": summary["total_calls"],
        "total_valid": summary["total_valid"],
        "valid_rate": summary["total_valid"] / max(summary["total_calls"], 1),
        "actual_input_tokens": in_tok,
        "actual_output_tokens": out_tok,
        "actual_cost_usd": cost,
        "elapsed_seconds": summary["elapsed_seconds"],
        "estimate_base_usd": float(
            protocol.get("cost_ceiling_usd_estimate", 12.0)
        ),  # ceiling recorded; actual estimate in cost card
    }
    (args.out_dir / "actual_cost.json").write_text(
        json.dumps(cost_payload, indent=2), encoding="utf-8"
    )

    # Compare to locked surrogates
    locked = list(
        csv.DictReader(
            (args.out_dir / "surrogate_predictions_locked.csv").open(encoding="utf-8")
        )
    )
    by_key = {(r["bundle"], r["run_id"]): r for r in locked}

    compare_rows = []
    for row in traj:
        run_id = row["run_id"]
        for bundle in ("v1", "v2"):
            s = by_key[(bundle, run_id)]
            compare_rows.append(
                {
                    "run_id": run_id,
                    "n_agents": row["n_agents"],
                    "coupling": row["coupling"],
                    "seed_index": row["seed_index"],
                    "bundle": bundle,
                    "llm_final_r1": row["final_r1"],
                    "llm_mean_r1": row["mean_r1"],
                    "llm_activity": row["mean_activity"],
                    "llm_t05": row["t_r1_gt_0.5"],
                    "llm_t09": row["t_r1_gt_0.9"],
                    "llm_tau": row["mean_tau_social"],
                    "llm_p_stay": row["mean_p_stay_emp"],
                    "surr_final_r1": float(s["mean_final_r"]),
                    "surr_mean_r1": float(s["mean_mean_r"]),
                    "surr_activity": float(s["mean_activity"]),
                    "surr_t05": float(s["mean_t_r1_gt_0.5"]),
                    "surr_t09": float(s["mean_t_r1_gt_0.9"]),
                    "surr_tau": float(s["mean_tau_social"]),
                    "err_final_r1": row["final_r1"] - float(s["mean_final_r"]),
                    "err_activity": row["mean_activity"] - float(s["mean_activity"]),
                    "err_t09": (
                        row["t_r1_gt_0.9"] - float(s["mean_t_r1_gt_0.9"])
                        if np.isfinite(row["t_r1_gt_0.9"])
                        and np.isfinite(float(s["mean_t_r1_gt_0.9"]))
                        else float("nan")
                    ),
                }
            )
    _write_csv(args.out_dir / "llm_vs_surrogate_per_run.csv", compare_rows)

    # Aggregate by (N, K) × bundle
    cells = sorted(
        {
            (int(r["n_agents"]), float(r["coupling"]))
            for r in traj
        },
        key=lambda x: (x[0], x[1]),
    )
    agg = []
    for n_agents, coupling in cells:
        llm_sub = [
            r
            for r in traj
            if int(r["n_agents"]) == n_agents
            and abs(float(r["coupling"]) - coupling) < 1e-9
        ]
        row = {
            "n_agents": n_agents,
            "coupling": float(coupling),
            "n_seeds": len(llm_sub),
            "llm_mean_final_r1": _nanmean([r["final_r1"] for r in llm_sub]),
            "llm_mean_mean_r1": _nanmean([r["mean_r1"] for r in llm_sub]),
            "llm_mean_activity": _nanmean([r["mean_activity"] for r in llm_sub]),
            "llm_mean_t05": _nanmean([r["t_r1_gt_0.5"] for r in llm_sub]),
            "llm_mean_t09": _nanmean([r["t_r1_gt_0.9"] for r in llm_sub]),
            "llm_mean_tau": _nanmean([r["mean_tau_social"] for r in llm_sub]),
            "llm_mean_p_stay": _nanmean([r["mean_p_stay_emp"] for r in llm_sub]),
            "llm_p_final_r1_gt_0.9": float(
                np.mean([r["final_r1"] > 0.9 for r in llm_sub])
            ),
        }
        for bundle in ("v1", "v2"):
            csub = [
                r
                for r in compare_rows
                if r["bundle"] == bundle
                and int(r["n_agents"]) == n_agents
                and abs(float(r["coupling"]) - coupling) < 1e-9
            ]
            row[f"{bundle}_mean_activity"] = _nanmean(
                [r["surr_activity"] for r in csub]
            )
            row[f"{bundle}_mean_final_r1"] = _nanmean(
                [r["surr_final_r1"] for r in csub]
            )
            row[f"{bundle}_mean_t09"] = _nanmean([r["surr_t09"] for r in csub])
            row[f"mae_activity_vs_{bundle}"] = _nanmean(
                [abs(r["err_activity"]) for r in csub]
            )
            row[f"mean_err_activity_vs_{bundle}"] = _nanmean(
                [r["err_activity"] for r in csub]
            )
            row[f"mae_final_r1_vs_{bundle}"] = _nanmean(
                [abs(r["err_final_r1"]) for r in csub]
            )
        agg.append(row)
    _write_csv(args.out_dir / "llm_vs_surrogate_by_K.csv", agg)

    # Teacher-forced / risk with both bundles
    teacher_summary = {}
    for label, model_path, bundle_path in (
        (
            "v1",
            "kernel_hurdle",
            ROOT / protocol["bundle_root_comparator"],
        ),
        (
            "v2",
            "regime_kernel_hurdle",
            ROOT / protocol["bundle_root_prediction"],
        ),
    ):
        if label == "v1":
            model = KernelHurdleMultinomial.load(bundle_path / model_path)
        else:
            model = RegimeAwareKernelHurdle.load(bundle_path / model_path)
        ad = ApplicabilityDomain.load(bundle_path / "applicability_domain")
        cal = RiskCalibrator.load(bundle_path / "risk_calibrator.json")
        thr = float(
            json.loads((bundle_path / "ood_policy.json").read_text(encoding="utf-8"))[
                "risk"
            ]["risk_threshold"]
        )
        risk_rows, _teacher_rows = analyze_llm_risk_and_teacher(
            session,
            model=model,
            ad=ad,
            calibrator=cal,
            risk_threshold=thr,
            representation=protocol["representation"],
            out_dir=args.out_dir / f"teacher_{label}",
        )
        teacher_summary[label] = {
            "mean_teacher_log_loss": float(
                np.mean([r["teacher_log_loss"] for r in risk_rows])
            ),
            "mean_pred_stay": float(np.mean([r["pred_stay_frac"] for r in risk_rows])),
            "mean_obs_stay": float(np.mean([r["obs_stay_frac"] for r in risk_rows])),
            "mean_Q_risk": float(np.mean([r["Q_risk"] for r in risk_rows])),
            "mean_abs_stay_err": float(
                np.mean(
                    [
                        abs(r["pred_stay_frac"] - r["obs_stay_frac"])
                        for r in risk_rows
                    ]
                )
            ),
        }

    verdict = {
        "session": str(session.relative_to(ROOT).as_posix()),
        "valid_rate": cost_payload["valid_rate"],
        "actual_cost_usd": cost,
        "by_K": agg,
        "activity_mae": {
            "vs_v1": _nanmean([r["mae_activity_vs_v1"] for r in agg]),
            "vs_v2": _nanmean([r["mae_activity_vs_v2"] for r in agg]),
        },
        "activity_mean_err": {
            "vs_v1": _nanmean([r["mean_err_activity_vs_v1"] for r in agg]),
            "vs_v2": _nanmean([r["mean_err_activity_vs_v2"] for r in agg]),
        },
        "final_r1_mae": {
            "vs_v1": _nanmean([r["mae_final_r1_vs_v1"] for r in agg]),
            "vs_v2": _nanmean([r["mae_final_r1_vs_v2"] for r in agg]),
        },
        "v2_closer_on_activity": (
            _nanmean([r["mae_activity_vs_v2"] for r in agg])
            < _nanmean([r["mae_activity_vs_v1"] for r in agg])
        ),
        "teacher": teacher_summary,
    }
    (args.out_dir / args.verdict_name).write_text(
        json.dumps(verdict, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                k: verdict[k]
                for k in (
                    "valid_rate",
                    "actual_cost_usd",
                    "activity_mae",
                    "activity_mean_err",
                    "final_r1_mae",
                    "v2_closer_on_activity",
                    "teacher",
                )
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

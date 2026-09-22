"""Rich offline dense-K compare: v1 vs v2 (activity, arrival, torque)."""

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
from analysis.stage3.models import KernelHurdleMultinomial, RegimeAwareKernelHurdle  # noqa: E402
from analysis.stage3.risk_calibrator import RiskCalibrator  # noqa: E402
from analysis.stage_c.run_surrogate_stage_c import run_one  # noqa: E402

REP = "moments_m1_m3"


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _nanmean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    return float(np.nanmean(arr)) if arr.size else float("nan")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-seeds", type=int, default=12)
    parser.add_argument("--n-steps", type=int, default=100)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "analysis" / "stage3a_artifacts",
    )
    args = parser.parse_args()

    v1_dir = args.out_dir / "moments_bundle_v1"
    v2_dir = args.out_dir / "moments_bundle_v2"
    models = {
        "v1": (
            KernelHurdleMultinomial.load(v1_dir / "kernel_hurdle"),
            ApplicabilityDomain.load(v1_dir / "applicability_domain"),
            RiskCalibrator.load(v1_dir / "risk_calibrator.json"),
            float(
                json.loads((v1_dir / "ood_policy.json").read_text(encoding="utf-8"))[
                    "risk"
                ]["risk_threshold"]
            ),
        ),
        "v2": (
            RegimeAwareKernelHurdle.load(v2_dir / "regime_kernel_hurdle"),
            ApplicabilityDomain.load(v2_dir / "applicability_domain"),
            RiskCalibrator.load(v2_dir / "risk_calibrator.json"),
            float(
                json.loads((v2_dir / "ood_policy.json").read_text(encoding="utf-8"))[
                    "risk"
                ]["risk_threshold"]
            ),
        ),
    }
    k_grid = sorted(
        set([-0.15, 0.0, 0.15] + [round(0.02 * i, 2) for i in range(1, 16)])
    )
    run_rows: list[dict] = []
    for label, (model, ad, cal, thr) in models.items():
        for n_agents in (9, 17):
            for k in k_grid:
                for seed_i in range(args.n_seeds):
                    seed = (
                        2026072500
                        + 1000 * n_agents
                        + int(abs(k) * 1000)
                        + seed_i
                    )
                    out = run_one(
                        model=model,
                        ad=ad,
                        calibrator=cal,
                        risk_threshold=thr,
                        n_agents=n_agents,
                        n_steps=args.n_steps,
                        coupling=float(k),
                        omega_halfwidth=0.05,
                        seed=seed,
                        representation=REP,
                        track_rich=True,
                    )
                    run_rows.append(
                        {
                            "bundle": label,
                            "n_agents": n_agents,
                            "coupling": float(k),
                            "seed_index": seed_i,
                            "final_r": out["final_r"],
                            "mean_r": out["mean_r"],
                            "t_r1_gt_0.5": out["t_r1_gt_0.5"],
                            "t_r1_gt_0.9": out["t_r1_gt_0.9"],
                            "mean_activity": out["mean_activity"],
                            "mean_tau_social": out["mean_tau_social"],
                        }
                    )
                print(f"{label} N={n_agents} K={k:+.2f} done")

    _write_csv(args.out_dir / "dense_k_rich_runs_v1_vs_v2.csv", run_rows)

    agg_rows = []
    for label in ("v1", "v2"):
        for n_agents in (9, 17):
            for k in k_grid:
                sub = [
                    r
                    for r in run_rows
                    if r["bundle"] == label
                    and r["n_agents"] == n_agents
                    and abs(r["coupling"] - k) < 1e-12
                ]
                finals = [r["final_r"] for r in sub]
                agg_rows.append(
                    {
                        "bundle": label,
                        "n_agents": n_agents,
                        "coupling": float(k),
                        "p_final_r1_gt_0.9": float(np.mean(np.asarray(finals) > 0.9)),
                        "mean_final_r1": float(np.mean(finals)),
                        "median_final_r1": float(np.median(finals)),
                        "mean_t_r1_gt_0.5": _nanmean([r["t_r1_gt_0.5"] for r in sub]),
                        "mean_t_r1_gt_0.9": _nanmean([r["t_r1_gt_0.9"] for r in sub]),
                        "mean_activity": _nanmean([r["mean_activity"] for r in sub]),
                        "mean_tau_social": _nanmean(
                            [r["mean_tau_social"] for r in sub]
                        ),
                        "n_seeds": len(sub),
                    }
                )
    _write_csv(args.out_dir / "dense_k_rich_v1_vs_v2.csv", agg_rows)

    # Delta table at shared K of interest
    focus_k = {
        9: [-0.15, 0.0, 0.06, 0.08, 0.10, 0.15],
        17: [-0.15, 0.0, 0.08, 0.10, 0.12, 0.15],
    }

    def _find(bundle: str, n_agents: int, k: float) -> dict:
        for r in agg_rows:
            if (
                r["bundle"] == bundle
                and int(r["n_agents"]) == int(n_agents)
                and abs(float(r["coupling"]) - float(k)) < 1e-9
            ):
                return r
        raise KeyError(f"missing agg row {bundle} N={n_agents} K={k}")

    deltas = []
    for n_agents, ks in focus_k.items():
        for k in ks:
            a = _find("v1", n_agents, k)
            b = _find("v2", n_agents, k)
            deltas.append(
                {
                    "n_agents": n_agents,
                    "coupling": k,
                    "delta_p90": b["p_final_r1_gt_0.9"] - a["p_final_r1_gt_0.9"],
                    "delta_mean_r1": b["mean_final_r1"] - a["mean_final_r1"],
                    "delta_mean_t05": b["mean_t_r1_gt_0.5"] - a["mean_t_r1_gt_0.5"],
                    "delta_mean_t09": b["mean_t_r1_gt_0.9"] - a["mean_t_r1_gt_0.9"],
                    "delta_activity": b["mean_activity"] - a["mean_activity"],
                    "delta_tau": b["mean_tau_social"] - a["mean_tau_social"],
                    "v1_activity": a["mean_activity"],
                    "v2_activity": b["mean_activity"],
                    "v1_t09": a["mean_t_r1_gt_0.9"],
                    "v2_t09": b["mean_t_r1_gt_0.9"],
                }
            )
    _write_csv(args.out_dir / "dense_k_rich_deltas_focus.csv", deltas)
    summary = {
        "n_seeds": args.n_seeds,
        "n_steps": args.n_steps,
        "focus_deltas": deltas,
        "artifacts": {
            "runs": "analysis/stage3a_artifacts/dense_k_rich_runs_v1_vs_v2.csv",
            "agg": "analysis/stage3a_artifacts/dense_k_rich_v1_vs_v2.csv",
            "deltas": "analysis/stage3a_artifacts/dense_k_rich_deltas_focus.csv",
        },
    }
    (args.out_dir / "dense_k_rich_v1_vs_v2_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps({"n_agg": len(agg_rows), "n_deltas": len(deltas)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

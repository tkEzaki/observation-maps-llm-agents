"""Offline dense K scan with selection-grade moments_bundle_v1 (no API).

Treat results as a *conservative / attenuated* response envelope — not as the
sole basis for Stage C v0.2 cell selection.
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
from analysis.stage3.models import KernelHurdleMultinomial  # noqa: E402
from analysis.stage3.risk_calibrator import RiskCalibrator  # noqa: E402
from analysis.stage_c.run_surrogate_stage_c import run_one  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle",
        type=Path,
        default=ROOT / "analysis" / "stage3a_artifacts" / "moments_bundle_v1",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "analysis" / "stage_c_artifacts" / "dense_k_v1",
    )
    parser.add_argument("--n-steps", type=int, default=100)
    parser.add_argument("--n-seeds", type=int, default=32)
    parser.add_argument("--omega-halfwidth", type=float, default=0.05)
    parser.add_argument("--base-seed", type=int, default=2026072499)
    args = parser.parse_args()

    model = KernelHurdleMultinomial.load(args.bundle / "kernel_hurdle")
    ad = ApplicabilityDomain.load(args.bundle / "applicability_domain")
    calibrator = RiskCalibrator.load(args.bundle / "risk_calibrator.json")
    ood = json.loads((args.bundle / "ood_policy.json").read_text(encoding="utf-8"))
    risk_threshold = float(ood["risk"]["risk_threshold"])
    representation = "moments_m1_m3"

    # Dense positive K + negative control + K=0
    k_grid = (
        [-0.20, -0.15, -0.10, -0.05]
        + [0.0]
        + [round(0.02 * i, 2) for i in range(1, 16)]  # 0.02..0.30
    )
    n_list = [9, 17]

    rows = []
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for n_agents in n_list:
        for k in k_grid:
            finals = []
            means = []
            t09 = []
            for seed_i in range(args.n_seeds):
                seed = args.base_seed + 1000 * n_agents + int(round(abs(k) * 1000)) + seed_i
                out = run_one(
                    model=model,
                    ad=ad,
                    calibrator=calibrator,
                    risk_threshold=risk_threshold,
                    n_agents=n_agents,
                    n_steps=args.n_steps,
                    coupling=float(k),
                    omega_halfwidth=args.omega_halfwidth,
                    seed=seed,
                    representation=representation,
                )
                finals.append(out["final_r"])
                means.append(out["mean_r"])
                # cheap arrival proxy from mean_r path not available; recompute lightly
                # run_one doesn't return series — use final only for lock fraction
            finals_a = np.asarray(finals)
            means_a = np.asarray(means)
            rows.append(
                {
                    "n_agents": n_agents,
                    "coupling": float(k),
                    "n_seeds": args.n_seeds,
                    "n_steps": args.n_steps,
                    "final_r1_mean": float(np.mean(finals_a)),
                    "final_r1_p10": float(np.quantile(finals_a, 0.10)),
                    "final_r1_p50": float(np.quantile(finals_a, 0.50)),
                    "final_r1_p90": float(np.quantile(finals_a, 0.90)),
                    "mean_r1_mean": float(np.mean(means_a)),
                    "p_final_r1_gt_0.5": float(np.mean(finals_a > 0.5)),
                    "p_final_r1_gt_0.9": float(np.mean(finals_a > 0.9)),
                    "envelope_note": "attenuated_v1_conservative",
                }
            )
            print(
                f"N={n_agents} K={k:+.2f}: "
                f"p(r>0.9)={rows[-1]['p_final_r1_gt_0.9']:.2f} "
                f"median={rows[-1]['final_r1_p50']:.3f}"
            )

    out_csv = args.out_dir / f"dense_k_T{args.n_steps}_s{args.n_seeds}.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Transition markers: first K where p(r>0.9) crosses 0.5 for each N
    summary = {"n_steps": args.n_steps, "n_seeds": args.n_seeds, "by_N": {}}
    for n_agents in n_list:
        sub = [r for r in rows if r["n_agents"] == n_agents and r["coupling"] >= 0]
        sub = sorted(sub, key=lambda r: r["coupling"])
        k_cross = None
        for r in sub:
            if r["p_final_r1_gt_0.9"] >= 0.5:
                k_cross = r["coupling"]
                break
        summary["by_N"][str(n_agents)] = {
            "k_p90_lock_ge_0.5": k_cross,
            "rows": [
                {
                    "K": r["coupling"],
                    "p_gt_0.9": r["p_final_r1_gt_0.9"],
                    "median_final_r1": r["final_r1_p50"],
                }
                for r in sub
            ],
        }
    summary["interpretation"] = (
        "v1 attenuates activity; treat k_cross as an upper-biased "
        "(harder-to-sync) envelope, not a true K_c estimate."
    )
    summary["artifact"] = str(out_csv.relative_to(ROOT).as_posix())
    (args.out_dir / f"dense_k_T{args.n_steps}_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "artifact": summary["artifact"],
        "k_cross": {n: summary["by_N"][n]["k_p90_lock_ge_0.5"] for n in summary["by_N"]},
        "note": summary["interpretation"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

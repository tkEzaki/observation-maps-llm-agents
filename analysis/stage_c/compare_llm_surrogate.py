"""Compare Stage C LLM trajectories to locked-bundle surrogate predictions."""

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
        "--session",
        type=Path,
        default=ROOT
        / "runs"
        / "stage_c"
        / "stage-c-v0.1_0291166d2e39"
        / "20260723T235620Z",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "analysis" / "stage_c_artifacts",
    )
    parser.add_argument("--input-price", type=float, default=0.75)
    parser.add_argument("--output-price", type=float, default=4.5)
    args = parser.parse_args()

    rep = json.loads((args.session / "session_summary.json").read_text(encoding="utf-8"))
    protocol = json.loads((args.session / "protocol.json").read_text(encoding="utf-8"))
    bundle = ROOT / protocol["bundle_root"]
    model = KernelHurdleMultinomial.load(bundle / "kernel_hurdle")
    ad = ApplicabilityDomain.load(bundle / "applicability_domain")
    calibrator = RiskCalibrator.load(bundle / "risk_calibrator.json")
    ood = json.loads((bundle / "ood_policy.json").read_text(encoding="utf-8"))
    risk_threshold = float(ood["risk"]["risk_threshold"])

    tin = sum(r["actual_input_tokens"] for r in rep["runs"])
    tout = sum(r["actual_output_tokens"] for r in rep["runs"])
    cost = (tin * args.input_price + tout * args.output_price) / 1_000_000

    rows = []
    for r in rep["runs"]:
        # Matched init seed + (N,K,T) so surrogate uses the same RNG start.
        s = run_one(
            model=model,
            ad=ad,
            calibrator=calibrator,
            risk_threshold=risk_threshold,
            n_agents=int(r["n_agents"]),
            n_steps=int(r["n_steps"]),
            coupling=float(r["coupling"]),
            omega_halfwidth=float(protocol["omega_halfwidth"]),
            seed=int(r["init_seed"]),
            representation=protocol["representation"],
        )
        rows.append(
            {
                "run_id": r["run_id"],
                "n_agents": r["n_agents"],
                "coupling": r["coupling"],
                "seed_index": r["seed_index"],
                "init_seed": r["init_seed"],
                "llm_final_r": r["final_r"],
                "llm_mean_r": r["mean_r"],
                "surr_final_r": float(s["final_r"]),
                "surr_mean_r": float(s["mean_r"]),
                "abs_err_final_r": abs(r["final_r"] - float(s["final_r"])),
                "abs_err_mean_r": abs(r["mean_r"] - float(s["mean_r"])),
            }
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.out_dir / "llm_vs_surrogate_T40.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    by_k = {}
    for k in (-0.15, 0.0, 0.15):
        sub = [x for x in rows if abs(float(x["coupling"]) - k) < 1e-12]
        by_k[str(k)] = {
            "llm_final_r": float(np.mean([x["llm_final_r"] for x in sub])),
            "surr_final_r": float(np.mean([x["surr_final_r"] for x in sub])),
            "mae_final_r": float(np.mean([x["abs_err_final_r"] for x in sub])),
        }
        print(
            f"K={k:+.2f}: llm_final={by_k[str(k)]['llm_final_r']:.3f} "
            f"surr_final={by_k[str(k)]['surr_final_r']:.3f} "
            f"mae={by_k[str(k)]['mae_final_r']:.3f}"
        )

    summary = {
        "session_dir": str(args.session),
        "total_calls": rep["total_calls"],
        "total_valid": rep["total_valid"],
        "valid_rate": rep["total_valid"] / max(rep["total_calls"], 1),
        "actual_input_tokens": tin,
        "actual_output_tokens": tout,
        "actual_cost_usd": cost,
        "overall_mae_final_r": float(np.mean([x["abs_err_final_r"] for x in rows])),
        "overall_mae_mean_r": float(np.mean([x["abs_err_mean_r"] for x in rows])),
        "by_K": by_k,
        "comparison_csv": str(out_csv.relative_to(ROOT).as_posix()),
    }
    out_json = args.out_dir / "stage_c_v0_1_report.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

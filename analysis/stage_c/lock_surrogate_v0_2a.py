"""Lock offline v1/v2 surrogate predictions for Stage C v0.2a cells (pre-paid)."""

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
from experiments.stage_c.run_collective import _run_specs  # noqa: E402

REP = "moments_m1_m3"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "experiments" / "stage_c" / "protocol_stage_c_v0_2a.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "analysis" / "stage_c_artifacts" / "stage_c_v0_2a",
    )
    parser.add_argument("--n-replicates", type=int, default=8)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    bundles = {
        "v1": ROOT / protocol["bundle_root_comparator"],
        "v2": ROOT / protocol["bundle_root_prediction"],
    }
    models = {
        "v1": KernelHurdleMultinomial.load(bundles["v1"] / "kernel_hurdle"),
        "v2": RegimeAwareKernelHurdle.load(bundles["v2"] / "regime_kernel_hurdle"),
    }
    ads = {
        label: ApplicabilityDomain.load(path / "applicability_domain")
        for label, path in bundles.items()
    }
    cals = {
        label: RiskCalibrator.load(path / "risk_calibrator.json")
        for label, path in bundles.items()
    }
    thrs = {
        label: float(
            json.loads((path / "ood_policy.json").read_text(encoding="utf-8"))["risk"][
                "risk_threshold"
            ]
        )
        for label, path in bundles.items()
    }

    specs = _run_specs(protocol)
    rows = []
    for label in ("v1", "v2"):
        for spec in specs:
            finals = []
            means = []
            acts = []
            t05s = []
            t09s = []
            taus = []
            for rep in range(args.n_replicates):
                seed = int(spec["init_seed"]) + 17_000 + 100 * rep
                out = run_one(
                    model=models[label],
                    ad=ads[label],
                    calibrator=cals[label],
                    risk_threshold=thrs[label],
                    n_agents=int(spec["n_agents"]),
                    n_steps=int(protocol["n_steps"]),
                    coupling=float(spec["coupling"]),
                    omega_halfwidth=float(protocol["omega_halfwidth"]),
                    seed=seed,
                    representation=REP,
                    track_rich=True,
                )
                finals.append(out["final_r"])
                means.append(out["mean_r"])
                acts.append(out["mean_activity"])
                t05s.append(out["t_r1_gt_0.5"])
                t09s.append(out["t_r1_gt_0.9"])
                taus.append(out["mean_tau_social"])
            rows.append(
                {
                    "bundle": label,
                    "run_id": spec["run_id"],
                    "n_agents": spec["n_agents"],
                    "coupling": spec["coupling"],
                    "seed_index": spec["seed_index"],
                    "n_replicates": args.n_replicates,
                    "mean_final_r": float(np.mean(finals)),
                    "mean_mean_r": float(np.mean(means)),
                    "mean_activity": float(np.nanmean(acts)),
                    "mean_t_r1_gt_0.5": float(np.nanmean(t05s)),
                    "mean_t_r1_gt_0.9": float(np.nanmean(t09s)),
                    "mean_tau_social": float(np.nanmean(taus)),
                    "p_final_r1_gt_0.9": float(np.mean(np.asarray(finals) > 0.9)),
                }
            )
            print(
                f"{label} {spec['run_id']}: "
                f"A={rows[-1]['mean_activity']:.3f} "
                f"rT={rows[-1]['mean_final_r']:.3f}"
            )

    out_csv = args.out_dir / "surrogate_predictions_locked.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lock = {
        "protocol_version": protocol["protocol_version"],
        "protocol_sha256": __import__("hashlib")
        .sha256(args.protocol.read_bytes())
        .hexdigest(),
        "n_replicates": args.n_replicates,
        "locked_before_paid_llm": True,
        "artifact": str(Path(out_csv).resolve().relative_to(ROOT).as_posix()),
        "v1_aggregate_sha256": json.loads(
            (bundles["v1"] / "manifest.json").read_text(encoding="utf-8")
        ).get("aggregate_sha256"),
        "v2_aggregate_sha256": json.loads(
            (bundles["v2"] / "manifest.json").read_text(encoding="utf-8")
        ).get("aggregate_sha256"),
    }
    (args.out_dir / "surrogate_lock.json").write_text(
        json.dumps(lock, indent=2), encoding="utf-8"
    )
    print(json.dumps(lock, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

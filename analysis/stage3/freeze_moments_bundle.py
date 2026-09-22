"""Hash-lock the moments Stage 3B production bundle (offline).

Persists kernel_hurdle + AD + risk calibrator + OOD policy and writes
sha256-locked manifest. Does NOT authorize paid Stage C by itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.stage3.applicability_domain import (  # noqa: E402
    ApplicabilityDomain,
    DEFAULT_TRAIN_PEERS,
)
from analysis.stage3.dataset import (  # noqa: E402
    build_unified_dataset,
    post_pilot_source_runs,
)
from analysis.stage3.models import KernelHurdleMultinomial  # noqa: E402
from analysis.stage3.oof_residuals import default_stay_indices  # noqa: E402
from analysis.stage3.risk_calibrator import RiskCalibrator  # noqa: E402
from circlemap.field_features import COMMON_FEATURE_NAMES  # noqa: E402

REP = "moments_m1_m3"
BUNDLE_VERSION = "stage3b-moments-v1"


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_json(payload: object) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _hash_tree(root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            rel = path.relative_to(root).as_posix()
            if rel == "manifest.json":
                continue
            hashes[rel] = _sha256_file(path)
    return hashes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "analysis" / "stage3a_artifacts" / "moments_bundle_v1",
    )
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=ROOT / "analysis" / "stage3a_artifacts",
    )
    args = parser.parse_args()
    out_dir = args.out_dir
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    print("Loading post-pilot dataset...")
    all_rows, packed = build_unified_dataset(
        ROOT, source_runs=post_pilot_source_runs(ROOT)
    )
    rep_rows = [r for r in all_rows if r.representation == REP]
    x = packed[REP]["features"]
    counts = packed[REP]["counts"]
    stay_idx = default_stay_indices()

    hyperparams = {
        "model": "kernel_hurdle",
        "representation": REP,
        "length_scale_stay": 1.2,
        "length_scale_dir": 1.5,
        "k_neighbors": 32,
        "dirichlet_alpha": 0.5,
        "stay_feature_indices": list(stay_idx),
        "stay_feature_names": [COMMON_FEATURE_NAMES[i] for i in stay_idx],
        "common_feature_names": list(COMMON_FEATURE_NAMES),
    }
    (out_dir / "hyperparameters.json").write_text(
        json.dumps(hyperparams, indent=2), encoding="utf-8"
    )

    print("Fitting production kernel_hurdle...")
    model = KernelHurdleMultinomial.fit(
        x,
        counts,
        stay_feature_indices=stay_idx,
        length_scale_stay=hyperparams["length_scale_stay"],
        length_scale_dir=hyperparams["length_scale_dir"],
        k_neighbors=hyperparams["k_neighbors"],
        alpha=hyperparams["dirichlet_alpha"],
    )
    model_dir = out_dir / "kernel_hurdle"
    model.save(model_dir)

    # Round-trip smoke check
    reloaded = KernelHurdleMultinomial.load(model_dir)
    probe = x[: min(32, x.shape[0])]
    delta = float(np.max(np.abs(model.predict_proba(probe) - reloaded.predict_proba(probe))))
    if delta > 1e-12:
        raise RuntimeError(f"kernel_hurdle reload mismatch: {delta}")

    print("Fitting applicability domain...")
    ad = ApplicabilityDomain.fit(
        x, representation=REP, train_peers=DEFAULT_TRAIN_PEERS
    )
    ad_dir = out_dir / "applicability_domain"
    ad.save(ad_dir)
    ad2 = ApplicabilityDomain.load(ad_dir)
    s1 = ad.score(probe)
    s2 = ad2.score(probe)
    for key in ("d_shape", "d_native", "d_orientation", "local_density"):
        err = float(np.max(np.abs(s1[key] - s2[key])))
        if err > 1e-12:
            raise RuntimeError(f"AD reload mismatch on {key}: {err}")

    # Risk calibrator: prefer post-pilot artifact; verify it loads.
    cal_src = args.artifacts / f"risk_calibrator_post_{REP}.json"
    if not cal_src.exists():
        raise FileNotFoundError(
            f"missing {cal_src}; run analysis.stage3.refit_post_pilot first"
        )
    shutil.copy2(cal_src, out_dir / "risk_calibrator.json")
    calibrator = RiskCalibrator.load(out_dir / "risk_calibrator.json")

    # Risk threshold from OOF table if present.
    oof_path = args.artifacts / "oof_residual_table_post_pilot.csv"
    risk_threshold = None
    if oof_path.exists():
        import csv

        oof_rows = []
        with oof_path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row["representation"] == REP:
                    oof_rows.append(row)
        if oof_rows:
            r_hat = calibrator.predict_rows(oof_rows)
            risk_threshold = float(np.quantile(r_hat, 0.8))

    decision = json.loads(
        (args.artifacts / "stage3b_decision.json").read_text(encoding="utf-8")
    )
    if risk_threshold is None:
        risk_threshold = float(
            decision["per_representation"][REP]["risk_threshold_oof80"]
        )

    ood_policy = {
        "version": "stage3b-moments-ood-v1",
        "representation": REP,
        "peer_hard_gate": {
            "allowed_peers": list(DEFAULT_TRAIN_PEERS),
            "action": "reject",
        },
        "collective_n_allowed": [9, 17],
        "risk": {
            "calibrator": "risk_calibrator.json",
            "feature_schema": [
                "d_shape",
                "d_native",
                "local_density",
                "u_ensemble",
                "d_orientation",
                "policy_a",
            ],
            "threshold_quantile": 0.8,
            "risk_threshold": risk_threshold,
            "interpretation": "ranking_applicability_not_precise_error",
        },
        "policy_a_near_zero": {
            "action": "flag",
            "description": "unresolved_near_zero band from field descriptors",
        },
        "unsupported_encodings": [
            "intervals_24_decimal6",
            "centers_24_standard",
        ],
        "stage_c": {
            "authorized": False,
            "requires": "explicit_user_go_after_hash_lock_review",
        },
    }
    (out_dir / "ood_policy.json").write_text(
        json.dumps(ood_policy, indent=2), encoding="utf-8"
    )

    # Persist moments training slice for exact reproducibility.
    np.savez_compressed(
        out_dir / "training_arrays_moments.npz",
        features=x,
        counts=counts,
        unresolved_near_zero=packed[REP]["unresolved_near_zero"],
        estimated_eps=packed[REP]["estimated_eps"],
    )
    # Also copy post-pilot row table if present.
    rows_src = args.artifacts / "training_rows_post_pilot.csv"
    if rows_src.exists():
        shutil.copy2(rows_src, out_dir / "training_rows_post_pilot.csv")

    source_counts: dict[str, int] = {}
    for row in rep_rows:
        source_counts[row.source_family] = source_counts.get(row.source_family, 0) + 1

    provenance = {
        "bundle_version": BUNDLE_VERSION,
        "representation": REP,
        "production_model": "kernel_hurdle",
        "n_training_rows_moments": len(rep_rows),
        "n_rows_by_source": source_counts,
        "gates_from_stage3b_decision": decision.get("gates_moments"),
        "moments_risk_spearman": decision.get("moments_risk_spearman"),
        "moments_closed_loop_mean_Q_risk_N9_N17": decision.get(
            "moments_closed_loop_mean_Q_risk_N9_N17"
        ),
        "reload_smoke_max_abs_delta": delta,
    }
    (out_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )

    file_hashes = _hash_tree(out_dir)
    manifest = {
        "status": "hash_locked",
        "bundle_version": BUNDLE_VERSION,
        "representation": REP,
        "production_model": "kernel_hurdle",
        "freeze_ready": True,
        "stage_c_authorized": False,
        "created_at": "2026-07-24",
        "bundle_root": str(out_dir.relative_to(ROOT).as_posix()),
        "file_sha256": file_hashes,
        "aggregate_sha256": _sha256_json(file_hashes),
        "ood_policy_sha256": file_hashes["ood_policy.json"],
        "hyperparameters_sha256": file_hashes["hyperparameters.json"],
        "risk_calibrator_sha256": file_hashes["risk_calibrator.json"],
        "constraints": {
            "peer_counts_allowed": list(DEFAULT_TRAIN_PEERS),
            "collective_n_allowed": [9, 17],
            "encodings_frozen": [REP],
            "encodings_exploratory": [
                "intervals_24_decimal6",
                "centers_24_standard",
            ],
        },
        "stage_c_gate": (
            "Hash-lock complete. Stage C still requires explicit user GO "
            "and a Stage C protocol freeze."
        ),
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Point stage3b_decision at the locked bundle.
    decision["status"] = "moments_hash_locked"
    decision["hash_lock"] = {
        "bundle_root": manifest["bundle_root"],
        "aggregate_sha256": manifest["aggregate_sha256"],
        "manifest": str(manifest_path.relative_to(ROOT).as_posix()),
        "freeze_ready": True,
        "stage_c_authorized": False,
    }
    (args.artifacts / "stage3b_decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )

    print(
        json.dumps(
            {
                "status": manifest["status"],
                "bundle_root": manifest["bundle_root"],
                "aggregate_sha256": manifest["aggregate_sha256"],
                "n_files": len(file_hashes),
                "freeze_ready": True,
                "stage_c_authorized": False,
                "reload_ok": True,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
